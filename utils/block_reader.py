"""
Random-access safetensors block reader for pipelined model loading.

Provides ``RandomAccessBlockReader``: a thread-safe file-I/O reader that
can seek to and load individual transformer-block weight groups from a
safetensors checkpoint without loading the entire file into RAM.

The reader uses a position-independent read for concurrent reads across
threads — ``os.pread`` on POSIX, and a per-fd locked ``lseek``+``read``
fallback on Windows (where ``os.pread`` does not exist). Either way, no
shared file position is mutated, so block-group reads are thread-safe.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# Cross-platform positioned read.
#
# POSIX: ``os.pread`` is atomic, position-independent and thread-safe.
# Windows: ``os.pread`` does not exist. Prefer pywin32's ``win32file.ReadFile``
# with an ``OVERLAPPED`` offset — a native positioned read, no lock, true
# parallelism across the prefetch workers. Fall back to a per-fd locked
# ``lseek``+``read`` if pywin32 is unavailable. (``os.dup``+``fdopen`` does
# NOT work: the Windows CRT shares the file offset across dup'd handles, so
# concurrent positioned reads would corrupt each other's position.)
_PREAD_LOCKS: dict = {}


def _pread(fd: int, n: int, offset: int) -> bytes:
    """Read *n* bytes from *fd* at *offset* without moving the shared file
    position. POSIX uses ``os.pread``; Windows uses pywin32 OVERLAPPED reads
    when available, else a per-fd locked lseek+read fallback."""
    if hasattr(os, "pread"):
        return os.pread(fd, n, offset)
    # Windows: prefer native overlapped read (lock-free, parallel)
    try:
        import msvcrt
        import pywintypes  # type: ignore
        import win32file  # type: ignore
        h = msvcrt.get_osfhandle(fd)
        ov = pywintypes.OVERLAPPED()
        ov.Offset = offset & 0xFFFFFFFF
        ov.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
        try:
            _, data = win32file.ReadFile(h, n, ov)
        except pywintypes.error as e:
            if e.winerror != win32file.ERROR_IO_PENDING:
                raise
            _, data = win32file.GetOverlappedResult(h, ov, True)
        return data
    except ImportError:
        lock = _PREAD_LOCKS.get(fd)
        if lock is None:
            lock = _PREAD_LOCKS.setdefault(fd, threading.Lock())
        with lock:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, n)

# Minimal dtype map (matches safetensors conventions).
_SAFETENSORS_DTYPE_MAP: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# Typed block-group plan: which safetensors keys belong to which block.
TypedBlockPlan = dict[int, dict[str, int]]
TypedBlockMeta = dict[str, object]


class RandomAccessBlockReader:
    """Thread-safe reader for on-demand block-group reads from safetensors.

    Opens the file once; each ``read_block(idx)`` call uses a positioned
    read (``os.pread`` on POSIX, locked ``lseek``+``read`` on Windows) to
    read tensor data directly from disk at the pre-computed byte offset,
    bypassing the shared file-position pointer.  Multiple threads may call
    ``read_block`` concurrently with no locking needed.

    Parameters
    ----------
    model_path:
        Path to the .safetensors checkpoint file.
    lora_specs:
        Optional list of (lora_path, strength) for inline LoRA merging.
    """

    def __init__(
        self,
        model_path: str,
        lora_specs: Optional[list[tuple[str, float]]] = None,
    ):
        self._path = model_path
        self._fd = os.open(model_path, os.O_RDONLY | os.O_BINARY)

        try:
            from safetensors.torch import _TYPES
            self._dtype_map = _TYPES
        except Exception:
            self._dtype_map = _SAFETENSORS_DTYPE_MAP

        header_len = struct.unpack("<Q", os.read(self._fd, 8))[0]
        self._header: dict = json.loads(os.read(self._fd, header_len))
        self._data_offset: int = 8 + header_len

        self._raw_keys: list[str] = [
            k for k, v in self._header.items()
            if isinstance(v, dict) and "dtype" in v
        ]

        self._lora_specs = lora_specs

        # ── Build block plan ───────────────────────────────────────────
        self._plan: TypedBlockPlan = {}
        self._peripheral: TypedBlockPlan = {}
        self._block_indices: set[int] = set()

        for raw_key in sorted(self._raw_keys):
            target_key = self._normalize_key(raw_key)
            info = self._header[raw_key]
            byte_offset = self._data_offset + info["data_offsets"][0]
            byte_length = info["data_offsets"][1] - info["data_offsets"][0]

            parts = target_key.split(".")
            if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
                bidx = int(parts[1])
                self._block_indices.add(bidx)
                self._plan.setdefault(bidx, {})[target_key] = (byte_offset, byte_length)
            else:
                self._peripheral.setdefault(target_key, {})[target_key] = (byte_offset, byte_length)

        self.num_layers: int = max(self._block_indices) + 1 if self._block_indices else 0

        logger.info(
            "[BlockReader] %s: %d blocks, %d peripheral groups, %d raw keys",
            os.path.basename(model_path), self.num_layers,
            len(self._peripheral), len(self._raw_keys),
        )

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def block_indices(self) -> set[int]:
        return self._block_indices

    def read_block(self, block_idx: int) -> dict[str, torch.Tensor]:
        """Read all tensors for transformer block *block_idx* from disk.

        Thread-safe: uses a positioned read so no file-position state is
        shared across threads (per-fd lock on Windows).
        Returns a dict of ``{canonical_key: cpu_tensor}``.
        """
        entries = self._plan.get(block_idx)
        if not entries:
            raise KeyError(f"Block {block_idx} not found in block plan")
        return self._read_entries(entries)

    def read_peripheral(self) -> dict[str, torch.Tensor]:
        """Read all non-block (peripheral) tensors from disk.

        These include patch_embedding, text/time embedding, time_projection,
        head, and any other top-level keys.  Typically called once at
        model-load time.
        """
        result: dict[str, torch.Tensor] = {}
        for key, (offset, length) in self._peripheral.items():
            raw = self._read_bytes(offset, length)
            dtype, shape = self._tensor_info(key)
            result[key] = torch.frombuffer(raw, dtype=dtype).reshape(shape)
        return result

    def shape_of(self, key: str) -> tuple[int, ...]:
        return tuple(self._header[self._resolve_raw_key(key)]["shape"])

    def dtype_of(self, key: str) -> torch.dtype:
        return self._dtype_map[self._header[self._resolve_raw_key(key)]["dtype"]]

    def has_key(self, key: str) -> bool:
        """Check if a canonical (normalised) key exists in the model."""
        try:
            self._resolve_raw_key(key)
            return True
        except KeyError:
            return False

    def close(self) -> None:
        """Close the underlying file descriptor."""
        try:
            os.close(self._fd)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    # ── Internal ────────────────────────────────────────────────────────

    def _read_entries(
        self, entries: dict[str, tuple[int, int]]
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, (offset, length) in entries.items():
            raw = self._read_bytes(offset, length)
            dtype, shape = self._tensor_info(key)
            result[key] = torch.frombuffer(raw, dtype=dtype).reshape(shape)
        return result

    def _read_bytes(self, offset: int, length: int) -> bytearray:
        """Read raw bytes from disk at *offset*, thread-safe via pread.

        Returns a *mutable* ``bytearray`` rather than immutable ``bytes`` so
        that ``torch.frombuffer`` yields a writable tensor and PyTorch's
        "buffer is not writable" warning is avoided.  The buffer is allocated
        once up front and filled via a memoryview slice so repeated
        ``bytearray.extend`` reallocations are avoided.
        """
        data = bytearray(length)
        view = memoryview(data)
        pos = 0
        while pos < length:
            chunk = _pread(self._fd, min(length - pos, 64 * 1024 * 1024), offset + pos)
            if not chunk:
                raise EOFError(
                    f"Unexpected EOF at offset {offset + pos}")
            view[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
        return data

    def _tensor_info(self, canonical_key: str) -> tuple[torch.dtype, tuple[int, ...]]:
        raw_key = self._resolve_raw_key(canonical_key)
        info = self._header[raw_key]
        dtype = self._dtype_map[info["dtype"]]
        shape = tuple(info["shape"])
        return dtype, shape

    def _resolve_raw_key(self, canonical_key: str) -> str:
        """Map a normalised key to its raw safetensors key."""
        if canonical_key in self._header:
            return canonical_key
        for raw_key in self._raw_keys:
            if self._normalize_key(raw_key) == canonical_key:
                return raw_key
        raise KeyError(f"Key not found: {canonical_key!r}")

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Canonicalise a safetensors key to model state-dict form.

        Only the checkpoint prefix (``model.diffusion_model.`` /
        ``diffusion_model.`` / ``model.``) is stripped.  The ``.weight`` /
        ``.bias`` / ``.weight_scale`` / ``.comfy_quant`` tensor suffix is
        KEPT, so the returned key matches the keys produced by
        ``_normalize_unet_key`` (the streaming loader) and ComfyUI's
        ``load_state_dict`` path.

        Dropping the suffix here was a bug: every tensor under a layer
        (``self_attn.q.weight``, ``.weight_scale``, ``.bias``,
        ``.comfy_quant``) collapsed to the same canonical ``self_attn.q``,
        so the block plan map overwrote itself and ``read_block`` handed
        ``_stream_load_group`` a single bogus key.  ComfyUI's
        ``_load_quantized_module`` then could not find ``self_attn.q.weight``
        and left the layer's ``weight`` as ``None`` -> "'NoneType' object has
        no attribute 'device'" at forward time.
        """
        k = key
        for prefix in (
            "model.diffusion_model.", "diffusion_model.", "model.",
        ):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        return k
