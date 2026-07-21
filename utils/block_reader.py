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
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

# Shared typed payloads
from .types import LoraTensorEntry
from .keys import _normalize_unet_key

from .log import get_logger as _get_logger

logger = _get_logger("BlockReader")
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


# ── Common utility: normalise safetensors key ──────────────────────────
# Canonical implementation lives in utils.keys (imported above).


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

        # Shared tensor-level I/O pool for parallel read_block.  Lazy-built;
        # a single pool per reader caps total in-flight reads regardless of
        # how many _DiskPrefetcher workers call read_block concurrently.
        self._io_pool: ThreadPoolExecutor | None = None

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
            "%s: %d blocks, %d peripheral groups, %d raw keys",
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
        """Close the underlying file descriptor and shut down the I/O pool."""
        try:
            os.close(self._fd)
        except OSError:
            pass
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=False)
            self._io_pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    # ── Internal ────────────────────────────────────────────────────────

    def _read_entries(
        self, entries: dict[str, tuple[int, int]]
    ) -> dict[str, torch.Tensor]:
        if len(entries) <= 1:
            return {
                key: self._read_one(key, offset, length)
                for key, (offset, length) in entries.items()
            }
        # Parallel positioned reads across the block's tensors, submitted in
        # file-offset order.  The shared pool keeps peak memory identical to
        # the sequential path — the whole block is materialised either way.
        ordered = sorted(entries.items(), key=lambda kv: kv[1][0])
        pool = self._get_io_pool()
        futs = [pool.submit(self._read_one, key, offset, length)
                for key, (offset, length) in ordered]
        return {key: fu.result() for (key, _), fu in zip(ordered, futs)}

    def _read_one(self, key: str, offset: int, length: int) -> torch.Tensor:
        """Read a single tensor from disk (thread-safe via ``_pread``)."""
        raw = self._read_bytes(offset, length)
        dtype, shape = self._tensor_info(key)
        return torch.frombuffer(raw, dtype=dtype).reshape(shape)

    def _get_io_pool(self) -> ThreadPoolExecutor:
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="block-reader-io")
        return self._io_pool

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

        Delegates to ``utils.keys._normalize_unet_key``, the canonical
        implementation shared with the checkpoint loader and LoRA folding.
        """
        return _normalize_unet_key(key)


# ── LoraBlockReader — on-demand per-block LoRA reader ───────────────────


class LoraBlockReader:
    """Per-block on-demand LoRA reader using random-access positioned reads.

    Opens each LoRA safetensors file once and builds an offset plan (no
    materialised tensors).  ``read_block_lora(idx)`` reads raw tensors from
    disk and groups them into the same format expected by the unified slot
    (``block._lora_payload``), matching the output of
    ``_build_streaming_lora_groups`` for a single block.

    Non-block LoRA entries (patch_embedding, head, norms, etc.) are
    collected by ``read_non_block()`` which materialises them at startup —
    these are few (typically < 10) and are needed during peripheral loading.

    Thread safety: reads use ``_pread`` (position-independent), so multiple
    ``read_block_lora`` calls can run concurrently on different block indices
    with no locking.  However the object itself is intended for single-thread
    usage in the lazy-mode disk prefetcher which already serialises by block.
    """

    # Suffixes and their kind labels.  Covers both Diffusers (lora_A/lora_B)
    # and Kohya (lora_down/lora_up) naming — same kind value, so the fold
    # math is identical.  Also handles DoRA (diff_b/diff) and alpha.
    _LORA_SUFFIXES = {
        ".lora_A.weight": "A",
        ".lora_B.weight": "B",
        ".lora_down.weight": "A",
        ".lora_up.weight": "B",
        ".alpha": "alpha",
        ".diff_b": "diff_b",
        ".diff": "diff",
    }

    def __init__(self, lora_specs: list[tuple[str, float]]):
        # -- first pass: deduplicate paths, open files, read headers --
        self._fds: list[int] = []
        self._path_fd_idx: dict[str, int] = {}
        self._path_headers: dict[str, tuple[dict, int]] = {}

        for path, strength in lora_specs:
            if strength == 0.0:
                continue
            if path not in self._path_headers:
                fd = os.open(path, os.O_RDONLY | os.O_BINARY)
                fd_idx = len(self._fds)
                self._fds.append(fd)
                self._path_fd_idx[path] = fd_idx
                hdr_len = struct.unpack("<Q", os.read(fd, 8))[0]
                header = json.loads(os.read(fd, hdr_len))
                self._path_headers[path] = (header, 8 + hdr_len)

        # -- second pass: build offset plan from headers --
        self._spec_strengths: list[float] = []
        self._plan: dict[int, dict[str, list[LoraTensorEntry]]] = {}
        self._non_block_plan: dict[str, list[LoraTensorEntry]] = {}

        valid_spec_idx = 0
        for path, strength in lora_specs:
            if strength == 0.0:
                continue
            header, data_offset = self._path_headers[path]
            fd_idx = self._path_fd_idx[path]
            self._spec_strengths.append(strength)

            unknown_count = 0
            unknown_examples: list[str] = []

            for raw_key, info in header.items():
                if not isinstance(info, dict) or "dtype" not in info:
                    continue  # metadata-only keys (__metadata__, etc.)
                normalized = _normalize_unet_key(raw_key)
                kind = self._identify_lora_kind(normalized)
                if kind is None:
                    unknown_count += 1
                    if len(unknown_examples) < 3:
                        unknown_examples.append(raw_key[-60:])
                    continue
                base_key = self._strip_lora_suffix(normalized) + ".weight"

                entry = LoraTensorEntry(
                    fd_idx=fd_idx,
                    offset=data_offset + info["data_offsets"][0],
                    length=info["data_offsets"][1] - info["data_offsets"][0],
                    dtype=_SAFETENSORS_DTYPE_MAP.get(info["dtype"], torch.float32),
                    shape=tuple(info["shape"]),
                    kind=kind,
                    spec_idx=valid_spec_idx,
                )

                parts = base_key.split('.')
                if len(parts) >= 2 and parts[0] == 'blocks' and parts[1].isdigit():
                    bidx = int(parts[1])
                    pname = '.'.join(parts[2:])  # e.g. "self_attn.q.weight"
                    self._plan.setdefault(bidx, {}).setdefault(pname, []).append(entry)
                else:
                    self._non_block_plan.setdefault(base_key, []).append(entry)

            if unknown_count > 0:
                logger.warning(
                    "%s: %d tensor(s) with unrecognised suffix "
                    "(silently skipped).  Examples: %s",
                    os.path.basename(path), unknown_count, unknown_examples,
                )

            valid_spec_idx += 1

        logger.info(
            "%d fd(s), %d valid spec(s), %d block(s) planned, "
            "%d non-block entry group(s)",
            len(self._fds), valid_spec_idx, len(self._plan), len(self._non_block_plan),
        )

    # ── Public API ──────────────────────────────────────────────────────

    def read_block_lora(self, block_idx: int) -> dict[str, list[dict]] | None:
        """Read and group LoRA tensors for *block_idx* from disk on demand.

        Returns ``{pname: [entry_per_spec, ...]}`` — the same format as
        ``_build_streaming_lora_groups`` for one block — or ``None`` when
        *block_idx* has no LoRA entries.  Each per-spec entry is a dict
        with keys ``A``, ``B``, ``alpha``, ``diff_b``, ``diff`` (may be
        ``None``) and ``strength``.
        """
        block_entries = self._plan.get(block_idx)
        if not block_entries:
            return None

        result: dict[str, list[dict]] = {}
        for pname, raw_entries in block_entries.items():
            by_spec: dict[int, dict[str, LoraTensorEntry]] = {}
            for e in raw_entries:
                by_spec.setdefault(e.spec_idx, {})[e.kind] = e

            spec_entries: list[dict] = []
            for sidx in sorted(by_spec):
                kinds = by_spec[sidx]
                entry: dict = {
                    "A": None, "B": None,
                    "alpha": None, "diff_b": None, "diff": None,
                    "strength": self._spec_strengths[sidx],
                }
                for k in ("A", "B", "alpha", "diff_b", "diff"):
                    if k in kinds:
                        entry[k] = self._read_tensor(kinds[k])
                spec_entries.append(entry)
            result[pname] = spec_entries

        return result

    def read_non_block(self) -> dict[str, list[dict]]:
        """Read and group ALL non-block LoRA entries at startup.

        Returns the same format as ``_build_streaming_lora_groups`` but only
        for keys that are NOT under any ``blocks.N.`` prefix
        (e.g. ``patch_embedding.weight``, ``head.weight``).  Typically called
        once, right after construction, to feed into the peripheral parameter
        merge path.
        """
        return self._group_and_materialise(self._non_block_plan)

    def close(self) -> None:
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _identify_lora_kind(normalized_key: str) -> str | None:
        """Return the LoRA kind ('A', 'B', 'alpha', 'diff_b', 'diff') or None."""
        for suffix, kind in LoraBlockReader._LORA_SUFFIXES.items():
            if normalized_key.endswith(suffix):
                return kind
        return None

    @staticmethod
    def _strip_lora_suffix(normalized_key: str) -> str:
        """Strip the LoRA suffix from a normalized key, returning the base.

        E.g. ``blocks.0.self_attn.q.lora_A.weight`` → ``blocks.0.self_attn.q``.
        """
        for suffix in LoraBlockReader._LORA_SUFFIXES:
            if normalized_key.endswith(suffix):
                return normalized_key[:-len(suffix)]
        return normalized_key

    def _read_tensor(self, entry: LoraTensorEntry) -> torch.Tensor:
        """Read a single tensor from disk using the entry's metadata."""
        fd = self._fds[entry.fd_idx]
        length = entry.length
        data = bytearray(length)
        view = memoryview(data)
        offset = entry.offset
        pos = 0
        while pos < length:
            chunk = _pread(fd, min(length - pos, 64 * 1024 * 1024), offset + pos)
            if not chunk:
                raise EOFError(f"Unexpected EOF at offset {offset + pos}")
            view[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
        return torch.frombuffer(data, dtype=entry.dtype).reshape(entry.shape)

    def _group_and_materialise(
        self, plan: dict[str, list[LoraTensorEntry]],
    ) -> dict[str, list[dict]]:
        """Read & group entries from *plan* into per-spec entry dicts."""
        result: dict[str, list[dict]] = {}
        for base_key, raw_entries in plan.items():
            by_spec: dict[int, dict[str, LoraTensorEntry]] = {}
            for e in raw_entries:
                by_spec.setdefault(e.spec_idx, {})[e.kind] = e

            spec_entries: list[dict] = []
            for sidx in sorted(by_spec):
                kinds = by_spec[sidx]
                entry: dict = {
                    "A": None, "B": None,
                    "alpha": None, "diff_b": None, "diff": None,
                    "strength": self._spec_strengths[sidx],
                }
                for k in ("A", "B", "alpha", "diff_b", "diff"):
                    if k in kinds:
                        entry[k] = self._read_tensor(kinds[k])
                spec_entries.append(entry)
            result[base_key] = spec_entries
        return result
