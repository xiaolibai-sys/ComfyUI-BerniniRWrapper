"""
Block-level GPU↔RAM offloading for Wan/Bernini transformer blocks.

Maintains a sliding window of blocks on GPU, offloading blocks outside the
window to CPU to save VRAM.  A dedicated CUDA stream overlaps H2D transfers
with transformer block computation, hiding transfer latency.

Architecture
------------
The module is organised into four components with single responsibilities:

``_BlockWindow``
    Pure state tracking — which blocks are on GPU vs CPU right now.
    No CUDA calls, no tensor operations.

``_TransferEngine``
    CUDA stream management + async H2D/D2H transfers.  Owns the dedicated
    stream, tracks in-flight prefetches, and provides safe move primitives.

``_VRAMBudget``
    VRAM headroom monitoring and CUDA cache management.  Decides when to
    flush the allocator cache to prevent fragmentation.

``BlockSwapManager``
    Thin orchestrator that wires the three components together and exposes
    the public API used by the model's forward pass.

Usage::

    mgr = BlockSwapManager(model, window_size=10, prefetch=True)
    mgr.prepare_pre_forward()
    for i, block in enumerate(model.blocks):
        mgr.prepare(i)
        mgr.prefetch_next(i)
        x = block(x, ...)
    mgr.prepare_head()
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .tensor_ops import (
    move_module_to,
    pin_module,
    record_module_stream,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# _BlockWindow — pure state tracking
# ═══════════════════════════════════════════════════════════════════════════

class _BlockWindow:
    """Tracks which transformer blocks are on GPU vs CPU.

    Pure data structure — no CUDA calls, no tensor operations.
    """

    __slots__ = ("on_gpu", "resident", "prewarmed", "total", "window_size")

    def __init__(self, total: int, window_size: int, prewarmed: int = 0):
        self.total = total
        self.window_size = max(1, window_size)
        self.prewarmed = prewarmed
        # Blocks known to be in CPU RAM (loaded during streaming pass).
        self.resident: set[int] = set(range(prewarmed, total))
        # Blocks currently on GPU.
        self.on_gpu: set[int] = set(range(prewarmed))

    # -- query ---------------------------------------------------------------

    def needed(self, block_idx: int) -> set[int]:
        """Blocks that should be on GPU for the window starting at *block_idx*."""
        start = max(0, block_idx)
        end = min(self.total, block_idx + self.window_size)
        return set(range(start, end))

    def to_offload(self, block_idx: int) -> set[int]:
        """Blocks on GPU that are no longer in the window."""
        return self.on_gpu - self.needed(block_idx)

    def to_load(self, block_idx: int) -> set[int]:
        """Blocks needed for the window that are not yet on GPU."""
        return self.needed(block_idx) - self.on_gpu

    def is_on_gpu(self, block_idx: int) -> bool:
        return block_idx in self.on_gpu

    # -- mutations -----------------------------------------------------------

    def mark_loaded(self, block_idx: int) -> None:
        self.on_gpu.add(block_idx)
        self.resident.discard(block_idx)

    def mark_offloaded(self, block_idx: int) -> None:
        self.on_gpu.discard(block_idx)
        self.resident.add(block_idx)

    def clear(self) -> None:
        self.on_gpu.clear()
        self.resident = set(range(self.total))


# ═══════════════════════════════════════════════════════════════════════════
# _TransferEngine — CUDA stream + async H2D/D2H
# ═══════════════════════════════════════════════════════════════════════════

class _TransferEngine:
    """Manages a dedicated CUDA stream for async block transfers.

    Owns the stream lifecycle and tracks in-flight prefetch operations
    via CUDA events.  All tensor movement goes through
    :mod:`tensor_ops` so ``QuantizedTensor`` (fp8) and other subclasses
    are handled correctly.
    """

    def __init__(
        self,
        device: torch.device | str,
        prefetch: bool = True,
        prefetch_count: int = 1,
        pin_memory: bool = False,
    ):
        self.device = device
        self.prefetch = prefetch and torch.cuda.is_available()
        self.prefetch_count = max(1, prefetch_count)
        self.pin_memory = pin_memory

        self._stream: Optional[torch.cuda.Stream] = None
        if self.prefetch:
            try:
                self._stream = torch.cuda.Stream(device=device)
            except Exception:
                pass

        # block_idx → CUDA event recorded after async transfer started
        self._events: dict[int, torch.cuda.Event] = {}

    # -- move ----------------------------------------------------------------

    def to_gpu(self, module: torch.nn.Module, non_blocking: bool = False) -> None:
        """Move *module* to GPU, optionally async on the transfer stream."""
        if non_blocking and self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())
            if self.pin_memory:
                pin_module(module)
            with torch.cuda.stream(self._stream):
                move_module_to(module, self.device, non_blocking=True)
                record_module_stream(module, self._stream)
        else:
            move_module_to(module, self.device)

    def to_cpu(
        self,
        module: torch.nn.Module,
        block_idx: int | None = None,
    ) -> None:
        """Move *module* to CPU, waiting for in-flight prefetch if needed."""
        if block_idx is not None and block_idx in self._events:
            event = self._events.pop(block_idx)
            torch.cuda.current_stream().wait_event(event)
        elif self._events:
            self.cancel_all()
        move_module_to(module, "cpu")

    # -- prefetch ------------------------------------------------------------

    def start_prefetch(self, block_idx: int, module: torch.nn.Module) -> None:
        """Begin async H2D transfer for *module*, tracked by *block_idx*."""
        if block_idx in self._events:
            return
        self.to_gpu(module, non_blocking=True)
        if self._stream is not None:
            event = torch.cuda.Event()
            event.record(self._stream)
            self._events[block_idx] = event

    def sync_prefetch(self, block_idx: int) -> bool:
        """Wait for *block_idx*'s async transfer to finish.  Returns True if waited."""
        event = self._events.pop(block_idx, None)
        if event is None:
            return False
        torch.cuda.current_stream().wait_event(event)
        return True

    def cancel_all(self) -> None:
        """Abort all in-flight prefetches."""
        if not self._events:
            return
        if self._stream is not None:
            self._stream.synchronize()
        self._events.clear()

    def sync_all(self) -> None:
        """Block until every CUDA stream (including compute) is done."""
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# _VRAMBudget — memory monitoring
# ═══════════════════════════════════════════════════════════════════════════

class _VRAMBudget:
    """Tracks per-block VRAM usage and manages CUDA cache flushes.

    Decides when to call ``torch.cuda.empty_cache()`` to defragment the
    allocator — too often hurts perf, too rarely causes OOM from
    fragmentation holes.
    """

    def __init__(self, block_mb: float):
        self.block_mb = block_mb

    @staticmethod
    def estimate_block_mb(block) -> float:
        """Rough VRAM for one block in MB (assumes bf16/fp16 = 2 bytes/param)."""
        total = sum(p.numel() for p in block.parameters()) * 2
        return total / (1024 * 1024)

    def free_mb(self, device: torch.device | str | None = None) -> float:
        if not torch.cuda.is_available():
            return float("inf")
        try:
            free, _ = torch.cuda.mem_get_info(device)
            return free / (1024 * 1024)
        except Exception:
            return float("inf")

    def is_fragmented(self, device: torch.device | str | None = None) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
            return reserved > allocated + self.block_mb * 2
        except Exception:
            return False

    def maybe_flush(
        self,
        device: torch.device | str | None = None,
        reserve_blocks: int = 2,
    ) -> None:
        """Release cached GPU memory if free space is tight or fragmented."""
        if not torch.cuda.is_available():
            return
        needed_mb = self.block_mb * max(1, reserve_blocks)
        if self.free_mb(device) < needed_mb or self.is_fragmented(device):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# BlockSwapManager — public API
# ═══════════════════════════════════════════════════════════════════════════

class BlockSwapManager:
    """Sliding-window GPU block cache with async CUDA-stream prefetch.

    Keeps at most *window_size* consecutive transformer blocks on GPU at any
    time.  Non-block modules are brought to GPU on demand.  A dedicated CUDA
    stream prefetches upcoming blocks while the current block computes.

    Parameters
    ----------
    model:
        The diffusion model (must have ``.blocks`` list and peripheral attrs).
    window_size:
        Number of transformer blocks to keep on GPU simultaneously.
    prefetch:
        If True, async H2D copies run on a dedicated CUDA stream.
    prefetch_count:
        Max blocks to prefetch ahead of the current window.
    pin_memory:
        If True, pin CPU copies before async transfer (uses more host RAM).
    """

    # Peripheral module names managed alongside transformer blocks.
    _PERIPHERAL_NAMES = (
        "patch_embedding",
        "text_embedding",
        "time_embedding",
        "time_projection",
        "head",
    )

    def __init__(
        self,
        model,
        window_size: int = 10,
        device: str = "cuda",
        prefetch: bool = True,
        prefetch_count: int = 1,
        pin_memory: bool = False,
    ):
        self.model = model
        self._blocks = model.blocks
        total = len(self._blocks)

        # Resolve pre-warmed count from the streaming loader.
        prewarmed = getattr(model, "_prewarmed", 0)

        # ---- sub-components ------------------------------------------------
        self._window = _BlockWindow(total, window_size, prewarmed)
        self._xfer = _TransferEngine(device, prefetch, prefetch_count, pin_memory)

        # Per-block VRAM estimate.
        _meta = getattr(model, "_block_meta", None)
        if _meta and _meta.get("avg_mb", 0) > 0:
            block_mb = _meta["avg_mb"]
        else:
            block_mb = _VRAMBudget.estimate_block_mb(self._blocks[0])
        self._budget = _VRAMBudget(block_mb)

        # Peripheral modules (embeddings, head, etc.)
        self._peri = {
            name: getattr(model, name, None)
            for name in self._PERIPHERAL_NAMES
        }

        logger.info(
            "[BlockSwap] %d blocks total, window=%d, per-block ~%.0f MB, "
            "target VRAM ~%.0f MB, prefetch=%s, count=%d, pin=%s, prewarmed=%d",
            total, window_size, block_mb, window_size * block_mb,
            prefetch, prefetch_count, pin_memory, prewarmed,
        )

        self._init_state()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        """Establish deterministic starting state.

        The streaming loader placed the first ``_window.prewarmed`` blocks
        on GPU; the rest are in CPU RAM.  Only peripherals may need moving
        (ComfyUI may have moved them to GPU).

        If ``_prewarmed`` was not set by the loader, detect GPU blocks now
        and self-correct.
        """
        self._xfer.cancel_all()

        # Defensive: detect GPU blocks if prewarmed wasn't set.
        if self._window.prewarmed == 0 and self._window.total > 0:
            gpu_blocks = []
            for i in range(self._window.total):
                try:
                    p0 = next(self._blocks[i].parameters(), None)
                    if p0 is not None and p0.is_cuda:
                        gpu_blocks.append(i)
                except Exception:
                    pass
            if gpu_blocks:
                self._window.prewarmed = max(gpu_blocks) + 1
                self._window.on_gpu = set(range(self._window.prewarmed))
                self._window.resident = set(
                    range(self._window.prewarmed, self._window.total))
                logger.warning(
                    "[BlockSwap] _prewarmed was 0 but detected %d GPU blocks "
                    "(0-%d); self-corrected.",
                    len(gpu_blocks), max(gpu_blocks),
                )

        # Ensure peripherals start on CPU (model's home device).
        for mod in self._peri.values():
            if mod is not None:
                move_module_to(mod, "cpu")
        self._budget.maybe_flush(self._xfer.device, reserve_blocks=0)

    # ------------------------------------------------------------------
    # Public API — called from the model's forward pass
    # ------------------------------------------------------------------

    def prepare(self, block_idx: int) -> None:
        """Ensure block *block_idx* and its window are on GPU.

        Offloads blocks that have fallen outside the window and loads newly
        needed blocks.  Call at the top of each loop iteration *before*
        using the block.
        """
        if block_idx == 0:
            self.prepare_blocks_entry()

        # Offload blocks no longer in window.
        for i in sorted(self._window.to_offload(block_idx)):
            self._xfer.to_cpu(self._blocks[i], block_idx=i)
            self._window.mark_offloaded(i)

        if self._window.to_offload(block_idx):
            self._budget.maybe_flush(self._xfer.device, reserve_blocks=2)

        # Load blocks newly entering the window.
        for i in sorted(self._window.to_load(block_idx)):
            if not self._xfer.sync_prefetch(i):
                self._xfer.to_gpu(self._blocks[i])
            self._window.mark_loaded(i)

    def prefetch_next(self, block_idx: int) -> None:
        """Start async H2D copies for blocks just beyond the window.

        Call after ``prepare(block_idx)`` and before the compute for
        *block_idx*.  Copies run on the transfer stream in parallel with
        the current block's forward pass.
        """
        start = block_idx + self._window.window_size
        if start >= self._window.total:
            return
        count = self._max_prefetch(start)
        end = min(self._window.total, start + count)
        for i in range(start, end):
            if self._window.is_on_gpu(i):
                continue
            self._xfer.start_prefetch(i, self._blocks[i])

    def prepare_pre_forward(self) -> None:
        """Ensure patch_embedding is on GPU before ``pre_forward``."""
        self._xfer.cancel_all()
        self._load_peripheral("patch_embedding")

    def prepare_blocks_entry(self) -> None:
        """Ensure embeddings/projection are on GPU for ``transformer_forward``."""
        for name in ("text_embedding", "time_embedding", "time_projection"):
            self._load_peripheral(name)

    def prepare_head(self) -> None:
        """Ensure head is on GPU before final unpatchify."""
        self._load_peripheral("head")

    def is_on_gpu(self, block_idx: int) -> bool:
        """Check if a block is currently on GPU (for S² drop / STG skip)."""
        return self._window.is_on_gpu(block_idx)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def evict_all(self) -> None:
        """Move all blocks and peripherals back to CPU."""
        self._xfer.cancel_all()
        self._xfer.sync_all()
        for i in sorted(self._window.on_gpu):
            self._xfer.to_cpu(self._blocks[i], block_idx=i)
        self._window.clear()
        for name in self._PERIPHERAL_NAMES:
            self._offload_peripheral(name)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Deterministic full teardown, safe to call on model unload/switch.

        Order matters on Windows: cancel prefetches, synchronise every
        CUDA stream, then evict everything.
        """
        try:
            self._xfer.cancel_all()
        except Exception:
            pass
        self._xfer.sync_all()
        self.evict_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_peripheral(self, name: str) -> None:
        mod = self._peri.get(name)
        if mod is not None:
            move_module_to(mod, self._xfer.device)

    def _offload_peripheral(self, name: str) -> None:
        mod = self._peri.get(name)
        if mod is not None:
            move_module_to(mod, "cpu")

    def _max_prefetch(self, start: int) -> int:
        """How many blocks we can safely prefetch given free VRAM."""
        if not self._xfer.prefetch or self._xfer._stream is None:
            return 0
        free_mb = self._budget.free_mb(self._xfer.device)
        reserve_mb = self._budget.block_mb * 2
        avail_mb = max(0.0, free_mb - reserve_mb)
        max_by_mem = int(avail_mb / (self._budget.block_mb * 1.2))
        return max(0, min(self._xfer.prefetch_count, max_by_mem))


# ═══════════════════════════════════════════════════════════════════════════
# StreamingBlockPipeline — pipelined disk→RAM→GPU loader
# ═══════════════════════════════════════════════════════════════════════════

class StreamingBlockPipeline:
    """Pipelined block loader that overlaps disk reads with GPU transfers.

    Reads block groups from a safetensors file and immediately moves them
    to the target device using an async CUDA stream.  The next block group
    is read from disk *while* the previous group's GPU transfer is still
    in flight — hiding disk I/O latency behind the H2D copy.

    Synchronisation points every *sync_every* blocks bound in-flight
    transfers so RAM usage stays tight.

    Parameters
    ----------
    dm:
        The diffusion model (must have ``.blocks`` list).
    load_device:
        Target GPU device.
    window_size:
        Number of blocks to pre-warm onto GPU.
    sync_every:
        Insert a CUDA sync after every N blocks.
    """

    def __init__(
        self,
        dm,
        load_device: torch.device,
        window_size: int = 10,
        sync_every: int = 5,
    ):
        self._dm = dm
        self._device = load_device
        self._window = window_size
        self._sync_every = max(1, sync_every)
        self._xfer = _TransferEngine(load_device, prefetch=True, prefetch_count=1)

    def run(self, block_groups: list[tuple[str, dict[str, torch.Tensor]]]) -> int:
        """Stream *block_groups* into the model with pipelined GPU transfers.

        Returns the number of pre-warmed blocks.
        """
        total = len(self._dm.blocks)
        warm_blocks = min(self._window, total)
        synced = 0

        for group_key, group in block_groups:
            self._load_group(group_key, group)

            if group_key.startswith("blocks."):
                try:
                    idx = int(group_key.split(".")[1])
                except (IndexError, ValueError):
                    continue
                if idx < warm_blocks:
                    self._xfer.to_gpu(self._dm.blocks[idx], non_blocking=True)

            synced += 1
            if synced >= self._sync_every:
                self._xfer.sync_all()
                synced = 0

        self._xfer.sync_all()
        return warm_blocks

    def _load_group(self, group_key: str, group: dict) -> None:
        """Load a block group into the CPU model (disk → RAM)."""
        prefix = group_key + "."
        if group_key.startswith("blocks."):
            try:
                idx = int(group_key.split(".")[1])
            except (IndexError, ValueError):
                self._dm.load_state_dict(group, strict=False, assign=False)
                return
            sub = self._dm.blocks[idx]
        elif hasattr(self._dm, group_key):
            sub = getattr(self._dm, group_key)
        else:
            self._dm.load_state_dict(group, strict=False, assign=False)
            return
        sub_group = {k[len(prefix):]: v for k, v in group.items()
                     if k.startswith(prefix)}
        sub.load_state_dict(sub_group, strict=False, assign=False)
