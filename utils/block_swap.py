"""
Block-level GPU↔RAM offloading for Wan/Bernini transformer blocks.

Maintains a sliding window of blocks on GPU, offloading blocks outside the
window to CPU to save VRAM.  A dedicated CUDA stream overlaps H2D transfers
with transformer block computation, hiding transfer latency.

When a ``RandomAccessBlockReader`` is provided (lazy-loading mode), blocks
are loaded from disk on demand by a worker thread pool.  In both modes the
off-GPU blocks reside in a fixed, pre-allocated CPU "home" pool sized to
``(N - window)`` blocks, so host RAM stays flat at ``(N-W)`` for the whole
run — there is no per-step allocation churn (the previous move-based swap
produced a periodic multi-GB sawtooth on Windows).

Architecture
------------
The module is organised into five components with single responsibilities:

``_BlockWindow``
    Pure state tracking — which blocks are on GPU vs CPU right now.
    No CUDA calls, no tensor operations.

``_TransferEngine``
    CUDA stream management + async H2D/D2H transfers.  Owns the dedicated
    stream, tracks in-flight prefetches, and provides safe move primitives.

``_VRAMBudget``
    VRAM headroom monitoring and CUDA cache management.  Decides when to
    flush the allocator cache to prevent fragmentation.

``_DiskPrefetcher``
    Disk→RAM loader running on a background thread pool.  Reads blocks from
    safetensors on demand, populates model parameters, and evicts unused
    blocks from RAM.

``BlockSwapManager``
    Thin orchestrator that wires the four components together and exposes
    the public API used by the model's forward pass.

Usage::

    mgr = BlockSwapManager(model, window_size=10, block_reader=reader)
    mgr.prepare_pre_forward()
    for i, block in enumerate(model.blocks):
        mgr.prepare(i)
        mgr.prefetch_next(i)
        x = block(x, ...)
    mgr.prepare_head()
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

from .tensor_ops import (
    free_module_storage,
    move_module_to,
    pin_module,
    record_module_stream,
)
from .types import DiskLoadRequest, SlotEntry

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# _BlockWindow — pure state tracking
# ═══════════════════════════════════════════════════════════════════════════

class _BlockWindow:
    """Tracks which transformer blocks are on GPU vs CPU vs disk.

    Pure data structure — no CUDA calls, no tensor operations.
    """

    __slots__ = ("on_gpu", "resident", "prewarmed", "total", "window_size",
                 "ram_resident", "disk_loading")

    def __init__(self, total: int, window_size: int, prewarmed: int = 0):
        self.total = total
        self.window_size = max(1, window_size)
        self.prewarmed = prewarmed
        # Blocks known to be in CPU RAM (weights populated).
        # In lazy mode this starts empty (blocks loaded on demand).
        # In eager mode (legacy) this holds all blocks.
        self.resident: set[int] = set(range(prewarmed, total))
        # Blocks whose CPU RAM parameters have data (for eviction tracking).
        # Mirrors self.resident initially; diverges in lazy mode.
        self.ram_resident: set[int] = set(range(total))
        # Blocks being read from disk by a background thread right now.
        self.disk_loading: set[int] = set()
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

    def is_in_ram(self, block_idx: int) -> bool:
        """True if the block's weights are loaded in CPU RAM."""
        return block_idx in self.ram_resident

    def prefetch_window(self, block_idx: int) -> set[int]:
        """Blocks just beyond the current window (candidate prefetch range)."""
        start = block_idx + self.window_size
        if start >= self.total:
            return set()
        count = min(self.window_size, self.total - start)
        return set(range(start, start + count))

    # -- mutations -----------------------------------------------------------

    def mark_loaded(self, block_idx: int) -> None:
        self.on_gpu.add(block_idx)
        self.resident.discard(block_idx)

    def mark_offloaded(self, block_idx: int) -> None:
        self.on_gpu.discard(block_idx)
        self.resident.add(block_idx)

    def mark_ram_loaded(self, block_idx: int) -> None:
        self.ram_resident.add(block_idx)
        self.disk_loading.discard(block_idx)

    def mark_ram_freed(self, block_idx: int) -> None:
        self.ram_resident.discard(block_idx)

    def mark_disk_loading(self, block_idx: int) -> None:
        self.disk_loading.add(block_idx)

    def clear(self) -> None:
        self.on_gpu.clear()
        self.resident = set(range(self.total))
        self.ram_resident = set(range(self.total))
        self.disk_loading.clear()


# ═══════════════════════════════════════════════════════════════════════════
# PinStage — pinned CPU staging ring, sized to GPU ring buffer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PinStage:
    """Pinned CPU staging buffers for truly async H2D transfers.

    Each slot holds a ``dict[str, SlotEntry]`` whose tensor memory is
    page-locked (pinned), so CUDA can DMA directly from it without a
    staging copy.  The ring cursor is shared with the GPU ring buffer:
    slot ``ring_idx % ring_size`` pairs with the GPU slot at the same index.

    Pinning only the transfer staging (``W + prefetch`` blocks) instead of
    the entire home pool (``N - W`` blocks) keeps the pinned memory footprint
    small — critical on Windows where large pinned allocations strain the
    allocator.
    """
    slots: list[dict | None] = field(default_factory=list)
    built: list[bool] = field(default_factory=list)

    @classmethod
    def new(cls, ring_size: int) -> "PinStage":
        return cls(slots=[None] * ring_size, built=[False] * ring_size)

    @property
    def size(self) -> int:
        return len(self.slots)

    def ensure(self, ring_idx: int, template: dict, pin_memory: bool = False) -> dict:
        """Build pinned ``SlotEntry`` for *ring_idx* from *template* (an
        existing ``dict[str, SlotEntry]``, e.g. a home slot or GPU slot)
        if not yet done."""
        idx = ring_idx % self.size if self.size else 0
        slot = self.slots[idx]
        if slot is not None and self.built[idx]:
            return slot
        self.slots[idx] = {
            n: SlotEntry.empty_like_entry(e, device="cpu", pin_memory=pin_memory)
            for n, e in template.items()
        }
        self.built[idx] = True
        return self.slots[idx]


# ═══════════════════════════════════════════════════════════════════════════
# _TransferEngine — CUDA stream + async H2D/D2H
# ═══════════════════════════════════════════════════════════════════════════

class _TransferEngine:
    """CUDA stream + fixed slot pools for churn-free block swaps.

    Two pools are allocated **once** at construction:

    * a pinned CPU "home" pool of ``n_home_slots`` block buffers — at steady
      state exactly the off-GPU blocks (``total - window`` of them) occupy
      these, so host RAM stays flat at ``(N-W)`` and is never returned to the
      OS allocator (no periodic churn / 9 GB sawtooth);
    * a GPU "slot" pool of ``n_gpu_slots`` block buffers
      (``window + prefetch``) used while a block computes.

    Blocks move between pools with ``copy_`` (never ``module.to``), so no new
    tensors are allocated on the hot path.  ``pin_memory`` makes the home pool
    page-locked, enabling async H2D on the dedicated stream.

    The legacy ``to_gpu``/``to_cpu`` *move* primitives are retained for
    ``StreamingBlockPipeline`` (one-off model pre-load), which does not use the
    slot pools.
    """

    def __init__(
        self,
        device: torch.device | str,
        prefetch: bool = True,
        prefetch_count: int = 1,
        pin_memory: bool = False,
        n_home_slots: int = 0,
        n_gpu_slots: int = 0,
        template_block=None,
        home: "BlockHome | None" = None,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.prefetch = prefetch and torch.cuda.is_available()
        self.prefetch_count = max(1, prefetch_count)
        # pin_memory only meaningful with a CUDA device
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self._home = home

        self._stream: Optional[torch.cuda.Stream] = None
        if self.prefetch:
            try:
                self._stream = torch.cuda.Stream(device=device)
            except Exception:
                pass

        # block_idx → CUDA event recorded after async transfer started
        self._events: dict[int, torch.cuda.Event] = {}

        # ---- slot pools ----------------------------------------------------
        # Each slot is a ``dict[str, SlotEntry]`` — one entry per parameter.
        # ``SlotEntry`` handles both regular tensors and QuantizedTensors via
        # a unified ``copy_from`` / ``assign_to`` interface, so the swap
        # engine never branches on parameter type.
        #
        # GPU and pin (staging) slots use a shared ring buffer cursor
        # (``_gpu_cursor``) — deterministic O(1) allocation.
        nh = max(0, n_home_slots)
        ng = max(0, n_gpu_slots)
        self._home_pool: list[dict | None] = [None] * nh
        self._home_built: list[bool] = [False] * nh
        self._home_free: list[int] = list(range(nh))

        self._pin: PinStage = PinStage.new(max(1, self.prefetch_count))  # pinned CPU staging (small — only for prefetch overlap)
        self._gpu_pool: list[dict | None] = [None] * ng   # GPU ring buffer
        self._gpu_built: list[bool] = [False] * ng

        self._gpu_cursor: int = 0
        self._block_home: dict[int, int] = {}
        self._block_gpu: dict[int, int] = {}

    # -- slot bookkeeping ----------------------------------------------------

    def _ring_acquire(self) -> int:
        """Advance the GPU ring-buffer cursor and return the next slot index."""
        ng = len(self._gpu_pool)
        idx = self._gpu_cursor % ng
        self._gpu_cursor = (self._gpu_cursor + 1) % ng
        return idx

    def _ensure_entries(self, pool, built, idx, block, device) -> dict:
        """Build a ``name → SlotEntry`` dict for *block* if not already done."""
        slot = pool[idx]
        if slot is not None and built[idx]:
            return slot
        pool[idx] = {
            n: SlotEntry.empty_like(
                p, device=device,
                pin_memory=(device == "cpu" and self.pin_memory),
            )
            for n, p in block.named_parameters()
        }
        built[idx] = True
        return pool[idx]

    def _swap_param(self, block, entries, non_blocking: bool = False) -> None:
        """Copy every param's data from *block* into *entries*, then assign
        the entries back to the block's ``_parameters`` — all via ``SlotEntry``,
        which handles both regular and QuantizedTensor params uniformly."""
        for n, p in block.named_parameters():
            e = entries.get(n)
            if e is None:
                continue
            e.copy_from(p, non_blocking=non_blocking)
            e.assign_to(block, n)

    def reset(self) -> None:
        """Return every slot to the free list and clear block mappings."""
        self._block_home.clear()
        self._block_gpu.clear()
        self._home_free = list(range(len(self._home_pool)))
        self._gpu_cursor = 0
        self._events.clear()

    def clear_pools(self) -> None:
        """Drop all pool tensor references so GC can reclaim RAM/VRAM."""
        nh, ng = len(self._home_pool), len(self._gpu_pool)
        self._home_pool = [None] * nh
        self._gpu_pool = [None] * ng
        self._pin = PinStage.new(ng)
        self._home_built = [False] * nh
        self._gpu_built = [False] * ng
        self._block_home.clear()
        self._block_gpu.clear()
        self._events.clear()

    def forget_home(self, block_idx: int) -> None:
        """Drop the home-slot mapping for *block_idx*."""
        self._block_home.pop(block_idx, None)

    def _acquire_home(self) -> int:
        if not self._home_free:
            raise RuntimeError("block_swap: out of home slots")
        return self._home_free.pop()

    def init_gpu(self, block_idx: int, block) -> None:
        """Place *block*'s weights onto a GPU ring slot at construction time."""
        gslot = self._ring_acquire()
        entries = self._ensure_entries(self._gpu_pool, self._gpu_built,
                                       gslot, block, self.device)
        self._swap_param(block, entries)
        self._block_gpu[block_idx] = gslot

    def init_home(self, block_idx: int, block) -> None:
        """Place *block*'s weights onto a home slot at construction time."""
        hslot = self._acquire_home()
        entries = self._ensure_entries(self._home_pool, self._home_built,
                                       hslot, block, "cpu")
        self._swap_param(block, entries)
        self._block_home[block_idx] = hslot

    # -- legacy move primitives (StreamingBlockPipeline) --------------------

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
        """Move *module* to CPU, waiting for its own in-flight prefetch if needed.

        Only the block being offloaded is waited on.  Other blocks' in-flight
        prefetches are intentionally left alone — they are still useful and will
        be consumed by ``sync_prefetch`` when those blocks enter the window.
        Cancelling them here (the previous behaviour) silently disabled all
        prefetch overlap and forced every load back to synchronous.
        """
        if block_idx is not None and block_idx in self._events:
            event = self._events.pop(block_idx)
            torch.cuda.current_stream().wait_event(event)
        move_module_to(module, "cpu")

    # -- slot-based block transfer (churn-free) -----------------------------

    def ensure_home(self, block_idx: int, block) -> int:
        """Ensure *block_idx* has a populated home slot; return its index."""
        if block_idx in self._block_home:
            return self._block_home[block_idx]
        hslot = self._acquire_home()
        entries = self._ensure_entries(self._home_pool, self._home_built,
                                       hslot, block, "cpu")
        if self._home is not None:
            self._home.read_into(block_idx, entries, block)
        self._block_home[block_idx] = hslot
        return hslot

    def load_block(self, block_idx: int, block, non_blocking: bool = True) -> None:
        """Copy *block*'s weights onto a GPU ring slot.

        When ``pin_memory`` is enabled, data flows through a pinned CPU
        staging ring for truly async H2D copies.  Otherwise it copies
        directly from block parameters to the GPU ring.
        """
        if block_idx in self._block_gpu:
            return
        self.ensure_home(block_idx, block)
        ridx = self._ring_acquire()

        gpu = self._ensure_entries(self._gpu_pool, self._gpu_built,
                                   ridx, block, self.device)
        nb = self._stream is not None and non_blocking

        if self.pin_memory and non_blocking and self._stream is not None:
            # ── Pinned staging path (prefetch only) ──────────────────────
            pin = self._pin.ensure(ridx, gpu, pin_memory=True)
            # Step 1: block params → pinned CPU staging
            for n, p in block.named_parameters():
                e = pin.get(n)
                if e is not None:
                    e.copy_from(p)
            # Step 2: pinned CPU → GPU on transfer stream
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                for n in pin:
                    pe, ge = pin[n], gpu.get(n)
                    if ge is None:
                        continue
                    if pe.is_qt:
                        ge.data.copy_(pe.data, non_blocking=True)
                        ge.scale.copy_(pe.scale, non_blocking=True)
                    else:
                        ge.data.copy_(pe.data, non_blocking=True)
            # Sync: compute stream waits for transfer stream BEFORE assign_to
            compute = torch.cuda.current_stream()
            ev = torch.cuda.Event()
            ev.record(self._stream)
            compute.wait_event(ev)
            nb = False  # pin path already synced; skip outer event recording
        elif self.pin_memory and non_blocking:
            # Pin enabled but no stream — fallback to direct sync copy
            for n, p in block.named_parameters():
                e = gpu.get(n)
                if e is None:
                    continue
                if e.is_qt:
                    e.data.copy_(p.data._qdata)
                    e.scale.copy_(p.data._params.scale)
                else:
                    e.data.copy_(p.data)
            # No event needed — synchronous copy completed above.
            nb = False
        else:
            # ── Direct path (no pin pool) ────────────────────────────────
            for n, p in block.named_parameters():
                e = gpu.get(n)
                if e is None:
                    continue
                if e.is_qt:
                    e.data.copy_(p.data._qdata, non_blocking=nb)
                    e.scale.copy_(p.data._params.scale, non_blocking=nb)
                else:
                    e.data.copy_(p.data, non_blocking=nb)

        # Assign GPU entries → block _parameters
        for n in gpu:
            gpu[n].assign_to(block, n)

        # Recycle home slot
        hslot = self._block_home.pop(block_idx, None)
        if hslot is not None:
            self._home_free.append(hslot)
        self._block_gpu[block_idx] = ridx
        if nb:
            event = torch.cuda.Event()
            event.record(self._stream)
            self._events[block_idx] = event

    def offload_block(self, block_idx: int, block, force: bool = False) -> bool:
        """Copy *block* back to a home slot; returns True if moved.

        Copies GPU slot entry → home slot entry directly (NOT through
        ``block.named_parameters()``, which may have been freed by
        ``free_module_storage`` during model switching).
        """
        event = self._events.pop(block_idx, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        gslot = self._block_gpu.pop(block_idx, None)
        if gslot is None:
            return False
        hslot = self._block_home.get(block_idx)
        if hslot is None:
            if self._home_free:
                hslot = self._home_free.pop()
            elif force:
                victim = min(self._block_home.keys())
                hslot = self._block_home.pop(victim)
                if self._home is not None:
                    self._home.release(victim)
            else:
                self._block_gpu[block_idx] = gslot
                return False
        # Build (or reuse) home slot entries from the block — at this point
        # the block may already have been freed (empty(0) params), so we
        # build from the GPU slot's metadata instead.
        gpu_entries = self._gpu_pool[gslot]
        home_entries = self._ensure_entries(self._home_pool, self._home_built,
                                            hslot, block, "cpu")
        # Copy data directly from GPU entries to home entries for params
        # present in both dicts.  This survives the block having been freed.
        for n in list(gpu_entries.keys() & home_entries.keys()):
            ge, he = gpu_entries[n], home_entries[n]
            if ge.is_qt:
                he.data.copy_(ge.data)
                he.scale.copy_(ge.scale)
            else:
                he.data.copy_(ge.data)
            he.assign_to(block, n)
        self._block_home[block_idx] = hslot
        return True

    # -- prefetch ------------------------------------------------------------

    def start_prefetch(self, block_idx: int, block) -> None:
        """Async-load *block* ahead of the window (no-op if already loaded)."""
        if block_idx in self._block_gpu or block_idx in self._events:
            return
        self.load_block(block_idx, block, non_blocking=True)

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
# _DiskPrefetcher — lazy disk→RAM→GPU pipeline
# ═══════════════════════════════════════════════════════════════════════════

class _DiskPrefetcher:
    """Background disk→RAM loader with on-demand CPU RAM eviction.

    Reads block weights from safetensors on a thread pool, populates the
    model's parameter data, and frees unused blocks' CPU RAM when they
    leave the GPU window + prefetch range.

    Parameters
    ----------
    model:
        The diffusion model (must have ``.blocks`` list and ``_lora_groups``).
    block_reader:
        Random-access safetensors reader.
    max_workers:
        Thread pool size (typically 2).
    """

    def __init__(
        self,
        model,
        block_reader,
        max_workers: int = 2,
    ):
        self._model = model
        self._blocks = model.blocks
        self._reader = block_reader
        self._lora_groups = getattr(model, "_lora_groups", None)

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="disk-prefetch",
        )
        self._pending: dict[int, Future] = {}  # block_idx → Future
        # Blocks whose weights are confirmed resident in CPU RAM (worker
        # finished, or a synchronous load completed).  Lets ``ensure_ram``
        # return immediately instead of re-reading from disk when a completed
        # future was reclaimed from ``_pending``.
        self._loaded: set[int] = set()
        self._lock = threading.Lock()
        self._shut_down = False

    def ensure_ram(self, block_idx: int) -> None:
        """Guarantee block *block_idx* is in CPU RAM (block until ready)."""
        # Already resident? No I/O, no locking past the check.
        with self._lock:
            if block_idx in self._loaded:
                return
            fut = self._pending.pop(block_idx, None)
        if fut is not None:
            fut.result()  # wait for disk read (re-raises worker errors)
        else:
            # Not loading yet — load synchronously on this thread.
            self._load_immediate(
                DiskLoadRequest(block_idx=block_idx, group_key=f"blocks.{block_idx}")
            )
        with self._lock:
            self._loaded.add(block_idx)

    def start_ram_load(self, block_idx: int) -> None:
        """Begin loading *block_idx* from disk in a background thread."""
        if self._shut_down:
            return  # pool torn down; caller must not dispatch new work
        req = DiskLoadRequest(block_idx=block_idx, group_key=f"blocks.{block_idx}")
        with self._lock:
            # Skip if already resident or an in-flight load exists.  Done
            # futures stay in ``_pending`` until ``ensure_ram`` consumes them,
            # so a completed prefetch is never silently dropped and re-read.
            if block_idx in self._loaded or block_idx in self._pending:
                return
            fut = self._executor.submit(self._load_immediate, req)
            self._pending[block_idx] = fut
            fut.add_done_callback(self._on_load_done)

    def evict_ram(self, block_idx: int) -> None:
        """Free block *block_idx*'s CPU RAM, discarding its weight data.

        Safe to call only after the block's GPU copy is confirmed.
        """
        try:
            block = self._blocks[block_idx]
        except IndexError:
            return
        free_module_storage(block)

    def evict_all_ram(self) -> None:
        """Free ALL blocks' CPU RAM."""
        for block_idx in range(len(self._blocks)):
            self.evict_ram(block_idx)
        with self._lock:
            self._loaded.clear()

    def cancel_all(self) -> None:
        """Abort all in-flight disk reads."""
        with self._lock:
            for fut in self._pending.values():
                fut.cancel()
            self._pending.clear()

    def _on_load_done(self, fut: Future) -> None:
        """Record a successfully completed load so ``ensure_ram`` won't re-read.

        Bound to each future via ``add_done_callback``; runs on the worker
        thread.  Errors are deliberately *not* swallowed here — they surface
        when ``ensure_ram`` calls ``fut.result()`` (or trigger a synchronous
        re-read that re-raises).
        """
        if fut.exception() is None:
            with self._lock:
                # ``_pending`` may already hold a newer entry for this block;
                # ``_loaded`` is keyed by block_idx, so just mark it resident.
                for bi in list(self._pending):
                    if self._pending.get(bi) is fut:
                        self._loaded.add(bi)
                        break

    def join(self, timeout: float = 30.0) -> None:
        """Block until every in-flight disk load finishes (no pool teardown).

        Call *before* freeing block RAM: a worker still executing
        ``_load_immediate`` could otherwise write into a parameter that
        ``evict_all_ram`` has shrunk to ``empty(0)`` -> copy_ crash or NaN.
        """
        with self._lock:
            futs = list(self._pending.values())
        for f in futs:
            try:
                f.result(timeout=timeout)
            except Exception:
                # Worker error — let ensure_ram / the forward re-raise if the
                # block is actually needed; nothing to free here.
                pass

    def shutdown(self) -> None:
        """Cancel I/O and shut down the thread pool.

        ``wait=True`` guarantees every worker has finished ``_load_immediate``
        (and its ``_stream_load_group`` writes into model params) *before* the
        caller frees any block storage. ``Future.cancel()`` is a no-op for
        already-running tasks, so without this a worker could still be writing
        while ``evict_all_ram()`` swaps ``param.data`` for an ``empty(0)``
        tensor -> copy_ crash or NaN weights.
        """
        try:
            self.cancel_all()
            try:
                self._executor.shutdown(wait=True, timeout=30)
            except Exception:
                # Last-resort: don't block teardown if the pool is wedged.
                self._executor.shutdown(wait=False)
            # Belt-and-suspenders: ensures any still-running worker finished
            # writing before evict_all_ram() frees the parameters.
            self.join()
            self.evict_all_ram()
        finally:
            # Always mark the prefetcher dead so evict_all()/start_ram_load()
            # can't dispatch or free against a torn-down pool.
            self._shut_down = True

    # ── internal ───────────────────────────────────────────────────────

    def _load_immediate(self, req: DiskLoadRequest) -> None:
        """Read *req.block_idx* from disk into CPU RAM (called from thread)."""
        from ..models.wan_model import _stream_load_group
        try:
            tensors = self._reader.read_block(req.block_idx)
        except Exception as e:
            logger.error(
                "[DiskPrefetch] Failed to read block %d: %s", req.block_idx, e
            )
            raise
        _stream_load_group(
            self._model, req.group_key, tensors, self._lora_groups
        )


# ═══════════════════════════════════════════════════════════════════════════
# BlockHome — unified block-weight source (resident RAM or disk)
# ═══════════════════════════════════════════════════════════════════════════

class BlockHome:
    """Source of a transformer block's weights.

    Resident mode (weights already in CPU RAM) and lazy mode (weights on
    disk) are expressed through this single interface, so the swap engine
    never branches on mode.  The engine owns a fixed pool of CPU "home"
    buffers; ``read_into`` populates one such buffer with a block's weights
    on demand (resident: already there; disk: stream from safetensors).
    """

    def __init__(self, model=None):
        self._model = model

    def read_into(self, block_idx: int, slot: dict, model_blocks) -> None:
        """Fill *slot* (``name -> Tensor``) with *block_idx*'s weights."""
        raise NotImplementedError

    def release(self, block_idx: int) -> None:
        """Drop cached weights for *block_idx* (no-op for resident RAM)."""
        pass

    def shutdown(self) -> None:
        """Tear down any background I/O (no-op for resident RAM)."""
        pass

    def join(self) -> None:
        """Wait for in-flight I/O (no-op for resident RAM)."""
        pass


class RamHome(BlockHome):
    """Resident mode: weights live in the model's CPU parameters.

    At manager init every block's weights are copied into a fixed home
    buffer owned by the transfer engine; afterwards the weights always
    reside in that buffer, so ``read_into`` is a no-op.
    """

    def read_into(self, block_idx: int, slot: dict, model_blocks) -> None:
        # Weights already live in the engine's home buffer (populated at
        # init, or written back on offload).  Nothing to fetch.
        return


class DiskHome(BlockHome):
    """Lazy mode: weights are streamed from safetensors on demand."""

    def __init__(self, model, block_reader, max_workers: int = 4):
        self._model = model
        self._disk = _DiskPrefetcher(model, block_reader, max_workers)

    def read_into(self, block_idx: int, slot: dict, block) -> None:
        # *block* is the manager's model block (identical object to
        # ``self._model.blocks[block_idx]``); ``ensure_ram`` populates its
        # parameters in place from safetensors.
        #
        # The home slot's ``SlotEntry`` entries are pre-allocated by
        # ``_ensure_entries`` from the block's initial (possibly
        # pre-materialised) parameter set.  For fp8 deferred-weight ops the
        # block's ``named_parameters()`` is empty *before* ``ensure_ram``
        # and complete *afterwards* — ``SlotEntry.copy_from`` handles both
        # regular and quantized tensor types transparently.
        if block_idx not in self._disk._loaded:
            self._disk.ensure_ram(block_idx)
        for n, p in block.named_parameters():
            e = slot.get(n)
            if e is None:
                # Deferred-weight materialisation: the param didn't exist
                # when the slot was built.  Build a new CPU SlotEntry now.
                slot[n] = SlotEntry.empty_like(
                    p, device="cpu", pin_memory=p.is_pinned(),
                )
                slot[n].copy_from(p)
            else:
                e.copy_from(p)

    def release(self, block_idx: int) -> None:
        self._disk.evict_ram(block_idx)
        self._disk._loaded.discard(block_idx)

    def shutdown(self) -> None:
        self._disk.shutdown()

    def join(self) -> None:
        self._disk.join()


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
        pin_memory: bool = True,
        block_reader=None,
        max_disk_workers: int = 4,
    ):
        self.model = model
        self._blocks = model.blocks
        self._prefetch = bool(prefetch)
        total = len(self._blocks)

        # Resolve pre-warmed count.  Resident mode (no block_reader) pre-warms
        # a full window onto GPU so that, at steady state, exactly
        # ``total - window`` blocks rest in the CPU home pool — host RAM stays
        # flat at ``(N-W)`` with no periodic allocation churn.  Lazy mode
        # pre-warms nothing (blocks are read from disk on demand).
        prewarmed = window_size if block_reader is None else 0

        # Unified weight source: RamHome (resident RAM) or DiskHome (lazy
        # disk).  The swap engine never branches on mode afterwards.
        if block_reader is not None:
            self._home: BlockHome = DiskHome(model, block_reader, max_disk_workers)
            self._lazy = True
        else:
            self._home: BlockHome = RamHome(model)
            self._lazy = False

        # ---- sub-components ------------------------------------------------
        self._window = _BlockWindow(total, window_size, prewarmed)
        self._xfer = _TransferEngine(
            device, prefetch, prefetch_count, pin_memory,
            n_home_slots=max(total - window_size, 1),
            n_gpu_slots=window_size + prefetch_count,
            template_block=self._blocks[0] if total else None,
            home=self._home,
        )

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
            "target VRAM ~%.0f MB, prefetch=%s, count=%d, pin=%s, "
            "prewarmed=%d, lazy=%s",
            total, window_size, block_mb, window_size * block_mb,
            prefetch, prefetch_count, pin_memory, prewarmed, self._lazy,
        )

        self._init_state()

    # ------------------------------------------------------------------
    # Public attribute accessors (used by the model's forward pass)
    # ------------------------------------------------------------------

    @property
    def window(self) -> int:
        """Sliding-window size in blocks (GPU-resident count).

        Mirrors ``_window.window_size`` so the forward pass can cheaply
        check whether the configured window changed without reaching into
        the internal ``_window`` state object.
        """
        return self._window.window_size

    @property
    def device(self) -> "torch.device | str":
        """GPU compute device blocks/activations are streamed to.

        Mirrors ``_xfer.device`` (defaults to ``"cuda"`` when the manager
        is constructed without an explicit device).
        """
        return self._xfer.device

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        """Establish deterministic starting state.

        Reset the slot pools, then bring every block to CPU (undo any
        pre-warm done by the streaming loader) so the engine's slot pools
        are the sole owner of block placement.  Finally place ``prewarmed``
        blocks onto GPU slots (resident mode); the rest stay in home slots.
        Lazy-mode blocks are left for on-demand disk reads when first used.
        """
        self._xfer.cancel_all()
        self._xfer.reset()

        # Undo any pipeline pre-warm: every block starts on CPU.
        for block in self._blocks:
            try:
                if any(p.is_cuda for p in block.parameters()):
                    move_module_to(block, "cpu")
            except Exception:
                pass

        prewarmed = self._window.prewarmed
        for i, block in enumerate(self._blocks):
            if i < prewarmed:
                self._xfer.init_gpu(i, block)
            elif isinstance(self._home, RamHome):
                self._xfer.init_home(i, block)
            # DiskHome off-GPU blocks: left for lazy load on first use.

        # Ensure peripherals start on CPU (model's home device).
        for mod in self._peri.values():
            if mod is not None:
                move_module_to(mod, "cpu")
        self._budget.maybe_flush(self._xfer.device, reserve_blocks=0)

        # Pin entries are NOT pre-built here — they are built lazily inside
        # ``load_block`` AFTER ``ensure_home`` has materialised the block's
        # weights on CPU, guaranteeing the ``SlotEntry`` keys match the real
        # parameter set (fp8 deferred-weight layers are only registered after
        # ``load_state_dict`` runs inside ``ensure_ram``).

    # ------------------------------------------------------------------
    # Public API — called from the model's forward pass
    # ------------------------------------------------------------------

    def prepare(self, block_idx: int) -> None:
        """Ensure block *block_idx* and its window are on GPU.

        Offloads blocks that have fallen outside the window and loads newly
        needed blocks.  Off-GPU blocks live in the fixed CPU home pool; the
        swap engine copies them to/from GPU slots with ``copy_`` — no new
        tensors are allocated, so host RAM stays flat at ``(N-W)``.  In lazy
        mode the home pool is populated from disk on demand (inside the
        engine's ``ensure_home``).

        Call at the top of each loop iteration *before* using the block.
        """
        if block_idx == 0:
            self.prepare_blocks_entry()

        to_offload = sorted(self._window.to_offload(block_idx))
        to_load_now = sorted(self._window.to_load(block_idx))
        did_offload = bool(to_offload)

        # Interleave load (home→GPU) and offload (GPU→home).  Crucially we
        # *load first*, so the incoming block frees its home slot before the
        # outgoing block needs one — the slot pool stays balanced at (N-W)
        # and host RAM never spikes toward the whole model.  Both sides use
        # the recycled slot pools, so the transient VRAM/RAM churn is bounded
        # to ~one block.
        #
        # A trailing block whose offload finds no free home slot is *deferred*
        # (left on GPU) rather than raising — at the tail of a pass the window
        # has shrunk below W, so more than N-W blocks would be off-GPU; the
        # deferred blocks simply linger on GPU and are fully evicted at
        # ``evict_all``.  This is what keeps RAM flat at (N-W) throughout.
        n = max(len(to_offload), len(to_load_now))
        for k in range(n):
            if k < len(to_load_now):
                j = to_load_now[k]
                if not self._xfer.sync_prefetch(j):
                    # Async load on the transfer stream, then sync the
                    # just-recorded event so the block's weights are fully
                    # resident on GPU before forward.  Using the transfer
                    # stream (``non_blocking=True``) lets the copy overlap
                    # with any remaining offload work.
                    self._xfer.load_block(j, self._blocks[j], non_blocking=True)
                    self._xfer.sync_prefetch(j)  # wait for async H2D
                self._window.mark_loaded(j)
            if k < len(to_offload):
                i = to_offload[k]
                if self._xfer.offload_block(i, self._blocks[i]):
                    self._window.mark_offloaded(i)
                # deferred -> block stays on GPU; retry on the next prepare

        # Bulletproof residency guarantee.
        for j in sorted(self._window.needed(block_idx)):
            if self._xfer._block_gpu.get(j) is None:
                if not self._xfer.sync_prefetch(j):
                    self._xfer.load_block(j, self._blocks[j], non_blocking=True)
                    self._xfer.sync_prefetch(j)
                self._window.mark_loaded(j)

        if did_offload:
            self._budget.maybe_flush(self._xfer.device, reserve_blocks=2)

    def prefetch_next(self, block_idx: int) -> None:
        """Start async H2D copies for blocks just beyond the window.

        Call after ``prepare(block_idx)`` and before the compute for
        *block_idx*.  Copies run on the transfer stream in parallel with
        the current block's forward pass.  In lazy mode the home pool is
        populated from disk as a side effect of the load.  No-op when
        prefetch is disabled.
        """
        if not self._prefetch:
            return
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
        """Move all blocks and peripherals back to CPU home slots.

        For lazy mode, also free the home-pool CPU RAM (disk is the source
        of truth, so the buffers are not needed until the next load).
        """
        self._xfer.cancel_all()
        self._xfer.sync_all()
        for i in sorted(self._window.on_gpu):
            self._xfer.offload_block(i, self._blocks[i], force=True)
        self._window.clear()
        for name in self._PERIPHERAL_NAMES:
            self._offload_peripheral(name)

        # Lazy mode: drop the in-RAM weights (disk-backed) and let the engine
        # re-populate the home slot on the next load.  Resident mode keeps
        # them — they are the model's persistent CPU copy.
        if self._home is not None and not isinstance(self._home, RamHome):
            for i in range(self._window.total):
                self._home.release(i)
                self._xfer.forget_home(i)
            self._home.join()

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Drop all pool tensor references — without this the home & ring
        # buffer SlotEntry tensors keep RAM/VRAM pinned across model switches.
        self._xfer.clear_pools()

    def shutdown(self) -> None:
        """Deterministic full teardown, safe to call on model unload/switch.

        Order matters on Windows: stop the disk workers *first* and let them
        finish (so no thread is still writing block weights into the model),
        then synchronise every CUDA stream, then evict everything.
        """
        try:
            self._xfer.cancel_all()
        except Exception:
            pass
        try:
            self._home.join()
        except Exception:
            pass
        self._xfer.sync_all()
        self.evict_all()
        try:
            self._home.shutdown()
        except Exception:
            pass
        # Drop all pool tensor references and block list so the old model's
        # Parameter objects have zero live references -> GC reclaims RAM+VRAM.
        self._xfer.clear_pools()
        self._blocks = []
        self._home = None

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
        if self._budget.block_mb <= 0:
            # A block group with no parameters (e.g. a norm-only slice)
            # would divide by zero below; nothing to prefetch.
            return 0
        free_mb = self._budget.free_mb(self._xfer.device)
        reserve_mb = self._budget.block_mb * 2
        avail_mb = max(0.0, free_mb - reserve_mb)
        max_by_mem = int(avail_mb / (self._budget.block_mb * 1.2))
        return max(0, min(self._xfer.prefetch_count, max_by_mem))

