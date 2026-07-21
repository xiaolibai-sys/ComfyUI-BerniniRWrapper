"""
TeaCache — skip redundant transformer block computations during sampling.

Ported from WanVideoWrapper.  When consecutive sampling steps produce
near-identical hidden states in early transformer blocks, the later
blocks are skipped and their cached outputs are reused.  1.5-2× speedup
with negligible quality loss on Wan 2.1 architecture.

Step counting and cache-allowance gating are handled externally by
the sampler via ``step()``.  Per-window tracking prevents CFG double-calls
and context-window multi-calls from corrupting the cache.
"""

from __future__ import annotations

from typing import Optional

import torch

from .log import get_logger as _get_logger

logger = _get_logger("TeaCache")
# ---------------------------------------------------------------------------
# Defaults — single source of truth for both the TeaCache engine and the
# ComfyUI widget definitions in sampler_teacache.py.
# ---------------------------------------------------------------------------

DEFAULT_START_BLOCK = 3
DEFAULT_MAX_SKIP_BLOCKS = 15
DEFAULT_REL_L1_THRESH = 0.08
DEFAULT_WARMUP_STEPS = 1   # skip first step (structure formation)
DEFAULT_COOLDOWN_STEPS = 2  # skip last 2 steps (detail refinement)


def _l1(x: torch.Tensor, y: torch.Tensor) -> float:
    """Scalar L1 distance (mean over all dims), returned as a Python float.

    ``(x - y).abs().float().mean().item()`` — single float32
    intermediate instead of converting both inputs separately.
    """
    return (x - y).abs().float().mean().item()


class TeaCache:
    """Attaches caching hooks to a WanModel's transformer blocks.

    Usage::

        cache = TeaCache(model, start_block=3, max_skip=15, thresh=0.08)
        cache.reset(total_steps=20)
        for step in range(total_steps):
            cache.step()          # advance counter (once per sampler step)
            model(...)
        cache.detach()
    """

    def __init__(
        self,
        model,
        *,
        start_block: int = DEFAULT_START_BLOCK,
        max_skip_blocks: int = DEFAULT_MAX_SKIP_BLOCKS,
        rel_l1_thresh: float = DEFAULT_REL_L1_THRESH,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        cooldown_steps: int = DEFAULT_COOLDOWN_STEPS,
        batch_size: int = 1,
    ):
        dm = self._get_wan_model(model)
        if dm is None:
            raise RuntimeError(
                "TeaCache: cannot locate WanModel on the given model patcher."
            )

        n_blocks = len(dm.blocks)
        self._wan = dm

        # ── torch.compile incompatibility guard ─────────────────────
        # If the model was compiled (BerniniR_CompileModel), the compiled
        # transformer_forward has *inlined* each block.forward into the traced
        # graph.  TeaCache patches block.forward at sample time, so a patched
        # block.forward would never be invoked → TeaCache silently does nothing
        # (no speedup, no error).  Restore the eager forward so our block hooks
        # actually fire.  This mirrors the existing block-swap guard in
        # InjectionContext.apply_block_swap.
        _orig_tf = getattr(self._wan, "_original_transformer_forward", None)
        _orig_fo = getattr(self._wan, "_original_forward_orig", None)
        if _orig_tf is not None:
            self._wan.transformer_forward = _orig_tf
            logger.warning(
                "TeaCache is incompatible with torch.compile — "
                "compile disabled for this sampling run."
            )
        elif _orig_fo is not None:
            self._wan.forward_orig = _orig_fo
            logger.warning(
                "TeaCache is incompatible with torch.compile — "
                "compile disabled for this sampling run."
            )

        self._start = max(0, min(start_block, n_blocks - 1))
        self._end = min(self._start + max_skip_blocks, n_blocks)
        self._thresh = rel_l1_thresh
        self._warmup = warmup_steps
        self._cooldown = cooldown_steps
        self._batch_gt_1 = batch_size > 1
        if self._batch_gt_1:
            logger.warning(
                "TeaCache batch_size=%d > 1: comparing only "
                "batch[0:1] for cache decisions.", batch_size
            )

        # State — step counter is managed externally via step()
        self._step: int = 0
        self._total_steps: int = 0
        self._skipping: bool = False
        self._cache_output_pending: bool = False
        self._orig_forwards: dict[int, callable] = {}
        self._patched: bool = False

        # Per-window cache (key 0 = no context window; tuple = window frames)
        self._window_cache: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        self._window_last_consumed: dict[object, int] = {}
        # Active-window temporaries loaded at start_block, saved at end-1
        self._active_window_key: object = None
        self._active_cached_input: Optional[torch.Tensor] = None
        self._active_cached_output: Optional[torch.Tensor] = None

        self._patch()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reset(self, total_steps: int):
        """Call at the start of every sampling run."""
        self._step = 0
        self._total_steps = total_steps
        self._skipping = False
        self._cache_output_pending = False
        self._window_cache.clear()
        self._window_last_consumed.clear()
        self._active_window_key = None
        self._active_cached_input = None
        self._active_cached_output = None

    def step(self):
        """Advance the step counter (call once per denoising step, NOT per model call)."""
        self._step += 1
        self._skipping = False  # belt-and-suspenders: never leak skip into next step

    def detach(self):
        """Restore original block forwards and release references (idempotent)."""
        if not self._patched:
            return
        for i, orig in self._orig_forwards.items():
            if i < len(self._wan.blocks):
                self._wan.blocks[i].forward = orig
        self._orig_forwards.clear()
        self._window_cache.clear()
        self._window_last_consumed.clear()
        self._active_cached_input = None
        self._active_cached_output = None
        self._wan = None
        self._patched = False
        logger.info("TeaCache detached.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _get_wan_model(model):
        """Walk model patcher → diffusion_model → WanModel."""
        for _ in range(3):
            candidate = getattr(model, "diffusion_model", None) or getattr(model, "model", None)
            if candidate is not None:
                model = candidate
            else:
                break
        return model if hasattr(model, "blocks") else None

    def _active(self) -> bool:
        return self._warmup < self._step <= self._total_steps - self._cooldown

    def _patch(self):
        if self._patched:
            return
        for i in range(self._start, self._end):
            blk = self._wan.blocks[i]
            self._orig_forwards[i] = blk.forward
            blk.forward = self._hook(blk, i)
        self._patched = True
        logger.info(
            "TeaCache: blocks [%d, %d) of %d, thresh=%.3f, warmup=%d",
            self._start, self._end, len(self._wan.blocks),
            self._thresh, self._warmup,
        )

    def _hook(self, block, idx: int):
        """Return a ``forward`` replacement that optionally skips the block.

        Maintains per-window cache isolation: reads the current context
        window identity from ``transformer_options["_context_window"]``
        (set by the context-window wrapper) so that each temporal window
        gets its own independent cache state.
        """
        orig = self._orig_forwards[idx]

        def forward(x, e, freqs, context, context_img_len=None,
                     transformer_options=None):
            # ── start_block: resolve window identity + decide skip ─────
            if idx == self._start:
                # Which context window are we processing right now?
                win_key = (transformer_options or {}).get("_context_window") or 0

                # First call for THIS window in THIS step?
                fresh = (
                    win_key not in self._window_last_consumed
                    or self._step != self._window_last_consumed[win_key]
                )

                if fresh:
                    self._window_last_consumed[win_key] = self._step
                    self._active_window_key = win_key

                    # Load cached state for this window (if any)
                    if win_key in self._window_cache:
                        self._active_cached_input, self._active_cached_output = (
                            self._window_cache[win_key]
                        )
                    else:
                        self._active_cached_input = None
                        self._active_cached_output = None

                    # Decide: skip or compute?
                    # When batch > 1, compare only batch[0:1] so the
                    # cache state stays consistent with the single-element
                    # L1 threshold (multiplying across batch inflates L1).
                    x_cmp = x[:1] if self._batch_gt_1 else x
                    if (
                        self._active()
                        and self._active_cached_input is not None
                        and _l1(x_cmp, self._active_cached_input) < self._thresh
                    ):
                        self._skipping = True
                    else:
                        self._skipping = False
                        self._active_cached_input = x_cmp.detach()
                        self._cache_output_pending = True
                else:
                    # Same step, same window — non-fresh (e.g. CFG uncond).
                    # Force compute to preserve CFG quality.  The cond pass
                    # already updated the cache; skipping here would risk
                    # reusing a stale cond-only cached output for uncond.
                    self._active_window_key = win_key
                    self._skipping = False

            # ── Pass-through path (skip) ─────────────────────────────
            if self._skipping:
                result = (
                    self._active_cached_output if idx == self._start else x
                )
                # When batch > 1, the cached output has shape [1,...],
                # expand to match the full-batch input.
                if idx == self._start and self._batch_gt_1:
                    result = result.expand(x.shape[0], -1, -1)
                if idx == self._end - 1:
                    self._skipping = False
                return result

            # ── Compute path ─────────────────────────────────────────
            result = orig(x, e=e, freqs=freqs, context=context,
                          context_img_len=context_img_len,
                          transformer_options=transformer_options)

            if idx == self._end - 1 and self._cache_output_pending:
                # When batch > 1, cache only batch[0:1] output so cache
                # shape stays consistent with the L1 comparison slice.
                result_cache = result[:1].detach() if self._batch_gt_1 else result.detach()
                self._window_cache[self._active_window_key] = (
                    self._active_cached_input,
                    result_cache,
                )
                self._cache_output_pending = False

            return result

        # Tell Dynamo to skip this function entirely — the
        # data-dependent skip/compute decision cannot be traced
        # without triggering recompilation on every state change.
        forward = torch._dynamo.disable(forward)
        return forward
