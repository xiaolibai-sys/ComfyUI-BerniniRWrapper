"""
Bernini-R self-contained sampling module.

Replaces ComfyUI's ``sampling_function`` / ``calc_cond_batch`` / ``cfg_function``
chain with a clean, extensible architecture.  All guidance modes live in one
file and the forward pass is a direct ``model.apply_model()`` call — no area
conditioning, no ControlNet multi-pass, no hook-group batching.

Supported guidance modes (extensible):
  - ``CFG``       — standard classifier-free guidance
  - ``APG``       — Adaptive Projected Guidance (parallel-component suppression)
  - ``RAAG``      — Ratio-Aware Adaptive Guidance (ρ‑based early‑step dampening)
  - ``S2``        — Stochastic Self‑Guidance (block‑dropped sub‑network repulsion)

Existing features that work *through* the model unchanged:
  - Context windows    (via ``model_function_wrapper``)
  - TeaCache           (block‑level hooks)
  - NAG                (``transformer_options["nag_context"]``)
  - NTK RoPE scaling   (``transformer_options["_rope_ntk_scale"]``)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.sampler_helpers
from comfy.samplers import get_area_and_mult, cond_cat

from ..utils.model_manager import BerniniRModelHandle, _cache_evict_patcher
from ..utils.vram import collect_garbage, log_memory, release_model_ram, log_system_ram
from ..utils.types import (
    BerniniContext,
    BerniniBlockSwap,
    BerniniGuidanceConfig,
    Conditioning,
    EncodedSegment,
    SegmentWindow,
)
from ..utils.injection import InjectionContext
# Context-window / segment prompt-travel wrapper lives in sampler.py. Imported
# here at top level now that sampler.py no longer imports bernini_sampling at
# module init (it lazily imports bernini_sample inside BerniniR_KSampler.sample),
# so there is no circular-import cycle.
from .sampler import _build_context_window_wrapper

logger = logging.getLogger(__name__)


def _resample_schedule(values: list[float], target_steps: int) -> list[float]:
    """Linearly resample a guidance schedule to *target_steps*.

    If the schedule node was configured for a different step count than the
    sampler, this auto-corrects the mismatch so users don't have to keep
    two ``steps`` widgets in sync manually.
    """
    old = len(values)
    if old == target_steps or old <= 1:
        return list(values)
    result = []
    for i in range(target_steps):
        t = i * (old - 1) / max(target_steps - 1, 1)
        lo = int(t)
        hi = min(lo + 1, old - 1)
        frac = t - lo
        result.append(round(values[lo] * (1.0 - frac) + values[hi] * frac, 2))
    return result


# ---------------------------------------------------------------------------
# Model wrapper — replaces CFGGuider for k-diffusion
# ---------------------------------------------------------------------------

class BerniniModelWrapper:
    """Drop-in replacement for ``CFGGuider`` in the k-diffusion sampling loop.

    k-diffusion samplers call ``model(x, sigma, **extra_args)`` and expect a
    denoised prediction back.  This wrapper:

    1. Runs 2 forward passes (uncond + cond), or 3 for S² mode.
    2. Combines predictions according to the active guidance mode.
    3. Delegates ``model_sampling``, ``process_latent_in/out`` etc. to the
       underlying ``BaseModel`` so that ``KSAMPLER.sample()`` noise scaling
       and latent pre/post-processing work transparently.
    """

    def __init__(
        self,
        inner_model,
        cond_pos: list,
        cond_neg: list | None,
        cfg: float = 6.0,
        mode: str = "CFG",
        schedule: list[float] | None = None,
        s2_omega: float = 1.0,
        stg_scale: float = 1.0,
        stg_mode: str = "A",
        stg_block_idx: list[int] | None = None,
        raag_alpha: float = 1.0,
        apg_eta: float = 0.15,
        apg_rescale: bool = True,
        apg_momentum: float = 0.0,
        z2_collapse: float = 0.3,
        seed: int = 0,
        model_options: dict | None = None,
        noise: torch.Tensor | None = None,
        total_steps: int = 1,
        injection: InjectionContext | None = None,
    ):
        self.inner_model = inner_model
        self.cond_pos = cond_pos
        self.cond_neg = cond_neg
        self._noise = noise
        self._total = max(1, total_steps)
        self.cfg = cfg
        self.mode = mode.upper()
        self._schedule = schedule if schedule is not None else [cfg]
        self._s2_omega = s2_omega
        self._stg_scale = stg_scale
        self._stg_mode = stg_mode.upper()
        self._stg_block_idx = stg_block_idx or []
        self._raag_alpha = raag_alpha
        self._apg_eta = apg_eta
        self._apg_rescale = apg_rescale
        self._apg_momentum = apg_momentum
        self._z2_collapse = z2_collapse
        self._seed = seed
        self._model_options = model_options or {}

        # Step counter — incremented per denoising step (not per forward).
        self._step = 0

        # APG momentum state (per‑step reverse momentum).
        self._prev_apg: torch.Tensor | None = None

        # ── Pre-computed dispatch index (avoids per-step string compare) ──
        _m = mode.upper()
        if _m.startswith("STG"):
            self._dispatch = 5  # STG
        elif _m == "S2":
            self._dispatch = 4
        elif _m == "APG":
            self._dispatch = 3
        elif _m == "RAAG":
            self._dispatch = 2
        elif _m == "Z2":
            self._dispatch = 1
        else:
            self._dispatch = 0  # CFG

        # ── Differential diffusion state — from InjectionContext ─────
        if injection is not None and self._noise is not None:
            self._dd_mask_mode = injection.dd_mask_mode
            self._dd_src_raw = injection.dd_src_latent
            self._dd_mask_raw = injection.dd_edit_mask
        elif injection is not None:
            self._dd_mask_mode = "anneal"
            self._dd_src_raw = None
            self._dd_mask_raw = None
        else:
            # Legacy path (tests / backward compat)
            self._dd_mask_mode = "anneal"
            self._dd_src_raw = None
            self._dd_mask_raw = None

        # Lazy device transfer state
        self._dd_src_prepared: torch.Tensor | None = None
        self._dd_mask_prepared: torch.Tensor | None = None
        self._dd_noise_prepared: torch.Tensor | None = None

    def _prepare_dd_state(self, device: torch.device, dtype: torch.dtype) -> None:
        """Lazy one-shot transfer of differential diffusion tensors to the
        sampling device.  Called on the first denoising step; subsequent steps
        reuse the cached tensors (device / dtype are stable across a run)."""
        if self._dd_src_raw is None:
            return
        self._dd_src_prepared = self._dd_src_raw.to(device=device, dtype=dtype)
        self._dd_mask_prepared = self._dd_mask_raw.to(device=device, dtype=dtype)
        self._dd_noise_prepared = self._noise.to(device=device, dtype=dtype)

    # -- delegate model_sampling etc. to the inner BaseModel ---------------

    @property
    def model_sampling(self):
        return self.inner_model.model_sampling

    def process_latent_in(self, latent):
        return self.inner_model.process_latent_in(latent)

    def process_latent_out(self, latent):
        return self.inner_model.process_latent_out(latent)

    @property
    def model_patcher(self):
        return None

    # -- k-diffusion interface ---------------------------------------------

    def __call__(
        self, x: torch.Tensor, sigma: torch.Tensor, **extra_args
    ) -> torch.Tensor:
        """k-diffusion entry point: noisy latent → denoised prediction."""
        model_options = extra_args.get("model_options", self._model_options)

        # ── Resolve per-step ceiling scale from schedule ─────────────
        step = self._step
        if step < len(self._schedule):
            cfg_step = self._schedule[step]
        else:
            cfg_step = self._schedule[-1]
        self._step += 1

        # ── Differential diffusion (per-step latent re-noise + blend) ──
        # Source and mask tensors are pre-extracted in __init__ and lazily
        # transferred to device on the first step.  No per-step cond dict reads.
        if self._dd_src_raw is not None:
            if self._dd_src_prepared is None:
                self._prepare_dd_state(x.device, x.dtype)

            src0 = self._dd_src_prepared
            if src0.shape[0] == 1 and x.shape[0] > 1:
                src0 = src0.repeat(x.shape[0], 1, 1, 1, 1)
            if src0.shape[2:] == x.shape[2:]:
                keep = 1.0 - self._dd_mask_prepared
                if keep.shape[0] == 1 and x.shape[0] > 1:
                    keep = keep.repeat(x.shape[0], 1, 1, 1, 1)
                if self._dd_mask_mode == "anneal":
                    th = float(step) / self._total
                    keep = (keep > th).to(x.dtype)
                sigma_b = sigma.view(-1, *([1] * (x.ndim - 1)))
                src_noisy = (1.0 - sigma_b) * src0 + sigma_b * self._dd_noise_prepared
                x = src_noisy * keep + x * (1.0 - keep)
            else:
                logger.warning(
                    "[BerniniR] Differential diffusion skipped: "
                    "source shape %s does not match latent shape %s.",
                    list(src0.shape), list(x.shape),
                )

        # ── Batched forward (cond + uncond) ──────────────────────────
        preds = self._forward_batched(
            x, sigma, [self.cond_pos, self.cond_neg], model_options,
        )
        pred_cond, pred_uncond = preds[0], preds[1]

        # ── Forward: sub (S² only) ───────────────────────────────────
        pred_sub = None
        if self._dispatch == 4:  # S2
            pred_sub = self._forward_single(
                x, sigma, self.cond_pos, model_options, sub_network=True,
            )

        # ── Forward: cond_skip (STG only) ────────────────────────────
        pred_cond_skip = None
        if self._dispatch == 5:  # STG
            pred_cond_skip = self._forward_single(
                x, sigma, self.cond_pos, model_options, stg_skip=True,
            )

        # ── Combine ──────────────────────────────────────────────────
        return self._combine(pred_uncond, pred_cond, pred_sub, pred_cond_skip,
                              cfg_step)

    def _forward_batched(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        conds: list[list],
        model_options: dict,
        sub_network: bool = False,
        stg_skip: bool = False,
    ) -> list[torch.Tensor]:
        """Batched forward — matches ``_calc_cond_batch`` semantics.

        If *sub_network* is True, ``_s2_sub_network`` / ``_s2_seed`` are injected.
        If *stg_skip* is True, ``_stg_skip_attn`` / ``_stg_mode`` / ``_stg_blocks``
        are injected so the model skips self-attention in specified blocks.
        """
        input_x_list = []
        c_list = []
        group_list = []  # per-entry cond-group index: 0 = positive, 1 = negative
        for _ci, cond in enumerate(conds):
            for entry in cond:
                p = get_area_and_mult(entry, x, sigma)
                if p is None:
                    continue
                input_x_list.append(p.input_x)
                c_list.append(p.conditioning)
                group_list.append(_ci)

        if not c_list:
            return [torch.zeros_like(x) for _ in conds]

        input_x = torch.cat(input_x_list)
        c = cond_cat(c_list)
        timestep_ = torch.cat([sigma] * len(c_list))

        topts = model_options.get("transformer_options", {}).copy()
        if sub_network:
            topts["_s2_sub_network"] = True
            topts["_s2_seed"] = self._seed + self._step
        if stg_skip:
            topts["_stg_skip_attn"] = True
            topts["_stg_mode"] = self._stg_mode
            topts["_stg_blocks"] = self._stg_block_idx
        c["transformer_options"] = topts

        # Use model_function_wrapper if set (context windows / prompt-travel).
        # cond_or_uncond marks the cond group per batch row (0 = positive,
        # 1 = negative) so the wrapper can swap the positive prompt per window.
        if 'model_function_wrapper' in model_options:
            batch_chunks = len(c_list)
            cond_or_uncond = group_list
            output = model_options['model_function_wrapper'](
                self.inner_model.apply_model,
                {"input": input_x, "timestep": timestep_, "c": c,
                 "cond_or_uncond": cond_or_uncond},
            ).chunk(batch_chunks)
            return list(output)

        output = self.inner_model.apply_model(input_x, timestep_, **c)
        return list(output.chunk(len(c_list)))

    def _forward_single(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: list | None,
        model_options: dict,
        sub_network: bool = False,
        stg_skip: bool = False,
    ) -> torch.Tensor:
        """Single-condition forward (S² sub-network / STG skip).

        When the model's ``transformer_forward`` is compiled, dynamic
        control flow (S² block dropping / STG attention skipping) would
        cause recompilation or errors.  We temporarily restore the
        uncompiled original for this call, then swap back.
        """
        if cond is None:
            return torch.zeros_like(x)

        _dm = getattr(self.inner_model, 'diffusion_model', None)
        _orig = getattr(_dm, '_original_transformer_forward', None) if _dm else None

        if _orig is not None and (sub_network or stg_skip):
            # Swap compiled → original for this call, then restore.
            _compiled = _dm.transformer_forward
            _dm.transformer_forward = _orig
            try:
                return self._forward_batched(
                    x, sigma, [cond], model_options,
                    sub_network=sub_network, stg_skip=stg_skip,
                )[0]
            finally:
                _dm.transformer_forward = _compiled

        return self._forward_batched(x, sigma, [cond], model_options,
                                      sub_network=sub_network,
                                      stg_skip=stg_skip)[0]

    # -- combination formulas --------------------------------------------

    def _combine(
        self,
        uncond: torch.Tensor,
        cond: torch.Tensor,
        sub: torch.Tensor | None,
        cond_skip: torch.Tensor | None,
        cfg_step: float,
    ) -> torch.Tensor:
        """Route to the correct combination function (dispatch index)."""
        d = self._dispatch
        if d == 5:   # STG
            return _combine_stg(uncond, cond, cond_skip, cfg_step,
                                self._stg_scale)
        elif d == 4:  # S2
            return _combine_s2(uncond, cond, sub, cfg_step, self._s2_omega)
        elif d == 3:  # APG
            return _combine_apg(uncond, cond, cfg_step,
                                self._apg_eta, self._apg_rescale,
                                self._apg_momentum, self)
        elif d == 2:  # RAAG
            return _combine_raag(uncond, cond, cfg_step, self._raag_alpha)
        elif d == 1:  # Z2
            return _combine_z2(uncond, cond, cfg_step,
                               collapse=self._z2_collapse, wrapper=self)
        else:          # CFG
            return _combine_cfg(uncond, cond, cfg_step)

    def reset_step(self):
        """Reset the step counter (call before each new sampling run)."""
        self._step = 0
        self._prev_apg = None
        self._prev_z2 = None


# ---------------------------------------------------------------------------
# Dual-expert wrapper — switches inner_model at a split step
# ---------------------------------------------------------------------------

class DualExpertModelWrapper:
    """Combines a high-noise model wrapper with a lazy low-noise factory.

    The *low_factory* is invoked exactly once when the denoising step counter
    reaches *split_step*.  This lets the dual-expert sampler unload the high
    model and load the low model from disk at the split point, ensuring only
    one model is resident in RAM/VRAM at a time.

    Delegates ``model_sampling`` / ``process_latent_*`` to the high-noise
    model (both models must share the same latent space and sigma schedule).
    """

    def __init__(
        self,
        high_wrapper: BerniniModelWrapper,
        low_factory: callable,
        split_step: int,
    ):
        self._high = high_wrapper
        self._low: BerniniModelWrapper | None = None
        self._low_factory = low_factory
        self._split = split_step
        self._step = 0
        self._switched = False

    def _active_wrapper(self):
        return self._low if self._low is not None else self._high

    @property
    def inner_model(self):
        return self._active_wrapper().inner_model

    @property
    def model_sampling(self):
        return self._active_wrapper().model_sampling

    def process_latent_in(self, x):
        return self._active_wrapper().process_latent_in(x)

    def process_latent_out(self, x):
        return self._active_wrapper().process_latent_out(x)

    def release_high(self):
        """Drop the high-model wrapper to allow RAM release after a switch."""
        self._high = None

    @property
    def model_patcher(self):
        return None

    def __call__(self, x, sigma, **extra_args):
        if self._step >= self._split and not self._switched:
            logger.info(
                "[BerniniR] Dual-expert split at step %d: unloading high, loading low.",
                self._step,
            )
            # Cancel any in-flight high-model prefetches before VRAM pressure
            # spikes; moving weights behind a running transfer stream is what
            # causes the STATUS_ACCESS_VIOLATION on Windows.
            if self._high is not None:
                _dm = getattr(self._high.inner_model, 'diffusion_model', None)
                _mgr = getattr(_dm, '_block_swap_mgr', None)
                if _mgr is not None:
                    try:
                        _mgr.evict_all()
                    except Exception as e:
                        logger.warning(
                            "[BerniniR] High block-swap evict before switch failed: %s",
                            e,
                        )
            self._low = self._low_factory()
            self._switched = True
            self.release_high()  # drop high wrapper so high model can leave RAM
        w = self._low if self._low is not None else self._high
        self._step += 1
        return w(x, sigma, **extra_args)

    def reset_step(self):
        self._step = 0
        self._switched = False
        if self._high is not None:
            self._high.reset_step()
        if self._low is not None:
            self._low.reset_step()


# ---------------------------------------------------------------------------
# Combination functions (stateless, importable)
# ---------------------------------------------------------------------------

def _combine_cfg(uncond: torch.Tensor, cond: torch.Tensor, scale: float) -> torch.Tensor:
    """Standard classifier-free guidance."""
    return uncond + (cond - uncond) * scale


def _combine_z2(
    uncond: torch.Tensor,
    cond: torch.Tensor,
    scale: float,
    collapse: float = 0.0,
    wrapper: BerniniModelWrapper | None = None,
) -> torch.Tensor:
    """Zero-Cost Zigzag Trajectories (Hang et al.) + trajectory-collapse
    stabilization for video.

        x0 = uncond + s*(cond - uncond)      # standard CFG prediction
        z0 = uncond + s*(uncond - x0)        # zigzag back-step
        v  = uncond + s*(x0 - z0)            # re-CFG combination

    *collapse* in [0, 1): EMA of the output velocity across denoising
    steps, killing the per-step temporal jitter Z² introduces in video
    (the "trajectory collapse" stabilization from the video extension).
    0 = off (pure image Z²).  State lives on *wrapper._prev_z2*.
    """
    x0 = uncond + scale * (cond - uncond)
    z0 = uncond + scale * (uncond - x0)
    v = uncond + scale * (x0 - z0)

    if collapse > 0.0 and wrapper is not None:
        prev = getattr(wrapper, "_prev_z2", None)
        if prev is not None and prev.shape == v.shape:
            v = collapse * prev + (1.0 - collapse) * v
        wrapper._prev_z2 = v.detach().clone()
    elif wrapper is not None:
        wrapper._prev_z2 = None

    return v


def _combine_apg(
    uncond: torch.Tensor,
    cond: torch.Tensor,
    scale: float,
    eta: float = 0.15,
    rescale: bool = True,
    momentum: float = 0.0,
    wrapper: BerniniModelWrapper | None = None,
) -> torch.Tensor:
    """Adaptive Projected Guidance (Sadat et al., ICLR 2025).

    Decomposes the CFG update into parallel and orthogonal components
    relative to the conditional prediction.  The parallel component is
    suppressed (via *eta*) to reduce oversaturation at high scales.
    """
    v_cfg = uncond + scale * (cond - uncond)

    delta = v_cfg - cond
    norm_sq = (cond * cond).sum() + 1e-8
    parallel = (delta * cond).sum() / norm_sq * cond
    orthogonal = delta - parallel

    v_apg = cond + eta * parallel + orthogonal

    if rescale:
        v_norm = v_apg.norm()
        c_norm = cond.norm()
        # Adaptive radius: higher CFG → larger allowed norm.
        target = c_norm * (1.0 + eta * (scale - 1.0))
        if v_norm > 1e-8:
            v_apg = v_apg * (target / v_norm)

    if momentum > 0.0 and wrapper is not None and wrapper._prev_apg is not None:
        v_apg = cond + momentum * (v_apg - wrapper._prev_apg)

    if wrapper is not None:
        wrapper._prev_apg = v_apg.detach().clone()

    return v_apg


def _combine_raag(
    uncond: torch.Tensor,
    cond: torch.Tensor,
    cfg_max: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Ratio-Aware Adaptive Guidance (Zhu et al., arXiv:2508.03442).

    Computes ρ = ‖cond‖ / ‖uncond‖ and dampens guidance at early steps
    where ρ spikes exponentially:
        w(ρ) = 1 + (cfg_max − 1) · exp(−α · ρ)
    """
    rho = (cond.norm() / (uncond.norm() + 1e-8)).item()
    w = 1.0 + (cfg_max - 1.0) * math.exp(-alpha * rho)
    return uncond + (cond - uncond) * w


def _combine_stg(
    uncond: torch.Tensor,
    cond: torch.Tensor,
    cond_skip: torch.Tensor,
    cfg: float,
    stg_scale: float = 1.0,
) -> torch.Tensor:
    """Spatiotemporal Skip Guidance (Hyung et al., CVPR 2025).

        D̃ = D_uncond + λ(D_cond − D_uncond) + ω_stg(D_cond − D_skip)

    D_skip is the cond prediction with self-attention skipped in specific
    transformer blocks, creating an implicit weak model without extra params.
    """
    return uncond + cfg * (cond - uncond) + stg_scale * (cond - cond_skip)


def _combine_s2(
    uncond: torch.Tensor,
    cond: torch.Tensor,
    sub: torch.Tensor,
    cfg: float,
    omega: float = 1.0,
) -> torch.Tensor:
    """Stochastic Self-Guidance (Chen et al., arXiv:2508.12880).

        D̃ = D_uncond + λ(D_cond − D_uncond) − ω(D_sub − D_cond)

    D_sub is the prediction from a randomly block‑dropped sub‑network.
    """
    return uncond + cfg * (cond - uncond) - omega * (sub - cond)


# ---------------------------------------------------------------------------
# Segment prompt-travel plumbing
#
# ``segment_specs`` / ``segment_overlap_latent`` are carried on the positive
# conditioning dict produced by ``BerniniR_SegmentSchedule``.  They are *data*
# only — the actual windowing + crossfade is performed by the existing
# context-window framework (``_build_context_window_wrapper`` in sampler.py),
# which is reused verbatim.  We strip the keys from ``positive`` before
# ``process_conds`` so the standard single-prompt pipeline never sees them.
# ---------------------------------------------------------------------------

def _extract_segment_specs(positive) -> tuple[list[EncodedSegment] | None, int]:
    """Read segment prompt-travel data from *positive* conditioning.

    Returns ``(segment_specs, segment_overlap_latent)``.  ``segment_specs`` is
    ``None`` when the schedule node did not attach any segment data, so callers
    can branch straight to the ordinary context-window path.
    """
    cond = Conditioning.from_comfy(positive)
    if not cond.items:
        return None, 1
    extra = cond.items[0].extra
    specs = extra.get("segment_specs")
    if specs is None:
        return None, int(extra.get("segment_overlap_latent", 1))
    return list(specs), int(extra.get("segment_overlap_latent", 1))


def _strip_segment_keys(positive):
    """Return *positive* with the segment bookkeeping keys removed from the
    conditioning extra dict, so ``process_conds`` only sees a clean
    single-prompt positive embedding.

    Uses the typed ``Conditioning`` wrapper to avoid ``positive[0][1]`` magic
    indexing, and to skip a full ``deepcopy`` of the prompt tensor (the extra
    dict is shallow-copied by ``from_comfy``; only that dict is mutated).
    """
    cond = Conditioning.from_comfy(positive)
    if not cond.items:
        return positive
    extra = cond.items[0].extra
    if "segment_specs" not in extra and "segment_overlap_latent" not in extra:
        return positive
    extra.pop("segment_specs", None)
    extra.pop("segment_overlap_latent", None)
    return cond.to_comfy()


def _build_segment_or_context_wrapper(
    context_options,
    segment_specs: list[EncodedSegment] | None,
    segment_overlap: int = 1,
):
    """Pick the right ``model_function_wrapper``.

    * Segment mode (``segment_specs`` is not ``None``): build a
      ``SegmentWindow`` per segment and delegate to the context-window wrapper
      in segment mode (static windows from segments + linear crossfade, prompt
      swapped per window).  ``context_options`` is unused here.
    * Context mode (``context_options`` provided): delegate to the context-window
      wrapper in ordinary context mode.
    * Neither: return ``None`` (no windowing).
    """
    if segment_specs is not None:
        seg_windows = [
            SegmentWindow(
                start_latent=int(s.start_latent),
                end_latent=int(s.end_latent),
                embed=s.embed,
            )
            for s in segment_specs
        ]
        return _build_context_window_wrapper(
            context_options=None,
            segment_windows=seg_windows,
            segment_overlap=int(segment_overlap),
        )
    if context_options is not None:
        return _build_context_window_wrapper(context_options)
    return None


# ---------------------------------------------------------------------------
# Public entry point — replaces comfy.sample.sample()
# ---------------------------------------------------------------------------

def bernini_sample(
    model_handle: BerniniRModelHandle,
    noise: torch.Tensor,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent_image: dict,
    denoise: float = 1.0,
    seed: int = 0,
    guidance_config: BerniniGuidanceConfig | None = None,
    guidance_schedule: list[float] | None = None,
    block_swap_args: dict | None = None,
    flow_shift: float = 3.0,
    context_options: dict | None = None,
    callback=None,
    disable_pbar: bool = False,
    pre_unload_callback: callable | None = None,
) -> dict:
    """Run Bernini-R denoising with self-contained forward orchestration.

    Detects ``segment_specs`` in positive conditioning and, when present, runs
    segment prompt-travel through the standard single-pass pipeline: the whole
    video is denoised in one pass and the positive text embedding is swapped
    per temporal window via the context-window framework (crossfading adjacent
    segments).  Otherwise runs standard single-pass sampling.
    """
    # ── Auto-detect segment prompt-travel schedule ────────────────────
    # Segments reuse the standard pipeline; the per-window embedding swap is
    # injected as the context-window wrapper below.  Strip segment-only keys
    # so process_conds sees a clean single-prompt positive (segment[0]).
    _segment_specs, _segment_overlap = _extract_segment_specs(positive)
    if _segment_specs is not None:
        positive = _strip_segment_keys(positive)

    # ── Standard single-pass sampling ─────────────────────────────────
    # ── Load model (this is the only time it enters RAM/VRAM) ─────────
    # Block swap ON -> the model is the single copy on CPU and BlockSwapManager
    # windows a slice onto the GPU; we must NOT let ComfyUI hoist the whole
    # model onto the GPU first (that would be a second full copy).
    model = model_handle.load(
        block_swap=(block_swap_args.block_to_swap > 0) if block_swap_args else False)

    # ── Prepare latent ────────────────────────────────────────────────
    latent = dict(latent_image)
    latent_samples = latent["samples"]
    latent_samples = comfy.sample.fix_empty_latent_channels(
        model, latent_samples,
        latent.get("downscale_ratio_spatial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent["samples"] = latent_samples

    # ── Noise mask (unused by BerniniR — kept for compatibility) ────────
    noise_mask = latent.get("noise_mask", None)

    # ── flow_shift ────────────────────────────────────────────────────
    use_seed = seed
    _prev_shift = None
    if hasattr(model.model, 'model_sampling'):
        ms = model.model.model_sampling
        _prev_shift = getattr(ms, 'shift', None)
        ms.set_parameters(shift=flow_shift)

    # ── Build schedule ────────────────────────────────────────────────
    if guidance_schedule is not None:
        schedule = _resample_schedule(list(guidance_schedule), steps)
    else:
        schedule = [cfg] * steps

    # ── Encode conditioning ───────────────────────────────────────────
    positive_copy = comfy.sampler_helpers.convert_cond(positive)
    negative_copy = comfy.sampler_helpers.convert_cond(negative)
    conds = {"positive": positive_copy, "negative": negative_copy}

    inner_model, conds, loaded_models = comfy.sampler_helpers.prepare_sampling(
        model, noise.shape, conds, model.model_options,
    )
    device = model.load_device

    inj: InjectionContext | None = None
    _prev_cw = None
    try:
        # ── process_conds ──────────────────────────────────────────
        latent_for_conds = latent_samples.to(device=device, dtype=torch.float32)
        latent_for_conds = inner_model.process_latent_in(latent_for_conds)
        conds = comfy.samplers.process_conds(
            inner_model, noise, conds, device,
            latent_image=latent_for_conds,
            seed=use_seed,
        )
        comfy.sampler_helpers.prepare_model_patcher(
            model, conds, model.model_options,
        )

        # ── Context window / segment prompt-travel wrapper ──────────
        _cw_wrapper = _build_segment_or_context_wrapper(
            context_options, _segment_specs, _segment_overlap)

        # ── Calculate sigmas ────────────────────────────────────────
        k_sampler = comfy.samplers.KSampler(
            model, steps=steps, device=device,
            sampler=sampler_name, scheduler=scheduler,
            denoise=denoise, model_options=model.model_options,
        )
        sigmas = k_sampler.sigmas.to(device)

        # ── Build unified injection context ─────────────────────────
        _dm = getattr(inner_model, 'diffusion_model', inner_model)
        total_blocks = len(_dm.blocks) if hasattr(_dm, 'blocks') else 0
        inj = InjectionContext.build(
            positive=positive,
            context_options=context_options,
            block_swap_args=block_swap_args,
            guidance_config=guidance_config,
            total_blocks=total_blocks,
            context_window_wrapper=_cw_wrapper,
        )

        # ── Build extra_model_options (single injection call) ───────
        extra_model_options = comfy.model_patcher.create_model_options_clone(
            model.model_options)
        inj.apply_options(extra_model_options, sigmas, noise.shape[0])

        model.pre_run()

        # ── Block swap (needs model access for compile check) ───────
        inj.apply_block_swap(extra_model_options, inner_model)

        # ── FreeNoise ───────────────────────────────────────────────
        noise = inj.apply_noise(noise, context_options, seed=use_seed)

        # ── Ensure noise is on device ───────────────────────────────
        noise = noise.to(device=device, dtype=torch.float32)

        # ── Create model wrapper ────────────────────────────────────
        # Resolve guidance params from config (or defaults).
        gc = guidance_config or BerniniGuidanceConfig()
        wrapper = BerniniModelWrapper(
            inner_model=inner_model,
            cond_pos=conds.get("positive"),
            cond_neg=conds.get("negative"),
            cfg=cfg,
            mode=gc.mode.value,
            schedule=schedule,
            s2_omega=gc.s2_omega,
            stg_scale=gc.stg_scale,
            stg_mode=inj.stg_mode,
            stg_block_idx=inj.stg_blocks,
            raag_alpha=gc.raag_alpha,
            apg_eta=gc.apg_eta,
            apg_rescale=gc.apg_rescale,
            apg_momentum=gc.apg_momentum,
            z2_collapse=gc.z2_collapse,
            seed=use_seed,
            model_options=extra_model_options,
            noise=noise,
            total_steps=steps,
            injection=inj,
        )

        # ── Run k-diffusion ─────────────────────────────────────────
        sigmas = sigmas.to(device=device)

        samp_obj = comfy.samplers.sampler_object(sampler_name)
        samples = samp_obj.sample(
            wrapper, sigmas,
            extra_args={"model_options": extra_model_options, "seed": use_seed},
            callback=callback,
            noise=noise,
            latent_image=latent_for_conds,
            denoise_mask=noise_mask,
            disable_pbar=disable_pbar,
        )
        samples = inner_model.process_latent_out(samples.to(torch.float32))

    finally:
        # ── Cleanup ─────────────────────────────────────────────────
        if _prev_shift is not None and hasattr(model.model, 'model_sampling'):
            model.model.model_sampling.set_parameters(shift=_prev_shift)
        try:
            comfy.sampler_helpers.cleanup_models(conds, loaded_models)
        except Exception:
            pass
        # Give callers a chance to clean up before the model is dropped.
        if pre_unload_callback is not None:
            try:
                pre_unload_callback()
            except Exception:
                pass
        # Fully unload the model from RAM/VRAM now that sampling is done.
        try:
            model_handle.unload()
        except Exception:
            pass

    # ── Build output ──────────────────────────────────────────────────
    out = latent.copy()
    out.pop("downscale_ratio_spatial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples

    return out


# ---------------------------------------------------------------------------
# Dual-expert entry point
# ---------------------------------------------------------------------------
def bernini_sample_dual(
    high_model: BerniniRModelHandle,
    low_model: BerniniRModelHandle,
    noise: torch.Tensor,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent_image: dict,
    split_step: int = 10,
    denoise: float = 1.0,
    seed: int = 0,
    guidance_config: BerniniGuidanceConfig | None = None,
    guidance_schedule: list[float] | None = None,
    flow_shift: float = 3.0,
    context_options: dict | None = None,
    block_swap_args: dict | None = None,
    callback=None,
    disable_pbar: bool = False,
) -> dict:
    """Dual-expert sampling: high-noise model early, low-noise model late.

    Auto-detects ``segment_specs`` and runs segment prompt-travel through the
    standard dual-expert pipeline (per-window text-embedding swap via the
    context-window wrapper), instead of a separate hard-cut path.
    """
    # ── Auto-detect segment prompt-travel schedule ────────────────────
    _segment_specs, _segment_overlap = _extract_segment_specs(positive)
    if _segment_specs is not None:
        positive = _strip_segment_keys(positive)

    use_seed = seed

    # ── Load high-noise model ─────────────────────────────────────────
    # Block swap ON -> single copy on CPU, BlockSwapManager owns the GPU.
    high_patcher = high_model.load(
        block_swap=(block_swap_args.block_to_swap > 0) if block_swap_args else False)

    # ── Prepare latent ────────────────────────────────────────────────
    latent = dict(latent_image)
    latent_samples = latent["samples"]
    latent_samples = comfy.sample.fix_empty_latent_channels(
        high_patcher, latent_samples,
        latent.get("downscale_ratio_spatial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent["samples"] = latent_samples
    noise_mask = latent.get("noise_mask", None)

    # ── flow_shift ────────────────────────────────────────────────────
    _prev_shift_high = None
    if hasattr(high_patcher.model, 'model_sampling'):
        ms = high_patcher.model.model_sampling
        _prev_shift_high = getattr(ms, 'shift', None)
        ms.set_parameters(shift=flow_shift)

    # ── Build schedule ────────────────────────────────────────────────
    if guidance_schedule is not None:
        schedule = _resample_schedule(list(guidance_schedule), steps)
    else:
        schedule = [cfg] * steps
    high_schedule = schedule[:split_step]
    low_schedule = schedule[split_step:]

    # ── Setup high-noise model ────────────────────────────────────────
    def _setup_model(mp, lat_for_conds):
        positive_copy = comfy.sampler_helpers.convert_cond(positive)
        negative_copy = comfy.sampler_helpers.convert_cond(negative)
        conds = {"positive": positive_copy, "negative": negative_copy}
        inner, conds, loaded = comfy.sampler_helpers.prepare_sampling(
            mp, noise.shape, conds, mp.model_options)
        conds = comfy.samplers.process_conds(
            inner, noise, conds, mp.load_device,
            latent_image=lat_for_conds, seed=use_seed)
        comfy.sampler_helpers.prepare_model_patcher(
            mp, conds, mp.model_options)
        return inner, conds, loaded

    device = high_patcher.load_device
    noise_dev = noise.to(device=device, dtype=torch.float32)
    lat_dev = latent_samples.to(device=device, dtype=torch.float32)
    lat_for_conds = high_patcher.model.process_latent_in(lat_dev)

    high_inner, high_conds, high_loaded = _setup_model(high_patcher, lat_for_conds)

    # ── Calculate sigmas ──────────────────────────────────────────────
    k_sampler = comfy.samplers.KSampler(
        high_patcher, steps=steps, device=device,
        sampler=sampler_name, scheduler=scheduler,
        denoise=denoise, model_options=high_patcher.model_options,
    )
    sigmas = k_sampler.sigmas.to(device)

    # ── Context window / segment prompt-travel wrapper ────────────────
    _cw_wrapper = _build_segment_or_context_wrapper(
        context_options, _segment_specs, _segment_overlap)

    # ── Build unified injection context ───────────────────────────────
    _dm = getattr(high_inner, 'diffusion_model', high_inner)
    total_blocks = len(_dm.blocks) if hasattr(_dm, 'blocks') else 0
    inj = InjectionContext.build(
        positive=positive,
        context_options=context_options,
        block_swap_args=block_swap_args,
        guidance_config=guidance_config,
        total_blocks=total_blocks,
        context_window_wrapper=_cw_wrapper,
    )

    # ── FreeNoise ───────────────────────────────────────────────
    # The dual-expert path previously never applied FreeNoise at all.  Apply
    # it here exactly once, mirroring the single-sampler path (single path
    # applies it inside the same InjectionContext.apply_noise call site).
    noise_dev = inj.apply_noise(noise_dev, context_options, seed=use_seed)

    # ── Build extra_model_options (single injection call) ─────────────
    extra_model_options = comfy.model_patcher.create_model_options_clone(
        high_patcher.model_options)
    inj.apply_options(extra_model_options, sigmas, noise.shape[0])

    high_patcher.pre_run()

    # ── Block swap for high model ─────────────────────────────────────
    inj.apply_block_swap(extra_model_options, high_inner)

    # ── High-noise wrapper ────────────────────────────────────────────
    gc = guidance_config or BerniniGuidanceConfig()
    high_wrapper = BerniniModelWrapper(
        inner_model=high_inner,
        cond_pos=high_conds.get("positive"),
        cond_neg=high_conds.get("negative"),
        cfg=cfg,
        mode=gc.mode.value,
        schedule=high_schedule,
        s2_omega=gc.s2_omega,
        stg_scale=gc.stg_scale,
        stg_mode=inj.stg_mode,
        stg_block_idx=inj.stg_blocks,
        raag_alpha=gc.raag_alpha,
        apg_eta=gc.apg_eta,
        apg_rescale=gc.apg_rescale,
        apg_momentum=gc.apg_momentum,
        z2_collapse=gc.z2_collapse,
        seed=use_seed,
        model_options=extra_model_options,
        noise=noise_dev,
        total_steps=steps,
        injection=inj,
    )

    # ── Lazy low-noise wrapper factory ────────────────────────────────
    low_inner = None
    low_conds = None
    low_loaded = None
    _prev_shift_low = None

    def _make_low_wrapper():
        nonlocal low_inner, low_conds, low_loaded, _prev_shift_low
        nonlocal high_patcher, high_inner, high_wrapper, high_model

        logger.info(
            "[BerniniR] Dual-expert split: fully releasing high-noise model before loading low.",
        )

        # 1. Restore the high model's original shift before we drop it, so the
        #    next workflow that reloads it from cache sees the unmodified value.
        if high_patcher is not None and _prev_shift_high is not None:
            try:
                if hasattr(high_patcher.model, 'model_sampling'):
                    high_patcher.model.model_sampling.set_parameters(shift=_prev_shift_high)
            except Exception:
                pass

        # 2. Stop high async transfers and move all high blocks to CPU,
        #    deterministically, before we drop the model.  shutdown() cancels
        #    prefetches, synchronises every CUDA stream, and evicts to CPU so
        #    freeing the weights cannot race an in-flight H2D copy.
        if high_patcher is not None:
            _dm = getattr(high_patcher.model, 'diffusion_model', None)
            _mgr = getattr(_dm, '_block_swap_mgr', None)
            if _mgr is not None:
                try:
                    _mgr.shutdown()
                except Exception as e:
                    logger.warning("[BerniniR] High block-swap shutdown failed: %s", e)

        # 3. Fully release the high model and drop every local reference so
        #    Python can free the weights *before* the low load peak.  unload()
        #    now always does a full release and notifies ComfyUI's model
        #    manager so VRAM is actually freed for the low model.
        log_system_ram("before high release")
        try:
            high_model.unload()
        except Exception as e:
            logger.warning("[BerniniR] High model unload before low load failed: %s", e)

        if high_patcher is not None:
            _cache_evict_patcher(high_patcher)

        # Immediately shrink the high model's weight tensors to zero so the
        # ~14 GB of host RAM is returned *synchronously* (gc.collect alone can
        # leave it resident on Windows).  This is what keeps the dual-expert
        # switch at a single-model RAM footprint instead of 2x.
        release_model_ram(high_patcher)

        high_wrapper = None
        high_inner = None
        high_patcher = None
        high_model = None
        try:
            dual.release_high()
        except Exception:
            pass
        # Aggressive cleanup: the HIGH model may have left PyTorch's CPU
        # allocator holding many gigabytes of RAM that it won't return to the
        # OS on Windows.  Trim the working set before the LOW load peak.
        collect_garbage(aggressive=True)
        log_system_ram("after high release, before low load")

        # 4. Now that high is fully gone, load the low-noise expert.
        # Block swap ON -> single copy on CPU; with high already deterministically
        # freed there is no whole-model GPU allocation to collide with.
        low_patcher = low_model.load(
            block_swap=(inj.block_to_swap > 0) if inj.block_to_swap > 0 else False)

        if hasattr(low_patcher.model, 'model_sampling'):
            ms = low_patcher.model.model_sampling
            _prev_shift_low = getattr(ms, 'shift', None)
            ms.set_parameters(shift=flow_shift)

        low_lat_for_conds = low_patcher.model.process_latent_in(lat_dev)
        low_inner, low_conds, low_loaded = _setup_model(low_patcher, low_lat_for_conds)

        low_patcher.pre_run()

        # Re-apply block swap for the low model (block count may differ).
        inj.apply_block_swap(extra_model_options, low_inner)

        return BerniniModelWrapper(
            inner_model=low_inner,
            cond_pos=low_conds.get("positive"),
            cond_neg=low_conds.get("negative"),
            cfg=cfg,
            mode=gc.mode.value,
            schedule=low_schedule,
            s2_omega=gc.s2_omega,
            stg_scale=gc.stg_scale,
            stg_mode=inj.stg_mode,
            stg_block_idx=inj.stg_blocks,
            raag_alpha=gc.raag_alpha,
            apg_eta=gc.apg_eta,
            apg_rescale=gc.apg_rescale,
            apg_momentum=gc.apg_momentum,
            z2_collapse=gc.z2_collapse,
            seed=use_seed,
            model_options=extra_model_options,
            noise=noise_dev,
            total_steps=steps,
            injection=inj,
        )

    dual = DualExpertModelWrapper(high_wrapper, _make_low_wrapper, split_step)

    # ── Run k-diffusion ───────────────────────────────────────────────
    sigmas_dev = sigmas.to(device=device)
    samp_obj = comfy.samplers.sampler_object(sampler_name)
    samples = samp_obj.sample(
        dual, sigmas_dev,
        extra_args={"model_options": extra_model_options, "seed": use_seed},
        callback=callback,
        noise=noise_dev,
        latent_image=lat_for_conds,
        denoise_mask=noise_mask,
        disable_pbar=disable_pbar,
    )
    final_inner = dual._low.inner_model if dual._low is not None else high_inner
    samples = final_inner.process_latent_out(samples.to(torch.float32))

    # ── Cleanup ───────────────────────────────────────────────────────
    if (high_patcher is not None and _prev_shift_high is not None
            and hasattr(high_patcher.model, 'model_sampling')):
        high_patcher.model.model_sampling.set_parameters(shift=_prev_shift_high)
    if low_inner is not None and _prev_shift_low is not None and hasattr(low_inner, 'model_sampling'):
        low_inner.model_sampling.set_parameters(shift=_prev_shift_low)
    try:
        comfy.sampler_helpers.cleanup_models(
            {"positive": high_conds["positive"], "negative": high_conds["negative"]},
            high_loaded)
    except Exception:
        pass
    if low_conds is not None:
        try:
            comfy.sampler_helpers.cleanup_models(
                {"positive": low_conds["positive"], "negative": low_conds["negative"]},
                low_loaded)
        except Exception:
            pass

    # Unload whichever model is currently resident.
    for handle in (low_model, high_model):
        try:
            handle.unload()
        except Exception:
            pass

    out = latent.copy()
    out.pop("downscale_ratio_spatial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out

