"""
Enhanced KSampler for Bernini-R with context window support.

Provides BerniniR_KSampler: a drop-in replacement for ComfyUI's KSampler that
natively supports temporal context window tiling for long video generation.

When context_options is connected:
  - Latent is split into overlapping windows along the temporal dimension.
  - Each window is denoised independently and blended with the configured
    fusion method (linear or pyramid).
  - FreeNoise shuffling reduces tiling artifacts.
  - Step-aware window scheduling (uniform_standard / uniform_looped) varies
    window positions across denoising steps.

When context_options is NOT connected:
  - Behaves identically to ComfyUI's built-in KSampler.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.utils

try:
    import latent_preview
except Exception:
    latent_preview = None

from ..context.windows import get_context_scheduler, create_window_mask
from ..utils.types import (
    BerniniContext,
    BerniniBlockSwap,
    BerniniTeaCache,
    BerniniGuidance,
    BerniniGuidanceConfig,
    SegmentWindow,
)
from ..utils.teacache import (
    TeaCache,
    DEFAULT_START_BLOCK,
    DEFAULT_MAX_SKIP_BLOCKS,
    DEFAULT_REL_L1_THRESH,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_COOLDOWN_STEPS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context window helpers
# ---------------------------------------------------------------------------

def _setup_context_windows(
    latent_frames: int,
    context_options: BerniniContext,
    step: int = 0,
) -> tuple[list[list[int]], int, int]:
    """Convert pixel-space context options to latent-space windows.

    Returns:
        (windows_list, context_latent_frames, overlap_latent)
    """
    ctx_latent = context_options.latent_frames
    overlap_latent = context_options.latent_overlap
    schedule = context_options.schedule.value

    # uniform_* schedulers treat context_stride as the number of stride
    # levels (log2 scale).  WanVideoWrapper divides pixel value by 4.
    ctx_stride_levels = max(1, context_options.latent_stride)

    scheduler = get_context_scheduler(schedule)

    if schedule == "static_standard":
        windows = scheduler(
            step=0, num_steps=1,
            num_frames=latent_frames,
            context_size=ctx_latent,
            context_stride=ctx_stride_levels,
            context_overlap=overlap_latent,
        )
    else:
        windows = scheduler(
            step=step, num_steps=1,
            num_frames=latent_frames,
            context_size=ctx_latent,
            context_stride=ctx_stride_levels,
            context_overlap=overlap_latent,
        )

    # uniform_looped returns a generator; convert to list for len().
    if not isinstance(windows, list):
        windows = list(windows)

    return windows, ctx_latent, overlap_latent


def _apply_freenoise(
    noise: torch.Tensor,
    context_options: BerniniContext,
    seed: int = 0,
) -> torch.Tensor:
    """Shuffle noise patterns between context windows to reduce tiling artifacts.

    Ported from AnimateDiff-Evolved via WanVideoWrapper.
    """
    ctx_latent = context_options.latent_frames
    overlap_latent = context_options.latent_overlap
    delta = context_options.context_frames - context_options.context_overlap
    if delta <= 0:
        return noise

    delta_latent = ctx_latent - overlap_latent
    if delta_latent <= 0:
        return noise

    B, C, T, H, W = noise.shape
    if T <= ctx_latent:
        return noise

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    for start_idx in range(0, T - ctx_latent, delta_latent):
        place_idx = start_idx + ctx_latent
        if place_idx >= T:
            break
        actual_delta = min(delta_latent, T - place_idx)
        if actual_delta <= 0:
            break

        idx = torch.randperm(actual_delta, generator=gen)
        src = noise[:, :, start_idx: start_idx + actual_delta, :, :]
        noise[:, :, place_idx: place_idx + actual_delta, :, :] = src[:, :, idx, :, :]

    return noise


def _normalize_timestep(ts) -> float:
    """Extract a canonical scalar from a timestep value for comparison.

    ComfyUI may pass timesteps as a 0-d tensor, 1-d tensor, list, or float.
    NOTE: ``.item()`` triggers a CPU-GPU sync.  This is called only on
    timestep transitions (~once per denoising step), not per-window, so the
    overhead is negligible (<0.1ms per step).
    """
    if isinstance(ts, torch.Tensor):
        return float(ts.reshape(-1)[0].item())  # handles 0-d and 1-d tensors
    if isinstance(ts, (list, tuple)):
        return float(ts[0])
    return float(ts)


def _build_segment_windows(
    latent_frames: int,
    segment_windows: list[SegmentWindow],
    overlap: int,
) -> list[tuple[list[int], torch.Tensor]]:
    """Expand user segments into overlapping latent windows for crossfade.

    Each segment ``[start_latent, end_latent)`` is extended by *overlap* latent
    frames into its neighbours so adjacent windows share a transition region
    that :func:`create_window_mask` crossfades.  The first window is not
    extended on the left, and the last not on the right (video boundaries).

    Returns a list of ``(window_indices, embed)`` tuples in segment order.
    """
    ov = max(0, int(overlap))
    n = len(segment_windows)
    out: list[tuple[list[int], torch.Tensor]] = []
    for i, sw in enumerate(segment_windows):
        a = int(sw.start_latent) - (ov if i > 0 else 0)
        b = int(sw.end_latent) + (ov if i < n - 1 else 0)
        a = max(0, a)
        b = min(latent_frames, b)
        if b <= a:
            continue
        out.append((list(range(a, b)), sw.embed))
    return out


def _swap_positive_embed(c: dict, embed: torch.Tensor, cond_or_uncond) -> dict:
    """Return a shallow copy of *c* whose positive ``c_crossattn`` rows are
    replaced by *embed* (the current window's segment prompt).

    ``cond_or_uncond`` marks each batch row: ``0`` = positive (cond), non-zero
    = negative (uncond).  Only positive rows are swapped so CFG still uses the
    shared negative prompt.  All Wan text embeds are ``(1, 512, 4096)`` so the
    in-place row assignment is always shape-safe.
    """
    cc = c.get("c_crossattn", None)
    if cc is None or embed is None:
        return c
    new_cc = cc.clone()
    emb = embed.to(device=cc.device, dtype=cc.dtype)
    for row, tag in enumerate(cond_or_uncond):
        if tag == 0 and row < new_cc.shape[0]:
            new_cc[row] = emb[0]
    c = dict(c)
    c["c_crossattn"] = new_cc
    return c


def _build_context_window_wrapper(
    context_options: BerniniContext | None,
    segment_windows: list[SegmentWindow] | None = None,
    segment_overlap: int = 1,
):
    """Return a model_function_wrapper that implements temporal window tiling.

    The wrapper splits the input latent along the temporal dimension, runs the
    model on each window independently, and blends predictions with configurable
    fusion masks.

    For static_standard schedules, windows and masks are cached after first
    computation since they never change across steps.

    **Segment prompt-travel mode**: when *segment_windows* is provided, windows
    come from the user's segments (extended by *segment_overlap* for crossfade)
    instead of ``context_options``, the schedule is forced to static+linear, and
    each window's positive text embedding is swapped to its segment prompt.  The
    overlap region linearly crossfades adjacent prompts.  In this mode
    *context_options* may be ``None``.

    **Lifecycle**: a new closure is created on every ``_build_context_window_wrapper``
    call (once per ``bernini_sample``).  All mutable state (step counter, buffers,
    caches) lives in the closure, never on a module-level global.  Callers MUST
    call this factory for each sampling run to avoid cross-run leakage.
    """
    segment_mode = segment_windows is not None
    if segment_mode:
        # Prompt-travel: static windows from segments + linear crossfade.
        schedule = "static_standard"
        fuse_method = "linear"
        is_static = True
    else:
        schedule = context_options.schedule.value
        fuse_method = context_options.fuse_method.value
        is_static = (schedule == "static_standard")

    # Step counter — resets to 0 on each new wrapper creation (i.e., each
    # new sampling run).  Uses a list for mutability within the closure.
    # In non-batched CFG mode, ComfyUI calls the wrapper TWICE per denoising
    # step (once for cond, once for uncond).  We track the timestep to only
    # increment on actual step transitions.  Starts at 0 (first increment
    # occurs on first new timestep) so ordered_halving(0) = 0.0 matches
    # WanVideoWrapper / AnimateDiff-Evolved reference.
    step_counter: list[int] = [0]
    _last_timestep = [None]  # tracks timestep of the last increment
    cache: dict = {}  # keyed by (T, device_str) → (windows_list, masks_list)

    buf_out: torch.Tensor | None = None
    buf_counter: torch.Tensor | None = None

    def wrapper(apply_model_func, kwargs):
        nonlocal step_counter, _last_timestep, cache, buf_out, buf_counter
        x = kwargs.get("input")
        timestep = kwargs.get("timestep")
        c = kwargs.get("c")
        cond_or_uncond = kwargs.get("cond_or_uncond")

        # Guard: context window requires a 5D video tensor (B, C, T, H, W).
        if not isinstance(x, torch.Tensor) or x.dim() != 5:
            return apply_model_func(x, timestep, **c, cond_or_uncond=cond_or_uncond)

        B, C, T, H, W = x.shape

        # Increment step only on new timestep values.  In non-batched CFG,
        # cond + uncond share the same timestep within one denoising step;
        # this ensures uniform_* schedulers receive consecutive step numbers
        # starting from 0 (matching WanVideoWrapper reference).
        ts_val = _normalize_timestep(timestep)
        if _last_timestep[0] is not None and ts_val != _last_timestep[0]:
            step_counter[0] += 1
        _last_timestep[0] = ts_val
        step = step_counter[0]

        # ── Resolve windows + masks ──────────────────────────────────
        cache_key = (T, str(x.device))
        if is_static and cache_key in cache:
            windows_list, masks_list, embeds_list = cache[cache_key]
        else:
            if segment_mode:
                seg_wins = _build_segment_windows(
                    T, segment_windows, segment_overlap)
                windows_raw = [w for (w, _e) in seg_wins]
                raw_embeds = [e for (_w, e) in seg_wins]
                overlap_latent = segment_overlap
            else:
                windows_raw, _, overlap_latent = _setup_context_windows(
                    T, context_options, step=step,
                )
                raw_embeds = [None] * len(windows_raw)
            if len(windows_raw) <= 1:
                return apply_model_func(x, timestep, **c, cond_or_uncond=cond_or_uncond)

            windows_list = []
            masks_list = []
            embeds_list = []
            _order = sorted(range(len(windows_raw)), key=lambda k: windows_raw[k][0])
            windows_sorted = [windows_raw[k] for k in _order]
            embeds_sorted = [raw_embeds[k] for k in _order]
            for i, w in enumerate(windows_sorted):
                w_idx = torch.tensor(w, dtype=torch.long, device=x.device)
                w_cpu = torch.tensor(w, dtype=torch.long)  # for CPU latents
                windows_list.append((w, w_idx, w_cpu))
                tw = len(w)
                if is_static:
                    # Measure actual overlap with neighbours for crossfade
                    left_ol = 0
                    right_ol = 0
                    if i > 0:
                        left_ol = max(0, windows_sorted[i-1][-1] - w[0] + 1)
                    if i < len(windows_sorted) - 1:
                        right_ol = max(0, w[-1] - windows_sorted[i+1][0] + 1)
                    left_ol = min(left_ol, tw)
                    right_ol = min(right_ol, tw)
                else:
                    # uniform_*: multi-window interleaving — use uniform
                    # weights normalised by frame-wise coverage later.
                    left_ol = 0
                    right_ol = 0
                mask_temporal = create_window_mask(
                    x[:, 0, :tw],
                    w, T,
                    context_overlap=overlap_latent,
                    left_overlap=left_ol,
                    right_overlap=right_ol,
                    window_type=fuse_method,
                )
                if mask_temporal.dim() == 4:
                    mask = mask_temporal.unsqueeze(1)
                else:
                    mask = mask_temporal
                masks_list.append(mask)
                embeds_list.append(embeds_sorted[i])

            if is_static:
                cache[cache_key] = (windows_list, masks_list, embeds_list)

        # ── Accumulate predictions ───────────────────────────────────
        if buf_out is None or buf_out.shape != x.shape:
            buf_out = torch.zeros_like(x)
            buf_counter = torch.zeros(B, 1, T, 1, 1, device=x.device, dtype=torch.float32)
        else:
            buf_out.zero_()
            buf_counter.zero_()

        # For uniform_* schedulers, 3+ windows may overlap at the same
        # frame (different strides).  The per-edge mask can't handle
        # >2-way overlaps.  Normalise by simple frame-wise coverage.
        if not is_static:
            coverage = torch.zeros(B, 1, T, 1, 1, device=x.device, dtype=torch.float32)
            for w_idx in windows_list:
                ones = torch.ones(B, 1, len(w_idx), 1, 1, device=x.device, dtype=torch.float32)
                coverage.index_add_(2, w_idx, ones)
            coverage.clamp_(min=1.0)

        # ── Expose window identity via transformer_options so hooks
        #     (TeaCache etc.) can maintain per-window state without a
        #     module-level global variable.
        mo = kwargs.get("model_options", {})
        to = mo.setdefault("transformer_options", {})

        for idx, ((w, w_idx, w_cpu), mask) in enumerate(zip(windows_list, masks_list)):
            x_win = x.index_select(2, w_idx)
            to["_context_window"] = tuple(w)

            # ── Per-window conditioning slice ──────────────────────
            # Time-varying conditioning (source_video, reference_video
            # encoded as context_latents) must be sliced to the current
            # window so the model only attends to temporally relevant
            # reference frames.  Single-frame latents (reference_images)
            # are left untouched so they remain global references.
            c_win = dict(c)  # shallow copy — other windows need original
            win_len = len(w)

            context_latents = c_win.get('context_latents', None)
            if context_latents is not None:
                sliced_latents = []
                for lat in context_latents:
                    if (isinstance(lat, torch.Tensor) and lat.dim() >= 5
                            and lat.shape[2] > 1
                            and w[-1] < lat.shape[2]):
                        # Use CPU copy of w_idx for CPU latents,
                        # GPU copy for GPU latents (same-device avoids D2H).
                        idx_t = w_idx if lat.device == x.device else w_cpu
                        sliced_latents.append(
                            lat.index_select(2, idx_t.to(lat.device)))
                    else:
                        sliced_latents.append(lat)
                c_win['context_latents'] = sliced_latents

            # c_concat (image conditioning) — slice iff its temporal
            # dimension covers all indices in the window.
            c_concat = c_win.get('c_concat', None)
            if (isinstance(c_concat, torch.Tensor) and c_concat.dim() >= 5
                    and c_concat.shape[2] > 1
                    and w[-1] < c_concat.shape[2]):
                idx_t = w_idx if c_concat.device == x.device else w_cpu
                c_win['c_concat'] = c_concat.index_select(
                    2, idx_t.to(c_concat.device))

            # Inject NTK RoPE scaling if configured (not in segment mode,
            # where context_options may be None).
            ntk_scale = (
                context_options.rope_ntk_scale
                if context_options is not None else 1.0
            )
            if ntk_scale != 1.0:
                c_win['_rope_ntk_scale'] = ntk_scale

            # ── Per-window prompt embedding (segment prompt-travel) ─
            # Swap the positive text embedding to this window's segment
            # prompt.  The overlap region between adjacent windows blends
            # the two prompts via the crossfade mask below.
            if segment_mode and embeds_list[idx] is not None:
                c_win = _swap_positive_embed(
                    c_win, embeds_list[idx], cond_or_uncond)

            # Global RoPE positioning: each window's main latent AND
            # multi-frame context latents (source_video, T>1) use the
            # real global frame index as t_start.  Single-frame context
            # latents (ref_images, T=1) keep t_start=0 —
            # pre_forward decides per source_id: t_start for T>1, 0 for T=1.
            c_win['_rope_t_start'] = w[0]
            pred = apply_model_func(
                x_win, timestep, **c_win, cond_or_uncond=cond_or_uncond,
            )
            if not is_static:
                cov_win = coverage.index_select(2, w_idx)
                buf_out.index_add_(2, w_idx, pred * mask / cov_win)
                mask_ctr = mask[:, :, :, :1, :1]
                buf_counter.index_add_(2, w_idx, mask_ctr / cov_win)
            else:
                buf_out.index_add_(2, w_idx, pred * mask)
                mask_ctr = mask[:, :, :, :1, :1]
                buf_counter.index_add_(2, w_idx, mask_ctr)
            del pred  # free model output before next window iteration
        to.pop("_context_window", None)

        buf_counter.clamp_(min=1e-8)
        return buf_out.div_(buf_counter)

    return wrapper


def _inject_context_window(model_patcher, context_options: dict):
    """Inject the context window wrapper into *model_patcher*'s options.

    Returns the previous wrapper (or None) so it can be restored.
    """
    cw_wrapper = _build_context_window_wrapper(context_options)
    topts = model_patcher.model_options
    prev = topts.get("model_function_wrapper", None)
    if prev is not None:
        logger.info("[BerniniR] model_function_wrapper already exists; saving for restore.")
    topts["model_function_wrapper"] = cw_wrapper
    logger.info("[BerniniR] Context window model_function_wrapper injected.")
    return prev


def _remove_context_window(model_patcher, prev_wrapper=None) -> None:
    """Remove the context window wrapper and restore the previous one if any."""
    topts = model_patcher.model_options
    if prev_wrapper is not None:
        topts["model_function_wrapper"] = prev_wrapper
        logger.info("[BerniniR] Previous model_function_wrapper restored.")
    elif "model_function_wrapper" in topts:
        del topts["model_function_wrapper"]
        logger.info("[BerniniR] Context window wrapper removed.")


# ---------------------------------------------------------------------------
# BerniniR_KSampler
# ---------------------------------------------------------------------------

class BerniniR_KSampler:
    """Enhanced KSampler with optional temporal context window tiling.

    When ``context_options`` is connected, the sampler splits long video latents
    into overlapping temporal windows, denoises each window independently, and
    blends the results.  This dramatically reduces peak VRAM for long videos.

    When ``context_options`` is NOT connected, this node behaves identically to
    ComfyUI's built-in ``KSampler``.

    Inputs (required):
        model_handle (BERNINI_MODEL_HANDLE): The lazy-loading model handle from BerniniR_ModelLoader.
        seed (INT): Random seed for noise generation.
        steps (INT): Number of denoising steps.
        cfg (FLOAT): Classifier-free guidance scale.
        sampler_name (COMBO): ComfyUI sampler (e.g. euler, dpmpp_2m).
        scheduler (COMBO): ComfyUI scheduler (e.g. normal, simple, ddim_uniform).
        positive (CONDITIONING): Positive conditioning.
        negative (CONDITIONING): Negative conditioning.
        latent_image (LATENT): Initial latent.
        denoise (FLOAT): Denoising strength (1.0 = full denoise).
        flow_shift (FLOAT): Sigma schedule shift. Bernini-R was trained
            with 3.0. Higher = more low-noise steps, better spatial structure.

    Inputs (optional):
        context_options (BERNINI_CTX): Context window configuration.
        block_swap_args (BERNINI_BLOCKSWAP): Block swap configuration.
        teacache_args (BERNINI_TEACACHE): TeaCache configuration for faster
            sampling.  Leave disconnected to disable TeaCache.
        add_noise (BOOLEAN): Whether to add initial noise (default True).
        guidance_schedule (BERNINI_GUIDANCE): Per-step guidance scale from
            BerniniR_GuidanceStrengthSchedule. Overrides static cfg.

    Output:
        LATENT: Denoised latent.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_handle": ("BERNINI_MODEL_HANDLE", {"tooltip": "Bernini-R model handle from BerniniR_ModelLoader"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed for reproducible generation"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of denoising steps. 20-40 for flow matching"}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale. 4-7 typical for video"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm. 'uni_pc' or 'euler' work well with flow matching"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Sigma schedule. 'simple' or 'normal' recommended"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning"}),
                "latent_image": ("LATENT", {"tooltip": "Input latent (noise or encoded image for img2vid)"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength. 1.0 = full generation, <1.0 = img2vid"}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1, "tooltip": "Sigma schedule shift. Bernini-R trained with 3.0. Higher = more low-noise steps, sharper details"}),
            },
            "optional": {
                "context_options": ("BERNINI_CTX", {"tooltip": "Context window config from BerniniR_ContextWindow. Connect to enable temporal window tiling for long videos."}),
                "guidance_config": ("BERNINI_GUIDANCE_CONFIG", {"tooltip": "Guidance strategy + params from BerniniR_GuidanceConfig. Leave disconnected for plain CFG."}),
                "block_swap_args": ("BERNINI_BLOCKSWAP", {"tooltip": "Block swap config from BerniniR_BlockSwapArgs. Leave disconnected to disable."}),
                "teacache_args": ("BERNINI_TEACACHE", {"tooltip": "TeaCache config from BerniniR_TeaCacheArgs. Leave disconnected to disable TeaCache."}),
                "add_noise": ("BOOLEAN", {"default": True, "tooltip": "Add initial noise. Disable for img2vid when denoise < 1.0"}),
                "guidance_schedule": ("BERNINI_GUIDANCE", {"tooltip": "Dynamic per-step guidance scale from BerniniR_GuidanceStrengthSchedule. Overrides static cfg."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Bernini-R/Sampling"
    DESCRIPTION = (
        "Enhanced KSampler with temporal context window tiling for long videos. "
        "Connect BerniniR_ContextWindow to enable windowed sampling; leave "
        "disconnected for standard single-pass sampling."
    )

    def sample(
        self,
        model_handle,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        latent_image,
        denoise: float = 1.0,
        flow_shift: float = 3.0,
        guidance_config: BerniniGuidanceConfig | None = None,
        block_swap_args: BerniniBlockSwap | None = None,
        teacache_args: BerniniTeaCache | None = None,
        context_options: BerniniContext | None = None,
        add_noise: bool = True,
        guidance_schedule: BerniniGuidance | None = None,
    ):
        # ── Prepare initial noise ─────────────────────────────────────
        # NOTE: FreeNoise is intentionally NOT applied here.  It is applied
        # exactly ONCE inside bernini_sample / bernini_sample_dual, via
        # InjectionContext.apply_noise, after the unified injection context is
        # built and with the canonical seed.  Applying it here too would shuffle
        # the noise a *second* time (permutation-squared) on the single-sampler
        # path; the dual-expert path used to skip it entirely.  Converging on
        # the single call site keeps both paths correct.
        latent_samples = latent_image["samples"]
        if add_noise:
            batch_inds = latent_image.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        else:
            noise = torch.zeros_like(latent_samples)

        # ── TeaCache: load model early and attach hooks ───────────────
        _tc = None
        _tc_detach = None
        if teacache_args is not None:
            patcher = model_handle.load()
            batch = latent_samples.shape[0]
            _tc = TeaCache(
                patcher,
                start_block=teacache_args.start_block,
                max_skip_blocks=teacache_args.max_skip_blocks,
                rel_l1_thresh=teacache_args.rel_l1_thresh,
                warmup_steps=teacache_args.warmup_steps,
                cooldown_steps=teacache_args.cooldown_steps,
                batch_size=batch,
            )
            _tc.reset(steps)
            _tc_detach = _tc.detach

        # ── Callback ─────────────────────────────────────────────────
        callback = None
        if _tc is not None:
            x0_output = {}
            _base_cb = None
            if latent_preview is not None:
                _base_cb = latent_preview.prepare_callback(patcher, steps, x0_output)

            def _tc_callback(step, x0, x, total_steps):
                _tc.step()
                if _base_cb is not None:
                    _base_cb(step, x0, x, total_steps)

            callback = _tc_callback
        elif latent_preview is not None:
            callback = latent_preview.prepare_callback(model_handle, steps)

        # ── Delegate to self-contained sampling ─────────────────
        # Lazy import to avoid a circular import: bernini_sampling imports the
        # context-window wrapper from this module at top level.
        from .bernini_sampling import bernini_sample
        result = bernini_sample(
            model_handle=model_handle,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            seed=seed,
            guidance_config=guidance_config,
            guidance_schedule=guidance_schedule.values if guidance_schedule is not None else None,
            block_swap_args=block_swap_args,
            flow_shift=flow_shift,
            context_options=context_options,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            pre_unload_callback=_tc_detach,
        )
        samples = result["samples"]

        # ── Build output ────────────────────────────────────────────
        out = latent_image.copy()
        out.pop("downscale_ratio_spatial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples

        return (out,)




