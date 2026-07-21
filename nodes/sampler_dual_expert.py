"""
Dual-expert sampler with block-swap VRAM offloading.

Provides ``BerniniR_DualExpertSampler``: a drop-in replacement for
``BerniniR_KSampler`` that uses two models — a high-noise model for
early denoising steps and a low-noise model for later steps.  Both
models support GPU↔RAM block swapping to save VRAM.

Supports all guidance modes: CFG, APG, RAAG, S2, STG_A, STG_R.
"""

from __future__ import annotations


import torch
import comfy.sample
import comfy.samplers
import comfy.utils

try:
    import latent_preview
except Exception:
    latent_preview = None

from ..utils.types import BerniniBlockSwap, BerniniGuidance, BerniniContext, BerniniGuidanceConfig
from .bernini_sampling import bernini_sample_dual

from ..utils.log import get_logger as _get_logger

logger = _get_logger("Sampler")
class BerniniR_DualExpertSampler:
    """Dual-model sampler with block-swap VRAM offloading.

    The ``high_noise_model`` runs for the first ``split_step`` denoising
    steps (high sigma), then ``low_noise_model`` takes over for the
    remaining steps.  This enables FlowBlending-style generation where
    a larger model handles the critical early/late stages and a smaller
    model covers the intermediate steps.

    Block swap config comes from an external ``BERNINI_BLOCKSWAP`` input
    (``BerniniR_BlockSwapArgs``).  This allows fitting two models that would
    otherwise not fit together.

    Inputs (required):
        high_noise_model (BERNINI_MODEL_HANDLE): Handle for high-noise (early) steps.
        low_noise_model (BERNINI_MODEL_HANDLE): Handle for low-noise (late) steps.
        split_step (INT): Number of steps to use the high-noise model.
        seed, steps, cfg, sampler_name, scheduler, positive, negative,
        latent_image, denoise, flow_shift: standard KSampler params.
        guidance_mode, apg_eta, apg_rescale, apg_momentum, raag_alpha,
        s2_omega, stg_scale, stg_block_idx: guidance params.

    Inputs (optional):
        block_swap_args (BERNINI_BLOCKSWAP): Block swap config (0 = disabled).
        context_options (BERNINI_CTX): context window config.
        add_noise (BOOLEAN): Whether to add initial noise.
        guidance_schedule (BERNINI_GUIDANCE): dynamic schedule.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "high_noise_model": ("BERNINI_MODEL_HANDLE", {"tooltip": "Handle for high-noise (early) denoising steps"}),
                "low_noise_model": ("BERNINI_MODEL_HANDLE", {"tooltip": "Handle for low-noise (late) denoising steps"}),
                "split_step": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "Switch from high_noise to low_noise model after this many steps"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "latent_image": ("LATENT", {}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "context_options": ("BERNINI_CTX", {"tooltip": "Context window config from BerniniR_ContextWindow. Connect to enable temporal window tiling."}),
                "guidance_config": ("BERNINI_GUIDANCE_CONFIG", {"tooltip": "Guidance strategy from BerniniR_GuidanceConfig."}),
                "block_swap_args": ("BERNINI_BLOCKSWAP", {"tooltip": "Block swap config from BerniniR_BlockSwapArgs. Leave disconnected to disable."}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "guidance_schedule": ("BERNINI_GUIDANCE", {}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Bernini-R/Sampling"
    DESCRIPTION = "Dual-expert sampler: high_noise model for early steps, low_noise model for late steps. Block swap saves VRAM."

    def sample(
        self,
        high_noise_model,
        low_noise_model,
        split_step: int,
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
        context_options: BerniniContext | None = None,
        add_noise: bool = True,
        guidance_schedule: BerniniGuidance | None = None,
    ):
        # ── Prepare noise ──────────────────────────────────────────
        latent_samples = latent_image["samples"]
        if add_noise:
            batch_inds = latent_image.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        else:
            noise = torch.zeros_like(latent_samples)

        # ── Callback ────────────────────────────────────────────────
        callback = None
        if latent_preview is not None:
            callback = latent_preview.prepare_callback(high_noise_model, steps)

        result = bernini_sample_dual(
            high_model=high_noise_model,
            low_model=low_noise_model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            split_step=split_step,
            denoise=denoise,
            seed=seed,
            guidance_config=guidance_config,
            guidance_schedule=guidance_schedule.values if guidance_schedule is not None else None,
            flow_shift=flow_shift,
            context_options=context_options,
            block_swap_args=block_swap_args,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        )
        return (result,)
