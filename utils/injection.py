"""
Centralised injection context for the Bernini-R sampling pipeline.

``InjectionContext`` is built once at the start of ``bernini_sample()`` /
``bernini_sample_dual()`` and holds all conditioning-extracted data, config
values, and parsed arguments that previously were scattered across ~100 lines
of ad-hoc extraction and ``extra_model_options`` mutation.

Having a single source of truth:
* Eliminates the duplicated extraction in the single- and dual-expert paths.
* Makes ``extra_model_options`` construction a single ``apply_options()`` call.
* Lets ``BerniniModelWrapper`` receive pre-extracted state instead of reading
  from raw conditioning dicts and the diffusion model directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from .parsing import parse_stg_block_indices
from .types import (
    BerniniBlockSwap,
    BerniniContext,
    BerniniGuidanceConfig,
    Conditioning,
    GuidanceMode,
)

logger = logging.getLogger(__name__)


@dataclass
class InjectionContext:
    """All injection data extracted once at sampling start.

    Built by :meth:`build` from the raw node inputs, then consumed by
    :meth:`apply_options`, :meth:`apply_block_swap`, and
    :meth:`apply_noise`.

    Attributes prefixed with ``_dd_`` feed the differential-diffusion path
    inside :class:`~.bernini_sampling.BerniniModelWrapper`.
    """

    # -- extracted from positive conditioning ---------------------------------
    nag_context: torch.Tensor | None = None
    nag_params: dict[str, Any] | None = None

    # -- differential-diffusion (extracted from conditioning, consumed by
    #    BerniniModelWrapper) --------------------------------------------------
    dd_edit_mask: torch.Tensor | None = None
    dd_mask_mode: str = "anneal"
    dd_src_latent: torch.Tensor | None = None

    # -- from config nodes ----------------------------------------------------
    freenoise: bool = False
    block_to_swap: int = 0
    block_swap_prefetch: bool = True
    block_swap_prefetch_count: int = 1
    block_swap_pin_memory: bool = False
    block_swap_loading_mode: str = "Streaming"
    context_window_wrapper: callable | None = None

    # -- parsed STG -----------------------------------------------------------
    stg_blocks: list[int] = field(default_factory=list)
    stg_mode: str = "A"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        positive,
        context_options: BerniniContext | None,
        block_swap_args: BerniniBlockSwap | None,
        guidance_config: BerniniGuidanceConfig | None,
        total_blocks: int,
        context_window_wrapper: callable | None = None,
    ) -> "InjectionContext":
        """Extract all injection data from node inputs in one call.

        Parameters
        ----------
        positive:
            ComfyUI CONDITIONING list (``[[tensor, dict], ...]``).
        context_options:
            :class:`BerniniContext` or ``None``.
        block_swap_args:
            :class:`BerniniBlockSwap` or ``None``.
        guidance_config:
            :class:`BerniniGuidanceConfig` or ``None`` (defaults to CFG).
        total_blocks:
            Number of transformer blocks in the model (for STG parsing).
        context_window_wrapper:
            Pre-built context-window wrapper callable, or ``None``.
        """
        ctx = cls()

        # -- conditioning extraction ------------------------------------------
        if positive:
            pos_cond = Conditioning.from_comfy(positive)

            ctx.nag_context = pos_cond.get_extra("nag_prompt_embeds")
            ctx.nag_params = pos_cond.get_extra("nag_params")
            if ctx.nag_context is not None and ctx.nag_params:
                logger.info(
                    "[BerniniR] NAG enabled: scale=%.1f, tau=%.1f, alpha=%.2f",
                    ctx.nag_params.get("nag_scale", 0),
                    ctx.nag_params.get("nag_tau", 0),
                    ctx.nag_params.get("nag_alpha", 0),
                )

            # Differential-diffusion
            _edit_mask = pos_cond.get_extra("edit_mask")
            if _edit_mask is not None:
                ctx.dd_edit_mask = _edit_mask
                ctx.dd_mask_mode = pos_cond.get_extra("mask_mode", "anneal")
                _src = pos_cond.get_extra("context_latents")
                if _src is not None:
                    if hasattr(_src, "cond"):
                        _src = _src.cond
                    if _src:
                        ctx.dd_src_latent = _src[0]

        # -- config nodes -----------------------------------------------------
        if context_options is not None:
            ctx.freenoise = context_options.freenoise
        if block_swap_args is not None:
            ctx.block_to_swap = block_swap_args.block_to_swap
            ctx.block_swap_prefetch = block_swap_args.prefetch
            ctx.block_swap_prefetch_count = block_swap_args.prefetch_count
            ctx.block_swap_pin_memory = block_swap_args.pin_memory
            ctx.block_swap_loading_mode = block_swap_args.loading_mode

        ctx.context_window_wrapper = context_window_wrapper

        # -- parsed STG / guidance -------------------------------------------
        if guidance_config is not None:
            if guidance_config.stg_block_idx.strip():
                ctx.stg_blocks = parse_stg_block_indices(
                    guidance_config.stg_block_idx, total_blocks)
            ctx.stg_mode = (
                guidance_config.mode.value[4:]
                if guidance_config.mode.value.startswith("STG") else "A"
            )
        else:
            ctx.stg_mode = "A"

        return ctx

    # ------------------------------------------------------------------
    # Apply methods — each mutates its target in one shot
    # ------------------------------------------------------------------

    def apply_options(
        self,
        extra_model_options: dict,
        sigmas: torch.Tensor,
        batch: int,
    ) -> None:
        """Apply all ``transformer_options`` / ``model_options`` injections.

        Call this **once** after ``create_model_options_clone()`` and before
        ``model.pre_run()``.
        """
        to = extra_model_options.setdefault("transformer_options", {})

        # 1. sample_sigmas — always injected (TeaCache + hooks need it)
        to["sample_sigmas"] = sigmas

        # 2. NAG — inject into cloned options so cache doesn't leak
        if self.nag_context is not None and self.nag_params:
            nag_ctx = self.nag_context
            if batch > 1 and nag_ctx.shape[0] == 1:
                nag_ctx = nag_ctx.repeat(batch, 1, 1)
            to.update({
                "nag_context": nag_ctx,
                "nag_params": self.nag_params,
            })

        # 3. Context window wrapper — on clone only, no patcher mutation
        if self.context_window_wrapper is not None:
            extra_model_options["model_function_wrapper"] = (
                self.context_window_wrapper
            )

    def apply_block_swap(
        self,
        extra_model_options: dict,
        inner_model,
    ) -> None:
        """Inject block-swap config and disable torch.compile if needed.

        Must be called *after* :meth:`apply_options` and *after*
        ``model.pre_run()`` because it needs access to the inner model to
        compute ``blocks_on_gpu`` and check for ``_original_transformer_forward``.
        """
        if self.block_to_swap <= 0:
            return

        dm = getattr(inner_model, "diffusion_model", inner_model)
        total_blocks = len(dm.blocks) if hasattr(dm, "blocks") else 0
        if total_blocks <= 0:
            return

        blocks_on_gpu = max(1, total_blocks - self.block_to_swap)

        # Dynamic GPU↔CPU block loading breaks the traced graph.
        _orig = getattr(dm, "_original_transformer_forward", None)
        if _orig is not None:
            dm.transformer_forward = _orig
            logger.warning(
                "[BerniniR] Block swap is incompatible with "
                "torch.compile — compile disabled for this run."
            )

        extra_model_options.setdefault("transformer_options", {}).update({
            "_block_swap": True,
        })

    def apply_noise(
        self,
        noise: torch.Tensor,
        context_options: BerniniContext | None,
        seed: int,
    ) -> torch.Tensor:
        """Apply FreeNoise shuffling if enabled.  Returns (possibly new) noise."""
        if not self.freenoise or context_options is None:
            return noise
        from ..nodes.sampler import _apply_freenoise
        return _apply_freenoise(noise, context_options, seed=seed)
