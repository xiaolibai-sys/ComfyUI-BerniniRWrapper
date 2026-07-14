"""
NAG (Normalized Attention Guidance) node for Bernini-R.

Mirrors WanVideoWrapper's ``WanVideoApplyNAG``: merges a negative prompt's
text embedding into the positive conditioning dict as ``nag_prompt_embeds``
and ``nag_params``.  The sampler detects these keys and forwards them through
``transformer_options`` so the cross-attention NAG hooks can use them.

Also provides a passthrough output for the negative conditioning so it can
be wired directly to the sampler for standard CFG.
"""
from __future__ import annotations

import logging

from ..utils.types import Conditioning

logger = logging.getLogger(__name__)


class BerniniR_ApplyNAG:
    """Attach NAG guidance data to positive conditioning.

    Extracts the negative prompt's text embedding and injects it into the
    positive conditioning as ``nag_prompt_embeds``, along with NAG hyper-
    params (scale, tau, alpha, inplace).  The sampler reads these keys from
    ``transformer_options`` to enable cross-attention NAG hooks.

    The negative conditioning is passed through on a separate output so it
    can be connected to the sampler for standard CFG alongside NAG.

    Inputs:
        positive (CONDITIONING):  positive text conditioning.
        nag_scale (FLOAT):  guidance strength  (default 11.0).
        nag_tau (FLOAT):    norm-clamp threshold  (default 2.5).
        nag_alpha (FLOAT):  blend factor  (default 0.25).
        inplace (BOOLEAN):  reuse tensors to save VRAM  (default True).

    Optional:
        negative (CONDITIONING):  prompt used for NAG guidance (typically the
            same negative prompt already encoded).  If not connected, NAG is
            disabled and both outputs pass through unchanged.

    Output:
        positive (CONDITIONING):  positive copy with ``nag_prompt_embeds``
            and ``nag_params`` injected.
        negative (CONDITIONING):  negative conditioning passthrough for CFG.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "Positive text conditioning"}),
                "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "NAG guidance strength. Higher = stronger attention steering"}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Norm-clamp threshold. Controls how aggressively outlier attention is suppressed"}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend factor between original and NAG-steered attention"}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "Negative text conditioning (used for both CFG and NAG guidance)"}),
                "inplace": ("BOOLEAN", {"default": True, "tooltip": "Reuse tensors in-place to save VRAM"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "Bernini-R/Conditioning"
    DESCRIPTION = (
        "Attaches NAG (Normalized Attention Guidance) data to the positive "
        "conditioning and passes through negative for CFG.  "
        "Connect the same negative prompt used for CFG as the NAG guidance "
        "signal.  If negative is not connected, both outputs pass through "
        "unchanged (NAG disabled).  "
        "https://github.com/ChenDarYen/Normalized-Attention-Guidance"
    )

    def apply(
        self,
        positive,
        negative=None,
        nag_scale: float = 11.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        inplace: bool = True,
    ):
        # ── NAG requires negative conditioning ───────────────────────
        if negative is None or len(negative) == 0 or len(negative[0]) == 0:
            raise ValueError(
                "[BerniniR] NAG requires a negative conditioning input. "
                "Connect both positive and negative from the prompt planner."
            )

        nag_embeds = Conditioning.from_comfy(negative).embed

        # ── Inject NAG data into a shallow copy of positive ──────────
        nag_params = {
            "nag_scale": nag_scale,
            "nag_tau": nag_tau,
            "nag_alpha": nag_alpha,
            "inplace": inplace,
        }

        out = Conditioning.from_comfy(positive).with_extra(
            nag_prompt_embeds=nag_embeds,
            nag_params=nag_params,
        )

        logger.info(
            "[BerniniR] NAG active: scale=%.1f, tau=%.1f, alpha=%.2f, inplace=%s",
            nag_scale, nag_tau, nag_alpha, inplace,
        )
        # Positive → sampler positive (with NAG), negative → sampler negative (for CFG)
        return (out.to_comfy(), negative)
