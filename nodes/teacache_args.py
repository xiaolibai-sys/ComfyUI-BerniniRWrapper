"""
Standalone TeaCache configuration node.

Decouples TeaCache settings from the sampler so it can be wired
independently and reused across samplers.
"""
from __future__ import annotations

from ..utils.types import BerniniTeaCache
from ..utils.teacache import (
    DEFAULT_START_BLOCK,
    DEFAULT_MAX_SKIP_BLOCKS,
    DEFAULT_REL_L1_THRESH,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_COOLDOWN_STEPS,
)


class BerniniR_TeaCacheArgs:
    """Build a TeaCache configuration object for Bernini-R samplers.

    TeaCache skips redundant transformer blocks when consecutive denoising
    steps produce near-identical hidden states — typically 1.5-2× speedup
    with negligible quality loss.

    Connect to the ``teacache_args`` input of ``BerniniR_KSampler``.
    Leave disconnected to disable TeaCache entirely.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_skip_blocks": ("INT", {
                    "default": DEFAULT_MAX_SKIP_BLOCKS,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Number of blocks in the caching window. Larger = more speedup but higher quality risk.",
                }),
                "start_block": ("INT", {
                    "default": DEFAULT_START_BLOCK,
                    "min": 0,
                    "max": 29,
                    "step": 1,
                    "tooltip": "Index of the first cacheable block. L1 distance is compared here to decide skip vs compute.",
                }),
                "rel_l1_thresh": ("FLOAT", {
                    "default": DEFAULT_REL_L1_THRESH,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Skip blocks when L1 distance < threshold. 0.04 = safe (minimal quality loss), 0.12 = fast (may degrade).",
                }),
                "warmup_steps": ("INT", {
                    "default": DEFAULT_WARMUP_STEPS,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "First N denoising steps that never cache (structure formation phase).",
                }),
                "cooldown_steps": ("INT", {
                    "default": DEFAULT_COOLDOWN_STEPS,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Last N denoising steps that never cache (detail refinement phase).",
                }),
            },
        }

    RETURN_TYPES = ("BERNINI_TEACACHE",)
    RETURN_NAMES = ("teacache_args",)
    FUNCTION = "build_args"
    CATEGORY = "Bernini-R/Config"
    DESCRIPTION = (
        "Configure TeaCache block caching for faster sampling. "
        "Connect to BerniniR_KSampler to enable; leave disconnected to disable."
    )

    def build_args(
        self,
        max_skip_blocks: int = DEFAULT_MAX_SKIP_BLOCKS,
        start_block: int = DEFAULT_START_BLOCK,
        rel_l1_thresh: float = DEFAULT_REL_L1_THRESH,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        cooldown_steps: int = DEFAULT_COOLDOWN_STEPS,
    ):
        return (BerniniTeaCache(
            max_skip_blocks=max_skip_blocks,
            start_block=start_block,
            rel_l1_thresh=rel_l1_thresh,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
        ),)
