"""
Context window configuration node for Bernini-R.

Mirrors WanVideoWrapper's WanVideoContextOptions node exactly:
pixel-frame parameters are converted to latent space when the
BerniniContext is constructed (see utils/types.py:70), not by the sampler.

Inputs configure how long videos are split into temporal windows during
denoising, including scheduling strategy, window size, overlap, and blending.
"""
from __future__ import annotations


from ..utils.types import BerniniContext, ContextSchedule, FuseMethod

from ..utils.log import get_logger as _get_logger

logger = _get_logger("CtxWin")
CONTEXT_SCHEDULES = ["uniform_standard", "uniform_looped", "static_standard"]
FUSE_METHODS = ["linear", "smooth", "pyramid"]


class BerniniR_ContextWindow:
    """Configure temporal tiling for long video generation.

    Parameters exactly mirror WanVideoWrapper's ``WanVideoContextOptions``
    with the same defaults and constraints.

    Inputs:
        context_schedule (COMBO): Window scheduling strategy.
            - ``uniform_standard``: Multi-stride windows, deduplicated.
            - ``uniform_looped``:  Cyclic windows for looped video.
            - ``static_standard``: Fixed sliding window (simplest, fastest).
        context_frames (INT): Number of pixel frames per window (default 81).
        context_stride (INT): Stride level (log2 scale, default 4).
        context_overlap (INT): Overlap between windows in pixel frames (default 16).
        freenoise (BOOLEAN): Shuffle noise across windows to reduce tiling artifacts.
        fuse_method (COMBO): Window blending function.
            - ``linear``:  Ramp 0→1 on left overlap, 1→0 on right overlap.
            - ``smooth``:  Smoothstep-eased crossfade (C1-continuous) — same as
                           linear but with zero slope at the edges, so the
                           transition band reads as far less of a brightness
                           bump / graying at window seams. Recommended default.
            - ``pyramid``: Triangle weights peaking at window centre.

    Output:
        BERNINI_CTX: Configuration dict consumed by BerniniR_KSampler.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_schedule": (CONTEXT_SCHEDULES, {"default": "static_standard", "tooltip": "Window scheduling: 'static_standard' = simple sliding (fastest), 'uniform_standard' = multi-stride (best quality), 'uniform_looped' = cyclic for seamless looping"}),
                "context_frames": ("INT", {"default": 81, "min": 2, "max": 1000, "step": 1, "tooltip": "Pixel frames per window. 81 ≈ 20 latent frames. Smaller = less VRAM, more windows"}),
                "context_stride": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1, "tooltip": "Stride levels in log2 scale. Higher = more window positions (uniform_* only)"}),
                "context_overlap": ("INT", {"default": 16, "min": 0, "max": 100, "step": 1, "tooltip": "Overlap between windows in pixel frames. Smooths transitions. 12-16 typical"}),
                "freenoise": ("BOOLEAN", {"default": True, "tooltip": "Shuffle noise across windows to reduce seam artifacts"}),
            },
            "optional": {
                "fuse_method": (FUSE_METHODS, {"default": "smooth", "tooltip": "Blending: 'smooth' (default) = smoothstep crossfade (softest seams), 'linear' = crossfade at edges, 'pyramid' = triangle weights peaking at center"}),
                "rope_ntk_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.05, "tooltip": "NTK RoPE frequency scaling for sequences longer than training. 1.0 = off. 2.0 = moderate (2x training length), 3.0+ = aggressive. Scales rope_embedder.theta to prevent positional collapse."}),
            },
        }

    RETURN_TYPES = ("BERNINI_CTX",)
    RETURN_NAMES = ("context_options",)
    FUNCTION = "build_context"
    CATEGORY = "Bernini-R/Sampling"
    DESCRIPTION = (
        "Temporal context windows for long video generation. "
        "Configures how the sampler splits video frames into overlapping windows "
        "for memory-efficient processing. Mirrors WanVideoWrapper's "
        "WanVideoContextOptions."
    )

    def build_context(
        self,
        context_schedule: str = "static_standard",
        context_frames: int = 81,
        context_stride: int = 4,
        context_overlap: int = 16,
        freenoise: bool = True,
        fuse_method: str = "smooth",
        rope_ntk_scale: float = 1.0,
    ):
        ctx = BerniniContext(
            schedule=ContextSchedule(context_schedule),
            context_frames=context_frames,
            context_stride=context_stride,
            context_overlap=context_overlap,
            freenoise=freenoise,
            fuse_method=FuseMethod(fuse_method),
            rope_ntk_scale=rope_ntk_scale,
        )
        logger.info(f"Context window config: schedule={context_schedule}, "
                    f"frames={context_frames}, overlap={context_overlap}, "
                    f"stride={context_stride}, freenoise={freenoise}, fuse={fuse_method}")
        return (ctx,)
