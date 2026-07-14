"""
Standalone block-swap configuration node.

Decouples the block-swap setting from the sampler so it can be wired
independently and reused across samplers.
"""
from __future__ import annotations

from ..utils.types import BerniniBlockSwap


class BerniniR_BlockSwapArgs:
    """Build a block-swap configuration object for Bernini-R samplers.

    ``block_to_swap`` is the number of transformer blocks to keep in RAM;
    the remaining blocks stay on GPU.  Set to 0 to disable block swap
    (keep all blocks on GPU).

    ``prefetch`` enables a dedicated CUDA stream that loads upcoming
    window blocks while the current block computes, hiding H2D transfer
    latency and allowing smaller GPU windows.

    ``prefetch_count`` controls how many blocks are prefetched at once.

    ``pin_memory`` pins the CPU copies of offloaded blocks.  This makes
    async transfers truly non-blocking, but increases host RAM usage.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "block_to_swap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of transformer blocks to keep in RAM. 0 = disable block swap.",
                }),
            },
            "optional": {
                "prefetch": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use a CUDA stream to prefetch upcoming blocks during compute.",
                }),
                "prefetch_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of blocks to prefetch ahead of the current window.",
                }),
                "pin_memory": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pin CPU copies for faster async transfer. Increases host RAM.",
                }),
            },
        }

    RETURN_TYPES = ("BERNINI_BLOCKSWAP",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "build_args"
    CATEGORY = "Bernini-R/Config"
    DESCRIPTION = (
        "Configure block-swap VRAM offloading. Connect to Bernini-R KSampler "
        "or Dual Expert Sampler."
    )

    def build_args(
        self,
        block_to_swap: int,
        prefetch: bool = True,
        prefetch_count: int = 1,
        pin_memory: bool = False,
    ):
        return (BerniniBlockSwap(
            block_to_swap=block_to_swap,
            prefetch=prefetch,
            prefetch_count=prefetch_count,
            pin_memory=pin_memory,
        ),)
