"""
VAE nodes for Bernini-R: tiled encode and decode with optional color matching.

Uses the bundled ``BerniniRVAE`` wrapper (``utils.vae_wrapper``), which is
loaded by ``BerniniR_VAELoader`` instead of ComfyUI's native ``comfy.sd.VAE``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from ..utils.color_match import COLORMATCH_METHODS, apply_color_match
from ..utils.vram import inference_mode, collect_garbage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BerniniR_VAEDecode
# ---------------------------------------------------------------------------

class BerniniR_VAEDecode:
    """VAE Decode with spatial tiling and optional color matching.

    Decodes latent → video pixels, optionally applying a colour transfer
    from a reference image to improve temporal consistency.

    Inputs (required):
        vae (VAE): The Bernini-R VAE.
        samples (LATENT): Latent to decode.

    Inputs (optional):
        colormatch (COMBO): Color matching algorithm.
        enable_tiling (BOOLEAN): Enable spatial tiling for VRAM-limited GPUs.
        tile_x (INT): Tile width in pixels.
        tile_y (INT): Tile height in pixels.
        tile_stride_x (INT): Horizontal stride in pixels.
        tile_stride_y (INT): Vertical stride in pixels.
        ref_image (IMAGE): Reference image for color matching.
            If not provided and colormatch is enabled, the first decoded
            frame is used as reference.
        blend_strength (FLOAT): Color match blend: 0.0 = no change, 1.0 = full match.

    Output:
        IMAGE: Decoded video frames (F, H, W, C).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "Bernini-R VAE (Wan 16-channel, 4x temporal downscale)"}),
                "samples": ("LATENT", {"tooltip": "Latent to decode (from sampler output)"}),
            },
            "optional": {
                "colormatch": (COLORMATCH_METHODS, {"default": "disabled",
                    "tooltip": "Color transfer algorithm. 'hm-mvgd-hm' is a good default for temporal consistency"}),
                "enable_tiling": ("BOOLEAN", {"default": False,
                    "tooltip": "Spatial tiling for VRAM-limited GPUs. Enables decoding very large frames"}),
                "tile_x": ("INT", {"default": 272, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Tile width in pixels"}),
                "tile_y": ("INT", {"default": 272, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Tile height in pixels"}),
                "tile_stride_x": ("INT", {"default": 144, "min": 16, "max": 4096, "step": 16,
                    "tooltip": "Horizontal stride (must be < tile_x). Larger = fewer tiles, more artifacts"}),
                "tile_stride_y": ("INT", {"default": 128, "min": 16, "max": 4096, "step": 16,
                    "tooltip": "Vertical stride (must be < tile_y). Larger = fewer tiles, more artifacts"}),
                "ref_image": ("IMAGE", {"tooltip": "Reference image for color matching. If not provided, uses first decoded frame per batch"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Color match blend. 0.0 = original, 1.0 = full transfer"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "Bernini-R/VAE"
    DESCRIPTION = (
        "Tiled VAE decode with optional colour matching. "
        "Matches decoded video colour distribution to a reference image "
        "for improved temporal consistency."
    )

    def decode(
        self,
        vae,
        samples,
        colormatch: str = "disabled",
        enable_tiling: bool = False,
        tile_x: int = 272,
        tile_y: int = 272,
        tile_stride_x: int = 144,
        tile_stride_y: int = 128,
        ref_image: Optional[torch.Tensor] = None,
        blend_strength: float = 1.0,
    ):
        # ── Validate tiles ───────────────────────────────────────────
        if enable_tiling:
            if tile_x <= tile_stride_x:
                raise ValueError(
                    f"tile_x ({tile_x}) must be > tile_stride_x ({tile_stride_x})"
                )
            if tile_y <= tile_stride_y:
                raise ValueError(
                    f"tile_y ({tile_y}) must be > tile_stride_y ({tile_stride_y})"
                )

        latent_samples = samples["samples"]

        # ── Decode ────────────────────────────────────────────────────
        with inference_mode():
            try:
                pixels = vae.decode(
                    latent_samples,
                    tiled=enable_tiling,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    tile_stride_x=tile_stride_x,
                    tile_stride_y=tile_stride_y,
                )
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    "[BerniniR] VAE decode ran out of VRAM. "
                    "Try enabling spatial tiling or reducing the number of frames."
                )

        # BerniniRVAE.decode returns (F, H, W, C) directly.
        pixels = pixels.clamp(0.0, 1.0)

        # ── Release VAE VRAM ─────────────────────────────────────────
        # The VAE decoder produces large intermediates (3D convs,
        # upsampling layers); the model itself can be offloaded and
        # the CUDA cache flushed so downstream nodes have headroom.
        del latent_samples
        if hasattr(vae, "offload"):
            try:
                vae.offload()
            except Exception:
                pass
        collect_garbage()

        # ── Optional color matching ──────────────────────────────────
        if colormatch != "disabled":
            if ref_image is None:
                ref_image = pixels[:1]
                logger.info(
                    "[BerniniR] No ref_image provided; "
                    "using first decoded frame as reference."
                )

            logger.info(
                f"[BerniniR] Applying color match: method={colormatch}, "
                f"blend={blend_strength}"
            )
            pixels = apply_color_match(
                pixels, ref_image,
                method=colormatch,
                blend_strength=blend_strength,
            )

        return (pixels,)


# ---------------------------------------------------------------------------
# BerniniR_VAEEncode
# ---------------------------------------------------------------------------

class BerniniR_VAEEncode:
    """VAE Encode with spatial tiling.

    Encodes video/image pixels → latent for denoising.

    Inputs:
        vae (VAE): The Bernini-R VAE.
        pixels (IMAGE): Video frames (F, H, W, C) or image (H, W, C).
        enable_tiling (BOOLEAN): Enable spatial tiling for large inputs.
        tile_x (INT): Tile width.
        tile_y (INT): Tile height.
        tile_stride_x (INT): Horizontal stride.
        tile_stride_y (INT): Vertical stride.

    Output:
        LATENT: Encoded latent dict.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "Bernini-R VAE (Wan 16-channel, 4x temporal downscale)"}),
                "pixels": ("IMAGE", {"tooltip": "Video frames (F, H, W, C) in [0, 1]"}),
            },
            "optional": {
                "enable_tiling": ("BOOLEAN", {"default": False,
                    "tooltip": "Spatial tiling for VRAM-limited GPUs"}),
                "tile_x": ("INT", {"default": 272, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Tile width in pixels"}),
                "tile_y": ("INT", {"default": 272, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Tile height in pixels"}),
                "tile_stride_x": ("INT", {"default": 144, "min": 16, "max": 4096, "step": 16,
                    "tooltip": "Horizontal stride (must be < tile_x)"}),
                "tile_stride_y": ("INT", {"default": 128, "min": 16, "max": 4096, "step": 16,
                    "tooltip": "Vertical stride (must be < tile_y)"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "Bernini-R/VAE"
    DESCRIPTION = "Tiled VAE encode from image/video pixels to latent."

    def encode(
        self,
        vae,
        pixels: torch.Tensor,
        enable_tiling: bool = False,
        tile_x: int = 272,
        tile_y: int = 272,
        tile_stride_x: int = 144,
        tile_stride_y: int = 128,
    ):
        # ── Validate tiles ───────────────────────────────────────────
        if enable_tiling:
            if tile_x <= tile_stride_x:
                raise ValueError(
                    f"tile_x ({tile_x}) must be > tile_stride_x ({tile_stride_x})"
                )
            if tile_y <= tile_stride_y:
                raise ValueError(
                    f"tile_y ({tile_y}) must be > tile_stride_y ({tile_stride_y})"
                )

        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)  # (H,W,C) → (1,H,W,C)

        # ── Encode ───────────────────────────────────────────────────
        with inference_mode():
            try:
                latent = vae.encode(
                    pixels,
                    tiled=enable_tiling,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    tile_stride_x=tile_stride_x,
                    tile_stride_y=tile_stride_y,
                )
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    "[BerniniR] VAE encode ran out of VRAM. "
                    "Try enabling spatial tiling or reducing the input resolution."
                )

        # ── Release VAE VRAM ─────────────────────────────────────────
        if hasattr(vae, "offload"):
            try:
                vae.offload()
            except Exception:
                pass
        collect_garbage()

        return ({"samples": latent},)
