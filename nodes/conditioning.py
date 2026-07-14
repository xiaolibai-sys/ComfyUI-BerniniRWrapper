"""
Bernini-R conditioning node.

Creates the initial latent and attaches in-context reference latents
(source video, reference video, reference images) to the conditioning
for Bernini-R's in-context generation.

Supports:
  - Chunked VAE encoding for long videos (avoids OOM).
  - Batch generation: batch_size > 1 creates multiple latents in parallel.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils
import node_helpers

from ..utils.vram import inference_mode, log_memory, collect_garbage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image / video preprocessing
# ---------------------------------------------------------------------------

def _resize_mask_to_latent(mask, t_latent, h_latent, w_latent, device):
    """Pixel/frame mask -> latent-grid soft mask.

    Pixel-domain feathering is the caller's responsibility; here we only do a
    latent-space trilinear resize to produce per-token continuous blend weights.
    Output shape: [B, 1, T_lat, H_lat, W_lat].
    """
    m = mask.to(device).float()
    if m.dim() == 3:                       # [B, H, W] single-image mask -> reused across all time
        m = m.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, H, W]
    elif m.dim() == 4:                     # [B, F, H, W] video, or [B, 1, H, W]
        if m.shape[1] == 1:
            m = m.unsqueeze(2)             # [B, 1, 1, H, W]
        else:
            m = m.unsqueeze(1)             # [B, 1, F, H, W]
    # now m is [B, 1, F, H, W]
    m = F.interpolate(
        m, size=(t_latent, h_latent, w_latent),
        mode="trilinear", align_corners=False,
    )
    return m


def _resize_long_edge(image: torch.Tensor, max_size: int, stride: int = 16) -> torch.Tensor:
    """Resize image so its long edge is at most max_size, snapped to stride."""
    B, H, W, C = image.shape
    if max(H, W) <= max_size:
        h_snapped = (H // stride) * stride
        w_snapped = (W // stride) * stride
        if h_snapped != H or w_snapped != W:
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1), w_snapped, h_snapped, "area", "disabled"
            ).movedim(1, -1)
        return image

    if H >= W:
        new_h = (max_size // stride) * stride
        scale = new_h / H
        new_w = int(W * scale)
        new_w = (new_w // stride) * stride
    else:
        new_w = (max_size // stride) * stride
        scale = new_w / W
        new_h = int(H * scale)
        new_h = (new_h // stride) * stride

    image = comfy.utils.common_upscale(
        image.movedim(-1, 1), new_w, new_h, "area", "disabled"
    ).movedim(1, -1)
    return image


# ---------------------------------------------------------------------------
# Chunked VAE encoding
# ---------------------------------------------------------------------------

def _encode_video_chunked(
    vae,
    video: torch.Tensor,
    chunk_frames: int = 16,
    temporal_overlap: int = 4,
) -> torch.Tensor:
    """VAE-encode video in temporal chunks to bound peak VRAM.

    Splits the video along the frame (batch) dimension into chunks of
    ``chunk_frames`` with ``temporal_overlap`` pixel frames of overlap
    between adjacent chunks.  Overlap frames are discarded after encoding
    to avoid boundary discontinuities caused by the VAE's 3D convolutions
    seeing truncated temporal context at chunk edges.

    Args:
        vae: ComfyUI VAE object.
        video: [F, H, W, C] in ComfyUI IMAGE format.
        chunk_frames: Max pixel frames per encoding chunk.
        temporal_overlap: Pixel frames of overlap between chunks.
            Must be ≥ 4 (one latent frame).  Overlap regions are
            encoded twice and the boundary portions discarded.

    Returns:
        Encoded latent tensor [1, z_dim*2, T_latent, H//8, W//8].
    """
    F = video.shape[0]
    # Snap chunk_frames to a multiple of 4 (VAE temporal stride).
    # Non-multiple values break the overlap-trimming arithmetic:
    # the trim formula assumes chunk_frames % 4 == 0, otherwise
    # T_lat = (C-1)//4+1 produces one extra latent frame per chunk
    # that accumulates across chunks, inflating the total count.
    # Minimum is 8 (two latent frames) so stride ≥ 4.
    chunk_frames = max((chunk_frames // 4) * 4, 8)
    if F <= chunk_frames:
        return _encode_single(vae, video)

    # Enforce minimum overlap of one latent frame (4 pixel frames)
    temporal_overlap = max(temporal_overlap, 4)
    stride = chunk_frames - temporal_overlap
    if stride <= 0:
        return _encode_single(vae, video)

    # Convert overlap from pixel frames to latent frames
    overlap_lat = temporal_overlap // 4

    logger.info(
        f"[BerniniR] Chunked VAE encode: {F} frames → "
        f"{math.ceil(F / stride)} chunks of ≤{chunk_frames}"
        f" (overlap={temporal_overlap}px / {overlap_lat}lat)"
    )

    latent_chunks = []
    prev_end = 0  # end pixel frame of the previous chunk
    for start in range(0, F, stride):
        end = min(start + chunk_frames, F)
        # Skip chunk that adds no new frames beyond previous coverage.
        if end <= prev_end:
            break
        chunk = video[start:end]  # [chunk_F, H, W, C]
        lat = _encode_single(vae, chunk)

        # Trim overlap from the left edge of all chunks except the first.
        # Guard: only trim if the chunk has more latent frames than the
        # overlap — a degenerate end chunk (e.g. 1 px frame → 1 lat frame)
        # must be kept intact.
        if latent_chunks and lat.shape[2] > overlap_lat:
            lat = lat[:, :, overlap_lat:]
        latent_chunks.append(lat)
        prev_end = end
        log_memory(f"Chunked encode {start}-{end} / {F}")

    # Concatenate along latent temporal dim (dim=2)
    result = torch.cat(latent_chunks, dim=2)
    return result


def _encode_single(vae, video: torch.Tensor) -> torch.Tensor:
    """VAE-encode a video chunk in one call.

    Args:
        vae: ComfyUI VAE object.
        video: [F, H, W, C] in ComfyUI IMAGE format.

    Returns:
        Latent tensor from VAE.encode.
    """
    if video.dim() == 3:
        video = video.unsqueeze(0)
    with inference_mode():
        latent = vae.encode(video)
    return latent


# ---------------------------------------------------------------------------
# BerniniR_Conditioning
# ---------------------------------------------------------------------------

class BerniniR_Conditioning:
    """Prepare conditioning and latent for Bernini-R in-context generation.

    Creates a zero latent and VAE-encodes any provided source/reference
    visuals into ``context_latents`` that are attached to both positive
    and negative conditioning.

    Features:
      - **Chunked encoding**: Long source videos are split into temporal
        chunks to keep peak VRAM bounded.
      - **Batch generation**: ``batch_size > 1`` creates multiple latents
        for parallel video generation in one sampling run.

    Inputs (required):
        positive (CONDITIONING): Positive conditioning.
        negative (CONDITIONING): Negative conditioning.
        vae (VAE): VAE for encoding visual context.
        width (INT): Output width in pixels.
        height (INT): Output height in pixels.
        length (INT): Total number of pixel frames.

    Inputs (optional):
        source_video (IMAGE): Source video for video-to-video.
        reference_video (IMAGE): Reference video for style/motion guidance.
        reference_images (IMAGE): Reference images (autogrow, up to 8 slots).
        ref_max_size (INT): Max long-edge size for reference images/video.
        batch_size (INT): Number of videos to generate in parallel.
        chunk_frames (INT): Max pixel frames per VAE encoding chunk.
            Set lower if encoding long source videos causes OOM.

    Outputs:
        positive (CONDITIONING): Positive conditioning with context_latents.
        negative (CONDITIONING): Negative conditioning with context_latents.
        latent (LATENT): Zero-initialized latent for sampling.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "ref_max_size": ("INT", {"default": 848, "min": 64, "max": 4096, "step": 16}),
                "chunk_frames": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Max pixel frames per VAE encoding chunk. Snapped to nearest multiple of 4 internally."}),
                "chunk_overlap": ("INT", {"default": 4, "min": 4, "max": 32, "step": 4,
                    "tooltip": "Pixel frames of overlap between chunks. Prevents temporal boundary artifacts from VAE 3D convolutions."}),
            },
            "optional": {
                "source_video": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "Mask of the region to edit. White (1) = regenerate, black (0) = keep source. From a segmenter like SAM2; connect directly without inverting. Connecting enables differential diffusion."}),
                "mask_mode": (["freeze", "anneal"], {"default": "anneal", "tooltip": "anneal = soft background anchoring with natural boundaries (default); freeze = pixel-level freeze, background stays completely still"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "prepare"
    CATEGORY = "Bernini-R/Conditioning"
    DESCRIPTION = (
        "Creates latent and attaches in-context visual references for Bernini-R. "
        "Encodes source_video, reference_video, and reference_images to "
        "context_latents with chunked encoding for VRAM efficiency. "
        "Supports batch_size > 1 for parallel generation."
    )

    def prepare(
        self,
        positive,
        negative,
        vae,
        width: int,
        height: int,
        length: int,
        batch_size: int = 1,
        ref_max_size: int = 848,
        chunk_frames: int = 16,
        chunk_overlap: int = 4,
        source_video: Optional[torch.Tensor] = None,
        reference_video: Optional[torch.Tensor] = None,
        reference_images: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mask_mode: str = "anneal",
    ):
        # ── Compute latent dimensions ────────────────────────────────
        latent_channels = 16  # fallback
        try:
            if hasattr(vae, "latent_channels"):
                latent_channels = vae.latent_channels
            else:
                latent_channels = vae.first_stage_model.z_dim  # z_dim (mu), not z_dim*2
        except Exception:
            pass

        t_latent = ((length - 1) // 4) + 1
        h_latent = height // 8
        w_latent = width // 8

        device = mm.intermediate_device()
        latent = torch.zeros(
            [batch_size, latent_channels, t_latent, h_latent, w_latent],
            device=device,
        )

        # ── Build context latents ────────────────────────────────────
        context_latents = []

        # 1) source_video → trim to length, strip to 3 channels, encode with chunking
        if source_video is not None:
            logger.info(
                f"[BerniniR] Encoding source_video: "
                f"shape={list(source_video.shape)}, chunks≤{chunk_frames}"
            )
            source_resized = comfy.utils.common_upscale(
                source_video[:length, :, :, :3].movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)  # [F, H, W, C]
            source_latent = _encode_video_chunked(vae, source_resized, chunk_frames, chunk_overlap)
            context_latents.append(source_latent)

        # 2) reference_video → trim, strip, encode with chunking
        if reference_video is not None:
            logger.info(
                f"[BerniniR] Encoding reference_video: "
                f"shape={list(reference_video.shape)}"
            )
            ref_resized = _resize_long_edge(reference_video[:length, :, :, :3], ref_max_size)
            ref_latent = _encode_video_chunked(vae, ref_resized, chunk_frames, chunk_overlap)
            context_latents.append(ref_latent)

        # 3) reference_images → encode one at a time
        if reference_images is not None:
            n_imgs = reference_images.shape[0]
            logger.info(
                f"[BerniniR] Encoding {n_imgs} reference_images: "
                f"shape={list(reference_images.shape)}"
            )
            for i in range(n_imgs):
                single_img = reference_images[i:i+1]  # [1, H, W, C]
                img_resized = _resize_long_edge(single_img[:, :, :, :3], ref_max_size)
                img_latent = _encode_single(vae, img_resized)
                context_latents.append(img_latent)

        # ── Attach context_latents to conditioning ───────────────────
        if context_latents:
            logger.info(
                f"[BerniniR] Attaching {len(context_latents)} context_latents "
                f"to conditioning (batch_size={batch_size})."
            )
            # Duplicate context_latents across batch dim if batch_size > 1
            if batch_size > 1:
                context_latents = [
                    lat.repeat(batch_size, 1, 1, 1, 1)
                    for lat in context_latents
                ]

            positive = node_helpers.conditioning_set_values(
                positive, {"context_latents": context_latents}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"context_latents": context_latents}
            )

        # ── Attach differential-diffusion mask to conditioning ────────
        # Connecting a mask is treated as enabling differential diffusion; no extra switch needed.
        if mask is not None:
            if source_video is None:
                logger.warning(
                    "[BerniniR] mask connected but no source_video provided; "
                    "edit_mask requires source_video as the background source."
                )
            else:
                edit_mask = _resize_mask_to_latent(
                    mask, t_latent, h_latent, w_latent, device
                )                                              # [B,1,T_lat,H_lat,W_lat]
                # Match mask batch dim to latent batch dim
                if edit_mask.shape[0] == 1 and batch_size > 1:
                    edit_mask = edit_mask.repeat(batch_size, 1, 1, 1, 1)
                dd_values = {
                    "edit_mask": edit_mask,
                    "mask_mode": mask_mode,
                }
                positive = node_helpers.conditioning_set_values(
                    positive, dd_values
                )
                negative = node_helpers.conditioning_set_values(
                    negative, dd_values
                )

        # ── Build output latent dict ─────────────────────────────────
        latent_out = {"samples": latent}

        # ── Release VAE VRAM (encoded latents stay in conditioning) ──
        if hasattr(vae, "offload"):
            try:
                vae.offload()
            except Exception:
                pass
        collect_garbage()
        log_memory("Conditioning done")

        return (positive, negative, latent_out)
