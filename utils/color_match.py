"""
Pure-PyTorch colour transfer algorithms for VAE decode post-processing.

All methods operate on ``(F, H, W, C)`` tensors in [0, 1] range.
Zero external dependencies beyond ``torch``.  GPU-native, fully
differentiable (though we only use inference).

Algorithms
----------
- ``reinhard``       Channel-wise mean / std alignment (RGB or CIELAB).
- ``hm``             Per-channel histogram matching.
- ``mvgd``           Multi-Variate Gaussian Distribution transfer.
- ``mkl``            Monge-Kantorovich Linearisation.
- ``hm-mvgd-hm``     HM → MVGD → HM  (perceptually most pleasing).
- ``hm-mkl-hm``      HM → MKL  → HM.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

COLORMATCH_METHODS = [
    "disabled",
    "mkl",
    "hm",
    "reinhard",
    "mvgd",
    "hm-mvgd-hm",
    "hm-mkl-hm",
]

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_color_match(
    pixels: torch.Tensor,
    ref_image: Optional[torch.Tensor],
    method: str = "disabled",
    blend_strength: float = 1.0,
    *,
    frames_per_batch: Optional[int] = None,
    max_stat_pixels: int = 500_000,
) -> torch.Tensor:
    """Apply colour matching to decoded video frames.

    Args:
        pixels:  Video frames ``(F, H, W, C)`` in [0, 1].
        ref_image:  Reference ``(1, H, W, C)`` or ``(H, W, C)`` or ``None``
            (``None`` → first frame used as reference).
        method:  One of ``COLORMATCH_METHODS``.
        blend_strength:  0 = no change, 1 = full transfer.
        frames_per_batch:  If set and ``ref_image is None``, split into
            batches that each use their own first frame as reference.
        max_stat_pixels:  Cap on pixels fed to covariance-based methods
            (MVGD / MKL) to keep large-video statistics cheap.

    Returns:
        Colour-matched pixels, same shape / dtype / device as *pixels*.
    """
    if method == "disabled" or blend_strength <= 0.0:
        return pixels

    total_frames = pixels.shape[0]

    # ── Per-batch auto-reference ─────────────────────────────────────
    if frames_per_batch is not None and ref_image is None:
        B = total_frames // frames_per_batch
        if B * frames_per_batch != total_frames:
            logger.warning(
                "[BerniniR] frames_per_batch=%d doesn't divide %d frames; "
                "falling back to single-ref.",
                frames_per_batch, total_frames,
            )
            ref_image = pixels[:1]
        else:
            matched = []
            for b in range(B):
                lo = b * frames_per_batch
                hi = lo + frames_per_batch
                batch = pixels[lo:hi]
                matched.append(
                    _apply_single_ref(batch, batch[:1], method,
                                      blend_strength, max_stat_pixels)
                )
            return torch.cat(matched, dim=0)

    # ── Single-reference ─────────────────────────────────────────────
    if ref_image is None:
        ref_image = pixels[:1]
    if ref_image.dim() == 3:
        ref_image = ref_image.unsqueeze(0)

    return _apply_single_ref(pixels, ref_image, method,
                             blend_strength, max_stat_pixels)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _apply_single_ref(
    pixels: torch.Tensor,       # (F, H, W, C)
    ref_image: torch.Tensor,    # (1, H, W, C)
    method: str,
    blend_strength: float,
    max_stat_pixels: int,
) -> torch.Tensor:
    """Route *method* to the appropriate torch-native algorithm."""
    if method == "reinhard":
        return _reinhard(pixels, ref_image, blend_strength)
    if method == "hm":
        return _histogram_match(pixels, ref_image, blend_strength)
    if method == "mvgd":
        return _mvgd(pixels, ref_image, blend_strength, max_stat_pixels)
    if method == "mkl":
        return _mkl(pixels, ref_image, blend_strength, max_stat_pixels)
    if method == "hm-mvgd-hm":
        return _pipeline_hm_mvgd_hm(pixels, ref_image, blend_strength, max_stat_pixels)
    if method == "hm-mkl-hm":
        return _pipeline_hm_mkl_hm(pixels, ref_image, blend_strength, max_stat_pixels)
    # Unknown method → reinhard fallback
    logger.warning("[BerniniR] Unknown method '%s' — using reinhard.", method)
    return _reinhard(pixels, ref_image, blend_strength)


# ===================================================================
# Algorithm implementations
# ===================================================================

# -------------------------------------------------------------------
# 1. Histogram Matching
# -------------------------------------------------------------------

def _histogram_match(
    pixels: torch.Tensor,       # (F, H, W, C)
    ref_image: torch.Tensor,    # (1, H, W, C)
    blend_strength: float,
) -> torch.Tensor:
    """Per-channel histogram matching via CDF interpolation.

    For each colour channel independently, map source intensities so
    that the output CDF matches the reference CDF.  Uses linear
    interpolation of the sorted pixel arrays — equivalent to the
    ``color_matcher`` "hm" method.
    """
    F, H, W, C = pixels.shape
    result = pixels.clone()

    for c in range(C):
        src_flat = pixels[..., c].reshape(-1)
        ref_flat = ref_image[..., c].reshape(-1)

        # Sort both channels (work in float32 for interpolation accuracy,
        # then cast back to the source dtype at the end).
        src_flat_f32 = src_flat.float()
        ref_flat_f32 = ref_flat.float()

        src_sorted, _ = src_flat_f32.sort()
        ref_sorted, _ = ref_flat_f32.sort()

        n_src = src_sorted.numel()
        n_ref = ref_sorted.numel()

        src_indices = torch.linspace(0, n_ref - 1, n_src, device=pixels.device)
        lo = src_indices.floor().long().clamp(0, n_ref - 1)
        hi = (lo + 1).clamp(0, n_ref - 1)
        alpha = (src_indices - lo.float()).clamp(0, 1)

        mapped_ref_f32 = ref_sorted[lo] + alpha * (ref_sorted[hi] - ref_sorted[lo])

        src_argsort = src_flat_f32.argsort()
        mapped = torch.empty_like(src_flat)
        mapped[src_argsort] = mapped_ref_f32.to(mapped.dtype)

        result[..., c] = mapped.reshape(F, H, W)

    if blend_strength < 1.0:
        result = torch.lerp(pixels.float(), result.float(), blend_strength)
    return result.clamp(0.0, 1.0).to(dtype=pixels.dtype)


# -------------------------------------------------------------------
# 2. Reinhard (mean / std) — CIELAB preferred, RGB fallback
# -------------------------------------------------------------------

def _reinhard(
    pixels: torch.Tensor,       # (F, H, W, C)
    ref_image: torch.Tensor,    # (1, H, W, C)
    blend_strength: float,
) -> torch.Tensor:
    """Perceptual Reinhard transfer (CIELAB via kornia, or RGB fallback)."""
    try:
        return _reinhard_lab(pixels, ref_image, blend_strength)
    except ImportError:
        logger.debug("[BerniniR] kornia not installed; using RGB reinhard.")
        return _reinhard_rgb(pixels, ref_image, blend_strength)


def _reinhard_lab(pixels, ref_image, blend_strength):
    import kornia
    # kornia requires float32
    pixels_f32 = pixels.float()
    ref_f32 = ref_image.float()
    pixels_lab = kornia.color.rgb_to_lab(pixels_f32.permute(0, 3, 1, 2))
    ref_lab = kornia.color.rgb_to_lab(ref_f32.permute(0, 3, 1, 2))
    src_mean = pixels_lab.mean(dim=(2, 3), keepdim=True)
    src_std = pixels_lab.std(dim=(2, 3), keepdim=True) + 1e-8
    ref_mean = ref_lab.mean(dim=(2, 3), keepdim=True)
    ref_std = ref_lab.std(dim=(2, 3), keepdim=True) + 1e-8
    pixels_lab = (pixels_lab - src_mean) / src_std * ref_std + ref_mean
    result = kornia.color.lab_to_rgb(pixels_lab).permute(0, 2, 3, 1)
    if blend_strength < 1.0:
        result = torch.lerp(pixels_f32, result, blend_strength)
    return result.clamp(0.0, 1.0).to(dtype=pixels.dtype)


def _reinhard_rgb(pixels, ref_image, blend_strength):
    F, H, W, C = pixels.shape
    Hr, Wr = ref_image.shape[1], ref_image.shape[2]
    src_flat = pixels.reshape(F, H * W, C)
    ref_flat = ref_image.reshape(1, Hr * Wr, C)
    src_mean = src_flat.mean(dim=1, keepdim=True)
    src_std = src_flat.std(dim=1, keepdim=True) + 1e-8
    ref_mean = ref_flat.mean(dim=1, keepdim=True)
    ref_std = ref_flat.std(dim=1, keepdim=True) + 1e-8
    result_flat = (src_flat - src_mean) / src_std * ref_std + ref_mean
    result = result_flat.reshape(F, H, W, C)
    if blend_strength < 1.0:
        result = torch.lerp(pixels.float(), result.float(), blend_strength)
    return result.clamp(0.0, 1.0).to(dtype=pixels.dtype)


# -------------------------------------------------------------------
# 3. MVGD — Multi-Variate Gaussian Distribution
# -------------------------------------------------------------------

def _mvgd(
    pixels: torch.Tensor,       # (F, H, W, C)
    ref_image: torch.Tensor,    # (1, H, W, C)
    blend_strength: float,
    max_pixels: int,
) -> torch.Tensor:
    r"""MVGD colour transfer.

    Transforms the source RGB distribution so that its mean and
    covariance match the reference::

        X' = T @ (X - μ_s) + μ_r

    where ``T = Σ_r^{1/2} @ Σ_s^{-1/2}``.
    """
    F, H, W, C = pixels.shape
    src_flat = pixels.reshape(-1, C).float()
    ref_flat = ref_image.reshape(-1, C).float()

    # ── Subsample for covariance if needed ──────────────────────────
    n_src = src_flat.shape[0]
    n_ref = ref_flat.shape[0]
    src_for_stats = _subsample(src_flat, max_pixels)
    ref_for_stats = _subsample(ref_flat, max_pixels)

    src_mean = src_for_stats.mean(dim=0)
    ref_mean = ref_for_stats.mean(dim=0)

    T = _mvgd_transform(src_for_stats - src_mean, ref_for_stats - ref_mean)

    # Apply to all pixels
    result = ((src_flat - src_mean) @ T.T) + ref_mean
    result = result.reshape(F, H, W, C)

    if blend_strength < 1.0:
        result = torch.lerp(pixels.float(), result, blend_strength)
    return result.clamp(0.0, 1.0).to(dtype=pixels.dtype)


def _mvgd_transform(
    src_centered: torch.Tensor,  # (N_s, C)
    ref_centered: torch.Tensor,  # (N_r, C)
) -> torch.Tensor:               # (C, C)
    """Compute MVGD transformation matrix ``T``."""
    n_s = src_centered.shape[0]
    n_r = ref_centered.shape[0]
    src_cov = (src_centered.T @ src_centered) / max(n_s - 1, 1)
    ref_cov = (ref_centered.T @ ref_centered) / max(n_r - 1, 1)

    src_cov_sqrt, src_cov_inv_sqrt = _cov_sqrt_pair(src_cov)
    ref_cov_sqrt, _ = _cov_sqrt_pair(ref_cov)

    return ref_cov_sqrt @ src_cov_inv_sqrt


# -------------------------------------------------------------------
# 4. MKL — Monge-Kantorovich Linearisation
# -------------------------------------------------------------------

def _mkl(
    pixels: torch.Tensor,       # (F, H, W, C)
    ref_image: torch.Tensor,    # (1, H, W, C)
    blend_strength: float,
    max_pixels: int,
) -> torch.Tensor:
    r"""MKL colour transfer.

    Uses the optimal-transport (Wasserstein-2) map for Gaussians::

        T = Σ_s^{-1/2} @ (Σ_s^{1/2} @ Σ_r @ Σ_s^{1/2})^{1/2} @ Σ_s^{-1/2}
    """
    F, H, W, C = pixels.shape
    src_flat = pixels.reshape(-1, C).float()
    ref_flat = ref_image.reshape(-1, C).float()

    src_for_stats = _subsample(src_flat, max_pixels)
    ref_for_stats = _subsample(ref_flat, max_pixels)

    src_mean = src_for_stats.mean(dim=0)
    ref_mean = ref_for_stats.mean(dim=0)

    T = _mkl_transform(src_for_stats - src_mean, ref_for_stats - ref_mean)

    result = ((src_flat - src_mean) @ T.T) + ref_mean
    result = result.reshape(F, H, W, C)

    if blend_strength < 1.0:
        result = torch.lerp(pixels.float(), result, blend_strength)
    return result.clamp(0.0, 1.0).to(dtype=pixels.dtype)


def _mkl_transform(
    src_centered: torch.Tensor,  # (N_s, C)
    ref_centered: torch.Tensor,  # (N_r, C)
) -> torch.Tensor:               # (C, C)
    """Compute MKL transformation matrix ``T``."""
    n_s = src_centered.shape[0]
    n_r = ref_centered.shape[0]
    src_cov = (src_centered.T @ src_centered) / max(n_s - 1, 1)
    ref_cov = (ref_centered.T @ ref_centered) / max(n_r - 1, 1)

    src_cov_sqrt, src_cov_inv_sqrt = _cov_sqrt_pair(src_cov)
    sandwich = src_cov_sqrt @ ref_cov @ src_cov_sqrt
    sandwich_sqrt, _ = _cov_sqrt_pair(sandwich)

    return src_cov_inv_sqrt @ sandwich_sqrt @ src_cov_inv_sqrt


# -------------------------------------------------------------------
# 5. Pipelines
# -------------------------------------------------------------------

def _pipeline_hm_mvgd_hm(
    pixels: torch.Tensor,
    ref_image: torch.Tensor,
    blend_strength: float,
    max_pixels: int,
) -> torch.Tensor:
    """HM → MVGD → HM pipeline."""
    out = _histogram_match(pixels, ref_image, 1.0)
    out = _mvgd(out, ref_image, 1.0, max_pixels)
    out = _histogram_match(out, ref_image, blend_strength)
    return out


def _pipeline_hm_mkl_hm(
    pixels: torch.Tensor,
    ref_image: torch.Tensor,
    blend_strength: float,
    max_pixels: int,
) -> torch.Tensor:
    """HM → MKL → HM pipeline."""
    out = _histogram_match(pixels, ref_image, 1.0)
    out = _mkl(out, ref_image, 1.0, max_pixels)
    out = _histogram_match(out, ref_image, blend_strength)
    return out


# ===================================================================
# Internal helpers
# ===================================================================

def _subsample(x: torch.Tensor, max_pixels: int) -> torch.Tensor:
    """Return *x* as-is or a uniform random subset capped at *max_pixels* rows."""
    n = x.shape[0]
    if n <= max_pixels:
        return x
    idx = torch.randperm(n, device=x.device)[:max_pixels]
    return x[idx]


def _cov_sqrt_pair(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return ``(Σ^{1/2}, Σ^{-1/2})`` for a symmetric PSD matrix via SVD.

    Uses the eigendecomposition path for 3×3 matrices because it is
    faster and avoids the full-SVD overhead.
    """
    # For 3×3 we can use torch.linalg.eigh (specialised for symmetric)
    S, U = torch.linalg.eigh(cov)          # S = eigenvalues, U = eigenvectors
    S = S.clamp(min=1e-8)                  # guard against negatives

    sqrt_S = torch.sqrt(S)
    inv_sqrt_S = 1.0 / sqrt_S

    # Σ^{1/2} = U @ diag(sqrt_S) @ U^T
    cov_sqrt = U @ (sqrt_S.unsqueeze(1) * U.T)
    cov_inv_sqrt = U @ (inv_sqrt_S.unsqueeze(1) * U.T)

    return cov_sqrt, cov_inv_sqrt
