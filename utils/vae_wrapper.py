"""
WanVideo VAE integration helper for Bernini-R.

Provides ``BerniniRVAE`` (a ComfyUI-compatible VAE wrapper) and the
``load_berninir_vae()`` factory used by the VAE loader node.

Imports the VAE model classes from the bundled ``..utils.wanvideo_vae`` module,
eliminating the external dependency on ComfyUI-WanVideoWrapper.
"""
from __future__ import annotations


import torch
import comfy.model_management as mm
import comfy.utils

from .wanvideo_vae import WanVideoVAE, WanVideoVAE38

from .log import get_logger as _get_logger

logger = _get_logger("VAE")
class BerniniRVAE:
    """ComfyUI-compatible VAE wrapper around ``WanVideoVAE``.

    Adapts the native ``WanVideoVAE.encode/decode`` (list of 5D tensors,
    device parameter, pixel-space tile sizes) to the simpler interface
    expected by the ``BerniniR_VAEEncode`` / ``BerniniR_VAEDecode`` nodes:

    * ``encode(pixels)`` takes a single 4D video tensor ``(F, H, W, C)`` in
      ``[0, 1]`` and returns a raw latent ``(1, C_lat, T, H_lat, W_lat)``.
    * ``decode(samples)`` takes a raw latent ``(B, C_lat, T, H_lat, W_lat)``
      and returns pixels ``(F, H, W, C)`` in ``[0, 1]``.

    ..  important::

        WanVideoVAE internally applies mean/std normalisation to latents
        during encode/decode.  ComfyUI's model wrapper (``process_latent_in`` /
        ``process_latent_out``) also applies the **same** mean/std.  To avoid
        double-normalisation this wrapper **undoes** the VAE's internal
        normalisation in ``encode()`` so that latents are "raw" (matching
        ComfyUI's native VAE convention).

        Decode receives **raw** latents (after ComfyUI's
        ``process_latent_out``) and converts them to the VAE's internal
        normalised space before calling the inner decoder.
    """

    def __init__(self, model: WanVideoVAE, dtype: torch.dtype):
        self.model = model
        self.dtype = dtype
        self._device: torch.device | None = None

        # Expose standard ComfyUI VAE attributes so that conditioning nodes
        # and samplers can discover the correct latent shape / spatial scale.
        self.latent_channels: int = model.z_dim
        self.downscale_ratio = model.upsampling_factor
        self.upscale_ratio = model.upsampling_factor
        self.latent_dim: int = 3
        self.latent_rgb_factors = None

    def offload(self) -> None:
        """Move the inner VAE model to CPU to free GPU memory.

        Matches the ComfyUI native VAE ``offload()`` interface so that
        ``_unload_vae()`` in conditioning.py can find it.
        """
        offload_device = mm.unet_offload_device()
        if self._device != offload_device:
            self.model = self.model.to(offload_device)
            self._device = offload_device
            mm.soft_empty_cache()

    def _ensure_device(self) -> torch.device:
        """Lazily move the inner model to GPU on first use."""
        device = mm.get_torch_device()
        if self._device != device:
            self.model = self.model.to(device)
            self._device = device
        return device

    # ------------------------------------------------------------------
    # Helpers: convert between "raw" (ComfyUI) and "normalised" (Wan VAE)
    # ------------------------------------------------------------------
    def _latent_to_raw(self, normalised: torch.Tensor) -> torch.Tensor:
        """Undo the VAE's internal mean/std so the latent is "raw".

        VAE internal:  ``norm = (raw - mean) * inv_std``
        Inverse:       ``raw = norm / inv_std + mean``

        Preserves the input rank: a 4D input returns 4D, a 5D input
        (including the leading batch dim) returns 5D.
        """
        input_dim = normalised.dim()
        mean = self.model.mean.to(normalised.device, normalised.dtype)
        inv_std = self.model.inv_std.to(normalised.device, normalised.dtype)
        raw = normalised / inv_std + mean
        # mean/inv_std are 5D (1, C, 1, 1, 1); broadcasting against a 4D
        # (C, T, H, W) input produces 5D (1, C, T, H, W).  Squeeze back
        # only when the input itself was 4D.
        if input_dim == 4 and raw.dim() == 5 and raw.shape[0] == 1:
            raw = raw.squeeze(0)
        return raw

    def _latent_to_normalised(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the VAE's internal mean/std (reverse of ``_latent_to_raw``).

        Preserves the input rank: a 4D input returns 4D, a 5D input
        (including the leading batch dim) returns 5D.
        """
        input_dim = raw.dim()
        mean = self.model.mean.to(raw.device, raw.dtype)
        inv_std = self.model.inv_std.to(raw.device, raw.dtype)
        normalised = (raw - mean) * inv_std
        if input_dim == 4 and normalised.dim() == 5 and normalised.shape[0] == 1:
            normalised = normalised.squeeze(0)
        return normalised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(
        self,
        pixels: torch.Tensor,
        tiled: bool = False,
        tile_x: int = 272,
        tile_y: int = 272,
        tile_stride_x: int = 144,
        tile_stride_y: int = 128,
    ) -> torch.Tensor:
        """Encode pixels ``(F, H, W, C)`` in ``[0, 1]`` to raw latent.

        The returned latent is *without* mean/std normalisation, matching
        ComfyUI's native VAE convention.  ComfyUI's model wrapper will apply
        ``process_latent_in`` later.
        """
        device = self._ensure_device()
        # (F, H, W, C) → (C, F, H, W)  in [-1, 1]; WanVideoVAE adds batch dim internally
        video = pixels.permute(3, 0, 1, 2).to(device=device, dtype=self.dtype)
        video = video.mul_(2.0).sub_(1.0)  # in-place scale+shift

        sf = self.model.upsampling_factor
        if tiled:
            tile_size = (max(1, tile_y // sf), max(1, tile_x // sf))
            tile_stride = (max(1, tile_stride_y // sf), max(1, tile_stride_x // sf))
        else:
            tile_size = None
            tile_stride = None

        normalised = self.model.encode(
            [video], device=device, tiled=tiled,
            tile_size=tile_size, tile_stride=tile_stride,
            pbar=True,
        )  # (1, C, T', H', W')  -- VAE-internally normalised
        return self._latent_to_raw(normalised)  # (1, C, T', H', W') raw latent

    def _decode_single(
        self,
        samples: torch.Tensor,
        device: torch.device,
        tiled: bool,
        tile_x: int,
        tile_y: int,
        tile_stride_x: int,
        tile_stride_y: int,
    ) -> torch.Tensor:
        """Decode a single video latent already in normalised space."""
        sf = self.model.upsampling_factor

        # ComfyUI latents are (B, C, T, H, W); WanVideoVAE expects (C, T, H, W)
        # Also convert to the model's dtype to avoid Half vs BFloat16 mismatch.
        if samples.dim() == 5 and samples.shape[0] == 1:
            hs = samples[0].to(dtype=self.dtype)  # (1, C, T, H, W) → (C, T, H, W)
        else:
            hs = samples.to(dtype=self.dtype)  # assume already (C, T, H, W)

        if tiled:
            # WanVideoVAE tile_size is in latent-space coordinates
            tile_size = (max(1, tile_y // sf), max(1, tile_x // sf))
            tile_stride = (max(1, tile_stride_y // sf), max(1, tile_stride_x // sf))
        else:
            tile_size = None
            tile_stride = None

        results = self.model.decode(
            [hs], device=device, tiled=tiled,
            tile_size=tile_size, tile_stride=tile_stride,
            pbar=True,
        )
        video = results[0]  # (C, F, H, W)
        # (C, F, H, W) → (F, H, W, C) in [0, 1] — in-place to save allocs
        pixels = video.permute(1, 2, 3, 0)    # view, no copy
        pixels = pixels.mul_(0.5).add_(0.5).clamp_(0, 1).float()
        return pixels

    def decode(
        self,
        samples: torch.Tensor,
        tiled: bool = False,
        tile_x: int = 272,
        tile_y: int = 272,
        tile_stride_x: int = 144,
        tile_stride_y: int = 128,
    ) -> torch.Tensor:
        """Decode raw latent to pixels ``(F, H, W, C)`` in ``[0, 1]``.

        The input latent follows ComfyUI's VAE convention (raw, mean/std
        *not* applied).  It is converted to the VAE's internal normalised
        space before being passed to the underlying ``WanVideoVAE`` decoder.
        """
        device = self._ensure_device()

        # Convert from ComfyUI raw latent space to VAE-internal normalised space.
        samples = self._latent_to_normalised(samples)

        # Multi-batch: decode each batch item independently and concatenate.
        if samples.dim() == 5 and samples.shape[0] > 1:
            frames = []
            for i in range(samples.shape[0]):
                frames.append(
                    self._decode_single(
                        samples[i:i + 1], device=device, tiled=tiled,
                        tile_x=tile_x, tile_y=tile_y,
                        tile_stride_x=tile_stride_x,
                        tile_stride_y=tile_stride_y,
                    )
                )
            return torch.cat(frames, dim=0)

        return self._decode_single(
            samples, device=device, tiled=tiled,
            tile_x=tile_x, tile_y=tile_y,
            tile_stride_x=tile_stride_x,
            tile_stride_y=tile_stride_y,
        )

    def to(self, dtype: torch.dtype) -> BerniniRVAE:
        self.model = self.model.to(dtype)
        self.dtype = dtype
        return self


def load_berninir_vae(
    vae_path: str,
    dtype: torch.dtype,
    prefer_wanvideo: bool = True,
) -> BerniniRVAE | None:
    """Load a Wan VAE wrapped in a ComfyUI-compatible ``BerniniRVAE``.

    Args:
        vae_path: Absolute path to the VAE ``.safetensors`` file.
        dtype: Target weight dtype (e.g. ``torch.bfloat16``).
        prefer_wanvideo: If ``True``, try the bundled WanVideoVAE first;
            fall back to ComfyUI's native VAE on failure.

    Returns:
        A ``BerniniRVAE`` instance, or ``None`` if loading fails and no
        fallback is available.
    """
    from comfy.sd import VAE as ComfyVAE

    sd = comfy.utils.load_torch_file(vae_path, safe_load=True)
    if not sd:
        logger.warning(f"[VAE] Empty state dict: {vae_path}")
        if not prefer_wanvideo:
            return None
        # Fall back to ComfyUI VAE
        logger.info("[VAE] Falling back to ComfyUI native VAE.")
        return ComfyVAE(sd=sd)

    z_dim = _detect_z_dim(sd)
    logger.info(f"[VAE] Detected VAE z_dim={z_dim} (16=Wan2.1, 48=WanI38B)")

    # WanVideoVAE wraps VideoVAE_ as self.model, so the state dict keys
    # need a "model." prefix.  Safetensors files distributed for Wan
    # may or may not include this prefix; add it if missing (matching
    # WanVideoWrapper's WanVideoVAELoader).
    if not any(k.startswith("model.") for k in sd):
        sd = {f"model.{k}": v for k, v in sd.items()}

    try:
        if z_dim == 48:
            model = WanVideoVAE38(z_dim=z_dim, dtype=dtype)
        else:
            model = WanVideoVAE(z_dim=z_dim, dtype=dtype)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(f"[VAE] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"[VAE] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        model = model.eval().requires_grad_(False).to(dtype)
        return BerniniRVAE(model, dtype)

    except Exception as e:
        logger.warning(f"[VAE] Failed to load WanVideoVAE: {e}")
        if not prefer_wanvideo:
            return None
        logger.info("[VAE] Falling back to ComfyUI native VAE.")
        return ComfyVAE(sd=sd)


def load_wanvideo_vae(vae_path: str, dtype: torch.dtype) -> WanVideoVAE | None:
    """Load a raw ``WanVideoVAE`` without the BerniniRVAE wrapper.

    Used internally or for direct access to the native VAE.
    """
    sd = comfy.utils.load_torch_file(vae_path, safe_load=True)
    z_dim = _detect_z_dim(sd)
    if not any(k.startswith("model.") for k in sd):
        sd = {f"model.{k}": v for k, v in sd.items()}
    try:
        if z_dim == 48:
            vae = WanVideoVAE38(z_dim=z_dim, dtype=dtype)
        else:
            vae = WanVideoVAE(z_dim=z_dim, dtype=dtype)
        vae.load_state_dict(sd, strict=False)
        return vae.eval().requires_grad_(False).to(dtype)
    except Exception as e:
        logger.warning(f"[VAE] Failed to load WanVideoVAE (raw): {e}")
        return None


def _detect_z_dim(sd: dict) -> int:
    """Detect VAE latent dimension from state dict keys.

    Wan 2.2 / WanI38B uses z_dim=48 (encoder head outputs 48 channels).
    Wan 2.1 uses z_dim=16 (encoder head outputs 16 channels).
    """
    # 1) Encoder head — most reliable: CausalConv3d(z_dim*2, z_dim, ...)
    #    Output channels = z_dim directly.
    for key in sd:
        if "head" in key and "weight" in key and sd[key].dim() >= 2:
            out_ch = sd[key].shape[0]
            if out_ch in (16, 48):
                return out_ch

    # 2) Middle conv1: CausalConv3d(z_dim*2, z_dim*2, 1)
    #    Output channels = z_dim * 2.  Exclude encoder/decoder conv1
    #    which have different channel counts.
    for key in sd:
        if "conv1.weight" in key and "encoder" not in key and "decoder" not in key and sd[key].dim() >= 2:
            out_ch = sd[key].shape[0]  # z_dim * 2
            z = out_ch // 2
            if z in (16, 48):
                return z

    # 3) Fallback: decoder conv1 input channels = z_dim
    for key in sd:
        if "decoder" in key and "conv1.weight" in key:
            in_ch = sd[key].shape[1]
            if in_ch in (16, 48):
                return in_ch

    return 16  # default to Wan2.1
