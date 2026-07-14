"""
Model loaders for Bernini-R.

Nodes:
  - BerniniR_ModelLoader:   Create a lazy-loading model handle.
  - BerniniR_CompileModel:  Store torch.compile config on a handle.
  - BerniniR_CLIPLoader:    Load the Wan text encoder.
  - BerniniR_VAELoader:     Load the Bernini-R VAE.
"""

from __future__ import annotations

import logging

import torch
import comfy.model_management as mm
import comfy.sd
import comfy.utils
import folder_paths

from ..utils.model_manager import load_model_handle
from ..utils.vae_wrapper import load_berninir_vae

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BerniniR_ModelLoader
# ---------------------------------------------------------------------------

class BerniniR_ModelLoader:
    """Create a lazy-loading Bernini-R diffusion model handle.

    The model is NOT loaded into RAM by this node.  The actual weights are
    loaded later by the sampler via ``handle.load()``.  This lets Bernini-R
    fully control model lifecycle (e.g. dual-expert split reloads).

    Inputs:
        model_name (COMBO): Model file from diffusion_models/.

    Optional:
        attn_backend_args (BERNINI_ATTN): Attention backend from
            BerniniR_AttentionConfig.  Stored on the handle and applied at
            load time.

    Output:
        BERNINI_MODEL_HANDLE: A lightweight handle.  Pass it to BerniniR_LoadLoRA,
        BerniniR_CompileModel, and the BerniniR samplers.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "Bernini-R model checkpoint (.safetensors)"}),
            },
            "optional": {
                "attn_backend_args": ("BERNINI_ATTN", {"tooltip": "Attention backend config (SageAttn3 → FlashAttn → xformers → SDPA)"}),
            },
        }

    RETURN_TYPES = ("BERNINI_MODEL_HANDLE",)
    RETURN_NAMES = ("model_handle",)
    FUNCTION = "load_model"
    CATEGORY = "Bernini-R/Loaders"
    DESCRIPTION = "Create a lazy-loading Bernini-R model handle. The actual weights are loaded by the sampler when needed."

    def load_model(
        self,
        model_name: str,
        attn_backend_args: dict = None,
    ):
        # ── Resolve model path ────────────────────────────────────────
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        # ── Create lazy handle ────────────────────────────────────────
        handle = load_model_handle(
            model_path=model_path,
            attn_backend_args=attn_backend_args,
        )
        return (handle,)


# ---------------------------------------------------------------------------
# BerniniR_CompileModel
# ---------------------------------------------------------------------------

class BerniniR_CompileModel:
    """Store torch.compile configuration on a Bernini-R model handle.

    The actual compile is applied when the sampler calls ``handle.load()``,
    after all LoRAs have been merged inline.

    Inputs:
        model_handle (BERNINI_MODEL_HANDLE): Handle from BerniniR_ModelLoader.
        compile_mode (COMBO): torch.compile mode.
        fullgraph (BOOLEAN): Require no graph breaks.
        dynamic_shapes (BOOLEAN): Allow dynamic shapes.

    Output:
        BERNINI_MODEL_HANDLE: Handle with compile config stored.
    """

    @classmethod
    def INPUT_TYPES(s):
        from ..models.wan_compile import COMPILE_MODES
        return {
            "required": {
                "model_handle": ("BERNINI_MODEL_HANDLE", {"tooltip": "Bernini-R model handle"}),
                "compile_mode": (COMPILE_MODES, {"default": "none", "tooltip": "torch.compile mode. 'none' = eager, 'default' = trace w/ graph breaks, 'reduce-overhead' → auto-downgraded to 'default' on Windows"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Require zero graph breaks (will likely fail with custom attention ops)"}),
                "dynamic_shapes": ("BOOLEAN", {"default": True, "tooltip": "Allow variable-length sequences in compiled graph"}),
            },
        }

    RETURN_TYPES = ("BERNINI_MODEL_HANDLE",)
    RETURN_NAMES = ("model_handle",)
    FUNCTION = "compile_model"
    CATEGORY = "Bernini-R/Loaders"
    DESCRIPTION = "Store torch.compile config on a Bernini-R model handle. Applied when the sampler loads the model."

    def compile_model(
        self,
        model_handle,
        compile_mode: str = "none",
        fullgraph: bool = False,
        dynamic_shapes: bool = True,
    ):
        if compile_mode == "none":
            return (model_handle,)
        new_handle = model_handle.clone_with_compile(
            mode=compile_mode,
            fullgraph=fullgraph,
            dynamic_shapes=dynamic_shapes,
        )
        return (new_handle,)


# ---------------------------------------------------------------------------
# BerniniR_CLIPLoader
# ---------------------------------------------------------------------------

class BerniniR_CLIPLoader:
    """Load the Wan text encoder for Bernini-R.

    Supports CPU offloading to save VRAM.

    Inputs:
        clip_name (COMBO): Text encoder file from text_encoders/.
        clip_type (COMBO): CLIP type (wan, etc.).
        device (COMBO): Device to load on. "cpu" saves VRAM.

    Output:
        CLIP: The loaded CLIP model.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "Wan T5 text encoder file"}),
                "clip_type": (["wan"], {"default": "wan", "tooltip": "CLIP architecture (Wan uses T5-XXL)"}),
                "device": (["default", "cpu"], {"default": "cpu", "tooltip": "'cpu' offloads text encoder to RAM to save VRAM"}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "Bernini-R/Loaders"
    DESCRIPTION = "Load the Wan text encoder for Bernini-R prompt encoding."

    def load_clip(
        self,
        clip_name: str,
        clip_type: str = "wan",
        device: str = "cpu",
    ):
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.WAN)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = torch.device("cpu")
            model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type_enum,
            model_options=model_options,
        )
        return (clip,)


# ---------------------------------------------------------------------------
# BerniniR_VAELoader
# ---------------------------------------------------------------------------

class BerniniR_VAELoader:
    """Load the Bernini-R VAE.

    Inputs:
        vae_name (COMBO): VAE file from vae/.

    Output:
        VAE: The loaded VAE.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "Wan 16-channel VAE (4x temporal, 8x8 spatial compression)"}),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "Bernini-R/Loaders"
    DESCRIPTION = "Load the Bernini-R VAE for encoding/decoding video latents."

    def load_vae(self, vae_name: str):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        dtype = mm.vae_dtype(
            device=mm.get_torch_device(),
            allowed_dtypes=[torch.bfloat16, torch.float16, torch.float32],
        )
        vae = load_berninir_vae(
            vae_path=vae_path,
            dtype=dtype,
            prefer_wanvideo=False,
        )
        if vae is None:
            raise RuntimeError(
                f"[BerniniR] Failed to load Bernini-R VAE from {vae_path}"
            )
        logger.info(f"[BerniniR] Loaded VAE with dtype={dtype}: {vae_path}")
        return (vae,)
