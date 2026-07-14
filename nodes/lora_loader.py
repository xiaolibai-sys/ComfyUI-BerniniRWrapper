"""
LoRA loader for Bernini-R / Wan diffusion models.

This node does not merge LoRA weights immediately.  Instead it appends the
LoRA specification to a ``BerniniRModelHandle``; the actual inline merge
happens when the sampler calls ``handle.load()``.  This gives Bernini-R full
control over weight loading and keeps inactive models out of RAM.
"""
from __future__ import annotations

import logging

import folder_paths

logger = logging.getLogger(__name__)


class BerniniR_LoadLoRA:
    """Append a LoRA to a Bernini-R model handle.

    Inputs:
        model_handle (BERNINI_MODEL_HANDLE): Handle from BerniniR_ModelLoader.
        lora_name (COMBO): LoRA file from loras/.
        strength_model (FLOAT): LoRA strength applied to the diffusion model.

    Output:
        model_handle (BERNINI_MODEL_HANDLE): New handle with the LoRA queued.
            The LoRA is merged inline when the model is loaded for sampling.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_handle": ("BERNINI_MODEL_HANDLE", {"tooltip": "Bernini-R model handle"}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file from loras/"}),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "LoRA strength for the diffusion model",
                }),
            },
        }

    RETURN_TYPES = ("BERNINI_MODEL_HANDLE",)
    RETURN_NAMES = ("model_handle",)
    FUNCTION = "load_lora"
    CATEGORY = "Bernini-R/Loaders"
    DESCRIPTION = (
        "Queue a Wan/Bernini-R LoRA on a model handle. "
        "Supports ComfyUI, Diffusers, Kohya, LyCORIS and Fun LoRA key formats."
    )

    def load_lora(self, model_handle, lora_name: str, strength_model: float = 1.0):
        if strength_model == 0.0:
            return (model_handle,)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        logger.info(
            "[BerniniR] Queued LoRA: %s (strength=%.3f)",
            lora_name, strength_model,
        )

        new_handle = model_handle.clone_with_lora(lora_path, strength_model)
        return (new_handle,)
