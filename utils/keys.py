"""Canonical safetensors/state-dict key normalisation.

Single authoritative implementation shared by the checkpoint loader
(``models.loader``), the random-access block readers (``utils.block_reader``)
and the LoRA folding helpers (``utils.lora``), so prefix handling can never
drift between the read path and the fold path.
"""

from __future__ import annotations


def _normalize_unet_key(k: str) -> str:
    """Strip common checkpoint prefixes so state-dict keys line up with the
    diffusion_model's own keys (``model.diffusion_model.blocks.0...`` →
    ``blocks.0...``).
    """
    if k.startswith("model.diffusion_model."):
        return k[len("model.diffusion_model."):]
    if k.startswith("model."):
        return k[len("model."):]
    if k.startswith("video_model."):
        return k[len("video_model."):].replace("modulation.modulation", "modulation")
    if k.startswith("diffusion_model."):
        return k[len("diffusion_model."):]
    return k
