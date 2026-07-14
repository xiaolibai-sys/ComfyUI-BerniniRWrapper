"""Attention backend configuration node for Bernini-R."""

from __future__ import annotations

import logging

from ..utils.types import BerniniAttention
from .backends import (
    BACKEND_NAMES,
    available_backends,
    best_available,
)

logger = logging.getLogger(__name__)


class BerniniR_AttentionConfig:
    """Select attention backend with automatic fallback.

    Configures which attention implementation to use during sampling.
    ``"auto"`` selects the best available backend (SageAttention 3 > 2 >
    FlashAttention > xformers > PyTorch SDPA).

    Connect to ``BerniniR_ModelLoader`` via its ``attn_backend_args`` input
    to apply the backend at model load time.

    Inputs:
        backend (COMBO): Attention backend name.
            ``"auto"`` — auto-select best available.
            Others only shown if the library is installed.
        force_backend (BOOLEAN): If True, error on unavailable backend
            instead of silently falling back.

    Output:
        BERNINI_ATTN: Configuration dict consumed by BerniniR_ModelLoader.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (BACKEND_NAMES, {"default": "auto", "tooltip": "Attention implementation. 'auto' = best available. Chain: Sage3 → Sage2/1 → FlashAttn → xformers → SDPA"}),
                "force_backend": ("BOOLEAN", {"default": False, "tooltip": "If True, error on unavailable backend. If False, silently fall back through the chain"}),
            },
        }

    RETURN_TYPES = ("BERNINI_ATTN",)
    RETURN_NAMES = ("attention_config",)
    FUNCTION = "configure"
    CATEGORY = "Bernini-R/Sampling"
    DESCRIPTION = (
        "Attention backend selector. 'auto' = best available. "
        "Chain: SageAttn3 → SageAttn2 → FlashAttn → xformers → SDPA."
    )

    def configure(self, backend: str = "auto", force_backend: bool = False):
        available = available_backends()
        best = best_available()
        config = BerniniAttention(
            backend=backend,
            force_backend=force_backend,
            available=tuple(available),
            best=best,
        )
        logger.info(
            f"[BerniniR] Attention config: backend={backend}, "
            f"force={force_backend}, available={available}, best={best}"
        )
        return (config,)
