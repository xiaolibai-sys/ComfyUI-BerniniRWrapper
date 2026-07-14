"""
Central parsing utilities for Bernini-R sampling options.

Keeps string→structured conversion in one place so producer and consumer nodes
share the same interpretation of user inputs.
"""

from __future__ import annotations

import logging
from typing import Optional

from .types import GuidanceMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guidance mode
# ---------------------------------------------------------------------------

def parse_guidance_mode(mode: str | GuidanceMode) -> GuidanceMode:
    """Normalise a guidance-mode string into a ``GuidanceMode`` enum.

    Accepts both the UI forms (``"STG_A"``, ``"STG-A"``) and the legacy
    sampler argument forms (``"STG-A"`` / ``"STG-R"``).
    """
    if isinstance(mode, GuidanceMode):
        return mode
    normalised = str(mode).strip().upper().replace("-", "_")
    try:
        return GuidanceMode(normalised)
    except ValueError as exc:
        logger.warning(
            "[BerniniR] Unknown guidance mode %r; falling back to CFG.",
            mode,
        )
        return GuidanceMode.CFG


# ---------------------------------------------------------------------------
# STG block indices
# ---------------------------------------------------------------------------

def parse_stg_block_indices(
    spec: str,
    total_blocks: int,
) -> list[int]:
    """Parse a comma-separated STG block-index spec into absolute indices.

    Supports absolute integers (``"10,20,27"``) and percentages
    (``"33%,67%,90%"``). Percentages are resolved against *total_blocks* and
    clamped to ``[0, total_blocks - 1]``.

    Out-of-range indices are logged and removed.
    """
    spec = (spec or "").strip()
    indices: list[int] = []
    if not spec or total_blocks <= 0:
        return indices

    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        try:
            if tok.endswith("%"):
                pct = int(tok[:-1])
                idx = max(0, min(total_blocks - 1, total_blocks * pct // 100))
            else:
                idx = int(tok)
            indices.append(idx)
        except ValueError:
            logger.warning("[BerniniR] Ignoring invalid STG block token %r.", raw)
            continue

    if indices:
        oob = [i for i in indices if i < 0 or i >= total_blocks]
        if oob:
            logger.warning(
                "[BerniniR] STG block indices %s out of range "
                "(model has %d blocks 0-%d); ignored.",
                oob, total_blocks, total_blocks - 1,
            )
            indices = [i for i in indices if 0 <= i < total_blocks]
    return indices


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def latent_frame_count(pixel_frames: int) -> int:
    """Convert pixel frames to Wan latent frames (4x temporal downscale)."""
    return max(1, (pixel_frames - 1) // 4 + 1)
