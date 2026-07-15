"""
Guidance strength schedule — per-step ceiling scale curves.

Provides ``BerniniR_GuidanceStrengthSchedule``: generates a list of per-step
guidance scale values using configurable curves (cosine / linear / piecewise).

The actual guidance combination logic lives in :mod:`.bernini_sampling`.
This node is purely a *schedule generator* — it outputs a ``list[float]``
that the sampler passes to :class:`~.bernini_sampling.BerniniModelWrapper`.
"""

from __future__ import annotations

import math
import logging

from ..utils.types import BerniniGuidance

logger = logging.getLogger(__name__)

CURVE_TYPES = ["cosine", "linear", "piecewise"]


# ---------------------------------------------------------------------------
# Curve functions
# ---------------------------------------------------------------------------

def cosine_schedule(
    steps: int,
    guidance_start: float = 7.0,
    guidance_end: float = 4.0,
    hold_start: float = 0.05,
    hold_end: float = 0.20,
) -> list[float]:
    """Cosine curve with head/tail plateaus."""
    if steps <= 1:
        return [guidance_start]
    schedule = []
    for i in range(steps):
        t = i / (steps - 1)
        if t < hold_start:
            schedule.append(guidance_start)
        elif t > 1.0 - hold_end:
            schedule.append(guidance_end)
        else:
            t2 = (t - hold_start) / (1.0 - hold_start - hold_end)
            val = guidance_end + (guidance_start - guidance_end) * (1.0 + math.cos(t2 * math.pi)) / 2.0
            schedule.append(round(val, 2))
    return schedule


def linear_schedule(
    steps: int,
    guidance_start: float = 7.0,
    guidance_end: float = 4.0,
) -> list[float]:
    """Linear ramp."""
    if steps <= 1:
        return [guidance_start]
    step_size = (guidance_end - guidance_start) / max(steps - 1, 1)
    return [round(guidance_start + step_size * i, 2) for i in range(steps)]


def piecewise_schedule(
    steps: int,
    guidance_start: float = 7.0,
    guidance_mid: float = 5.5,
    guidance_end: float = 4.0,
    transition: float = 0.40,
) -> list[float]:
    """Three flat segments."""
    if steps <= 1:
        return [guidance_start]
    t_mid = 0.70
    schedule = []
    for i in range(steps):
        t = i / (steps - 1)
        if t < transition:
            schedule.append(guidance_start)
        elif t < t_mid:
            schedule.append(guidance_mid)
        else:
            schedule.append(guidance_end)
    return schedule


CURVE_FUNCTIONS = {
    "cosine":    cosine_schedule,
    "linear":    linear_schedule,
    "piecewise": piecewise_schedule,
}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class BerniniR_GuidanceStrengthSchedule:
    """Generate a per-step guidance ceiling schedule.

    Connect the output to the ``guidance_schedule`` input of
    ``BerniniR_KSampler``.  The sampler
    passes this schedule to :class:`~.bernini_sampling.BerniniModelWrapper`
    which uses each value as the per-step *w_max* for the selected guidance
    mode (CFG / APG / RAAG / S2).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 200,
                    "tooltip": "Number of steps for the curve. Auto-resampled to match sampler steps if they differ — you don't need to keep them in sync manually."}),
                "curve": (CURVE_TYPES, {"default": "cosine"}),
                "guidance_start": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Guidance scale at the first denoising step."}),
                "guidance_end": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Guidance scale at the last denoising step."}),
                # ── cosine-specific ──
                "hold_start": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Fraction of steps to hold guidance_start flat (cosine only)."}),
                "hold_end": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Fraction of steps to hold guidance_end flat (cosine only)."}),
                # ── piecewise-specific ──
                "guidance_mid": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Guidance scale in the middle segment (piecewise only)."}),
                "transition": ("FLOAT", {"default": 0.40, "min": 0.05, "max": 0.65, "step": 0.01,
                    "tooltip": "When to switch from guidance_start to guidance_mid (piecewise only)."}),
            },
        }

    RETURN_TYPES = ("BERNINI_GUIDANCE",)
    RETURN_NAMES = ("guidance_schedule",)
    FUNCTION = "build"
    CATEGORY = "Bernini-R/Sampling"
    DESCRIPTION = "Pre-computed per-step guidance ceiling list for dynamic guidance."

    def build(
        self,
        steps: int,
        curve: str,
        guidance_start: float,
        guidance_end: float,
        hold_start: float = 0.05,
        hold_end: float = 0.20,
        guidance_mid: float = 5.5,
        transition: float = 0.40,
    ):
        fn = CURVE_FUNCTIONS[curve]

        if curve == "cosine":
            result = fn(steps, guidance_start, guidance_end, hold_start, hold_end)
        elif curve == "linear":
            result = fn(steps, guidance_start, guidance_end)
        else:
            result = fn(steps, guidance_start, guidance_mid, guidance_end, transition)

        schedule = BerniniGuidance(
            values=result,
            curve=curve,
            steps=steps,
            start=guidance_start,
            end=guidance_end,
        )
        logger.info(
            "[BerniniR] Guidance strength schedule: %s, %d steps, %.1f -> %.1f",
            curve, steps, guidance_start, guidance_end,
        )
        return (schedule,)
