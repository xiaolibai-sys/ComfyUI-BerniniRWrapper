"""
Standalone guidance configuration node.

Decouples guidance-mode selection and per-mode hyper-parameters from the
sampler so the sampler UI stays lean.  Follows the same pattern as
``BerniniR_BlockSwapArgs`` / ``BerniniR_TeaCacheArgs``.
"""

from __future__ import annotations

from ..utils.types import BerniniGuidanceConfig, GuidanceMode

GUIDANCE_MODES = ["CFG", "APG", "RAAG", "S2", "Z2", "STG_A", "STG_R"]


class BerniniR_GuidanceConfig:
    """Select guidance strategy and tune its hyper-parameters.

    Connect to the ``guidance_config`` input of ``BerniniR_KSampler``.
    Leave disconnected to use plain CFG with defaults.

    Inputs:
        mode (COMBO): Guidance strategy.
            - ``CFG`` — standard classifier-free guidance (no extra params).
            - ``APG`` — Adaptive Projected Guidance (eta, rescale, momentum).
            - ``RAAG`` — Ratio-Aware Adaptive Guidance (alpha).
            - ``S2`` — Stochastic Self-Guidance (omega).
            - ``Z2`` — Zero-Cost Zigzag Trajectories (collapse).
            - ``STG_A`` / ``STG_R`` — Spatiotemporal Skip Guidance (scale, block_idx).

    Output:
        BERNINI_GUIDANCE_CONFIG: Dataclass consumed by the sampler.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (GUIDANCE_MODES, {
                    "default": "CFG",
                    "tooltip": "Guidance strategy. CFG = standard. Others provide stronger prompt adherence.",
                }),
                # ── APG ──
                "apg_eta": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "APG: parallel-component attenuation. 0.15 = mild, 0.5 = strong.",
                }),
                "apg_rescale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "APG: rescale output norm to target radius (adaptive to CFG).",
                }),
                "apg_momentum": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "APG: reverse momentum across steps. 0 = off.",
                }),
                # ── RAAG ──
                "raag_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "RAAG: decay rate. Higher = stronger early-step damping.",
                }),
                # ── S² ──
                "s2_omega": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "S²: repulsion strength from sub-network prediction.",
                }),
                # ── STG ──
                "stg_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "STG: guidance strength. 1.0 is a safe start.",
                }),
                "stg_block_idx": ("STRING", {
                    "default": "10,20,27", "multiline": False,
                    "tooltip": "STG: comma-separated block indices or percentages for attention skip.",
                }),
                # ── Z² ──
                "z2_collapse": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Z²: trajectory-collapse stabilization. 0 = off, higher = smoother.",
                }),
            },
        }

    RETURN_TYPES = ("BERNINI_GUIDANCE_CONFIG",)
    RETURN_NAMES = ("guidance_config",)
    FUNCTION = "build"
    CATEGORY = "Bernini-R/Config"
    DESCRIPTION = (
        "Guidance strategy selector. Connect to BerniniR_KSampler to override "
        "the default CFG behaviour. Leave disconnected for plain CFG."
    )

    def build(
        self,
        mode: str = "CFG",
        apg_eta: float = 0.15,
        apg_rescale: bool = True,
        apg_momentum: float = 0.0,
        raag_alpha: float = 1.0,
        s2_omega: float = 1.0,
        stg_scale: float = 1.0,
        stg_block_idx: str = "10,20,27",
        z2_collapse: float = 0.3,
    ):
        return (BerniniGuidanceConfig(
            mode=GuidanceMode(mode),
            apg_eta=apg_eta,
            apg_rescale=apg_rescale,
            apg_momentum=apg_momentum,
            raag_alpha=raag_alpha,
            s2_omega=s2_omega,
            stg_scale=stg_scale,
            stg_block_idx=stg_block_idx,
            z2_collapse=z2_collapse,
        ),)
