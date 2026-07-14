"""
ComfyUI-BerniniR: Custom node package for the Bernini-R 1.3B video model.

Features:
  - Model loading with separate LoRA-aware torch.compile node
  - Temporal context window tiling for long video generation (4 schedules)
  - Enhanced KSampler with native context window support
  - Six-mode guidance family: CFG, APG, RAAG, S2, STG_A, STG_R
  - Prompt planner with 12 task presets, system prompts, and CLIP caching
  - In-context conditioning for video editing
  - Five attention backends: SageAttention 3, SageAttention, FlashAttention, xFormers, SDPA
  - TeaCache acceleration and GPU<->CPU block swap

Registered nodes (17):
  BerniniR_ModelLoader        — Load diffusion model
  BerniniR_CompileModel       — Apply torch.compile after LoRA loading
  BerniniR_CLIPLoader         — Load Wan text encoder
  BerniniR_VAELoader          — Load Bernini-R VAE
  BerniniR_ContextWindow      — Configure temporal context windows
  BerniniR_KSampler           — Enhanced sampler with context window support
  BerniniR_KSamplerTeaCache   — Sampler with TeaCache block skipping
  BerniniR_DualExpertSampler  — High/low-noise model switching per step
  BerniniR_Conditioning       — Create latents + attach in-context references
  BerniniR_PromptPlanner      — Task-aware prompt planning & CLIP encoding
  BerniniR_ApplyNAG           — Normalized Attention Guidance for detail enhancement
  BerniniR_GuidanceStrengthSchedule — Per-step guidance scale curve
  BerniniR_AttentionConfig    — Select attention backend
  BerniniR_VAEDecode          — VAE decode with color matching
  BerniniR_VAEEncode          — VAE encode with tiling
  BerniniR_LoadLoRA           — Append LoRA spec (inline merge)
  BerniniR_BlockSwapArgs      — Configure GPU<->CPU block swapping
"""

import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

logger = logging.getLogger(__name__)
logger.info("[BerniniR] ComfyUI-BerniniR node package loaded successfully.")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
