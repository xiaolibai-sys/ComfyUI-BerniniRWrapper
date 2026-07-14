"""ComfyUI-BerniniR: Custom node package for the Bernini-R 1.3B/14B video model.

Features:
  - Model loading with separate LoRA-aware torch.compile node
  - Temporal context window tiling for long video generation
  - Enhanced KSampler with native context window support
  - Seven-mode guidance family: CFG, APG, RAAG, S2, Z2, STG_A, STG_R
  - Prompt embedding with task presets, system prompts, and CLIP caching
  - In-context conditioning for video editing
  - Five attention backends: SageAttention 3, SageAttention, FlashAttention, xFormers, SDPA
  - TeaCache acceleration and GPU<->CPU block swap

Registered nodes (19):
  BerniniR_ModelLoader        — Load diffusion model
  BerniniR_CompileModel       — Apply torch.compile after LoRA loading
  BerniniR_CLIPLoader         — Load Wan text encoder
  BerniniR_VAELoader          — Load Bernini-R VAE
  BerniniR_Conditioning       — Create latents + attach in-context references
  BerniniR_KSampler           — Enhanced sampler with context window support
  BerniniR_TeaCacheArgs       — TeaCache block-skipping args
  BerniniR_ContextWindow      — Configure temporal context windows
  BerniniR_PromptEmbedding    — Task-aware prompt embedding & CLIP encoding
  BerniniR_VAEDecode          — VAE decode with color matching
  BerniniR_VAEEncode          — VAE encode with tiling
  BerniniR_ApplyNAG           — Normalized Attention Guidance for detail enhancement
  BerniniR_GuidanceStrengthSchedule — Per-step guidance scale curve
  BerniniR_DualExpertSampler  — High/low-noise model switching per step
  BerniniR_BlockSwapArgs      — Configure GPU<->CPU block swapping
  BerniniR_GuidanceConfig     — Select guidance mode & parameters
  BerniniR_SegmentSchedule    — Per-segment schedule
  BerniniR_LoadLoRA           — Append LoRA spec (inline merge)
  BerniniR_AttentionConfig    — Select attention backend
"""

import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

logger = logging.getLogger(__name__)
logger.info("[BerniniR] ComfyUI-BerniniR node package loaded successfully.")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
