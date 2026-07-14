"""Bernini-R node package — re-exports all node classes."""
from .loaders import BerniniR_ModelLoader, BerniniR_CompileModel, BerniniR_CLIPLoader, BerniniR_VAELoader
from .conditioning import BerniniR_Conditioning
from .sampler import BerniniR_KSampler
from .teacache_args import BerniniR_TeaCacheArgs
from .context_window import BerniniR_ContextWindow
from .prompt_embedding import BerniniR_PromptEmbedding
from .vae import BerniniR_VAEDecode, BerniniR_VAEEncode
from .nag import BerniniR_ApplyNAG
from .guidance_schedule import BerniniR_GuidanceStrengthSchedule
from .sampler_dual_expert import BerniniR_DualExpertSampler
from .block_swap_args import BerniniR_BlockSwapArgs
from .guidance_config import BerniniR_GuidanceConfig
from .segment_schedule import BerniniR_SegmentSchedule
from .lora_loader import BerniniR_LoadLoRA
from ..attention.config_node import BerniniR_AttentionConfig

NODE_CLASS_MAPPINGS = {
    "BerniniR_ModelLoader": BerniniR_ModelLoader,
    "BerniniR_CompileModel": BerniniR_CompileModel,
    "BerniniR_CLIPLoader": BerniniR_CLIPLoader,
    "BerniniR_VAELoader": BerniniR_VAELoader,
    "BerniniR_Conditioning": BerniniR_Conditioning,
    "BerniniR_KSampler": BerniniR_KSampler,
    "BerniniR_TeaCacheArgs": BerniniR_TeaCacheArgs,
    "BerniniR_ContextWindow": BerniniR_ContextWindow,
    "BerniniR_PromptEmbedding": BerniniR_PromptEmbedding,
    "BerniniR_VAEDecode": BerniniR_VAEDecode,
    "BerniniR_VAEEncode": BerniniR_VAEEncode,
    "BerniniR_ApplyNAG": BerniniR_ApplyNAG,
    "BerniniR_GuidanceStrengthSchedule": BerniniR_GuidanceStrengthSchedule,
    "BerniniR_DualExpertSampler": BerniniR_DualExpertSampler,
    "BerniniR_BlockSwapArgs": BerniniR_BlockSwapArgs,
    "BerniniR_GuidanceConfig": BerniniR_GuidanceConfig,
    "BerniniR_SegmentSchedule": BerniniR_SegmentSchedule,
    "BerniniR_LoadLoRA": BerniniR_LoadLoRA,
    "BerniniR_AttentionConfig": BerniniR_AttentionConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BerniniR_ModelLoader": "Bernini-R Model Loader",
    "BerniniR_CompileModel": "Bernini-R Compile Model",
    "BerniniR_CLIPLoader": "Bernini-R CLIP Loader",
    "BerniniR_VAELoader": "Bernini-R VAE Loader",
    "BerniniR_Conditioning": "Bernini-R Conditioning",
    "BerniniR_KSampler": "Bernini-R KSampler",
    "BerniniR_TeaCacheArgs": "Bernini-R TeaCache Args",
    "BerniniR_ContextWindow": "Bernini-R Context Window",
    "BerniniR_PromptEmbedding": "Bernini-R Prompt Embedding",
    "BerniniR_VAEDecode": "Bernini-R VAE Decode",
    "BerniniR_VAEEncode": "Bernini-R VAE Encode",
    "BerniniR_ApplyNAG": "Bernini-R Apply NAG",
    "BerniniR_GuidanceStrengthSchedule": "Bernini-R Guidance Strength Schedule",
    "BerniniR_DualExpertSampler": "Bernini-R Dual Expert Sampler",
    "BerniniR_BlockSwapArgs": "Bernini-R Block Swap Args",
    "BerniniR_GuidanceConfig": "Bernini-R Guidance Config",
    "BerniniR_SegmentSchedule": "Bernini-R Segment Schedule",
    "BerniniR_LoadLoRA": "Bernini-R Load LoRA",
    "BerniniR_AttentionConfig": "Bernini-R Attention Config",
}
