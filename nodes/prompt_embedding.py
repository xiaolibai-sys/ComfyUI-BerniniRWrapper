"""
Prompt planner for Bernini-R.

Unified prompt planning + text encoding node:
  - Selects a task-specific system prompt from 12 predefined options.
  - Concatenates with user prompt and encodes via CLIP.
  - Caches embeddings to disk for repeated prompts.
  - Optionally unloads CLIP after encoding to free VRAM.

Portable version of the Bernini-R Prompt Planner, usable as a standalone node.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import comfy.model_management as mm
import folder_paths

from ..utils.text_cache import (
    _load_cached_conditioning,
    _save_cached_conditioning,
    _encode_text,
    _load_clip_internal,
)
from ..utils.vram import collect_garbage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts — task-specific instruction prefixes
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a helpful assistant specialized in text-to-image generation.",
    "You are a helpful assistant specialized in text-to-video generation.",
    "You are a helpful assistant specialized in image editing.",
    "You are a helpful assistant specialized in subject-to-image generation.",
    "You are a helpful assistant specialized in image-to-video generation.",
    "You are a helpful assistant specialized in video editing.",
    "You are a helpful assistant specialized in video editing on content propagation.",
    "You are a helpful assistant specialized in video editing with reference.",
    "You are a helpful assistant specialized in ads insertion.",
    "You are a helpful assistant for editing. You may need to adjust the subject's action or position.",
    "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, "
    "colors, textures, and the subject's pose or action.",
]

TASK_OPTIONS = [
    "Default / General",
    "Text to Image",
    "Text to Video",
    "Image Editing",
    "Subject-to-Image",
    "Image-to-Video",
    "Video Editing",
    "Video Editing (Content Propagation)",
    "Video Editing with Reference",
    "Ads / Content Insertion",
    "Video Editing (Action / Position)",
    "Video Editing (Style / Motion)",
]

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class BerniniR_PromptEmbedding:
    """Unified prompt embedding + text encoding node.

    Selects a task-specific system prompt, concatenates it with the user prompt,
    and encodes both positive and negative prompts through CLIP.

    Caches embeddings to disk so repeated prompts skip CLIP entirely.
    Optionally unloads CLIP after encoding to free VRAM.

    Inputs (required):
        task_type (COMBO): Task type for system prompt selection.
        user_prompt (STRING): User's prompt (multiline).

    Inputs (optional):
        clip (CLIP): Externally-loaded CLIP (legacy/compatibility). If not
            provided, clip_name is used to load CLIP on demand.
        clip_name (COMBO): Text encoder file to load on demand.
        clip_type (COMBO): CLIP type, defaults to "wan".
        clip_device (COMBO): "default" (GPU) or "cpu" (VRAM-saving).
        negative_prompt (STRING): Negative prompt (multiline).
        delimiter (STRING): Delimiter between system and user prompt.
        force_offload (BOOLEAN): Unload CLIP after encoding to free VRAM.
        use_disk_cache (BOOLEAN): Cache/restore embeddings to disk.
        cache_tag (STRING): Tag to isolate caches across CLIP models.

    Outputs:
        positive (CONDITIONING): Encoded positive conditioning.
        negative (CONDITIONING): Encoded negative conditioning.
        system_prompt (STRING): Selected system prompt (for debugging).
        full_prompt (STRING): Concatenated positive prompt (for debugging).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task_type": (TASK_OPTIONS, {"default": TASK_OPTIONS[0], "tooltip": "Pre-built task type. Selects a system prompt + default negative tailored to the task"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Your creative prompt describing the desired video content"}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "CLIP text encoder. If connected, CLIP loader inputs below are ignored"}),
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "CLIP model file (ignored if 'clip' input is connected)"}),
                "clip_type": (["wan"], {"default": "wan", "tooltip": "CLIP architecture type"}),
                "clip_device": (["default", "cpu"], {"default": "cpu", "tooltip": "'cpu' saves VRAM during encoding, 'default' = GPU"}),
                "negative_prompt": ("STRING", {"default": DEFAULT_NEGATIVE_PROMPT, "multiline": True, "tooltip": "Negative prompt. Leave empty to use task default"}),
                "delimiter": ("STRING", {"default": "\n", "tooltip": "Separator between system prompt and user prompt"}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload CLIP to CPU after encoding to free VRAM"}),
                "use_disk_cache": ("BOOLEAN", {"default": True, "tooltip": "Cache encoded embeddings to disk for reuse across sessions"}),
                "cache_tag": ("STRING", {"default": "", "tooltip": "Optional tag to isolate caches (e.g. per model or project)"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "system_prompt", "full_prompt")
    FUNCTION = "plan"
    CATEGORY = "Bernini-R/Conditioning"
    DESCRIPTION = "Task-aware prompt planner: selects system prompt, encodes text with CLIP, caches embeddings."

    @staticmethod
    def _derive_clip_tag(clip) -> str:
        try:
            model = clip.patcher.model
            total = sum(p.numel() for p in model.parameters())
            dtypes = sorted({str(p.dtype) for p in model.parameters()})
            return f"{model.__class__.__name__}_{total}_{'-'.join(dtypes)}"
        except Exception:
            return "unknown"

    def plan(
        self,
        task_type: str = TASK_OPTIONS[0],
        user_prompt: str = "",
        clip=None,
        clip_name: str = "",
        clip_type: str = "wan",
        clip_device: str = "cpu",
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        delimiter: str = "\n",
        force_offload: bool = True,
        use_disk_cache: bool = True,
        cache_tag: str = "",
    ):
        clip_loaded_internally = False
        # ── Select system prompt ──────────────────────────────────────
        task_index = TASK_OPTIONS.index(task_type) if task_type in TASK_OPTIONS else 0
        system_prompt = SYSTEM_PROMPTS[task_index]
        full_prompt = system_prompt + delimiter + user_prompt

        # ── Determine cache tag ───────────────────────────────────────
        user_tag = cache_tag.strip()
        if user_tag:
            tag = user_tag
        elif clip is not None:
            tag = self._derive_clip_tag(clip)
        elif clip_name:
            tag = f"{clip_name}:{clip_type}:{clip_device}"
        else:
            tag = "unknown"

        # ── Try disk cache first ──────────────────────────────────────
        positive = _load_cached_conditioning(full_prompt, tag) if use_disk_cache else None
        negative = _load_cached_conditioning(negative_prompt, tag) if use_disk_cache else None

        if positive is not None and negative is not None:
            logger.info("[BerniniR] Both prompts served from disk cache; skipping CLIP load.")
            if force_offload and clip is not None and clip_loaded_internally:
                clip = None  # drop reference to our internally-loaded CLIP
            collect_garbage()
            return (positive, negative, system_prompt, full_prompt)

        try:
            # ── Load CLIP if needed ───────────────────────────────────
            if clip is None:
                if not clip_name:
                    raise ValueError(
                        "No CLIP input connected and clip_name not set; "
                        "cannot encode uncached prompt."
                    )
                logger.info(f"[BerniniR] Loading CLIP on demand: {clip_name}")
                clip = _load_clip_internal(clip_name, clip_type, clip_device)
                clip_loaded_internally = True

            # ── Encode positive ───────────────────────────────────────
            if positive is None:
                positive = _encode_text(clip, full_prompt)
                if use_disk_cache:
                    _save_cached_conditioning(full_prompt, positive, tag)

            # ── Encode negative ───────────────────────────────────────
            if negative is None:
                negative = _encode_text(clip, negative_prompt)
                if use_disk_cache:
                    _save_cached_conditioning(negative_prompt, negative, tag)

        finally:
            if force_offload and clip is not None and clip_loaded_internally:
                # Only release a CLIP we loaded ourselves; a CLIP passed in by
                # the user is shared input and must not be unloaded here.
                del clip
                collect_garbage()

        logger.info(f"[BerniniR] PromptEmbedding task='{task_type}' → system line {task_index}")
        return (positive, negative, system_prompt, full_prompt)
