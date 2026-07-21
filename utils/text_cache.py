"""
Shared text-embedding disk cache and CLIP encoding helpers.

Used by both ``PromptEmbedding`` and ``SegmentSchedule`` nodes to avoid
duplicated encoding of identical prompts across workflow runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch
import comfy.sd
import folder_paths

from .log import get_logger as _get_logger

logger = _get_logger("TextCache")
# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent.parent / "text_embed_cache"
_MAX_CACHE_FILES = 200  # evict oldest files when exceeded


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _evict_oldest_if_needed() -> None:
    """Remove oldest cache files if count exceeds _MAX_CACHE_FILES."""
    try:
        files = sorted(_CACHE_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        while len(files) > _MAX_CACHE_FILES:
            oldest = files.pop(0)
            oldest.unlink(missing_ok=True)
            logger.debug(f"Evicted old cache file: {oldest.name}")
    except Exception:
        pass  # never fail on cache maintenance


def _cache_key(prompt: str, tag: str = "") -> str:
    payload = prompt if not tag else f"{tag}::{prompt}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _cache_path(prompt: str, tag: str = "") -> Path:
    return _ensure_cache_dir() / f"{_cache_key(prompt, tag)}.pt"


def _load_cached_conditioning(prompt: str, tag: str = "") -> Optional[list]:
    path = _cache_path(prompt, tag)
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        cond = data.get("cond")
        pooled = data.get("pooled", {})
        if cond is None:
            return None
        return [[cond, pooled]]
    except Exception as e:
        logger.warning(f"Corrupt cache {path.name}: {e}")
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _save_cached_conditioning(prompt: str, conditioning: list, tag: str = "") -> None:
    if not conditioning or len(conditioning[0]) < 2:
        return
    cond, pooled = conditioning[0]
    safe_pooled = {}
    if isinstance(pooled, dict):
        for k, v in pooled.items():
            if isinstance(v, (torch.Tensor, str, int, float, bool)) or v is None:
                safe_pooled[k] = v
    try:
        path = _cache_path(prompt, tag)
        torch.save({"cond": cond.cpu(), "pooled": safe_pooled}, path)
        logger.info(f"Saved text embed cache: {path.name}")
        _evict_oldest_if_needed()
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def _encode_text(clip, text: str):
    """Encode text using ComfyUI's CLIPTextEncode logic."""
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def _load_clip_internal(clip_name: str, clip_type: str = "wan", device: str = "cpu"):
    """Load a Wan-series text encoder on demand."""
    clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.WAN)
    model_options = {}
    if device == "cpu":
        model_options["load_device"] = torch.device("cpu")
        model_options["offload_device"] = torch.device("cpu")
    return comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type_enum,
        model_options=model_options,
    )
