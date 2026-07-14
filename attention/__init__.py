"""Attention backend module for Bernini-R.

Provides:
  - ``backends.py``: backend implementations with auto-detection.
  - ``config_node.py``: ComfyUI node for selecting backends.
"""
from .backends import (
    BACKEND_NAMES,
    available_backends,
    best_available,
    create_attention_override,
)
from .config_node import BerniniR_AttentionConfig
