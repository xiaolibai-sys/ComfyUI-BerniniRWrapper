"""
Attention backends with auto-detection and fallback chain.

Fallback order (best → worst):
  SageAttention 3 (Blackwell sm_100+) → SageAttention 2/1 → FlashAttention →
  xformers → PyTorch SDPA (always available)

Each backend is a standalone function that can be used as
``transformer_options["optimized_attention_override"]``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ..utils.log import get_logger as _get_logger

logger = _get_logger("Attn")
# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

_AVAILABLE: dict[str, bool] = {}

# -- SageAttention 3 (Blackwell GPUs: 5090, 5070 Ti, B200, etc.) --
try:
    from sageattn3 import sageattn3_blackwell  # noqa: F401
    _AVAILABLE["sage3"] = True
except ImportError:
    _AVAILABLE["sage3"] = False

# -- SageAttention 2/1 (older GPUs) --
try:
    from sageattention import sageattn  # noqa: F401
    _AVAILABLE["sage"] = True
except ImportError:
    _AVAILABLE["sage"] = False

# -- FlashAttention --
try:
    from flash_attn import flash_attn_func  # noqa: F401
    _AVAILABLE["flash"] = True
except ImportError:
    _AVAILABLE["flash"] = False

# -- xformers --
try:
    import xformers.ops  # noqa: F401
    _AVAILABLE["xformers"] = True
except ImportError:
    _AVAILABLE["xformers"] = False

# -- PyTorch SDPA (always available on CUDA) --
_AVAILABLE["sdpa"] = True


def available_backends() -> list[str]:
    """Return list of available backend names, best first."""
    out = []
    for name in ["sage3", "sage", "flash", "xformers", "sdpa"]:
        if _AVAILABLE.get(name, False):
            out.append(name)
    return out


def best_available() -> str:
    """Return the best available backend name."""
    for name in ["sage3", "sage", "flash", "xformers", "sdpa"]:
        if _AVAILABLE.get(name, False):
            return name
    return "sdpa"


# ---------------------------------------------------------------------------
# Utility: reshape for attention backends
# ---------------------------------------------------------------------------

def _sdpa_core(q, k, v, heads):
    """Core SDPA call.  q/k/v are [B, seq, dim], returns [B, seq, dim]."""
    b, sq, dim = q.shape
    head_dim = dim // heads
    q = q.view(b, sq, heads, head_dim).transpose(1, 2)
    sk = k.shape[1]
    k = k.view(b, sk, heads, head_dim).transpose(1, 2)
    sv = v.shape[1]
    v = v.view(b, sv, heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, sq, dim)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------
# Custom CUDA kernels (Sage/Flash/xformers) are registered as PyTorch
# custom ops with fake tensor support.  This allows Dynamo to trace
# through them without graph breaks — no C++ glue code needed.
#
# Fallback: if custom_op fails (older PyTorch), use @torch.compiler.disable
# which works on Linux/macOS but may need C++ compiler on Windows.
# ---------------------------------------------------------------------------

def _register_custom_op(name, fn, fake_fn):
    """Register fn as a torch custom op with fake tensor support.

    Custom ops let Dynamo/Inductor treat the kernel as a black box:
      - Dynamo inserts a graph break at the call site (same as
        @torch.compiler.disable but without the "skipped function"
        cache-invalidation problem).
      - The fake implementation allows Dynamo to propagate tensor
        metadata (shape/dtype/device) through the graph without
        tracing into the C++/CUDA kernel.

    Requires PyTorch >= 2.4 and type-annotated function parameters
    (PyTorch 2.12+ enforces this strictly).
    """
    try:
        op = torch.library.custom_op(f"bernini::{name}", mutates_args=())(fn)
        op.register_fake(fake_fn)
        logger.debug(f"Registered custom op bernini::{name}")
        return op
    except Exception as e:
        logger.warning(
            f"Failed to register custom op bernini::{name}: {e}. "
            f"Using raw function (Dynamo will graph-break at this op)."
        )
        return fn  # fallback: use raw function (may cause graph breaks)

# --- Fake implementations (return same-shape tensor, no compute) ---
def _fake_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.empty_like(q)

# --- Define raw backend functions (always defined, only called if available) ---
def _sage3_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from sageattn3 import sageattn3_blackwell
    # SageAttn3 only supports fp16/bf16.  torch.compile may fuse away
    # upstream dtype casts leaving float32 inputs — clamp explicitly.
    orig_dtype = q.dtype
    target_dtype = orig_dtype if orig_dtype in (torch.float16, torch.bfloat16) else torch.float16
    q = q.to(target_dtype) if q.dtype != target_dtype else q
    k = k.to(target_dtype) if k.dtype != target_dtype else k
    v = v.to(target_dtype) if v.dtype != target_dtype else v
    return sageattn3_blackwell(q, k, v, is_causal=False).to(orig_dtype)

def _sage_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             mask: Optional[torch.Tensor]) -> torch.Tensor:
    from sageattention import sageattn
    return sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout="HND")

def _flash_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from flash_attn import flash_attn_func
    return flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

def _xformers_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 mask: Optional[torch.Tensor]) -> torch.Tensor:
    import xformers.ops as xops
    return xops.memory_efficient_attention(q, k, v, attn_bias=mask)

# --- Register as custom ops for Dynamo (only if available) ---
_SAGE3_OP = _register_custom_op("sage3_attn", _sage3_fn, _fake_sdpa) if _AVAILABLE.get("sage3") else None
_SAGE_OP = _register_custom_op("sage_attn", _sage_fn, _fake_sdpa) if _AVAILABLE.get("sage") else None
_FLASH_OP = _register_custom_op("flash_attn", _flash_fn, _fake_sdpa) if _AVAILABLE.get("flash") else None
_XFORMERS_OP = _register_custom_op("xformers_attn", _xformers_fn, _fake_sdpa) if _AVAILABLE.get("xformers") else None


def backend_sage3(original_fn, q, k, v, heads, mask=None, **kwargs):
    """SageAttention 3 — Blackwell GPUs (RTX 50xx, B200).

    The Sage3 kernel is registered as a torch custom op (bernini::sage3_attn).
    Dynamo traces the reshape logic below, then graph-breaks cleanly at the
    custom op call — it never sees fp4quant_cuda.
    """
    b, sq, dim = q.shape
    head_dim = dim // heads
    q = q.view(b, sq, heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(b, k.shape[1], heads, head_dim).transpose(1, 2).contiguous()
    v = v.view(b, v.shape[1], heads, head_dim).transpose(1, 2).contiguous()
    fn = _SAGE3_OP if _SAGE3_OP is not None else _sage3_fn
    out = fn(q, k, v)
    return out.transpose(1, 2).reshape(b, sq, dim)


def backend_sage(original_fn, q, k, v, heads, mask=None, **kwargs):
    """SageAttention 1/2 — pre-Blackwell GPUs."""
    b, sq, dim = q.shape
    head_dim = dim // heads
    q = q.view(b, sq, heads, head_dim)
    k = k.view(b, k.shape[1], heads, head_dim)
    v = v.view(b, v.shape[1], heads, head_dim)
    fn = _SAGE_OP if _SAGE_OP is not None else _sage_fn
    out = fn(q, k, v, mask)
    return out.reshape(b, sq, dim)


def backend_flash(original_fn, q, k, v, heads, mask=None, **kwargs):
    """FlashAttention 2/3."""
    b, sq, dim = q.shape
    head_dim = dim // heads
    q = q.view(b, sq, heads, head_dim)
    k = k.view(b, k.shape[1], heads, head_dim)
    v = v.view(b, v.shape[1], heads, head_dim)
    fn = _FLASH_OP if _FLASH_OP is not None else _flash_fn
    out = fn(q, k, v)
    return out.reshape(b, sq, dim)


def backend_xformers(original_fn, q, k, v, heads, mask=None, **kwargs):
    """xformers memory_efficient_attention."""
    b, sq, dim = q.shape
    head_dim = dim // heads
    q = q.view(b, sq, heads, head_dim)
    k = k.view(b, k.shape[1], heads, head_dim)
    v = v.view(b, v.shape[1], heads, head_dim)
    fn = _XFORMERS_OP if _XFORMERS_OP is not None else _xformers_fn
    out = fn(q, k, v, mask)
    return out.reshape(b, sq, dim)


def backend_sdpa(original_fn, q, k, v, heads, mask=None, **kwargs):
    """PyTorch SDPA — always available, fallback of last resort."""
    return _sdpa_core(q, k, v, heads)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Callable] = {
    "sage3":    backend_sage3,
    "sage":     backend_sage,
    "flash":    backend_flash,
    "xformers": backend_xformers,
    "sdpa":     backend_sdpa,
}

BACKEND_NAMES = ["auto"] + [k for k in _BACKENDS if _AVAILABLE.get(k, False)]
# Always include sdpa as final fallback in the dropdown even if others are listed
if "sdpa" not in BACKEND_NAMES:
    BACKEND_NAMES.append("sdpa")


def get_backend_fn(name: str) -> Optional[Callable]:
    """Get a backend function by name. ``"auto"`` returns the best available."""
    if name == "auto":
        name = best_available()
    return _BACKENDS.get(name)


def create_attention_override(
    backend: str = "auto",
    *,
    force_backend: bool = False,
) -> Callable:
    """Create an ``optimized_attention_override`` function.

    Args:
        backend: Backend name or ``"auto"`` for automatic selection.
        force_backend: If True, raise an error if the backend is unavailable.
            If False, silently fall back through the chain to SDPA.

    Returns:
        An override function suitable for
        ``transformer_options["optimized_attention_override"]``.
    """
    if backend == "auto":
        backend = best_available()

    fn = _BACKENDS.get(backend)
    if fn is not None:
        logger.debug(f"Attention backend: {backend}")
        return fn

    if force_backend:
        raise ValueError(
            f"Attention backend '{backend}' is not available. "
            f"Available: {available_backends()}"
        )

    # Fallback chain — try each backend in order until one works
    for name, fallback_fn in _BACKENDS.items():
        if _AVAILABLE.get(name, False):
            logger.warning(
                f"Backend '{backend}' unavailable; "
                f"falling back to '{name}'."
            )
            return fallback_fn

    # Should never reach here — SDPA is always available
    return backend_sdpa


# ---------------------------------------------------------------------------
# Log available backends at import time
# ---------------------------------------------------------------------------

_avail = available_backends()
logger.debug(f"Attention backends available: {_avail}; best: {best_available()}")
