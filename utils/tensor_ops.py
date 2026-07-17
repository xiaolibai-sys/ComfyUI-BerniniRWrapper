"""
Generic tensor operations safe for all ``torch.Tensor`` subclasses.

``QuantizedTensor`` (fp8) is a ``torch.Tensor`` subclass whose
``__torch_dispatch__`` intercepts ``.data`` assignments.  The old pattern
``param.data = torch.empty(0)`` silently keeps the fp8 storage alive
because the subclass never forwards the assignment to the underlying
storage.  The result is a ~14 GB RAM leak after model unload.

All operations in this module use **parameter/buffer replacement**
(``module._parameters[key] = ...``) instead of ``.data`` assignment,
which works uniformly for plain ``Tensor``, ``QuantizedTensor``, and
any future subclass.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _is_subclass_tensor(t: torch.Tensor) -> bool:
    """Return True if *t* is a non-plain torch.Tensor subclass."""
    return type(t) is not torch.Tensor and isinstance(t, torch.Tensor)


def _dtype_of(t: torch.Tensor | torch.nn.Parameter) -> torch.dtype:
    """Safe dtype extraction — some subclasses hide ``.dtype`` behind
    a property that may raise."""
    try:
        return t.dtype
    except Exception:
        return torch.float32


def _device_of(t: torch.Tensor | torch.nn.Parameter) -> torch.device:
    """Safe device extraction."""
    try:
        return t.device
    except Exception:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Parameter / buffer replacement (QuantizedTensor-safe)
# ---------------------------------------------------------------------------

def replace_param(
    module: torch.nn.Module,
    key: str,
    new_value: torch.Tensor | torch.nn.Parameter | None,
) -> None:
    """Replace a parameter object on *module*, dereferencing the old one.

    Unlike ``param.data = ...``, this works for ``QuantizedTensor`` and
    any other ``torch.Tensor`` subclass because the old object is simply
    dropped — no ``__torch_dispatch__`` interception possible.
    """
    if new_value is not None and not isinstance(new_value, torch.nn.Parameter):
        new_value = torch.nn.Parameter(new_value, requires_grad=False)
    try:
        module._parameters[key] = new_value
    except Exception:
        pass


def replace_buffer(
    module: torch.nn.Module,
    key: str,
    new_value: torch.Tensor | None,
) -> None:
    """Replace a buffer on *module*, dereferencing the old one."""
    try:
        module._buffers[key] = new_value
    except Exception:
        pass


def zero_param(module: torch.nn.Module, key: str) -> None:
    """Replace a parameter with a zero-size tensor to free its storage."""
    p = module._parameters.get(key)
    if p is None:
        return
    dtype = _dtype_of(p)
    replace_param(module, key, torch.empty((0,), dtype=dtype))


def zero_buffer(module: torch.nn.Module, key: str) -> None:
    """Replace a buffer with a zero-size tensor to free its storage."""
    b = module._buffers.get(key)
    if b is None:
        return
    dtype = _dtype_of(b)
    replace_buffer(module, key, torch.empty((0,), dtype=dtype))


# ---------------------------------------------------------------------------
# Module-level free
# ---------------------------------------------------------------------------

def free_module_storage(module: torch.nn.Module) -> None:
    """Replace every parameter and buffer in *module* with an empty tensor.

    This is the generic equivalent of iterating ``.data = empty(0)`` but
    works correctly for ``QuantizedTensor`` and any subclass.

    Used by :func:`~.vram.release_model_ram` to synchronously return
    host RAM after a dual-expert model switch.
    """
    for key in list(module._parameters.keys()):
        zero_param(module, key)
    for key in list(module._buffers.keys()):
        zero_buffer(module, key)


# ---------------------------------------------------------------------------
# Safe device movement
# ---------------------------------------------------------------------------

def move_module_to(
    module: torch.nn.Module,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> None:
    """Move *module* to *device*, safe for all tensor subclasses.

    Uses ``module.to(device)`` (which dispatches through
    ``torch.Tensor.to`` and works for subclasses) but falls back to
    parameter-by-parameter movement if the bulk ``.to()`` call fails
    (e.g. a QuantizedTensor op that errors on the target device).
    """
    try:
        module.to(device, non_blocking=non_blocking)
        return
    except Exception:
        pass

    # Fallback: move parameters and buffers individually.
    for key, p in list(module._parameters.items()):
        if p is not None:
            try:
                moved = p.to(device, non_blocking=non_blocking)
                replace_param(module, key, moved)
            except Exception:
                pass
    for key, b in list(module._buffers.items()):
        if b is not None:
            try:
                replace_buffer(module, key, b.to(device, non_blocking=non_blocking))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Safe pin_memory
# ---------------------------------------------------------------------------

def pin_module(module: torch.nn.Module) -> None:
    """Pin CPU tensors in *module* for fast async H2D transfers.

    Skips ``QuantizedTensor`` and other subclasses that don't support
    pinning.  Uses parameter/buffer replacement so the old unpinned
    storage is dereferenced immediately.
    """
    for key, p in list(module._parameters.items()):
        if p is None or not p.is_cpu or p.is_pinned():
            continue
        if _is_subclass_tensor(p):
            continue  # QuantizedTensor etc. — can't pin
        try:
            pinned = p.data.pin_memory()
            replace_param(module, key, pinned)
        except Exception:
            pass

    for key, b in list(module._buffers.items()):
        if b is None or not b.is_cpu or b.is_pinned():
            continue
        if _is_subclass_tensor(b):
            continue
        try:
            replace_buffer(module, key, b.pin_memory())
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Record stream (for async transfer safety)
# ---------------------------------------------------------------------------

def record_module_stream(module: torch.nn.Module, stream: torch.cuda.Stream) -> None:
    """Record *stream* on every CUDA tensor in *module*.

    Without this the caching allocator may recycle a tensor's memory
    while an async copy on *stream* is still reading it — producing a
    ``STATUS_ACCESS_VIOLATION`` on Windows.
    """
    for p in module.parameters():
        if p.is_cuda:
            try:
                p.data.record_stream(stream)
            except Exception:
                pass
    for b in module.buffers():
        if b.is_cuda:
            try:
                b.data.record_stream(stream)
            except Exception:
                pass
