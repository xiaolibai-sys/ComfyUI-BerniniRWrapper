"""
VRAM management utilities for Bernini-R.

Lightweight helpers wrapping ComfyUI's model_management and PyTorch's CUDA APIs.
"""
from __future__ import annotations

import gc

import torch
import comfy.model_management as mm

from .tensor_ops import free_module_storage

from .log import get_logger as _get_logger

logger = _get_logger("VRAM")
def get_compute_device() -> torch.device:
    """Return ComfyUI's configured compute device."""
    return mm.get_torch_device()


def soft_empty_cache() -> None:
    """Safe wrapper around ComfyUI's soft_empty_cache."""
    try:
        mm.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _empty_workingset() -> None:
    """Trim the Windows process working set to return physical pages."""
    import sys
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_bool
        h = kernel32.GetCurrentProcess()
        psapi.EmptyWorkingSet(h)
    except Exception:
        pass


def collect_garbage(aggressive: bool = False) -> None:
    """Force Python GC + CUDA stream sync + cache flush.

    The ``synchronize()`` is load-bearing on Windows: a model's block-swap
    unload may still have async copies in flight on a CUDA stream.  Freeing /
    re-allocating VRAM behind a running transfer stream is exactly what triggers
    ``STATUS_ACCESS_VIOLATION`` (0xC0000005) during the *next* model load — the
    dual-expert HIGH->LOW switch in particular.  Waiting for all streams to
    finish before we reclaim memory makes the switch safe.

    When *aggressive* is True (used during the HIGH->LOW model switch), also
    trim the Windows working set so physical RAM held by PyTorch's CPU
    allocator is returned to the OS before the next large allocation.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    soft_empty_cache()
    try:
        torch._C._host_emptyCache()
    except Exception:
        pass
    if aggressive:
        _empty_workingset()
        gc.collect()
        soft_empty_cache()


def get_free_vram_mb() -> float:
    """Get free VRAM in MiB."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, total = torch.cuda.mem_get_info()
        return free / (1024 * 1024)
    except Exception:
        return 0.0


def log_system_ram(tag: str = "") -> None:
    """Log available host RAM (MiB) so we can tell a leak from a tight machine.

    Used during the dual-expert HIGH->LOW switch: if available RAM is still
    ~one model short of what LOW needs after we release HIGH, the problem is
    the machine's RAM size, not a reference leak.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        label = f" [{tag}]" if tag else ""
        logger.info(
            "RAM%s: %.1f / %.1f GiB free (%.1f%% used)",
            label, vm.available / (1024 ** 3), vm.total / (1024 ** 3),
            vm.percent,
        )
    except Exception:
        # psutil not available — fall back to ctypes for a rough number.
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                ]
            ms = _MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(ms)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            avail = ms.ullAvailPhys / (1024 ** 3)
            total = ms.ullTotalPhys / (1024 ** 3)
            label = f" [{tag}]" if tag else ""
            logger.info(
                "RAM%s: %.1f / %.1f GiB free", label, avail, total)
        except Exception:
            pass


def release_model_ram(patcher: Any) -> None:
    """Immediately shrink a model's weight tensors to release host RAM.

    Uses :func:`tensor_ops.free_module_storage` which replaces every
    parameter/buffer with a zero-size tensor.  This works correctly for
    ``QuantizedTensor`` (fp8) and any other ``torch.Tensor`` subclass
    because the old object is dereferenced — no ``__torch_dispatch__``
    interception possible.
    """
    try:
        model = getattr(patcher, "model", None)
        if model is None:
            return
        for module in model.modules():
            free_module_storage(module)
    except Exception as e:
        logger.warning("release_model_ram shrink failed: %s", e)
    collect_garbage(aggressive=True)


def log_memory(tag: str = "") -> None:
    """Log current VRAM usage."""
    if not torch.cuda.is_available():
        return
    try:
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / (1024 * 1024 * 1024)
        total_gb = total / (1024 * 1024 * 1024)
        label = f" [{tag}]" if tag else ""
        logger.info(f"VRAM{label}: {used:.2f} / {total_gb:.2f} GiB used")
    except Exception:
        pass


def inference_mode():
    """Context manager: torch.inference_mode() with fallback to no_grad."""
    try:
        return torch.inference_mode()
    except AttributeError:
        return torch.no_grad()
