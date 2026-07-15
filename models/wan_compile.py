"""
Wan model torch.compile adaptation for Bernini-R.

Compiles ``forward_orig`` while keeping the original ``forward`` →
``_forward`` → ``forward_orig`` chain intact.  Also patches the
forward for context-window RoPE ``t_start`` and NTK scaling support.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPILE_MODES = ["none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]

import os

import torch._dynamo as _dynamo
_dynamo.config.suppress_errors = True
_dynamo.config.recompile_limit = 32

# ── Inductor C++ codegen: inject Windows SDK include/lib paths ────
# The standalone Python env lacks MSVC/WinSDK headers (crtdbg.h) and
# import libs (kernel32.lib).  Resolve the installed toolchain and add
# the missing dirs to INCLUDE / LIB so inductor's cl.exe invocations
# succeed.
#
# Resolution order (per include/lib pair):
#   1. Explicit override env vars  BERNINI_MSVC_INCLUDE / BERNINI_MSVC_LIB
#   2. SDK root env vars written by the VS / Windows SDK installer
#      (WindowsSdkDir, UniversalCRTSdkDir)
#   3. Scan well-known default install locations, derived from
#      %ProgramFiles(x86)%, %ProgramFiles%, %SystemDrive% plus a few
#      common extra drives (no single hardcoded drive letter).
def _winsdk_roots() -> list[tuple[str, str]]:
    """Yield (include_root, lib_root) candidate pairs to probe.

    De-duplicated; order reflects the resolution priority documented
    above (SDK env vars before drive-rooted default locations).
    """
    seen: set[tuple[str, str]] = set()
    roots: list[tuple[str, str]] = []

    def _add(inc: str, lib: str) -> None:
        key = (os.path.normcase(inc), os.path.normcase(lib))
        if key not in seen:
            seen.add(key)
            roots.append((inc, lib))

    # (2) SDK root env vars set by the Windows SDK / VS installer.
    for env in ("WindowsSdkDir", "UniversalCRTSdkDir"):
        sdk = os.environ.get(env)
        if sdk:
            _add(os.path.join(sdk, "Include"), os.path.join(sdk, "Lib"))

    # (3a) ProgramFiles-style env vars already name the base dir.
    for env in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.path.normpath(os.environ.get(env)) if os.environ.get(env) else None
        if base:
            _add(os.path.join(base, "Windows Kits", "10", "Include"),
                 os.path.join(base, "Windows Kits", "10", "Lib"))

    # (3b) Drive-rooted default locations, derived from %SystemDrive%
    #      plus a handful of common extra drives as a last resort.
    #      Normalised to absolute "X:\" form so os.path.join yields a
    #      real absolute path regardless of the current working dir.
    drives: list[str] = []
    sysdrv = os.environ.get("SystemDrive")
    if sysdrv:
        drives.append(sysdrv.rstrip("\\/") + "\\")
    for d in ("C", "D", "E", "F"):
        drv = f"{d}:\\"
        if drv not in drives:
            drives.append(drv)
    for drv in drives:
        for pf in ("Program Files (x86)", "Program Files", ""):
            _add(os.path.join(drv, pf, "Windows Kits", "10", "Include"),
                 os.path.join(drv, pf, "Windows Kits", "10", "Lib"))
    return roots


def _probe_winsdk() -> tuple[str | None, str | None]:
    """Return (include_paths, lib_paths) for the newest Windows Kit."""
    for inc_root, lib_root in _winsdk_roots():
        if not (os.path.isdir(inc_root) and os.path.isdir(lib_root)):
            continue

        versions = sorted(
            (d for d in os.listdir(inc_root)
             if os.path.isdir(os.path.join(inc_root, d))),
            reverse=True,
        )
        for v in versions:
            ucrt_inc = os.path.join(inc_root, v, "ucrt")
            if not os.path.isdir(ucrt_inc):
                continue
            inc_parts = [ucrt_inc]
            for sub in ("shared", "um"):
                p = os.path.join(inc_root, v, sub)
                if os.path.isdir(p):
                    inc_parts.append(p)

            # Match version on lib side (same SDK version)
            lib_ver_dir = os.path.join(lib_root, v)
            if not os.path.isdir(lib_ver_dir):
                continue
            lib_parts = []
            for sub in ("ucrt/x64", "um/x64"):
                p = os.path.join(lib_ver_dir, sub)
                if os.path.isdir(p):
                    lib_parts.append(p)

            return os.pathsep.join(inc_parts), os.pathsep.join(lib_parts) if lib_parts else None
    return None, None

_inc_path, _lib_path = os.environ.get("BERNINI_MSVC_INCLUDE"), os.environ.get("BERNINI_MSVC_LIB")
if not _inc_path or not _lib_path:
    _probed_inc, _probed_lib = _probe_winsdk()
    if not _inc_path:
        _inc_path = _probed_inc
    if not _lib_path:
        _lib_path = _probed_lib

if _inc_path:
    _existing = os.environ.get("INCLUDE", "")
    os.environ["INCLUDE"] = f"{_inc_path}{os.pathsep}{_existing}" if _existing else _inc_path
    logger.info("[BerniniR] Injected WinSDK include: %s", _inc_path)
else:
    logger.warning(
        "[BerniniR] Windows Kits include dir not found. "
        "Set BERNINI_MSVC_INCLUDE, or point WindowsSdkDir/UniversalCRTSdkDir "
        "at the Windows 10 SDK."
    )

if _lib_path:
    _existing = os.environ.get("LIB", "")
    os.environ["LIB"] = f"{_lib_path}{os.pathsep}{_existing}" if _existing else _lib_path
    logger.info("[BerniniR] Injected WinSDK lib: %s", _lib_path)
else:
    logger.warning(
        "[BerniniR] Windows Kits lib dir not found (kernel32.lib). "
        "Set BERNINI_MSVC_LIB, or point WindowsSdkDir/UniversalCRTSdkDir "
        "at the Windows 10 SDK."
    )

import torch._inductor.config as _inductor_config
_inductor_config.cpp_wrapper = False

# ── Scoped, safe compile cache ──────────────────────────────────────
# Previously this module did ``shutil.rmtree(torch._inductor.codecache.cache_dir())``
# on EVERY import.  That path is torch's *global* inductor cache, shared by every
# other project/model on the machine — so a Bernini-R restart would wipe everyone
# else's compiled graphs and force full recompiles.  We now isolate Bernini-R's
# compiled artifacts in a dedicated directory and only purge them when (a) this
# package's compiled-graph contract changes (a version sentinel) or (b) the user
# explicitly asks via BERNINI_PURGE_COMPILE_CACHE.
import tempfile as _tempfile
import torch._inductor.codecache as _codecache
import shutil as _shutil

# Bump when a code change invalidates previously compiled graphs (e.g. the
# forward signature, RoPE handling, or hook behaviour changes).
_BERNINI_CACHE_VERSION = "1"


def _bernini_compile_cache_dir() -> str:
    """Return a Bernini-R-isolated inductor/dynamo cache directory.

    Honours BERNINI_COMPILE_CACHE_DIR; otherwise a process-local temp dir.
    This is NOT torch's global cache, so purging it never touches other
    projects.
    """
    env = os.environ.get("BERNINI_COMPILE_CACHE_DIR")
    if env:
        return os.path.abspath(env)
    return os.path.join(_tempfile.gettempdir(), "bernini_r_inductor_cache")


_BERNINI_CACHE = _bernini_compile_cache_dir()
os.makedirs(_BERNINI_CACHE, exist_ok=True)
# inductor (and dynamo's FX graph cache) resolve their cache dir from these
# env vars *at call time*, so setting them here — at import, before any
# compile — redirects every compiled artifact into our scoped directory.
# (Assigning torch._inductor.config.cache_dir raises on torch>=2.10, hence
# the env-var route.)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _BERNINI_CACHE
os.environ["TORCH_COMPILE_CACHE_DIR"] = _BERNINI_CACHE


def purge_compile_cache(force: bool = False) -> bool:
    """Delete Bernini-R's isolated compile cache.

    Safe by construction: only ever touches the directory returned by
    ``_bernini_compile_cache_dir`` (a Bernini-R temp dir), never the global
    torch cache shared by other projects.  Returns True if a purge occurred.
    """
    target = _bernini_compile_cache_dir()
    if not os.path.isdir(target):
        return False
    try:
        _shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        return True
    except Exception as e:
        logger.warning("[BerniniR] Failed to purge compile cache %s: %s", target, e)
        return False


def _maybe_auto_purge_compile_cache() -> None:
    """Auto-purge stale compiled graphs only when the code contract changes.

    Replaces the old unconditional global rmtree: normal restarts keep the
    cache (fast), but an upgrade that changes the compiled graph invalidates
    it exactly once.  BERNINI_PURGE_COMPILE_CACHE=1 forces a manual purge.
    """
    sentinel = os.path.join(_BERNINI_CACHE, ".bernini_cache_version")
    prev = ""
    if os.path.isfile(sentinel):
        try:
            with open(sentinel, "r") as _f:
                prev = _f.read().strip()
        except Exception:
            prev = ""
    if prev != _BERNINI_CACHE_VERSION:
        if purge_compile_cache():
            logger.info(
                "[BerniniR] Compile cache purged on version change (%s -> %s).",
                prev or "<none>", _BERNINI_CACHE_VERSION,
            )
        try:
            with open(sentinel, "w") as _f:
                _f.write(_BERNINI_CACHE_VERSION)
        except Exception:
            pass
    if os.environ.get("BERNINI_PURGE_COMPILE_CACHE", "").lower() in ("1", "true", "yes"):
        if purge_compile_cache():
            logger.info("[BerniniR] Compile cache purged (BERNINI_PURGE_COMPILE_CACHE).")


_maybe_auto_purge_compile_cache()


# ===========================================================================
# Public API
# ===========================================================================

def apply_torch_compile(
    model_patcher,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> None:
    """Apply torch.compile to a loaded Wan/Bernini model.

    Compiles only the main transformer forward (``forward_orig`` for
    upstream WanModel, ``transformer_forward`` for BerniniRWanModel).
    The pre-processing (``pre_forward``) stays in eager mode.
    """
    if mode == "none":
        logger.info("[BerniniR] torch.compile disabled (mode='none').")
        return model_patcher

    base_model = getattr(model_patcher, 'model', None)
    if base_model is None:
        logger.warning("[BerniniR] No .model found on patcher.")
        return model_patcher

    wan_model = getattr(base_model, 'diffusion_model', None)
    if wan_model is None:
        logger.warning("[BerniniR] No .diffusion_model found.")
        return model_patcher

    if not torch.cuda.is_available():
        logger.warning("[BerniniR] CUDA not available; skipping compile.")
        return model_patcher

    # ── Enable TF32 for matmul (faster, prevents fp32 trace fusion) ─
    torch.set_float32_matmul_precision('high')

    # ── Resolve the compilable method name ─────────────────────────
    # BerniniRWanModel uses ``transformer_forward``; upstream WanModel
    # uses ``forward_orig``.  Detect whichever is present.
    if hasattr(wan_model, 'transformer_forward'):
        fwd_attr = 'transformer_forward'
    elif hasattr(wan_model, 'forward_orig'):
        fwd_attr = 'forward_orig'
    else:
        logger.warning("[BerniniR] No compilable forward method found.")
        return model_patcher

    # ── Save original ─────────────────────────────────────────────
    _orig_key = '_original_' + fwd_attr
    if not hasattr(wan_model, _orig_key):
        setattr(wan_model, _orig_key, getattr(wan_model, fwd_attr))

    # ── Compile ───────────────────────────────────────────────────
    compilable_fn = getattr(wan_model, _orig_key)

    torch.compiler.reset()

    if mode in ("reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        logger.info(f"[BerniniR] Mode '{mode}' → 'default' (no C++ SDK)")
        mode = "default"

    try:
        compile_kwargs = {"dynamic": dynamic}
        if fullgraph:
            compile_kwargs["fullgraph"] = True

        compiled_fn = torch.compile(compilable_fn, mode=mode, **compile_kwargs)
        setattr(wan_model, fwd_attr, compiled_fn)
        logger.info(f"[BerniniR] torch.compile applied to {fwd_attr}: "
                    f"mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")
    except Exception as e:
        logger.warning(f"[BerniniR] torch.compile failed: {e}. Falling back to uncompiled.")
        setattr(wan_model, fwd_attr, compilable_fn)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return model_patcher


def compile_wan_model(model_patcher, compile_mode: str = "none",
                      fullgraph: bool = False, dynamic: bool = True):
    """Public API: apply torch.compile to a Wan model patcher."""
    if compile_mode not in COMPILE_MODES:
        raise ValueError(f"Unknown compile_mode '{compile_mode}'. Options: {COMPILE_MODES}")
    return apply_torch_compile(model_patcher, mode=compile_mode,
                               fullgraph=fullgraph, dynamic=dynamic)


def restore_model(model_patcher) -> None:
    """Restore a model to its original (uncompiled) state."""
    base_model = getattr(model_patcher, 'model', None)
    if base_model is None:
        return
    wan_model = getattr(base_model, 'diffusion_model', None)
    if wan_model is None:
        return

    for attr in ('forward_orig', 'transformer_forward'):
        orig_key = '_original_' + attr
        if hasattr(wan_model, orig_key):
            setattr(wan_model, attr, getattr(wan_model, orig_key))
            delattr(wan_model, orig_key)

    if hasattr(wan_model, '_original_forward'):
        wan_model.forward = wan_model._original_forward
        delattr(wan_model, '_original_forward')

    logger.info("[BerniniR] Model restored to uncompiled state.")
