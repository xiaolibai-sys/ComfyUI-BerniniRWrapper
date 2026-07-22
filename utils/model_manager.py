"""
Self-managed model lifecycle for Bernini-R.

Provides ``BerniniRModelHandle``: a lightweight object that knows how to load
and unload a Bernini-R / Wan model from disk on demand.  LoRA weights are
merged inline when the model is loaded, so the handle never depends on
ComfyUI's ``load_lora_for_models`` / patcher mechanism for model management.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

import torch
import comfy.model_management as mm

from .vram import collect_garbage

from .log import get_logger as _get_logger

logger = _get_logger("ModelMgr")
# ---------------------------------------------------------------------------
# Lightweight proxy for latent_preview.prepare_callback compatibility
# ---------------------------------------------------------------------------

class _LatentFormatProxy:
    """Minimal model-like object that provides ``latent_format`` without
    loading the full model.  ``prepare_callback`` only accesses:

        model.load_device         – set on the handle itself
        model.model.latent_format – provided here

    So callers can create a preview callback before the handle is loaded.
    """

    def __init__(self):
        import comfy.latent_formats
        self.latent_format = comfy.latent_formats.Wan21()


# ---------------------------------------------------------------------------
# Module-level model cache — avoids a full checkpoint disk read + build on re-runs
# ---------------------------------------------------------------------------
_MAX_MODEL_CACHE: int = 1
_model_cache: OrderedDict[str, Any] = OrderedDict()


def _make_cache_key(
    model_path: str,
    lora_specs: list | None,
    compile_cfg: dict | None,
    attn_backend_args: dict | None,
    block_swap: bool = False,
) -> str:
    """Stable hash of the full model config for cache lookup.

    ``block_swap`` MUST be part of the key: it changes the patcher's
    ``load_device`` and attaches a ``BlockSwapManager`` to
    ``diffusion_model`` (see ``wan_model.load_bernini_model``).  Reusing a
    non-block-swap patcher for a block-swap request would leave the model on
    the GPU with no manager, breaking block swapping at runtime.
    """
    _loras = sorted([(str(p), float(s)) for p, s in (lora_specs or [])])
    _compile = dict(sorted(compile_cfg.items())) if compile_cfg else {}
    # attn_backend_args may be a BerniniAttention dataclass — not a dict.
    _attn = {}
    if attn_backend_args is not None:
        import dataclasses as _dc
        if _dc.is_dataclass(attn_backend_args) and not isinstance(attn_backend_args, dict):
            _attn = dict(sorted(_dc.asdict(attn_backend_args).items()))
        elif isinstance(attn_backend_args, dict):
            _attn = dict(sorted(attn_backend_args.items()))
    raw = json.dumps(
        dict(path=model_path, loras=_loras, compile=_compile, attn=_attn,
             block_swap=block_swap),
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_evict_patcher(patcher: Any) -> bool:
    """Remove a specific patcher from the LRU cache and force cleanup."""
    removed = False
    for key, value in list(_model_cache.items()):
        if value is patcher:
            _model_cache.pop(key, None)
            logger.info("Evicted model from cache: %s", key[:8])
            removed = True
    if removed:
        collect_garbage()
    return removed


def _cache_put(key: str, patcher: Any) -> None:
    """Store a built patcher in the LRU cache, evicting oldest if full."""
    _model_cache[key] = patcher
    _model_cache.move_to_end(key)
    while len(_model_cache) > _MAX_MODEL_CACHE:
        oldest_key, oldest_patcher = _model_cache.popitem(last=False)
        logger.info("Model cache evict: %s", oldest_key[:8])
        try:
            # Cancel any async block-swap transfers before unloading.
            _dm = getattr(getattr(oldest_patcher, "model", None),
                          "diffusion_model", None)
            _mgr = getattr(_dm, "_block_swap_mgr", None)
            if _mgr is not None:
                _mgr.evict_all()
                try:
                    delattr(_dm, "_block_swap_mgr")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            # Fully release from ComfyUI's model manager so VRAM is freed.
            mm.unload_model_and_clones(oldest_patcher)
        except Exception:
            pass
        del oldest_patcher
    collect_garbage()


class BerniniRModelHandle:
    """Lazy-loading handle for a Bernini-R diffusion model.

    The handle itself is tiny (only paths and configs).  The actual model is
    loaded only when ``load()`` is called, and fully released with
    ``unload()``.  A module-level LRU cache (``_MAX_MODEL_CACHE = 1``) keeps
    the most recently used patcher object alive so re-runs skip the build
    step; the weights are offloaded to CPU by ComfyUI when not in use.

    Attributes
    ----------
    model_path:
        Path to the base model checkpoint on disk.
    attn_backend_args:
        Optional attention backend config from ``BerniniR_AttentionConfig``.
    lora_specs:
        List of ``(lora_path, strength)`` to merge inline at load time.
    compile_cfg:
        Optional ``{"mode": ..., "fullgraph": ..., "dynamic_shapes": ...}``.
    """

    def __init__(
        self,
        model_path: str,
        attn_backend_args: dict | None = None,
        lora_specs: list[tuple[str, float]] | None = None,
        compile_cfg: dict | None = None,
        block_swap: bool = False,
    ):
        self.model_path = model_path
        self.attn_backend_args = attn_backend_args
        self.lora_specs = list(lora_specs) if lora_specs else []
        self.compile_cfg = compile_cfg
        self.block_swap = block_swap
        self._model_patcher: Any | None = None

        # Preview-compatible interface (no model load required):
        #   prepare_callback(model, steps)  needs:
        #     model.load_device          -> mm.get_torch_device()
        #     model.model.latent_format  -> Wan21()
        self.load_device = mm.get_torch_device()
        self.model = _LatentFormatProxy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, block_swap_config: Any = None) -> Any:
        """Load the model from disk, apply LoRAs/compile, and return patcher.

        *block_swap_config* is a ``BerniniBlockSwap`` dataclass (or ``None``
        to disable block swap).  When present, the model always goes through
        ``BlockSwapManager`` (ring buffer + ``SlotEntry``) for GPU placement,
        regardless of ``block_to_swap``.  When ``block_to_swap > 0``, extra
        blocks are offloaded to CPU/disk.
        """
        has_config = block_swap_config is not None
        need_swap = has_config and block_swap_config.block_to_swap > 0
        self.block_swap = need_swap
        block_swap = need_swap
        lazy = block_swap_config.lazy if need_swap else False

        # 1. Module-level cache — cross-execution reuse
        cache_key = _make_cache_key(
            self.model_path, self.lora_specs,
            self.compile_cfg, self.attn_backend_args,
            block_swap=self.block_swap,
        )
        cached: Any | None = _model_cache.get(cache_key)
        if cached is not None:
            _model_cache.move_to_end(cache_key)  # LRU bump
            self._model_patcher = cached
            logger.debug("Cache hit: %s", cache_key[:8])
            if need_swap:
                # BlockSwapManager owns GPU placement — keep on offload device.
                cached.load_device = cached.offload_device
                # Ensure _block_swap_config is set on the model (may be missing
                # if the cached patcher predates the unified config path).
                dm = getattr(cached.model, "diffusion_model", None)
                if dm is not None and not hasattr(dm, '_block_swap_config'):
                    dm._block_swap_config = block_swap_config
            else:
                mm.load_models_gpu([cached])
            return cached

        # 2. Guard: handle still holds a loaded patcher (not yet unloaded)
        if self._model_patcher is not None:
            if need_swap:
                self._model_patcher.load_device = self._model_patcher.offload_device
                dm = getattr(self._model_patcher.model, "diffusion_model", None)
                if dm is not None and not hasattr(dm, '_block_swap_config'):
                    dm._block_swap_config = block_swap_config
            else:
                mm.load_models_gpu([self._model_patcher])
            return self._model_patcher

        logger.info(
            "Loading model from disk: %s (loras=%d)",
            self.model_path,
            len(self.lora_specs),
        )

        # 1. Build model options (attention backend override).
        model_opts = self._build_model_options()

        # 2. In block-swap mode, create the RandomAccessBlockReader so the
        #    streaming loader skips block weights and defers to
        #    BlockSwapManager's _DiskPrefetcher for on-demand disk reads.
        #    Only needed when blocks are actually offloaded (block_to_swap > 0).
        _reader = None
        if has_config and block_swap_config.block_to_swap > 0:
            from ..utils.block_reader import RandomAccessBlockReader
            _reader = RandomAccessBlockReader(
                self.model_path,
                lora_specs=self.lora_specs or None,
            )

        # 3. Build the model.  For .safetensors we use a streaming loader that
        #    never holds the full state dict in RAM, so the load peak drops
        #    from ~2x model size to ~1x model size + one block group.
        from ..models.loader import load_bernini_model
        patcher = load_bernini_model(
            self.model_path,
            model_options=model_opts,
            state_dict=None,
            lora_specs=self.lora_specs or None,
            block_swap=need_swap,
            lazy=lazy,
            block_reader=_reader,
        )

        # 3. Set device targets.
        #    - Block swap config present: the model goes through
        #      BlockSwapManager (ring buffer + SlotEntry).  We set
        #      load_device == offload_device so ComfyUI's executor never
        #      hoists the whole model onto the GPU first (that would create
        #      a second full copy and race BlockSwapManager's transfers).
        #    - No config: standard ComfyUI behaviour (model on GPU).
        patcher.offload_device = mm.unet_offload_device()
        if need_swap:
            patcher.load_device = patcher.offload_device
        else:
            patcher.load_device = mm.get_torch_device()

        # Store the full block swap config on the diffusion model so the
        # forward pass can read it directly (only when block swap is active).
        dm = getattr(patcher.model, "diffusion_model", None)
        if dm is not None and need_swap:
            dm._block_swap_config = block_swap_config

        # 4. torch.compile if requested.
        if self.compile_cfg and self.compile_cfg.get("mode", "none") != "none":
            from ..models.wan_compile import compile_wan_model
            compile_wan_model(
                patcher,
                compile_mode=self.compile_cfg["mode"],
                fullgraph=self.compile_cfg.get("fullgraph", False),
                dynamic=self.compile_cfg.get("dynamic_shapes", True),
            )

        # 5. Store in module-level cache so re-runs skip disk + build
        _cache_put(cache_key, patcher)
        self._model_patcher = patcher
        return patcher

    def unload(self) -> None:
        """Fully unload the model from VRAM / RAM and drop the patcher.

        We always do a full release instead of caching the model on CPU:
        moving a large model to CPU before deletion leaves PyTorch's CPU
        allocator holding the RAM on Windows, which causes the next large
        model load to run out of memory.  This also ensures ComfyUI's own
        model manager removes the patcher from its loaded-model list.
        """
        if self._model_patcher is None:
            return

        logger.info("Unloading model from memory: %s", self.model_path)

        patcher = self._model_patcher

        # 0. Close block reader (disk fd) if in lazy mode.
        base_model = getattr(patcher, "model", None)
        dm = getattr(base_model, "diffusion_model", None)
        if dm is not None:
            _reader = getattr(dm, '_block_reader', None)
            if _reader is not None:
                try:
                    _reader.close()
                except Exception:
                    pass
                try:
                    delattr(dm, '_block_reader')
                except Exception:
                    pass

        # 1. Shrink all parameter tensors to zero-size before ComfyUI's unload.
        #    mm.unload_model_and_clones only offloads to CPU — the PyTorch CPU
        #    allocator never returns pages to the OS on Windows.
        from .vram import release_model_ram as _release_model_ram
        _release_model_ram(patcher)

        # 1b. Evict this patcher from the module-level cache so the next
        #     load() rebuilds a *fresh* model instead of reusing a patcher
        #     whose parameters were just shrunk to zero-size (1-D) tensors.
        #     Without this, load()'s cache hit returns the corpse and the
        #     forward pass crashes with "mat2 must be a matrix, got 1-D
        #     tensor" (e.g. text_embedding's F.linear).  The dual-expert path
        #     already evicts its high patcher the same way right after freeing.
        _cache_evict_patcher(patcher)

        # 1. Evict block-swap blocks before dropping the model so in-flight
        #    async transfers don't touch freed memory.  shutdown() cancels any
        #    prefetch on the transfer stream, synchronises *all* CUDA streams,
        #    and moves every block/peripheral back to CPU deterministically.
        base_model = getattr(patcher, "model", None)
        dm = getattr(base_model, "diffusion_model", None)
        mgr = getattr(dm, "_block_swap_mgr", None)
        if mgr is not None:
            try:
                mgr.shutdown()
            except Exception as e:
                logger.warning("BlockSwap shutdown failed: %s", e)
            try:
                delattr(dm, "_block_swap_mgr")
            except Exception:
                pass

        # 2. Tell ComfyUI's model manager to free this model and remove it
        #    from the loaded-model list.  Without this, ComfyUI keeps counting
        #    its VRAM as in-use and won't free memory for the next model.
        try:
            mm.unload_model_and_clones(patcher)
        except Exception as e:
            logger.warning("unload_model_and_clones failed: %s", e)

        # 3. Drop our reference.
        self._model_patcher = None
        del patcher

        # 4. Blocking CUDA cleanup.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        collect_garbage()

    def clone_with_lora(self, lora_path: str, strength: float) -> "BerniniRModelHandle":
        """Return a new handle with an additional LoRA spec.

        The actual LoRA weights are not touched until ``load()`` is called.
        """
        if strength == 0.0:
            return self
        new_specs = list(self.lora_specs)
        new_specs.append((lora_path, strength))
        return BerniniRModelHandle(
            model_path=self.model_path,
            attn_backend_args=self.attn_backend_args,
            lora_specs=new_specs,
            compile_cfg=self.compile_cfg,
            block_swap=self.block_swap,
        )

    def clone_with_compile(
        self,
        mode: str,
        fullgraph: bool,
        dynamic_shapes: bool,
    ) -> "BerniniRModelHandle":
        """Return a new handle with compile config stored (lazy, like LoRA)."""
        cfg = {"mode": mode, "fullgraph": fullgraph, "dynamic_shapes": dynamic_shapes}
        return BerniniRModelHandle(
            model_path=self.model_path,
            attn_backend_args=self.attn_backend_args,
            lora_specs=list(self.lora_specs),
            compile_cfg=cfg,
            block_swap=self.block_swap,
        )

    def is_loaded(self) -> bool:
        """Return True if the model is currently resident in memory."""
        return self._model_patcher is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model_options(self) -> dict:
        """Build the model_options dict passed to ``load_bernini_model``."""
        opts = {}
        if self.attn_backend_args is not None:
            from ..attention.backends import create_attention_override
            ov = create_attention_override(
                backend=self.attn_backend_args.backend,
                force_backend=self.attn_backend_args.force_backend,
            )
            opts.setdefault("transformer_options", {})[
                "optimized_attention_override"
            ] = ov
        return opts


def load_model_handle(
    model_path: str,
    attn_backend_args: dict | None = None,
    block_swap: bool = False,
) -> BerniniRModelHandle:
    """Factory for creating a lazy-loading Bernini-R model handle."""
    return BerniniRModelHandle(
        model_path=model_path,
        attn_backend_args=attn_backend_args,
        block_swap=block_swap,
    )