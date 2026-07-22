"""
Checkpoint loading for Bernini-R / Wan models.

Everything needed to turn a .safetensors / .pt file into a ModelPatcher:
  - ``_SafetensorsFileReader``: thread-safe positioned-read file reader
    (no memory-mapping, no full state dict in RAM)
  - ``_detect_model_config``: architecture & quantization detection
  - ``_build_bernini_base``: BaseModel + BerniniRWanModel construction
  - ``_load_bernini_model_safetensors_streaming``: pipelined group loader
  - ``load_bernini_model``: public entry point

Model *definition* lives in ``models.wan_model``; LoRA folding lives in
``utils.lora``; key normalisation lives in ``utils.keys``.
"""

from __future__ import annotations

import os

import torch

from ..utils.keys import _normalize_unet_key
from ..utils.lora import (
    _build_streaming_lora_groups,
    _collect_block_lora,
    _fold_group_loras,
    _fold_module_loras,
)
from .wan_model import BerniniRWanModel

from ..utils.log import get_logger as _get_logger

logger = _get_logger("Loader")


def _apply_model_options(patcher, model_options):
    """Merge model_options (transformer_options etc.) into ModelPatcher.

    ``_build_bernini_base`` only consumes ``dtype``/``device`` from
    ``model_options``.  This helper transfers everything else — the
    attention-backend override, NAG, TeaCache hooks — into the patcher
    so the model forward pass can see them.
    """
    if model_options and model_options.get("transformer_options"):
        patcher.model_options.setdefault("transformer_options", {}).update(
            model_options["transformer_options"])


def _build_bernini_base(
    unet_config: dict,
    model_options: dict,
    fp8: bool,
    quantization: str | None,
    parameters: int,
    weight_dtype,
    block_swap: bool = False,
):
    """Construct BaseModel + BerniniRWanModel and return (base, load_device, offload_device).

    When *block_swap* is True the model skeleton is allocated on the offload
    device (CPU) so it becomes the single source of truth for the weights.
    BlockSwapManager then windows a slice onto the GPU during the forward pass,
    keeping GPU + CPU holding exactly one copy of the model (never two).  When
    *block_swap* is False the skeleton is built on the compute device (GPU),
    the historical behaviour — one full copy resident on the GPU.
    """
    import comfy.model_base
    import comfy.model_management
    import comfy.latent_formats
    import comfy.utils

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    unet_dtype = model_options.get('dtype') or model_options.get('weight_dtype')
    if unet_dtype is None:
        unet_dtype = comfy.model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=supported_dtypes,
            weight_dtype=weight_dtype or comfy.utils.weight_dtype({}),
        )

    class _Cfg:
        latent_format = comfy.latent_formats.Wan21
        supported_inference_dtypes = supported_dtypes
        custom_operations = None
        quant_config = None
        manual_cast_dtype = None
        optimizations = {}
        sampling_settings = {"shift": 8.0}
        memory_usage_factor = 0.9
        def __init__(self, uc):
            self.unet_config = dict(uc)
            self.latent_format = self.latent_format()
            self.optimizations = self.optimizations.copy()
            self.sampling_settings = self.sampling_settings.copy()
        def set_inference_dtype(self, dt, m):
            self.manual_cast_dtype = m
            self.unet_config['dtype'] = dt
        def process_unet_state_dict(self, sd): return sd
        def process_unet_state_dict_for_saving(self, sd): return sd

    cfg = _Cfg(unet_config)
    if fp8:
        cfg.optimizations["fp8"] = True
    if quantization and 'scaled' in quantization:
        cfg.quant_config = {"mixed_ops": True}

    class _BerniniBaseModel(comfy.model_base.BaseModel):
        def extra_conds(self, **kw):
            out = super().extra_conds(**kw)
            cl = kw.get("context_latents")
            if cl is not None:
                import comfy.conds as _conds
                out['context_latents'] = _conds.CONDList(
                    [self.process_latent_in(l) for l in cl])
            return out

    load_device = model_options.get('load_device',
                                    comfy.model_management.get_torch_device())
    offload_device = model_options.get('offload_device',
                                       comfy.model_management.unet_offload_device())

    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        unet_dtype, load_device, supported_dtypes)
    cfg.set_inference_dtype(unet_dtype, manual_cast_dtype)

    # Block swap owns the GPU: build the skeleton on the offload device (CPU)
    # so it is the single copy of the weights.  Otherwise build on the compute
    # device (GPU) — the historical behaviour, one full copy on the GPU.
    build_device = offload_device if block_swap else load_device

    base = _BerniniBaseModel(
        cfg, model_type=comfy.model_base.ModelType.FLOW,
        device=build_device, unet_model=BerniniRWanModel)
    dm = base.diffusion_model
    dm._ntk_theta_stack = []

    if not comfy.model_management.is_device_cpu(offload_device):
        base.to(offload_device)

    return base, load_device, offload_device


# Fallback dtype mapping in case ``safetensors.torch._TYPES`` moves.
_SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E4M3FNUZ": getattr(torch, "float8_e4m3fnuz", None),
    "F8_E5M2": torch.float8_e5m2,
    "F8_E5M2FNUZ": getattr(torch, "float8_e5m2fnuz", None),
    "C64": torch.complex64,
    "U64": getattr(torch, "uint64", None),
    "U32": getattr(torch, "uint32", None),
    "U16": getattr(torch, "uint16", None),
}
_SAFETENSORS_DTYPE_MAP = {k: v for k, v in _SAFETENSORS_DTYPE_MAP.items() if v is not None}


class _SafetensorsFileReader:
    """File-I/O based safetensors reader that avoids memory-mapping.

    ``safetensors.safe_open`` memory-maps the whole file on Windows, which
    causes STATUS_ACCESS_VIOLATION when a second large model is loaded after
    the first one has been resident in RAM (e.g. dual-expert HIGH->LOW switch
    with block swap).  This reader keeps the file descriptor open and reads
    each tensor's bytes on demand, preserving the streaming / low-peak-memory
    behaviour.

    Reads use ``utils.block_reader._pread`` (position-independent), so
    ``get_tensor`` is thread-safe and multiple tensors may be read
    concurrently from a thread pool.
    """

    def __init__(self, path: str):
        try:
            from safetensors.torch import _TYPES
            self._dtype_map = _TYPES
        except Exception:
            self._dtype_map = _SAFETENSORS_DTYPE_MAP
        from ..utils.block_reader import _pread
        self._pread = _pread
        self._path = path
        self._fd = os.open(path, os.O_RDONLY | os.O_BINARY)
        import struct, json
        header_len = struct.unpack("<Q", os.read(self._fd, 8))[0]
        self._header = json.loads(os.read(self._fd, header_len))
        self._data_offset = 8 + header_len
        self._keys = [
            k for k, v in self._header.items()
            if isinstance(v, dict) and "dtype" in v
        ]

    def keys(self):
        return list(self._keys)

    def metadata(self) -> dict:
        """Return the safetensors ``__metadata__`` block (may be empty)."""
        return self._header.get("__metadata__", {}) or {}

    def get_slice(self, key: str):
        return _SafetensorsSlice(self._header[key]["shape"])

    def dtype_of(self, key: str):
        """Return the tensor dtype from the header only — no data read."""
        return self._dtype_map[self._header[key]["dtype"]]

    def _offset_of(self, key: str) -> int:
        """Absolute file offset of the tensor's data region."""
        return self._data_offset + self._header[key]["data_offsets"][0]

    def get_tensor(self, key: str):
        info = self._header[key]
        dtype = self._dtype_map[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        offset = self._data_offset + start
        nbytes = end - start
        # Read into a writable bytearray; ``torch.frombuffer`` keeps the
        # buffer alive for the tensor's lifetime, so no defensive clone is
        # needed (that clone used to double the per-tensor peak and added a
        # full checkpoint-sized memcpy to every load).
        buffer = bytearray(nbytes)
        view = memoryview(buffer)
        read = 0
        while read < nbytes:
            chunk = self._pread(
                self._fd, min(nbytes - read, 64 * 1024 * 1024), offset + read)
            if not chunk:
                raise EOFError(
                    f"Unexpected EOF reading tensor {key!r} "
                    f"({read}/{nbytes} bytes)"
                )
            view[read:read + len(chunk)] = chunk
            read += len(chunk)
        return torch.frombuffer(buffer, dtype=dtype).reshape(shape)

    def tensor_meta(self, key: str):
        """Return ``(nbytes, numel)`` for *key* from the header only.

        Reads nothing from the tensor data region — the byte count comes
        straight from the safetensors ``data_offsets`` and the element
        count from ``shape``.  Used by the lazy block-swap loader so it can
        record per-block VRAM/byte estimates without pulling the whole
        checkpoint through host RAM just to count bytes.
        """
        info = self._header[key]
        start, end = info["data_offsets"]
        nbytes = end - start
        numel = 1
        for s in info["shape"]:
            numel *= s
        return nbytes, numel

    def close(self) -> None:
        """Close the underlying file descriptor."""
        try:
            os.close(self._fd)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class _SafetensorsSlice:
    """Dummy slice returned by ``_SafetensorsFileReader.get_slice``."""

    def __init__(self, shape):
        self._shape = shape

    def get_shape(self):
        return self._shape


# ---------------------------------------------------------------------------
# Shared model-config detection (used by both streaming and full-dict paths)
# ---------------------------------------------------------------------------

def _detect_model_config(lookup_shape, *, keys=None, has_dtype=None):
    """Detect model architecture & quantization from a shape/dtype lookup.

    Parameters
    ----------
    lookup_shape:
        ``Callable[[str], tuple[int, ...]]`` — returns tensor shape for a
        canonical key (e.g. ``"blocks.0.self_attn.q.weight"``).
    keys:
        Optional iterable of canonical keys to scan for ``.weight_scale``
        suffixes (faster than probing one by one).  If ``None`` the function
        probes a few known keys.
    has_dtype:
        ``Callable[[str], torch.dtype | None]`` — returns tensor dtype for
        a key, or ``None`` if unavailable.  Used for fp8 detection.

    Returns
    -------
    dict
        ``unet_config`` ready to pass to ``_build_bernini_base``.
    str | None
        Quantization format string (e.g. ``"fp8_e4m3fn_scaled"``).
    torch.dtype | None
        Weight dtype (e.g. ``torch.float8_e4m3fn``).
    int
        Total parameter count (for VRAM estimation).
    """
    dim = lookup_shape("patch_embedding.weight")[0]
    num_heads = dim // 128
    in_dim = lookup_shape("patch_embedding.weight")[1]

    # Block indices → num_layers
    block_keys = [k for k in (keys or ()) if k.startswith("blocks.")]
    if block_keys:
        indices = {int(k.split(".")[1]) for k in block_keys
                   if k.split(".")[1].isdigit()}
        num_layers = max(indices) + 1
    else:
        num_layers = 30  # fallback

    ffn_dim = lookup_shape("blocks.0.ffn.0.weight")[0]

    try:
        out_dim = lookup_shape("head.head.weight")[0] // 4
    except (KeyError, IndexError):
        out_dim = 16  # fallback

    # Variant
    if dim == 5120:
        model_variant = "14B"
    elif dim == 3072:
        model_variant = "5B"
    elif dim == 1536:
        model_variant = "1_3B"
    else:
        model_variant = "unknown"

    # Quantization detection
    quantization = None
    weight_dtype = None
    is_scaled_fp8 = False
    if keys:
        is_scaled_fp8 = any(
            k.endswith((".scale_weight", ".weight_scale", ".weight_scale_2"))
            for k in keys
        )
        if is_scaled_fp8:
            for k in keys:
                if k.endswith(".weight_scale_2"):
                    quantization = "nvfp4"
                    break

    if has_dtype is not None:
        for probe in ("head.modulation", "time_projection.0.weight",
                      "time_embedding.0.weight", "blocks.0.self_attn.q.weight"):
            dt = has_dtype(probe)
            if dt is None:
                continue
            weight_dtype = dt
            if dt in (torch.float8_e4m3fn, torch.float8_e5m2):
                quantization = "fp8_e4m3fn" if dt == torch.float8_e4m3fn else "fp8_e5m2"
                break

    if is_scaled_fp8 and quantization:
        quantization += "_scaled"

    # Total parameters
    parameters = 0
    if keys:
        for k in keys:
            s = lookup_shape(k)
            if len(s) >= 2:
                parameters += s[0] * s[1]
            elif len(s) == 1:
                parameters += s[0]
            # scalars (0-dim, shape=()) contribute 1 element

    unet_config = {
        "dim": dim, "out_dim": out_dim, "num_heads": num_heads,
        "ffn_dim": ffn_dim, "num_layers": num_layers,
        "patch_size": (1, 2, 2), "freq_dim": 256, "in_dim": in_dim,
        "qk_norm": True, "cross_attn_norm": True, "eps": 1e-6,
        "window_size": (-1, -1), "text_dim": 4096,
        "model_variant": model_variant,
    }
    return unet_config, quantization, weight_dtype, parameters, num_layers


def _load_bernini_model_safetensors_streaming(
    model_path: str,
    model_options: dict,
    lora_specs: list | None,
    block_swap: bool = False,
    lazy: bool = False,
    block_reader: object | None = None,
) -> object:
    """Memory-efficient loader for safetensors checkpoints.

    Instead of loading the full state dict into RAM and then copying it into
    the model, we read one block group at a time.  Group N+1 is read from
    disk while group N is being copied into the model, so peak host RAM is
    roughly the model size plus two block groups (vs ~2x the model size for
    the full-dict path).
    """
    import comfy.model_patcher
    import comfy.utils
    from concurrent.futures import ThreadPoolExecutor

    if model_options is None:
        model_options = {}

    with _SafetensorsFileReader(model_path) as f:
        raw_keys = list(f.keys())

        # Canonical (model-state-dict) key -> raw safetensors key
        norm_map = {}
        for k in raw_keys:
            nk = _normalize_unet_key(k)
            norm_map.setdefault(nk, k)

        def _shape(key: str):
            return tuple(f.get_slice(key).get_shape())

        # ── Config detection via shared helper ─────────────────────────
        # dtype probes read the header only — the old path pulled entire
        # tensors from disk just to inspect their dtype.
        _canonical_keys = list(norm_map.keys())
        def _stream_has_dtype(k):
            rk = norm_map.get(k)
            if rk is None:
                return None
            try:
                return f.dtype_of(rk)
            except Exception:
                return None

        unet_config, quantization, weight_dtype_val, parameters, num_layers = \
            _detect_model_config(
                lambda k: _shape(norm_map[k]),
                keys=_canonical_keys,
                has_dtype=_stream_has_dtype,
            )

        # Build model
        fp8 = quantization is not None and 'fp8' in quantization
        base, load_device, offload_device = _build_bernini_base(
            unet_config, model_options, fp8, quantization,
            parameters=parameters, weight_dtype=weight_dtype_val,
            block_swap=block_swap,
        )
        dm = base.diffusion_model

        # Diffusers/ModelOpt-style ``_quantization_metadata`` (per-layer
        # ``{"format": "nvfp4"}`` JSON).  Native ComfyUI converts this into
        # per-layer ``comfy_quant`` tensors inside load_torch_file; our
        # streaming reader bypasses that, so we parse it here and let
        # ``_stream_load_group`` inject the entries per group.  Checkpoints
        # with native comfy_quant tensors (fp8_scaled) skip this entirely.
        import json as _json
        _qm_raw = f.metadata().get("_quantization_metadata")
        if _qm_raw:
            try:
                _qm = _json.loads(_qm_raw)
                dm._quant_layer_conf = {
                    _normalize_unet_key(k): v
                    for k, v in _qm.get("layers", {}).items()
                }
                logger.info("Quant metadata: %d quantized layer(s)",
                            len(dm._quant_layer_conf))
            except Exception as e:
                logger.warning("Failed to parse _quantization_metadata: %s", e)

        # In streaming block-swap mode the transformer-block ops have no ``weight``
        # attribute yet (it is deferred to load_state_dict).  Register a plain
        # ``None`` weight slot on each of them so ComfyUI's ModelPatcher load walk
        # (partially_load -> _load_list -> get_key_weight) does not raise
        # ``'Linear' object has no attribute 'weight'``.  No RAM is allocated; the
        # real fp8 weights are filled shortly after by the streaming pass below.
        if block_swap:
            _materialize_block_weight_slots(dm)

        # In lazy mode, a RandomAccessBlockReader is provided to the
        # BlockSwapManager so it can load blocks from disk on demand.
        # Only peripheral modules are loaded now; block weights stay empty.
        if lazy and block_reader is not None:
            dm._block_reader = block_reader

        # Pre-load / prepare LoRAs so they can be folded block-by-block.
        # Resident mode: load all LoRA into a RAM dict (aligned with the model
        # weights already in CPU RAM).  Lazy mode: build a LoraBlockReader that
        # reads block LoRA from disk on demand — no RAM pool.  Non-block keys
        # (patch_embedding, head, norms) are read at startup for both modes.
        lora_groups: dict = {}
        if lora_specs:
            if lazy and block_swap:
                from ..utils.block_reader import LoraBlockReader
                dm._lora_reader = LoraBlockReader(lora_specs)
                non_block_lora = dm._lora_reader.read_non_block()
                if non_block_lora:
                    lora_groups = non_block_lora
            else:
                lora_groups = _build_streaming_lora_groups(lora_specs)
                dm._lora_groups = lora_groups

        # Per-block metadata for block-swap VRAM estimates.
        block_plan: dict = {}
        block_bytes: dict = {}
        block_mb: dict = {}

        # Group plan from the header only — same grouping as the old
        # sorted-key loop: "blocks.N" groups plus first-component groups,
        # in sorted raw-key order.
        seen_keys = set()
        group_plan: list = []
        for raw_key in sorted(raw_keys):
            target_key = _normalize_unet_key(raw_key)
            seen_keys.add(target_key)
            parts = target_key.split('.')
            if len(parts) >= 2 and parts[0] == 'blocks':
                group_key = f"{parts[0]}.{parts[1]}"
            else:
                group_key = parts[0] if parts else raw_key
            if group_plan and group_plan[-1][0] == group_key:
                group_plan[-1][1].append((target_key, raw_key))
            else:
                group_plan.append((group_key, [(target_key, raw_key)]))

        # Each group's tensors are read in file-offset order by a 4-thread
        # pool; a 1-thread prefetcher reads group N+1 while group N is being
        # copied into the model, overlapping disk I/O with the load copies.
        io_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="bernini-io")
        prefetch_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bernini-prefetch")

        def _read_group(group_key, entries):
            # In lazy block-swap mode the block weights are NOT loaded here
            # (the _DiskPrefetcher reads them on demand during sampling).
            # Only per-block byte/param counts are needed for the VRAM
            # estimate, taken from the header instead of reading tensors —
            # this avoids a pointless full-checkpoint disk read.
            if lazy and block_swap and group_key.startswith("blocks."):
                return {t: f.tensor_meta(r) for t, r in entries}
            ordered = sorted(entries, key=lambda e: f._offset_of(e[1]))
            futs = [io_pool.submit(f.get_tensor, r) for _, r in ordered]
            return {t: fu.result() for (t, _), fu in zip(ordered, futs)}

        folded_total = [0]  # plain-mode GPU fold count (visibility)

        def _load_group(group_key, group):
            if group_key.startswith("blocks."):
                if lazy:
                    # Lazy mode: metadata only — do NOT load block weights
                    # into RAM.  BlockSwapManager's _DiskPrefetcher loads
                    # them on demand.
                    if block_swap:
                        _record_block_meta(
                            group_key, group, block_plan, block_bytes, block_mb)
                else:
                    # Always load raw weights here; LoRA folding is done
                    # per-mode below.  Passing None to _stream_load_group in
                    # ALL modes avoids any CPU-side fp32 fold pass.
                    _stream_load_group(dm, group_key, group, None)
                    if block_swap:
                        _record_block_meta(
                            group_key, group, block_plan, block_bytes, block_mb)
                        # Co-locate this block's LoRA with its slot
                        # (unified (block, lora) → (block) design); folded
                        # later on the GPU ring — never here, that would
                        # double-apply.
                        bidx = int(group_key.split('.')[1])
                        blk = _collect_block_lora(lora_groups, bidx)
                        if blk:
                            dm.blocks[bidx]._lora_payload = blk
                    else:
                        # Plain mode (no block swap): the module is
                        # GPU-resident, so fold LoRA on GPU right after the
                        # weights land — nothing reads dm._lora_groups, and
                        # without this block LoRAs were silently dropped.
                        try:
                            bidx = int(group_key.split('.')[1])
                        except (IndexError, ValueError):
                            bidx = None
                        if bidx is not None:
                            folded_total[0] += _fold_module_loras(
                                dm.blocks[bidx], group_key + ".", lora_groups)
            else:
                _stream_load_group(dm, group_key, group, lora_groups)
            group.clear()

        try:
            pending = None
            if group_plan:
                pending = prefetch_pool.submit(_read_group, *group_plan[0])
            for gi, (group_key, _) in enumerate(group_plan):
                group = pending.result()
                if gi + 1 < len(group_plan):
                    pending = prefetch_pool.submit(
                        _read_group, *group_plan[gi + 1])
                _load_group(group_key, group)
        finally:
            prefetch_pool.shutdown(wait=False)
            io_pool.shutdown(wait=True)

        if folded_total[0]:
            logger.info("Folded LoRA into %d block weights (GPU)",
                        folded_total[0])

    for missing_base in set(lora_groups) - seen_keys:
        logger.warning(
            "LoRA base key not found in model state dict: %s",
            missing_base,
        )

    # The global LoRA pool is no longer needed in resident mode: every block's
    # LoRA has been co-located into its slot (or already folded into non-block
    # params, e.g. embeddings/norms), so dropping the pool frees the 1.5GB.
    # Lazy mode never builds a RAM pool — LoraBlockReader reads per-block
    # from disk on demand.
    if not lazy and block_swap:
        lora_groups.clear()
        dm._lora_groups = None

    # In lazy mode, block weights are NOT loaded into CPU RAM yet.
    # BlockSwapManager's _DiskPrefetcher reads them from disk on demand
    # when prepare() is called.  Only VRAM metadata is stored.
    if block_swap:
        avg_mb = sum(block_mb.values()) / len(block_mb) if block_mb else 0.0
        dm._block_meta = {'block_mb': dict(block_mb), 'avg_mb': avg_mb}
        if lazy:
            # No pre-warm — BlockSwapManager loads first window on first
            # prepare(0) call via _DiskPrefetcher.
            dm._prewarmed = 0
            logger.info("Lazy mode: %d blocks, metadata only, no pre-warm",
                        num_layers)
        else:
            DEFAULT_WINDOW = 10
            warm_blocks = min(DEFAULT_WINDOW, num_layers)
            for idx in range(warm_blocks):
                dm.blocks[idx].to(load_device)
            dm._prewarmed = warm_blocks
            logger.info("Pre-warmed block window: %d / %d blocks to GPU",
                        warm_blocks, num_layers)

    mp = comfy.model_patcher.ModelPatcher(
        base, load_device=load_device, offload_device=offload_device)

    _apply_model_options(mp, model_options)

    # In lazy mode, block weights are NOT loaded yet — skip per-block
    # verification and only check peripheral modules.
    if lazy:
        _verify_weights_loaded(
            dm, skip_blocks=True,
            warning_msg="lazy-load — blocks will be loaded on demand",
        )
    else:
        _verify_weights_loaded(dm, skip_blocks=False)

    mode = "lazy" if lazy else "eager"
    logger.info("Stream-loaded (%(mode)s): dim=%(dim)d heads=%(num_heads)d "
                "layers=%(num_layers)d ffn=%(ffn_dim)d "
                "variant=%(model_variant)s quant=%(quant)s",
                {"mode": mode,
                 "dim": unet_config["dim"],
                 "num_heads": unet_config["num_heads"],
                 "num_layers": unet_config["num_layers"],
                 "ffn_dim": unet_config["ffn_dim"],
                 "model_variant": unet_config["model_variant"],
                 "quant": quantization or "none"})
    return mp


def _record_block_meta(current_group, group, block_plan, block_bytes, block_mb):
    """Record per-block key lists and byte counts from already-loaded tensors.

    Unlike the old ``_finalize_block_meta`` (which was called in place of
    loading so the tensors were dropped immediately), this is called AFTER
    ``_stream_load_group`` has loaded the block into the CPU model.  The
    tensors in *group* are still live and only used for counting bytes.
    """
    try:
        idx = int(current_group.split(".")[1])
    except (IndexError, ValueError):
        return
    block_plan[idx] = list(group.keys())
    nb = 0
    nv = 0
    for t in group.values():
        if isinstance(t, tuple):
            # Lazy mode: (nbytes, numel) from the header — no tensor read.
            nbytes, numel = t
            nb += nbytes
            nv += numel * 2  # VRAM budget estimated as half precision
        else:
            nb += t.numel() * t.element_size()
            nv += t.numel() * 2  # VRAM budget estimated as half precision
    block_bytes[idx] = nb
    block_mb[idx] = nv / (1024 * 1024)


def _fixup_quant_orig_dtype(module, dtype=None) -> None:
    """Default ``Params.orig_dtype`` on quantized params when it is None.

    ComfyUI sets ``orig_dtype`` from the compute dtype when materialising a
    QuantizedTensor (ops.py:1125); when the model needs no manual cast (e.g.
    a bf16 checkpoint on a bf16-capable GPU) it stays ``None`` — which crashes
    nvfp4's dequantize (``DTYPE_TO_CODE[None]``) and the matmul output-dtype
    default (``a._params.orig_dtype``).  Fall back to the model's dtype
    (bf16 for Bernini-R/Wan).
    """
    if dtype is None:
        dtype = torch.bfloat16
    for p in module.parameters():
        # The Parameter itself IS the QuantizedTensor — mutate ``p._params``
        # (canonical).  ``p.data`` returns an ephemeral wrapper whose params
        # are throwaway copies, so writes there are silently lost.
        params = getattr(p, "_params", None)
        if params is None:
            d = getattr(p, "data", None)
            params = getattr(d, "_params", None) if hasattr(d, "_qdata") else None
        if params is not None and getattr(params, "orig_dtype", None) is None:
            # Params is a frozen dataclass — bypass with object.__setattr__.
            object.__setattr__(params, "orig_dtype", dtype)


def _stream_load_group(dm, group_key, group, lora_groups=None):
    """Load one streamed group into the correct submodule.

    Critical: target the submodule directly, NOT the whole ``dm``.

    ``mixed_precision_ops.Linear`` registers ``weight`` lazily and, inside
    ``_load_quantized_module``, sets ``module.weight = None`` whenever the
    filtered sub-state-dict for that call lacks the ``weight`` key.  Flushing
    ``dm.load_state_dict(group)`` per block visits *every* block (and head);
    blocks absent from the current group get an empty sub-dict and are wiped
    to ``None`` -- and the final group (e.g. ``time_projection``) wipes them all,
    producing the "Weight not loaded" errors.  Loading the submodule with the
    group prefix stripped keeps each flush local to its intended target.

    LoRA folding happens here (not in the per-key loop) so that the per-layer
    ``weight_scale`` is available for fp8_scaled weights: the stored fp8 value
    is dequantized (``stored * scale``), the deltas are folded in float32, then
    the result is re-quantized back to fp8 with a fresh ``weight_scale``.
    """
    prefix = group_key + "."
    if group_key.startswith("blocks."):
        try:
            idx = int(group_key.split(".")[1])
        except (IndexError, ValueError):
            with torch.inference_mode():
                dm.load_state_dict(group, strict=False, assign=False)
            return
        sub = dm.blocks[idx]
    elif hasattr(dm, group_key):
        sub = getattr(dm, group_key)
    else:
        # Unknown top-level key — fall back to a full-model load.
        with torch.inference_mode():
            dm.load_state_dict(group, strict=False, assign=False)
        return
    sub_group = {k[len(prefix):]: v for k, v in group.items() if k.startswith(prefix)}

    # Inject per-layer ``comfy_quant`` metadata synthesised from the
    # checkpoint's ``_quantization_metadata`` (see loader).  Without these,
    # mixed_precision_ops cannot know a layer is nvfp4/mxfp8 and would load
    # the packed quantized bytes as a plain tensor.
    layer_conf = getattr(dm, "_quant_layer_conf", None)
    if layer_conf:
        import json as _json
        for layer, lconf in layer_conf.items():
            if layer.startswith(prefix):
                local = layer[len(prefix):]
                if local:
                    key = local + ".comfy_quant"
                    if key not in sub_group:
                        sub_group[key] = torch.tensor(
                            list(_json.dumps(lconf).encode("utf-8")),
                            dtype=torch.uint8)
            elif layer == group_key:
                if "comfy_quant" not in sub_group:
                    sub_group["comfy_quant"] = torch.tensor(
                        list(_json.dumps(lconf).encode("utf-8")),
                        dtype=torch.uint8)

    # Fold any LoRAs targeting this group now that weight_scale is in scope.
    _fold_group_loras(sub_group, prefix, lora_groups)

    # Block params are inference tensors (model built under InferenceMode).
    # load_state_dict(assign=False) does an in-place copy_ which is forbidden
    # outside InferenceMode -> wrap the copy so it is allowed while keeping the
    # params as inference tensors.
    with torch.inference_mode():
        sub.load_state_dict(sub_group, strict=False, assign=False)
    _fixup_quant_orig_dtype(sub, getattr(dm, "dtype", None))


def _materialize_block_weight_slots(dm):
    """Register a plain ``None`` ``weight`` (and ``bias``) attribute on every
    transformer-block op that was built without one.

    In streaming block-swap mode the block weights are NOT loaded at build
    time -- ``comfy.ops.mixed_precision_ops.Linear`` (fp8) and the
    ``disable_weight_init`` family defer weight creation to ``load_state_dict``
    and leave ``self.weight`` undefined until then.  ComfyUI's
    ``ModelPatcher.load`` -> ``partially_load`` -> ``_load_list`` walks *every*
    module and calls ``get_key_weight(op, "X.weight")`` which does
    ``getattr(op, "weight")`` and raises
    ``AttributeError: 'Linear' object has no attribute 'weight'``.

    Setting a plain ``None`` attribute makes ``get_key_weight`` return ``None``
    (-> 0 offload memory, no crash) without allocating host RAM and without
    turning ``weight`` into a registered Parameter (which would change the
    ``load_state_dict`` path that fills the real fp8 weights on demand via
    the block reader (``RandomAccessBlockReader`` / ``_DiskPrefetcher``).  The real
    weights are still materialised later, on first GPU move.
    """
    for block in getattr(dm, "blocks", []):
        for m in block.modules():
            # get_key_weight (via _load_list) is only ever called for modules
            # that advertise comfy_cast_weights (the quant / manual_cast ops).
            # Those are exactly the ops whose weight is deferred, so only patch
            # those -- never touch container modules like WanAttentionBlock.
            if not hasattr(m, "comfy_cast_weights"):
                continue
            try:
                m.weight
            except AttributeError:
                try:
                    object.__setattr__(m, "weight", None)
                except Exception:
                    pass
            try:
                m.bias
            except AttributeError:
                try:
                    object.__setattr__(m, "bias", None)
                except Exception:
                    pass


def _verify_weights_loaded(dm, skip_blocks: bool = False, warning_msg: str = ""):
    """Warn if any ComfyUI ops layer still has a None weight after loading.

    When *skip_blocks* is set (streaming block-swap / lazy), transformer blocks
    are intentionally not resident at load time -- they are read from disk on
    demand -- so only peripheral / top-level modules are checked.
    """
    for name, module in dm.named_modules():
        if not hasattr(module, 'weight') or not hasattr(module, 'comfy_cast_weights'):
            continue
        if isinstance(module, torch.nn.LayerNorm) and not module.elementwise_affine:
            continue
        if isinstance(module, torch.nn.RMSNorm) and not getattr(module, 'elementwise_affine', True):
            continue
        if skip_blocks and name.startswith("blocks."):
            continue
        if module.weight is None:
            suffix = f" ({warning_msg})" if warning_msg else ""
            logger.error(
                "Weight not loaded: %s — state dict key mismatch?%s",
                name, suffix)


def load_bernini_model(model_path, model_options=None, state_dict=None, lora_specs=None, block_swap: bool = False, lazy: bool = False, block_reader=None) -> object:
    """Load a Bernini-R / Wan checkpoint.  Fully self-contained —
    no ``model_detection`` / ``supported_models`` dependency.

    For ``.safetensors`` files this now uses a streaming loader that avoids
    holding the full state dict in RAM.  ``.pt`` / ``.ckpt`` files still fall
    back to the full-dict path.

    When *lazy* is True, transformer block weights are NOT loaded into CPU RAM
    during this call.  Instead, a ``RandomAccessBlockReader`` is stored on the
    model for on-demand block loading by ``BlockSwapManager``'s
    ``_DiskPrefetcher``.
    """
    import comfy.model_patcher
    import comfy.model_management
    import comfy.utils

    if model_options is None:
        model_options = {}

    if state_dict is not None:
        sd = state_dict
    else:
        lower_path = model_path.lower()
        if lower_path.endswith(".safetensors") or lower_path.endswith(".sft"):
            return _load_bernini_model_safetensors_streaming(
                model_path, model_options, lora_specs, block_swap,
                lazy=lazy, block_reader=block_reader)
        sd = comfy.utils.load_torch_file(model_path)

    if lora_specs and state_dict is None:
        from ..utils.lora import apply_loras_to_state_dict
        sd = apply_loras_to_state_dict(sd, lora_specs)

    # Prefix normalisation
    first_key = next(iter(sd)) if sd else ""
    if first_key.startswith("model.diffusion_model."):
        sd = {k.replace("model.diffusion_model.", "", 1): v
              for k, v in sd.items()}
    elif first_key.startswith("model."):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    elif first_key.startswith("video_model."):
        sd = {k.replace("video_model.", "", 1)
               .replace("modulation.modulation", "modulation"): v
              for k, v in sd.items()}

    # Config detection via shared helper
    def _sd_shape(k):
        return tuple(sd[k].shape)
    def _sd_has_dtype(k):
        t = sd.get(k)
        return t.dtype if t is not None else None
    canon_keys = [k for k in sd if isinstance(sd.get(k), torch.Tensor)]

    unet_config, quantization, weight_dtype, parameters, num_layers = \
        _detect_model_config(
            _sd_shape, keys=canon_keys, has_dtype=_sd_has_dtype,
        )
    fp8 = quantization is not None and 'fp8' in quantization

    base, load_device, offload_device = _build_bernini_base(
        unet_config, model_options, fp8, quantization,
        parameters=parameters,
        weight_dtype=weight_dtype or comfy.utils.weight_dtype(sd),
        block_swap=block_swap,
    )
    dm = base.diffusion_model

    base.load_model_weights(sd, "", assign=False)

    mp = comfy.model_patcher.ModelPatcher(
        base, load_device=load_device, offload_device=offload_device)

    _apply_model_options(mp, model_options)

    _verify_weights_loaded(dm)

    logger.info("Loaded: dim=%(dim)d heads=%(num_heads)d "
                "layers=%(num_layers)d ffn=%(ffn_dim)d "
                "variant=%(model_variant)s quant=%(quant)s",
                {"dim": unet_config["dim"], "num_heads": unet_config["num_heads"],
                 "num_layers": unet_config["num_layers"],
                 "ffn_dim": unet_config["ffn_dim"],
                 "model_variant": unet_config["model_variant"],
                 "quant": quantization or "none"})
    return mp