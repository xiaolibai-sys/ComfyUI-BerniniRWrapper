"""
Self-contained WanModel for Bernini-R.

Copied from ComfyUI's ``comfy/ldm/wan/model.py`` and stripped of
unused extensions (VACE, audio, I2V, control, etc.).  All dependencies
(rope, EmbedND, pad_to_patch_size) are inlined so no ComfyUI internals
are required for inference — only ``optimized_attention`` is imported
as a stable API.

Modifications vs upstream:
  - forward / pre_forward:  ``_rope_t_start`` and ``_rope_ntk_scale`` kwargs
  - pre_forward:            source_id-aware t_start (multi-frame vs single-frame)
  - WanT2VCrossAttention:   NAG (Normalized Attention Guidance) built-in
  - WrapperExecutor:        bypassed — ``forward`` calls ``pre_forward`` directly
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

# ── stable ComfyUI API (the ONLY import) ─────────────────────────────
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope1  # use upstream CUDA kernel for bit-identical RoPE

from ..utils.log import get_logger as _get_logger

logger = _get_logger("Model")
# ═══════════════════════════════════════════════════════════════════════
# Inlined dependencies (no ComfyUI internals)
# ═══════════════════════════════════════════════════════════════════════

def _rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """RoPE rotation matrices.  Copied from comfy.ldm.flux.math."""
    assert dim % 2 == 0
    # Compute on CUDA only when the input is *already* on a capable CUDA
    # device (sm_70+).  With block-swap the model skeleton (and thus ``pos``)
    # lives on CPU, so we must never hand a CPU tensor to
    # ``torch.cuda.get_device_capability`` — that raises
    # ``ValueError: Expected a cuda device, but got: cpu``.  The rotation is
    # device-agnostic math, so computing on CPU is bit-identical to GPU and
    # only matters for the (small) precompute speed.
    if pos.device.type == "cuda" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(pos.device)[0]
        device = pos.device if cap >= 7 else torch.device("cpu")
    else:
        device = torch.device("cpu")
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2,
                           dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd",
                       pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out),
                       torch.sin(out),  torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def _apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a single tensor.  Copied from comfy.ldm.flux.math."""
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 \
            and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


class EmbedND(nn.Module):
    """3D RoPE embedding.  Copied from comfy.ldm.flux.layers."""

    def __init__(self, dim: int, theta: float, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [_rope(ids[..., i], self.axes_dim[i], self.theta)
             for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def _pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    """Pad to patch multiple.  Copied from comfy.ldm.common_dit."""
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i])
               % patch_size[i]) + pad
    return torch.nn.functional.pad(img, pad, mode=padding_mode)


def _cast_to(tensor, *, dtype=None, device=None):
    """Lightweight replacement for comfy.model_management.cast_to."""
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    return tensor


# ═══════════════════════════════════════════════════════════════════════
# Sinusoidal embedding (upstream, unchanged)
# ═══════════════════════════════════════════════════════════════════════

def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


# ═══════════════════════════════════════════════════════════════════════
# Attention blocks (upstream, unchanged)
# ═══════════════════════════════════════════════════════════════════════

class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, eps=1e-6, kv_dim=None, operation_settings=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        kv_dim = kv_dim or dim
        ops = operation_settings or {}
        self.q = ops.get("operations").Linear(dim, dim, device=ops.get("device"), dtype=ops.get("dtype"))
        self.k = ops.get("operations").Linear(kv_dim, dim, device=ops.get("device"), dtype=ops.get("dtype"))
        self.v = ops.get("operations").Linear(kv_dim, dim, device=ops.get("device"), dtype=ops.get("dtype"))
        self.o = ops.get("operations").Linear(dim, dim, device=ops.get("device"), dtype=ops.get("dtype"))
        self.norm_q = ops.get("operations").RMSNorm(dim, eps=eps, elementwise_affine=True, device=ops.get("device"), dtype=ops.get("dtype")) if qk_norm else nn.Identity()
        self.norm_k = ops.get("operations").RMSNorm(dim, eps=eps, elementwise_affine=True, device=ops.get("device"), dtype=ops.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, freqs, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        patches = transformer_options.get("patches", {})
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def _qkv_q(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            return apply_rope1(q, freqs)

        def _qkv_k(x):
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            return apply_rope1(k, freqs)

        q = _qkv_q(x)
        k = _qkv_k(x)

        x = optimized_attention(
            q.view(b, s, n * d), k.view(b, s, n * d),
            self.v(x).view(b, s, n * d),
            heads=self.num_heads, transformer_options=transformer_options)

        if "attn1_patch" in patches:
            for p in patches["attn1_patch"]:
                x = p({"x": x, "q": q, "k": k, "transformer_options": transformer_options})
        return self.o(x)


class WanT2VCrossAttention(WanSelfAttention):
    """Cross-attention for text conditioning.

    Supports NAG (Normalized Attention Guidance) with dual-attention
    path, falling back to standard broadcast-context cross-attention.
    """

    def forward(self, x, context, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        nag_params = transformer_options.get("nag_params")
        nag_context = transformer_options.get("nag_context")

        # ── NAG dual-attention path ─────────────────────────────────────
        if nag_params and nag_context is not None:
            q = self.norm_q(self.q(x))
            k_pos = self.norm_k(self.k(context))
            v_pos = self.v(context)
            x_pos = optimized_attention(
                q, k_pos, v_pos, heads=self.num_heads,
                transformer_options=transformer_options)
            del k_pos, v_pos
            k_neg = self.norm_k(self.k(nag_context))
            v_neg = self.v(nag_context)
            x_neg = optimized_attention(
                q, k_neg, v_neg, heads=self.num_heads,
                transformer_options=transformer_options)
            del k_neg, v_neg, q
            x_pos = x_pos.flatten(2) if x_pos.dim() > 3 else x_pos
            x_neg = x_neg.flatten(2) if x_neg.dim() > 3 else x_neg
            x = _apply_nag(x_pos, x_neg, nag_params)
            del x_pos, x_neg
            x = self.o(x)
            return x
        else:
            # ── Standard broadcast path ──────────────────────────────────
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(context))
            v = self.v(context)
            x = optimized_attention(q, k, v, heads=self.num_heads,
                                    transformer_options=transformer_options)
            x = self.o(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
# NAG helper
# ═══════════════════════════════════════════════════════════════════════

def _apply_nag(x_pos, x_neg, nag_params):
    scale = nag_params['nag_scale']
    alpha = nag_params['nag_alpha']
    tau = nag_params['nag_tau']
    inplace = nag_params.get('inplace', True)
    if inplace:
        guided = x_neg.mul_(scale - 1).neg_().add_(x_pos, alpha=scale)
    else:
        guided = x_pos * scale - x_neg * (scale - 1)
    del x_neg
    norm_pos = torch.norm(x_pos, p=1, dim=-1, keepdim=True)
    norm_guided = torch.norm(guided, p=1, dim=-1, keepdim=True)
    # Guard against near-zero norms (all-zero or degenerate input).
    # When norm_pos ≈ 0, the ratio/clamp logic would zero the output;
    # instead skip the norm-clamping step entirely.
    eps = 1e-7
    ratio = norm_guided / (norm_pos + eps)
    ratio = torch.where(norm_pos < eps, torch.zeros_like(ratio), ratio)
    torch.nan_to_num_(ratio, nan=10.0)
    mask = ratio > tau
    del ratio
    adjustment = (norm_pos * tau) / (norm_guided + eps)
    del norm_pos, norm_guided
    guided.mul_(torch.where(mask, adjustment, 1.0))
    del mask, adjustment
    if inplace:
        guided.sub_(x_pos).mul_(alpha).add_(x_pos)
    else:
        guided = guided * alpha + x_pos * (1 - alpha)
    del x_pos
    return guided


# ═══════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════

def _repeat_e(e, x):
    repeats = 1
    if e.size(1) > 1:
        repeats = x.size(1) // e.size(1)
    if repeats == 1:
        return e
    if repeats * e.size(1) == x.size(1):
        return torch.repeat_interleave(e, repeats, dim=1)
    return torch.repeat_interleave(e, repeats + 1, dim=1)[:, :x.size(1)]


# ═══════════════════════════════════════════════════════════════════════
# WanAttentionBlock (upstream, unchanged except cast_to → _cast_to)
# ═══════════════════════════════════════════════════════════════════════

class WanAttentionBlock(nn.Module):

    def __init__(self, dim, ffn_dim, num_heads,
                 window_size=(-1, -1), qk_norm=True, cross_attn_norm=False,
                 eps=1e-6, operation_settings=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        ops = operation_settings or {}
        self.norm1 = ops.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=ops.get("device"), dtype=ops.get("dtype"))
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)
        self.norm3 = ops.get("operations").LayerNorm(dim, eps, elementwise_affine=True, device=ops.get("device"), dtype=ops.get("dtype")) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, operation_settings=operation_settings)
        self.norm2 = ops.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=ops.get("device"), dtype=ops.get("dtype"))
        self.ffn = nn.Sequential(
            ops.get("operations").Linear(dim, ffn_dim, device=ops.get("device"), dtype=ops.get("dtype")),
            nn.GELU(approximate='tanh'),
            ops.get("operations").Linear(ffn_dim, dim, device=ops.get("device"), dtype=ops.get("dtype")))
        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=ops.get("device"), dtype=ops.get("dtype")))

    def forward(self, x, e, freqs, context, context_img_len=257,
                transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        patches = transformer_options.get("patches", {})

        if e.ndim < 4:
            e = (_cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        else:
            e = (_cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)

        # ── STG: skip self-attention in specified blocks ──────────
        stg_skip = transformer_options.get("_stg_skip_attn", False)
        stg_blocks = transformer_options.get("_stg_blocks", ())
        stg_mode = transformer_options.get("_stg_mode", "A")
        block_idx = transformer_options.get("block_index", -1)
        # Clamp: out-of-range indices are silently ignored
        total_blocks = transformer_options.get("total_blocks", 0)
        _do_stg = stg_skip and 0 <= block_idx < total_blocks and block_idx in stg_blocks

        x = x.contiguous()
        if not (_do_stg and stg_mode == "A"):
            # STG-A: skip self-attention entirely in marked blocks
            y = self.self_attn(
                torch.addcmul(_repeat_e(e[0], x), self.norm1(x), 1 + _repeat_e(e[1], x)),
                freqs, transformer_options=transformer_options)
        else:
            y = torch.zeros_like(x)

        if not (_do_stg and stg_mode == "R"):
            # STG-R: skip the residual from self-attention
            x = torch.addcmul(x, y, _repeat_e(e[2], x))
        # else: STG-R drops the residual — x stays unchanged
        del y

        x = x + self.cross_attn(self.norm3(x), context,
                                context_img_len=context_img_len,
                                transformer_options=transformer_options)
        if "attn2_patch" in patches:
            for p in patches["attn2_patch"]:
                x = p({"x": x, "transformer_options": transformer_options})

        y = self.ffn(torch.addcmul(_repeat_e(e[3], x), self.norm2(x),
                                    1 + _repeat_e(e[4], x)))
        x = torch.addcmul(x, y, _repeat_e(e[5], x))
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings=None):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        ops = operation_settings or {}
        out_dim = math.prod(patch_size) * out_dim
        self.norm = ops.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=ops.get("device"), dtype=ops.get("dtype"))
        self.head = ops.get("operations").Linear(dim, out_dim, device=ops.get("device"), dtype=ops.get("dtype"))
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=ops.get("device"), dtype=ops.get("dtype")))

    def forward(self, x, e):
        if e.ndim < 3:
            e = (_cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        else:
            e = (_cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e.unsqueeze(2)).unbind(2)
        return self.head(torch.addcmul(_repeat_e(e[0], x), self.norm(x),
                                        1 + _repeat_e(e[1], x)))


# ═══════════════════════════════════════════════════════════════════════
# BerniniRWanModel — self-contained, no ComfyUI patching needed
# ═══════════════════════════════════════════════════════════════════════

class BerniniRWanModel(nn.Module):
    """Wan diffusion backbone with built-in RoPE t_start, NTK, and NAG.

    Drop-in replacement for ``comfy.ldm.wan.model.WanModel``.  All
    dependencies are inlined so no ComfyUI internals are needed beyond
    ``optimized_attention``.
    """

    def __init__(self,
                 patch_size=(1, 2, 2),
                 in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256,
                 text_dim=4096, out_dim=16, num_heads=16, num_layers=32,
                 window_size=(-1, -1), qk_norm=True, cross_attn_norm=True,
                 eps=1e-6, device=None, dtype=None, operations=None,
                 **kwargs):  # absorb ComfyUI config extras (image_model, etc.)
        super().__init__()
        # dtype prioritised: explicit arg → unet_config['dtype'] → op setting
        if dtype is None:
            uc = kwargs.get('unet_config', {})
            dtype = uc.get('dtype', None)
        if dtype is None and operations is not None:
            dtype = getattr(operations, 'unet_dtype', None)
        self.dtype = dtype
        op = {"operations": operations, "device": device, "dtype": dtype}

        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.model_type = 't2v'
        self.text_len = 512

        # Embeddings
        self.patch_embedding = operations.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size,
            device=op.get("device"), dtype=torch.float32)
        self.text_embedding = nn.Sequential(
            operations.Linear(text_dim, dim, device=op.get("device"), dtype=op.get("dtype")),
            nn.GELU(approximate='tanh'),
            operations.Linear(dim, dim, device=op.get("device"), dtype=op.get("dtype")))
        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=op.get("device"), dtype=op.get("dtype")),
            nn.SiLU(),
            operations.Linear(dim, dim, device=op.get("device"), dtype=op.get("dtype")))
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, dim * 6, device=op.get("device"), dtype=op.get("dtype")))

        # Blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size,
                              qk_norm, cross_attn_norm, eps,
                              operation_settings=op)
            for _ in range(num_layers)
        ])

        # Head
        self.head = Head(dim, out_dim, patch_size, eps, operation_settings=op)

        # RoPE embedder
        d_head = dim // num_heads
        self.rope_embedder = EmbedND(
            dim=d_head, theta=10000.0,
            axes_dim=[d_head - 4 * (d_head // 6),
                       2 * (d_head // 6), 2 * (d_head // 6)])

        # T2V only — no img_emb, no ref_conv
        self.img_emb = None
        self.ref_conv = None

        # ── NTK re-entrancy stack ─────────────────────────────────
        self._ntk_theta_stack: list = []

    # ═══════════════════════════════════════════════════════════════
    # FFN quantization state management
    # ═══════════════════════════════════════════════════════════════

    def _apply_ffn_quant_config(self, transformer_options: dict):
        return  # no-op (FFN quantization removed)

    # ═══════════════════════════════════════════════════════════════
    # forward — entry point (pops RoPE/NTK kwargs, applies NTK, NAG)
    # ═══════════════════════════════════════════════════════════════

    def forward(self, x, timestep, context,
                clip_fea=None, time_dim_concat=None,
                transformer_options=None, **kwargs):
        """Entry point.  Pops RoPE/NTK kwargs, applies NTK scaling,
        then delegates to ``pre_forward``."""
        if transformer_options is None:
            transformer_options = {}

        t_start: int = kwargs.pop('_rope_t_start', 0) if kwargs else 0
        ntk_scale: float = kwargs.pop('_rope_ntk_scale', 1.0) if kwargs else 1.0

        # ── Block swap: create manager early so the forward pass can
        #     use it.  Only active when ``block_to_swap > 0``.
        _bswap = None
        cfg: Any = getattr(self, '_block_swap_config', None)
        if cfg is not None and cfg.block_to_swap > 0:
            from ..utils.block_swap import BlockSwapManager
            total = len(self.blocks) if hasattr(self, 'blocks') else 0
            window = max(1, total - cfg.block_to_swap)
            _bswap = getattr(self, '_block_swap_mgr', None)
            if _bswap is None or _bswap.window != window:
                _bswap = BlockSwapManager(
                    self,
                    window_size=window,
                    prefetch=cfg.prefetch,
                    prefetch_count=cfg.prefetch_count,
                    pin_memory=cfg.pin_memory,
                    block_reader=getattr(self, '_block_reader', None),
                    lora_reader=getattr(self, '_lora_reader', None),
                    max_disk_workers=getattr(cfg, "disk_workers", 4),
                )
                self._block_swap_mgr = _bswap
            kwargs.pop('_block_swap_mgr', None)
            kwargs['_block_swap_mgr'] = _bswap
            _bswap.prepare_blocks_entry()

        # ── NAG context projection ────────────────────────────────
        # Track whether transformer_options was already cloned by NAG
        if 'nag_context' in transformer_options:
            nag_ctx = transformer_options['nag_context']
            if nag_ctx is not None:
                text_dim = self.text_embedding[0].in_features
                if nag_ctx.shape[-1] == text_dim:
                    transformer_options = dict(transformer_options)
                    transformer_options['nag_context'] = (
                        self.text_embedding(nag_ctx).to(
                            dtype=x.dtype, device=x.device))

        # ── NTK theta scaling ─────────────────────────────────────
        theta_modified = False
        if ntk_scale != 1.0:
            self._ntk_theta_stack.append(self.rope_embedder.theta)
            self.rope_embedder.theta = self.rope_embedder.theta * ntk_scale
            theta_modified = True

        try:
            return self.pre_forward(
                x, timestep, context,
                clip_fea=clip_fea,
                time_dim_concat=time_dim_concat,
                transformer_options=transformer_options,
                t_start=t_start,
                **kwargs,
            )
        finally:
            if theta_modified:
                self.rope_embedder.theta = self._ntk_theta_stack.pop()

    # ═══════════════════════════════════════════════════════════════
    # pre_forward — padding + RoPE + context_latents
    # ═══════════════════════════════════════════════════════════════

    def pre_forward(self, x, timestep, context,
                    clip_fea=None, time_dim_concat=None,
                    transformer_options=None,
                    t_start: int = 0, **kwargs):
        bs, c, t, h, w = x.shape
        x = _pad_to_patch_size(x, self.patch_size)

        t_len = t
        if time_dim_concat is not None:
            time_dim_concat = _pad_to_patch_size(time_dim_concat, self.patch_size)
            x = torch.cat([x, time_dim_concat], dim=2)
            t_len = x.shape[2]

        # ── Main latent RoPE ──────────────────────────────────────
        rope_opts = (transformer_options or {}).get("rope_options", None)
        freqs = self._freqs(
            t_len, h, w, t_start=t_start,
            rope_options=rope_opts, device=x.device, dtype=x.dtype,
            source_id=0)

        # ── Context latent(s) RoPE + pre-patch ────────────────────
        _bswap = kwargs.get("_block_swap_mgr")
        context_latents = kwargs.get("context_latents", None)
        if context_latents is not None:
            context_latents = [
                _pad_to_patch_size(lat, self.patch_size)
                for lat in context_latents
            ]
            # Pre-patch context latents here so transformer_forward
            # can reuse them across windows without re-convolving.
            patched_context = []
            for i, lat in enumerate(context_latents):
                if lat.ndim < 3:
                    raise ValueError(
                        f"Context latent {i} has {lat.ndim} dimensions; "
                        f"expected ≥ 3 (B, C, T, ...).")
                ctx_t_start = t_start if lat.shape[-3] > 1 else 0
                freqs = torch.cat([
                    freqs,
                    self._freqs(
                        lat.shape[-3], lat.shape[-2], lat.shape[-1],
                        t_start=ctx_t_start,
                        rope_options=rope_opts,
                        device=x.device, dtype=x.dtype,
                        source_id=i + 1,
                    ),
                ], dim=1)
                if _bswap is not None:
                    _bswap.prepare_pre_forward()
                patched_context.append(
                    self.patch_embedding(lat.float().to(x.device)).to(x.dtype
                    ).flatten(2).transpose(1, 2))
            kwargs = {**kwargs,
                      "context_latents": context_latents,
                      "_patched_context_latents": patched_context}

        return self.transformer_forward(
            x, timestep, context,
            clip_fea=clip_fea, freqs=freqs,
            transformer_options=transformer_options,
            **kwargs,
        )[:, :, :t, :h, :w]

    # ═══════════════════════════════════════════════════════════════
    # transformer_forward — main transformer (patch → blocks → head)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_s2_drop_set(transformer_options: dict, total_blocks: int) -> set:
        """Build the set of block indices to skip for S² sub-network forward.

        Returns an empty set unless ``_s2_sub_network`` is True in
        *transformer_options*.  When active, ~10 % of blocks (excluding the
        first block) are randomly selected for dropping.  The seed
        ``_s2_seed`` is used for deterministic per-step mask generation.
        """
        if not transformer_options.get("_s2_sub_network", False):
            return set()

        seed = transformer_options.get("_s2_seed", 0)
        # Exclude first block (structural) and block 0
        eligible = list(range(1, total_blocks))
        drop_count = max(1, len(eligible) // 10)  # ~10 %

        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        indices = torch.randperm(len(eligible), generator=rng)[:drop_count]
        return {eligible[i.item()] for i in indices}

    def transformer_forward(self, x, t, context,
                            clip_fea=None, freqs=None,
                            transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}

        # Capture the device ComfyUI handed the latent to us.  In block-swap
        # mode ``load_device == offload_device == CPU``, so the latent (and
        # thus ``model_sampling.calculate_denoised``'s inputs) live on CPU
        # while we internally compute on GPU.  The forward output must be
        # returned on this *input* device to stay consistent with what
        # ComfyUI subtracts against — otherwise we hit a cuda:0/cpu mismatch
        # in ``model_sampling.calculate_denoised``.
        input_device = x.device

        # ── FFN activation quantization (Beta) ─────────────────────────
        pass  # FFN quant removed (was: _apply_ffn_quant_config)

        # ── Block swap: GPU↔RAM offloading ───────────────────────────
        _bswap = kwargs.get("_block_swap_mgr")
        if _bswap is None:
            cfg: Any = getattr(self, '_block_swap_config', None)
            if cfg is not None and cfg.block_to_swap > 0:
                from ..utils.block_swap import BlockSwapManager
                total = len(self.blocks) if hasattr(self, 'blocks') else 0
                window = max(1, total - cfg.block_to_swap)
                _bswap = getattr(self, '_block_swap_mgr', None)
                if _bswap is None or _bswap.window != window:
                    _bswap = BlockSwapManager(
                        self,
                        window_size=window,
                        prefetch=cfg.prefetch,
                        prefetch_count=cfg.prefetch_count,
                        pin_memory=cfg.pin_memory,
                        block_reader=getattr(self, '_block_reader', None),
                        lora_reader=getattr(self, '_lora_reader', None),
                        max_disk_workers=getattr(cfg, "disk_workers", 4),
                    )
                    self._block_swap_mgr = _bswap

        # Patch embedding — matches upstream: x.float() ensures float32
        # Conv3d accumulation (patch_embedding weights are float32).
        # Removing .float() changes numerical precision and breaks I2V
        # conditioning fidelity.
        if _bswap is not None:
            _bswap.prepare_pre_forward()
            _bswap.prepare_blocks_entry()

        # ── Block swap: activations must live on the GPU compute device ──
        # The model *weights* stay on CPU (single copy) and only a sliding
        # window of blocks is moved to GPU by BlockSwapManager, but the
        # activations (x, context, timestep, RoPE freqs, pre-patched context)
        # must be on GPU for those blocks to compute.  ComfyUI hands us CPU
        # activations because ``load_device == offload_device`` (CPU) in this
        # mode, so move them explicitly here.  Non-block-swap mode already has
        # everything on GPU and skips this.
        if _bswap is not None:
            dev = _bswap.device
            x = x.to(dev)
            if context is not None:
                context = context.to(dev)
            # nag_context was projected onto x.device in pre_forward, which is
            # CPU in block-swap mode — relocate it too so NAG cross-attention
            # doesn't see q on GPU but k/v on CPU.
            nag_ctx = transformer_options.get("nag_context", None)
            if nag_ctx is not None:
                transformer_options["nag_context"] = nag_ctx.to(dev)
            if t is not None:
                t = t.to(dev)
            if freqs is not None:
                freqs = freqs.to(dev)
            patched_context = kwargs.get("_patched_context_latents")
            if patched_context is not None:
                kwargs["_patched_context_latents"] = [c.to(dev) for c in patched_context]

        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        transformer_options["grid_sizes"] = grid_sizes
        x = x.flatten(2).transpose(1, 2)

        # ── Prompt Travel: slice to current window ────────────────────
        # Time embedding
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()
            ).to(dtype=x.dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # Context latents — use pre-patched tensors from pre_forward
        context_latents = kwargs.get("context_latents", None)
        patched_context = kwargs.get("_patched_context_latents", None)
        main_len = x.shape[1]
        if patched_context is not None:
            for cl in patched_context:
                x = torch.cat([x, cl], dim=1)

        # Text embedding
        context = self.text_embedding(context)

        # Blocks
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        has_patches = bool(blocks_replace)
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"

        # ── S² Stochastic Self-Guidance: block dropping ──────────────
        s2_drop = self._build_s2_drop_set(transformer_options, len(self.blocks))

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i

            # S²: skip this block entirely — don't load to GPU
            if i in s2_drop:
                continue

            # Block swap: ensure this block is on GPU before use
            if _bswap is not None:
                _bswap.prepare(i)
                _bswap.prefetch_next(i)

            if has_patches and ("double_block", i) in blocks_replace:
                def _wrap(args):
                    out = {}
                    out["img"] = block(
                        args["img"], context=args["txt"],
                        e=args["vec"], freqs=args["pe"],
                        context_img_len=None,
                        transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)](
                    {"img": x, "txt": context, "vec": e0, "pe": freqs,
                     "transformer_options": transformer_options},
                    {"original_block": _wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context,
                          context_img_len=None,
                          transformer_options=transformer_options)

        # Head
        if _bswap is not None:
            _bswap.prepare_head()
        x = self.head(x, e)

        # Trim context tokens
        if context_latents is not None:
            x = x[:, :main_len]

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)

        # ── Return output on the input (ComfyUI ``load_device``) device ──
        # Block-swap keeps the latent on CPU (load_device == CPU); we only
        # windowed the weights/activations onto GPU for compute.  Move the
        # final tensor back so model_sampling.calculate_denoised stays on one
        # device.  No-op in non-block-swap mode (input already on GPU).
        x = x.to(input_device)
        return x

    # ═══════════════════════════════════════════════════════════════
    # RoPE — position ID computation, called directly from pre_forward
    # ═══════════════════════════════════════════════════════════════

    def _freqs(self, t, h, w, *, t_start=0, source_id=0,
               rope_options=None, device=None, dtype=None):
        """Core RoPE frequency computation — no keyword parsing.
        Called inline from ``pre_forward`` so torch.compile sees
        the entire position-ID → EmbedND → freqs chain as one graph.
        """
        ps = self.patch_size
        t_len = ((t + (ps[0] // 2)) // ps[0])
        h_len = ((h + (ps[1] // 2)) // ps[1])
        w_len = ((w + (ps[2] // 2)) // ps[2])
        steps_t, steps_h, steps_w = t_len, h_len, w_len
        h_start = 0
        w_start = 0

        if rope_options is not None:
            # Unpack once — avoids 6 dict.get calls in the hot path.
            t_len = (t_len - 1.0) * rope_options.get("scale_t", 1.0) + 1.0
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0
            t_start += rope_options.get("shift_t", 0.0)
            h_start += rope_options.get("shift_y", 0.0)
            w_start += rope_options.get("shift_x", 0.0)

        # Compute 3D position IDs with linspace broadcasts
        img_ids = torch.empty((steps_t, steps_h, steps_w, 3),
                              device=device, dtype=dtype)
        img_ids[:, :, :, 0] = torch.linspace(
            t_start, t_start + (t_len - 1),
            steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = torch.linspace(
            h_start, h_start + (h_len - 1),
            steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = torch.linspace(
            w_start, w_start + (w_len - 1),
            steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        freqs = self.rope_embedder(img_ids).movedim(1, 2)

        # source_id=0 produces identity rotation ([1,0,1,0,...]) —
        # skip the useless einsum.
        if source_id > 0:
            d = self.dim // self.num_heads
            pos = torch.tensor([[float(source_id)]], device=freqs.device,
                              dtype=torch.float32)
            id_rot = _rope(pos, d, self.rope_embedder.theta
                           ).reshape(1, 1, 1, d // 2, 2, 2).to(freqs.dtype)
            freqs = torch.einsum('...ij,...jk->...ik', freqs, id_rot)
        return freqs


    # ═══════════════════════════════════════════════════════════════
    # Unpatchify
    # ═══════════════════════════════════════════════════════════════

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u
