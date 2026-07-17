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
import logging
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

# ── stable ComfyUI API (the ONLY import) ─────────────────────────────
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope1  # use upstream CUDA kernel for bit-identical RoPE

logger = logging.getLogger(__name__)


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


# ═══════════════════════════════════════════════════════════════════════
# Model loader
# ═══════════════════════════════════════════════════════════════════════

def _normalize_unet_key(k: str) -> str:
    """Strip common checkpoint prefixes so state-dict keys line up with the
    diffusion_model's own keys (``model.diffusion_model.blocks.0...`` →
    ``blocks.0...``).
    """
    if k.startswith("model.diffusion_model."):
        return k[len("model.diffusion_model."):]
    if k.startswith("model."):
        return k[len("model."):]
    if k.startswith("video_model."):
        return k[len("video_model."):].replace("modulation.modulation", "modulation")
    if k.startswith("diffusion_model."):
        return k[len("diffusion_model."):]
    return k


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
    with block swap).  This reader keeps the file handle open and reads each
    tensor's bytes on demand, preserving the streaming / low-peak-memory
    behaviour.
    """

    def __init__(self, path: str):
        try:
            from safetensors.torch import _TYPES
            self._dtype_map = _TYPES
        except Exception:
            self._dtype_map = _SAFETENSORS_DTYPE_MAP
        self._path = path
        self._file = open(path, "rb")
        import struct, json
        header_len = struct.unpack("<Q", self._file.read(8))[0]
        self._header = json.loads(self._file.read(header_len))
        self._data_offset = 8 + header_len
        self._keys = [
            k for k, v in self._header.items()
            if isinstance(v, dict) and "dtype" in v
        ]

    def keys(self):
        return list(self._keys)

    def get_slice(self, key: str):
        return _SafetensorsSlice(self._header[key]["shape"])

    def get_tensor(self, key: str):
        info = self._header[key]
        dtype = self._dtype_map[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        self._file.seek(self._data_offset + start)
        nbytes = end - start
        # Read directly into a writable bytearray so ``torch.frombuffer``
        # doesn't warn about a read-only buffer, then clone so the tensor
        # owns its memory and the temporary buffer can be freed immediately.
        buffer = bytearray(nbytes)
        view = memoryview(buffer)
        read = 0
        while read < nbytes:
            n = self._file.readinto(view[read:])
            if n == 0:
                raise EOFError(
                    f"Unexpected EOF reading tensor {key!r} "
                    f"({read}/{nbytes} bytes)"
                )
            read += n
        return torch.frombuffer(buffer, dtype=dtype).reshape(shape).clone()

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
        return False


class _SafetensorsSlice:
    """Dummy slice returned by ``_SafetensorsFileReader.get_slice``."""

    def __init__(self, shape):
        self._shape = shape

    def get_shape(self):
        return self._shape


def _build_streaming_lora_groups(lora_specs):
    """Load all LoRAs and group A/B/alpha by canonical base weight key."""
    if not lora_specs:
        return {}
    from ..utils.lora import load_lora_state_dict
    groups = {}
    for lora_path, strength in lora_specs:
        if strength == 0.0:
            continue
        try:
            lora_sd = load_lora_state_dict(lora_path)
        except Exception as e:
            logger.error("[BerniniR] Failed to load LoRA %s: %s", lora_path, e)
            raise
        per = {}
        for k, v in lora_sd.items():
            if k.endswith(".lora_A.weight"):
                base = k[:-len(".lora_A.weight")] + ".weight"
                per.setdefault(base, {})["A"] = v
            elif k.endswith(".lora_B.weight"):
                base = k[:-len(".lora_B.weight")] + ".weight"
                per.setdefault(base, {})["B"] = v
            elif k.endswith(".alpha"):
                base = k[:-len(".alpha")] + ".weight"
                per.setdefault(base, {})["alpha"] = v
            elif k.endswith(".diff_b"):
                base = k[:-len(".diff_b")] + ".weight"
                per.setdefault(base, {})["diff_b"] = v
            elif k.endswith(".diff"):
                base = k[:-len(".diff")] + ".weight"
                per.setdefault(base, {})["diff"] = v
        for base, parts in per.items():
            if "A" not in parts or "B" not in parts:
                # Norm-only diffs / bias diffs (no A/B pair)
                if parts.get("diff_b") is not None or parts.get("diff") is not None:
                    norm_base = _normalize_unet_key(base)
                    groups.setdefault(norm_base, []).append({
                        "A": None, "B": None, "alpha": None,
                        "diff_b": parts.get("diff_b"),
                        "diff": parts.get("diff"),
                        "strength": float(strength),
                    })
                continue
            norm_base = _normalize_unet_key(base)
            groups.setdefault(norm_base, []).append({
                "A": parts["A"],
                "B": parts["B"],
                "alpha": parts.get("alpha"),
                "diff_b": parts.get("diff_b"),
                "diff": parts.get("diff"),
                "strength": float(strength),
            })
        logger.info("[BerniniR] Inline merged LoRA: %s (strength=%.3f)",
                    lora_path, strength)
    return groups


def _collect_block_lora(lora_groups: dict, bidx: int) -> dict | None:
    """Return ``{pname: entries}`` for transformer block *bidx* and remove
    those keys from *lora_groups*.

    This co-locates a block's LoRA with the block itself (``block._lora_payload``)
    and lets the global LoRA pool shrink toward empty as every block is attached —
    the unified (block, lora) → (block) slot design needs no persistent pool.
    """
    if not lora_groups:
        return None
    prefix = f"blocks.{bidx}."
    out = {}
    to_pop = []
    for key, val in lora_groups.items():
        if key.startswith(prefix) and key.endswith(".weight"):
            # Keep the full param name (incl. ".weight") so it matches the
            # slot's ``named_parameters()`` key, e.g. "self_attn.q.weight".
            pname = key[len(prefix):]
            out[pname] = val
            to_pop.append(key)
    for k in to_pop:
        del lora_groups[k]
    return out or None


def _apply_streaming_loras(base: torch.Tensor, groups: list, scale: torch.Tensor | None = None):
    """Fold pre-grouped LoRAs/DoRAs into a single base weight tensor.

    ``scale`` is the ``weight_scale`` for fp8_scaled (quantized) weights.  When
    provided the base is dequantized (``stored * scale``), the LoRA deltas are
    folded in float32, then the result is re-quantized back to fp8 and returned
    together with a fresh ``scale`` (caller must update the group).  When
    ``scale`` is None the weight is plain bf16/fp16 and the original dtype is kept.

    DoRA support (from ``diff_b`` / ``diff`` in the group dicts):
      - ``diff_b`` on 2-D (Linear) weights: DoRA magnitude delta applied row-wise.
      - ``diff``  on 1-D (norm) weights: direct additive delta to the norm weight.

    Returns ``(weight, scale_or_None)``.
    """
    # Quick return if nothing to do: no groups, or groups with only A=None entries.
    has_work = False
    # Collect DoRA params from the group list (take last non-None values).
    dora_diff_b = None
    dora_diff = None
    for g in groups:
        if g.get("A") is not None and g.get("B") is not None:
            has_work = True
        if g.get("diff_b") is not None:
            dora_diff_b = g["diff_b"]
        if g.get("diff") is not None:
            dora_diff = g["diff"]
    if not has_work and dora_diff_b is None and dora_diff is None:
        return base, scale

    from ..utils.lora import _lora_delta, _requantize_fp8

    if scale is not None:
        base_f = base.to(torch.float32) * scale.to(torch.float32)
        is_fp8 = True
    else:
        base_f = base.to(torch.float32)
        is_fp8 = False

    # Capture the ORIGINAL base row/vector norm BEFORE any LoRA delta is applied.
    # DoRA needs the target norm = ||W0|| + diff_b, so we must snapshot ||W0|| now
    # (after the delta loop base_f would already be W0 + Δ).
    if base.dim() == 2:
        init_norm = base_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        init_norm = None

    # 1. Apply LoRA deltas
    for g in groups:
        if g.get("A") is not None and g.get("B") is not None:
            delta, _ = _lora_delta(g["A"], g["B"], g.get("alpha"), g["strength"], base.shape)
            base_f = base_f + delta

    # 2. DoRA for 2-D (Linear) weights
    #    W_final = (||W0|| + diff_b) * W_temp / ||W_temp||
    #    where W_temp = W0 + Σ delta, and init_norm is ||W0|| captured above.
    #    NOTE: diff_b is the magnitude *difference* (tiny, ~1e-4), NOT the target
    #    magnitude.  Using m = diff_b alone (without ||W0||) zeroes the weight.
    if dora_diff_b is not None and base.dim() == 2:
        temp_norm = base_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
        m = (init_norm + dora_diff_b.to(torch.float32).reshape(-1, 1)).clamp(min=0.0)
        base_f = m * base_f / temp_norm
    else:
        # Norm diff for 1-D weights (handled here only for the LoRA+diff case;
        # pure norm-only groups are applied by _fold_group_loras).
        if dora_diff is not None and base.dim() == 1:
            base_f = base_f + dora_diff.to(torch.float32)

    if is_fp8:
        return _requantize_fp8(base_f)
    return base_f.to(base.dtype), None


def _fold_group_loras(sub_group: dict, prefix: str, lora_groups: dict | None):
    """Fold LoRAs whose base key lives under *prefix* into *sub_group* in place.

    Used by the eager streaming loader so block-swap can fold LoRA at the
    exact moment a block group is read from disk.
    """
    if not lora_groups:
        return

    def _cast_back(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Norm weights/biases in this model are fp16 (not fp8), so a plain
        # cast back to the original dtype is sufficient.  The fp8 branch is
        # kept only for theoretical fp8-scaled norms.
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            from ..utils.lora import _requantize_fp8
            return _requantize_fp8(t.to(torch.float32))[0]
        return t.to(dtype)

    for full_key, g_list in lora_groups.items():
        if not full_key.startswith(prefix):
            continue
        local = full_key[len(prefix):]  # e.g. "self_attn.q.weight"
        if local not in sub_group:
            continue
        # Split into LoRA entries (have A & B) and norm-only entries (no A/B).
        lora_entries = [g for g in g_list
                        if g.get("A") is not None and g.get("B") is not None]
        norm_only = [g for g in g_list
                     if not (g.get("A") is not None and g.get("B") is not None)]

        if lora_entries:
            # sub_group keys keep the full tensor name (e.g. "self_attn.q.weight_scale"),
            # so the scale key is local + "_scale" (local already ends in ".weight").
            scale = sub_group.get(local + "_scale")
            new_w, new_scale = _apply_streaming_loras(sub_group[local], lora_entries, scale)
            sub_group[local] = new_w
            if new_scale is not None:
                sub_group[local + "_scale"] = new_scale

        # Norm-only groups: mirror merge_lora_into_state_dict's pure
        # norm-diff / bias-diff branch.  diff -> norm weight (1-D);
        # diff_b -> corresponding bias (1-D).
        for g in norm_only:
            tgt = sub_group[local]
            if g.get("diff") is not None and tgt.dim() == 1:
                sub_group[local] = _cast_back(
                    tgt.to(torch.float32) + g["diff"].to(torch.float32), tgt.dtype)
            if g.get("diff_b") is not None:
                bias_local = local[: -len(".weight")] + ".bias"
                if bias_local in sub_group:
                    bt = sub_group[bias_local]
                    sub_group[bias_local] = _cast_back(
                        bt.to(torch.float32) + g["diff_b"].to(torch.float32), bt.dtype)


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
    the model, we read one block group at a time.  Peak host RAM drops from
    ~2x the model size to roughly the model size plus one block group.
    """
    import comfy.model_patcher
    import comfy.utils

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
        def _sd_key(k):
            return norm_map.get(k)
        _canonical_keys = list(norm_map.keys())
        def _stream_has_dtype(k):
            rk = norm_map.get(k)
            if rk is None:
                return None
            try:
                return f.get_tensor(rk).dtype
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

    # Stream weights into the model one block group at a time.
    seen_keys = set()
    with _SafetensorsFileReader(model_path) as f:
        group = {}
        current_group = None
        for raw_key in sorted(raw_keys):
            target_key = _normalize_unet_key(raw_key)
            seen_keys.add(target_key)

            parts = target_key.split('.')
            if len(parts) >= 2 and parts[0] == 'blocks':
                group_key = f"{parts[0]}.{parts[1]}"
            else:
                group_key = parts[0] if parts else raw_key

            if current_group is not None and group_key != current_group:
                if current_group.startswith("blocks."):
                    if lazy:
                        # In lazy mode: record metadata only — do NOT load
                        # block weights into RAM.  BlockSwapManager's
                        # _DiskPrefetcher will load them on demand.
                        if block_swap:
                            _record_block_meta(
                                current_group, group, block_plan, block_bytes, block_mb)
                    else:
                        _stream_load_group(dm, current_group, group, None)
                        if block_swap:
                            _record_block_meta(
                                current_group, group, block_plan, block_bytes, block_mb)
                            # Co-locate this block's LoRA with its slot
                            # (unified (block, lora) → (block) design).
                            bidx = int(current_group.split('.')[1])
                            blk = _collect_block_lora(lora_groups, bidx)
                            if blk:
                                dm.blocks[bidx]._lora_payload = blk
                else:
                    _stream_load_group(dm, current_group, group, lora_groups)
                group.clear()

            current_group = group_key

            # In lazy block-swap mode the block weights are NOT loaded here
            # (the _DiskPrefetcher reads them on demand during sampling).  We
            # only need per-block byte/param counts for the VRAM estimate, so
            # take them from the header instead of reading the full tensor —
            # this avoids a pointless sequential disk read of the full 14B
            # checkpoint that otherwise dominates load time (~25 s).
            is_block_key = len(parts) >= 2 and parts[0] == 'blocks'
            if lazy and block_swap and is_block_key:
                group[target_key] = f.tensor_meta(raw_key)  # (nbytes, numel)
            else:
                group[target_key] = f.get_tensor(raw_key)

        if group:
            if current_group.startswith("blocks."):
                if lazy:
                    if block_swap:
                        _record_block_meta(current_group, group, block_plan, block_bytes, block_mb)
                else:
                    _stream_load_group(dm, current_group, group, None)
                    if block_swap:
                        _record_block_meta(
                            current_group, group, block_plan, block_bytes, block_mb)
                        # Co-locate this block's LoRA with its slot.
                        bidx = int(current_group.split('.')[1])
                        blk = _collect_block_lora(lora_groups, bidx)
                        if blk:
                            dm.blocks[bidx]._lora_payload = blk
            else:
                _stream_load_group(dm, current_group, group, lora_groups)
            group.clear()

    for missing_base in set(lora_groups) - seen_keys:
        logger.warning(
            "[BerniniR] LoRA base key not found in model state dict: %s",
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
            logger.info("[BerniniR] Lazy mode: %d blocks, metadata only, no pre-warm",
                        num_layers)
        else:
            DEFAULT_WINDOW = 10
            warm_blocks = min(DEFAULT_WINDOW, num_layers)
            for idx in range(warm_blocks):
                dm.blocks[idx].to(load_device)
            dm._prewarmed = warm_blocks
            logger.info("[BerniniR] Pre-warmed block window: %d / %d blocks to GPU",
                        warm_blocks, num_layers)

    mp = comfy.model_patcher.ModelPatcher(
        base, load_device=load_device, offload_device=offload_device)

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
    logger.info("[BerniniR] Stream-loaded (%(mode)s): dim=%(dim)d heads=%(num_heads)d "
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

    # Fold any LoRAs targeting this group now that weight_scale is in scope.
    _fold_group_loras(sub_group, prefix, lora_groups)

    # Block params are inference tensors (model built under InferenceMode).
    # load_state_dict(assign=False) does an in-place copy_ which is forbidden
    # outside InferenceMode -> wrap the copy so it is allowed while keeping the
    # params as inference tensors.
    with torch.inference_mode():
        sub.load_state_dict(sub_group, strict=False, assign=False)


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
                "[BerniniR] Weight not loaded: %s — state dict key mismatch?%s",
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

    _verify_weights_loaded(dm)

    logger.info("[BerniniR] Loaded: dim=%(dim)d heads=%(num_heads)d "
                "layers=%(num_layers)d ffn=%(ffn_dim)d "
                "variant=%(model_variant)s quant=%(quant)s",
                {"dim": unet_config["dim"], "num_heads": unet_config["num_heads"],
                 "num_layers": unet_config["num_layers"],
                 "ffn_dim": unet_config["ffn_dim"],
                 "model_variant": unet_config["model_variant"],
                 "quant": quantization or "none"})
    return mp
