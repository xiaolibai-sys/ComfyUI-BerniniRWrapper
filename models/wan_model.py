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
from typing import Optional

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

        # ── Block swap: create manager early so pre_forward can use it ──
        _bswap = None
        if transformer_options.get("_block_swap", False):
            from ..utils.block_swap import BlockSwapManager
            bswap_args = transformer_options.get("_block_swap_args", {})
            window = bswap_args.get("window_size", 10)
            _bswap = getattr(self, '_block_swap_mgr', None)
            if _bswap is None or _bswap.window != window:
                _bswap = BlockSwapManager(
                    self,
                    window_size=window,
                    prefetch=bswap_args.get("prefetch", True),
                    prefetch_count=bswap_args.get("prefetch_count", 1),
                    pin_memory=bswap_args.get("pin_memory", False),
                )
                self._block_swap_mgr = _bswap
            kwargs.pop('_block_swap_mgr', None)
            kwargs['_block_swap_mgr'] = _bswap
            _bswap.prepare_blocks_entry()

        # ── NAG context projection ────────────────────────────────
        # Track whether transformer_options was already cloned by NAG
        _orig_topt = transformer_options
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
        if _bswap is None and transformer_options.get("_block_swap", False):
            from ..utils.block_swap import BlockSwapManager
            bswap_args = transformer_options.get("_block_swap_args", {})
            window = bswap_args.get("window_size", 10)
            _bswap = getattr(self, '_block_swap_mgr', None)
            if _bswap is None or _bswap.window != window:
                _bswap = BlockSwapManager(
                    self,
                    window_size=window,
                    prefetch=bswap_args.get("prefetch", True),
                    prefetch_count=bswap_args.get("prefetch_count", 1),
                    pin_memory=bswap_args.get("pin_memory", False),
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
    from ..utils.lora import standardize_lora_keys, load_lora_state_dict
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
        for base, parts in per.items():
            if "A" not in parts or "B" not in parts:
                continue
            norm_base = _normalize_unet_key(base)
            groups.setdefault(norm_base, []).append({
                "A": parts["A"],
                "B": parts["B"],
                "alpha": parts.get("alpha"),
                "strength": float(strength),
            })
        logger.info("[BerniniR] Inline merged LoRA: %s (strength=%.3f)",
                    lora_path, strength)
    return groups


def _apply_streaming_loras(base: torch.Tensor, groups: list, scale: torch.Tensor | None = None):
    """Fold pre-grouped LoRAs into a single base weight tensor.

    ``scale`` is the ``weight_scale`` for fp8_scaled (quantized) weights.  When
    provided the base is dequantized (``stored * scale``), the LoRA deltas are
    folded in float32, then the result is re-quantized back to fp8 and returned
    together with a fresh ``scale`` (caller must update the group).  When
    ``scale`` is None the weight is plain bf16/fp16 and the original dtype is kept.

    Returns ``(weight, scale_or_None)``.
    """
    if not groups:
        return base, scale
    from ..utils.lora import _lora_delta, _requantize_fp8

    if scale is not None:
        base_f = base.to(torch.float32) * scale.to(torch.float32)
        is_fp8 = True
    else:
        base_f = base.to(torch.float32)
        is_fp8 = False

    for g in groups:
        delta, _ = _lora_delta(g["A"], g["B"], g.get("alpha"), g["strength"], base.shape)
        base_f = base_f + delta

    if is_fp8:
        return _requantize_fp8(base_f)
    return base_f.to(base.dtype), None


def _fold_group_loras(sub_group: dict, prefix: str, lora_groups: dict | None):
    """Fold LoRAs whose base key lives under *prefix* into *sub_group* in place.

    Shared by the eager streaming loader and the on-demand ``BlockDiskSource``
    so block-swap can fold LoRA at the exact moment a block is read from disk.
    """
    if not lora_groups:
        return
    for full_key, g_list in lora_groups.items():
        if not full_key.startswith(prefix):
            continue
        local = full_key[len(prefix):]
        if local not in sub_group:
            continue
        scale = sub_group.get(local + ".weight_scale")
        new_w, new_scale = _apply_streaming_loras(sub_group[local], g_list, scale)
        sub_group[local] = new_w
        if new_scale is not None:
            sub_group[local + ".weight_scale"] = new_scale


class BlockDiskSource:
    """On-demand, per-block weight loader for **streaming block-swap**.

    Only cheap metadata is kept resident: the safetensors index, the list of
    tensor keys belonging to each block, the pre-grouped LoRA folds, and a
    per-block size estimate.  The ~14 GB of weights themselves are *never* all
    in RAM at once -- each block is read from disk the instant the swap manager
    needs it on GPU, its LoRA is fused inline, and the CPU copy is released on
    eviction.  Peak host RAM therefore stays at roughly one GPU window of blocks
    plus a single in-flight block, which is what lets a 23 GB machine run the
    dual-expert switch without both 14 GB models resident simultaneously.
    """

    def __init__(self, model_path, norm_map, block_plan, lora_groups,
                 block_bytes, block_mb, param_dtype=None):
        self.model_path = model_path
        self.norm_map = norm_map            # norm_key -> raw safetensors key
        self.block_plan = block_plan        # idx -> [norm_key, ...]
        self.lora_groups = lora_groups      # norm_full_key -> [ {A,B,alpha,strength} ]
        self.block_bytes = block_bytes      # idx -> total bytes (host RAM)
        self._block_mb = block_mb           # idx -> VRAM MB estimate
        self.param_dtype = param_dtype
        self._reader = None

    # ── size estimates (for VRAM budgeting) ──────────────────────────
    def estimate_block_mb(self) -> float:
        """Average per-block VRAM estimate in MB."""
        if self._block_mb:
            return sum(self._block_mb.values()) / len(self._block_mb)
        return 0.0

    # ── lifecycle ────────────────────────────────────────────────────
    def _open(self):
        if self._reader is None:
            self._reader = _SafetensorsFileReader(self.model_path)
        return self._reader

    def close(self):
        if self._reader is not None:
            try:
                self._reader.__exit__(None, None, None)
            except Exception:
                pass
            self._reader = None

    # ── on-demand block load ─────────────────────────────────────────
    def load_block(self, idx: int, dm):
        """Read block *idx* from disk, fuse its LoRA inline, fill dm.blocks[idx].

        The freshly loaded block lives on CPU (host RAM); the caller is
        responsible for moving it to GPU (BlockSwapManager._to_gpu) and for
        freeing the CPU copy on eviction.
        """
        if idx not in self.block_plan:
            return
        reader = self._open()
        prefix = "blocks.%d." % idx
        group = {}
        for nk in self.block_plan[idx]:
            group[nk] = reader.get_tensor(self.norm_map[nk])
        # Fold LoRAs targeting this block (mirrors _stream_load_group).
        _fold_group_loras(group, prefix, self.lora_groups)
        sub_group = {k[len(prefix):]: v for k, v in group.items()
                     if k.startswith(prefix)}
        dm.blocks[idx].load_state_dict(sub_group, strict=False, assign=False)


def _load_bernini_model_safetensors_streaming(
    model_path: str,
    model_options: dict,
    lora_specs: list | None,
    block_swap: bool = False,
) -> object:
    """Memory-efficient loader for safetensors checkpoints.

    Instead of loading the full state dict into RAM and then copying it into
    the model, we read one block group at a time.  Peak host RAM drops from
    ~2x the model size to roughly the model size plus one block group.
    """
    import math
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

        # ── Config detection from metadata only ─────────────────────────
        if "patch_embedding.weight" in norm_map:
            dim = _shape(norm_map["patch_embedding.weight"])[0]
        elif "patch_embedding.0.weight" in norm_map:
            dim = _shape(norm_map["patch_embedding.0.weight"])[0]
        elif "head.modulation" in norm_map:
            dim = _shape(norm_map["head.modulation"])[-1]
        else:
            raise KeyError("Cannot detect model dimension: no patch_embedding "
                           "or head.modulation found")

        num_heads = dim // 128
        block_indices = {
            int(k.split('.')[1])
            for k in norm_map
            if k.startswith('blocks.') and k.split('.')[1].isdigit()
        }
        num_layers = max(block_indices) + 1 if block_indices else 30

        ffn_dim = None
        for k in ('blocks.0.ffn.0.weight', 'blocks.0.ffn.w1.weight',
                  'blocks.0.ffn.fc1.weight', 'blocks.0.ffn.0.bias'):
            if k in norm_map:
                ffn_dim = _shape(norm_map[k])[0]
                break
        if ffn_dim is None:
            raise KeyError("Cannot detect FFN dimension")

        in_dim = None
        for k in ('patch_embedding.weight', 'patch_embedding.0.weight'):
            if k in norm_map:
                in_dim = _shape(norm_map[k])[1]
                break
        if in_dim is None:
            raise KeyError("Cannot detect in_dim")

        out_dim = 16
        for k in ('head.head.weight', 'head.head.0.weight'):
            if k in norm_map:
                out_dim = _shape(norm_map[k])[0] // 4
                break

        if dim == 5120:
            model_variant = "14B"
        elif dim == 3072:
            model_variant = "5B"
        elif dim == 1536:
            model_variant = "1_3B"
        else:
            model_variant = "unknown"

        # Quantization detection
        weight_dtype_val = None
        quantization = None
        is_scaled_fp8 = any(
            k.endswith((".scale_weight", ".weight_scale", ".weight_scale_2"))
            for k in norm_map
        )
        if is_scaled_fp8:
            for k in norm_map:
                if k.endswith(".weight_scale_2"):
                    quantization = "nvfp4"
                    break

        for k in ('head.modulation', 'time_projection.0.weight',
                  'time_embedding.0.weight', 'blocks.0.self_attn.q.weight'):
            if k not in norm_map:
                continue
            try:
                dtype = f.get_tensor(norm_map[k]).dtype
            except Exception:
                continue
            weight_dtype_val = dtype
            if dtype == torch.float8_e4m3fn:
                quantization, weight_dtype_val = "fp8_e4m3fn", torch.float8_e4m3fn
                break
            elif dtype == torch.float8_e5m2:
                quantization, weight_dtype_val = "fp8_e5m2", torch.float8_e5m2
                break

        if is_scaled_fp8 and quantization:
            quantization += "_scaled"

        parameters = sum(math.prod(_shape(k)) for k in raw_keys)

    # Build model
    unet_config = {
        'dim': dim, 'out_dim': out_dim, 'num_heads': num_heads,
        'ffn_dim': ffn_dim, 'num_layers': num_layers,
        'patch_size': (1, 2, 2), 'freq_dim': 256, 'in_dim': in_dim,
        'qk_norm': True, 'cross_attn_norm': True, 'eps': 1e-6,
        'window_size': (-1, -1), 'text_dim': 4096,
        'model_variant': model_variant,
    }

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

    # Pre-load LoRAs (small) so they can be folded block-by-block.
    lora_groups = _build_streaming_lora_groups(lora_specs)

    # Per-block metadata for block-swap VRAM estimates.  Blocks are now
    # streamed into the CPU model during the pass below (no longer dropped),
    # but we still record key lists and byte counts for BlockSwapManager.
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
                    # Stream: load block into CPU model immediately (no more
                    # wasteful read-and-discard).  In block-swap mode we also
                    # record per-block metadata so BlockSwapManager has VRAM
                    # size estimates without a separate pre-scan pass.
                    _stream_load_group(dm, current_group, group, lora_groups)
                    if block_swap:
                        _record_block_meta(
                            current_group, group, block_plan, block_bytes, block_mb)
                else:
                    _stream_load_group(dm, current_group, group, lora_groups)
                group.clear()

            current_group = group_key

            tensor = f.get_tensor(raw_key)
            group[target_key] = tensor

        if group:
            if current_group.startswith("blocks."):
                _stream_load_group(dm, current_group, group, lora_groups)
                if block_swap:
                    _record_block_meta(
                        current_group, group, block_plan, block_bytes, block_mb)
            else:
                _stream_load_group(dm, current_group, group, lora_groups)
            group.clear()

    for missing_base in set(lora_groups) - seen_keys:
        logger.warning(
            "[BerniniR] LoRA base key not found in model state dict: %s",
            missing_base,
        )

    # In block-swap mode we pre-warm the first window of blocks to GPU
    # during the streaming pass so the GPU window is ready immediately.
    # Blocks beyond the window stay in CPU RAM; BlockSwapManager manages
    # GPU↔CPU movement with no further disk I/O.
    if block_swap:
        DEFAULT_WINDOW = 10
        warm_blocks = min(DEFAULT_WINDOW, num_layers)
        for idx in range(warm_blocks):
            dm.blocks[idx].to(load_device)
        avg_mb = sum(block_mb.values()) / len(block_mb) if block_mb else 0.0
        dm._block_meta = {'block_mb': dict(block_mb), 'avg_mb': avg_mb}
        dm._prewarmed = warm_blocks
        logger.info("[BerniniR] Pre-warmed block window: %d / %d blocks to GPU",
                    warm_blocks, num_layers)

    mp = comfy.model_patcher.ModelPatcher(
        base, load_device=load_device, offload_device=offload_device)

    # Blocks are loaded now (streamed into model during the pass above),
    # so verify ALL weights — no skips needed.
    _verify_weights_loaded(dm, skip_blocks=False)

    logger.info("[BerniniR] Stream-loaded: dim=%d heads=%d layers=%d ffn=%d "
                "variant=%s quant=%s", dim, num_heads, num_layers, ffn_dim,
                model_variant, quantization or 'none')
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
            dm.load_state_dict(group, strict=False, assign=False)
            return
        sub = dm.blocks[idx]
    elif hasattr(dm, group_key):
        sub = getattr(dm, group_key)
    else:
        # Unknown top-level key — fall back to a full-model load.
        dm.load_state_dict(group, strict=False, assign=False)
        return
    sub_group = {k[len(prefix):]: v for k, v in group.items() if k.startswith(prefix)}

    # Fold any LoRAs targeting this group now that weight_scale is in scope.
    _fold_group_loras(sub_group, prefix, lora_groups)

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
    ``BlockDiskSource.load_block``).  The real weights are still materialised
    later, on first GPU move.
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


def _verify_weights_loaded(dm, skip_blocks: bool = False):
    """Warn if any ComfyUI ops layer still has a None weight after loading.

    When *skip_blocks* is set (streaming block-swap), transformer blocks are
    intentionally not resident at load time -- they are read from disk on
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
            logger.error(
                "[BerniniR] Weight not loaded: %s — state dict key mismatch?",
                name)


def load_bernini_model(model_path, model_options=None, state_dict=None, lora_specs=None, block_swap: bool = False) -> object:
    """Load a Bernini-R / Wan checkpoint.  Fully self-contained —
    no ``model_detection`` / ``supported_models`` dependency.

    For ``.safetensors`` files this now uses a streaming loader that avoids
    holding the full state dict in RAM.  ``.pt`` / ``.ckpt`` files still fall
    back to the full-dict path.
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
                model_path, model_options, lora_specs, block_swap)
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

    # Config detection
    if 'patch_embedding.weight' in sd:
        dim = sd['patch_embedding.weight'].shape[0]
    elif 'patch_embedding.0.weight' in sd:
        dim = sd['patch_embedding.0.weight'].shape[0]
    elif 'head.modulation' in sd:
        dim = sd['head.modulation'].shape[-1]
    else:
        raise KeyError("Cannot detect model dimension: no patch_embedding "
                       "or head.modulation found")

    num_heads = dim // 128
    block_indices = {int(k.split('.')[1]) for k in sd
                     if k.startswith('blocks.') and k.split('.')[1].isdigit()}
    num_layers = max(block_indices) + 1 if block_indices else 30

    ffn_dim = None
    for k in ('blocks.0.ffn.0.weight', 'blocks.0.ffn.w1.weight',
              'blocks.0.ffn.fc1.weight', 'blocks.0.ffn.0.bias'):
        if k in sd: ffn_dim = sd[k].shape[0]; break
    if ffn_dim is None:
        raise KeyError("Cannot detect FFN dimension")

    in_dim = None
    for k in ('patch_embedding.weight', 'patch_embedding.0.weight'):
        if k in sd: in_dim = sd[k].shape[1]; break
    if in_dim is None:
        raise KeyError("Cannot detect in_dim")

    out_dim = None
    for k in ('head.head.weight', 'head.head.0.weight'):
        if k in sd:
            out_dim = sd[k].shape[0] // 4
            break
    if out_dim is None:
        out_dim = 16

    if dim == 5120:
        model_variant = "14B"
    elif dim == 3072:
        model_variant = "5B"
    elif dim == 1536:
        model_variant = "1_3B"
    else:
        model_variant = "unknown"

    # Quantization detection
    weight_dtype = None
    quantization = None
    is_scaled_fp8 = False
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float8_e4m3fn:
                quantization, weight_dtype = 'fp8_e4m3fn', torch.float8_e4m3fn
            elif v.dtype == torch.float8_e5m2:
                quantization, weight_dtype = 'fp8_e5m2', torch.float8_e5m2
        if k.endswith('.scale_weight') or k.endswith('.weight_scale'):
            is_scaled_fp8 = True
        if k.endswith('.weight_scale_2'):
            is_scaled_fp8 = True
            if quantization is None:
                quantization = 'nvfp4'
        if quantization and is_scaled_fp8:
            break
    if is_scaled_fp8 and quantization:
        quantization += '_scaled'

    parameters = comfy.utils.calculate_parameters(sd)
    fp8 = quantization is not None and 'fp8' in quantization

    unet_config = {
        'dim': dim, 'out_dim': out_dim, 'num_heads': num_heads,
        'ffn_dim': ffn_dim, 'num_layers': num_layers,
        'patch_size': (1, 2, 2), 'freq_dim': 256, 'in_dim': in_dim,
        'qk_norm': True, 'cross_attn_norm': True, 'eps': 1e-6,
        'window_size': (-1, -1), 'text_dim': 4096,
        'model_variant': model_variant,
    }

    base, load_device, offload_device = _build_bernini_base(
        unet_config, model_options, fp8, quantization,
        parameters=parameters, weight_dtype=weight_dtype or comfy.utils.weight_dtype(sd),
        block_swap=block_swap,
    )
    dm = base.diffusion_model

    base.load_model_weights(sd, "", assign=False)

    mp = comfy.model_patcher.ModelPatcher(
        base, load_device=load_device, offload_device=offload_device)

    _verify_weights_loaded(dm)

    logger.info("[BerniniR] Loaded: dim=%d heads=%d layers=%d ffn=%d "
                "variant=%s quant=%s", dim, num_heads, num_layers, ffn_dim,
                model_variant, quantization or 'none')
    return mp
