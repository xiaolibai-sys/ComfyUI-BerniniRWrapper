"""
Portable Wan 2.2 VAE implementation — copied from ComfyUI-WanVideoWrapper.

Removes the external dependency on WanVideoWrapper while maintaining full
compatibility with Wan 2.1 (z_dim=16) and Wan 2.2 (WanI38B, z_dim=48) VAEs.

Dependencies: einops, tqdm (both already present in ComfyUI environments).
"""
from __future__ import annotations


from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import comfy.model_management as mm
import comfy.ops
from comfy.utils import ProgressBar

from .log import get_logger as _get_logger

logger = _get_logger("VAE")
ops = comfy.ops.disable_weight_init

CACHE_T = 2


def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    return False


class CausalConv3d(ops.Conv3d):
    """Causal 3d convolution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)
        ) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                        ], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device), cache_x
                        ], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        if self.mode == 'downsample3d':
            if feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class Resample38(Resample):

    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super(Resample, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0, cpu_cache=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cpu_cache = cpu_cache
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        if self.cpu_cache:
            return self._forward_cpu_cache(x, feat_cache, feat_idx)
        return self._forward(x, feat_cache, feat_idx)

    def _forward(self, x, feat_cache=None, feat_idx=None):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                    ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h

    def _forward_cpu_cache(self, x, feat_cache=None, feat_idx=None):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cached_frame = feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device)
                    cache_x = torch.cat([cached_frame, cache_x], dim=2)
                prev_cache = feat_cache[idx].to(x.device) if feat_cache[idx] is not None else None
                x = layer(x, prev_cache)
                feat_cache[idx] = cache_x.to("cpu", non_blocking=True)
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Causal self-attention with a single head."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x + identity


class AvgDown3D(nn.Module):

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x):
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        x = x.view(B, C, T // self.factor_t, self.factor_t, H // self.factor_s, self.factor_s, W // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x, first_chunk=False):
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.size(0), self.out_channels, self.factor_t, self.factor_s, self.factor_s, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor_t, x.size(4) * self.factor_s, x.size(6) * self.factor_s)
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample38(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=None):
        # avg_shortcut creates a new tensor via F.pad+view+permute+mean
        # — it never mutates x, so no .clone() needed.
        x_shortcut = self.avg_shortcut(x)
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)
        return x + x_shortcut


class Up_ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False):
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2 if up_flag else 1)
        else:
            self.avg_shortcut = None
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample38(out_dim, mode=mode))
        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        # upsamples loop and avg_shortcut both create new tensors — no
        # in-place mutation of x, so .clone() is unnecessary.
        x_main = x
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        return x_main


class Encoder3d(nn.Module):

    def __init__(self, dim=128, z_dim=4, dim_mult=None, num_res_blocks=2,
                 attn_scales=None, temperal_downsample=None, dropout=0.0,
                 pruning_rate=0.0, cpu_cache=False):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [True, True, False]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.cpu_cache = cpu_cache
        dims = [int(dim * u * (1 - pruning_rate)) for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout, cpu_cache=cpu_cache))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout, cpu_cache=cpu_cache),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout, cpu_cache=cpu_cache))
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                    ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Encoder3d_38(nn.Module):

    def __init__(self, dim=128, z_dim=4, dim_mult=None, num_res_blocks=2,
                 attn_scales=None, temperal_downsample=None, dropout=0.0,
                 pruning_rate=0.0, cpu_cache=False):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.cpu_cache = cpu_cache
        dims = [int(dim * u * (1 - pruning_rate)) for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(Down_ResidualBlock(
                in_dim=in_dim, out_dim=out_dim, dropout=dropout, mult=num_res_blocks,
                temperal_downsample=t_down_flag, down_flag=i != len(dim_mult) - 1))
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout, cpu_cache=cpu_cache),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout, cpu_cache=cpu_cache))
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                    ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self, dim=128, z_dim=4, dim_mult=None, num_res_blocks=2,
                 attn_scales=None, temperal_upsample=None, dropout=0.0,
                 pruning_rate=0.0, cpu_cache=False):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_upsample is None:
            temperal_upsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.cpu_cache = cpu_cache
        dims = [int(dim * u * (1 - pruning_rate)) for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout, cpu_cache=cpu_cache),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout, cpu_cache=cpu_cache))
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout, cpu_cache=cpu_cache))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                    ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d_38(nn.Module):

    def __init__(self, dim=128, z_dim=4, dim_mult=None, num_res_blocks=2,
                 attn_scales=None, temperal_upsample=None, dropout=0.0,
                 pruning_rate=0.0, cpu_cache=False):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_upsample is None:
            temperal_upsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        dims = [int(dim * u * (1 - pruning_rate)) for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout, cpu_cache=cpu_cache),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout, cpu_cache=cpu_cache))
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(Up_ResidualBlock(
                in_dim=in_dim, out_dim=out_dim, dropout=dropout, mult=num_res_blocks + 1,
                temperal_upsample=t_up_flag, up_flag=i != len(dim_mult) - 1))
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx, first_chunk)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                    ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class VideoVAE_(nn.Module):

    def __init__(self, dim=96, z_dim=16, dim_mult=None, num_res_blocks=2,
                 attn_scales=None, temperal_downsample=None, dropout=0.0,
                 mean=None, inv_std=None, pruning_rate=0.0,
                 cpu_cache=False, verbose=False):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.mean = mean
        self.inv_std = inv_std
        self.verbose = verbose
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout, pruning_rate, cpu_cache=cpu_cache)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout, pruning_rate, cpu_cache=cpu_cache)

    def encode(self, x, pbar=True, sample=False):
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        if pbar:
            pbar = ProgressBar(iter_)
        try:
            torch.cuda.reset_peak_memory_stats(mm.get_torch_device())
        except Exception:
            pass

        # ── chunk 0: first frame only ──────────────────────────────────
        self._enc_conv_idx = [0]
        out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        if pbar:
            pbar.update(1)

        if iter_ == 1:
            self.clear_cache()
            mu, log_var = self.conv1(out).chunk(2, dim=1)
            mu = (mu - self.mean.to(mu)) * self.inv_std.to(mu)
            return mu

        # ── pre-allocate full output buffer ────────────────────────────
        full = torch.empty(
            out.shape[0], out.shape[1], iter_, out.shape[3], out.shape[4],
            device=out.device, dtype=out.dtype,
        )
        full[:, :, 0:1, :, :] = out

        for i in tqdm(range(1, iter_), desc="WanVAE encoding", disable=not pbar):
            self._enc_conv_idx = [0]
            chunk = self.encoder(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx,
            )
            full[:, :, i:i + 1, :, :] = chunk
            if pbar:
                pbar.update(1)

        self.clear_cache()
        mu, log_var = self.conv1(full).chunk(2, dim=1)
        mu = (mu - self.mean.to(mu)) * self.inv_std.to(mu)
        if self.verbose:
            logger.info(f"WanVAE encoded to {full.shape}")
        return mu

    def decode(self, z, pbar=True):
        z = z / self.inv_std.to(z) + self.mean.to(z)
        iter_ = z.shape[2]
        if pbar:
            pbar = ProgressBar(iter_)
        try:
            torch.cuda.reset_peak_memory_stats(mm.get_torch_device())
        except Exception:
            pass
        x = self.conv2(z)
        for i in tqdm(range(iter_), desc="WanVAE decoding", disable=not pbar):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
            if pbar:
                pbar.update(1)
        self.clear_cache()
        if self.verbose:
            logger.info(f"WanVAE decoded to {out.shape}")
        return out

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        return rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        return rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=patch_size, r=patch_size)
    raise ValueError(f"Invalid input shape: {x.shape}")


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        return rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        return rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=patch_size, r=patch_size)
    raise ValueError(f"Invalid input shape: {x.shape}")


class VideoVAE38_(VideoVAE_):

    def __init__(self, dim=160, z_dim=48, dec_dim=256, dim_mult=None,
                 num_res_blocks=2, attn_scales=None, temperal_downsample=None,
                 dropout=0.0, dtype=torch.bfloat16, mean=None, inv_std=None,
                 pruning_rate=0.0, cpu_cache=False, verbose=False):
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        super(VideoVAE_, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.dtype = dtype
        self.mean = mean
        self.inv_std = inv_std
        self.cpu_cache = cpu_cache
        self.verbose = verbose
        self.encoder = Encoder3d_38(dim, z_dim * 2, dim_mult, num_res_blocks,
                                    attn_scales, self.temperal_downsample, dropout, pruning_rate, cpu_cache=cpu_cache)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d_38(dec_dim, z_dim, dim_mult, num_res_blocks,
                                    attn_scales, self.temperal_upsample, dropout, pruning_rate, cpu_cache=cpu_cache)

    def encode(self, x, pbar=True, sample=False):
        self.clear_cache()
        try:
            torch.cuda.reset_peak_memory_stats(mm.get_torch_device())
        except Exception:
            pass
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        if pbar:
            pbar = ProgressBar(iter_)

        # ── chunk 0 ────────────────────────────────────────────────────
        self._enc_conv_idx = [0]
        out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        if pbar:
            pbar.update(1)

        if iter_ == 1:
            mu = self.conv1(out).chunk(2, dim=1)[0]
            mu = (mu - self.mean.to(mu)) * self.inv_std.to(mu)
            self.clear_cache()
            return mu

        # ── pre-allocate + slice-copy ───────────────────────────────────
        full = torch.empty(
            out.shape[0], out.shape[1], iter_, out.shape[3], out.shape[4],
            device=out.device, dtype=out.dtype,
        )
        full[:, :, 0:1, :, :] = out

        for i in tqdm(range(1, iter_), desc="WanVAE encoding", disable=not pbar):
            self._enc_conv_idx = [0]
            chunk = self.encoder(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx,
            )
            full[:, :, i:i + 1, :, :] = chunk
            if pbar:
                pbar.update(1)

        mu = self.conv1(full).chunk(2, dim=1)[0]
        mu = (mu - self.mean.to(mu)) * self.inv_std.to(mu)
        self.clear_cache()
        if self.verbose:
            logger.info(f"WanVAE38 encoded to {full.shape}")
        return mu

    def decode(self, z, pbar=True):
        self.clear_cache()
        try:
            torch.cuda.reset_peak_memory_stats(mm.get_torch_device())
        except Exception:
            pass
        z = z / self.inv_std.to(z) + self.mean.to(z)
        iter_ = z.shape[2]
        if pbar:
            pbar = ProgressBar(iter_)
        x = self.conv2(z)
        for i in tqdm(range(iter_), desc="WanVAE decoding", disable=not pbar):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map,
                                   feat_idx=self._conv_idx, first_chunk=True)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
            if pbar:
                pbar.update(1)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        if self.verbose:
            logger.info(f"WanVAE38 decoded to {out.shape}")
        return out


class WanVideoVAE(nn.Module):
    """Wan 2.1 / Wan2.2 video VAE (z_dim=16)."""

    def __init__(self, z_dim=16, dtype=torch.float32, pruning_rate=0.0, cpu_cache=False, verbose=False):
        super().__init__()
        self.dtype = dtype
        self.cpu_cache = cpu_cache
        self.verbose = verbose
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean).view(1, z_dim, 1, 1, 1)
        self.inv_std = (1.0 / torch.tensor(std)).view(1, z_dim, 1, 1, 1)
        self.z_dim = z_dim
        self.model = VideoVAE_(z_dim=z_dim, mean=self.mean, inv_std=self.inv_std,
                               pruning_rate=pruning_rate, cpu_cache=self.cpu_cache,
                               verbose=self.verbose).eval().requires_grad_(False)
        self.upsampling_factor = 8

    def encode(self, videos, device, tiled=False, tile_size=None, tile_stride=None, pbar=True, sample=False):
        self.model.clear_cache()
        # Only move to CPU when tiled (frees GPU before tile processing);
        # for non-tiled the round-trip is wasteful.
        videos = [v.to("cpu") if tiled else v for v in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                hs = self._tiled_encode(video, device, tile_size, tile_stride, pbar=pbar)
            else:
                hs = self.model.encode(video.to(device), pbar=pbar).float()
            hs = hs.squeeze(0)
            hidden_states.append(hs)
        return torch.stack(hidden_states)

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), pbar=True):
        self.model.clear_cache()
        hs_list = [hs.to("cpu") if tiled else hs for hs in hidden_states]
        videos = []
        for hs in hs_list:
            hs = hs.unsqueeze(0)
            if tiled:
                video = self._tiled_decode(hs, device, tile_size, tile_stride, pbar=pbar)
            else:
                video = self.model.decode(hs.to(device), pbar=pbar).float()
            video = video.squeeze(0)
            videos.append(video)
        return videos

    def _tiled_decode(self, hidden_states, device, tile_size, tile_stride, pbar=True):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))
        weight = None
        values = None
        if pbar:
            pbar = ProgressBar(len(tasks))
        for h, h_, w, w_ in tqdm(tasks, desc="VAE tiled decode"):
            batch = hidden_states[:, :, :, h:h_, w:w_].to(device)
            self.model.clear_cache()  # fresh cache per tile
            decoded = self.model.decode(batch, pbar=False).to("cpu")
            sf = self.upsampling_factor
            if weight is None:
                weight = torch.zeros((1, 1, decoded.shape[2], H * sf, W * sf), dtype=hidden_states.dtype, device="cpu")
            if values is None:
                values = torch.zeros((1, 3, decoded.shape[2], H * sf, W * sf), dtype=decoded.dtype, device="cpu")
            mask = self._build_blend_mask(
                decoded, (h == 0, h_ >= H, w == 0, w_ >= W),
                ((size_h - stride_h) * sf, (size_w - stride_w) * sf)).to(dtype=decoded.dtype, device="cpu")
            th = h * sf
            tw = w * sf
            values[:, :, :, th:th + decoded.shape[3], tw:tw + decoded.shape[4]] += decoded * mask
            weight[:, :, :, th:th + decoded.shape[3], tw:tw + decoded.shape[4]] += mask
            if pbar:
                pbar.update(1)
        values = (values / weight).float().clamp_(-1, 1)
        return values

    def _tiled_encode(self, video, device, tile_size, tile_stride, pbar=True):
        _, _, T, H, W = video.shape
        if tile_size is None:
            size_h, size_w = H // 2, W // 2
            stride_h, stride_w = size_h // 2, size_w // 2
        else:
            sf = self.upsampling_factor
            size_h, size_w = tile_size[0] * sf, tile_size[1] * sf
            stride_h, stride_w = tile_stride[0] * sf, tile_stride[1] * sf
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))
        out_T = (T + 3) // 4
        sf = self.upsampling_factor
        weight = torch.zeros((1, 1, out_T, H // sf, W // sf), dtype=video.dtype, device=device)
        values = torch.zeros((1, self.z_dim, out_T, H // sf, W // sf), dtype=video.dtype, device=device)
        if pbar:
            pbar = ProgressBar(len(tasks))
        for h, h_, w, w_ in tqdm(tasks, desc="VAE tiled encode"):
            batch = video[:, :, :, h:h_, w:w_].to(device)
            encoded = self.model.encode(batch).float()
            mask = self._build_blend_mask(
                encoded, (h == 0, h_ >= H, w == 0, w_ >= W),
                ((size_h - stride_h) // sf, (size_w - stride_w) // sf)).to(dtype=video.dtype, device=device)
            th = h // sf
            tw = w // sf
            values[:, :, :, th:th + encoded.shape[3], tw:tw + encoded.shape[4]] += encoded * mask
            weight[:, :, :, th:th + encoded.shape[3], tw:tw + encoded.shape[4]] += mask
            if pbar:
                pbar.update(1)
        return values / weight

    def _build_blend_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h_mask = torch.ones(H)
        if not is_bound[0]:
            h_mask[:border_width[0]] = (torch.arange(border_width[0]) + 1) / border_width[0]
        if not is_bound[1]:
            h_mask[-border_width[0]:] = torch.flip((torch.arange(border_width[0]) + 1) / border_width[0], dims=(0,))
        w_mask = torch.ones(W)
        if not is_bound[2]:
            w_mask[:border_width[1]] = (torch.arange(border_width[1]) + 1) / border_width[1]
        if not is_bound[3]:
            w_mask[-border_width[1]:] = torch.flip((torch.arange(border_width[1]) + 1) / border_width[1], dims=(0,))
        h_mask = repeat(h_mask, "H -> H W", H=H, W=W)
        w_mask = repeat(w_mask, "W -> H W", H=H, W=W)
        mask = torch.stack([h_mask, w_mask]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def clear_cache(self):
        self.model.clear_cache()


class WanVideoVAE38(WanVideoVAE):
    """WanI38B VAE (z_dim=48, upsampling_factor=16)."""

    def __init__(self, z_dim=48, dim=160, dtype=torch.bfloat16, pruning_rate=0.0, cpu_cache=False, verbose=False):
        super(WanVideoVAE, self).__init__()
        mean = [-0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
                -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
                -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
                -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
                -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
                0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667]
        std = [0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
               0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
               0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
               0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
               0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
               0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744]
        self.mean = torch.tensor(mean).view(1, z_dim, 1, 1, 1)
        self.inv_std = (1.0 / torch.tensor(std)).view(1, z_dim, 1, 1, 1)
        self.dtype = dtype
        self.z_dim = z_dim
        self.cpu_cache = cpu_cache
        self.verbose = verbose
        self.model = VideoVAE38_(z_dim=z_dim, dim=dim, dtype=dtype, mean=self.mean, inv_std=self.inv_std,
                                 pruning_rate=pruning_rate, cpu_cache=cpu_cache,
                                 verbose=verbose).eval().requires_grad_(False)
        self.upsampling_factor = 16
