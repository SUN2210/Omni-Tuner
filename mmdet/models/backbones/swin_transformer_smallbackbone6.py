
import os
from typing import Dict, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import math
import torch.distributed as dist
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv.runner import BaseModule
# NOTE: The following imports might need adjustment based on your environment.
# Assuming mmcv_custom, mmdet, and ..builder are in your PYTHONPATH.
try:
    from mmcv_custom import load_checkpoint
    from mmdet.utils import get_root_logger
    from ..builder import BACKBONES
except (ImportError, ModuleNotFoundError):
    # Dummy registry for standalone execution
    class DummyRegistry:
        def __init__(self):
            self.module_dict = {}
        def register_module(self):
            def decorator(cls):
                self.module_dict[cls.__name__] = cls
                return cls
            return decorator
    BACKBONES = DummyRegistry()
    get_root_logger = lambda: print # Simple print for logger
    load_checkpoint = lambda model, path, strict, logger: None


class ASPPInspiredModule(BaseModule):

    def __init__(self, in_dim, hidden_dim=64, dilation_rates=(1, 3, 5), use_global_branch=True):
        super().__init__()
        dilation_rates = tuple(dilation_rates)
        self.use_global_branch = use_global_branch
        self.down_proj = nn.Linear(in_dim, hidden_dim)
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=rate,
                      dilation=rate, groups=hidden_dim, bias=False)
            for rate in dilation_rates
        ])
        if self.use_global_branch:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.global_conv = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        else:
            self.global_pool = None
            self.global_conv = None
        self.fusion_conv = nn.Conv2d(
            hidden_dim * (len(dilation_rates) + (1 if self.use_global_branch else 0)),
            hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        identity = x
        x = self.norm(x) * self.gamma + identity
        x = self.down_proj(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        multi_scale_features = [conv(x) for conv in self.multi_scale_convs]
        if self.use_global_branch:
            global_feat = self.global_pool(x)
            global_feat = self.global_conv(global_feat)
            global_feat = F.interpolate(
                global_feat, size=(H, W), mode='bilinear', align_corners=False)
            multi_scale_features.append(global_feat)
        fused = torch.cat(multi_scale_features, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.bn(fused)
        fused = self.act(fused)
        fused = fused.permute(0, 2, 3, 1).reshape(B, N, -1)
        fused = self.up_proj(fused)
        fused = self.dropout(fused)
        return identity + fused


class Mlp(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FNetBlock(nn.Module):
    """Apply 2D FFT (frequency prompt) as in reference implementation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("FNetBlock expects input with at least 2 dimensions")
        freq = torch.fft.fft(x, dim=-1)
        freq = torch.fft.fft(freq, dim=-2)
        return freq.real


def window_partition(x, window_size):
    """将特征图划分为窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """将窗口恢复为特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., num_prompts=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_prompts = num_prompts
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        if self.num_prompts > 0:
            N_img = N - self.num_prompts
            attn[:, :, self.num_prompts:, self.num_prompts:] = attn[:, :, self.num_prompts:, self.num_prompts:] + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            if self.num_prompts > 0:
                N_img = N - self.num_prompts
                attn_img = attn[:, :, self.num_prompts:, self.num_prompts:].view(B_ // nW, nW, self.num_heads, N_img, N_img)
                attn_img = attn_img + mask.unsqueeze(1).unsqueeze(0)
                attn[:, :, self.num_prompts:, self.num_prompts:] = attn_img.view(-1, self.num_heads, N_img, N_img)
            else:
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_aspp_module=True, num_prompts=0,
                 adapter1_dilation_rates=(1, 3, 5), adapter1_use_global_branch=True, adapter1_hidden_dim=64,
                 adapter2_dilation_rates=(1, 3, 5), adapter2_use_global_branch=True, adapter2_hidden_dim=64,
                 use_fourier_prompts=False, fourier_mix_cfg=None, fourier_ema_momentum=0.99):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_aspp_module = use_aspp_module
        self.num_prompts = num_prompts
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window-size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            num_prompts=num_prompts)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        
        if num_prompts > 0:
            self.my_module_prompt_embeddings = nn.Parameter(torch.zeros(1, num_prompts, dim))
            val = math.sqrt(6. / float(3 * 16 * 16 + dim))
            nn.init.uniform_(self.my_module_prompt_embeddings, -val, val)
        else:
            self.my_module_prompt_embeddings = None

        self.use_fourier_prompts = bool(use_fourier_prompts and num_prompts > 0)
        if self.use_fourier_prompts:
            self.my_module_fourier_transform = FNetBlock()
            momentum = float(fourier_ema_momentum)
            momentum = max(0.0, min(momentum, 1.0))
            self.fourier_ema_momentum = momentum
            self._init_fourier_mix(fourier_mix_cfg)
        else:
            self.my_module_fourier_transform = None
            self.fourier_ema_momentum = 0.0
            self.fourier_mix_mode = 'none'
        
        # --- OPTIMIZATION START ---
        # Conditionally create ASPP adapter modules only if they are going to be used.
        # This prevents registering unused parameters.
        if self.use_aspp_module:
            self.my_module_adapter_1 = ASPPInspiredModule(
                dim, hidden_dim=adapter1_hidden_dim,
                dilation_rates=adapter1_dilation_rates,
                use_global_branch=adapter1_use_global_branch)
            self.my_module_adapter_2 = ASPPInspiredModule(
                dim, hidden_dim=adapter2_hidden_dim,
                dilation_rates=adapter2_dilation_rates,
                use_global_branch=adapter2_use_global_branch)
        else:
            self.my_module_adapter_1 = None
            self.my_module_adapter_2 = None
        # --- OPTIMIZATION END ---


    def forward(self, x, mask):
        H, W = self.H, self.W
        B, L, C = x.shape

        if self.num_prompts > 0:
            prompts = self.my_module_prompt_embeddings.expand(B, -1, -1)
            if self.use_fourier_prompts:
                alpha = self._get_fourier_alpha(device=prompts.device, dtype=prompts.dtype)
                alpha = alpha.view(1, self.num_prompts, 1)
                prompts_freq = self.my_module_fourier_transform(prompts)
                prompts = prompts_freq * alpha + prompts * (1.0 - alpha)
            x_with_prompts = torch.cat([prompts, x], dim=1)
        else:
            x_with_prompts = x

        shortcut = x_with_prompts
        x_norm = self.norm1(x_with_prompts)

        if self.num_prompts > 0:
            prompts_norm = x_norm[:, :self.num_prompts, :]
            x_img = x_norm[:, self.num_prompts:, :]
        else:
            x_img = x_norm

        assert L == H * W, "input feature has wrong size"
        x_img = x_img.view(B, H, W, C)

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x_img = F.pad(x_img.permute(0, 3, 1, 2), (0, pad_r, 0, pad_b)).permute(0, 2, 3, 1)

        H_pad, W_pad = H + pad_b, W + pad_r

        if self.shift_size > 0:
            shifted_x = torch.roll(x_img, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_img

        x_windows = window_partition(shifted_x, self.window_size)
        nW = x_windows.shape[0] // B

        if self.num_prompts > 0:
            prompts_norm_win = prompts_norm.unsqueeze(1).expand(-1, nW, -1, -1).reshape(nW * B, self.num_prompts, C)
            x_windows = torch.cat([prompts_norm_win, x_windows], dim=1)

        attn_output_windows = self.attn(x_windows, mask=mask)

        if self.num_prompts > 0:
            updated_prompts_windows = attn_output_windows[:, :self.num_prompts, :]
            updated_img_windows = attn_output_windows[:, self.num_prompts:, :]
            updated_prompts = updated_prompts_windows.view(B, nW, self.num_prompts, C).mean(dim=1)
        else:
            updated_img_windows = attn_output_windows

        updated_img_windows = updated_img_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(updated_img_windows, self.window_size, H_pad, W_pad)

        if self.shift_size > 0:
            x_img_updated_padded = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_img_updated_padded = shifted_x

        if pad_b > 0 or pad_r > 0:
            x_img_updated = x_img_updated_padded[:, :H, :W, :].contiguous()
        else:
            x_img_updated = x_img_updated_padded

        x_img_updated = x_img_updated.view(B, H * W, C)

        if self.num_prompts > 0:
            x_updated_with_prompts = torch.cat([updated_prompts, x_img_updated], dim=1)
        else:
            x_updated_with_prompts = x_img_updated

        x_residual1 = shortcut + self.drop_path(x_updated_with_prompts)

        x_residual2 = x_residual1 + self.drop_path(self.mlp(self.norm2(x_residual1)))

        if self.num_prompts > 0:
            x_img_out = x_residual2[:, self.num_prompts:, :]
        else:
            x_img_out = x_residual2

        if self.use_aspp_module:
            x_img_out = self.my_module_adapter_1(x_img_out, (H, W))
            x_img_out = self.my_module_adapter_2(x_img_out, (H, W))

        return x_img_out

    def _init_fourier_mix(self, mix_cfg):
        if self.num_prompts <= 0:
            self.fourier_mix_mode = 'none'
            self.my_module_fourier_mix_param = None
            return

        cfg = 0.5 if mix_cfg is None else mix_cfg
        eps = 1e-6

        self.fourier_mix_mode = 'static'
        self.my_module_fourier_mix_param = None

        # Clean existing buffers if any (first registration only in practice)
        if 'my_module_fourier_mix_constant' in self._buffers:
            self._buffers.pop('my_module_fourier_mix_constant')
        if 'my_module_fourier_mix_ema' in self._buffers:
            self._buffers.pop('my_module_fourier_mix_ema')

        # EMA or learnable shortcut tokens expressed via strings
        if isinstance(cfg, str):
            token = cfg.strip()
            upper = token.upper()
            if upper.startswith('EMA@'):
                try:
                    init = float(token.split('@', 1)[1])
                except ValueError as exc:
                    raise ValueError(f'Invalid EMA fourier mix config "{cfg}"') from exc
                init = max(0.0, min(1.0, init))
                init_for_param = max(eps, min(1.0 - eps, init))
                logit = math.log(init_for_param / (1.0 - init_for_param))
                param_init = torch.full((self.num_prompts,), logit, dtype=torch.float32)
                self.my_module_fourier_mix_param = nn.Parameter(param_init)
                ema_buffer = torch.full((self.num_prompts,), init, dtype=torch.float32)
                self.register_buffer('my_module_fourier_mix_ema', ema_buffer)
                self.fourier_mix_mode = 'ema'
                return
            if token.lower() in {'x', 'learnable'}:
                self.my_module_fourier_mix_param = nn.Parameter(torch.zeros(self.num_prompts, dtype=torch.float32))
                self.fourier_mix_mode = 'learnable'
                return
            # Fallthrough: try to parse as a numeric string
            try:
                cfg = float(token)
            except ValueError as exc:
                raise ValueError(f'Invalid fourier mix config "{cfg}"') from exc

        # Handle iterable constants (per-token specification)
        if isinstance(cfg, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(cfg, (float, int)):
            values = torch.as_tensor(cfg, dtype=torch.float32)
            if values.numel() == 1:
                values = values.repeat(self.num_prompts)
            elif values.numel() != self.num_prompts:
                raise ValueError(
                    f'fourier mix config expects {self.num_prompts} values but got {values.numel()}')
            values = torch.clamp(values, 0.0, 1.0)
            self.register_buffer('my_module_fourier_mix_constant', values)
            return

        try:
            value = float(cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid fourier mix config "{cfg}"') from exc
        value = max(0.0, min(1.0, value))
        constant = torch.full((self.num_prompts,), value, dtype=torch.float32)
        self.register_buffer('my_module_fourier_mix_constant', constant)

    def _get_fourier_alpha(self, device, dtype) -> torch.Tensor:
        if self.fourier_mix_mode == 'static':
            return self.my_module_fourier_mix_constant.to(device=device, dtype=dtype)
        if self.fourier_mix_mode == 'ema':
            raw = torch.sigmoid(self.my_module_fourier_mix_param)
            ema_val = self.my_module_fourier_mix_ema
            momentum = self.fourier_ema_momentum
            if self.training:
                updated = momentum * ema_val + (1.0 - momentum) * raw.detach()
                self.my_module_fourier_mix_ema.copy_(updated)
            blended = momentum * self.my_module_fourier_mix_ema + (1.0 - momentum) * raw
            return blended.to(device=device, dtype=dtype)
        if self.fourier_mix_mode == 'learnable':
            return torch.sigmoid(self.my_module_fourier_mix_param).to(device=device, dtype=dtype)
        return torch.zeros(self.num_prompts, dtype=dtype, device=device)


class PatchMerging(nn.Module):
    """ Patch Merging Layer (Simplified, no prompt handling). """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 3, 1, 2), (0, W % 2, 0, H % 2)).permute(0, 2, 3, 1)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x_merged = torch.cat([x0, x1, x2, x3], -1)
        x_merged = x_merged.view(B, -1, 4 * C)

        x_merged = self.norm(x_merged)
        x_merged = self.reduction(x_merged)
        return x_merged


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage. """
    def __init__(self, dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_aspp_in_deep_layers=True, num_prompts=0,
                 skip_half_block=True,
                 adapter1_dilation_rates=(1, 3, 5), adapter1_use_global_branch=True, adapter1_hidden_dim=64,
                 adapter2_dilation_rates=(1, 3, 5), adapter2_use_global_branch=True, adapter2_hidden_dim=64,
                 use_fourier_prompts=False, fourier_mix_cfg=None, fourier_ema_momentum=0.99): # <-- [MODIFIED] Added new argument
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, 
                use_aspp_module=use_aspp_in_deep_layers and ((not skip_half_block) or (i % 2 != 1)),
                # --- [END MODIFICATION] ---
                num_prompts=num_prompts,
                adapter1_dilation_rates=adapter1_dilation_rates,
                adapter1_use_global_branch=adapter1_use_global_branch,
                adapter1_hidden_dim=adapter1_hidden_dim,
                adapter2_dilation_rates=adapter2_dilation_rates,
                adapter2_use_global_branch=adapter2_use_global_branch,
                adapter2_hidden_dim=adapter2_hidden_dim,
                use_fourier_prompts=use_fourier_prompts,
                fourier_mix_cfg=fourier_mix_cfg,
                fourier_ema_momentum=fourier_ema_momentum)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """图像到Patch嵌入"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


@BACKBONES.register_module()
class SwinTransformer_smallbackbone6(nn.Module):

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 out_indices=(0, 1, 2, 3), frozen_stages=-1,
                 use_checkpoint=False,
                 num_prompts=48,  
                 use_aspp_in_stages=[True, True, True, True],
                 skip_half_block_in_stages=[False, False, True, False],
                 adapter1_dilation_rates=(1, 3, 5), adapter1_use_global_branch=True, adapter1_hidden_dim=64,
                 adapter2_dilation_rates=(1, 3, 5), adapter2_use_global_branch=True, adapter2_hidden_dim=64,
                 use_fourier_prompts_in_stages=[True, True, True, False],  
                 fourier_prompt_mix_cfg="learnable", 
                 freeze_fourier_mix_params_on_load=False,   
                 fourier_prompt_ema_momentum=0.6,  
                 halve_prompts_in_deep_stages=True,
                 style_adapter=None):

        super().__init__()
        self.freeze_fourier_mix_params_on_load = freeze_fourier_mix_params_on_load

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if use_fourier_prompts_in_stages is None:
            use_fourier_prompts_in_stages = [False] * self.num_layers
        elif len(use_fourier_prompts_in_stages) != self.num_layers:
            raise ValueError('use_fourier_prompts_in_stages must match num_layers in length')
        else:
            use_fourier_prompts_in_stages = [bool(flag) for flag in use_fourier_prompts_in_stages]

        def _broadcast_stage_cfg(cfg, name, default_value=None):
            if isinstance(cfg, (list, tuple)):
                if len(cfg) != self.num_layers:
                    raise ValueError(f'{name} must match num_layers in length')
                return list(cfg)
            if cfg is None:
                return [default_value for _ in range(self.num_layers)]
            return [cfg for _ in range(self.num_layers)]

        fourier_mix_cfg_list = _broadcast_stage_cfg(fourier_prompt_mix_cfg, 'fourier_prompt_mix_cfg', None)
        fourier_ema_momentum_list = _broadcast_stage_cfg(fourier_prompt_ema_momentum, 'fourier_prompt_ema_momentum', 0.99)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if halve_prompts_in_deep_stages:
                stage_num_prompts = num_prompts if i_layer < 2 else num_prompts // 2
            else:
                stage_num_prompts = num_prompts
            
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_aspp_in_deep_layers=use_aspp_in_stages[i_layer],
                skip_half_block=skip_half_block_in_stages[i_layer], # <-- [MODIFIED] Pass the flag
                num_prompts=stage_num_prompts,
                adapter1_dilation_rates=adapter1_dilation_rates,
                adapter1_use_global_branch=adapter1_use_global_branch,
                adapter1_hidden_dim=adapter1_hidden_dim,
                adapter2_dilation_rates=adapter2_dilation_rates,
                adapter2_use_global_branch=adapter2_use_global_branch,
                adapter2_hidden_dim=adapter2_hidden_dim,
                use_fourier_prompts=use_fourier_prompts_in_stages[i_layer],
                fourier_mix_cfg=fourier_mix_cfg_list[i_layer],
                fourier_ema_momentum=fourier_ema_momentum_list[i_layer]
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._style_stage_names = {i: f'stage{i}' for i in range(self.num_layers)}
        self._style_stage_pre_keys = {i: f'{name}_pre' for i, name in self._style_stage_names.items()}
        self._style_stage_post_keys = {i: f'{name}_post' for i, name in self._style_stage_names.items()}
        self._style_patch_key = 'patch_embed'
        self.style_propagation_mode = 'full'
        self._init_style_adapter(style_adapter)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _resolve_style_stage_key(self, key: Optional[str]) -> Optional[str]:
        if key is None:
            return None
        key = str(key).strip()
        if not key:
            return None
        if key == self._style_patch_key:
            return key
        if not key.startswith('stage'):
            return None

        suffix = None
        if key.endswith('_pre'):
            suffix = 'pre'
            base = key[:-4]
        elif key.endswith('_post'):
            suffix = 'post'
            base = key[:-5]
        else:
            suffix = 'pre'
            base = key

        if len(base) <= 5 or not base[5:].isdigit():
            return None

        idx = int(base[5:])
        if idx < 0 or idx >= self.num_layers:
            return None

        base_name = self._style_stage_names.get(idx, f'stage{idx}')
        if suffix == 'pre':
            return self._style_stage_pre_keys[idx]
        if suffix == 'post':
            return self._style_stage_post_keys[idx]
        return None

    def _configure_mix_alpha_controls(
        self,
        scope_value,
        mix_alpha_cfg,
        mix_alpha_init_cfg,
        key_order: Tuple[str, ...],
        logger=None,
    ) -> None:
        """Normalize mix_alpha configuration and set up shared/per-stage controls."""

        self.style_mix_alpha_scope = 'shared'
        self.style_mix_alpha_per_stage = {}
        self.style_mix_alpha_shared = 1.0
        self.style_mix_alpha = 1.0
        self.my_module_style_mix_alpha = None

        def _to_float(value, default, context: str):
            try:
                return float(value)
            except (TypeError, ValueError):
                if logger:
                    logger.warning(f'Invalid {context} value "{value}"; falling back to {default}.')
                return default

        def _clamp_unit(val: float) -> float:
            return max(0.0, min(val, 1.0))

        def _normalize_stage_key(raw_key):
            if raw_key is None:
                return None
            text = str(raw_key).strip()
            if not text:
                return None
            normalized = self._resolve_style_stage_key(text)
            if normalized is not None:
                return normalized
            if text == self._style_patch_key:
                return text
            return None

        scope_token = 'shared'
        if scope_value is not None:
            token = scope_value if isinstance(scope_value, str) else str(scope_value)
            token = token.strip().lower().replace('-', '_')
            if token in {'shared', 'per_stage'}:
                scope_token = token
            else:
                if logger:
                    logger.warning(f'Unknown mix_alpha_scope "{scope_value}"; defaulting to "shared".')

        stage_keys = list(key_order) if key_order else sorted(self.style_apply_to)
        if scope_token == 'per_stage' and not stage_keys:
            scope_token = 'shared'
            if logger:
                logger.warning('mix_alpha_scope "per_stage" requested but no style stages are active; using shared instead.')

        if scope_token == 'shared':
            effective_cfg = mix_alpha_cfg
            if isinstance(effective_cfg, dict):
                # dictionaries are only meaningful for per-stage; pick first value to preserve behavior
                try:
                    effective_cfg = next(iter(effective_cfg.values()))
                    if logger:
                        logger.warning('mix_alpha dict supplied while using shared scope; using the first value instead.')
                except StopIteration:
                    effective_cfg = 1.0

            if isinstance(effective_cfg, str) and effective_cfg.lower() == 'x':
                init_default = 1.0
                if mix_alpha_init_cfg is not None:
                    init_default = _clamp_unit(_to_float(mix_alpha_init_cfg, 1.0, 'mix_alpha_init'))
                param = nn.Parameter(torch.tensor(init_default, dtype=torch.float32))
                self.my_module_style_mix_alpha = param
                self.style_mix_alpha_shared = param
                self.style_mix_alpha = param
            else:
                value = _clamp_unit(_to_float(effective_cfg, 1.0, 'mix_alpha'))
                self.style_mix_alpha_shared = value
                self.style_mix_alpha = value
            self.style_mix_alpha_scope = 'shared'
            return

        # per-stage configuration
        self.style_mix_alpha_scope = 'per_stage'
        per_stage_map: Dict[str, Union[float, torch.nn.Parameter]] = {}

        learnable = isinstance(mix_alpha_cfg, str) and mix_alpha_cfg.lower() == 'x'
        if learnable:
            init_default = 1.0
            init_lookup: Dict[str, float] = {}
            if isinstance(mix_alpha_init_cfg, dict):
                for raw_key, raw_val in mix_alpha_init_cfg.items():
                    canonical = _normalize_stage_key(raw_key)
                    if canonical is None:
                        if logger:
                            logger.warning(f'Unknown mix_alpha_init stage "{raw_key}"; skipping.')
                        continue
                    init_lookup[canonical] = _clamp_unit(_to_float(raw_val, 1.0, f'mix_alpha_init[{raw_key}]'))
            elif mix_alpha_init_cfg is not None:
                init_default = _clamp_unit(_to_float(mix_alpha_init_cfg, 1.0, 'mix_alpha_init'))

            for stage_key in stage_keys:
                start_val = init_lookup.get(stage_key, init_default)
                param = nn.Parameter(torch.tensor(start_val, dtype=torch.float32))
                param_name = f'my_module_style_mix_alpha_{stage_key}'.replace('.', '_')
                setattr(self, param_name, param)
                per_stage_map[stage_key] = param
            self.style_mix_alpha_shared = init_default
        else:
            base_default = 1.0
            value_lookup: Dict[str, float] = {}
            if isinstance(mix_alpha_cfg, dict):
                for raw_key, raw_val in mix_alpha_cfg.items():
                    canonical = _normalize_stage_key(raw_key)
                    if canonical is None:
                        if logger:
                            logger.warning(f'Unknown mix_alpha stage "{raw_key}"; skipping.')
                        continue
                    value_lookup[canonical] = _clamp_unit(_to_float(raw_val, 1.0, f'mix_alpha[{raw_key}]'))
            else:
                base_default = _clamp_unit(_to_float(mix_alpha_cfg, 1.0, 'mix_alpha'))

            for stage_key in stage_keys:
                per_stage_map[stage_key] = value_lookup.get(stage_key, base_default)
            self.style_mix_alpha_shared = base_default

        self.style_mix_alpha_per_stage = per_stage_map
        self.style_mix_alpha = self.style_mix_alpha_shared
        if logger and per_stage_map:
            joined = ', '.join(f'{k}' for k in per_stage_map.keys())
            logger.info(f'Configured per-stage mix_alpha for stages: {joined}')

    def _get_mix_alpha_value(self, stage_key: Optional[str]):
        if self.style_mix_alpha_scope == 'per_stage':
            value = self.style_mix_alpha_per_stage.get(stage_key)
            if value is not None:
                return value
        return self.style_mix_alpha_shared

    def _init_style_adapter(self, style_cfg):
        self.style_adapter_enabled = False
        self.style_buffers = {}
        self.style_apply_to = set()
        self.style_momentum = 1.0
        self.style_eps = 1e-6
        self.style_min_std = 1e-6
        self.style_mix_alpha_scope: str = 'shared'
        self.style_mix_alpha_shared: Union[float, torch.nn.Parameter] = 1.0
        self.style_mix_alpha_per_stage: Dict[str, Union[float, torch.nn.Parameter]] = {}
        self.style_mix_alpha: Union[float, torch.nn.Parameter] = 1.0
        self.my_module_style_mix_alpha: Optional[torch.nn.Parameter] = None
        self._style_active_key_order: Tuple[str, ...] = tuple()
        self._style_collect_pre_default = True
        self._style_collect_post_default = True
        if style_cfg is None:
            return

        logger = None
        try:
            logger = get_root_logger()
        except Exception:
            logger = None

        style_cfg = style_cfg.copy()
        raw_mix_alpha_scope = style_cfg.pop('mix_alpha_scope', None)
        legacy_mix_alpha_mode = style_cfg.pop('mix_alpha_mode', None)
        if raw_mix_alpha_scope is None and legacy_mix_alpha_mode is not None:
            raw_mix_alpha_scope = legacy_mix_alpha_mode
            if logger:
                logger.info('style_adapter.mix_alpha_mode is deprecated; use mix_alpha_scope instead.')
        mix_alpha_cfg = style_cfg.get('mix_alpha', 1.0)
        mix_alpha_init_cfg = style_cfg.get('mix_alpha_init', None)
        raw_stage_norm_mode = style_cfg.pop('stage_norm_mode', None)
        parsed_norm_mode = None
        if isinstance(raw_stage_norm_mode, str):
            parsed_norm_mode = raw_stage_norm_mode.strip().lower() or None
        elif raw_stage_norm_mode is not None:
            parsed_norm_mode = str(raw_stage_norm_mode).strip().lower() or None

        valid_norm_modes = {'pre', 'post', 'both', 'none'}
        if parsed_norm_mode is not None and parsed_norm_mode not in valid_norm_modes:
            if logger:
                logger.warning(f'Invalid style_adapter.stage_norm_mode "{parsed_norm_mode}"; ignoring.')
            parsed_norm_mode = None

        if parsed_norm_mode is not None and logger:
            logger.info('style_adapter.stage_norm_mode is deprecated; enumerate specific stages in style_adapter.apply_to instead.')

        if parsed_norm_mode == 'pre' or parsed_norm_mode is None:
            self._style_collect_pre_default = True
            self._style_collect_post_default = False
        elif parsed_norm_mode == 'post':
            self._style_collect_pre_default = False
            self._style_collect_post_default = True
        elif parsed_norm_mode == 'both':
            self._style_collect_pre_default = True
            self._style_collect_post_default = True
        elif parsed_norm_mode == 'none':
            self._style_collect_pre_default = False
            self._style_collect_post_default = False

        propagation_mode = style_cfg.get('propagation', 'full')
        if isinstance(propagation_mode, str):
            propagation_mode = propagation_mode.lower()
        else:
            propagation_mode = 'full'
        valid_modes = {'full', 'fpn_only', 'none'}
        if propagation_mode not in valid_modes:
            if logger:
                logger.warning(f'Invalid style propagation mode "{propagation_mode}", defaulting to "full".')
            propagation_mode = 'full'
        self.style_propagation_mode = propagation_mode
        if self.style_propagation_mode == 'none':
            if logger:
                logger.info('Style adapter propagation set to "none"; skipping style correction initialization.')
            return

        self.style_momentum = float(style_cfg.get('momentum', 1.0))
        self.style_momentum = max(0.0, min(self.style_momentum, 1.0))
        self.style_eps = float(style_cfg.get('eps', 1e-6))
        self.style_min_std = float(style_cfg.get('min_std', 1e-6))
        self.style_min_std = max(self.style_min_std, 1e-6)
        stats_file = style_cfg.get('stats_file')
        if not stats_file:
            if style_cfg.get('enabled', False) and logger:
                logger.warning('Style adapter enabled but no stats_file provided; skipping style correction.')
            return

        stats_file = os.path.expanduser(stats_file)
        if not os.path.isfile(stats_file):
            if logger:
                logger.warning(f'Style stats file not found: {stats_file}. Style adapter disabled.')
            return

        try:
            payload = torch.load(stats_file, map_location='cpu')
        except Exception as exc:
            if logger:
                logger.warning(f'Failed to load style stats from {stats_file}: {exc}')
            return

        stats_dict = payload.get('style_stats', payload)
        if not isinstance(stats_dict, dict):
            if logger:
                logger.warning(f'Style stats file {stats_file} has unexpected format; expected dict.')
            return

        available_stats = set(stats_dict.keys())
        requested_apply_to = style_cfg.get('apply_to')
        normalized_keys = []
        if requested_apply_to is None:
            if self._style_patch_key in available_stats:
                normalized_keys.append(self._style_patch_key)
            if self._style_collect_pre_default:
                for idx in range(self.num_layers):
                    key = self._style_stage_pre_keys[idx]
                    if key in available_stats:
                        normalized_keys.append(key)
            if self._style_collect_post_default:
                for idx in range(self.num_layers):
                    key = self._style_stage_post_keys[idx]
                    if key in available_stats:
                        normalized_keys.append(key)
        else:
            for raw_key in requested_apply_to:
                normalized = self._resolve_style_stage_key(raw_key)
                if normalized is None:
                    if logger:
                        logger.warning(f'Unknown style stage key "{raw_key}"; skipping.')
                    continue
                if normalized not in available_stats:
                    if logger:
                        logger.warning(f'Style stats missing for requested key "{normalized}" in {stats_file}.')
                    continue
                normalized_keys.append(normalized)

        # remove duplicates while preserving order
        seen_keys = set()
        filtered_keys = []
        for key in normalized_keys:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            filtered_keys.append(key)

        for key in filtered_keys:
            stats_lookup_key = key
            if stats_lookup_key not in stats_dict and stats_lookup_key.endswith('_pre'):
                legacy_key = stats_lookup_key[:-4]
                if legacy_key in stats_dict:
                    stats_lookup_key = legacy_key
            if key == self._style_patch_key:
                stage_dim = self.embed_dim
            elif key.endswith('_pre') or key.endswith('_post'):
                base_key = key.rsplit('_', 1)[0]
                if len(base_key) <= 5 or not base_key[5:].isdigit():
                    continue
                idx = int(base_key[5:])
                if idx < 0 or idx >= len(self.num_features):
                    continue
                stage_dim = self.num_features[idx]
            else:
                continue

            stage_stats = stats_dict.get(stats_lookup_key)
            if stage_stats is None:
                if logger:
                    logger.warning(f'Style stats missing for stage "{key}" in {stats_file}.')
                continue

            mu = stage_stats.get('mean') if isinstance(stage_stats, dict) else None
            if mu is None:
                mu = stage_stats.get('mu') if isinstance(stage_stats, dict) else None
            sigma = stage_stats.get('std') if isinstance(stage_stats, dict) else None
            if sigma is None:
                sigma = stage_stats.get('sigma') if isinstance(stage_stats, dict) else None

            if mu is None or sigma is None:
                if logger:
                    logger.warning(f'Style stats for stage "{key}" missing mean/std entries.')
                continue

            mu = torch.as_tensor(mu, dtype=torch.float32)
            sigma = torch.as_tensor(sigma, dtype=torch.float32)
            if mu.dim() == 0:
                mu = mu.unsqueeze(0)
            if sigma.dim() == 0:
                sigma = sigma.unsqueeze(0)

            if stage_dim is not None and mu.numel() != stage_dim:
                if logger:
                    logger.warning(
                        f'Style stats for stage "{key}" expect {stage_dim} channels but got {mu.numel()}.'
                    )
                continue

            buffer_mu_name = f'style_{key}_mu'
            buffer_sigma_name = f'style_{key}_sigma'
            self.register_buffer(buffer_mu_name, mu.clone())
            self.register_buffer(buffer_sigma_name, sigma.clone())
            self.style_buffers[key] = {
                'mu': getattr(self, buffer_mu_name),
                'sigma': getattr(self, buffer_sigma_name),
            }
            self.style_apply_to.add(key)

        self._style_active_key_order = tuple(filtered_keys)
        self._configure_mix_alpha_controls(
            raw_mix_alpha_scope,
            mix_alpha_cfg,
            mix_alpha_init_cfg,
            self._style_active_key_order,
            logger=logger,
        )

        if self.style_buffers:
            self.style_adapter_enabled = True
            if logger:
                stages_text = ', '.join(sorted(self.style_apply_to))
                logger.info(f'Loaded style stats from {stats_file} for stages: {stages_text}')

    def _style_adapt_feature(self, stage_key, feat):
        if not self.style_adapter_enabled:
            return feat
        buffers = self.style_buffers.get(stage_key)
        if buffers is None:
            return feat

        mu_buf = buffers['mu']
        sigma_buf = buffers['sigma']
        channels = feat.size(1)
        if mu_buf.numel() != channels or sigma_buf.numel() != channels:
            return feat

        device = feat.device
        per_sample_mean = feat.mean(dim=(2, 3), keepdim=True)
        per_sample_var = feat.var(dim=(2, 3), keepdim=True, unbiased=False)
        per_sample_std = torch.sqrt(torch.clamp(per_sample_var, min=self.style_min_std))

        target_mu = mu_buf.view(1, -1, 1, 1).to(device, non_blocking=True)
        target_std = torch.clamp(sigma_buf.view(1, -1, 1, 1), min=self.style_min_std).to(device, non_blocking=True)

        adapted_core = (feat - per_sample_mean) / (per_sample_std + self.style_eps) * target_std + target_mu

        mix_alpha = self._get_mix_alpha_value(stage_key)
        if isinstance(mix_alpha, torch.nn.Parameter):
            alpha = torch.clamp(mix_alpha, 0.0, 1.0)
        elif isinstance(mix_alpha, torch.Tensor):
            alpha = torch.clamp(mix_alpha, 0.0, 1.0)
        else:
            alpha = max(0.0, min(float(mix_alpha), 1.0))

        adapted = adapted_core * alpha + feat * (1.0 - alpha)

        if self.training and self.style_momentum < 1.0:
            batch_mu = per_sample_mean.mean(dim=0).squeeze().detach()
            batch_std = per_sample_std.mean(dim=0).squeeze().detach()

            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(batch_mu)
                dist.all_reduce(batch_std)
                batch_mu /= world_size
                batch_std /= world_size

            target_device = mu_buf.device
            if batch_mu.device != target_device:
                batch_mu = batch_mu.to(target_device, non_blocking=True)
                batch_std = batch_std.to(target_device, non_blocking=True)

            momentum = self.style_momentum
            with torch.no_grad():
                mu_buf.mul_(momentum).add_(batch_mu, alpha=1 - momentum)
                sigma_buf.mul_(momentum).add_(batch_std, alpha=1 - momentum)

        return adapted

    def forward_style_features(
        self,
        x: torch.Tensor,
        stages: Optional[Set[str]] = None,
        include_patch: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Collect pre-normalization backbone features for style statistics.

        Args:
            x (Tensor): Input image tensor of shape (N, C, H, W).
            stages (Optional[Set[str]]): Specific stage keys to collect, e.g. {'stage1'}.
            include_patch (bool): Whether to include patch embedding outputs.

        Returns:
            Dict[str, Tensor]: Mapping from stage key to feature map in (N, C, H, W).
        """

        stage_filter = set(stages) if stages is not None else None
        include_patch_flag = include_patch
        if stage_filter is not None and self._style_patch_key in stage_filter:
            include_patch_flag = True

        alias_without_suffix: Set[str] = set()
        if stage_filter is not None:
            alias_without_suffix = {
                name for name in stage_filter
                if name in self._style_stage_names.values()
            }

        collected: Dict[str, torch.Tensor] = {}

        patch_feat = self.patch_embed(x)
        if include_patch_flag:
            collected[self._style_patch_key] = patch_feat

        Wh, Ww = patch_feat.size(2), patch_feat.size(3)
        tokens = patch_feat.flatten(2).transpose(1, 2)
        tokens = self.pos_drop(tokens)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_stage, H, W, tokens, Wh, Ww = layer(tokens, Wh, Ww)
            base_key = self._style_stage_names.get(i, f'stage{i}')
            pre_key = self._style_stage_pre_keys.get(i, f'{base_key}_pre')
            post_key = self._style_stage_post_keys.get(i, f'{base_key}_post')

            if stage_filter is None:
                need_pre = self._style_collect_pre_default
                need_post = self._style_collect_post_default
            else:
                need_pre = (
                    pre_key in stage_filter
                    or base_key in alias_without_suffix
                )
                need_post = (
                    post_key in stage_filter
                    or base_key in alias_without_suffix
                )

            norm_layer = getattr(self, f'norm{i}', None)
            if norm_layer is None:
                need_post = False

            if need_pre:
                feat_pre = x_stage.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                collected[pre_key] = feat_pre

            if need_post:
                x_norm = norm_layer(x_stage)
                feat_post = x_norm.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                collected[post_key] = feat_post

        return collected

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            if logger:
                logger.info(f'loading model from: {pretrained}')
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            if self.freeze_fourier_mix_params_on_load:
                self._freeze_fourier_mix_parameters()
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        for name, param in self.named_parameters():
            if 'my_module' not in name:
                param.requires_grad = False

    def _freeze_fourier_mix_parameters(self):
        for module in self.modules():
            param = getattr(module, 'my_module_fourier_mix_param', None)
            if isinstance(param, nn.Parameter) and param.requires_grad:
                param.requires_grad = False

    def forward(self, x):
        patch_feat = self.patch_embed(x)
        if (
            self.style_propagation_mode == 'full'
            and self._style_patch_key in self.style_apply_to
        ):
            patch_feat = self._style_adapt_feature(self._style_patch_key, patch_feat)

        Wh, Ww = patch_feat.size(2), patch_feat.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            tokens = (patch_feat + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            tokens = patch_feat.flatten(2).transpose(1, 2)
        tokens = self.pos_drop(tokens)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_stage, H, W, x_down_orig, next_H_orig, next_W_orig = layer(tokens, Wh, Ww)

            base_key = self._style_stage_names.get(i, f'stage{i}')
            pre_key = self._style_stage_pre_keys.get(i, f'{base_key}_pre')
            post_key = self._style_stage_post_keys.get(i, f'{base_key}_post')

            apply_pre = (
                pre_key in self.style_apply_to
                and self.style_propagation_mode != 'none'
            )
            apply_post = (
                post_key in self.style_apply_to
                and self.style_propagation_mode != 'none'
            )
            propagate_pre = apply_pre and self.style_propagation_mode == 'full'

            tokens_for_norm = x_stage
            x_down = x_down_orig
            next_H = next_H_orig
            next_W = next_W_orig

            if apply_pre:
                stage_feat_pre = x_stage.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                adapted_pre_feat = self._style_adapt_feature(pre_key, stage_feat_pre)
                adapted_pre_tokens = adapted_pre_feat.permute(0, 2, 3, 1).reshape(-1, H * W, self.num_features[i])
                tokens_for_norm = adapted_pre_tokens
                if propagate_pre:
                    if layer.downsample is not None:
                        x_down = layer.downsample(adapted_pre_tokens, H, W)
                        next_H, next_W = (H + 1) // 2, (W + 1) // 2
                    else:
                        x_down = adapted_pre_tokens
                        next_H, next_W = H, W

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_norm = norm_layer(tokens_for_norm)
                out = x_norm.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                if apply_post:
                    out = self._style_adapt_feature(post_key, out)
                outs.append(out)

            tokens = x_down
            Wh, Ww = next_H, next_W

        return tuple(outs)

    def train(self, mode=True):
        super(SwinTransformer_smallbackbone6, self).train(mode)
        self._freeze_stages()
