"""LCAT implementation used by the final LCD-YOLO model."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("SwinTransformerBlock_v2a",)


def _dwconv_block(c, k=3):
    return nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False)


class _Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class SwinTransformerBlock_v2a(nn.Module):
    """LCAT with pre-norm attention and depthwise-convolution positional encoding."""

    def __init__(self, c1, c2=None, num_heads=0, dropout=0.1):
        super().__init__()
        c2 = c2 or c1
        self.c1 = c1
        self.c2 = c2

        if num_heads <= 0:
            num_heads = max(c2 // 32, 1)
        self.num_heads = num_heads

        self.input_proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.dwconv = _dwconv_block(c2, k=3)
        self.norm1 = nn.LayerNorm(c2)
        self.attn = nn.MultiheadAttention(c2, num_heads=num_heads, dropout=dropout, batch_first=False)

        self.norm2 = nn.LayerNorm(c2)
        self.mlp = _Mlp(c2, dropout=dropout)

        self.out_proj = nn.Conv2d(c2, c2, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        x = self.input_proj(x)

        x = x + self.dwconv(x)

        x_flat = x.flatten(2).permute(2, 0, 1)

        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        x_norm2 = self.norm2(x_flat)
        x_flat = x_flat + self.mlp(x_norm2)

        x = x_flat.permute(1, 2, 0).view(B, self.c2, H, W)
        x = self.out_proj(x)

        if C == self.c2:
            x = x + shortcut
        return x
