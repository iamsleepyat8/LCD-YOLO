"""CMCA implementation used by the final LCD-YOLO model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("SM_block_v2a",)


class CoordAtt(nn.Module):
    """Coordinate Attention — preserves spatial position for small-object detection.

    Instead of compressing spatial dims to 1x1 (CBAM-style), CoordAtt preserves
    X and Y coordinate information through directional pooling.
    """

    def __init__(self, c, reduction=8):
        super().__init__()
        mid = max(c // reduction, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, c, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = y.split([h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))
        return identity * a_h * a_w


class CascadeBranch(nn.Module):
    """Single branch with BN+SiLU and optional depthwise-separable conv."""

    def __init__(self, c, kernel_size=3, dilation=1, use_dw=True):
        super().__init__()
        if use_dw and kernel_size > 1:
            pad = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
            self.conv = nn.Sequential(
                nn.Conv2d(c, c, kernel_size, 1, pad, groups=c, dilation=dilation, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
        elif kernel_size > 1:
            self.conv = nn.Sequential(
                nn.Conv2d(c, c, kernel_size, 1, kernel_size // 2, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class _SM_Cascade_Core(nn.Module):
    """Cascade-branch core used by CMCA.

    Three branches with cascade flow:
        branch1 (1x1)  <- split_0
        branch2 (DW 3x3) <- split_1 + branch1_output
        branch3 (DW 3x3 d=2, receptive=5) <- split_2 + branch2_output
    """

    def __init__(self, c, kernel_sizes=(1, 3, 5), mid_expand_ratio=2.0, use_dw=True):
        super().__init__()
        branch_num = len(kernel_sizes)
        self.branch_channel = max(int(c * mid_expand_ratio / branch_num), 1)
        hidden = self.branch_channel * branch_num
        self.branch_num = branch_num

        self.in_conv = Conv(c, hidden, k=1, s=1)

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            if k == 1:
                self.branches.append(CascadeBranch(self.branch_channel, kernel_size=1, use_dw=use_dw))
            elif k <= 3:
                self.branches.append(CascadeBranch(self.branch_channel, kernel_size=3, dilation=1, use_dw=use_dw))
            else:
                self.branches.append(CascadeBranch(self.branch_channel, kernel_size=3, dilation=2, use_dw=use_dw))

    def forward(self, x):
        y = self.in_conv(x)
        parts = torch.split(y, self.branch_channel, dim=1)
        outs = []
        for i, part in enumerate(parts):
            if i > 0 and outs:
                part = part + outs[-1]
            outs.append(self.branches[i](part))
        return outs


class SM_block_v2a(nn.Module):
    """CMCA with cascaded branches, dilated depthwise convolution, and coordinate attention."""

    def __init__(self, c1, c2=None, kernel_sizes=(1, 3, 5), mid_expand_ratio=2.0):
        super().__init__()
        c2 = c2 or c1
        self.core = _SM_Cascade_Core(c1, kernel_sizes=kernel_sizes, mid_expand_ratio=mid_expand_ratio, use_dw=True)
        hidden = self.core.branch_channel * self.core.branch_num

        self.ca = CoordAtt(hidden)
        self.proj = nn.Conv2d(hidden, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.short = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        residual = self.short(x)
        outs = self.core(x)
        out = torch.cat(outs, dim=1)
        out = self.ca(out)
        out = self.act(self.bn(self.proj(out)))
        return out + residual
