# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DS-MSBlock implementation used by the final LCD-YOLO model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv, DWConv

__all__ = ("MSBlock",)


class SeparableConv(nn.Module):
    """Depthwise + pointwise conv used by DS-MSBlock variants."""

    def __init__(self, c: int, k: int):
        super().__init__()
        self.dw = DWConv(c, c, k=k, s=1)
        self.pw = Conv(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class BranchGate(nn.Module):
    """Per-branch gate: GAP -> 1x1 -> SiLU -> 1x1 -> Sigmoid."""

    def __init__(self, c: int):
        super().__init__()
        hidden = max(c // 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, hidden, kernel_size=1, stride=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        return self.sigmoid(w)


class MSBlockUnit(nn.Module):
    """Single final DS-MSBlock unit."""

    def __init__(
        self,
        c: int,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
        mid_expand_ratio: float = 2.0,
        start_branch_id: int = 1,
        shortcut: bool = True,
        f4_kernel: int = 3,
        shared_kernel: int = 3,
    ):
        super().__init__()
        assert len(kernel_sizes) > 0, "kernel_sizes must not be empty"

        self.start_branch_id = start_branch_id
        self.shortcut = shortcut

        branch_num = len(kernel_sizes)
        branch_channel = max(int(c * mid_expand_ratio / branch_num), 1)
        hidden = branch_channel * branch_num
        self.branch_channels = [branch_channel] * branch_num

        self.in_conv = Conv(c, hidden, k=1, s=1)
        self.shared_stem = SeparableConv(hidden, k=shared_kernel)
        self.mid_convs = nn.ModuleList(
            [
                nn.Identity()
                if k == 1
                else Conv(self.branch_channels[i], self.branch_channels[i], k=1, s=1)
                for i, k in enumerate(kernel_sizes)
            ]
        )
        self.f4_proj = Conv(c, branch_channel, k=1, s=1)
        self.f4_conv = SeparableConv(branch_channel, k=f4_kernel)
        self.gates = nn.ModuleList([BranchGate(branch_channel) for _ in range(branch_num + 1)])
        self.out_conv = Conv(branch_channel, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.in_conv(x)
        y = self.shared_stem(y)
        parts = list(torch.split(y, self.branch_channels, dim=1))

        outs = []
        for i, part in enumerate(parts):
            if i >= self.start_branch_id and outs:
                part = part + outs[-1]
            part = self.mid_convs[i](part)
            outs.append(part)

        f4_base = self.f4_proj(x)
        branches = [*outs, f4_base + self.f4_conv(f4_base)]
        weighted = [branch * gate(branch) for branch, gate in zip(branches, self.gates)]
        y = self.out_conv(sum(weighted))

        return x + y if self.shortcut else y


class MSBlock(nn.Module):
    """Parser-friendly wrapper for the final DS-MSBlock."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
        down_ratio: float = 1.0,
        mid_expand_ratio: float = 2.0,
        start_branch_id: int = 1,
        shortcut: bool = False,
        f4_kernel: int = 3,
        shared_kernel: int = 3,
    ):
        super().__init__()
        _ = down_ratio  # Keep parser arg compatibility.
        self.proj = Conv(c1, c2, k=1, s=1)
        self.blocks = nn.Sequential(
            *[
                MSBlockUnit(
                    c2,
                    kernel_sizes=tuple(kernel_sizes),
                    mid_expand_ratio=mid_expand_ratio,
                    start_branch_id=start_branch_id,
                    shortcut=shortcut,
                    f4_kernel=f4_kernel,
                    shared_kernel=shared_kernel,
                )
                for _ in range(max(n, 1))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))
