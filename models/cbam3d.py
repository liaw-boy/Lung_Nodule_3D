"""3D CBAM attention — channel + spatial gating for volumetric features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention3D(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_ch // reduction)
        self.fc1 = nn.Conv3d(in_ch, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(hidden, in_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool3d(x, 1)
        mx = F.adaptive_max_pool3d(x, 1)
        att = torch.sigmoid(self.fc2(F.relu(self.fc1(avg))) +
                            self.fc2(F.relu(self.fc1(mx))))
        return x * att


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM3D(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16, spatial_k: int = 7):
        super().__init__()
        self.channel = ChannelAttention3D(in_ch, reduction)
        self.spatial = SpatialAttention3D(spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))
