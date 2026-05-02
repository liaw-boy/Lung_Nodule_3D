"""3D ResNet18 backbone with optional CBAM blocks.

Adapted from Hara et al. "Can Spatiotemporal 3D CNNs Retrace the History of
2D CNNs and ImageNet?" (CVPR 2018) — pure 3D Conv variant tuned for nodule
volumes (D=16, H=W=128 default).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam3d import CBAM3D


def conv3x3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_cbam: bool = True):
        super().__init__()
        self.conv1 = conv3x3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = conv3x3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.cbam = CBAM3D(out_ch) if use_cbam else nn.Identity()
        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = out + identity
        return F.relu(out, inplace=True)


class ResNet3D18(nn.Module):
    """3D ResNet18 with CBAM, returns (feature_vec, intermediate_maps)."""

    def __init__(self, in_channels: int = 1, base_width: int = 32,
                 use_cbam: bool = True, feature_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_width, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.layer1 = self._make_layer(base_width,     base_width,     blocks=2,
                                       stride=1, use_cbam=use_cbam)
        self.layer2 = self._make_layer(base_width,     base_width * 2, blocks=2,
                                       stride=2, use_cbam=use_cbam)
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, blocks=2,
                                       stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(base_width * 4, base_width * 8, blocks=2,
                                       stride=2, use_cbam=use_cbam)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_width * 8, feature_dim)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int,
                    use_cbam: bool) -> nn.Sequential:
        layers = [BasicBlock3D(in_ch, out_ch, stride=stride, use_cbam=use_cbam)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_ch, out_ch, stride=1, use_cbam=use_cbam))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x).flatten(1)
        return F.relu(self.fc(x), inplace=True)
