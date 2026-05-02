"""3D nodule classifier with Attribute Feedback (JIMI 2022 idea, ported to 3D).

Forward returns (logits, aux_preds):
  - logits   : (B, 2)  — benign / malignant
  - aux_preds: (B, n_aux)  — sigmoid'd attributes (lobulation/spiculation/margin)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .resnet3d import ResNet3D18


class NoduleClassifier3D(nn.Module):
    def __init__(self, in_channels: int = 1, feature_dim: int = 256, n_aux: int = 3,
                 use_attribute_feedback: bool = True):
        super().__init__()
        self.use_attribute_feedback = use_attribute_feedback
        self.backbone = ResNet3D18(in_channels=in_channels, feature_dim=feature_dim)

        # Aux head — predicts clinical attributes from shared features
        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_aux),
            nn.Sigmoid(),
        )

        # Compress shared feature → 128-d cls feature, then concat aux feedback
        self.cls_proj = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        if use_attribute_feedback:
            self.malignancy_head = nn.Linear(128 + n_aux, 2)
        else:
            self.malignancy_head = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor):
        shared = self.backbone(x)              # (B, feature_dim)
        aux_preds = self.aux_head(shared)       # (B, n_aux)
        cls_feat = self.cls_proj(shared)        # (B, 128)
        if self.use_attribute_feedback:
            cls_feat = torch.cat([cls_feat, aux_preds], dim=1)
        logits = self.malignancy_head(cls_feat) # (B, 2)
        return logits, aux_preds
