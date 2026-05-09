from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision import models


class MultiViewEfficientNet(nn.Module):
    """Shared-image encoder + multi-view aggregation.

    Input shape: [batch, views, channels, height, width]
    Output shape: [batch], raw logits for BCEWithLogitsLoss.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.25) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Could not load EfficientNet weights metadata: {exc}. Using random init.")
                weights = None

        try:
            backbone = models.efficientnet_b0(weights=weights)
        except Exception as exc:
            warnings.warn(f"Could not initialize pretrained EfficientNet-B0: {exc}. Using random init.")
            backbone = models.efficientnet_b0(weights=None)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features * 2),
            nn.Dropout(dropout),
            nn.Linear(in_features * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)
        feats = self.backbone(x)  # [B*V, F]
        feats = feats.view(b, v, -1)

        mean_feats = feats.mean(dim=1)
        max_feats = feats.max(dim=1).values
        agg = torch.cat([mean_feats, max_feats], dim=1)
        logits = self.classifier(agg).squeeze(1)
        return logits
