"""PyTorch model builders for MFuRe-style multi-backbone classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models

from .torch_layers import FusionResidualBlockTorch


@dataclass(frozen=True)
class TorchModelConfig:
    """Typed configuration for PyTorch MFuRe-style classifier construction."""

    num_classes: int = 4
    dropout_rate: float = 0.1
    dense_units: int = 256
    freeze_backbones: bool = True


class MFuReTorch(nn.Module):
    """PyTorch counterpart for a multi-fused residual classifier."""

    def __init__(self, config: TorchModelConfig):
        super().__init__()
        self.config = config

        mobile = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.mobile_features = mobile.features
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])

        if config.freeze_backbones:
            for module in (self.mobile_features, self.resnet_features):
                for param in module.parameters():
                    param.requires_grad = False

        mobile_channels = 1280
        resnet_channels = 2048
        fusion_channels = 512

        self.fusion = FusionResidualBlockTorch(
            left_channels=mobile_channels,
            right_channels=resnet_channels,
            out_channels=fusion_channels,
            dropout_rate=config.dropout_rate,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.AlphaDropout(p=config.dropout_rate),
            nn.Linear(fusion_channels, config.dense_units),
            nn.SELU(inplace=True),
            nn.Linear(config.dense_units, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mobile = self.mobile_features(x)
        resnet = self.resnet_features(x)
        fused = self.fusion(mobile, resnet)
        pooled = self.pool(fused)
        return self.classifier(pooled)


def build_mfure_torch_classifier(config: TorchModelConfig | None = None) -> MFuReTorch:
    """Factory for a production-ready PyTorch MFuRe-style classifier."""

    return MFuReTorch(config or TorchModelConfig())
