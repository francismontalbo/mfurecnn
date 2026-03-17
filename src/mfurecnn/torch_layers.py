"""PyTorch custom layers for MFuRe-style architectures."""

from __future__ import annotations

import torch
from torch import nn


class FusionResidualBlockTorch(nn.Module):
    """Fusion residual block aligned with the paper's multi-fused residual idea.

    Inputs:
        left:  (B, C1, H, W)
        right: (B, C2, H, W)

    Output:
        (B, out_channels, H, W)
    """

    def __init__(self, left_channels: int, right_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        in_channels = left_channels + right_channels

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=dropout_rate),
        )
        self.skip = nn.Conv2d(left_channels, out_channels, kernel_size=1, bias=False)
        self.out_activation = nn.SELU(inplace=True)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([left, right], dim=1)
        fused = self.fuse(merged)
        residual = self.skip(left)
        return self.out_activation(fused + residual)
