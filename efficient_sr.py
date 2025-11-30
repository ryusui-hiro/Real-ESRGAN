from typing import Tuple

import torch
from torch import nn


class ECALite(nn.Module):
    """Lightweight channel attention using 1x1 convolutions.

    This module uses global average pooling followed by a pointwise convolution
    and a sigmoid gate. It stores the latest attention map for inspection in
    tests.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.latest_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.avg_pool(x)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        self.latest_attention = attn.detach()
        return x * attn


class ERBlock(nn.Module):
    """Efficient Residual Block with depthwise separable convolutions and ECA-Lite."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.attention = ECALite(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise(x)
        out = self.act(out)
        out = self.pointwise(out)
        out = self.attention(out)
        return residual + out


class PixelShuffleBlock(nn.Module):
    def __init__(self, channels: int, upscale: int = 2, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (upscale**2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.shuffle(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class EfficientSRNet(nn.Module):
    """High-efficiency depthwise-separable super-resolution network.

    Args:
        in_channels: Number of input channels (default: 3 for RGB images).
        num_features: Base number of feature channels (NF).
        num_blocks: Number of ERBlocks in the deep feature extractor.
        upscale: Upscaling factor. Supports 2 or 4 (implemented as two x2 stages for x4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 8,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        if upscale not in (2, 4):
            raise ValueError("EfficientSRNet currently supports upscale factors of 2 or 4.")

        self.shallow = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ERBlock(num_features) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Two PixelShuffle stages to reach x4, or one if upscale=2
        activation = nn.ReLU(inplace=True)
        self.up1 = PixelShuffleBlock(num_features, upscale=2, activation=activation)
        self.up2 = PixelShuffleBlock(num_features, upscale=2, activation=activation) if upscale == 4 else None

        self.final_conv = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.shallow(x)
        deep = self.blocks(f0)
        deep = self.trunk_conv(deep)
        deep = deep + f0

        out = self.up1(deep)
        if self.up2 is not None:
            out = self.up2(out)

        out = self.final_conv(out)
        return out

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def infer_shape(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> Tuple[int, ...]:
        dummy = torch.zeros(input_shape)
        with torch.no_grad():
            out = model(dummy)
        return tuple(out.shape)
