"""3D U-Net model for brain tumor segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Two consecutive 3D convolutions with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
                track_running_stats=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
                track_running_stats=False,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    """Downsampling block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv3D(out_channels + skip_channels, out_channels)

    @staticmethod
    def _match_size(
        x: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Pad/crop x so its spatial dimensions match reference."""

        target = reference.shape[2:]
        current = x.shape[2:]

        # Crop if necessary.
        slices = [slice(None), slice(None)]
        for current_size, target_size in zip(current, target):
            if current_size > target_size:
                start = (current_size - target_size) // 2
                slices.append(slice(start, start + target_size))
            else:
                slices.append(slice(None))

        x = x[tuple(slices)]

        # Pad if necessary.
        padding = []
        for current_size, target_size in reversed(
            list(zip(x.shape[2:], target))
        ):
            difference = target_size - current_size
            padding.extend(
                [
                    difference // 2,
                    difference - difference // 2,
                ]
            )

        if any(padding):
            x = nn.functional.pad(x, padding)

        return x

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        x = self._match_size(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric medical image segmentation."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv3D(
            in_channels,
            base_channels,
        )

        self.down1 = Down3D(
            base_channels,
            base_channels * 2,
        )

        self.down2 = Down3D(
            base_channels * 2,
            base_channels * 4,
        )

        self.down3 = Down3D(
            base_channels * 4,
            base_channels * 8,
        )

        # Bottleneck
        self.bottleneck = DoubleConv3D(
            base_channels * 8,
            base_channels * 16,
        )

        # Decoder
        #
        # x5: 1/8 resolution
        # x4: 1/8 resolution
        # x3: 1/4 resolution
        # x2: 1/2 resolution
        # x1: full resolution
        #
        # Therefore:
        # x5 -> x3
        # -> x2
        # -> x1

        self.up1 = Up3D(
            base_channels * 16,
            base_channels * 4,
            base_channels * 8,
        )

        self.up2 = Up3D(
            base_channels * 8,
            base_channels * 2,
            base_channels * 4,
        )

        self.up3 = Up3D(
            base_channels * 4,
            base_channels,
            base_channels * 2,
        )

        self.out_conv = nn.Conv3d(
            base_channels * 2,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the U-Net forward pass."""

        if x.ndim != 5:
            raise ValueError(
                "UNet3D expects a 5D input tensor with shape "
                "(N, C, D, H, W)."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[1]}."
            )

        # Encoder
        x1 = self.inc(x)       # 128³
        x2 = self.down1(x1)    # 64³
        x3 = self.down2(x2)    # 32³
        x4 = self.down3(x3)    # 16³
        x5 = self.bottleneck(x4)  # 16³

        # Decoder
        x = self.up1(x5, x3)   # 32³
        x = self.up2(x, x2)    # 64³
        x = self.up3(x, x1)    # 128³

        return self.out_conv(x)


# Backwards-compatible alias.
UNet = UNet3D


__all__ = [
    "DoubleConv3D",
    "Down3D",
    "Up3D",
    "UNet3D",
    "UNet",
]