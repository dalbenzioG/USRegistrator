"""LocalNet3D – MONAI LocalNet + Warp for 3D registration."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.nets import LocalNet
from monai.networks.blocks import Warp

from .registry import register_model


class LocalNet3D(nn.Module):
    """
    MONAI LocalNet + Warp for 3D registration.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        in_channels: int = 2,
        num_channel_initial: int = 16,
        depth: int = 3,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "border",
    ):
        super().__init__()

        self.net = LocalNet(
            spatial_dims=3,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            depth=depth,
        )
        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, D, H, W)
        ddf = self.net(x)                      # (B, 3, D, H, W)
        warped = self.warp(moving, ddf)
        return warped, ddf


@register_model("localnet3d")
def create_localnet3d(
    image_size: Sequence[int],
    in_channels: int = 2,
    num_channel_initial: int = 16,
    depth: int = 3,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "border",
) -> nn.Module:
    """
    Factory for LocalNet3D.

    Config example:
        model:
          name: localnet3d
          in_channels: 2
          num_channel_initial: 16
          depth: 3
    """
    return LocalNet3D(
        image_size=image_size,
        in_channels=in_channels,
        num_channel_initial=num_channel_initial,
        depth=depth,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
    )
