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
        extract_levels: Sequence[int] = (0, 1, 2, 3),
        out_channels: int = 3,
        pooling: bool = True,
        concat_skip: bool = False,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "border",
    ):
        super().__init__()

        self.net = LocalNet(
            spatial_dims=3,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            extract_levels=list(extract_levels),
            out_channels=out_channels,
            pooling=pooling,
            concat_skip=concat_skip,
        )
        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, D, H, W)
        ddf = self.net(x)  # (B, 3, D, H, W)
        warped = self.warp(moving, ddf)
        return warped, ddf


@register_model("localnet3d")
def create_localnet3d(
    image_size: Sequence[int],
    in_channels: int = 2,
    num_channel_initial: int = 16,
    extract_levels: Sequence[int] = (0, 1, 2, 3),
    out_channels: int = 3,
    pooling: bool = True,
    concat_skip: bool = False,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "border",
) -> nn.Module:
    return LocalNet3D(
        image_size=image_size,
        in_channels=in_channels,
        num_channel_initial=num_channel_initial,
        extract_levels=extract_levels,
        out_channels=out_channels,
        pooling=pooling,
        concat_skip=concat_skip,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
    )