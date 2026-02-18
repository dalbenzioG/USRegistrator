"""UNetReg3D – UNet-based 3D registration network (MONAI UNet + Warp)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.nets import UNet
from monai.networks.blocks import Warp

from .registry import register_model


class UNetReg3D(nn.Module):
    """
    UNet-based 3D registration network (MONAI UNet + Warp).

    This predicts a dense displacement field with a standard UNet.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        in_channels: int = 2,
        out_channels: int = 3,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        strides: Sequence[int] = (2, 2, 2, 2),
        num_res_units: int = 2,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "border",
    ):
        super().__init__()

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=list(channels),
            strides=list(strides),
            num_res_units=num_res_units,
        )
        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)
        ddf = self.net(x)  # (B,3,D,H,W)
        warped = self.warp(moving, ddf)
        return warped, ddf


@register_model("unetreg3d")
def create_unetreg3d(
    image_size: Sequence[int],
    in_channels: int = 2,
    out_channels: int = 3,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "border",
) -> nn.Module:
    """
    Factory for UNetReg3D.

    Config example:
        model:
          name: unetreg3d
          in_channels: 2
          out_channels: 3
          channels: [16, 32, 64, 128, 256]
          strides: [2, 2, 2, 2]
          num_res_units: 2
    """
    return UNetReg3D(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
    )
