"""GlobalNet3D – MONAI GlobalNet + Warp for 3D registration."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.nets import GlobalNet
from monai.networks.blocks import Warp

from .registry import register_model


class GlobalNet3D(nn.Module):
    """
    MONAI GlobalNet + Warp for 3D registration.

    Components:
      - GlobalNet: predicts 3D displacement field (ddf)
      - Warp: applies ddf to the moving image

    Inputs:
      moving: (B, 1, D, H, W)
      fixed:  (B, 1, D, H, W)

    Outputs:
      warped: (B, 1, D, H, W)  -- moving warped into fixed space
      ddf:    (B, 3, D, H, W)  -- displacement field
    """

    def __init__(
        self,
        image_size: Sequence[int],
        num_channel_initial: int = 16,
        depth: int = 3,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "border",
        max_disp: float = 0.2,
    ):
        super().__init__()

        if len(image_size) != 3:
            raise ValueError(
                f"image_size must be length 3 (D, H, W), got {image_size}"
            )

        self.image_size = [int(s) for s in image_size]
        self.max_disp = float(max_disp)

        self.net = GlobalNet(
            image_size=self.image_size,
            spatial_dims=3,
            in_channels=2,  # moving + fixed concatenated along channel dim
            num_channel_initial=num_channel_initial,
            depth=depth,
        )

        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

        # Zero-init / near-zero-init convs to make initial deformation gentle
        for m in self.net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=1e-5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print(f"[GlobalNet3D] Using max_disp = {self.max_disp}")

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        """
        Args:
            moving: (B, 1, D, H, W)
            fixed:  (B, 1, D, H, W)

        Returns:
            warped: (B, 1, D, H, W)
            ddf:    (B, 3, D, H, W)
        """
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, D, H, W)

        # Predict raw displacement
        ddf_raw = self.net(x)  # (B, 3, D, H, W)

        # Restrict DVF into a configurable numeric range
        ddf = torch.tanh(ddf_raw) * self.max_disp

        # Warp moving -> fixed space
        warped = self.warp(moving, ddf)

        return warped, ddf


@register_model("globalnet3d")
def create_globalnet3d(
    image_size: Sequence[int],
    num_channel_initial: int = 16,
    depth: int = 3,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "border",
    max_disp: float = 0.2,
) -> nn.Module:
    """
    Factory for GlobalNet3D.

    Config example:
        model:
          name: globalnet3d
          num_channel_initial: 16
          depth: 3
          warp_mode: bilinear
          warp_padding_mode: border
          max_disp: 0.2
    """
    return GlobalNet3D(
        image_size=image_size,
        num_channel_initial=num_channel_initial,
        depth=depth,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
        max_disp=max_disp,
    )
