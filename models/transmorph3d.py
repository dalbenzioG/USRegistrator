"""TransMorph3D – TransMorph-style Swin Transformer registration (MONAI SwinUNETR + Warp)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.nets import SwinUNETR
from monai.networks.blocks import Warp

from .registry import register_model


class TransMorph3D(nn.Module):
    """
    TransMorph-style 3D registration network (MONAI SwinUNETR + Warp).

    Follows the TransMorph design (Chen et al., 2022): a Swin Transformer
    encoder with a convolutional decoder predicts a dense displacement
    field from the concatenated moving/fixed pair, which is then applied
    to the moving image with a spatial warp.

    Components:
      - SwinUNETR: Swin Transformer encoder + CNN decoder predicting the ddf
      - Warp: applies ddf to the moving image

    Note: SwinUNETR downsamples 5x, so each spatial dimension of the
    input must be divisible by 32 (e.g. 64, 96, 128).

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
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "border",
    ):
        super().__init__()

        if len(image_size) != 3:
            raise ValueError(
                f"image_size must be length 3 (D, H, W), got {image_size}"
            )

        self.image_size = [int(s) for s in image_size]
        for dim in self.image_size:
            if dim % 32 != 0:
                raise ValueError(
                    f"TransMorph3D requires each spatial dimension to be divisible "
                    f"by 32, got image_size={self.image_size}."
                )

        self.net = SwinUNETR(
            in_channels=2,  # moving + fixed concatenated along channel dim
            out_channels=3,
            feature_size=feature_size,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

        # Near-zero-init the ddf head so training starts close to the
        # identity transform (standard practice for registration networks).
        for m in self.net.out.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=1e-5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        ddf = self.net(x)  # (B, 3, D, H, W)
        warped = self.warp(moving, ddf)
        return warped, ddf


@register_model("transmorph3d")
def create_transmorph3d(
    image_size: Sequence[int],
    feature_size: int = 48,
    depths: Sequence[int] = (2, 2, 2, 2),
    num_heads: Sequence[int] = (3, 6, 12, 24),
    window_size: int = 7,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
    use_checkpoint: bool = False,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "border",
) -> nn.Module:
    """
    Factory for TransMorph3D.

    Config example:
        model:
          name: transmorph3d
          feature_size: 48
          depths: [2, 2, 2, 2]
          num_heads: [3, 6, 12, 24]
          window_size: 7
          warp_mode: bilinear
          warp_padding_mode: border
    """
    return TransMorph3D(
        image_size=image_size,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
    )
