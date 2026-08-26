"""VoxelMorph3D – MONAI VoxelMorph + VoxelMorphUNet registration network."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.nets import VoxelMorph, VoxelMorphUNet

from .registry import register_model


class VoxelMorph3D(nn.Module):
    """
    VoxelMorph-based 3D registration network (MONAI VoxelMorph + VoxelMorphUNet).

    The VoxelMorphUNet backbone predicts a dense field from the concatenated
    (moving, fixed) pair, and MONAI's VoxelMorph wrapper turns it into a DDF and
    warps the moving image.

    ``integration_steps`` selects the variant:
      * ``0`` – no integration; the backbone field is used directly as the DDF
        (non-diffeomorphic, CVPR 2018 variant). This is the repo default because
        it matches the ``gt_dvf`` produced by ``datasets/deepreg_synthetic.py``
        and the behaviour of GlobalNet3D / LocalNet3D / UNetReg3D.
      * ``> 0`` – the backbone field is treated as a stationary velocity field
        and integrated by scaling-and-squaring (``2 ** integration_steps``
        effective sub-steps) into a diffeomorphic DDF (MICCAI 2018 variant).

    Note: MONAI's ``VoxelMorph`` hardcodes ``Warp(mode="bilinear",
    padding_mode="zeros")`` and applies no ``tanh`` clamp to the field, so
    ``warp_mode`` / ``warp_padding_mode`` / ``max_disp`` are intentionally not
    exposed here – they would be silently ignored. Its ``Warp`` also works in
    voxel units, whereas ``gt_dvf`` is in normalized grid-sample coordinates,
    so the ``epe`` metric and the ``dvf_weight * MSE(pred_dvf, gt_dvf)`` term
    are on a different scale than for GlobalNet3D. Keep ``dvf_weight`` low and
    rely on the LNCC image term.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        in_channels: int = 2,
        unet_out_channels: int = 32,
        channels: Sequence[int] = (16, 32, 32, 32, 32, 32),
        final_conv_channels: Sequence[int] = (16, 16),
        integration_steps: int = 0,
        half_res: bool = False,
    ):
        super().__init__()

        backbone = VoxelMorphUNet(
            spatial_dims=3,
            in_channels=in_channels,
            unet_out_channels=unet_out_channels,
            channels=tuple(channels),
            final_conv_channels=tuple(final_conv_channels),
        )
        self.net = VoxelMorph(
            backbone=backbone,
            integration_steps=integration_steps,
            half_res=half_res,
            spatial_dims=3,
        )

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        warped, ddf = self.net(moving, fixed)
        return warped, ddf


@register_model("voxelmorph3d")
def create_voxelmorph3d(
    image_size: Sequence[int],
    in_channels: int = 2,
    unet_out_channels: int = 32,
    channels: Sequence[int] = (16, 32, 32, 32, 32, 32),
    final_conv_channels: Sequence[int] = (16, 16),
    integration_steps: int = 0,
    half_res: bool = False,
) -> nn.Module:
    """
    Factory for VoxelMorph3D.

    ``image_size`` is accepted for registry uniformity and is unused internally.

    Config example:
        model:
          name: voxelmorph3d
          in_channels: 2
          unet_out_channels: 32
          channels: [16, 32, 32, 32, 32, 32]
          final_conv_channels: [16, 16]
          integration_steps: 0
          half_res: false
    """
    return VoxelMorph3D(
        image_size=image_size,
        in_channels=in_channels,
        unet_out_channels=unet_out_channels,
        channels=channels,
        final_conv_channels=final_conv_channels,
        integration_steps=integration_steps,
        half_res=half_res,
    )
