"""LNCC + DVF supervision loss (image similarity + flow MSE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from monai.losses import LocalNormalizedCrossCorrelationLoss
from .utils import register_loss, validate_smoothing_params


class LNCCWithDVFSupervision(nn.Module):
    """
    Combined loss: LNCC-based image similarity + DVF MSE supervision.

    Usage:
        loss = criterion(warped, fixed, pred_dvf, gt_dvf)
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 9,
        kernel_type: str = "rectangular",
        reduction: str = "mean",
        smooth_nr: float = 1e-4,
        smooth_dr: float = 1e-2,
        image_weight: float = 1.0,
        dvf_weight: float = 1.0,
    ):
        super().__init__()
        # Ensure numeric fields (YAML may give strings)
        smooth_nr = float(smooth_nr)
        smooth_dr = float(smooth_dr)
        image_weight = float(image_weight)
        dvf_weight = float(dvf_weight)

        # Validate and adjust smoothing parameters
        smooth_nr, smooth_dr = validate_smoothing_params(smooth_nr, smooth_dr)

        self.image_loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=spatial_dims,
            kernel_size=kernel_size,
            kernel_type=kernel_type,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.image_weight = float(image_weight)
        self.dvf_weight = float(dvf_weight)

    def forward(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        pred_dvf: torch.Tensor | None = None,
        gt_dvf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Image similarity component
        sim_loss = self.image_loss(warped, fixed)

        # DVF MSE supervision
        dvf_loss = torch.tensor(0.0, device=warped.device)
        if self.dvf_weight > 0.0 and pred_dvf is not None and gt_dvf is not None:
            dvf_loss = F.mse_loss(pred_dvf, gt_dvf)

        return self.image_weight * sim_loss + self.dvf_weight * dvf_loss


@register_loss("lncc_dvf")
def create_lncc_dvf_loss(
    spatial_dims: int = 3,
    kernel_size: int = 9,
    kernel_type: str = "rectangular",
    reduction: str = "mean",
    smooth_nr: float = 1e-4,
    smooth_dr: float = 1e-2,
    image_weight: float = 1.0,
    dvf_weight: float = 1.0,
) -> nn.Module:
    """
    Factory for LNCC + DVF loss when config specifies:
        loss:
          name: lncc_dvf
    """
    return LNCCWithDVFSupervision(
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        kernel_type=kernel_type,
        reduction=reduction,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        image_weight=image_weight,
        dvf_weight=dvf_weight,
    )

