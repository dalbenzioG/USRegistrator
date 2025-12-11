# losses.py

from __future__ import annotations

from typing import Dict, Callable
import torch
import torch.nn.functional as F
from torch import nn
from monai.losses import LocalNormalizedCrossCorrelationLoss

LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_loss(name: str):
    def decorator(fn: Callable[..., nn.Module]):
        LOSS_REGISTRY[name] = fn
        return fn

    return decorator


def build_loss(cfg: dict) -> nn.Module:
    """
    Build a loss function from config:
        cfg = {
            "name": "lncc",
            "patch_size": 9,    # or "kernel_size": 9
            ...
        }
    """
    name = cfg["name"]
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return LOSS_REGISTRY[name](**kwargs)


@register_loss("mse")
def create_mse(**kwargs) -> nn.Module:
    """
    Standard MSE similarity loss.
    """
    return nn.MSELoss()


@register_loss("lncc")
def create_lncc(
        kernel_size: int | None = None,
        patch_size: int | None = None,
        spatial_dims: int = 3,
        kernel_type: str = "rectangular",
        reduction: str = "mean",
        smooth_nr: float = 0.0,
        smooth_dr: float = 1e-5,
        **_,
) -> nn.Module:
    """
    Local normalized cross-correlation (intensity-based similarity).

    Supports both:
        loss:
          name: lncc
          patch_size: 9

    and:
        loss:
          name: lncc
          kernel_size: 9
    """
    # Handle both kernel_size and patch_size (they mean the same thing)
    if kernel_size is None and patch_size is not None:
        kernel_size = patch_size
    if kernel_size is None:
        kernel_size = 3  # default

    # Convert to float (YAML may load as string)
    smooth_nr = float(smooth_nr)
    smooth_dr = float(smooth_dr)

    # Increase smoothing for better numerical stability, especially with AMP
    # Use much larger values to prevent NaN in mixed precision
    # smooth_dr prevents division by zero in variance calculations
    # smooth_nr prevents issues in the numerator
    if smooth_dr < 1e-2:
        smooth_dr = 1e-2  # Significantly increase minimum smoothing for denominator
    if smooth_nr < 1e-4:
        smooth_nr = 1e-4  # Add smoothing to numerator as well

    return LocalNormalizedCrossCorrelationLoss(
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        kernel_type=kernel_type,
        reduction=reduction,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
    )

# -------------------------------------------------------------------------
# LNCC + DVF supervision (image similarity + flow MSE)
# -------------------------------------------------------------------------


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

        if smooth_dr < 1e-2:
            smooth_dr = 1e-2
        if smooth_nr < 1e-4:
            smooth_nr = 1e-4

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
