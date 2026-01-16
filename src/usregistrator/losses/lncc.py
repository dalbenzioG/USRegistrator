"""LNCC (Local Normalized Cross-Correlation) loss function."""

from __future__ import annotations

from monai.losses import LocalNormalizedCrossCorrelationLoss
from .utils import register_loss, validate_smoothing_params


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
) -> LocalNormalizedCrossCorrelationLoss:
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

    # Validate and adjust smoothing parameters
    smooth_nr, smooth_dr = validate_smoothing_params(smooth_nr, smooth_dr)

    return LocalNormalizedCrossCorrelationLoss(
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        kernel_type=kernel_type,
        reduction=reduction,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
    )

