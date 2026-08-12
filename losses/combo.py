"""Combined registration losses (e.g. LNCC + Dice)."""

from __future__ import annotations

import torch
from torch import nn
from monai.networks.blocks import Warp

from .lncc import create_lncc
from .utils import register_loss


class LNCCWithDiceLoss(nn.Module):
    """
    Combined LNCC + Dice loss for image registration with labels.

    Expected usage:
        loss = criterion(
            warped,
            fixed,
            pred_dvf=ddf,
            fixed_label=fixed_label,
            moving_label=moving_label,
        )
    """

    def __init__(
        self,
        kernel_size: int | None = None,
        patch_size: int | None = None,
        spatial_dims: int = 3,
        kernel_type: str = "rectangular",
        reduction: str = "mean",
        smooth_nr: float = 0.0,
        smooth_dr: float = 1e-5,
        lncc_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth_weight: float = 0.0,
        include_background: bool = True,
        dice_smooth_nr: float = 1e-5,
        dice_smooth_dr: float = 1e-5,
        label_warp_mode: str = "nearest",
        label_warp_padding_mode: str = "border",
        fixed_label_key: str = "fixed_label",
        moving_label_key: str = "moving_label",
        warped_moving_label_key: str = "warped_moving_label",
        **_,
    ) -> None:
        super().__init__()
        self.image_loss = create_lncc(
            kernel_size=kernel_size,
            patch_size=patch_size,
            spatial_dims=spatial_dims,
            kernel_type=kernel_type,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.lncc_weight = float(lncc_weight)
        self.dice_weight = float(dice_weight)
        self.include_background = bool(include_background)
        self.smooth_weight = float(smooth_weight)
        self.dice_smooth_nr = float(dice_smooth_nr)
        self.dice_smooth_dr = float(dice_smooth_dr)

        self.fixed_label_key = str(fixed_label_key)
        self.moving_label_key = str(moving_label_key)
        self.warped_moving_label_key = str(warped_moving_label_key)
        self.label_warp = Warp(
            mode=label_warp_mode,
            padding_mode=label_warp_padding_mode,
        )

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss over spatial dimensions, averaged over batch/channel."""
        if pred.shape != target.shape:
            raise ValueError(
                "Dice expects prediction and target with the same shape, "
                f"got pred={tuple(pred.shape)} and target={tuple(target.shape)}."
            )
        if pred.ndim < 3:
            raise ValueError(
                "Dice expects MONAI-style tensors with shape (B, C, ...), "
                f"got ndim={pred.ndim}."
            )

        pred = pred.float()
        target = target.float()

        if not self.include_background and pred.shape[1] > 1:
            pred = pred[:, 1:]
            target = target[:, 1:]

        reduce_dims = tuple(range(2, pred.ndim))
        intersection = torch.sum(pred * target, dim=reduce_dims)
        denominator = torch.sum(pred, dim=reduce_dims) + torch.sum(target, dim=reduce_dims)

        dice_score = (2.0 * intersection + self.dice_smooth_nr) / (
            denominator + self.dice_smooth_dr
        )
        return 1.0 - dice_score.mean()

    def _resolve_label_inputs(
        self,
        fixed_label: torch.Tensor | None,
        moving_label: torch.Tensor | None,
        warped_moving_label: torch.Tensor | None,
        pred_dvf: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve required label tensors, warping moving labels when necessary."""
        missing_fields: list[str] = []

        if fixed_label is None:
            missing_fields.append(self.fixed_label_key)

        if warped_moving_label is None:
            if moving_label is None:
                missing_fields.append(self.moving_label_key)
            if pred_dvf is None:
                missing_fields.append("pred_dvf/ddf")

        if missing_fields:
            formatted = ", ".join(missing_fields)
            raise ValueError(
                "lncc_dice requires segmentation labels to compute Dice loss. "
                f"Missing required fields: {formatted}. "
                f"Provide `{self.fixed_label_key}` and either "
                f"`{self.warped_moving_label_key}` or (`{self.moving_label_key}` + pred_dvf/ddf)."
            )

        if warped_moving_label is None:
            warped_moving_label = self.label_warp(moving_label.float(), pred_dvf)
            if moving_label.dtype.is_floating_point:
                warped_moving_label = warped_moving_label.to(dtype=moving_label.dtype)

        return fixed_label, warped_moving_label

    @staticmethod
    def _smoothness(pred_dvf: torch.Tensor) -> torch.Tensor:
        """Mean squared first difference of the displacement field, per axis.

        Without this term nothing in the objective penalises a rough field: the network
        can tear tissue apart to gain mask overlap, which shows up as rising folding and a
        TRE that degrades past the rigid initialisation while Dice still creeps up.
        """
        dz = (pred_dvf[:, :, 1:, :, :] - pred_dvf[:, :, :-1, :, :]).pow(2).mean()
        dy = (pred_dvf[:, :, :, 1:, :] - pred_dvf[:, :, :, :-1, :]).pow(2).mean()
        dx = (pred_dvf[:, :, :, :, 1:] - pred_dvf[:, :, :, :, :-1]).pow(2).mean()
        return dz + dy + dx

    def forward(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        pred_dvf: torch.Tensor | None = None,
        gt_dvf: torch.Tensor | None = None,
        *,
        fixed_label: torch.Tensor | None = None,
        moving_label: torch.Tensor | None = None,
        warped_moving_label: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        """Compute weighted LNCC + Dice loss."""
        del gt_dvf  # Unused here; kept for compatibility with current training call path.

        lncc_loss = self.image_loss(warped, fixed)
        total_loss = self.lncc_weight * lncc_loss

        if self.smooth_weight > 0.0:
            if pred_dvf is None:
                raise ValueError(
                    "lncc_dice needs the predicted displacement field to apply "
                    "smooth_weight; pred_dvf/ddf was not provided."
                )
            total_loss = total_loss + self.smooth_weight * self._smoothness(pred_dvf)

        if self.dice_weight <= 0.0:
            return total_loss

        fixed_label, warped_moving_label = self._resolve_label_inputs(
            fixed_label=fixed_label,
            moving_label=moving_label,
            warped_moving_label=warped_moving_label,
            pred_dvf=pred_dvf,
        )
        dice_loss = self._dice_loss(warped_moving_label, fixed_label)
        return total_loss + self.dice_weight * dice_loss


@register_loss("lncc_dice")
def create_lncc_dice_loss(
    kernel_size: int | None = None,
    patch_size: int | None = None,
    spatial_dims: int = 3,
    kernel_type: str = "rectangular",
    reduction: str = "mean",
    smooth_nr: float = 0.0,
    smooth_dr: float = 1e-5,
    lncc_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth_weight: float = 0.0,
    include_background: bool = True,
    dice_smooth_nr: float = 1e-5,
    dice_smooth_dr: float = 1e-5,
    label_warp_mode: str = "nearest",
    label_warp_padding_mode: str = "border",
    fixed_label_key: str = "fixed_label",
    moving_label_key: str = "moving_label",
    warped_moving_label_key: str = "warped_moving_label",
    **kwargs,
) -> nn.Module:
    """
    Factory for LNCC + Dice loss.

    Supports:
        loss:
          name: lncc_dice
          params:
            lncc_weight: 1.0
            dice_weight: 1.0
    """
    return LNCCWithDiceLoss(
        kernel_size=kernel_size,
        patch_size=patch_size,
        spatial_dims=spatial_dims,
        kernel_type=kernel_type,
        reduction=reduction,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        lncc_weight=lncc_weight,
        dice_weight=dice_weight,
        smooth_weight=smooth_weight,
        include_background=include_background,
        dice_smooth_nr=dice_smooth_nr,
        dice_smooth_dr=dice_smooth_dr,
        label_warp_mode=label_warp_mode,
        label_warp_padding_mode=label_warp_padding_mode,
        fixed_label_key=fixed_label_key,
        moving_label_key=moving_label_key,
        warped_moving_label_key=warped_moving_label_key,
        **kwargs,
    )
