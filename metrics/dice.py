"""Dice metrics for registration evaluation."""

from __future__ import annotations

import torch
from torch import Tensor
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp


def dice_score(
    y_pred: Tensor,
    y: Tensor,
    threshold: float = 0.5,
    include_background: bool = True,
) -> float:
    """
    Compute hard Dice score using MONAI DiceMetric.

    Both inputs are thresholded (>= threshold) for binary-mask style evaluation.
    """
    if y_pred.shape != y.shape:
        raise ValueError(f"Dice: shape mismatch {tuple(y_pred.shape)} vs {tuple(y.shape)}")

    pred_bin = (y_pred >= threshold).float()
    target_bin = (y >= threshold).float()

    metric = DiceMetric(
        include_background=include_background,
        reduction="mean",
        ignore_empty=False,
    )
    metric.reset()
    metric(y_pred=pred_bin, y=target_bin)
    score_tensor = metric.aggregate()
    metric.reset()
    return float(score_tensor.item())


def registration_dice(
    fixed_label: Tensor,
    moving_label: Tensor | None = None,
    warped_moving_label: Tensor | None = None,
    ddf: Tensor | None = None,
    threshold: float = 0.5,
    include_background: bool = True,
    warp_mode: str = "nearest",
    warp_padding_mode: str = "border",
) -> float:
    """
    Compute Dice between fixed label and warped moving label.

    If `warped_moving_label` is not provided, moving labels are warped using `ddf`.
    """
    if fixed_label is None:
        raise ValueError("registration_dice requires fixed_label.")

    pred_label = warped_moving_label
    if pred_label is None:
        if moving_label is None or ddf is None:
            raise ValueError(
                "registration_dice requires either warped_moving_label, or moving_label + ddf."
            )
        pred_label = Warp(mode=warp_mode, padding_mode=warp_padding_mode)(
            moving_label.float(), ddf
        )
        if moving_label.dtype.is_floating_point:
            pred_label = pred_label.to(dtype=moving_label.dtype)

    return dice_score(
        y_pred=pred_label,
        y=fixed_label,
        threshold=threshold,
        include_background=include_background,
    )
