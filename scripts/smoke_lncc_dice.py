#!/usr/bin/env python3
"""Smoke checks for the registered lncc_dice loss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses import LOSS_REGISTRY, build_loss  # noqa: E402


def _assert_scalar(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise AssertionError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 0:
        raise AssertionError(f"{name} must be scalar, got shape={tuple(value.shape)}.")
    if not torch.isfinite(value):
        raise AssertionError(f"{name} must be finite, got value={value.item()}.")


def main() -> None:
    print("[1/4] Checking registry entry...")
    if "lncc_dice" not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise AssertionError(f"'lncc_dice' not registered. Available losses: {available}")

    print("[2/4] Building from YAML-style config (with nested params)...")
    cfg = {
        "name": "lncc_dice",
        "params": {
            "kernel_size": 3,
            "lncc_weight": 1.0,
            "dice_weight": 1.0,
        },
    }
    loss_fn = build_loss(cfg)

    print("[3/4] Running scalar forward pass...")
    b, c, d, h, w = 1, 1, 12, 12, 12
    moving = torch.rand((b, c, d, h, w), dtype=torch.float32)
    fixed = torch.rand((b, c, d, h, w), dtype=torch.float32)
    ddf = torch.zeros((b, 3, d, h, w), dtype=torch.float32)
    fixed_label = (fixed > 0.5).float()
    moving_label = (moving > 0.5).float()
    scalar_loss = loss_fn(
        moving,
        fixed,
        pred_dvf=ddf,
        fixed_label=fixed_label,
        moving_label=moving_label,
    )
    _assert_scalar(scalar_loss, "lncc_dice forward output")

    print("[4/4] Verifying clear error for missing labels...")
    try:
        loss_fn(moving, fixed, pred_dvf=ddf)
    except ValueError as exc:
        message = str(exc)
        if "fixed_label" not in message or "moving_label" not in message:
            raise AssertionError(
                "Missing-label error should mention required label fields. "
                f"Actual error: {message}"
            ) from exc
    else:
        raise AssertionError("Expected ValueError when labels are missing, but no error was raised.")

    print("Smoke checks passed for lncc_dice.")


if __name__ == "__main__":
    main()
