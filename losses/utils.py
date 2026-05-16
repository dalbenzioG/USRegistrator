"""Shared utilities for loss functions."""

from __future__ import annotations

from typing import Dict, Callable
from torch import nn


LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_loss(name: str):
    """Decorator to register a loss function factory."""
    def decorator(fn: Callable[..., nn.Module]):
        LOSS_REGISTRY[name] = fn
        return fn

    return decorator


def build_loss(cfg: dict) -> nn.Module:
    """
    Build a loss function from config.

    Supported forms:
        cfg = {
            "name": "lncc",
            "patch_size": 9,    # or "kernel_size": 9
            ...
        }

        cfg = {
            "name": "lncc_dice",
            "params": {
                "lncc_weight": 1.0,
                "dice_weight": 1.0,
            },
        }
    """
    name = cfg["name"]
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    params = kwargs.pop("params", None)
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError(
                f"Expected 'loss.params' to be a dict, got {type(params).__name__}."
            )
        # Keep backward compatibility: top-level kwargs override nested params keys.
        kwargs = {**params, **kwargs}
    return LOSS_REGISTRY[name](**kwargs)


def validate_smoothing_params(smooth_nr: float, smooth_dr: float) -> tuple[float, float]:
    """
    Validate and adjust smoothing parameters for numerical stability.
    
    Increases smoothing for better numerical stability, especially with AMP.
    Use much larger values to prevent NaN in mixed precision.
    smooth_dr prevents division by zero in variance calculations.
    smooth_nr prevents issues in the numerator.
    
    Args:
        smooth_nr: Numerator smoothing parameter
        smooth_dr: Denominator smoothing parameter
        
    Returns:
        Tuple of (adjusted_smooth_nr, adjusted_smooth_dr)
    """
    smooth_nr = float(smooth_nr)
    smooth_dr = float(smooth_dr)
    
    if smooth_dr < 1e-2:
        smooth_dr = 1e-2  # Significantly increase minimum smoothing for denominator
    if smooth_nr < 1e-4:
        smooth_nr = 1e-4  # Add smoothing to numerator as well
    
    return smooth_nr, smooth_dr

