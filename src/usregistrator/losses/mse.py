"""MSE (Mean Squared Error) loss function."""

from __future__ import annotations

from torch import nn
from .utils import register_loss


@register_loss("mse")
def create_mse(**kwargs) -> nn.Module:
    """
    Standard MSE similarity loss.
    """
    return nn.MSELoss()

