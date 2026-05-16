"""Loss functions for image registration."""

from __future__ import annotations

# Import all loss modules to register them
from . import mse
from . import lncc
from . import lncc_dvf
from . import combo

# Import utilities and registry
from .utils import (
    LOSS_REGISTRY,
    register_loss,
    build_loss,
    validate_smoothing_params,
)

# Import loss classes for direct access
from .lncc_dvf import LNCCWithDVFSupervision
from .combo import LNCCWithDiceLoss

# Import factory functions
from .mse import create_mse
from .lncc import create_lncc
from .lncc_dvf import create_lncc_dvf_loss
from .combo import create_lncc_dice_loss

__all__ = [
    "LOSS_REGISTRY",
    "register_loss",
    "build_loss",
    "validate_smoothing_params",
    "LNCCWithDVFSupervision",
    "LNCCWithDiceLoss",
    # Factory functions
    "create_mse",
    "create_lncc",
    "create_lncc_dvf_loss",
    "create_lncc_dice_loss",
]

