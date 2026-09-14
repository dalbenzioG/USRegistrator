"""Registration models for 3D image registration."""

from __future__ import annotations

# Import all model modules to trigger @register_model decorators
from . import globalnet3d
from . import localnet3d
from . import unetreg3d
from . import transmorph_original3d  # Optional dependencies load on construction only.

# Import registry utilities
from .registry import (
    MODEL_REGISTRY,
    register_model,
    build_model,
)

# Import model classes for direct access
from .globalnet3d import GlobalNet3D
from .localnet3d import LocalNet3D
from .unetreg3d import UNetReg3D
from .transmorph_original3d import TransMorphOriginal3D

# Import factory functions
from .globalnet3d import create_globalnet3d
from .localnet3d import create_localnet3d
from .unetreg3d import create_unetreg3d
from .transmorph_original3d import create_transmorph_original3d

__all__ = [
    # Registry
    "MODEL_REGISTRY",
    "register_model",
    "build_model",
    # Model classes
    "GlobalNet3D",
    "LocalNet3D",
    "UNetReg3D",
    "TransMorphOriginal3D",
    # Factory functions
    "create_globalnet3d",
    "create_localnet3d",
    "create_unetreg3d",
    "create_transmorph_original3d",
]
