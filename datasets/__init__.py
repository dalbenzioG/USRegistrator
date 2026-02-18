"""Datasets for 3D image registration."""

from __future__ import annotations

# Import all dataset modules to trigger @register_dataset decorators
from . import synthetic_ellipsoids
from . import deepreg_synthetic

# Import registry utilities
from .registry import (
    DATASET_REGISTRY,
    register_dataset,
    build_dataset,
)

# Import dataset / generator classes for direct access
from .synthetic_ellipsoids import (
    SyntheticEllipsoidsGenerator,
    SyntheticEllipsoidsMonaiDataset,
)
from .deepreg_synthetic import DeepRegLikeDVFSyntheticGenerator

# Import factory functions
from .synthetic_ellipsoids import create_synthetic_ellipsoids
from .deepreg_synthetic import create_deepreg_synthetic

__all__ = [
    # Registry
    "DATASET_REGISTRY",
    "register_dataset",
    "build_dataset",
    # Generator / dataset classes
    "SyntheticEllipsoidsGenerator",
    "SyntheticEllipsoidsMonaiDataset",
    "DeepRegLikeDVFSyntheticGenerator",
    # Factory functions
    "create_synthetic_ellipsoids",
    "create_deepreg_synthetic",
]
