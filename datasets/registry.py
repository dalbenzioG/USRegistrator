"""Dataset registry and factory utilities."""

from __future__ import annotations

from typing import Dict, Callable

from monai.data import Dataset


DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {}


def register_dataset(name: str):
    """
    Decorator to register a dataset factory under a string key.

    Usage:
        @register_dataset("synthetic_ellipsoids")
        def create_synthetic_ellipsoids(...):
            ...
    """
    def decorator(fn: Callable[..., Dataset]):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


def build_dataset(cfg: dict, split: str, transforms=None) -> Dataset:
    """
    Build a dataset from config.

    Expected config format:
        cfg = {
            "name": "synthetic_ellipsoids",
            ... other kwargs passed to the factory ...
        }

    Args:
        cfg: dataset configuration dictionary.
        split: one of "train", "val", "test".
        transforms: optional MONAI transforms to apply.

    Returns:
        Instantiated Dataset.
    """
    name = cfg["name"]

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )

    factory = DATASET_REGISTRY[name]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}

    return factory(split=split, transforms=transforms, **kwargs)
