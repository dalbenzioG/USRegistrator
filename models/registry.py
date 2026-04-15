"""Model registry and factory utilities."""

from __future__ import annotations

from typing import Dict, Callable, Sequence

from torch import nn


MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """
    Decorator to register a model factory under a string key.

    Usage:
        @register_model("globalnet3d")
        def create_globalnet3d(...):
            ...
    """
    def decorator(fn: Callable[..., nn.Module]):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def build_model(cfg: dict, image_size: Sequence[int]) -> nn.Module:
    """
    Build a model from a config dict.

    Expected config format:
        cfg = {
            "name": "globalnet3d",
            ... other kwargs passed to the factory ...
        }

    Args:
        cfg: model configuration dictionary.
        image_size: (D, H, W) of the 3D volume.

    Returns:
        Instantiated nn.Module.
    """
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return MODEL_REGISTRY[name](image_size=image_size, **kwargs)
