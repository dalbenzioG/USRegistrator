"""Optional adapter for the pinned upstream TransMorph implementation."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from threading import RLock
from typing import Sequence

import torch
from torch import nn

from .registry import register_model


UPSTREAM_COMMIT = "6357a1d7fc44c36db9b1d1ccaa372409253142cf"
SOURCE_SHA256 = {
    "models/TransMorph.py": "bf4d4820c5c6847cbf8f7a4c1bfadf587d478221e7cb84d44363696c73a86581",
    "models/configs_TransMorph.py": "1515d500908b65d07751dabd8f5862053e789b405e7b7a41abfc4c8e3e9fc7d7",
}
CONFIG_FACTORIES = {
    "base": "get_3DTransMorph_config",
    "small": "get_3DTransMorphSmall_config",
    "tiny": "get_3DTransMorphTiny_config",
}
_IMPORT_LOCK = RLock()


def resolve_transmorph_root(source_root: str | None = None) -> Path:
    default = Path(__file__).resolve().parents[1] / ".third_party" / "TransMorph" / "TransMorph"
    return Path(source_root or os.environ.get("USREGISTRATOR_TRANSMORPH_ROOT", str(default))).expanduser().resolve()


def verify_transmorph_source(source_root: str | Path) -> dict[str, str]:
    """Verify the two executed upstream files, permitting only newline conversion."""
    root = Path(source_root)
    actual = {}
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing TransMorph source: {path}. See docs/transmorph-validation.md."
            )
        actual[relative] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual[relative] != expected:
            raise ValueError(
                f"TransMorph source mismatch: {path}; expected source from {UPSTREAM_COMMIT}. "
                "Use the documented checkout; do not silently run a different implementation."
            )
    return actual


def _load_transmorph(root: str):
    # Check on every construction, including when imports are already cached.
    verify_transmorph_source(root)
    try:
        return _load_verified_transmorph(root)
    except ModuleNotFoundError as exc:
        raise ImportError(
            "TransMorph optional dependencies are missing. Install requirements-transmorph.txt "
            "with matching torch/torchvision builds."
        ) from exc


def _module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TransMorph source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@lru_cache(maxsize=None)
def _load_verified_transmorph(root: str):
    """Load upstream without replacing USRegistrator's models package or sys.path."""
    source_root = Path(root)
    config_path = source_root / "models" / "configs_TransMorph.py"
    model_path = source_root / "models" / "TransMorph.py"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"Missing TransMorph source under {source_root}. "
            "See docs/transmorph-validation.md or set USREGISTRATOR_TRANSMORPH_ROOT."
        )
    suffix = hashlib.sha256(str(source_root).encode()).hexdigest()[:12]
    with _IMPORT_LOCK:
        config_module = _module_from_file(f"_usregistrator_tm_config_{suffix}", config_path)
        parent = sys.modules[__package__]
        alias = "models.configs_TransMorph"
        sentinel = object()
        previous_module = sys.modules.get(alias, sentinel)
        previous_attribute = getattr(parent, "configs_TransMorph", sentinel)
        sys.modules[alias] = config_module
        setattr(parent, "configs_TransMorph", config_module)
        try:
            model_module = _module_from_file(f"_usregistrator_tm_{suffix}", model_path)
        except ModuleNotFoundError as exc:
            raise ImportError(
                "TransMorph optional dependencies are missing. "
                "Install requirements-transmorph.txt with matching torch/torchvision builds."
            ) from exc
        finally:
            if previous_module is sentinel:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = previous_module
            if previous_attribute is sentinel:
                delattr(parent, "configs_TransMorph")
            else:
                setattr(parent, "configs_TransMorph", previous_attribute)
    return model_module.TransMorph, config_module


class TransMorphOriginal3D(nn.Module):
    def __init__(
        self,
        image_size: Sequence[int],
        variant: str = "tiny",
        transmorph_root: str | None = None,
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in CONFIG_FACTORIES:
            raise ValueError(f"Unknown TransMorph variant {variant!r}; choose {sorted(CONFIG_FACTORIES)}")
        image_size = tuple(int(size) for size in image_size)
        if len(image_size) != 3 or any(size < 32 or size % 32 for size in image_size):
            raise ValueError("TransMorph image_size must contain three positive multiples of 32")
        root = resolve_transmorph_root(transmorph_root)
        upstream_model, config_module = _load_transmorph(str(root))
        config = getattr(config_module, CONFIG_FACTORIES[variant])()
        config.img_size = image_size
        self.variant = variant
        self.net = upstream_model(config)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        return self.net(torch.cat((moving, fixed), dim=1))


@register_model("transmorph_original3d")
def create_transmorph_original3d(
    image_size: Sequence[int],
    variant: str = "tiny",
    transmorph_root: str | None = None,
) -> nn.Module:
    return TransMorphOriginal3D(image_size, variant=variant, transmorph_root=transmorph_root)
