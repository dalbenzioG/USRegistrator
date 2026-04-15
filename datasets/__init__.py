from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent.parent / "datasets.py"
_SPEC = importlib.util.spec_from_file_location("usregistrator_datasets_file", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load dataset definitions from {_MODULE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_dataset = _MODULE.build_dataset
DATASET_REGISTRY = _MODULE.DATASET_REGISTRY

__all__ = ["build_dataset", "DATASET_REGISTRY"]
