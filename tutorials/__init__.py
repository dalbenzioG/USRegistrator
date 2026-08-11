"""Runnable, self-documenting worked examples for USRegistrator.

Each subpackage holds a USRegistrator-compatible config plus the small amount of code
needed to resolve it against wherever the data actually lives, exposed as a callable so
`main.py` can run it directly.
"""

from __future__ import annotations

# Import tutorial modules to trigger @register_tutorial decorators.
from . import trusted_ct_us_localnet  # noqa: F401
from .registry import (
    TUTORIAL_REGISTRY,
    describe_tutorials,
    list_tutorials,
    register_tutorial,
    run_tutorial,
)

__all__ = [
    "TUTORIAL_REGISTRY",
    "describe_tutorials",
    "list_tutorials",
    "register_tutorial",
    "run_tutorial",
]
