"""Registry of runnable tutorials.

A tutorial is a callable that builds a complete USRegistrator config and hands it to
`train.run_training`. It exists so a worked example is *executable* rather than a set of
instructions that rot: the config lives next to the code that resolves it.

Register one with the decorator, then run it from the CLI or from Python:

    python main.py --tutorial trusted_ct_us_localnet --data-root /path/to/data

    from tutorials import run_tutorial
    run_tutorial("trusted_ct_us_localnet", data_root="/path/to/data", epochs=5)
"""

from __future__ import annotations

from typing import Any, Callable

TUTORIAL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_tutorial(name: str):
    """Register a tutorial entry point under a string key."""

    def decorator(fn: Callable[..., Any]):
        if name in TUTORIAL_REGISTRY:
            raise ValueError(f"Tutorial '{name}' is already registered.")
        TUTORIAL_REGISTRY[name] = fn
        return fn

    return decorator


def list_tutorials() -> list[str]:
    return sorted(TUTORIAL_REGISTRY)


def describe_tutorials() -> str:
    lines = []
    for name in list_tutorials():
        doc = (TUTORIAL_REGISTRY[name].__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else "(no description)"
        lines.append(f"  {name}\n      {summary}")
    return "\n".join(lines) if lines else "  (none registered)"


def run_tutorial(name: str, **kwargs: Any) -> Any:
    """Run a registered tutorial by name, forwarding keyword arguments to it."""
    if name not in TUTORIAL_REGISTRY:
        raise KeyError(
            f"Unknown tutorial '{name}'. Available:\n{describe_tutorials()}"
        )
    return TUTORIAL_REGISTRY[name](**kwargs)
