"""Check pinned source and optional imports before starting a training run."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_model
from models.transmorph_original3d import (
    UPSTREAM_COMMIT, resolve_transmorph_root, verify_transmorph_source,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    root = resolve_transmorph_root(args.source_root)
    report = {"source_root": str(root), "expected_commit": UPSTREAM_COMMIT,
              "normalized_source_sha256": verify_transmorph_source(root)}
    if not args.source_only:
        model = build_model(
            {"name": "transmorph_original3d", "variant": "tiny", "transmorph_root": str(root)},
            [64, 64, 64],
        )
        report["parameters"] = sum(p.numel() for p in model.parameters())
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "torchvision", "monai", "timm", "ml-collections")}
    report["status"] = "PASS"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
