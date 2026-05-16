#!/usr/bin/env python3
"""Debug custom_dataset loading, pairing integrity, and orientation metadata."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import nibabel as nib
except Exception:  # noqa: BLE001
    nib = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect custom_dataset loading and pair consistency."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config containing train_dataset/val_dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of sample items to inspect.",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_stats(t: torch.Tensor) -> str:
    return (
        f"shape={tuple(t.shape)}, dtype={t.dtype}, "
        f"min={float(t.min()):.5f}, max={float(t.max()):.5f}, "
        f"nan={bool(torch.isnan(t).any())}, inf={bool(torch.isinf(t).any())}"
    )


def _path_for_manifest(path_str: str, manifest_path: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (manifest_path.parent / p).resolve()
    return p


def _orientation_info(path: Path) -> tuple[str, tuple[float, float, float], tuple[int, int, int]]:
    if nib is None:
        return "n/a (nibabel unavailable)", (math.nan, math.nan, math.nan), (-1, -1, -1)
    img = nib.load(str(path))
    ax = "".join(nib.aff2axcodes(img.affine))
    zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
    shape = tuple(int(x) for x in img.shape[:3])
    return ax, zooms, shape


def inspect_manifest_pairs(dataset_cfg: dict[str, Any], split: str) -> None:
    if dataset_cfg.get("name") != "custom_dataset":
        print("Skipping manifest pair checks: dataset name is not 'custom_dataset'.")
        return

    json_file = dataset_cfg.get("json_file")
    if not json_file:
        raise ValueError("custom_dataset requires json_file in YAML.")

    manifest_path = Path(json_file).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows = manifest.get(split, [])
    if not isinstance(rows, list):
        raise ValueError(f"Manifest split '{split}' must be a list.")

    print(f"Manifest: {manifest_path}")
    print(f"Split '{split}' records: {len(rows)}")

    bad_suffix: list[str] = []
    missing_case_id = 0
    orientation_mismatches: list[str] = []

    for i, row in enumerate(rows):
        case_id = row.get("case_id")
        if not case_id:
            missing_case_id += 1
            case_id = f"<row_{i}>"

        moving_path = _path_for_manifest(str(row["moving"]), manifest_path)
        fixed_path = _path_for_manifest(str(row["fixed"]), manifest_path)
        moving_label = row.get("moving_label")
        fixed_label = row.get("fixed_label")
        moving_label_path = _path_for_manifest(moving_label, manifest_path) if moving_label else None
        fixed_label_path = _path_for_manifest(fixed_label, manifest_path) if fixed_label else None

        # TRUSTED-specific suffix consistency checks
        if not moving_path.name.endswith("_imgCT.nii.gz"):
            bad_suffix.append(f"{case_id}: moving filename not *_imgCT.nii.gz -> {moving_path.name}")
        if not fixed_path.name.endswith("_imgUS.nii.gz"):
            bad_suffix.append(f"{case_id}: fixed filename not *_imgUS.nii.gz -> {fixed_path.name}")
        if moving_label_path and not moving_label_path.name.endswith("_maskCT.nii.gz"):
            bad_suffix.append(f"{case_id}: moving_label filename not *_maskCT.nii.gz -> {moving_label_path.name}")
        if fixed_label_path and not fixed_label_path.name.endswith("_maskUS.nii.gz"):
            bad_suffix.append(f"{case_id}: fixed_label filename not *_maskUS.nii.gz -> {fixed_label_path.name}")

        if nib is not None:
            mov_ax, mov_sp, mov_shape = _orientation_info(moving_path)
            fix_ax, fix_sp, fix_shape = _orientation_info(fixed_path)
            if mov_ax != fix_ax:
                orientation_mismatches.append(
                    f"{case_id}: moving({mov_ax}) vs fixed({fix_ax})"
                )
            if i < 3:
                print(
                    f"[pair {i}] case_id={case_id} "
                    f"moving_ax={mov_ax} fixed_ax={fix_ax} "
                    f"moving_shape={mov_shape} fixed_shape={fix_shape} "
                    f"moving_spacing={mov_sp} fixed_spacing={fix_sp}"
                )

    if missing_case_id:
        print(f"Warning: {missing_case_id} case(s) missing 'case_id' in manifest.")

    if bad_suffix:
        print("Pair suffix warnings:")
        for msg in bad_suffix[:20]:
            print(f"  - {msg}")
        if len(bad_suffix) > 20:
            print("  - ...")
    else:
        print("Pair suffix checks passed for TRUSTED naming.")

    if nib is None:
        print("Orientation metadata checks skipped (nibabel not available).")
    elif orientation_mismatches:
        print("Orientation mismatches detected:")
        for msg in orientation_mismatches[:20]:
            print(f"  - {msg}")
        if len(orientation_mismatches) > 20:
            print("  - ...")
    else:
        print("Orientation checks passed: moving/fixed orientation matches for inspected pairs.")


def inspect_loaded_samples(dataset_cfg: dict[str, Any], split: str, num_samples: int) -> None:
    try:
        from datasets import build_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Unable to import dataset loading stack (MONAI/PyData environment issue). "
            "Fix the Python environment, then rerun this debug command."
        ) from exc

    dataset = build_dataset(dataset_cfg, split=split)
    print(f"Loaded split '{split}' with {len(dataset)} samples.")

    to_read = min(num_samples, len(dataset))
    for idx in range(to_read):
        item = dataset[idx]
        print(f"\n[sample {idx}] keys={sorted(item.keys())}")

        for key in ("moving", "fixed", "moving_label", "fixed_label"):
            if key in item:
                value = item[key]
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {_safe_stats(value)}")
                else:
                    print(f"  {key}: non-tensor value type={type(value).__name__}")

        for key in ("moving_landmarks", "fixed_landmarks"):
            if key in item:
                value = item[key]
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value).__name__}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    dataset_key = f"{args.split}_dataset"
    if dataset_key not in cfg:
        raise KeyError(f"Missing '{dataset_key}' in config.")

    dataset_cfg = cfg[dataset_key]
    inspect_manifest_pairs(dataset_cfg, split=args.split)
    inspect_loaded_samples(dataset_cfg, split=args.split, num_samples=args.num_samples)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
