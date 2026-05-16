#!/usr/bin/env python3
"""Generate a custom_dataset JSON manifest from TRUSTED folder structure.

Expected root layout:
  <root>/
    CT_images/
    CT_masks/
    US_images/
    US_masks/

Expected file naming per folder:
  CT_images: <CASE_ID>_imgCT.nii.gz
  CT_masks:  <CASE_ID>_maskCT.nii.gz
  US_images: <CASE_ID>_imgUS.nii.gz
  US_masks:  <CASE_ID>_maskUS.nii.gz
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


FOLDER_PATTERN = {
    "CT_images": re.compile(r"^(?P<id>.+)_imgCT\.nii\.gz$"),
    "CT_masks": re.compile(r"^(?P<id>.+)_maskCT\.nii\.gz$"),
    "US_images": re.compile(r"^(?P<id>.+)_imgUS\.nii\.gz$"),
    "US_masks": re.compile(r"^(?P<id>.+)_maskUS\.nii\.gz$"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create custom_dataset manifest JSON from TRUSTED data folders."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="TRUSTED split root containing CT_images/CT_masks/US_images/US_masks.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio in [0, 1). Default: 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic case split. Default: 42",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write paths relative to manifest directory instead of absolute paths.",
    )
    return parser.parse_args()


def collect_folder_cases(folder: Path, folder_name: str) -> dict[str, Path]:
    pattern = FOLDER_PATTERN[folder_name]
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing required folder: '{folder}'")

    result: dict[str, Path] = {}
    invalid: list[str] = []

    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if not path.name.endswith(".nii.gz"):
            continue

        match = pattern.match(path.name)
        if not match:
            invalid.append(path.name)
            continue

        case_id = match.group("id")
        if case_id in result:
            raise ValueError(
                f"Duplicate case ID '{case_id}' in {folder_name}: "
                f"'{result[case_id].name}' and '{path.name}'."
            )
        result[case_id] = path.resolve()

    if invalid:
        joined = ", ".join(invalid[:10])
        suffix = " ..." if len(invalid) > 10 else ""
        raise ValueError(
            f"Found invalid filenames in {folder_name} (unexpected naming). "
            f"Examples: {joined}{suffix}"
        )

    if not result:
        raise ValueError(f"No NIfTI files found in '{folder}'.")

    return result


def split_case_ids(case_ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1). Got {val_ratio}.")

    ids = list(case_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    if len(ids) <= 1:
        return ids, []

    n_val = int(round(len(ids) * val_ratio))
    n_val = max(1, n_val) if val_ratio > 0 else 0
    n_val = min(n_val, len(ids) - 1)

    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


def format_path(path: Path, manifest_dir: Path, relative_paths: bool) -> str:
    if relative_paths:
        return str(path.relative_to(manifest_dir))
    return str(path)


def build_record(
    case_id: str,
    ct_img: Path,
    us_img: Path,
    ct_mask: Path,
    us_mask: Path,
    manifest_dir: Path,
    relative_paths: bool,
) -> dict[str, str]:
    return {
        "case_id": case_id,
        "moving_modality": "CT",
        "fixed_modality": "US",
        "moving": format_path(ct_img, manifest_dir, relative_paths),
        "fixed": format_path(us_img, manifest_dir, relative_paths),
        "moving_label": format_path(ct_mask, manifest_dir, relative_paths),
        "fixed_label": format_path(us_mask, manifest_dir, relative_paths),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = out_path.parent

    folder_maps = {
        name: collect_folder_cases(root / name, name)
        for name in FOLDER_PATTERN
    }

    case_sets = [set(m.keys()) for m in folder_maps.values()]
    common_ids = sorted(set.intersection(*case_sets))
    if not common_ids:
        raise ValueError(
            "No paired IDs found across CT_images, CT_masks, US_images, US_masks."
        )

    all_ids = sorted(set.union(*case_sets))
    missing_report: list[str] = []
    for case_id in all_ids:
        missing = [name for name, id_map in folder_maps.items() if case_id not in id_map]
        if missing:
            missing_report.append(f"{case_id}: missing in {', '.join(missing)}")
    if missing_report:
        preview = "\n".join(missing_report[:20])
        suffix = "\n..." if len(missing_report) > 20 else ""
        raise ValueError(
            "Found unpaired case IDs across folders. Fix naming/coverage before manifest generation.\n"
            f"{preview}{suffix}"
        )

    train_ids, val_ids = split_case_ids(common_ids, val_ratio=args.val_ratio, seed=args.seed)

    manifest = {"train": [], "val": []}
    for case_id in train_ids:
        manifest["train"].append(
            build_record(
                case_id=case_id,
                ct_img=folder_maps["CT_images"][case_id],
                us_img=folder_maps["US_images"][case_id],
                ct_mask=folder_maps["CT_masks"][case_id],
                us_mask=folder_maps["US_masks"][case_id],
                manifest_dir=manifest_dir,
                relative_paths=args.relative_paths,
            )
        )
    for case_id in val_ids:
        manifest["val"].append(
            build_record(
                case_id=case_id,
                ct_img=folder_maps["CT_images"][case_id],
                us_img=folder_maps["US_images"][case_id],
                ct_mask=folder_maps["CT_masks"][case_id],
                us_mask=folder_maps["US_masks"][case_id],
                manifest_dir=manifest_dir,
                relative_paths=args.relative_paths,
            )
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest: {out_path}")
    print(f"Total paired cases: {len(common_ids)}")
    print(f"Train cases: {len(train_ids)}")
    print(f"Val cases: {len(val_ids)}")
    if train_ids:
        print(f"Train preview: {', '.join(train_ids[:5])}")
    if val_ids:
        print(f"Val preview: {', '.join(val_ids[:5])}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
