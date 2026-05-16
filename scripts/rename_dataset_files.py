#!/usr/bin/env python3
"""Rename dataset files to a consistent ID-based naming scheme.

Renaming patterns:
  - CT_images: <ID>_imgCT.nii.gz
  - CT_masks:  <ID>_maskCT.nii.gz
  - US_images: <ID>_imgUS.nii.gz
  - US_masks:  <ID>_maskUS.nii.gz

The <ID> is extracted as the text before the first underscore in each filename.
By default, this script runs in dry-run mode and only prints planned changes.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenamePlan:
    """A single source -> target filename mapping within one folder."""

    source: Path
    target: Path


FOLDER_TO_SUFFIX = {
    "CT_images": "imgCT",
    "CT_masks": "maskCT",
    "US_images": "imgUS",
    "US_masks": "maskUS",
}


def _extract_id(filename: str) -> str | None:
    """Return token before first underscore, or None if malformed."""
    stem = filename[:-7] if filename.endswith(".nii.gz") else filename
    first, sep, _rest = stem.partition("_")
    if not sep or not first:
        return None
    return first


def _collect_plans(folder_path: Path, suffix: str) -> tuple[list[RenamePlan], list[str], list[str]]:
    """Collect planned renames and validation errors for one folder."""
    if not folder_path.is_dir():
        return [], [f"Missing required folder: {folder_path}"], []

    plans: list[RenamePlan] = []
    errors: list[str] = []
    skipped_non_nii: list[str] = []
    target_to_sources: dict[str, list[Path]] = {}

    for child in sorted(folder_path.iterdir(), key=lambda p: p.name):
        if not child.is_file():
            continue
        if not child.name.endswith(".nii.gz"):
            skipped_non_nii.append(child.name)
            continue

        sample_id = _extract_id(child.name)
        if sample_id is None:
            errors.append(
                f"{folder_path.name}: malformed filename (cannot parse ID before first underscore): {child.name}"
            )
            continue

        new_name = f"{sample_id}_{suffix}.nii.gz"
        target = child.with_name(new_name)
        plans.append(RenamePlan(source=child, target=target))
        target_to_sources.setdefault(new_name, []).append(child)

    for target_name, sources in sorted(target_to_sources.items()):
        if len(sources) > 1:
            joined = ", ".join(src.name for src in sources)
            errors.append(
                f"{folder_path.name}: collision for target '{target_name}' from sources: {joined}"
            )

    return plans, errors, skipped_non_nii


def _print_preview(plans_by_folder: dict[str, list[RenamePlan]], apply: bool) -> int:
    """Print per-folder rename preview and return count of actual changes."""
    rename_count = 0
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n[{mode}] Planned renames\n")

    for folder, plans in plans_by_folder.items():
        print(f"{folder}:")
        if not plans:
            print("  (no .nii.gz files found)")
            continue
        changed_in_folder = 0
        for plan in plans:
            if plan.source.name == plan.target.name:
                continue
            changed_in_folder += 1
            rename_count += 1
            print(f"  {plan.source.name} -> {plan.target.name}")
        if changed_in_folder == 0:
            print("  (all files already match target naming)")
    print()
    return rename_count


def _apply_renames(plans_by_folder: dict[str, list[RenamePlan]]) -> None:
    """Perform a two-phase rename per folder to avoid in-place conflicts."""
    for folder, plans in plans_by_folder.items():
        pending = [p for p in plans if p.source.name != p.target.name]
        if not pending:
            continue

        temp_mappings: list[tuple[Path, Path]] = []
        for plan in pending:
            temp_name = f".rename_tmp_{uuid.uuid4().hex}_{plan.source.name}"
            temp_path = plan.source.with_name(temp_name)
            os.rename(plan.source, temp_path)
            temp_mappings.append((temp_path, plan.target))

        for temp_path, final_path in temp_mappings:
            os.rename(temp_path, final_path)

        print(f"Applied {len(pending)} rename(s) in {folder}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename CT/US image and mask files to a fixed ID-based pattern.",
        epilog=(
            "Examples:\n"
            "  python scripts/rename_dataset_files.py --root split_90_10/train\n"
            "  python scripts/rename_dataset_files.py --root split_90_10/train --apply"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Base folder containing CT_images, CT_masks, US_images, and US_masks.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute renames. If omitted, only prints a dry-run preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()

    plans_by_folder: dict[str, list[RenamePlan]] = {}
    errors: list[str] = []
    skipped: dict[str, list[str]] = {}

    for folder_name, suffix in FOLDER_TO_SUFFIX.items():
        folder_path = root / folder_name
        plans, folder_errors, skipped_non_nii = _collect_plans(folder_path, suffix)
        plans_by_folder[folder_name] = plans
        errors.extend(folder_errors)
        if skipped_non_nii:
            skipped[folder_name] = skipped_non_nii

    if skipped:
        for folder_name, names in skipped.items():
            print(
                f"Warning: {folder_name} has {len(names)} non-.nii.gz file(s), ignored."
            )

    if errors:
        print("\nValidation failed. No files were renamed.")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    rename_count = _print_preview(plans_by_folder, apply=args.apply)
    if rename_count == 0:
        print("No changes required.")
        return

    if not args.apply:
        print("Dry-run only. Re-run with --apply to perform renames.")
        return

    _apply_renames(plans_by_folder)
    print(f"\nDone. Total files renamed: {rename_count}")


if __name__ == "__main__":
    main()
