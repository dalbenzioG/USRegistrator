"""Generate a custom_dataset manifest for the co-registered TRUSTED split.

The co-registered ("in US space") TRUSTED volumes live on the Modal volume
`split_90_10_new`, pre-split into train/ and test/. This writes the manifest that
`configs/trusted_localnet3d_aligned.yaml` consumes (test split -> "val").

Regenerate the file listing with:

    for s in train test; do for d in CT_images CT_masks US_images US_masks landmarks_lps; do
      modal volume ls split_90_10_new split_90_10_new/$s/$d; done; done > listing.txt

then:  python scripts/generate_trusted_manifest_split9010.py listing.txt
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA_ROOT = "/data/split_90_10_new"
OUT = Path("configs/trusted_manifest_split9010.json")

# manifest key -> (directory, filename regex capturing the case id)
SPEC = {
    "moving": ("CT_images", r"(?P<cid>\d+[LR])_imgCT_in_US_space_crop.*\.nii\.gz$"),
    "moving_label": ("CT_masks", r"(?P<cid>\d+[LR])_seg_in_US_space_crop.*\.nii\.gz$"),
    "fixed": ("US_images", r"(?P<cid>\d+[LR])_imgUS_crop\.nii\.gz$"),
    "fixed_label": ("US_masks", r"(?P<cid>\d+[LR])_maskUS_crop\.nii\.gz$"),
}

# Optional keys — landmarks ship only with the test split, so a case without them is fine.
# Coordinates are LPS world mm (hence `landmarks_space: lps` in the config).
OPTIONAL_SPEC = {
    "moving_landmarks": ("landmarks_lps", r"(?P<cid>\d+[LR])_ldkCT_aligned_lps\.txt$"),
    "fixed_landmarks": ("landmarks_lps", r"(?P<cid>\d+[LR])_ldkUS_lps\.txt$"),
}


def parse_listing(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """listing lines -> {split: {case_id: {manifest_key: abs path}}}"""
    found: dict[str, dict[str, dict[str, str]]] = {"train": defaultdict(dict), "test": defaultdict(dict)}
    for raw in path.read_text().splitlines():
        line = raw.strip().lstrip("|").strip()
        if not line:
            continue
        parts = line.split("/")
        if len(parts) < 4:
            continue
        split, subdir, name = parts[-3], parts[-2], parts[-1]
        if split not in found:
            continue
        for key, (want_dir, pattern) in {**SPEC, **OPTIONAL_SPEC}.items():
            if subdir != want_dir:
                continue
            match = re.match(pattern, name)
            if match:
                found[split][match.group("cid")][key] = f"{DATA_ROOT}/{split}/{subdir}/{name}"
    return found


def build(found) -> dict[str, list[dict]]:
    manifest: dict[str, list[dict]] = {}
    for split, out_split in (("train", "train"), ("test", "val")):
        cases = []
        for cid in sorted(found[split]):
            paths = found[split][cid]
            missing = [k for k in SPEC if k not in paths]
            if missing:
                print(f"  skip {split}/{cid}: missing {missing}")
                continue
            case = {
                "case_id": cid,
                "moving_modality": "CT",
                "fixed_modality": "US",
                **{k: paths[k] for k in SPEC},
            }
            case.update({k: paths[k] for k in OPTIONAL_SPEC if k in paths})
            cases.append(case)
        manifest[out_split] = cases
    return manifest


def build_local(root: Path) -> dict[str, list[dict]]:
    """Manifest over a flat local directory of co-registered cases (e.g. test_data/).

    Used for smoke tests and the notebook, where only a handful of cases are on disk.
    The same cases go in both splits — this is a pipeline check, not an experiment.
    """
    per_case: dict[str, dict[str, str]] = defaultdict(dict)
    for path in sorted(list(root.glob("*.nii.gz")) + list(root.glob("*_lps.txt"))):
        for key, (_, pattern) in {**SPEC, **OPTIONAL_SPEC}.items():
            match = re.match(pattern, path.name)
            if match:
                per_case[match.group("cid")][key] = str(path.resolve())

    cases = []
    for cid in sorted(per_case):
        missing = [k for k in SPEC if k not in per_case[cid]]
        if missing:
            print(f"  skip {cid}: missing {missing}")
            continue
        case = {
            "case_id": cid,
            "moving_modality": "CT",
            "fixed_modality": "US",
            **{k: per_case[cid][k] for k in SPEC},
        }
        case.update({k: per_case[cid][k] for k in OPTIONAL_SPEC if k in per_case[cid]})
        cases.append(case)
    return {"train": cases, "val": cases}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        root = Path(sys.argv[2] if len(sys.argv) > 2 else "test_data")
        out = Path("configs/trusted_manifest_local.json")
        manifest = build_local(root)
    else:
        listing = Path(sys.argv[1] if len(sys.argv) > 1 else "listing.txt")
        out = OUT
        manifest = build(parse_listing(listing))

    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}: {len(manifest['train'])} train / {len(manifest['val'])} val cases")
