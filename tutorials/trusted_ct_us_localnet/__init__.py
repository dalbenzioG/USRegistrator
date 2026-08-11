"""TRUSTED CT->US kidney registration with LocalNet3D — runnable tutorial.

Resolves `config.yaml` in this directory against wherever your data lives, then hands it
to `train.run_training`. See README.md for the walkthrough.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..registry import register_tutorial

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"

# Filename patterns for the co-registered TRUSTED layout (`*_in_US_space_*` CT volumes),
# as found on the `split_90_10_new` volume. manifest key -> (subdirectory, regex).
LAYOUT = {
    "moving": ("CT_images", r"(?P<cid>\d+[LR])_imgCT_in_US_space_crop.*\.nii\.gz$"),
    "moving_label": ("CT_masks", r"(?P<cid>\d+[LR])_seg_in_US_space_crop.*\.nii\.gz$"),
    "fixed": ("US_images", r"(?P<cid>\d+[LR])_imgUS_crop\.nii\.gz$"),
    "fixed_label": ("US_masks", r"(?P<cid>\d+[LR])_maskUS_crop\.nii\.gz$"),
}
OPTIONAL_LAYOUT = {
    "moving_landmarks": ("landmarks_lps", r"(?P<cid>\d+[LR])_ldkCT_aligned_lps\.txt$"),
    "fixed_landmarks": ("landmarks_lps", r"(?P<cid>\d+[LR])_ldkUS_lps\.txt$"),
}


def _scan_split(split_root: Path) -> list[dict[str, str]]:
    """Collect complete cases under one split directory (train/ or test/)."""
    per_case: dict[str, dict[str, str]] = defaultdict(dict)
    for key, (subdir, pattern) in {**LAYOUT, **OPTIONAL_LAYOUT}.items():
        directory = split_root / subdir
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            match = re.match(pattern, path.name)
            if match:
                per_case[match.group("cid")][key] = str(path.resolve())

    cases = []
    for cid in sorted(per_case):
        found = per_case[cid]
        missing = [key for key in LAYOUT if key not in found]
        if missing:
            print(f"  skipping {cid}: missing {missing}")
            continue
        cases.append({
            "case_id": cid,
            "moving_modality": "CT",
            "fixed_modality": "US",
            **found,
        })
    return cases


def build_manifest(data_root: str | Path, output: str | Path | None = None) -> Path:
    """Write a custom_dataset manifest for a TRUSTED data root.

    `data_root` must contain `train/` and `test/` subdirectories in the co-registered
    TRUSTED layout. The test split becomes the manifest's "val" split.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    manifest: dict[str, list[dict[str, str]]] = {}
    for split_dir, split_key in (("train", "train"), ("test", "val")):
        split_root = root / split_dir
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Expected '{split_dir}/' under {root}. The tutorial assumes the "
                "pre-split TRUSTED layout (train/ and test/)."
            )
        manifest[split_key] = _scan_split(split_root)

    if not manifest["train"] or not manifest["val"]:
        raise RuntimeError(
            f"No complete cases found under {root}. Expected files such as "
            "train/CT_images/<id>_imgCT_in_US_space_crop*.nii.gz"
        )

    output_path = Path(output) if output else HERE / "manifest.generated.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {output_path} "
          f"({len(manifest['train'])} train / {len(manifest['val'])} val cases)")
    return output_path


def build_config(
    data_root: str | Path | None = None,
    manifest: str | Path | None = None,
    epochs: int | None = None,
    image_size: int | None = None,
    lr: float | None = None,
    align: bool | None = None,
    run_name: str | None = None,
    save_dir: str | Path | None = None,
    wandb_enabled: bool | None = None,
) -> dict[str, Any]:
    """Load `config.yaml` and apply the tutorial's overrides. Returns the config dict.

    Exactly one data source is needed: `manifest` (a ready manifest) or `data_root`
    (scanned to generate one). With neither, the config's own `json_file` is used as-is.
    """
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if manifest and data_root:
        raise ValueError("Pass either data_root or manifest, not both.")
    # Validate before scanning the data root, so a bad argument fails immediately.
    if image_size is not None and int(image_size) % 16 != 0:
        raise ValueError(
            f"image_size must be divisible by 16 for extract_levels "
            f"{cfg['model']['extract_levels']} (got {image_size})."
        )
    if data_root:
        manifest = build_manifest(data_root)
    if manifest:
        for split in ("train_dataset", "val_dataset"):
            cfg[split]["json_file"] = str(Path(manifest).resolve())

    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    if lr is not None:
        cfg["optimizer"]["lr"] = float(lr)
    if image_size is not None:
        size = [int(image_size)] * 3
        cfg["image_size"] = size
        for split in ("train_dataset", "val_dataset"):
            cfg[split]["image_size"] = size
    if align is not None:
        for split in ("train_dataset", "val_dataset"):
            cfg[split]["align_shared_grid"] = bool(align)
    if run_name is not None:
        cfg["wandb"]["run_name"] = run_name
    if save_dir is not None:
        cfg["training"]["save_dir"] = str(save_dir)
    if wandb_enabled is not None:
        cfg["wandb"]["enabled"] = bool(wandb_enabled)

    return cfg


def initial_dice(cfg: dict[str, Any], split: str = "val") -> dict[str, float]:
    """Mask Dice per case after preprocessing, with NO warp applied.

    This is the cheapest sanity check in the whole pipeline and the one worth running
    before any training: it is the score the network has to beat. On correctly aligned
    TRUSTED data it is ~0.86; ~0.58 means `align_shared_grid` is off or the data is not
    co-registered.
    """
    from datasets.registry import build_dataset

    dataset = build_dataset(cfg["val_dataset" if split == "val" else "train_dataset"], split=split)
    scores: dict[str, float] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        moving = sample["moving_label"] > 0.5
        fixed = sample["fixed_label"] > 0.5
        overlap = 2.0 * (moving & fixed).sum() / (moving.sum() + fixed.sum() + 1e-8)
        scores[str(sample.get("case_id", index))] = float(overlap)
        del sample
    return scores


@register_tutorial("trusted_ct_us_localnet")
def run(
    data_root: str | Path | None = None,
    manifest: str | Path | None = None,
    epochs: int | None = None,
    image_size: int | None = None,
    lr: float | None = None,
    align: bool | None = None,
    run_name: str | None = None,
    save_dir: str | Path | None = None,
    wandb_enabled: bool | None = None,
    check_only: bool = False,
) -> Any:
    """TRUSTED CT->US kidney registration with LocalNet3D (shared-grid preprocessing).

    Args:
        data_root: TRUSTED root containing train/ and test/; a manifest is generated.
        manifest: use an existing manifest instead of scanning.
        epochs, image_size, lr: config overrides.
        align: toggle `align_shared_grid`. Set False to reproduce the misaligned baseline.
        run_name, save_dir, wandb_enabled: run bookkeeping.
        check_only: report initial mask Dice and exit without training.
    """
    cfg = build_config(
        data_root=data_root,
        manifest=manifest,
        epochs=epochs,
        image_size=image_size,
        lr=lr,
        align=align,
        run_name=run_name,
        save_dir=save_dir,
        wandb_enabled=wandb_enabled,
    )

    print("--- initial mask Dice (no warp) ---")
    scores = initial_dice(cfg, split="val")
    for case_id, value in scores.items():
        print(f"  {case_id}: {value:.3f}")
    mean = sum(scores.values()) / max(len(scores), 1)
    print(f"  mean: {mean:.3f}")
    if mean < 0.4:
        print("  WARNING: this is very low. Check align_shared_grid and that the CT/US "
              "volumes really are co-registered before spending GPU hours.")
    if check_only:
        return scores

    from train import run_training

    return run_training(config_path=str(CONFIG_PATH), cfg=cfg)
