"""Modal runner for the TRUSTED CT->US LocalNet3D track.

Data: the co-registered ("in US space") TRUSTED volumes on the `split_90_10_new`
volume, mounted at /data -> data root /data/split_90_10_new. This is the same
volume and the same 53/6 split the VoxelMorph kidney baseline uses, so TRUSTED
LocalNet3D numbers are directly comparable to it.

Entrypoints:
    modal run train_modal_trusted.py::align_check   # init Dice, alignment off vs on
    modal run train_modal_trusted.py::smoke         # 1 epoch end-to-end
    modal run --detach train_modal_trusted.py::train_one --run-name <name>
    modal run --detach train_modal_trusted.py::sweep

Everything runs on the `aniketh` profile (workspace anikethvij464); W&B goes to
entity anikethvij-personal.
"""

import os

import modal

REPO_ROOT = "/root/usregistrator"
# Resolve from this file, not os.getcwd() — the notebook imports this module from
# notebooks/, and cwd would then upload that directory instead of the repo.
LOCAL_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/data/split_90_10_new"
MANIFEST = "configs/trusted_manifest_split9010.json"
# The untouched TRUSTED volumes, before any pre-registration (see align_check_raw).
RAW_MANIFEST = "configs/trusted_manifest_modal.json"
BASE_CONFIG = "configs/trusted_localnet3d_aligned.yaml"

image = (
    modal.Image.debian_slim()
    .pip_install("PyYAML>=6.0.0", "numpy>=1.24.0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "monai>=1.5.0",
        "wandb>=0.23.0",
        "nibabel>=5.0.0",
        "SimpleITK>=2.3.0",
        "einops>=0.7.0",
        "matplotlib>=3.7.0",
    )
    # Pin the W&B entity so a run can never land in amrita-medicalai.
    .env({"WANDB_ENTITY": "anikethvij-personal"})
    .add_local_dir(
        LOCAL_REPO_ROOT,
        remote_path=REPO_ROOT,
        copy=True,
        ignore=[
            "**/.claude/**",
            "**/wandb/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/venv/**",
            "**/test_data/**",
            "**/checkpoints/**",
            "**/results/**",
            "**/visualizations/**",
        ],
    )
)

app = modal.App("usregistrator-trusted")
data_volume = modal.Volume.from_name("split_90_10_new", create_if_missing=False)
raw_volume = modal.Volume.from_name("kidney-dataset", create_if_missing=False)
wandb_secret = modal.Secret.from_name("wandb-secret")

# Cheapest-first GPU choice. T4 (16 GB) is the cheapest tier Modal offers, L4 (24 GB) the
# next one up; A100 and L40S are rejected on this workspace at *app-definition* time (no
# payment method), which fails every entrypoint in the file, not just the big one.
GPU = "T4"
BIG_GPU = "L4"


# --------------------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------------------
def _load_base_config() -> dict:
    import yaml

    with open(os.path.join(REPO_ROOT, BASE_CONFIG), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_overrides(
    cfg: dict,
    run_name: str,
    epochs: int | None = None,
    lr: float | None = None,
    image_size: int | None = None,
    lncc_weight: float | None = None,
    dice_weight: float | None = None,
    num_channel_initial: int | None = None,
    smooth_weight: float | None = None,
    align_shared_grid: bool | None = None,
) -> dict:
    cfg["wandb"]["run_name"] = run_name
    cfg["training"]["save_dir"] = f"/data/checkpoints/trusted/{run_name}"

    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    if lr is not None:
        cfg["optimizer"]["lr"] = float(lr)
    if image_size is not None:
        size = [int(image_size)] * 3
        cfg["image_size"] = size
        for split in ("train_dataset", "val_dataset"):
            cfg[split]["image_size"] = size
    if lncc_weight is not None:
        cfg["loss"]["params"]["lncc_weight"] = float(lncc_weight)
    if dice_weight is not None:
        cfg["loss"]["params"]["dice_weight"] = float(dice_weight)
    if num_channel_initial is not None:
        cfg["model"]["num_channel_initial"] = int(num_channel_initial)
    if smooth_weight is not None:
        cfg["loss"]["params"]["smooth_weight"] = float(smooth_weight)
    if align_shared_grid is not None:
        for split in ("train_dataset", "val_dataset"):
            cfg[split]["align_shared_grid"] = bool(align_shared_grid)
    return cfg


def _run_training(cfg: dict, run_name: str):
    """Write the resolved config next to the checkpoints, then hand off to train.py."""
    import subprocess
    import sys

    import yaml

    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    resolved = os.path.join(save_dir, "config.yaml")
    with open(resolved, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    data_volume.commit()

    print(f"=== {run_name} ===")
    print(yaml.safe_dump(cfg, sort_keys=False))
    result = subprocess.run(
        [sys.executable, "train.py", "--config", resolved],
        cwd=REPO_ROOT,
        check=False,
    )
    data_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited {result.returncode} for run '{run_name}'")
    return {"run_name": run_name, "save_dir": save_dir}


# --------------------------------------------------------------------------------------
# alignment check
# --------------------------------------------------------------------------------------
@app.function(image=image, volumes={"/data": data_volume}, timeout=3600, cpu=4, memory=32768)
def align_check_remote(split: str = "val", max_cases: int = 0):
    """Initial mask Dice per case with `align_shared_grid` off vs on.

    With no warp applied, Dice(moving_label, fixed_label) after preprocessing is the
    starting point the network has to improve on. The `off` column is what the sweep
    would have measured; `on` is what the shared-grid resample gives.
    """
    import sys

    sys.path.insert(0, REPO_ROOT)
    os.chdir(REPO_ROOT)

    import numpy as np

    from datasets.registry import build_dataset

    def dice(a, b):
        a = (a > 0.5).float()
        b = (b > 0.5).float()
        return float(2 * (a * b).sum() / (a.sum() + b.sum() + 1e-8))

    cfg = _load_base_config()
    ds_cfg = dict(cfg["val_dataset" if split == "val" else "train_dataset"])
    ds_cfg["json_file"] = MANIFEST

    results = {}
    for label, flag in (("off", False), ("on", True)):
        ds = build_dataset({**ds_cfg, "align_shared_grid": flag}, split=split)
        n = len(ds) if max_cases <= 0 else min(len(ds), max_cases)
        for i in range(n):
            sample = ds[i]
            cid = sample.get("case_id", str(i))
            results.setdefault(cid, {})[label] = dice(sample["moving_label"], sample["fixed_label"])
            del sample

    print(f"\n{'case':8} {'align off':>10} {'align on':>10} {'delta':>8}")
    for cid in sorted(results):
        row = results[cid]
        print(f"{cid:8} {row['off']:10.3f} {row['on']:10.3f} {row['on'] - row['off']:8.3f}")
    off = np.array([results[c]["off"] for c in sorted(results)])
    on = np.array([results[c]["on"] for c in sorted(results)])
    print(f"{'mean':8} {off.mean():10.3f} {on.mean():10.3f} {on.mean() - off.mean():8.3f}")
    print(f"{'std':8} {off.std():10.3f} {on.std():10.3f}")
    return results


@app.local_entrypoint()
def align_check(split: str = "val", max_cases: int = 0):
    align_check_remote.remote(split=split, max_cases=max_cases)


@app.function(image=image, volumes={"/data": raw_volume}, timeout=3600, cpu=4, memory=32768)
def align_check_raw_remote(split: str = "val", max_cases: int = 0):
    """Same check, but on the RAW TRUSTED volumes — no vendor pre-registration.

    `split_90_10_new` holds CT already resampled into US space (`*_in_US_space_*`), i.e. a
    precomputed warp. This runs against the untouched volumes on `kidney-dataset`, where CT
    is still in scanner coordinates, to isolate how much of the alignment the shared-grid
    resample achieves on its own versus how much it inherits from that pre-registration.
    """
    import sys

    sys.path.insert(0, REPO_ROOT)
    os.chdir(REPO_ROOT)

    import numpy as np

    from datasets.registry import build_dataset

    def dice(a, b):
        a = (a > 0.5).float()
        b = (b > 0.5).float()
        return float(2 * (a * b).sum() / (a.sum() + b.sum() + 1e-8))

    cfg = _load_base_config()
    ds_cfg = dict(cfg["val_dataset" if split == "val" else "train_dataset"])
    ds_cfg["json_file"] = RAW_MANIFEST

    results = {}
    for label, flag in (("off", False), ("on", True)):
        ds = build_dataset({**ds_cfg, "align_shared_grid": flag}, split=split)
        n = len(ds) if max_cases <= 0 else min(len(ds), max_cases)
        for i in range(n):
            sample = ds[i]
            cid = sample.get("case_id", str(i))
            results.setdefault(cid, {})[label] = dice(sample["moving_label"], sample["fixed_label"])
            del sample

    print(f"\nRAW volumes (no pre-registration) — {split} split")
    print(f"{'case':8} {'align off':>10} {'align on':>10} {'delta':>8}")
    for cid in sorted(results):
        row = results[cid]
        print(f"{cid:8} {row['off']:10.3f} {row['on']:10.3f} {row['on'] - row['off']:8.3f}")
    off = np.array([results[c]["off"] for c in sorted(results)])
    on = np.array([results[c]["on"] for c in sorted(results)])
    print(f"{'mean':8} {off.mean():10.3f} {on.mean():10.3f} {on.mean() - off.mean():8.3f}")
    print(f"{'std':8} {off.std():10.3f} {on.std():10.3f}")
    return results


@app.local_entrypoint()
def align_check_raw(split: str = "val", max_cases: int = 0):
    align_check_raw_remote.remote(split=split, max_cases=max_cases)


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600, cpu=4, memory=32768)
def tutorial_check_remote(align: bool = True):
    """Run `tutorials/trusted_ct_us_localnet` in --check-only mode against the volume.

    Exercises the tutorial exactly as a user would from the CLI, including the manifest
    scanner, so the tutorial stays honest about the real data layout.
    """
    import subprocess
    import sys

    os.chdir(REPO_ROOT)
    command = [
        sys.executable, "main.py",
        "--tutorial", "trusted_ct_us_localnet",
        "--data-root", DATA_ROOT,
        "--check-only",
    ]
    if not align:
        command.append("--no-align")
    print(" ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"main.py exited {result.returncode}")


@app.local_entrypoint()
def tutorial_check(align: bool = True):
    tutorial_check_remote.remote(align=align)


@app.function(image=image, volumes={"/data": data_volume}, timeout=1800, cpu=2, memory=16384)
def geometry_remote(split: str = "val", max_cases: int = 3):
    """Per-case grid geometry (headers only) plus the kidney centroid distance in mm.

    Shows why the resize alone cannot work: the volumes share a world frame but not a
    grid, so each one's field of view is a different size.
    """
    import json

    import numpy as np
    import SimpleITK as sitk

    with open(os.path.join(REPO_ROOT, MANIFEST), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows = []
    for case in manifest[split][: max_cases if max_cases > 0 else None]:
        row = {"case_id": case["case_id"]}
        for key, tag in (("moving", "CT"), ("fixed", "US")):
            reader = sitk.ImageFileReader()
            reader.SetFileName(case[key])
            reader.ReadImageInformation()
            size = np.array(reader.GetSize())
            spacing = np.array(reader.GetSpacing())
            row[f"{tag}_shape"] = tuple(int(s) for s in size)
            row[f"{tag}_spacing_mm"] = tuple(round(float(s), 3) for s in spacing)
            row[f"{tag}_fov_mm"] = tuple(round(float(v), 1) for v in size * spacing)
            row[f"{tag}_origin_mm"] = tuple(round(float(v), 1) for v in reader.GetOrigin())

        centroids = {}
        for key, tag in (("moving_label", "CT"), ("fixed_label", "US")):
            mask = sitk.ReadImage(case[key], sitk.sitkUInt8)
            idx = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)  # (z, y, x)
            centre = idx.mean(axis=0)[::-1]  # -> (x, y, z)
            centroids[tag] = np.array(
                mask.TransformContinuousIndexToPhysicalPoint([float(v) for v in centre])
            )
            row[f"{tag}_centroid_mm"] = tuple(round(float(v), 1) for v in centroids[tag])
            del mask
        row["centroid_dist_mm"] = round(float(np.linalg.norm(centroids["CT"] - centroids["US"])), 1)
        rows.append(row)

    for row in rows:
        print(json.dumps(row))
    return rows


@app.function(image=image, volumes={"/data": data_volume}, timeout=1800, cpu=4, memory=32768)
def preview_remote(case_id: str | None = None, split: str = "val"):
    """Render mid-slice CT/US overlays with alignment off vs on. Returns PNG bytes.

    Top row: alignment off (what the sweep would have trained on). Bottom row: on.
    Grey = US (fixed), red contour = US kidney mask, cyan contour = warped-CT kidney
    mask position. The two contours should sit on top of each other.
    """
    import io
    import sys

    sys.path.insert(0, REPO_ROOT)
    os.chdir(REPO_ROOT)

    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from datasets.registry import build_dataset

    cfg = _load_base_config()
    ds_cfg = dict(cfg["val_dataset" if split == "val" else "train_dataset"])
    ds_cfg["json_file"] = MANIFEST

    # Resolve the case index from the manifest — building a sample just to read its id
    # would load every volume in the split.
    with open(os.path.join(REPO_ROOT, MANIFEST), "r", encoding="utf-8") as f:
        ids = [c["case_id"] for c in json.load(f)[split]]
    index = ids.index(case_id) if case_id in ids else 0

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for row, (label, flag) in enumerate((("align OFF", False), ("align ON", True))):
        ds = build_dataset({**ds_cfg, "align_shared_grid": flag}, split=split)
        sample = ds[index]
        us = sample["fixed"][0].numpy()
        us_mask = sample["fixed_label"][0].numpy() > 0.5
        ct_mask = sample["moving_label"][0].numpy() > 0.5
        dice = 2 * (us_mask & ct_mask).sum() / (us_mask.sum() + ct_mask.sum() + 1e-8)
        centre = [s // 2 for s in us.shape]

        for col, (plane, name) in enumerate(((0, "sagittal"), (1, "coronal"), (2, "axial"))):
            slicer = [slice(None)] * 3
            slicer[plane] = centre[plane]
            ax = axes[row, col]
            ax.imshow(us[tuple(slicer)].T, cmap="gray", origin="lower")
            ax.contour(us_mask[tuple(slicer)].T, levels=[0.5], colors="red", linewidths=1.4)
            ax.contour(ct_mask[tuple(slicer)].T, levels=[0.5], colors="cyan", linewidths=1.4)
            ax.set_title(f"{label} — {name}" + (f"  (Dice {dice:.3f})" if col == 1 else ""))
            ax.axis("off")
        del sample, ds

    fig.suptitle(
        f"TRUSTED {split} case {case_id or 'first'} — US (grey) with US mask (red) "
        "vs CT mask (cyan), before any warp",
        fontsize=12,
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


@app.local_entrypoint()
def preview(case_id: str = "206L", split: str = "val", out: str = "visualizations/trusted_alignment.png"):
    png = preview_remote.remote(case_id=case_id, split=split)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.write(png)
    print(f"wrote {out} ({len(png) / 1024:.0f} KB)")


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    timeout=3600 * 8,
    volumes={"/data": data_volume},
    secrets=[wandb_secret],
    cpu=8,
    memory=32768,
)
def train_remote(run_name: str, **overrides):
    cfg = _apply_overrides(_load_base_config(), run_name, **overrides)
    return _run_training(cfg, run_name)


@app.function(
    image=image,
    gpu=BIG_GPU,
    timeout=3600 * 12,
    volumes={"/data": data_volume},
    secrets=[wandb_secret],
    cpu=8,
    memory=65536,
)
def train_remote_big(run_name: str, **overrides):
    """Route for the memory-heavy configs (208^3, num_channel_initial >= 12).

    This workspace has no payment method, so A100/L40S are rejected at app-definition
    time and BIG_GPU is the same A10G as `train_remote`. 208^3 and channels 16 may OOM
    in 24 GB — if they do, add a payment method and set BIG_GPU = "A100".
    """
    cfg = _apply_overrides(_load_base_config(), run_name, **overrides)
    return _run_training(cfg, run_name)


@app.local_entrypoint()
def smoke(align_shared_grid: bool = True):
    """1-epoch end-to-end check on the A10G."""
    train_remote.remote(
        run_name="trusted_localnet_smoke" + ("" if align_shared_grid else "_noalign"),
        epochs=1,
        align_shared_grid=align_shared_grid,
    )


@app.local_entrypoint()
def train_one(
    run_name: str = "trusted_localnet_aligned_baseline",
    epochs: int = 30,
    lr: float = 1e-5,
    image_size: int = 192,
    lncc_weight: float = 1.0,
    dice_weight: float = 0.5,
    num_channel_initial: int = 8,
    big: bool = False,
):
    fn = train_remote_big if big else train_remote
    fn.remote(
        run_name=run_name,
        epochs=epochs,
        lr=lr,
        image_size=image_size,
        lncc_weight=lncc_weight,
        dice_weight=dice_weight,
        num_channel_initial=num_channel_initial,
    )


# --------------------------------------------------------------------------------------
# sweep: one factor at a time around the baseline
# --------------------------------------------------------------------------------------
# Each entry varies exactly one axis from the baseline (lr 1e-5, lncc/dice 1.0/0.5,
# size 192, channels 8). `big` routes memory-heavy configs to the A100.
# smooth_weight leads because it is the term that was missing entirely: the unregularised
# baseline gains ~0.006 Dice while TRE degrades past the rigid initialisation and folding
# climbs. Until a value is found that keeps TRE below rigid, the other axes are measuring
# variations of the same failure.
SWEEP_RUNS = [
    dict(run_name="trusted_aligned_baseline", lr=1e-5, image_size=192, lncc_weight=1.0, dice_weight=0.5, num_channel_initial=8, smooth_weight=0.1),
    dict(run_name="trusted_aligned_sw0p02", smooth_weight=0.02),
    dict(run_name="trusted_aligned_sw0p5", smooth_weight=0.5),
    dict(run_name="trusted_aligned_lr2e5", lr=2e-5),
    dict(run_name="trusted_aligned_lr5e5", lr=5e-5),
    dict(run_name="trusted_aligned_dice0p3", lncc_weight=1.0, dice_weight=0.3),
    dict(run_name="trusted_aligned_dice1p0", lncc_weight=0.5, dice_weight=1.0),
    dict(run_name="trusted_aligned_sz176", image_size=176),
    dict(run_name="trusted_aligned_sz208", image_size=208, big=True),
    dict(run_name="trusted_aligned_ch12", num_channel_initial=12, big=True),
]


@app.local_entrypoint()
def sweep(epochs: int = 30):
    """Fan the sweep out over Modal containers — one container per run, all concurrent.

    This is a fixed one-factor-at-a-time grid, not a W&B HPO sweep: no search
    algorithm, no early stopping. W&B only records the runs.
    """
    small = [dict(r) for r in SWEEP_RUNS if not r.get("big")]
    large = [dict(r) for r in SWEEP_RUNS if r.get("big")]
    for runs in (small, large):
        for r in runs:
            r.pop("big", None)
            r["epochs"] = epochs

    print(f"launching {len(small)} A10G runs + {len(large)} A100 runs")
    handles = [train_remote.spawn(**r) for r in small]
    handles += [train_remote_big.spawn(**r) for r in large]
    for handle in handles:
        print(handle.get())
