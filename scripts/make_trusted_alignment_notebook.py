"""Generate notebooks/trusted_alignment_and_localnet3d.ipynb."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md("""
# TRUSTED CT→US: the data-alignment fix, and how to run LocalNet3D

This notebook is the worked version of
[`docs/deepreg-pipeline/05_trusted_data_alignment_issue.md`](../docs/deepreg-pipeline/05_trusted_data_alignment_issue.md).
It diagnoses why the LocalNet3D sweep was paused, verifies the fix on real data, and
launches the runs.

**Everything heavy runs on Modal** (profile `aniketh`, workspace `anikethvij464`). The
cells below only send RPCs and print what comes back — no volumes are loaded locally.

## TL;DR

| | init Dice (val, n=6) | init Dice (train, n=53) |
|---|---|---|
| raw `kidney-dataset` volumes, as the paused sweep would have seen them | **0.14** | — |
| co-registered `split_90_10_new`, `align_shared_grid: false` | **0.580 ± 0.101** | **0.533 ± 0.099** |
| co-registered `split_90_10_new`, `align_shared_grid: true` | **0.856 ± 0.033** | **0.855 ± 0.063** |

Two separate problems were stacked on top of each other:

1. **Wrong data.** The manifest pointed at `kidney-dataset`, which holds the *raw*
   TRUSTED volumes — CT in scanner coordinates (world z ≈ 1 m), US in its own probe
   frame. Kidney centroids ~1 m apart. The co-registered volumes were on Modal all
   along, on the `split_90_10_new` volume (CT files named `*_in_US_space_*`).
2. **A loader that assumes its inputs share a grid.** Even co-registered, CT and US
   have different fields of view, and `InterpolateToSized` stretches each one onto the
   192³ cube by a *different* factor. Fixing the data alone only gets to 0.58.

The fix is both: repoint the manifest, and add a shared-grid resample
(`align_shared_grid: true`) before the resize.
""")

md("""
## 0. Setup

Nothing is imported from the dataset here — `train_modal_trusted` is just the Modal app
definition, and `app.run()` opens an ephemeral app so the functions can be called
directly from the notebook (no `modal deploy` needed).
""")

code("""
import sys
sys.path.insert(0, "..")   # notebook lives in notebooks/

import modal
import train_modal_trusted as tmt

print("app:", tmt.app.name)
print("volume:", tmt.DATA_ROOT)
print("manifest:", tmt.MANIFEST)
print("base config:", tmt.BASE_CONFIG)
""")

code("""
import json

with open("../" + tmt.MANIFEST) as f:
    manifest = json.load(f)

print({split: len(cases) for split, cases in manifest.items()})
print("val cases:", [c["case_id"] for c in manifest["val"]])
print()
print(json.dumps(manifest["val"][0], indent=2))
""")

md("""
The val cases are the same six the VoxelMorph kidney baseline holds out, so LocalNet3D
numbers from this manifest are directly comparable to the VoxelMorph table in
[`docs/active-kidney-experiments.md`](../docs/active-kidney-experiments.md).
""")

md("""
## 1. Why the resize alone cannot work

`geometry_remote` reads only the NIfTI headers plus the mask centroids. Watch the
**field of view** row: CT and US share a world frame (same 0.3 mm spacing, same
direction, origins tens of mm apart — not the ~1 m of the raw data), but they cover
*different amounts of space* on *different grids*.
""")

code("""
with modal.enable_output(), tmt.app.run():
    geom = tmt.geometry_remote.remote(split="val", max_cases=3)

for row in geom:
    print(f"--- {row['case_id']}  (kidney centroids {row['centroid_dist_mm']} mm apart)")
    for tag in ("CT", "US"):
        print(f"    {tag}: shape {row[f'{tag}_shape']}  spacing {row[f'{tag}_spacing_mm']} mm")
        print(f"        FoV {row[f'{tag}_fov_mm']} mm   origin {row[f'{tag}_origin_mm']} mm")
""")

md("""
So for case 206L the CT covers ~173 × 173 × 161 mm on a 576 × 576 × 535 grid while the US
covers ~216 × 168 × 245 mm on a 720 × 560 × 816 grid. `InterpolateToSized` squeezes each
of those onto the same 192³ cube, which applies a different mm-per-voxel scaling per
modality and shifts the kidney to a different place in each cube. The affines say the two
volumes are aligned; the tensors that reach the model are not.
""")

md("""
## 2. The fix: one shared grid before the resize

`AlignToSharedGridd` (in `datasets/custom_dataset.py`) fixes a single target grid per
case and resamples everything onto it:

1. Crop `align_reference` (default `fixed_label`, the US kidney mask) to its foreground
   bounding box padded by `align_margin_mm`. That cropped reference *is* the target grid.
2. Resample every other key onto it through its affine — nearest for `*_label` /
   `*_mask`, bilinear for images.

Then `InterpolateToSized` applies one common scaling to volumes that already share a
grid, so alignment survives.

The ROI comes from **one** reference for all keys. Cropping each modality to its own mask
would re-centre CT and US on each other and inflate the "before registration" score —
the open methodological point flagged in `docs/active-kidney-experiments.md`. This keeps
the residual misalignment honest.
""")

code("""
import inspect
sys.path.insert(0, "..")
from datasets.custom_dataset import AlignToSharedGridd

print(inspect.getsource(AlignToSharedGridd))
""")

md("""
## 3. Verify it on real data

`align_check_remote` builds the dataset twice — `align_shared_grid` off, then on — and
reports `Dice(moving_label, fixed_label)` with **no warp applied**. That is the starting
point the network has to improve on, so it is the number the paused sweep got wrong.

Pass `split="train"` for all 53 training cases (slower; recorded result 0.533 -> 0.855,
every case improving, deltas +0.036 to +0.514). Recorded val result:

```
case      align off   align on    delta
206L          0.696      0.898    0.203
258R          0.484      0.831    0.347
348L          0.734      0.815    0.080
371R          0.571      0.827    0.257
641R          0.513      0.884    0.371
721L          0.480      0.884    0.403
mean          0.580      0.856    0.277
std           0.101      0.033
```
""")

code("""
with modal.enable_output(), tmt.app.run():
    scores = tmt.align_check_remote.remote(split="val")

off = [scores[c]["off"] for c in sorted(scores)]
on = [scores[c]["on"] for c in sorted(scores)]
print(f"mean init Dice: {sum(off)/len(off):.3f} (off) -> {sum(on)/len(on):.3f} (on)")
""")

md("""
Across all 59 cases (both splits) every single one improves, and none lands near the
`< 0.2` threshold the VoxelMorph sanity check flags. The hardest cases after alignment are
263L (0.628), 398R (0.698), 517R (0.703) and 314L (0.763) — real residual misalignment
left by the rigid co-registration, which is exactly what the deformable model is for.

0.856 is not an arbitrary improvement — it is the *same* number the VoxelMorph track's
pre-flight check reports on these six cases (mean 0.856, 206L 0.898). Two independent
preprocessing paths landing on the same value is the strongest evidence available that
the geometry is now being handled correctly.

A full-resolution cross-check on 206L (CT mask resampled onto the US mask grid, no cube
resize at all) also gives **0.898** — identical to what the aligned 192³ pipeline
produces. The resize to the cube costs nothing in alignment once the grids are shared.
""")

md("""
## 4. Look at it

Grey = US, red contour = US kidney mask, cyan contour = CT kidney mask, before any warp.
Top row is `align_shared_grid: false`, bottom row is `true`. Rendered on Modal; only the
PNG comes back.
""")

code("""
from IPython.display import Image, display

with modal.enable_output(), tmt.app.run():
    png = tmt.preview_remote.remote(case_id="206L", split="val")

display(Image(data=png))
""")

md("""
## 5. Launch training

Run these from a terminal (`modal profile activate aniketh` first). Use `--detach` so a
closed laptop does not kill the run — a non-detached `modal run` dies with its client.

A 1-epoch check on this config (W&B run `j1vih79m`) reaches `val_dice = 0.8571` with the
field still near-identity (`ddf_abs = 0.36` vox at `lr=1e-5`) — training starts from the
corrected baseline. Watch `val_logjac_std` (1.21 at epoch 1) and `val_neg_jac` (0.69%):
both are high for a field that small, where log|J| should be ~0.

```bash
# 1-epoch end-to-end check on an A10G
modal run --detach train_modal_trusted.py::smoke

# the aligned baseline, 30 epochs
modal run --detach train_modal_trusted.py::train_one \\
    --run-name trusted_localnet_aligned_baseline --epochs 30

# the one-factor-at-a-time sweep (9 runs, all concurrent)
modal run --detach train_modal_trusted.py::sweep
```

Checkpoints and the resolved `config.yaml` land on the volume under
`/data/checkpoints/trusted/<run-name>/`. W&B goes to
`anikethvij-personal/USRegistrator` — the image pins `WANDB_ENTITY`, so a run cannot
land in `amrita-medicalai`.

**GPUs are the cheapest tiers available:** `GPU = "T4"` (16 GB, cheapest Modal offers)
and `BIG_GPU = "L4"` (24 GB, next cheapest) for the 208³ / channels 12-16 points. A100 and
L40S are rejected when the app is *defined*, not when it runs, so a single A100 function
makes **every** entrypoint in the file fail — do not add one speculatively.
""")

md("""
## 6. About the sweep — it is not a W&B HPO sweep

`train_modal_trusted.py::sweep` is a **hand-rolled one-factor-at-a-time grid**: a Python
list of 9 configs, each varying exactly one axis from the baseline, fanned out over Modal
containers with `.spawn()`. There is no search algorithm, no `wandb.sweep()`, no sweep
agent polling a controller, and no early stopping. W&B is only the logging sink.
""")

code("""
for run in tmt.SWEEP_RUNS:
    print(run)
""")

md("""
| Axis | Values |
|---|---|
| Learning rate | 1e-5 (base), 2e-5, 5e-5 |
| Loss balance (lncc, dice) | (1.0, 0.5) base, (1.0, 0.3), (0.5, 1.0) |
| Image size | 176, 192 (base), 208 |
| `num_channel_initial` | 8 (base), 12, 16 |

One-factor-at-a-time is 9 runs instead of 81, but it only finds the best value of each
axis *holding the others at baseline* — it cannot see interactions (e.g. a higher LR that
only helps at 16 channels). If you want interactions, a real `wandb.sweep()` with random
or Bayesian search is the upgrade; it would replace the `SWEEP_RUNS` list with a sweep
config and have each Modal container run `wandb.agent`. Worth doing only if the OFAT pass
suggests the axes are not independent.

Things to watch per run (`val/` keys in W&B):

| Metric | Healthy |
|---|---|
| `val/dice` | climbing from ~0.86 — **note the new baseline**, not the ~0.63 in the old tutorial |
| `val/tre_mm` | below `val/tre_mm_before` (the rigid init). If it goes *above*, the field is Dice-overfitting — the exact failure VoxelMorph hit |
| `val/ncc` | slow rise; stays low for CT–US, expected |
| `val/neg_jac_ratio` | small and stable; a spike means the field is folding |
| `val/log_jac_std` | SDlogJ over non-folding voxels. Falls as the field smooths: a white-noise field with \|u\| = 0.36 vox scores ~0.73, a smooth one of the same magnitude ~0.15 |
| `val/ddf_abs_mean` | non-zero but not exploding |

`val/tre_mm` is computed on the 6 val cases only — landmarks ship with the test split, not
the 53 training cases — and requires `align_shared_grid: true` (without a shared grid there
is no single geometry to map world mm onto, so it stays NaN).

Dice climbing while folding stays low is the signal. Dice climbing while
`neg_jac_ratio` / `log_jac_std` blow up is Dice-overfitting — exactly what bit the
VoxelMorph run, whose TRE came out *worse* than its rigid init at `smooth_weight=0.1`.
Starting from 0.856 instead of 0.58 means there is much less Dice headroom to chase, so
watch for the field doing large deformations for small Dice gains.
""")

md("""
## 7. Open items

- **The old tutorial's numbers are not reproducible as written.** The ~0.63 → ~0.78 in
  `04_trusted_localnet3d_tutorial.md` matches co-registered data *without* the shared-grid
  resample (init 0.58). Runs on this config start at 0.856, so the sweep result table
  needs re-filling from scratch rather than comparison against those figures.
- **`train_modal.py` is gone from the working tree.** Only a stale
  `__pycache__/train_modal.cpython-310.pyc` (the VoxelMorph/kidney variant) survives; the
  TRUSTED entrypoints it once had (`train_one`, `train_one_big`, `sweep`) are in no
  branch. `train_modal_trusted.py` is a fresh runner, deliberately under a new name so it
  cannot clobber the kidney one if that gets restored from `synthetic-data-tutorials`.
- **Landmarks / mTRE.** `split_90_10_new/test/landmarks_lps/` has paired
  `*_ldkCT_aligned_lps.txt` / `*_ldkUS_lps.txt` files, but they are not in the manifest
  and `AlignToSharedGridd` does not yet map point coordinates onto the new grid. TRE on
  the TRUSTED LocalNet3D track is blocked on that, and given VoxelMorph's TRE regression
  it is worth wiring up before trusting Dice alone.
- **`align_margin_mm: 6.0`** is inherited from the VoxelMorph track's 20 voxels at
  0.3 mm. It sets how much context around the kidney the model sees; it is untuned.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("/home/aniketh/Projects/USRegistrator/notebooks/trusted_alignment_and_localnet3d.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}: {len(cells)} cells")
