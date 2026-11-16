# TRUSTED CT→US Registration with LocalNet3D

> **Superseded for running experiments (2026-08-11).** The data + preprocessing this
> tutorial describes are wrong: the manifest points at the *raw* TRUSTED volumes, and the
> loader assumes CT and US share a grid when they do not. Use
> [`configs/trusted_localnet3d_aligned.yaml`](../../configs/trusted_localnet3d_aligned.yaml)
> with `train_modal_trusted.py`, and read
> [`05_trusted_data_alignment_issue.md`](05_trusted_data_alignment_issue.md) plus
> [`notebooks/trusted_alignment_and_localnet3d.ipynb`](../../notebooks/trusted_alignment_and_localnet3d.ipynb).
> **The ~0.63 → ~0.78 Dice figures below are not reproducible** — aligned runs start at
> ~0.86. The model/loss/metrics sections are still accurate.

This tutorial covers deformable registration on the **TRUSTED** CT→US kidney dataset
using **LocalNet3D**. In our current setup LocalNet3D clearly outperforms
GlobalNet3D on TRUSTED (val Dice ~0.63 → ~0.78 over the first 10 epochs), so it is
the recommended starting point.

- **Direction:** moving = CT, fixed = US (warp the pre-op CT into intra-op US space).
- **Model:** `localnet3d` (encoder–decoder that regresses a dense displacement field).
- **Loss:** `lncc_dice` — LNCC on intensities + Dice on the warped CT mask vs US mask.
- **Reference config:** [`configs/trusted_localnet3d.yaml`](../../configs/trusted_localnet3d.yaml).

## Why LocalNet3D over GlobalNet3D

GlobalNet3D predicts a single global affine-like transform, which cannot capture the
local kidney deformation between CT and intra-op US. LocalNet3D predicts a dense
per-voxel field, so it can refine the rigid pre-alignment. On TRUSTED:

| Model | val_dice @ ep1 | val_dice @ ep10 | notes |
|---|---|---|---|
| globalnet3d | ~0.63 | plateaus low | global transform only |
| localnet3d | ~0.63 | **~0.78** | learns non-trivial deformation |

## Data layout

Each TRUSTED case provides four volumes; the manifest maps them to the loader keys:

| Manifest key | File | Role |
|---|---|---|
| `moving` | `CT_images/<id>_imgCT.nii.gz` | moving image (CT) |
| `fixed` | `US_images/<id>_imgUS.nii.gz` | fixed image (US) |
| `moving_label` | `CT_masks/<id>_maskCT.nii.gz` | CT kidney mask (Dice + warp) |
| `fixed_label` | `US_masks/<id>_maskUS.nii.gz` | US kidney mask (Dice target) |

- `configs/trusted_manifest.json` — original manifest (paths on the data owner's machine).
- `configs/trusted_manifest_modal.json` — same 48 train / 5 val split, paths rewritten to
  the Modal `kidney-dataset` volume (`/data/Kidney_Dataset/TRUSTED/...`). Regenerate with
  the snippet in [Appendix](#appendix-regenerating-the-modal-manifest).

## Running on Modal (recommended)

Training runs on the **`anikethvij464`** Modal workspace. The TRUSTED data lives on the
`kidney-dataset` volume and W&B is wired to the **`anikethvij-personal`** entity via the
`wandb-secret` (the image also hard-pins `WANDB_ENTITY=anikethvij-personal`, so a run can
never log to `amrita-medicalai`).

```bash
# activate the workspace that holds the data + secret
modal profile activate aniketh          # workspace: anikethvij464

# 1) cheap 1-epoch end-to-end check (A10G)
modal run train_modal.py::smoke

# 2) a single configurable run (detached), overriding any hyperparameter
modal run --detach train_modal.py::train_one \
  --run-name trusted_localnet_baseline --lr 1e-5 --image-size 192
# ...use train_one_big (A100) for memory-heavy configs (208^3 or channels 12/16):
modal run --detach train_modal.py::train_one_big --run-name trusted_localnet_sz208 --image-size 208

# 3) the full one-factor-at-a-time sweep (detached) — auto-routes each run to A10G or A100
modal run --detach train_modal.py::sweep
```

GPU routing: `train_one` runs on **A10G** (24 GB) for the light configs; `train_one_big`
runs on **A100** (40 GB) for the memory-heavy ones. The `sweep` entrypoint dispatches each
run automatically (208³ and channels 12/16 → A100, everything else → A10G).

Checkpoints and the resolved `config.yaml` persist to the volume under
`/data/checkpoints/trusted/<run-name>/` (`best_val_loss.pt`, `last.pt`). W&B run names
match `<run-name>`.

## Running locally

Point the manifest at data reachable from the machine and run:

```bash
python train.py --config configs/trusted_localnet3d.yaml
```

Validation runs inside the training loop (`training.val_every: 1`).

## Metrics to monitor

The validation loop logs these every epoch (W&B keys under `val/`, and a summary line in
stdout). For TRUSTED, watch:

| Metric | W&B key | What good looks like |
|---|---|---|
| Dice (mask overlap after warp) | `val/dice` | **primary** — should climb from ~0.63 toward ~0.78+ |
| LNCC / NCC (image agreement) | `val/ncc` | slow rise; stays low for CT–US (modality gap), that's expected |
| Total val loss | `val/loss` | steadily decreasing |
| Negative-Jacobian ratio (folding) | `val/neg_jac_ratio` | keep small; a spike means the field is folding |
| log-Jacobian std (field regularity) | `val/log_jac_std` | moderate; runaway values ⇒ over-deforming |
| DDF magnitude | `val/ddf_abs_mean`, `val/ddf_l2_mean` | non-zero (learning) but not exploding |

A healthy LocalNet3D run shows **Dice rising while folding stays low**. If Dice climbs
but `neg_jac_ratio`/`log_jac_std` blow up, the field is Dice-overfitting — lower the
learning rate or reduce `dice_weight`.

## Hyperparameter sweep

`train_modal.py::sweep` runs a one-factor-at-a-time sweep around the baseline
(`lr=1e-5`, `lncc/dice=1.0/0.5`, `size=192`, `channels=8`):

| Axis | Values |
|---|---|
| Learning rate | 1e-5 (base), 2e-5, 5e-5 |
| Loss balance (lncc, dice) | (1.0, 0.5) base, (1.0, 0.3), (0.5, 1.0) |
| Image size | 176, 192 (base), 208 |
| `num_channel_initial` | 8 (base), 12, 16 |

### Results

<!-- Filled in after the sweep completes. -->

| Run | lr | lncc/dice | size | ch | best val_dice | neg_jac | notes |
|---|---|---|---|---|---|---|---|
| trusted_localnet_baseline | 1e-5 | 1.0/0.5 | 192 | 8 | _tbd_ | _tbd_ | |

## Troubleshooting

- **Image size must be divisible by 2^(#levels).** With `extract_levels: [0,1,2,3]` (4
  levels) the edge must be divisible by 16: 176 (=11×16), 192 (=12×16), 208 (=13×16) all
  work; 175 does not. Pick sizes from that set for the size sweep.
- **CUDA out of memory** at 208³ or `num_channel_initial: 16`: use `train_one_big` (A100)
  instead of `train_one` (the `sweep` already routes these to A100 automatically).
- **Epoch runs very slowly / times out:** the config uses `num_workers: 4` and `cpu=8` on
  Modal for parallel NIfTI loading + resize. The first epoch is slower (cold reads from the
  volume); later epochs speed up once files are cached in the container.
- **Deformation collapse (near-identity field):** `val/ddf_abs_mean` ≈ 0 and Dice flat.
  The rigid pre-alignment already gives ~0.6 Dice, so some smallness is expected — but a
  truly flat field means the model isn't learning; raise the learning rate.
- **Dice up but folding up too:** reduce `dice_weight` (e.g. 0.3) or lower `lr`.
- **`references missing file`:** a manifest path doesn't resolve. For Modal, confirm the
  case exists under `/data/Kidney_Dataset/TRUSTED/` (59 cases have all four files).
- **W&B logging to the wrong place:** the Modal image pins `WANDB_ENTITY=anikethvij-personal`;
  verify with `modal run train_modal.py::whoami`.

## Appendix: regenerating the Modal manifest

```python
import json
src = json.load(open("configs/trusted_manifest.json"))
ROOT = "/data/Kidney_Dataset/TRUSTED"
def conv(c):
    cid = c["case_id"]
    return {
        "case_id": cid, "moving_modality": "CT", "fixed_modality": "US",
        "moving":       f"{ROOT}/CT_images/{cid}_imgCT.nii.gz",
        "fixed":        f"{ROOT}/US_images/{cid}_imgUS.nii.gz",
        "moving_label": f"{ROOT}/CT_masks/{cid}_maskCT.nii.gz",
        "fixed_label":  f"{ROOT}/US_masks/{cid}_maskUS.nii.gz",
    }
out = {s: [conv(c) for c in src[s]] for s in ("train", "val")}
json.dump(out, open("configs/trusted_manifest_modal.json", "w"), indent=2)
```
