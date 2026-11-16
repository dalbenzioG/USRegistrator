# Active Kidney Experiments — VoxelMorph (ASMUS paper)

Owner: Aniketh. SOTA baseline = **VoxelMorph** for the kidney CT→US comparison
(TransMorph = Shrisharanyan, ConvexAdam = Mengting, MultiGradICON = Gabriella).

## TL;DR / current state

- **Headline config:** `configs/kidney_vxm_lncc_dice.yaml` — LNCC+Dice loss, **176³**
  (matches Mengting's ConvexAdam; see U-Net note below for why not 175³),
  `smooth_weight=0.1`, `lr=5e-4`, cosine schedule, Dice-based model selection.
- **Run it:** `modal run --detach train_modal.py::full_lncc_dice` (uses an **A10G**,
  not T4 — 176³ VoxelMorph does not fit in 16 GB).
- **U-Net size constraint:** input edge must be divisible by 2^(#downsamplings).
  `nb_features` length = #downsamplings. Default 5 entries → needs /32 (175 and 176
  both fail → skip-concat crash). The config pins `nb_features: [16,32,32,32]`
  (4 levels → /16), so **176³ works (176/16=11)**. Don't revert to 5 levels at 176³.
- **Why the old runs looked like an identity transform:** see
  ["Near-identity" note](#why-it-looked-like-identity) — it is mostly *expected*,
  not a pure bug. The rigid pre-registration + shared mask-bbox crop already align
  the kidneys to ~0.9 mask Dice, so the deformable step only refines.

## Data + W&B (verified 2026-06-15)

Gabriella's **corrected** dataset is on the Modal volume **`split_90_10_new`**
(created 2026-06-15), mounted at `/data` → data root `/data/split_90_10_new`. The
earlier `split-90-10` volume had origins corrupted by independent cropping and
should not be used.

Pre-flight sanity check (`modal run train_modal.py::sanity`) on the test split
confirmed alignment + modality direction:

| case | pre-reg mask Dice |
|---|---|
| 206L | 0.898 |
| 258R | 0.828 |
| 348L | 0.815 |
| 371R | 0.829 |
| 641R | 0.883 |
| 721L | 0.884 |
| **mean** | **0.856** |

Mean ~0.86 (no warp) → CT and US kidneys are co-located and the CT→US direction is
correct (a swap/misalignment would read near 0). The sanity step logs per-case
pre-reg Dice and flags any case `< 0.2`; rerun it whenever the data changes.

**W&B routes to the personal account.** Runs land in
`wandb.ai/anikethvij-personal/USRegistrator` (entity `anikethvij-personal`, set
explicitly in the config + every `wandb.init`), never `amrita-medicalai`.

`datasets/kidney.py` resamples CT and US onto a *single shared US-mask world grid*
(identity direction), which is the correct fix for the affine issue — but it can
only work if the source NIfTI affines are themselves correct, hence the sanity check.

## Shared setup

- Direction: **moving = CT**, **fixed = US** (warp CT into intra-op US space).
- Volume: `split_90_10_new`; data root `/data/split_90_10_new` (presplit `train/` + `test/`).
- Image size: **176³** (headline). 128³ configs kept for quick smoke tests.
- Optimizer: Adam, `lr=5e-4`, `weight_decay=1e-5`, cosine schedule, grad-clip 1.0.
- `batch_size=1`, `num_workers=0`, `amp=false`, `val_every=1`, `seed=42`.
- Model backend: official `voxelmorph` repo (`VxmPairwise`, displacement field,
  `integration_steps=0`, flow init std `1e-3`).

## Experiments

### 1. LNCC + Dice (headline) — `configs/kidney_vxm_lncc_dice.yaml`

```text
total = 1.0 * LNCC(warped_CT, US) + 1.0 * Dice(warped_CT_mask, US_mask) + 0.1 * ||∇DDF||²
```

- Selection: `dice` (max). Was previously over-regularized (`smooth_weight=0.25`,
  `lr=1e-4`) → `grad_l2≈0.07`, visually identical to the input. Now `0.1 / 5e-4`.
- NOTE: this file used to be mislabeled `loss.name: mind_dice` — fixed to `lncc_dice`.

### 2. MIND variants (ablation rows)

- `configs/kidney_vxm.yaml` — **MIND image-only** (no Dice), 128³, legacy baseline
  (`loss.name: mind`). Run via `train_modal.py::full`.
- MIND is more modality-invariant for CT–US than LNCC, so MIND+Dice is a natural
  ablation against the headline. The earlier checkpoint
  `kidney_voxelmorph_mind_dice_ct2us` (128³ MIND+Dice, epoch 66) reached val Dice
  0.879 but grad_l2 0.067 — i.e. near-identity (see note below). To redo it at 176³,
  copy the headline config and set `loss.name: mind_dice` (the `mind_dice` loss is
  registered in `losses/`).

## Metrics (paper table — aligned with ConvexAdam/MultiGradICON)

Computed in the dedicated inference pass (`eval_checkpoint_modal`):

| Metric | Source | Notes |
|---|---|---|
| Dice | `metrics/segmentation.py` | mask overlap after warp |
| HD95 | `metrics/segmentation.py` | re-enabled at inference |
| ASD | `metrics/segmentation.py` | average symmetric surface dist (new) |
| SDlogJ | `metrics/jacobian.py` | std of log\|J\| over non-folding voxels (new) |
| neg_jac_frac | `metrics/jacobian.py` | folding ratio |
| jac_det_mean/std, grad_l2 | | field regularity |
| mse, mae, ncc | | image agreement (ncc low for CT–US, expected) |

- **HD95/ASD are skipped during per-epoch train/val** (expensive + unstable) and
  computed only at inference.
- **mTRE [mm]:** Gabriella asked for it, but the cropped dataset does **not** retain
  paired landmark files (same limitation Mengting hit for ConvexAdam). TRE needs the
  original landmarks mapped into the resampled 176³ space — blocked until those are
  recovered.
- **EPE:** not applicable (no ground-truth DVF for real data).
- Always report **Dice before vs after** so VoxelMorph's actual contribution over the
  rigid init is explicit.

## Result — test set (best.pt, epoch 83, 6 cases, mean ± std)

Run `kidney_vxm_lncc_dice_176_sw0p1_lr5e4` (W&B run 4g28kvpq):

> **Checkpoint provenance (2026-08-07).** The epoch-83 weights behind this table were
> overwritten on the volume on 2026-06-25 by a later rerun (epoch 110, val Dice 0.8772).
> They were recovered from the W&B artifact
> `kidney_vxm_lncc_dice_176_sw0p1_lr5e4-checkpoints:v0` and now live at
> `/checkpoints/kidney_vxm_lncc_dice_176_sw0p1_lr5e4_epoch83/best.pt`. Per-case metrics
> for the camera-ready statistical analysis: `results/voxelmorph_trusted_test/`.
> Note TRE here is over **all 7 landmarks**; `eval_checkpoint_modal` had drifted to a
> hardcoded 3-landmark subset, and `chamfer` had drifted to the *sum* of directed means
> (2× this table). Both are fixed and the numbers below reproduce exactly.

| Metric | VoxelMorph | comparable to table? |
|---|---|---|
| TRE [mm] ↓ | 5.87 ± 1.54 (rigid-only 4.92 ± 2.03) | ⚠️ **worse than rigid** |
| Chamfer [mm] ↓ | 1.99 ± 0.73 | ❌ definition mismatch (≈ASD; others ~5) |
| Dice ↑ | 0.882 ± 0.032 | ✅ competitive (others ~0.86) |
| MI ↑ | 0.291 ± 0.130 | ❌ scale mismatch (others ~0.10) |
| HD95 [mm] ↓ | 6.25 ± 2.63 | ✅ competitive (others 6–7) |
| folding (neg-Jac) | 0.050 | high; \|DDF\|max ≈ 50 vox |

**Caveats before this goes in the paper:**
- **TRE degraded vs the rigid init on all 6 cases** — the field Dice-overfits (smooth_weight=0.1 too low, \|DDF\|max ≈ 50 vox ≈ 25 mm, 5% folding). Consistent with the team's "TRE best with rigid only" observation. Fix: retrain with smooth_weight ~0.2 and/or diffeomorphic integration (`integration_steps>0`).
- **Chamfer and MI are not yet comparable**: this repo's Chamfer is symmetric mean surface distance (≈ ASD ≈ 2 mm) vs the table's ~5; MI is whole-volume nats (~0.29) vs the table's ~0.10. Reconcile against the team's eval code before merging into the shared table. Dice + HD95 (mm) are standard and safe to report.

## Deformation-field export

`eval_checkpoint_modal` now writes, per test case, under
`/data/checkpoints/<run>/inference_<split>/`:

- `<i>_<id>_ddf.nii.gz` — predicted displacement field, 3-component vector NIfTI,
  voxel units, channels (dz, dy, dx), on the identity-direction cube.
- `<i>_<id>_warpedCT.nii.gz`, `<i>_<id>_warpedCTmask.nii.gz` — warped moving volume/mask.
- `<i>_<id>_ddf.png` — |DDF| tri-planar heatmap + coronal quiver (logged to W&B `*/ddf`).

Local equivalent: `python visualize_registration.py --checkpoint <ckpt> --data-root
test_data --image-size 176` (preprocessing matches training; CT→US direction).

## <a name="why-it-looked-like-identity"></a>"Near-identity" note

Pre-registration mask Dice on case 206L is **0.898** with no warp applied — the
rigid pre-registration + shared US-mask-bbox crop already co-center and co-scale the
kidneys. So the optimal deformable correction is genuinely small, and a near-identity
field with high Dice is partly *correct behavior*, not only a collapse. Levers that
still make the field do real (smooth, low-folding) refinement:

1. **Dice loss term** — modality-invariant, strong boundary gradient (LNCC alone is
   weak across CT/US). Already `dice_weight=1.0`.
2. **Lower smoothness** — `0.25 → 0.1` (folding stayed ~6e-4, lots of headroom).
3. **Higher LR** — `1e-4 → 5e-4` with cosine decay + grad clipping.

Open methodological point for the team: cropping both modalities to the *US mask*
bbox co-centers the kidneys and inflates the "before registration" Dice (~0.9) vs
Mengting's ConvexAdam "before" (~0.58). A fixed field-of-view crop (not re-centered
per mask) would give a fairer before/after delta. Raise with Gabriella before
finalizing the comparison table.

## How to run

```bash
# 0. Verify data alignment (cheap)
modal run train_modal.py::sanity

# 1. Headline training run (176³ LNCC+Dice on A10G, detached)
modal run --detach train_modal.py::full_lncc_dice

# 2. Inference + metrics + DDF export on the test split
modal run train_modal.py::eval_checkpoint \
  --checkpoint-run-name kidney_vxm_lncc_dice_176_sw0p1_lr5e4 \
  --split test --max-cases 6
```
