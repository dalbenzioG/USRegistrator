# TRUSTED CT→US data-alignment issue (blocked the LocalNet3D sweep)

**Status: resolved 2026-08-11** — options 2+3 implemented, verified on Modal. The sweep is
unblocked; see [Resolution](#resolution) and the worked notebook
[`notebooks/trusted_alignment_and_localnet3d.ipynb`](../../notebooks/trusted_alignment_and_localnet3d.ipynb).
**Filed:** 2026-07-26. **Workspace:** Modal `anikethvij464`, W&B `anikethvij-personal/USRegistrator`.

## Resolution

There were **two** stacked problems, and fixing only the data gets you halfway.

1. **Wrong data.** The co-registered TRUSTED volumes were on Modal the whole time — the
   `split_90_10_new` volume (created 2026-06-15 for the VoxelMorph kidney track), where the
   CT files are named `*_imgCT_in_US_space_crop*.nii.gz`. Option 1 needed no ask to
   Gabriella. The manifest was pointing at `kidney-dataset`, which holds the raw volumes.
2. **The loader still assumed a shared grid.** Even co-registered, CT and US cover
   different fields of view (206L: CT ~173×173×161 mm on 576×576×535, US ~216×168×245 mm on
   720×560×816, both 0.3 mm iso, same world frame). `InterpolateToSized` stretches each FoV
   onto the 192³ cube by a *different* factor, so the pair still arrives misaligned.

Verified initial mask Dice, no warp applied (`train_modal_trusted.py::align_check`, run on
Modal over both splits — 53 train + 6 val, all 59 cases):

| data + preprocessing | init Dice (val, n=6) | init Dice (train, n=53) |
|---|---|---|
| raw `kidney-dataset`, no alignment (the paused smoke run) | 0.14 | — |
| co-registered `split_90_10_new`, `align_shared_grid: false` | 0.580 ± 0.101 | 0.533 ± 0.099 |
| **co-registered `split_90_10_new`, `align_shared_grid: true`** | **0.856 ± 0.033** | **0.855 ± 0.063** |

Every one of the 59 cases improves (train deltas +0.036 to +0.514, mean +0.323), and none
lands anywhere near the `< 0.2` threshold the VoxelMorph sanity check flags. The hardest
cases after alignment are **263L (0.628), 398R (0.698), 517R (0.703), 314L (0.763)** —
that is real residual misalignment left by the rigid co-registration, not a pipeline
failure, and it is where the deformable model has the most to do.

A 1-epoch smoke run on this config (`train_modal_trusted.py::smoke`) trains end-to-end and
lands at `val_dice = 0.8571`, with the field still near-identity (`ddf_abs = 0.36` vox at
`lr=1e-5`) — i.e. it starts from the corrected baseline instead of climbing out of a
misalignment hole. Identical Dice on both an A10G (W&B `j1vih79m`) and a **T4**, so the
cheapest GPU tier is enough for LocalNet3D at 192³ / 8 channels:

| | A10G run | T4 run (current config) |
|---|---|---|
| `val_dice` | 0.8571 | 0.8571 |
| `val_ddf_abs` | 0.3585 | 0.3615 |
| `val_neg_jac` | 0.0069 | 0.0070 |
| `val_logjac_std` | 1.2083 *(clamped, pre-fix)* | **0.4303** *(true SDlogJ)* |
| `val_tre_mm` | not yet wired | **4.903** (rigid 4.917) |

The `logjac_std` drop is the metric fix, not a change in the field — `ddf_abs` and
`neg_jac` are the same to three decimals across the two runs.

**TRE sanity check.** `val/tre_mm_before` = **4.917 mm** independently reproduces the
VoxelMorph track's rigid-only TRE of **4.92 ± 2.03 mm** on the same 6 cases and 7
landmarks, which confirms landmark parsing and CT↔US pairing (a distance is invariant to
the LPS/RAS sign flip, so that part is what it checks). The grid mapping and DDF axis order
are verified separately, by a synthetic constant-DDF test that recovers an exact zero
residual in both `(z, y, x)` and `(x, y, z)` orders. `tre_after` barely moves at epoch 1
because the field is still noise — a trained run is what will really exercise it.

0.856 is the same figure the VoxelMorph pre-flight check reports on these six cases
(mean 0.856, 206L 0.898) — two independent preprocessing paths agreeing. A full-resolution
cross-check on 206L (CT mask resampled onto the US mask grid, no cube resize) also gives
0.898, so the resize to 192³ costs nothing once the grids are shared.

### What changed

| File | Change |
|---|---|
| `datasets/custom_dataset.py` | new `AlignToSharedGridd` + `align_shared_grid` / `align_reference` / `align_margin_mm`; records `grid_origin_xyz` / `grid_spacing_xyz`; `landmarks_space` for LPS→RAS; `InterpolateToSized` align_corners fix |
| `configs/trusted_manifest_split9010.json` | manifest over the co-registered volumes, 53 train / 6 val, landmarks on the 6 val cases |
| `scripts/generate_trusted_manifest_split9010.py` | regenerates it (`--local <dir>` for a local subset) |
| `configs/trusted_localnet3d_aligned.yaml` | LocalNet3D on that manifest with alignment on |
| `train_modal_trusted.py` | Modal runner: `align_check`, `geometry`, `preview`, `smoke`, `train_one`, `sweep` |
| `metrics/landmarks.py` | `axis_order` parameter — the old code assumed `(z, y, x)` tensors and silently mis-applied the DDF to RAS `(x, y, z)` ones |
| `metrics/jacobian.py` | `log_jac_std` now excludes folding voxels (real SDlogJ) instead of clamping them to `eps` |
| `train.py` | logs `val/tre_mm` and `val/tre_mm_before` when a case has landmarks + grid geometry |

### Three follow-on bugs found while wiring this up

1. **`InterpolateToSized` sheared image against label.** Images resized with
   `align_corners=True` (scale `(N-1)/(M-1)`) while labels used `nearest` (scale `N/M`).
   Sub-voxel but systematic, worst at the far edge of each axis. Both now use the
   half-pixel convention.
2. **`metrics/landmarks.py` assumed `(z, y, x)` tensors.** That holds for
   `datasets/kidney.py` (SimpleITK arrays) but not for `custom_dataset`, which is
   `(x, y, z)` after `Orientationd(RAS)`. Reusing it as-is would have produced
   plausible-but-wrong TRE. It now takes `axis_order`, verified against a synthetic
   constant DDF in both orders.
3. **`log_jac_std` was not SDlogJ.** It clamped folding voxels to `eps = 1e-6`, so each
   contributed `log(1e-6) = -13.8` and the metric mostly re-reported the folding fraction.
   A white-noise field with |u| = 0.36 vox and 1% folding scored **1.55** clamped vs
   **0.73** excluding folds; a *smooth* field of the same magnitude scores 0.15 either way.
   Folding voxels are now excluded, matching the definition this repo already documented.

`AlignToSharedGridd` crops the `align_reference` key (default `fixed_label`, the US kidney
mask) to its foreground bbox padded by `align_margin_mm`, then resamples every other key
onto that one grid via its affine. The ROI comes from a *single* reference for all keys, so
residual CT↔US misalignment survives — cropping each modality to its own mask would
re-centre them and inflate the "before" score, the concern raised in
[`docs/active-kidney-experiments.md`](../active-kidney-experiments.md).

### Consequences for the sweep

- Runs now start at **~0.86**, not the ~0.63 in
  [`04_trusted_localnet3d_tutorial.md`](04_trusted_localnet3d_tutorial.md). Those
  documented 0.63 → 0.78 figures match co-registered data *without* the shared-grid
  resample, so the sweep table must be filled from scratch, not compared against them.
- The manifest uses the same 53/6 split as the VoxelMorph baseline, so TRUSTED LocalNet3D
  is now directly comparable to the VoxelMorph numbers.
- Less Dice headroom above 0.86 means more risk of the field deforming heavily for small
  Dice gains — the failure that made VoxelMorph's TRE worse than its rigid init. Watch
  `val/neg_jac_ratio` and `val/ddf_abs_mean`, not just Dice.
- **GPU: cheapest tiers only.** `train_modal_trusted.py` runs `GPU = "T4"` (16 GB, the
  cheapest Modal offers) with `BIG_GPU = "L4"` (24 GB, next cheapest) for the 208³ and
  channels 12/16 sweep points. A100 and L40S are unavailable on this workspace (no payment
  method) and are rejected at *app-definition* time, which fails **every** entrypoint in
  the file, not just the big one — so do not add an A100 function "just in case".

### Does the shared-grid step work without the pre-registration?

No — and the failure is informative. `split_90_10_new` holds CT already resampled into US
space (`*_in_US_space_*`), i.e. a precomputed warp. Running the same check against the
untouched volumes on `kidney-dataset`
(`train_modal_trusted.py::align_check_raw`, 5 val cases):

| raw volumes, no pre-registration | initial Dice |
|---|---|
| `align_shared_grid: false` | 0.142 ± 0.038 |
| `align_shared_grid: true` | **0.000 ± 0.000** |

Zero is the correct answer: the US-mask ROI contains no CT kidney whatsoever, so resampling
the CT onto that grid produces an empty mask. The shared-grid resample *consumes* the
co-registration encoded in the affines — it cannot manufacture one.

Which also reframes the 0.14 that started this whole investigation. It is not "a bit of
alignment"; it is **accidental overlap** between two volumes independently squashed into the
same 192³ cube. A number that looks like partial alignment and means nothing. The 0.142 here
reproduces it on a different 5-case split, confirming it is an artefact of the squash rather
than anything about the data.

Practical consequence: the rigid pre-registration is **load-bearing**. The fix in this
document is necessary but not sufficient — both are required, in that order.

### Still open

- **TRE covers the val split only.** Landmarks exist for the 6 test/val cases, not the 53
  training cases, so `val/tre_mm` is computed over 6 cases and `train` reports nothing.
  TRE also needs `align_shared_grid: true`: without it there is no single grid both
  modalities live on, so the geometry keys are absent and `tre_mm` stays NaN (correctly —
  the registration itself is meaningless in that configuration).
- **SDlogJ values are not comparable to earlier runs.** Anything logged before the
  `log_jac_std` fix used the clamped definition and reads high; do not mix them in one
  table.
- **`train_modal.py` is missing from the working tree, and the branch copy is NOT a valid
  restore.** Constants recovered from the stale `__pycache__/train_modal.cpython-310.pyc`
  show the lost file had 17 functions — including `eval_checkpoint_modal`,
  `full_lncc_dice`, `full_lr1e2`, `train_on_modal_a10g` and `eval_convexadam_modal` — and
  already pointed at the **`split_90_10_new`** volume. The copy on
  `synthetic-data-tutorials` lacks those entrypoints and still points at the corrupted
  `split-90-10`, so restoring it would be a silent regression for the kidney track. It also
  referenced `configs/kidney_{convexadam,localnet,transmorph,unetreg}.yaml`, none of which
  are in the working tree either — so more than one file was lost. Recover from a W&B run's
  uploaded code, another machine, or rewrite; do not `git show` the branch version over it.
  Historical note: its docstring records that "176³ VoxelMorph does not fit on a 16 GB T4",
  which is why the kidney track used an A10G — that is about VoxelMorph at 176³, not
  LocalNet3D at 192³.
  `train_modal_trusted.py` is deliberately a new filename so any later restore cannot
  clobber it.
- `align_margin_mm: 6.0` (= the VoxelMorph track's 20 voxels at 0.3 mm) is untuned.

The original diagnosis follows, unchanged.

## TL;DR

The TRUSTED CT and US volumes on the Modal `kidney-dataset` volume are **raw, not
co-registered** — each case's CT and US live in different physical coordinate frames,
on different grids, with different spacing and field-of-view. The `custom_dataset`
loader does **no cross-modal alignment**: it just resizes each volume independently to
192³. So at initialization the warped CT kidney mask and the US kidney mask barely
overlap (**val_dice ≈ 0.14**), instead of the **~0.63** the plan expects. The pipeline
runs end-to-end and trains, but a sweep on this data measures registration starting from
a near-non-overlapping state, not the documented setup.

## How this surfaced

A 1-epoch Modal smoke run of the LocalNet3D baseline
(`configs/trusted_localnet3d.yaml`, 192³, lr 1e-5, lncc/dice 1.0/0.5) completed cleanly
and logged to W&B (run `vktlvjmv`):

| metric | epoch 1 (this run) | plan's documented epoch 1 |
|---|---|---|
| **val_dice** | **0.1429** | **~0.6337** |
| val_loss | 0.3844 | ~-0.2854 |
| val_ncc | 0.1978 | ~0.42 |
| ddf_abs (mean disp.) | 0.46 vox | — |

The displacement field is tiny at epoch 1 (`ddf_abs≈0.46` vox), so `val_dice` there is
essentially `Dice(CT_mask, US_mask)` **before** meaningful warping. 0.14 means the two
masks are almost non-overlapping once each is squeezed into the 192³ cube.

## Root-cause evidence (case 200R)

Headers pulled directly from the Modal volume
(`Kidney_Dataset/TRUSTED/{CT_masks/200R_seg, US_masks/200R_maskUS}.nii.gz`):

| property | CT mask (`200R_seg`) | US mask (`200R_maskUS`) |
|---|---|---|
| shape (vox) | 116 × 113 × 106 | 1024 × 768 × 822 |
| spacing (mm) | 0.914 × 0.914 × 1.5 | 0.3 × 0.3 × 0.3 |
| field of view (mm) | ~106 × 103 × 159 | ~307 × 230 × 247 |
| world origin (mm) | (127.8, 177.7, 1003.0) | (0.0, 0.0, 0.0) |
| **mask centroid, world (mm)** | **(76.5, 125.4, 1084.4)** | **(−125.8, −132.8, 119.5)** |

The kidney centroids are **~1000 mm apart in world Z** and hundreds of mm apart in X/Y —
the CT is in scanner/table coordinates (z≈1 m), the US in its own probe frame (origin 0).
They are not in a shared space, the grids differ by ~10× in voxel count, spacing differs
by ~3–5×, and the US FOV is ~3× larger (the kidney fills a small, off-center part of the
US volume while it fills most of the cropped CT volume).

### Why independent resize can't fix this

The `custom_dataset` "multigradicon" transform pipeline
(`datasets/custom_dataset.py::_build_transforms`) is:

1. `LoadImaged` → `EnsureChannelFirstd`
2. `Orientationd(RAS)` — reorders axes only; **does not** resample to a common grid
3. `Spacingd` — **skipped** because the config sets `spacing: null`
4. `InterpolateToSized` — `F.interpolate` each volume **independently** to `[192,192,192]`
5. `ModalityAwareIntensityNormd`

There is no `Spacingd` to a shared pixdim, no crop to a common ROI/FOV, and no rigid or
affine registration. Step 4 stretches whatever FOV each volume has onto the same cube, so
a CT cropped tight around the kidney and a US covering 3× the FOV end up with the kidney at
different normalized positions and scales → low overlap. The loader **assumes its inputs
are already spatially aligned**.

## Why the documented 0.63 → 0.78 differs

The plan's evidence came from a config pointing at `configs/trusted_manifest.json`, whose
paths are `C:\Users\gabridal\Documents\TRUSTED_reg\...` — the data owner's Windows machine,
which never resolved on Modal. That run must have used a **pre-aligned / co-registered**
version of TRUSTED (both modalities resampled to a shared FOV before loading), which starts
at ~0.6 Dice. The flat volumes under `kidney-dataset/Kidney_Dataset/TRUSTED/` are the raw,
un-co-registered images, so they start at ~0.14. This is a **data-preparation gap**, not a
code bug in the model/loss/training loop (all verified working).

## Impact on the sweep

- A 9-run × 30-epoch sweep on this data is still an internally-consistent *relative*
  comparison, but it **will not reproduce the documented absolute numbers** and the
  "LocalNet3D clearly beats GlobalNet3D" conclusion may look different from a 0.14 start.
- Registering across ~1 m of world offset via a dense DDF is a much harder (and less
  meaningful) problem than refining an already-rigid-aligned pair — results would be hard
  to interpret and likely not paper-usable.
- Estimated cost avoided by pausing: ~15–25 GPU-hours (~$20) on a workspace that already
  hit its spend limit once today.

## Options to fix (recommended order)

1. **Get the co-registered TRUSTED data used for the documented 0.63 run.** Ask the data
   owner (Gabriella) for the pre-aligned `TRUSTED_reg` volumes (or the exact preprocessing
   that produced them) and upload that to the Modal volume. Cleanest and reproduces the
   documented baseline.

2. **Resample both modalities onto one shared world grid before training** — the approach
   already used on the kidney track (`datasets/kidney.py` resamples CT and US onto a single
   shared US-mask world grid with identity direction; see
   [`docs/active-kidney-experiments.md`](../active-kidney-experiments.md)). Port that
   resampling into a preprocessing step / new dataset transform so `custom_dataset` receives
   aligned inputs. Requires the source affines to be correct (they look plausible here:
   distinct but valid world coordinates).

3. **Rigid pre-registration per case** (e.g. mask-centroid alignment + isotropic resample to
   a common FOV) as a preprocessing pass, then feed the aligned pairs to the existing loader.
   A minimal version — resample to common spacing + crop to the US-mask bbox around the
   kidney — would already lift the starting Dice substantially.

Option 3 with just a shared-grid resample (via `Spacingd` + a common ROI crop) is the
smallest change; option 1 is the most faithful to the documented results.

## Verified working (not part of the issue)

- Modal pipeline (`train_modal.py`): image build, `kidney-dataset` volume mount, checkpoint
  persistence to `/data/checkpoints/trusted/<run>/`, detached runs.
- W&B routing: logs to `anikethvij-personal` (username `anikethvij`); no `amrita-medicalai`
  access, and the image pins `WANDB_ENTITY=anikethvij-personal`.
- Manifest `configs/trusted_manifest_modal.json`: all 48 train / 5 val cases resolve
  (note CT masks are named `<id>_seg.nii.gz`, not `<id>_maskCT.nii.gz`).
- Fixed a pre-existing `IndentationError` in `datasets/deepreg_synthetic.py` that was
  breaking the entire `datasets` package import.

## Reproduce

```bash
# smoke run that produced val_dice=0.14
modal profile activate aniketh            # workspace anikethvij464
modal run train_modal.py::smoke           # W&B run under anikethvij-personal/USRegistrator

# inspect a case's geometry
modal volume get kidney-dataset Kidney_Dataset/TRUSTED/CT_masks/200R_seg.nii.gz .
modal volume get kidney-dataset Kidney_Dataset/TRUSTED/US_masks/200R_maskUS.nii.gz .
python -c "import nibabel as nib; [print(f, nib.load(f).shape, nib.load(f).header.get_zooms(), nib.load(f).affine[:3,3]) for f in ('200R_seg.nii.gz','200R_maskUS.nii.gz')]"
```
