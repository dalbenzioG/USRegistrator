# TRUSTED CT→US kidney registration with LocalNet3D

A runnable worked example: warp a pre-operative **CT** kidney onto an intra-operative
**ultrasound** volume with a dense displacement field, and — the part that actually decides
whether it works — preprocess the two modalities so they are still aligned by the time the
network sees them.

Files here:

| File | What it is |
|---|---|
| `config.yaml` | A plain USRegistrator config. Nothing tutorial-specific about the format. |
| `__init__.py` | The callable: resolves the config against your data, then runs it. |
| `manifest.generated.json` | Written for you when you pass `--data-root` (gitignored). |

## Three ways to run it

**1. Straight config** — if your manifest paths are already correct:

```bash
python main.py --config tutorials/trusted_ct_us_localnet/config.yaml
```

**2. The tutorial hook** — resolves data paths for you by scanning a data root:

```bash
python main.py --list-tutorials
python main.py --tutorial trusted_ct_us_localnet --data-root /path/to/split_90_10_new

# cheap sanity check: report initial mask Dice, don't train
python main.py --tutorial trusted_ct_us_localnet --data-root /path/to/data --check-only

# short run, smaller cube
python main.py --tutorial trusted_ct_us_localnet --data-root /path/to/data \
    --epochs 5 --image-size 176

# reproduce the misaligned baseline for comparison
python main.py --tutorial trusted_ct_us_localnet --data-root /path/to/data \
    --check-only --no-align
```

**3. From Python** — the same entry point, for notebooks or scripts:

```python
from tutorials import run_tutorial

run_tutorial("trusted_ct_us_localnet", data_root="/path/to/split_90_10_new", epochs=5)
```

Or compose it yourself when you want to inspect or patch the config first:

```python
from tutorials.trusted_ct_us_localnet import build_config, initial_dice
from train import run_training

cfg = build_config(data_root="/path/to/split_90_10_new", epochs=5)
print(initial_dice(cfg))            # check before spending GPU time
cfg["loss"]["params"]["dice_weight"] = 1.0
run_training(cfg=cfg)
```

## Expected data layout

`--data-root` must contain `train/` and `test/` (the test split becomes the manifest's
`val` split), in the co-registered TRUSTED layout:

```
<data-root>/
  train/
    CT_images/<id>_imgCT_in_US_space_crop*.nii.gz     -> moving
    CT_masks/<id>_seg_in_US_space_crop*.nii.gz        -> moving_label
    US_images/<id>_imgUS_crop.nii.gz                  -> fixed
    US_masks/<id>_maskUS_crop.nii.gz                  -> fixed_label
  test/
    ... same, plus optionally:
    landmarks_lps/<id>_ldkCT_aligned_lps.txt          -> moving_landmarks
    landmarks_lps/<id>_ldkUS_lps.txt                  -> fixed_landmarks
```

Cases missing any of the four required files are skipped with a message. Landmarks are
optional; when present you also get TRE in millimetres (`val/tre_mm`).

Different filenames? Edit the `LAYOUT` regexes at the top of `__init__.py`, or skip the
scanner entirely and pass `--manifest` with your own JSON:

```json
{
  "train": [
    {
      "case_id": "206L",
      "moving_modality": "CT", "fixed_modality": "US",
      "moving": "/abs/path/206L_imgCT.nii.gz",
      "moving_label": "/abs/path/206L_maskCT.nii.gz",
      "fixed": "/abs/path/206L_imgUS.nii.gz",
      "fixed_label": "/abs/path/206L_maskUS.nii.gz"
    }
  ],
  "val": []
}
```

## The one thing to understand

CT and US in this dataset are **already co-registered** — same world coordinate system, the
kidney at the same physical place in both. What differs is the *sampling*: different array
shapes, different origins, and above all **different fields of view** (for case 206L, CT
covers ~173×173×161 mm on a 576×576×535 grid; US covers ~216×168×245 mm on 720×560×816).

A resize to a common cube therefore applies a **different scale factor per modality** — it
magnifies the tightly-cropped CT relative to the wide US and shifts it. Two volumes that
agreed in world millimetres no longer agree in voxel indices. Nothing errors, the shapes are
all correct, and the loss still goes down: the model just trains on misaligned pairs.

The three `align_*` keys in `config.yaml` fix it by putting everything on one grid *before*
the resize:

```yaml
align_shared_grid: true       # resample every volume onto one shared grid
align_reference: fixed_label  # the US kidney mask defines the ROI grid
align_margin_mm: 6.0          # how much context to keep around the organ
```

Concretely: crop the reference to its foreground bounding box padded by `align_margin_mm`
(that cropped volume *is* the target grid), then resample every other key onto it through
its affine. The resize afterwards applies one common scale factor, so alignment survives.

Measured on the real TRUSTED data, initial mask Dice with **no warp applied**:

| | val (6 cases) | train (53 cases) |
|---|---|---|
| `align_shared_grid: false` | 0.580 ± 0.101 | 0.533 ± 0.099 |
| `align_shared_grid: true` | **0.856 ± 0.033** | **0.855 ± 0.063** |

Two design notes worth carrying to other datasets:

- The ROI comes from **one** reference for all keys. Cropping each modality to *its own*
  mask would re-centre CT and US on each other, hiding the misalignment the network is
  supposed to fix and inflating the "before registration" score.
- **This step aligns *grids*, not anatomy.** It consumes the co-registration already
  encoded in the affines; it cannot create one. Measured on the *raw* TRUSTED volumes,
  where CT is still in scanner coordinates and the kidney centroids are ~1 m apart in
  world Z:

  | raw volumes, no pre-registration | initial Dice (5 cases) |
  |---|---|
  | `align_shared_grid: false` | 0.142 ± 0.038 |
  | `align_shared_grid: true` | **0.000 ± 0.000** |

  Zero is the *correct* answer: the US ROI contains no CT kidney at all, so resampling the
  CT onto it yields an empty mask. And the 0.142 is worse than useless — it is accidental
  overlap between two volumes independently squashed into the same cube, a number that
  looks like partial alignment while meaning nothing.

  So a rigid (or affine) pre-registration is **load-bearing**, not an optional extra: this
  tutorial's preprocessing is necessary but not sufficient. If `--check-only` reports ~0.0,
  your data is not co-registered; if it reports ~0.1–0.2, suspect that whatever you are
  measuring is coincidence.

**Always run `--check-only` first.** It is seconds of CPU and tells you whether a
disappointing result is your model or your data.

## Reading the metrics

| Metric | What healthy looks like |
|---|---|
| `val/dice` | climbing above the initial ~0.86. Report before *and* after, always. |
| `val/tre_mm` | below `val/tre_mm_before` (the rigid baseline). Above it means the field is Dice-overfitting — raise smoothing. |
| `val/ncc` | slow rise; stays low for CT–US. Expected, not a bug. |
| `val/neg_jac_ratio` | near zero. Rising = the field is folding, i.e. tearing tissue. |
| `val/log_jac_std` | SDlogJ over non-folding voxels. Starts high (a random-init field is noisy) and should fall as the field smooths. |
| `val/ddf_abs_mean` | non-zero but not exploding. Exactly zero = the model learned nothing. |

Dice up while folding is also up is the classic failure: the field maximises overlap by
deforming implausibly. Lower `dice_weight` or the learning rate.

## Notes

- `image_size` edges must be divisible by 16 for `extract_levels: [0,1,2,3]` — 176, 192 and
  208 work, 175 does not. `build_config` rejects bad values rather than letting you hit a
  shape-mismatch crash in a skip connection.
- `lr: 1e-5` is deliberately conservative for 192³. For a quick look, `--epochs 5` with
  `--lr 1e-4` moves faster.
- W&B is off by default. Set `wandb.enabled: true` in the config, or pass
  `wandb_enabled=True` to `run()`, once `wandb login` is configured.

## Adding your own tutorial

Create `tutorials/<name>/` with a `config.yaml` and an `__init__.py` that registers a
callable, then import it from `tutorials/__init__.py`:

```python
from ..registry import register_tutorial

@register_tutorial("my_experiment")
def run(data_root=None, epochs=None, **kwargs):
    """One-line summary — this shows up in --list-tutorials."""
    cfg = build_config(...)
    from train import run_training
    return run_training(cfg=cfg)
```

It is then available as `python main.py --tutorial my_experiment` and
`run_tutorial("my_experiment", ...)`.
