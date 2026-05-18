# Custom Dataset Training and Validation

This guide explains how to train and validate USRegistrator on your own medical registration dataset without modifying source code.

## Overview

USRegistrator now supports a registry dataset named `custom_dataset` that reads a JSON manifest file declared in YAML:

- `train_dataset.json_file`
- `val_dataset.json_file`

The JSON manifest contains top-level `train` and `val` case lists. Each case must provide:

- `moving`: path to moving image
- `fixed`: path to fixed image

The pipeline uses MONAI dictionary transforms for loading/preprocessing and reuses the existing training/validation loop in `train.py`.

## Preprocessing Style

`custom_dataset` uses a multigradicon-style preprocessing flow:

- Resizes `moving`/`fixed` with trilinear interpolation to `image_size`.
- Resizes labels/masks with nearest-neighbor interpolation.
- Applies modality-aware intensity normalization for `moving`/`fixed`:
  - CT uses fixed `ct_window`.
  - Non-CT uses `quantile_range`.

Optional fields:

- `ct_window: [low, high]` (used for CT normalization)
- `quantile_range: [q_low, q_high]` (used for non-CT normalization)
- `default_is_ct: bool` (fallback when modality metadata is absent)

## Supported Formats

For image-like fields (`moving`, `fixed`, optional labels/masks), loading is done with MONAI `LoadImaged`, so formats supported by MONAI readers are supported (including NIfTI such as `.nii` / `.nii.gz`).

Landmarks are supported through:

- inline list format in JSON (list of 3D points)
- file paths with extensions `.npy`, `.npz`, `.txt`, `.csv`

## Expected Dataset Layout

You can organize your data as needed. One common structure is:

```text
your_project_data/
├── train/
│   ├── case_001/
│   │   ├── moving.nii.gz
│   │   ├── fixed.nii.gz
│   │   ├── moving_label.nii.gz          # optional
│   │   ├── fixed_label.nii.gz           # optional
│   │   ├── moving_mask.nii.gz           # optional
│   │   ├── fixed_mask.nii.gz            # optional
│   │   ├── moving_landmarks.csv         # optional
│   │   └── fixed_landmarks.csv          # optional
│   └── ...
└── val/
    ├── case_001/
    │   ├── moving.nii.gz
    │   └── fixed.nii.gz
    └── ...
```

## JSON Manifest Schema

Create a JSON file (example: `my_manifest.json`) with this schema:

```json
{
  "train": [
    {
      "moving": "train/case_001/moving.nii.gz",
      "fixed": "train/case_001/fixed.nii.gz",
      "moving_label": "train/case_001/moving_label.nii.gz",
      "fixed_label": "train/case_001/fixed_label.nii.gz",
      "moving_mask": "train/case_001/moving_mask.nii.gz",
      "fixed_mask": "train/case_001/fixed_mask.nii.gz",
      "moving_landmarks": "train/case_001/moving_landmarks.csv",
      "fixed_landmarks": "train/case_001/fixed_landmarks.csv",
      "modality": "MRI",
      "subject_id": "001"
    }
  ],
  "val": [
    {
      "moving": "val/case_001/moving.nii.gz",
      "fixed": "val/case_001/fixed.nii.gz"
    }
  ]
}
```

Notes:

- Paths can be absolute or relative to the JSON file location.
- Extra metadata fields are allowed and kept in each sample dictionary.
- Labels/masks/landmarks are optional and are loaded only when present.

## Example YAML Configuration

Use `configs/custom_dataset_example.yaml` as a template:

```yaml
train_dataset:
  name: custom_dataset
  json_file: docs/examples/custom_dataset_manifest.json
  image_size: [96, 96, 96]
  orientation: RAS
  spacing: null
  preprocess_style: multigradicon
  ct_window: [-1000.0, 1000.0]
  quantile_range: [0.01, 0.99]
  default_is_ct: false
  normalize_intensity: true

val_dataset:
  name: custom_dataset
  json_file: docs/examples/custom_dataset_manifest.json
  image_size: [96, 96, 96]
  orientation: RAS
  spacing: null
  preprocess_style: multigradicon
  ct_window: [-1000.0, 1000.0]
  quantile_range: [0.01, 0.99]
  default_is_ct: false
  normalize_intensity: true
```

Multigradicon-style example:

```yaml
train_dataset:
  name: custom_dataset
  json_file: configs/trusted_manifest.json
  image_size: [180, 180, 180]
  orientation: RAS
  spacing: null
  preprocess_style: multigradicon
  ct_window: [-1000.0, 1000.0]
  quantile_range: [0.01, 0.99]
  default_is_ct: false
```

You can point `train_dataset.json_file` and `val_dataset.json_file` to the same file (with both `train` and `val` lists) or to different files if desired.

## Run Training and Validation

Validation is part of the training loop (`val_every` in config), so running training also runs validation.

```bash
python train.py --config configs/custom_dataset_example.yaml
```

If you want more frequent validation, set in YAML:

```yaml
training:
  val_every: 1
```

## Validation and Error Checks

The dataset loader validates:

- `json_file` exists
- JSON has top-level `train` and `val` lists
- each case has required `moving` and `fixed`
- referenced files exist for required and optional path fields
- landmarks have valid shape `(N, 3)` when loaded

Errors are raised with case index and split context to simplify debugging.

## Troubleshooting

- **`missing 'json_file'`**: add `json_file:` under both `train_dataset` and `val_dataset` when using `name: custom_dataset`.
- **`references missing file`**: fix path typos or move files; relative paths are resolved from the JSON file directory.
- **shape/collation issues**: ensure samples are consistently resampled/cropped via `image_size` and optional `spacing`.
- **EPE appears zero**: if your dataset does not provide ground-truth DVF (`dvf`), EPE is not supervised and remains zero.


