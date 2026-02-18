# DeepReg Pipeline — Setup & Configuration

This guide walks you through configuring and preparing the DeepReg-style DVF-supervised training pipeline.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [The DeepReg Config File](#2-the-deepreg-config-file)
3. [Dataset Configuration Deep Dive](#3-dataset-configuration-deep-dive)
4. [Loss Configuration](#4-loss-configuration)
5. [Choosing `max_disp` and `cp_spacing`](#5-choosing-max_disp-and-cp_spacing)
6. [Creating Your Own Config](#6-creating-your-own-config)

---

## 1. Prerequisites

Before using the DeepReg pipeline, make sure you've completed the basic setup:

- ✅ Environment set up ([Getting Started](../synthetic-training/01_getting_started.md))
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ CUDA-capable GPU (recommended but not required)

No additional dependencies are needed beyond the standard USRegistrator setup. The DeepReg-style pipeline uses the same `train.py` entry point.

---

## 2. The DeepReg Config File

The pre-built config is at `configs/deepreg_synth.yaml`:

```yaml
project: "monai-3d-registration"
image_size: [64, 64, 64]

# ---- Dataset: DeepReg-style with ground-truth DVFs ----
train_dataset:
  name: deepreg_synthetic          # ← Key difference from basic config
  image_size: [64, 64, 64]
  num_samples: 4000
  max_disp: 0.2                    # Maximum displacement magnitude
  cp_spacing: 8                    # Control point grid spacing
  noise_std: 0.03
  smooth: true
  seed: 123

val_dataset:
  name: deepreg_synthetic          # ← Same dataset type for validation
  image_size: [64, 64, 64]
  num_samples: 200
  max_disp: 0.2
  cp_spacing: 8
  noise_std: 0.03
  smooth: true
  seed: 456                        # Different seed!

# ---- Model ----
model:
  name: globalnet3d
  num_channel_initial: 16
  depth: 3
  warp_mode: bilinear
  warp_padding_mode: border

# ---- Loss: LNCC + DVF supervision ----
loss:
  name: lncc_dvf                   # ← DVF-aware loss
  kernel_size: 9
  smooth_nr: 1e-4
  smooth_dr: 1e-2
  image_weight: 1.0                # Weight for LNCC component
  dvf_weight: 0.1                  # Weight for DVF MSE component

# ---- Optimizer ----
optimizer:
  name: Adam
  lr: 1e-4
  weight_decay: 1e-5

# ---- Training loop ----
training:
  epochs: 50
  batch_size: 2
  num_workers: 4
  amp: true
  val_every: 1
  seed: 42

# ---- Logging ----
wandb:
  enabled: true
  project: "usregistrator"
  run_name: "deepreg_synth"
  offline: true                    # Set to false for online logging
```

### Key Differences from Basic Config

| Setting | Basic Config | DeepReg Config | Why |
|---------|-------------|---------------|-----|
| `train_dataset.name` | `synthetic_ellipsoids` | `deepreg_synthetic` | Enables DVF generation |
| `loss.name` | `lncc` | `lncc_dvf` | Uses DVF supervision component |
| `loss.dvf_weight` | — | `0.1` | Controls DVF supervision strength |
| `loss.image_weight` | — | `1.0` | Controls image similarity strength |
| Dataset has `max_disp` | No | Yes | Controls deformation magnitude |
| Dataset has `cp_spacing` | No | Yes | Controls DVF smoothness |

---

## 3. Dataset Configuration Deep Dive

### How Samples Are Generated

For each sample, the `DeepRegLikeDVFSyntheticGenerator`:

1. **Creates a base ellipsoid volume** → this becomes the **fixed image**
2. **Generates a random smooth DVF**:
   - Samples a low-resolution Gaussian field on a coarse grid of size `(D/cp_spacing, H/cp_spacing, W/cp_spacing)`
   - Scales by a random magnitude sampled from `U(0, max_disp)`
   - Upsamples to full resolution via trilinear interpolation
3. **Warps the fixed image** using the DVF → this becomes the **moving image**
4. Returns all three: `{moving, fixed, dvf}`

### Parameter Effects

#### `max_disp` (Maximum Displacement)

Controls how far voxels can be displaced. This value is in **normalized coordinates** (range [-1, 1]):

| Value | Physical Meaning | Use Case |
|-------|-----------------|----------|
| `0.05` | Very small deformations (~1.5 voxels at 64³) | Fine-grained alignment |
| `0.1` | Small deformations (~3 voxels at 64³) | Conservative training |
| `0.2` | Medium deformations (~6 voxels at 64³) | **Recommended default** |
| `0.4` | Large deformations (~13 voxels at 64³) | Aggressive deformations |

> **Important**: The model's `max_disp` in `models/globalnet3d.py` (the `tanh` scaling) should match the dataset's `max_disp`. Both default to `0.2`.

#### `cp_spacing` (Control Point Spacing)

Controls the **smoothness** of generated DVFs:

| Value | Coarse Grid Size (at 64³) | Effect |
|-------|--------------------------|--------|
| `4` | 16 × 16 × 16 | Higher frequency, more detailed deformations |
| `8` | 8 × 8 × 8 | **Recommended** — balanced smoothness |
| `16` | 4 × 4 × 4 | Very smooth, global-like deformations |
| `32` | 2 × 2 × 2 | Nearly affine deformations |

Smaller `cp_spacing` → more control points → higher-frequency deformations.

---

## 4. Loss Configuration

The `lncc_dvf` loss combines two objectives:

```
total_loss = image_weight × LNCC(warped, fixed) + dvf_weight × MSE(ddf, gt_dvf)
```

### Tuning the Weights

| `image_weight` | `dvf_weight` | Regime | Notes |
|----------------|-------------|--------|-------|
| 1.0 | 0.0 | Image-only | Equivalent to basic `lncc` loss |
| 1.0 | 0.1 | **Recommended** | DVF regularization without overpowering |
| 1.0 | 1.0 | Balanced | Equal weight — may dominate early epochs |
| 0.0 | 1.0 | DVF-only | Only supervise on flow, no image similarity |

**Guidelines**:

- Start with `dvf_weight: 0.1` — this provides light DVF supervision while letting the image loss drive alignment
- If EPE is high but the warped images look good, increase `dvf_weight`
- If EPE is low but image quality is poor, decrease `dvf_weight`
- The LNCC loss typically has magnitude ≈ –0.5 to –1.0, while DVF MSE can be ≈ 0.001–0.01, so the weight ratio accounts for this scale difference

### Using a Simpler Loss

You can also use the basic `lncc` or `mse` loss with the `deepreg_synthetic` dataset. The training loop will simply ignore the GT DVF:

```yaml
# Works fine — GT DVF is available but unused by the loss
loss:
  name: lncc
  kernel_size: 9
```

In this case, the dataset still provides `dvf` in each batch, and EPE will still be computed during validation, but the loss function only uses image similarity.

---

## 5. Choosing `max_disp` and `cp_spacing`

These two parameters together define the "difficulty" of the registration task.

### General Guideline

```
Difficulty = max_disp / cp_spacing_effect
```

| Difficulty Level | `max_disp` | `cp_spacing` | Description |
|-----------------|-----------|-------------|-------------|
| Easy | 0.05 | 16 | Small, smooth deformations |
| Medium | 0.2 | 8 | **Default** — realistic range |
| Hard | 0.3 | 4 | Large, high-frequency deformations |
| Very Hard | 0.4 | 4 | Extreme — may need larger model |

### Matching Model Capacity

- **GlobalNet3D** (`depth=3, channels=16`) handles medium difficulty well
- For hard tasks, increase `depth` or `num_channel_initial`
- For very hard tasks, consider `localnet3d` or `unetreg3d` which can capture finer details

---

## 6. Creating Your Own Config

```bash
# Start from the template
cp configs/deepreg_synth.yaml configs/my_deepreg_experiment.yaml
```

Edit the key sections:

```yaml
# Example: Large deformations with stronger DVF supervision
train_dataset:
  name: deepreg_synthetic
  image_size: [64, 64, 64]
  num_samples: 8000          # More training samples
  max_disp: 0.3              # Larger deformations
  cp_spacing: 4              # Higher frequency

loss:
  name: lncc_dvf
  kernel_size: 9
  image_weight: 1.0
  dvf_weight: 0.5            # Stronger DVF supervision for harder task

model:
  name: globalnet3d
  num_channel_initial: 32    # More capacity
  depth: 4                   # Deeper network

training:
  epochs: 100                # More epochs for harder task
  batch_size: 2
```

---

## Next Steps

- **[Running & Evaluating →](03_running_and_evaluating.md)** — Launch training and interpret the results
- **[DeepReg Overview ←](01_deepreg_overview.md)** — Understanding the theory
