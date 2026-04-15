# Training Tutorial — Synthetic Ellipsoids

This tutorial walks you through training a 3D deformable image registration model using USRegistrator's **synthetic ellipsoid** dataset. By the end, you will understand the full pipeline: configuration → data generation → model training → evaluation.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Understanding the Synthetic Dataset](#2-understanding-the-synthetic-dataset)
3. [Configuration File Walkthrough](#3-configuration-file-walkthrough)
4. [Running Training](#4-running-training)
5. [Monitoring with Weights & Biases](#5-monitoring-with-weights--biases)
6. [Understanding the Output](#6-understanding-the-output)
7. [Experimenting & Iterating](#7-experimenting--iterating)

---

## 1. Overview

The training pipeline follows this flow:

```
┌──────────────┐     ┌───────────┐     ┌──────────┐     ┌────────────┐
│ YAML Config  │────▶│ Dataset   │────▶│  Model   │────▶│  Loss +    │
│              │     │ Generator │     │ (forward) │     │  Optimizer │
└──────────────┘     └───────────┘     └──────────┘     └────────────┘
                          │                 │                  │
                          ▼                 ▼                  ▼
                    moving + fixed     warped + DDF      backprop + step
                                            │
                                            ▼
                                     ┌────────────┐
                                     │  Metrics   │
                                     │ (NCC, MSE, │
                                     │  MAE, etc.)│
                                     └────────────┘
```

The key idea: two 3D volumes (**moving** and **fixed**) are fed into a registration network. The network predicts a **displacement field (DDF)** that warps the moving image to look like the fixed image. A loss function measures how well the warped image matches the fixed image, and the optimizer updates the network weights.

---

## 2. Understanding the Synthetic Dataset

The built-in `synthetic_ellipsoids` dataset generates 3D volumes containing **random ellipsoids** with:

- **Configurable position and size** — centers ∈ [-0.25, 0.25], radii ∈ [0.35, 0.55]
- **Smooth boundaries** — optional average-pooling to soften edges
- **Textural variation** — a smooth random field is added to guarantee non-zero local variance (important for LNCC loss)
- **Boundary enhancement** — extra texture near ellipsoid boundaries to create strong registration signals
- **Gaussian noise** — configurable noise for realism

Each sample returns:

```python
{
    "moving": torch.Tensor,  # shape (1, D, H, W)
    "fixed":  torch.Tensor,  # shape (1, D, H, W)
}
```

> **Note**: In the basic `synthetic_ellipsoids` mode, the moving and fixed images are identical (no ground-truth DVF). The model learns general registration features from the loss supervision alone. For DVF-supervised training, see the [DeepReg Pipeline Tutorial](../deepreg-pipeline/).

---

## 3. Configuration File Walkthrough

The template config lives at `configs/config_template.yaml`. Here's every section:

```yaml
# ---- Global ----
project: "monai-3d-registration"
image_size: [64, 64, 64]           # Volume resolution (D, H, W)

# ---- Experiment Tracking ----
wandb:
  enabled: True                     # Toggle W&B logging
  project: "usregistrator"          # W&B project name
  run_name: "globalnet3d"           # Run display name
  offline: False                    # Offline mode (no internet required)

# ---- Training Dataset ----
train_dataset:
  name: synthetic_ellipsoids        # Dataset type (from DATASET_REGISTRY)
  image_size: [64, 64, 64]
  num_samples: 4000                 # Number of training pairs
  noise_std: 0.03                   # Gaussian noise strength
  smooth: true                      # Smooth ellipsoid boundaries
  seed: 123                         # Reproducible seed

# ---- Validation Dataset ----
val_dataset:
  name: synthetic_ellipsoids
  image_size: [64, 64, 64]
  num_samples: 200
  noise_std: 0.03
  smooth: true
  seed: 999                         # Different seed from training!

# ---- Model ----
model:
  name: globalnet3d                 # Model type (from MODEL_REGISTRY)
  num_channel_initial: 16           # Initial feature channels
  depth: 3                          # Network depth (encoder levels)
  warp_mode: bilinear               # Image warping interpolation
  warp_padding_mode: border         # Border handling for warping

# ---- Loss Function ----
loss:
  name: lncc                        # Loss type (from LOSS_REGISTRY)
  kernel_size: 9                    # Local window size
  smooth_nr: 1e-4                   # Numerator smoothing
  smooth_dr: 1e-2                   # Denominator smoothing (NaN prevention)

# ---- Optimizer ----
optimizer:
  name: Adam                        # Adam or AdamW
  lr: 1e-4                          # Learning rate
  weight_decay: 1e-5                # L2 regularization

# ---- Training Loop ----
training:
  epochs: 50                        # Total epochs
  batch_size: 2                     # Samples per batch
  num_workers: 4                    # DataLoader worker processes
  amp: true                         # Mixed precision (CUDA only)
  val_every: 1                      # Validate every N epochs
  seed: 42                          # Global random seed
```

### Key Design Decisions

| Parameter | Why it matters |
|-----------|---------------|
| `smooth_dr: 1e-2` | Prevents division-by-zero in LNCC, especially critical with AMP (float16) |
| `num_channel_initial: 16` | Balances model capacity vs. memory. Increase for larger volumes |
| `warp_padding_mode: border` | Avoids black edges in warped images |
| `noise_std: 0.03` | Adds realism; too high will overwhelm signal |

---

## 4. Running Training

### Basic Run

```bash
python train.py --config configs/config_template.yaml
```

### Using a Custom Config

```bash
# Copy the template
cp configs/config_template.yaml configs/my_experiment.yaml

# Edit it (change epochs, lr, model, etc.)
nano configs/my_experiment.yaml

# Run
python train.py --config configs/my_experiment.yaml
```

### What Happens During Training

1. **Seed initialization** — deterministic training
2. **Dataset creation** — samples are generated on-the-fly from the ellipsoid generator
3. **Model construction** — GlobalNet3D (or whichever you configured)
4. **Epoch loop**:
   - Forward pass: model predicts DDF → warps moving image
   - Loss computation: LNCC (or MSE) between warped and fixed
   - Backward pass with gradient clipping (`max_norm=1.0`)
   - Mixed precision if AMP is enabled
5. **Validation** every `val_every` epochs
6. **Logging** to console + Weights & Biases

### Console Output

You'll see output like this each epoch:

```
Device: cuda
AMP: True
Train samples: 4000, Val samples: 200
[Epoch 001/050] train_loss = -0.2134, val_loss = -0.2456, val_ncc = 0.8723, val_epe = 0.0000, val_grad_l2 = 0.0012
[Epoch 002/050] train_loss = -0.3012, val_loss = -0.3189, val_ncc = 0.9001, val_epe = 0.0000, val_grad_l2 = 0.0009
...
```

> **Why is the loss negative?** LNCC is a *similarity* metric that is maximized. MONAI returns it as a negative value so that minimizing the loss maximizes similarity. A loss approaching –1.0 means near-perfect alignment.

---

## 5. Monitoring with Weights & Biases

If `wandb.enabled: true`, training automatically logs:

- **Train/Val loss** per epoch
- **Validation metrics**: NCC, MSE, MAE, Gradient L2, EPE
- **Slice visualizations**: mid-axial slices of moving, fixed, and warped volumes

### First-time Setup

```bash
# Login to W&B (one-time)
wandb login
# Paste your API key from https://wandb.ai/authorize
```

### Disabling W&B

Set `wandb.enabled: false` in your config, or run in offline mode (`wandb.offline: true`) to log locally without internet.

### Viewing Results

Navigate to your W&B project dashboard to see:
- Loss curves
- Metric trends
- Image comparisons across epochs

---

## 6. Understanding the Output

### Metrics Explained

| Metric | What it measures | Good values |
|--------|-----------------|------------|
| **NCC** (Normalized Cross-Correlation) | Global intensity similarity | → 1.0 |
| **MSE** (Mean Squared Error) | Pixel-wise intensity error | → 0.0 |
| **MAE** (Mean Absolute Error) | Pixel-wise absolute error | → 0.0 |
| **Grad L2** (Gradient L2 Norm) | Smoothness of the deformation field | Low ≈ smooth DVF |
| **EPE** (Endpoint Error) | Distance between predicted and ground-truth DVF | → 0.0 (requires GT DVF) |

> **Note**: EPE will be `0.0` for `synthetic_ellipsoids` because this dataset doesn't provide ground-truth DVFs. Use the `deepreg_synthetic` dataset for EPE evaluation — see the [DeepReg Pipeline Tutorial](../deepreg-pipeline/).

### Loss Interpretation for LNCC

- **≈ 0.0**: No correlation (bad)
- **≈ –0.5**: Moderate alignment
- **≈ –1.0**: Near-perfect alignment (excellent)

---

## 7. Experimenting & Iterating

Here are some experiments to try next:

### Change the Model

```yaml
model:
  name: localnet3d          # Try LocalNet instead of GlobalNet
  num_channel_initial: 16
  depth: 3
```

Or try the UNet-based model:

```yaml
model:
  name: unetreg3d
  channels: [16, 32, 64, 128, 256]
  strides: [2, 2, 2, 2]
  num_res_units: 2
```

### Switch the Loss

```yaml
loss:
  name: mse                 # Simple MSE instead of LNCC
```

### Increase Volume Resolution

```yaml
image_size: [128, 128, 128]
train_dataset:
  image_size: [128, 128, 128]
val_dataset:
  image_size: [128, 128, 128]
# ⚠️ You may also need to reduce batch_size to fit in GPU memory
training:
  batch_size: 1
```

### Hyperparameter Tuning

| Parameter | Try... | Effect |
|-----------|--------|--------|
| `lr` | 1e-3, 1e-4, 1e-5 | Faster/slower convergence |
| `depth` | 2, 3, 4 | Shallow/deeper encoder |
| `num_channel_initial` | 8, 16, 32 | Model capacity |
| `kernel_size` (LNCC) | 5, 9, 13 | Local context window |
| `batch_size` | 1, 2, 4 | Training stability vs. speed |

---

## Next Steps

- **[Creating Custom Models →](03_custom_models.md)** — Implement your own registration networks
- **[Creating Custom Losses →](04_custom_losses.md)** — Implement your own loss functions
- **[Creating Custom Metrics →](05_custom_metrics.md)** — Add new evaluation metrics
- **[Configuration Reference →](06_configuration_reference.md)** — Full config option reference
- **[DeepReg Pipeline →](../deepreg-pipeline/)** — DVF-supervised training with ground-truth deformations
