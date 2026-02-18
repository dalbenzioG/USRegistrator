# DeepReg Pipeline — Running & Evaluating

This guide covers launching the DeepReg-style training pipeline, monitoring progress, interpreting metrics, and troubleshooting common issues.

---

## Table of Contents

1. [Launching Training](#1-launching-training)
2. [Understanding the Console Output](#2-understanding-the-console-output)
3. [Monitoring with Weights & Biases](#3-monitoring-with-weights--biases)
4. [Interpreting DeepReg-Specific Metrics](#4-interpreting-deepreg-specific-metrics)
5. [Comparing to Unsupervised Training](#5-comparing-to-unsupervised-training)
6. [Troubleshooting](#6-troubleshooting)
7. [Advanced: Multi-Stage Training](#7-advanced-multi-stage-training)

---

## 1. Launching Training

### Quick Start

```bash
python train.py --config configs/deepreg_synth.yaml
```

### Verifying the Pipeline

Before a full training run, you can verify the dataset produces valid samples:

```python
# Quick sanity check (run interactively)
python -c "
from datasets import build_dataset

cfg = {
    'name': 'deepreg_synthetic',
    'image_size': [64, 64, 64],
    'num_samples': 10,
    'max_disp': 0.2,
    'cp_spacing': 8,
    'noise_std': 0.03,
    'smooth': True,
    'seed': 123,
}

ds = build_dataset(cfg, split='train')
sample = ds[0]

print('Keys:', list(sample.keys()))
print('Moving shape:', sample['moving'].shape)
print('Fixed shape:', sample['fixed'].shape)
print('DVF shape:', sample['dvf'].shape)
print('DVF range:', sample['dvf'].min().item(), 'to', sample['dvf'].max().item())
"
```

**Expected output:**

```
Keys: ['moving', 'fixed', 'dvf']
Moving shape: torch.Size([1, 64, 64, 64])
Fixed shape: torch.Size([1, 64, 64, 64])
DVF shape: torch.Size([3, 64, 64, 64])
DVF range: -0.1842 to 0.1756
```

---

## 2. Understanding the Console Output

The training loop prints detailed information each epoch:

```
---- Dataset sanity check ----
moving: 0.0 0.9876
fixed: 0.0 0.9812
any NaN: False False
any Inf: False False
--------------------------------
Device: cuda
AMP: True
Train samples: 4000, Val samples: 200
[Epoch 001/050] train_loss = -0.1523, val_loss = -0.1834, val_ncc = 0.8201, val_epe = 0.0312, val_grad_l2 = 0.0018
[Epoch 002/050] train_loss = -0.2156, val_loss = -0.2489, val_ncc = 0.8567, val_epe = 0.0245, val_grad_l2 = 0.0015
[Epoch 010/050] train_loss = -0.5234, val_loss = -0.5567, val_ncc = 0.9234, val_epe = 0.0098, val_grad_l2 = 0.0008
...
```

### What Each Column Means

| Column | Description | Good Trend |
|--------|-------------|-----------|
| `train_loss` | Combined LNCC + DVF loss | ↓ decreasing (more negative) |
| `val_loss` | Validation combined loss | ↓ decreasing (tracking train) |
| `val_ncc` | Global NCC on warped vs fixed | ↑ toward 1.0 |
| `val_epe` | **Endpoint Error** (pred DDF vs GT DVF) | ↓ toward 0.0 |
| `val_grad_l2` | DDF smoothness | ↓ or stable |

### The `val_epe` Column

This is the **key DeepReg-specific metric**. It measures the average Euclidean distance between the model's predicted displacement and the ground-truth displacement at every voxel:

```
EPE = mean(||ddf_pred - dvf_gt||₂)
```

Typical progression:

| Epoch Range | EPE | Interpretation |
|------------|-----|----------------|
| 1–5 | 0.03–0.05 | Model barely learning |
| 10–20 | 0.01–0.02 | Significant improvement |
| 30–50 | 0.005–0.01 | Good convergence |
| 50+ | < 0.005 | Excellent — near-perfect flow prediction |

> The EPE scale depends on your `max_disp` setting. With `max_disp=0.2`, initial random predictions give EPE ≈ `max_disp / √3 ≈ 0.115`.

---

## 3. Monitoring with Weights & Biases

When W&B is enabled, additional visualizations are logged:

### Logged Metrics

- `train/loss` — training loss per epoch
- `val/loss` — validation loss per epoch
- `val/ncc` — Normalized Cross-Correlation
- `val/mse` — Mean Squared Error
- `val/mae` — Mean Absolute Error
- `val/epe` — Endpoint Error (**most important for DeepReg**)
- `val/grad_l2` — Deformation smoothness
- `val/slices` — visual comparison of moving, fixed, warped (mid-axial slice)

### Recommended W&B Panel Layout

Create a custom dashboard with:

1. **Loss panel**: `train/loss` and `val/loss` overlaid
2. **EPE panel**: `val/epe` — the primary convergence indicator
3. **NCC panel**: `val/ncc` — image quality check
4. **Smoothness panel**: `val/grad_l2` — deformation quality
5. **Image panel**: `val/slices` — visual sanity check

### Offline Mode

If you're running on a machine without internet:

```yaml
wandb:
  enabled: true
  offline: true    # Logs saved locally in wandb/ directory
```

Sync later with:

```bash
wandb sync wandb/offline-run-*
```

---

## 4. Interpreting DeepReg-Specific Metrics

### EPE (Endpoint Error)

The EPE measures flow prediction accuracy. Monitor this alongside the image-based metrics:

| EPE | NCC | Interpretation |
|-----|-----|----------------|
| ↓ Low | ↑ High | ✅ Model is learning both flow and appearance |
| ↓ Low | ↓ Low | ⚠️ Flow is accurate but warped images look poor — possible numerical issue |
| ↑ High | ↑ High | ⚠️ Images look aligned but flow is wrong — model found a shortcut |
| ↑ High | ↓ Low | ❌ Not converging — check config |

### DVF Loss Component

The total loss with `lncc_dvf` is:

```
total = image_weight × LNCC(warped, fixed) + dvf_weight × MSE(ddf, gt_dvf)
```

If the LNCC component is ≈ –0.8 and DVF MSE is ≈ 0.005:

```
total = 1.0 × (–0.8) + 0.1 × 0.005 = –0.7995
```

The DVF component is small but provides consistent gradient signal on the deformation field itself.

### Grad L2 (Deformation Smoothness)

With DeepReg-style training, `grad_l2` should be naturally low because:
- The GT DVFs are smooth by construction (trilinear upsampling)
- DVF supervision encourages smooth predictions

If `grad_l2` increases over training, the model may be overfitting or the `dvf_weight` is too low.

---

## 5. Comparing to Unsupervised Training

To see the benefit of DVF supervision, run both pipelines and compare:

### Experiment Setup

```bash
# Run 1: Basic unsupervised
python train.py --config configs/config_template.yaml

# Run 2: DeepReg DVF-supervised
python train.py --config configs/deepreg_synth.yaml
```

### Expected Differences

| Metric | Unsupervised | DVF-Supervised |
|--------|-------------|---------------|
| NCC convergence speed | Slower | Faster |
| EPE | Not meaningful (0.0) | Trackable and decreasing |
| Grad L2 | May be higher | Lower (smoother DVFs) |
| Final NCC | ≈ 0.90–0.95 | ≈ 0.93–0.98 |

### When Unsupervised Wins

- Real paired images without GT DVFs
- When the synthetic DVF distribution doesn't match real deformations
- When you want the model to discover its own optimal deformation strategy

### When DVF-Supervised Wins

- Pre-training for fine-tuning on limited real data
- When you need quantitative DVF evaluation (EPE)
- When training stability is important (stronger gradient signal)

---

## 6. Troubleshooting

### Issue: EPE Not Decreasing

**Possible causes:**

1. **`dvf_weight` too low** — the model ignores DVF supervision
   - Fix: Increase `dvf_weight` from 0.1 to 0.5 or 1.0
   
2. **`max_disp` mismatch** — model's tanh scaling doesn't match dataset
   - Verify: model uses `max_disp = 0.2` (in `models/globalnet3d.py`) matching dataset's `max_disp: 0.2`
   
3. **Learning rate too low** — model converges too slowly
   - Fix: Try `lr: 5e-4` (but watch for instability)

### Issue: NaN in Loss

**Possible causes:**

1. **AMP with LNCC** — float16 precision issues
   - Fix: Ensure `smooth_dr: 1e-2` (minimum safe value)
   
2. **Large displacements** — warping creates out-of-bounds values
   - Fix: Reduce `max_disp` or ensure `warp_padding_mode: border`

3. **Learning rate too high**
   - Fix: Reduce `lr` to `1e-5`

### Issue: EPE Oscillating

**Possible causes:**

1. **Batch size too small** — high variance in gradient estimates
   - Fix: Increase `batch_size` or reduce `lr`
   
2. **`dvf_weight` too high** — DVF loss dominates and fights image loss
   - Fix: Reduce `dvf_weight`

### Issue: Warning Messages "NaN/Inf detected"

The training loop automatically skips batches with NaN/Inf. If you see many of these:

1. Check your CUDA drivers are up to date
2. Try disabling AMP: `training.amp: false`
3. Reduce learning rate
4. Check that `smooth_dr` is at least `1e-2`

---

## 7. Advanced: Multi-Stage Training

A powerful strategy is to combine both pipelines in stages:

### Stage 1: Pre-train with DVF Supervision

```yaml
# configs/stage1_deepreg.yaml
train_dataset:
  name: deepreg_synthetic
  num_samples: 8000
  max_disp: 0.2

loss:
  name: lncc_dvf
  dvf_weight: 0.5          # Strong DVF supervision

training:
  epochs: 30
```

```bash
python train.py --config configs/stage1_deepreg.yaml
```

### Stage 2: Fine-tune with Image-Only Loss

After stage 1, load the pretrained weights and fine-tune with just LNCC:

```yaml
# configs/stage2_finetune.yaml
train_dataset:
  name: synthetic_ellipsoids   # or your real dataset
  num_samples: 4000

loss:
  name: lncc                   # Image-only loss
  kernel_size: 9

training:
  epochs: 20
  # TODO: Add checkpoint loading support
```

> **Note**: Checkpoint saving/loading is not yet implemented in the current version of `train.py`. You would need to add `torch.save()` and `torch.load()` calls. This is a good area for future development.

### Why Multi-Stage Works

1. **Stage 1** teaches the model what a "good" deformation looks like (smooth, bounded, physically plausible)
2. **Stage 2** fine-tunes for image appearance matching without the constraint of matching a synthetic DVF distribution

---

## Summary Cheat Sheet

```bash
# Quick start
python train.py --config configs/deepreg_synth.yaml

# Watch for these metrics:
#   val_epe ↓   → model learning correct deformations
#   val_ncc ↑   → warped images aligning well
#   grad_l2 ↓   → smooth deformations

# Key config knobs:
#   max_disp     → deformation magnitude (match model!)
#   cp_spacing   → deformation smoothness
#   dvf_weight   → DVF supervision strength
#   image_weight → image similarity strength
```

---

## See Also

- **[DeepReg Overview ←](01_deepreg_overview.md)** — Theory and architecture
- **[Setup & Configuration ←](02_setup_and_configuration.md)** — Config options
- **[Synthetic Training Tutorial](../synthetic-training/02_training_tutorial.md)** — Basic unsupervised pipeline
- **[Configuration Reference](../synthetic-training/06_configuration_reference.md)** — All config options
