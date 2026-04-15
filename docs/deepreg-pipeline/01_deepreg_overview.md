# DeepReg Pipeline — Overview

This tutorial explains the **DeepReg-style synthetic DVF pipeline** in USRegistrator: what it is, why you'd use it, and how it differs from the basic synthetic training approach.

---

## Table of Contents

1. [What is DeepReg?](#1-what-is-deepreg)
2. [Unsupervised vs. DVF-Supervised Registration](#2-unsupervised-vs-dvf-supervised-registration)
3. [How USRegistrator Implements DeepReg-Style Training](#3-how-usregistrator-implements-deepreg-style-training)
4. [When to Use the DeepReg Pipeline](#4-when-to-use-the-deepreg-pipeline)
5. [Pipeline Architecture](#5-pipeline-architecture)

---

## 1. What is DeepReg?

[DeepReg](https://github.com/DeepRegNet/DeepReg) is an open-source framework for deep learning–based medical image registration. One of its key contributions is the idea of **training registration networks on synthetically generated deformations** — rather than relying only on image similarity losses, you can also supervise the model directly on **ground-truth displacement vector fields (DVFs)**.

USRegistrator implements a DeepReg-*inspired* approach: the same training pipeline (`train.py`), but with a different **dataset generator** that produces:

```python
{
    "moving": (1, D, H, W),  # warped anatomy
    "fixed":  (1, D, H, W),  # original anatomy
    "dvf":    (3, D, H, W),  # ground-truth displacement field
}
```

---

## 2. Unsupervised vs. DVF-Supervised Registration

### Unsupervised (Basic `synthetic_ellipsoids`)

```
                    ┌─────────────┐
  moving ──────────▶│             │── warped ──▶ Loss(warped, fixed)
                    │   Model     │
  fixed  ──────────▶│             │── ddf
                    └─────────────┘
```

- The loss only looks at **image similarity** (e.g., LNCC)
- No ground truth for the deformation itself
- Model learns to align images, but the predicted DVF has no direct supervision
- **EPE metric is not meaningful** (no GT DVF to compare against)

### DVF-Supervised (DeepReg-style `deepreg_synthetic`)

```
                    ┌─────────────┐
  moving ──────────▶│             │── warped ──▶ Image Loss(warped, fixed)
                    │   Model     │                       │
  fixed  ──────────▶│             │── ddf ─────▶ DVF Loss(ddf, gt_dvf)
                    └─────────────┘                       │
                                                          ▼
  gt_dvf (from dataset) ─────────────────────────▶  Total Loss
```

- The loss has **two components**:
  1. **Image similarity** — same as before (LNCC or MSE on warped vs. fixed)
  2. **DVF supervision** — MSE between predicted DDF and ground-truth DVF
- Model receives direct gradient signal on the displacement field
- **EPE metric becomes meaningful** — you can track how close the predicted flow is to ground truth

---

## 3. How USRegistrator Implements DeepReg-Style Training

The implementation consists of three parts:

### 3.1. Synthetic DVF Dataset (`datasets/deepreg_synthetic.py`)

The `DeepRegLikeDVFSyntheticGenerator` class:

1. **Generates a fixed image** — reuses the ellipsoid generator for anatomy
2. **Creates a random smooth DVF** — Gaussian random field on a coarse grid, scaled by `U(0, max_disp)`, then upsampled via trilinear interpolation
3. **Warps the fixed image** to produce the moving image using `F.grid_sample()`

The DVF generation process:

```
Coarse grid         Upsample           Scale
(D/8 × H/8 × W/8)  ──────────▶  (D × H × W)  ×  U(0, max_disp)
    │                                                      │
    ▼                                                      ▼
Gaussian noise                                    Smooth dense DVF
N(0, 1)                                           in [-max_disp, +max_disp]
```

The resulting DVF is smooth because:
- It starts on a coarse grid (e.g., 8×8×8 for a 64³ volume)
- Trilinear upsampling produces C⁰-continuous fields
- Low-frequency deformations are realistic for soft tissue

### 3.2. Combined Loss (`losses/lncc_dvf.py`)

`LNCCWithDVFSupervision` computes:

```
total_loss = image_weight × LNCC(warped, fixed) + dvf_weight × MSE(ddf, gt_dvf)
```

The `dvf_weight` controls how strongly the model is supervised on the displacement field itself (vs. just image appearance).

### 3.3. Training Loop (`train.py`)

The training loop automatically detects whether ground-truth DVFs are available:

```python
gt_dvf = batch.get("dvf", None)
if gt_dvf is not None:
    gt_dvf = gt_dvf.to(device, non_blocking=True)

# Loss call tries 4-arg signature first, then falls back to 2-arg
try:
    loss = loss_fn(warped, fixed, ddf, gt_dvf)
except TypeError:
    loss = loss_fn(warped, fixed)
```

This means the same `train.py` handles both pipelines transparently.

---

## 4. When to Use the DeepReg Pipeline

| Scenario | Recommended Pipeline |
|----------|---------------------|
| Initial prototyping | Basic `synthetic_ellipsoids` |
| You want direct DVF supervision | ✅ `deepreg_synthetic` |
| You need EPE tracking during training | ✅ `deepreg_synthetic` |
| You have real paired images without GT DVF | Basic `synthetic_ellipsoids` |
| Pre-training before fine-tuning on real data | ✅ `deepreg_synthetic` |
| You want to validate deformation quality | ✅ `deepreg_synthetic` |

### Key Benefits

1. **Faster convergence** — direct DVF loss provides stronger gradients
2. **Better regularization** — the model learns smooth, realistic deformations
3. **Quantitative evaluation** — EPE gives a direct measure of deformation accuracy
4. **Pre-training** — DVF supervision on synthetic data → fine-tune on real images with image-only loss

---

## 5. Pipeline Architecture

```
configs/deepreg_synth.yaml
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  train.py                                                   │
│                                                              │
│  ┌──────────────────────┐    ┌────────────────────────────┐ │
│  │ DeepRegLikeDVF       │    │ GlobalNet3D / LocalNet3D   │ │
│  │ SyntheticGenerator   │───▶│ / UNetReg3D                │ │
│  │                      │    │                             │ │
│  │ Returns:             │    │ Returns:                    │ │
│  │  • moving (1,D,H,W)  │    │  • warped (1,D,H,W)        │ │
│  │  • fixed  (1,D,H,W)  │    │  • ddf    (3,D,H,W)        │ │
│  │  • dvf    (3,D,H,W)  │    └────────────────────────────┘ │
│  └──────────────────────┘              │                     │
│                                        ▼                     │
│                           ┌────────────────────────────┐    │
│                           │ LNCCWithDVFSupervision      │    │
│                           │                             │    │
│                           │ image_w × LNCC(warped,fixed)│    │
│                           │ + dvf_w × MSE(ddf, gt_dvf)  │    │
│                           └────────────────────────────┘    │
│                                        │                     │
│                                        ▼                     │
│                              ┌──────────────────┐           │
│                              │ Adam Optimizer    │           │
│                              │ + AMP + GradClip  │           │
│                              └──────────────────┘           │
│                                        │                     │
│                                        ▼                     │
│                              ┌──────────────────┐           │
│                              │ Metrics:          │           │
│                              │ NCC, MSE, MAE,    │           │
│                              │ Grad L2, EPE ✓    │           │
│                              └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

- **[Setup & Configuration →](02_setup_and_configuration.md)** — Configure and run the DeepReg pipeline
- **[Running & Evaluating →](03_running_and_evaluating.md)** — Training, monitoring, and interpreting results
