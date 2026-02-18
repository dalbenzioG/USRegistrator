# Creating Custom Metrics

This guide explains how to add your own evaluation metrics to USRegistrator's **metrics** system.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Metric Contract](#2-the-metric-contract)
3. [Step-by-Step: Adding a Custom Metric](#3-step-by-step-adding-a-custom-metric)
4. [Complete Examples](#4-complete-examples)
5. [Class-Based vs Functional Metrics](#5-class-based-vs-functional-metrics)
6. [Using MONAI's Built-in Metric Classes](#6-using-monais-built-in-metric-classes)
7. [Available Built-in Metrics](#7-available-built-in-metrics)

---

## 1. Architecture Overview

Metrics live in the `metrics/` package. Unlike models and losses, metrics do **not** use a decorator-based registry. Instead, they are manually registered in `metrics/__init__.py`:

```python
# metrics/__init__.py

METRICS = {
    "mse": mse,
    "mae": mae,
    "ncc": global_ncc,
    "grad_l2": gradient_l2,
    "epe": epe,
}
```

Each entry maps a metric name to a **callable** (either a function or a class instance).

The evaluation loop in `train.py` iterates over `METRICS` and calls each one:

```python
for name, fn in METRICS.items():
    if name == "grad_l2":
        metric_totals[name] += fn(ddf) * bs
    elif name == "epe":
        if gt_dvf is not None:
            metric_totals[name] += fn(ddf, gt_dvf) * bs
    else:
        metric_totals[name] += fn(warped, fixed) * bs
```

---

## 2. The Metric Contract

Metrics fall into three categories based on their input signature:

### Image-Based Metrics

Compare warped vs. fixed images:

```python
def my_metric(warped: torch.Tensor, fixed: torch.Tensor) -> float:
    # warped: (B, 1, D, H, W)
    # fixed:  (B, 1, D, H, W)
    return scalar_value
```

### DDF-Only Metrics

Evaluate the displacement field alone:

```python
def my_metric(ddf: torch.Tensor) -> float:
    # ddf: (B, 3, D, H, W)
    return scalar_value
```

### DDF + Ground-Truth Metrics

Compare predicted DDF to ground truth:

```python
def my_metric(ddf: torch.Tensor, gt_dvf: torch.Tensor) -> float:
    # ddf:    (B, 3, D, H, W)
    # gt_dvf: (B, 3, D, H, W)
    return scalar_value
```

> **Return type**: Metrics should return a plain Python `float`, not a tensor. Use `.item()` to convert.

---

## 3. Step-by-Step: Adding a Custom Metric

### Step 1: Create a New File

Create `metrics/my_metric.py`:

```python
"""My custom evaluation metric."""

import torch
from torch import Tensor


def my_metric(warped: Tensor, fixed: Tensor) -> float:
    """
    Compute my custom metric between warped and fixed images.
    """
    # Your computation here
    result = ...
    return result.item()  # ← must return float
```

### Step 2: (Optional) Add a Class-Based Version

```python
from monai.metrics import Metric

class MyMetricClass(Metric):
    def __init__(self, ...):
        super().__init__()
    
    def __call__(self, y_pred: Tensor, y: Tensor) -> Tensor:
        return torch.tensor(my_metric(y_pred, y))
```

### Step 3: Register in `metrics/__init__.py`

```python
# Add the import
from .my_metric import my_metric  # and optionally MyMetricClass

# Add to the METRICS dict
METRICS = {
    "mse": mse,
    "mae": mae,
    "ncc": global_ncc,
    "grad_l2": gradient_l2,
    "epe": epe,
    "my_metric": my_metric,   # ← add this line
}
```

### Step 4: Update the Evaluation Loop (if needed)

If your metric uses the **image-based** signature `(warped, fixed)`, it works automatically — no changes to `train.py` needed.

If your metric uses the **DDF-only** or **DDF + GT** signature, you need to add a condition in the `evaluate()` function in `train.py`:

```python
for name, fn in METRICS.items():
    if name == "grad_l2":
        metric_totals[name] += fn(ddf) * bs
    elif name == "epe":
        if gt_dvf is not None:
            metric_totals[name] += fn(ddf, gt_dvf) * bs
    elif name == "my_ddf_metric":          # ← your new condition
        metric_totals[name] += fn(ddf) * bs
    else:
        metric_totals[name] += fn(warped, fixed) * bs
```

---

## 4. Complete Examples

### Example 1: Structural Similarity Index (SSIM)

```python
"""SSIM metric for 3D volumes."""

import torch
from torch import Tensor
import torch.nn.functional as F


def ssim_3d(
    warped: Tensor,
    fixed: Tensor,
    window_size: int = 7,
    eps: float = 1e-6,
) -> float:
    """
    Compute mean SSIM over 3D volume slices.
    
    A simplified SSIM computed on the mean intensity along each axis.
    """
    # Compute along the depth axis (mid-slice approach)
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu_w = F.avg_pool3d(warped, window_size, stride=1, padding=window_size // 2)
    mu_f = F.avg_pool3d(fixed, window_size, stride=1, padding=window_size // 2)
    
    mu_w_sq = mu_w ** 2
    mu_f_sq = mu_f ** 2
    mu_wf = mu_w * mu_f
    
    sigma_w_sq = F.avg_pool3d(warped ** 2, window_size, stride=1, padding=window_size // 2) - mu_w_sq
    sigma_f_sq = F.avg_pool3d(fixed ** 2, window_size, stride=1, padding=window_size // 2) - mu_f_sq
    sigma_wf = F.avg_pool3d(warped * fixed, window_size, stride=1, padding=window_size // 2) - mu_wf
    
    numerator = (2 * mu_wf + C1) * (2 * sigma_wf + C2)
    denominator = (mu_w_sq + mu_f_sq + C1) * (sigma_w_sq + sigma_f_sq + C2)
    
    ssim_map = numerator / (denominator + eps)
    return ssim_map.mean().item()
```

### Example 2: Jacobian Determinant (Folding Detection)

```python
"""Jacobian determinant metric for detecting folding in the DVF."""

import torch
from torch import Tensor


def jacobian_det_fraction(ddf: Tensor, threshold: float = 0.0) -> float:
    """
    Fraction of voxels with negative Jacobian determinant.
    
    A negative Jacobian determinant indicates local "folding" of the 
    deformation field — physically implausible deformations.
    
    ddf: (B, 3, D, H, W)
    Returns: fraction of voxels with det(J) < threshold
    """
    # Compute spatial gradients via finite differences
    # ddf[:, 0] = z-displacement, ddf[:, 1] = y-displacement, ddf[:, 2] = x-displacement
    
    # Add identity to get the deformation field (phi = id + ddf)
    # Jacobian = I + grad(ddf)
    
    dudz = torch.diff(ddf[:, 0:1], dim=2)  # ∂u_z/∂z
    dudy = torch.diff(ddf[:, 0:1], dim=3)  # ∂u_z/∂y  
    dudx = torch.diff(ddf[:, 0:1], dim=4)  # ∂u_z/∂x
    
    dvdz = torch.diff(ddf[:, 1:2], dim=2)  # ∂u_y/∂z
    dvdy = torch.diff(ddf[:, 1:2], dim=3)  # ∂u_y/∂y
    dvdx = torch.diff(ddf[:, 1:2], dim=4)  # ∂u_y/∂x
    
    dwdz = torch.diff(ddf[:, 2:3], dim=2)  # ∂u_x/∂z
    dwdy = torch.diff(ddf[:, 2:3], dim=3)  # ∂u_x/∂y
    dwdx = torch.diff(ddf[:, 2:3], dim=4)  # ∂u_x/∂x
    
    # Crop to common size (finite differences reduce dimension by 1)
    min_d = min(dudz.shape[2], dvdz.shape[2], dwdz.shape[2])
    min_h = min(dudy.shape[3], dvdy.shape[3], dwdy.shape[3])
    min_w = min(dudx.shape[4], dvdx.shape[4], dwdx.shape[4])
    
    # Jacobian = I + grad(ddf)
    # J = [[1+dudz, dudy, dudx],
    #      [dvdz, 1+dvdy, dvdx],
    #      [dwdz, dwdy, 1+dwdx]]
    
    j00 = 1 + dudz[:, :, :min_d, :min_h, :min_w]
    j01 = dudy[:, :, :min_d, :min_h, :min_w]
    j02 = dudx[:, :, :min_d, :min_h, :min_w]
    j10 = dvdz[:, :, :min_d, :min_h, :min_w]
    j11 = 1 + dvdy[:, :, :min_d, :min_h, :min_w]
    j12 = dvdx[:, :, :min_d, :min_h, :min_w]
    j20 = dwdz[:, :, :min_d, :min_h, :min_w]
    j21 = dwdy[:, :, :min_d, :min_h, :min_w]
    j22 = 1 + dwdx[:, :, :min_d, :min_h, :min_w]
    
    # det(J) = j00*(j11*j22 - j12*j21) - j01*(j10*j22 - j12*j20) + j02*(j10*j21 - j11*j20)
    det = (j00 * (j11 * j22 - j12 * j21)
         - j01 * (j10 * j22 - j12 * j20)
         + j02 * (j10 * j21 - j11 * j20))
    
    negative_fraction = (det < threshold).float().mean().item()
    return negative_fraction
```

Register both:

```python
# metrics/__init__.py
from .ssim import ssim_3d
from .jacobian import jacobian_det_fraction

METRICS = {
    ...
    "ssim": ssim_3d,
    "jac_neg_frac": jacobian_det_fraction,
}
```

And update `train.py` evaluation loop for the DDF-only metric:

```python
elif name == "jac_neg_frac":
    metric_totals[name] += fn(ddf) * bs
```

---

## 5. Class-Based vs Functional Metrics

USRegistrator uses **functional** metrics (plain functions returning `float`) in the `METRICS` dict. However, each metric also provides a **class-based** version inheriting from MONAI's `Metric` base class.

| Style | Use When |
|-------|----------|
| **Functional** (`def my_metric(...)`) | USRegistrator's `METRICS` dict (default) |
| **Class-based** (`class MyMetric(Metric)`) | MONAI integration, stateful accumulation, or external frameworks |

### Why Both?

The functional style is simpler for the training loop. The class-based style is useful if you want to:
- Accumulate metrics over multiple batches before computing the final value
- Use MONAI's metric composition and reduction framework
- Plug into other MONAI-based workflows

### MONAI Base Classes

```python
from monai.metrics import Metric           # Most general base
from monai.metrics import RegressionMetric  # For regression-type metrics (MSE, MAE, etc.)
```

---

## 6. Using MONAI's Built-in Metric Classes

MONAI provides many pre-built metrics you can wrap:

```python
# Example: wrapping MONAI's DiceMetric
from monai.metrics import DiceMetric as MonaiDice

_dice = MonaiDice(include_background=True, reduction="mean")

def dice_metric(warped: Tensor, fixed: Tensor) -> float:
    """
    Compute Dice coefficient.
    
    Note: Dice is typically used for segmentation masks, not raw intensities.
    You may need to threshold the images first.
    """
    # Threshold to create binary masks
    w_binary = (warped > 0.5).float()
    f_binary = (fixed > 0.5).float()
    result = _dice(w_binary, f_binary)
    return result.mean().item()
```

Other useful MONAI metrics:
- `monai.metrics.HausdorffDistanceMetric`
- `monai.metrics.SurfaceDistanceMetric`
- `monai.metrics.ConfusionMatrixMetric`

---

## 7. Available Built-in Metrics

| Name | Function | Input | Description |
|------|----------|-------|-------------|
| `mse` | `mse(warped, fixed)` | Image-based | Mean Squared Error |
| `mae` | `mae(warped, fixed)` | Image-based | Mean Absolute Error |
| `ncc` | `global_ncc(warped, fixed)` | Image-based | Global Normalized Cross-Correlation |
| `grad_l2` | `gradient_l2(ddf)` | DDF-only | L2 norm of DDF spatial gradients |
| `epe` | `epe(ddf, gt_dvf)` | DDF + GT | Endpoint Error (requires GT DVF) |

### Where They're Defined

| Metric | File |
|--------|------|
| MSE, MAE | `metrics/regression.py` |
| NCC | `metrics/ncc.py` |
| Gradient L2 | `metrics/smoothness.py` |
| EPE | `metrics/epe.py` |

---

## Next Steps

- **[Configuration Reference →](06_configuration_reference.md)**
- **[Creating Custom Models ←](03_custom_models.md)**
- **[Creating Custom Losses ←](04_custom_losses.md)**
