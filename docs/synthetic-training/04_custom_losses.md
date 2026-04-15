# Creating Custom Losses

This guide explains how to implement and register your own loss functions in USRegistrator's **loss registry** system.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Loss Contract](#2-the-loss-contract)
3. [Step-by-Step: Building a Custom Loss](#3-step-by-step-building-a-custom-loss)
4. [Complete Example: Dice + Smoothness Loss](#4-complete-example-dice--smoothness-loss)
5. [Working with DVF-Aware Losses](#5-working-with-dvf-aware-losses)
6. [Numerical Stability Tips](#6-numerical-stability-tips)
7. [Available Built-in Losses](#7-available-built-in-losses)

---

## 1. Architecture Overview

Loss functions live in the `losses/` package and follow the same registry pattern as models:

```
losses/
├── __init__.py       # Imports all modules to trigger registration
├── utils.py          # Registry, build_loss(), validate_smoothing_params()
├── lncc.py           # Local NCC loss
├── lncc_dvf.py       # LNCC + DVF MSE supervision
└── mse.py            # Standard MSE loss
```

The registry:

```python
LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}
```

The training script calls `build_loss(cfg)` which looks up the loss name and passes remaining config keys as kwargs to the factory.

---

## 2. The Loss Contract

Every loss function must be an `nn.Module` with one of two `forward()` signatures:

### Simple (image-only) Loss

```python
def forward(self, warped: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `warped` | `(B, 1, D, H, W)` | Model's warped output |
| `fixed` | `(B, 1, D, H, W)` | Target image |
| **Returns** | scalar `Tensor` | Loss value to minimize |

### DVF-Aware Loss

```python
def forward(
    self,
    warped: torch.Tensor,
    fixed: torch.Tensor,
    pred_dvf: torch.Tensor | None = None,
    gt_dvf: torch.Tensor | None = None,
) -> torch.Tensor:
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `pred_dvf` | `(B, 3, D, H, W)` | Predicted displacement field |
| `gt_dvf` | `(B, 3, D, H, W)` or `None` | Ground-truth DVF (if available) |

> **Backward compatibility**: The training loop first tries calling the loss with `(warped, fixed, ddf, gt_dvf)`. If that raises a `TypeError` (your loss doesn't accept those args), it falls back to `(warped, fixed)`. So simple losses "just work."

---

## 3. Step-by-Step: Building a Custom Loss

### Step 1: Create a New File

Create `losses/my_loss.py`:

```python
"""My custom loss function."""

from __future__ import annotations

import torch
from torch import nn
from .utils import register_loss
```

### Step 2: Implement the Loss Module

```python
class MyCustomLoss(nn.Module):
    """
    Example: Weighted combination of MSE and L1.
    """
    
    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
    
    def forward(self, warped: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        mse = torch.nn.functional.mse_loss(warped, fixed)
        l1 = torch.nn.functional.l1_loss(warped, fixed)
        return self.mse_weight * mse + self.l1_weight * l1
```

### Step 3: Register It

```python
@register_loss("my_custom_loss")
def create_my_custom_loss(
    mse_weight: float = 1.0,
    l1_weight: float = 0.5,
    **_,   # absorb extra config keys gracefully
) -> nn.Module:
    return MyCustomLoss(mse_weight=mse_weight, l1_weight=l1_weight)
```

### Step 4: Import in `losses/__init__.py`

Add the import to trigger registration:

```python
from . import my_loss   # ← add this line
```

### Step 5: Use in Config

```yaml
loss:
  name: my_custom_loss
  mse_weight: 1.0
  l1_weight: 0.5
```

---

## 4. Complete Example: Dice + Smoothness Loss

Here's a more advanced example combining image similarity with deformation smoothness:

```python
"""Dice similarity loss + deformation field smoothness regularization."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from .utils import register_loss


class DiceSmoothLoss(nn.Module):
    """
    Combined loss:
      - Soft Dice: penalizes misalignment of intensity distributions
      - Gradient smoothness: penalizes jagged/folding deformations
    
    Total = dice_weight * dice_loss + smooth_weight * smoothness_loss
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        smooth_weight: float = 0.01,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth_weight = smooth_weight
        self.epsilon = epsilon

    def _soft_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss (1 - Dice coefficient)."""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - dice

    def _gradient_smoothness(self, ddf: torch.Tensor) -> torch.Tensor:
        """L2 norm of spatial gradients of the displacement field."""
        dz = torch.diff(ddf, dim=2)
        dy = torch.diff(ddf, dim=3)
        dx = torch.diff(ddf, dim=4)
        return dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()

    def forward(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        pred_dvf: torch.Tensor | None = None,
        gt_dvf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Image similarity
        dice_loss = self._soft_dice(warped, fixed)
        total = self.dice_weight * dice_loss

        # Smoothness regularization (only if DDF is available)
        if self.smooth_weight > 0 and pred_dvf is not None:
            smooth_loss = self._gradient_smoothness(pred_dvf)
            total = total + self.smooth_weight * smooth_loss

        return total


@register_loss("dice_smooth")
def create_dice_smooth(
    dice_weight: float = 1.0,
    smooth_weight: float = 0.01,
    epsilon: float = 1e-5,
    **_,
) -> nn.Module:
    return DiceSmoothLoss(
        dice_weight=dice_weight,
        smooth_weight=smooth_weight,
        epsilon=epsilon,
    )
```

Config:

```yaml
loss:
  name: dice_smooth
  dice_weight: 1.0
  smooth_weight: 0.01
```

---

## 5. Working with DVF-Aware Losses

If your loss needs access to the displacement field (e.g., for regularization or supervision), use the 4-argument signature:

```python
def forward(
    self,
    warped: torch.Tensor,
    fixed: torch.Tensor,
    pred_dvf: torch.Tensor | None = None,
    gt_dvf: torch.Tensor | None = None,
) -> torch.Tensor:
```

The training loop passes:
- `pred_dvf` — always available (the model always outputs a DDF)
- `gt_dvf` — only available when the dataset provides ground-truth DVFs (e.g., `deepreg_synthetic`)

**Handle the `None` case gracefully**:

```python
if gt_dvf is not None:
    dvf_loss = F.mse_loss(pred_dvf, gt_dvf)
else:
    dvf_loss = torch.tensor(0.0, device=warped.device)
```

See `losses/lncc_dvf.py` for a production example of this pattern.

---

## 6. Numerical Stability Tips

### Mixed Precision (AMP)

When AMP is enabled, tensors may be `float16`. Key considerations:

- **Use large smoothing denominators** (`1e-2` minimum) to prevent division by zero in float16
- **Use `validate_smoothing_params()`** from `losses/utils.py` to enforce safe bounds:

```python
from .utils import validate_smoothing_params

smooth_nr, smooth_dr = validate_smoothing_params(smooth_nr, smooth_dr)
# smooth_dr is clamped to minimum 1e-2
# smooth_nr is clamped to minimum 1e-4
```

### Gradient Clipping

The training loop already clips gradients at `max_norm=1.0`:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### NaN Detection

The training loop skips batches with NaN/Inf values in inputs or outputs. If your loss produces NaN, investigate:

1. Division by near-zero values
2. `log()` of zero or negative values
3. Very large loss values causing overflow in float16

---

## 7. Available Built-in Losses

| Name | Class/Source | Description |
|------|-------------|-------------|
| `lncc` | MONAI `LocalNormalizedCrossCorrelationLoss` | Local normalized cross-correlation |
| `lncc_dvf` | `LNCCWithDVFSupervision` | LNCC + MSE between predicted and GT DVFs |
| `mse` | PyTorch `MSELoss` | Simple mean squared error |

### Choosing a Loss

| Use Case | Recommended Loss |
|----------|-----------------|
| Intensity-based (same modality) | `lncc` or `mse` |
| With ground-truth DVFs | `lncc_dvf` |
| Segmentation-based | Custom Dice loss |
| Multi-objective | Custom weighted combination |

---

## Next Steps

- **[Creating Custom Metrics →](05_custom_metrics.md)**
- **[Configuration Reference →](06_configuration_reference.md)**
