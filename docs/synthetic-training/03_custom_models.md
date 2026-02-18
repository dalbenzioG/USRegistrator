# Creating Custom Models

This guide explains how to add your own registration model to USRegistrator using the **model registry** pattern.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Model Contract](#2-the-model-contract)
3. [Step-by-Step: Building a Custom Model](#3-step-by-step-building-a-custom-model)
4. [Registering Your Model](#4-registering-your-model)
5. [Using Your Model in a Config](#5-using-your-model-in-a-config)
6. [Complete Example: VoxelMorph-style Model](#6-complete-example-voxelmorph-style-model)
7. [Tips & Best Practices](#7-tips--best-practices)

---

## 1. Architecture Overview

USRegistrator uses a **registry pattern** for models. Each model lives in its own file inside the `models/` package and is registered into a global dictionary:

```python
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}
```

The training script calls `build_model(cfg, image_size)` which:
1. Looks up the model name in `MODEL_REGISTRY`
2. Passes all other config keys as keyword arguments to the factory function
3. Returns the instantiated `nn.Module`

---

## 2. The Model Contract

Every registration model **must** follow this interface:

### Input

```python
def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `moving` | `(B, 1, D, H, W)` | The image to be deformed |
| `fixed` | `(B, 1, D, H, W)` | The target/reference image |

### Output

| Return Value | Shape | Description |
|-------------|-------|-------------|
| `warped` | `(B, 1, D, H, W)` | Moving image warped by the displacement field |
| `ddf` | `(B, 3, D, H, W)` | Dense displacement field (3 channels = x, y, z displacement) |

> **Critical**: The model must return **both** the warped image and the DDF. The training loop uses both — the warped image for loss computation, and the DDF for smoothness regularization and EPE metrics.

### Factory Function

```python
@register_model("my_model_name")
def create_my_model(image_size: Sequence[int], **kwargs) -> nn.Module:
    return MyModel(image_size=image_size, **kwargs)
```

The factory **must** accept `image_size` as its first argument.

---

## 3. Step-by-Step: Building a Custom Model

### Step 1: Define the Module Class

```python
import torch
from torch import nn
from monai.networks.blocks import Warp

class MyRegNet(nn.Module):
    """
    Custom 3D registration network.
    
    Architecture:
      1. Concatenate moving + fixed → 2-channel input
      2. Your backbone predicts a DDF
      3. Warp the moving image using the DDF
    """
    
    def __init__(
        self,
        image_size: Sequence[int],
        # ... your custom parameters ...
    ):
        super().__init__()
        
        # Your backbone network
        self.backbone = ...  # e.g., a UNet, ResNet encoder-decoder, etc.
        
        # Final conv to produce 3-channel DDF
        self.head = nn.Conv3d(
            in_channels=...,    # last feature channels from backbone
            out_channels=3,      # x, y, z displacements
            kernel_size=1,
        )
        
        # MONAI's Warp module to apply the DDF
        self.warp = Warp(mode="bilinear", padding_mode="border")
        
        # ⚡ Initialize the head to near-zero (identity warp at start)
        nn.init.normal_(self.head.weight, std=1e-5)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, D, H, W)
        
        features = self.backbone(x)             # Your backbone logic
        ddf_raw = self.head(features)            # (B, 3, D, H, W)
        
        # Optional: constrain DDF magnitude with tanh
        max_disp = 0.2
        ddf = torch.tanh(ddf_raw) * max_disp
        
        warped = self.warp(moving, ddf)          # (B, 1, D, H, W)
        
        return warped, ddf
```

### Step 2: Register It

```python
from models.registry import register_model

@register_model("myregnet")
def create_myregnet(
    image_size: Sequence[int],
    # expose whatever params you want configurable via YAML
    num_features: int = 32,
    **kwargs,
) -> nn.Module:
    return MyRegNet(image_size=image_size, num_features=num_features)
```

### Step 3: Add It to the `models/` Package

Create a new file `models/myregnet.py` with your class definition and factory function, then import it in `models/__init__.py` to trigger registration.

---

## 4. Registering Your Model

The `@register_model("name")` decorator is the key. Here's how it works internally:

```python
def register_model(name: str):
    """
    Decorator to register a model factory under a string key.
    """
    def decorator(fn: Callable[..., nn.Module]):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator
```

The name you pass to `@register_model()` is exactly the string you use in your YAML config:

```yaml
model:
  name: myregnet    # ← must match the registered name
```

---

## 5. Using Your Model in a Config

Once registered, simply reference it by name:

```yaml
model:
  name: myregnet
  num_features: 32    # ← these become **kwargs to your factory
```

All keys except `name` are forwarded to your factory function. The `image_size` is passed separately by the training script from the top-level config.

---

## 6. Complete Example: VoxelMorph-style Model

Here's a fully working example of a simplified VoxelMorph-style architecture:

```python
# Save this as models/voxelmorph.py

import torch
from torch import nn
from typing import Sequence
from monai.networks.blocks import Warp


class VoxelMorphNet(nn.Module):
    """
    Simplified VoxelMorph-style registration network.
    
    Uses a basic encoder-decoder with skip connections.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        enc_channels: Sequence[int] = (16, 32, 32, 32),
        dec_channels: Sequence[int] = (32, 32, 32, 16),
        warp_mode: str = "bilinear",
        max_disp: float = 0.2,
    ):
        super().__init__()
        self.max_disp = max_disp

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        in_ch = 2  # moving + fixed
        for out_ch in enc_channels:
            self.encoders.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch

        # ---- Decoder ----
        self.decoders = nn.ModuleList()
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i + 2)] if i < len(enc_channels) - 1 else 2
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose3d(in_ch + skip_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch

        # ---- DDF head ----
        self.head = nn.Conv3d(dec_channels[-1], 3, kernel_size=1)
        nn.init.normal_(self.head.weight, std=1e-5)
        nn.init.zeros_(self.head.bias)

        # ---- Warp ----
        self.warp = Warp(mode=warp_mode, padding_mode="border")

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)

        # Encoder with skip connections
        skips = [x]
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Decoder with skip connections
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            # Resize if needed
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # DDF prediction
        if x.shape[2:] != moving.shape[2:]:
            x = nn.functional.interpolate(x, size=moving.shape[2:], mode="trilinear", align_corners=True)

        ddf = torch.tanh(self.head(x)) * self.max_disp
        warped = self.warp(moving, ddf)

        return warped, ddf


@register_model("voxelmorph")
def create_voxelmorph(
    image_size: Sequence[int],
    enc_channels: Sequence[int] = (16, 32, 32, 32),
    dec_channels: Sequence[int] = (32, 32, 32, 16),
    warp_mode: str = "bilinear",
    max_disp: float = 0.2,
) -> nn.Module:
    return VoxelMorphNet(
        image_size=image_size,
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        warp_mode=warp_mode,
        max_disp=max_disp,
    )
```

And the config:

```yaml
model:
  name: voxelmorph
  enc_channels: [16, 32, 32, 32]
  dec_channels: [32, 32, 32, 16]
  max_disp: 0.2
```

---

## 7. Tips & Best Practices

### Initialize the DDF Head to Near-Zero

This ensures the model starts with an **identity transformation** (no warping). Without this, the initial random DDF creates highly distorted warped images that can destabilize training.

```python
nn.init.normal_(self.head.weight, std=1e-5)
nn.init.zeros_(self.head.bias)
```

### Constrain DDF Magnitude

Use `torch.tanh()` + scaling to prevent unreasonably large displacements:

```python
ddf = torch.tanh(ddf_raw) * max_disp  # e.g., max_disp = 0.2
```

The `max_disp` value should match your dataset's expected deformation range.

### Use MONAI's `Warp`

Don't implement your own warping — MONAI's `Warp` module handles edge cases (padding, interpolation modes) correctly and is GPU-optimized.

### Available Built-in Models

| Name | Class | Description |
|------|-------|-------------|
| `globalnet3d` | `GlobalNet3D` | MONAI GlobalNet — global affine-like + deformable |
| `localnet3d` | `LocalNet3D` | MONAI LocalNet — local deformable only |
| `unetreg3d` | `UNetReg3D` | Standard UNet encoder-decoder for DDF prediction |

### Adding New Models to the Package

The `models/` package is already organized as individual files. To add a new model:

1. Create `models/my_custom_model.py` with the class and `@register_model("my_model")` factory
2. Import it in `models/__init__.py` to trigger registration:

```python
# In models/__init__.py, add:
from . import my_custom_model
from .my_custom_model import MyCustomModel, create_my_custom_model
```

This automatically makes `my_model` available via YAML config:

```yaml
model:
  name: my_model
```

---

## Next Steps

- **[Creating Custom Losses →](04_custom_losses.md)**
- **[Creating Custom Metrics →](05_custom_metrics.md)**
