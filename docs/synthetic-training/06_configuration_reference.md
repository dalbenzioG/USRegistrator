# Configuration Reference

Complete reference for every YAML configuration option in USRegistrator.

---

## Config File Structure

```yaml
# Top-level settings
project: "..."
image_size: [D, H, W]

# Sections
wandb: { ... }
train_dataset: { ... }
val_dataset: { ... }
model: { ... }
loss: { ... }
optimizer: { ... }
training: { ... }
```

---

## Top-Level Settings

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `project` | `str` | No | — | Project identifier (informational) |
| `image_size` | `[int, int, int]` | **Yes** | — | Volume dimensions `[D, H, W]`. Passed to the model constructor |

---

## `wandb` — Experiment Tracking

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `enabled` | `bool` | **Yes** | — | Enable/disable W&B logging |
| `project` | `str` | Yes (if enabled) | — | W&B project name |
| `run_name` | `str` | Yes (if enabled) | — | Display name for this run |
| `offline` | `bool` | No | `false` | If `true`, logs locally without internet |

---

## `train_dataset` / `val_dataset` — Data

Both sections share the same schema. The `name` field selects the dataset type.

### `synthetic_ellipsoids`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | **Yes** | — | Must be `"synthetic_ellipsoids"` |
| `image_size` | `[int, int, int]` | No | `[64, 64, 64]` | Volume dimensions |
| `num_samples` | `int` | **Yes** | — | Number of samples to generate |
| `noise_std` | `float` | No | `0.03` | Gaussian noise standard deviation |
| `smooth` | `bool` | No | `true` | Smooth ellipsoid boundaries |
| `seed` | `int` | No | `123` | Random seed (auto-offset per split) |

Returns: `{"moving": (1,D,H,W), "fixed": (1,D,H,W)}`

### `deepreg_synthetic`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | **Yes** | — | Must be `"deepreg_synthetic"` |
| `image_size` | `[int, int, int]` | No | `[64, 64, 64]` | Volume dimensions |
| `num_samples` | `int` | **Yes** | — | Number of samples to generate |
| `max_disp` | `float` | No | `0.2` | Maximum displacement magnitude (normalized coordinates) |
| `cp_spacing` | `int` | No | `8` | Control point spacing for DVF generation |
| `noise_std` | `float` | No | `0.03` | Gaussian noise standard deviation |
| `smooth` | `bool` | No | `true` | Smooth ellipsoid boundaries |
| `seed` | `int` | No | `123` | Random seed (auto-offset per split) |

Returns: `{"moving": (1,D,H,W), "fixed": (1,D,H,W), "dvf": (3,D,H,W)}`

---

## `model` — Registration Network

The `name` field selects the model. All other keys are passed as keyword arguments.

### `globalnet3d`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | **Yes** | — | Must be `"globalnet3d"` |
| `num_channel_initial` | `int` | No | `16` | Initial feature channels |
| `depth` | `int` | No | `3` | Encoder depth (number of downsampling levels) |
| `warp_mode` | `str` | No | `"bilinear"` | Warp interpolation: `"bilinear"` or `"nearest"` |
| `warp_padding_mode` | `str` | No | `"border"` | Warp padding: `"border"`, `"zeros"`, or `"reflection"` |

### `localnet3d`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | **Yes** | — | Must be `"localnet3d"` |
| `in_channels` | `int` | No | `2` | Input channels (moving + fixed) |
| `num_channel_initial` | `int` | No | `16` | Initial feature channels |
| `depth` | `int` | No | `3` | Encoder depth |
| `warp_mode` | `str` | No | `"bilinear"` | Warp interpolation |
| `warp_padding_mode` | `str` | No | `"border"` | Warp padding |

### `unetreg3d`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | **Yes** | — | Must be `"unetreg3d"` |
| `in_channels` | `int` | No | `2` | Input channels |
| `out_channels` | `int` | No | `3` | Output channels (3 for DDF) |
| `channels` | `[int, ...]` | No | `[16,32,64,128,256]` | Feature channels per level |
| `strides` | `[int, ...]` | No | `[2,2,2,2]` | Downsampling strides |
| `num_res_units` | `int` | No | `2` | Residual units per block |
| `warp_mode` | `str` | No | `"bilinear"` | Warp interpolation |
| `warp_padding_mode` | `str` | No | `"border"` | Warp padding |

---

## `loss` — Loss Function

### `lncc`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | **Yes** | — | Must be `"lncc"` |
| `kernel_size` | `int` | No | `3` | Local window size (also accepts `patch_size`) |
| `spatial_dims` | `int` | No | `3` | Spatial dimensions |
| `kernel_type` | `str` | No | `"rectangular"` | Kernel shape |
| `reduction` | `str` | No | `"mean"` | Reduction method |
| `smooth_nr` | `float` | No | `0.0` | Numerator smoothing (clamped to ≥ 1e-4) |
| `smooth_dr` | `float` | No | `1e-5` | Denominator smoothing (clamped to ≥ 1e-2) |

### `lncc_dvf`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | **Yes** | — | Must be `"lncc_dvf"` |
| `kernel_size` | `int` | No | `9` | Local window size |
| `spatial_dims` | `int` | No | `3` | Spatial dimensions |
| `kernel_type` | `str` | No | `"rectangular"` | Kernel shape |
| `reduction` | `str` | No | `"mean"` | Reduction method |
| `smooth_nr` | `float` | No | `1e-4` | Numerator smoothing |
| `smooth_dr` | `float` | No | `1e-2` | Denominator smoothing |
| `image_weight` | `float` | No | `1.0` | Weight for image similarity component |
| `dvf_weight` | `float` | No | `1.0` | Weight for DVF MSE supervision component |

### `mse`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | **Yes** | — | Must be `"mse"` |

No additional parameters.

---

## `optimizer` — Optimizer

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | `str` | No | `"Adam"` | `"Adam"` or `"AdamW"` |
| `lr` | `float` | **Yes** | — | Learning rate. Recommended: `1e-4` to `1e-5` |
| `weight_decay` | `float` | No | `0.0` | L2 weight decay. Recommended: `1e-5` |

> **Warning**: Learning rates above `1e-3` will trigger a console warning. Very high learning rates can cause NaN with LNCC loss.

---

## `training` — Training Loop

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `epochs` | `int` | **Yes** | — | Total training epochs |
| `batch_size` | `int` | **Yes** | — | Samples per batch |
| `num_workers` | `int` | No | `4` | DataLoader worker processes |
| `amp` | `bool` | No | `true` | Enable mixed precision (CUDA only) |
| `val_every` | `int` | No | `1` | Run validation every N epochs (0 = disabled) |
| `seed` | `int` | No | `42` | Global random seed for reproducibility |

---

## Complete Template

```yaml
project: "monai-3d-registration"
image_size: [64, 64, 64]

wandb:
  enabled: true
  project: "usregistrator"
  run_name: "experiment-001"
  offline: false

train_dataset:
  name: synthetic_ellipsoids
  image_size: [64, 64, 64]
  num_samples: 4000
  noise_std: 0.03
  smooth: true
  seed: 123

val_dataset:
  name: synthetic_ellipsoids
  image_size: [64, 64, 64]
  num_samples: 200
  noise_std: 0.03
  smooth: true
  seed: 999

model:
  name: globalnet3d
  num_channel_initial: 16
  depth: 3
  warp_mode: bilinear
  warp_padding_mode: border

loss:
  name: lncc
  kernel_size: 9
  smooth_nr: 1e-4
  smooth_dr: 1e-2

optimizer:
  name: Adam
  lr: 1e-4
  weight_decay: 1e-5

training:
  epochs: 50
  batch_size: 2
  num_workers: 4
  amp: true
  val_every: 1
  seed: 42
```

---

## See Also

- **[Training Tutorial](02_training_tutorial.md)** — How to use a config to train
- **[DeepReg Config](../deepreg-pipeline/02_setup_and_configuration.md)** — Config specific to DeepReg-style datasets
