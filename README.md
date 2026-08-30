# USRegistrator - 3D Medical Image Registration

A PyTorch-based framework for 3D medical image registration using MONAI. This repository implements deep learning models for deformable image registration with support for synthetic data generation, multiple loss functions, and comprehensive training pipelines.

## Features

- **3D Image Registration**: Deformable registration using GlobalNet architecture
- **Synthetic Data Generation**: Built-in synthetic ellipsoid dataset for testing
- **Custom Dataset Support**: JSON-driven training/validation on user medical datasets
- **Multiple Loss Functions**: LNCC, MSE, and combined LNCC+Dice (`lncc_dice`)
- **Training Pipeline**: Complete training loop with validation, metrics, and logging
- **Weights & Biases Integration**: Automatic experiment tracking and visualization
- **Validation Diagnostics**: Segmentation overlay slices and deformation-field diagnostics
- **Mixed Precision Training**: AMP support for faster training on modern GPUs

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended)
- Windows/Linux/macOS

## Setup

### 1. Create Virtual Environment

#### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv monai-reg

# Activate virtual environment
.\monai-reg\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
# Create virtual environment
python -m venv monai-reg

# Activate virtual environment
monai-reg\Scripts\activate.bat
```

#### Linux/macOS
```bash
# Create virtual environment
python3 -m venv monai-reg

# Activate virtual environment
source monai-reg/bin/activate
```

### 2. Install Dependencies

After activating the virtual environment, install the required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai wandb pyyaml numpy
```

Or install from a requirements file (if you have one):

```bash
pip install -r requirements.txt
```

**Note**: The PyTorch installation above is for CUDA 11.8. Adjust the CUDA version and index URL based on your system:
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CPU only: `pip install torch torchvision torchaudio`

### 3. Verify Installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import monai; print(monai.__version__)"
```

## Usage

### Basic Training

Train a model using the default configuration:

```bash
python train.py --config configs/config_template.yaml
```

### Custom Configuration

1. Copy the template configuration:
   ```bash
   cp configs/config_template.yaml configs/my_experiment.yaml
   ```

2. Edit `configs/configs_template.yaml` with your desired settings

3. Run training:
   ```bash
   python train.py --config configs/my_experiment.yaml
   ```

### Train on Your Own Dataset

Use the custom dataset workflow documented in `docs/custom-training/custom_dataset.md` with:

- JSON manifest containing `train` and `val` cases
- YAML dataset blocks using `name: custom_dataset` and `json_file: ...`
- Example config: `configs/custom_dataset_example.yaml`
- Full docs index: `docs/README.md`

## Configuration Guide

The configuration file is a YAML file with the following sections:

### Dataset Configuration

```yaml
train_dataset:
  name: synthetic_ellipsoids
  image_size: [64, 64, 64]
  num_samples: 4000
  noise_std: 0.03
  smooth: true
  seed: 123
```

### Model Configuration

```yaml
model:
  name: globalnet3d
  num_channel_initial: 16  # Initial number of channels
  depth: 3                 # Network depth
  warp_mode: bilinear      # Interpolation mode
  warp_padding_mode: border
```

### Loss Configuration

```yaml
loss:
  name: lncc              # Options: lncc, mse
  kernel_size: 9          # Kernel size for LNCC
  smooth_nr: 1e-4         # Numerator smoothing
  smooth_dr: 1e-2         # Denominator smoothing (prevents NaN)
```

### Optimizer Configuration

```yaml
optimizer:
  name: Adam              # Options: Adam, AdamW
  lr: 1e-4                # Learning rate
  weight_decay: 1e-5      # Weight decay
```

### Training Configuration

```yaml
training:
  epochs: 50
  batch_size: 2
  num_workers: 4          # DataLoader workers
  amp: true               # Mixed precision training
  val_every: 1            # Validation frequency
  seed: 42
```

## Project Structure

```
USRegistrator/
├── train.py              # Main training script
├── models/               # Registration models (modular)
│   ├── __init__.py       # Registry & re-exports
│   ├── registry.py       # MODEL_REGISTRY, register_model, build_model
│   ├── globalnet3d.py    # GlobalNet3D
│   ├── localnet3d.py     # LocalNet3D
│   └── unetreg3d.py      # UNetReg3D
├── datasets/             # Dataset generators (modular)
│   ├── __init__.py       # Registry & re-exports
│   ├── registry.py       # DATASET_REGISTRY, register_dataset, build_dataset
│   ├── synthetic_ellipsoids.py  # Ellipsoid generator
│   ├── deepreg_synthetic.py     # DeepReg-style DVF generator
│   └── custom_dataset.py        # JSON-driven custom medical dataset
├── losses/               # Loss functions (modular)
│   ├── __init__.py
│   ├── utils.py          # Registry & helpers
│   ├── lncc.py           # Local NCC loss
│   ├── lncc_dvf.py       # LNCC + DVF supervision
│   ├── combo.py          # Combined losses (e.g., LNCC + Dice)
│   └── mse.py            # MSE loss
├── metrics/              # Evaluation metrics (modular)
│   ├── __init__.py
│   ├── ncc.py            # Global NCC metric
│   ├── epe.py            # Endpoint Error
│   ├── regression.py     # MSE / MAE metrics
│   ├── smoothness.py     # Gradient smoothness
│   ├── jacobian.py       # Jacobian-based deformation diagnostics
│   └── dice.py           # Dice overlap metrics
├── configs/
│   ├── config_template.yaml   # Synthetic ellipsoids config
│   ├── deepreg_synth.yaml     # DeepReg-style DVF config
│   └── custom_dataset_example.yaml  # Custom dataset config template
├── docs/
│   ├── README.md          # Documentation hub (synthetic / deepreg / custom dataset)
│   └── custom-training/
│       └── custom_dataset.md  # Custom dataset setup and usage guide
├── monai-reg/            # Virtual environment (gitignored)
└── wandb/                # Weights & Biases logs (gitignored)
```

## Key Components

### Models

- **GlobalNet3D**: 3D deformable registration network using MONAI's GlobalNet architecture
- Returns: `(warped_image, displacement_field)`

### Datasets

- **SyntheticEllipsoids**: Generates synthetic 3D ellipsoid volumes for training/testing
- Supports configurable image size, noise, and smoothing

### Loss Functions

- **LNCC (Local Normalized Cross-Correlation)**: Intensity-based similarity metric
- **MSE**: Mean Squared Error
- **LNCC+Dice**: Combined image-similarity and label-overlap supervision (`lncc_dice`)

### Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **NCC**: Global Normalized Cross-Correlation
- **Dice**: Segmentation overlap metric for labeled data
- **Grad_L2**: Gradient magnitude of displacement field
- **Jacobian Diagnostics**: Folding ratio and Jacobian statistics

## Weights & Biases Integration

The training script automatically logs to Weights & Biases:

1. Create a W&B account at https://wandb.ai
2. Login when prompted: `wandb login`
3. Training metrics, losses, and visualizations will be logged automatically

To disable W&B, you can modify `train.py` to skip `wandb.init()`.


## Contributing

### Optional original TransMorph

Use `model.name: transmorph_original3d` to load a pinned original TransMorph
implementation. See [setup, runnable Tiny example and validation](docs/transmorph-validation.md)
and the [experiment/decision history](docs/experiment-history.md). Only Tiny is
integration-tested; the source checkout and optional dependencies are not required
for existing core models. This is distinct from a MONAI SwinUNETR-based model.

[Add contribution guidelines if applicable]

