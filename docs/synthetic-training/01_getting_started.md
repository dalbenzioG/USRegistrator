# Getting Started

This guide walks you through setting up the USRegistrator environment and verifying everything works before your first training run.

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|------------|----------------|-------|
| Python | 3.12+ | Tested on 3.12 |
| CUDA | 11.8+ | Optional but strongly recommended |
| GPU VRAM | ≥ 4 GB | For 64³ volumes with batch size 2 |

---

## 1. Clone the Repository

```bash
git clone https://github.com/TheHuntsman4/USRegistrator.git
cd USRegistrator
```

---

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python3 -m venv monai-reg
source monai-reg/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv monai-reg
.\monai-reg\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```cmd
python -m venv monai-reg
monai-reg\Scripts\activate.bat
```

---

## 3. Install Dependencies

### Step 1 — PyTorch (CUDA-aware)

Choose the command matching your CUDA version:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (no GPU acceleration)
pip install torch torchvision torchaudio
```

### Step 2 — Remaining dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **MONAI** (≥ 1.5) — medical imaging deep-learning framework
- **Weights & Biases** (≥ 0.23) — experiment tracking
- **PyYAML** (≥ 6.0) — configuration parsing
- **NumPy** (≥ 1.24) — numerical computing

---

## 4. Verify the Installation

Run the following quick checks to make sure everything is working:

```bash
# Check PyTorch + CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Check MONAI
python -c "import monai; print('MONAI:', monai.__version__)"

# Check the project imports work
python -c "from datasets import build_dataset; from models import build_model; from losses import build_loss; from metrics import METRICS; print('All imports OK ✓')"
```

**Expected output** (versions may vary):

```
PyTorch: 2.x.x
CUDA available: True
MONAI: 1.x.x
All imports OK ✓
```

> **Troubleshooting**: If `CUDA available: False`, make sure your NVIDIA drivers and CUDA toolkit are installed correctly. Training will still work on CPU but will be significantly slower.

---

## 5. Project Layout

Before diving into training, familiarize yourself with the project structure:

```
USRegistrator/
├── train.py                  # Main training entry-point
├── models/                   # Registration models (modular)
│   ├── __init__.py           # Registry & re-exports
│   ├── registry.py           # MODEL_REGISTRY, register_model, build_model
│   ├── globalnet3d.py        # GlobalNet3D
│   ├── localnet3d.py         # LocalNet3D
│   └── unetreg3d.py          # UNetReg3D
├── datasets/                 # Dataset generators (modular)
│   ├── __init__.py           # Registry & re-exports
│   ├── registry.py           # DATASET_REGISTRY, register_dataset, build_dataset
│   ├── synthetic_ellipsoids.py  # Ellipsoid generator & MONAI wrapper
│   └── deepreg_synthetic.py     # DeepReg-style DVF generator
├── losses/                   # Loss functions (modular)
│   ├── __init__.py
│   ├── utils.py              # Registry & helpers
│   ├── lncc.py               # Local NCC loss
│   ├── lncc_dvf.py           # LNCC + DVF supervision
│   └── mse.py                # MSE loss
├── metrics/                  # Evaluation metrics (modular)
│   ├── __init__.py
│   ├── ncc.py                # Global NCC metric
│   ├── epe.py                # Endpoint Error
│   ├── regression.py         # MSE / MAE metrics
│   └── smoothness.py         # Gradient smoothness
├── configs/                  # YAML experiment configs
│   ├── config_template.yaml  # Synthetic ellipsoids
│   └── deepreg_synth.yaml    # DeepReg-style DVF
├── docs/                     # Documentation (you are here)
├── requirements.txt
└── README.md
```

---

## Next Steps

Once your environment is ready, proceed to the **[Training Tutorial →](02_training_tutorial.md)** for a complete walkthrough of training a registration model on the synthetic dataset.
