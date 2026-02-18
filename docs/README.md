# USRegistrator Documentation

Welcome to the **USRegistrator** documentation. This folder contains tutorials, reference guides, and how-to documents for working with the USRegistrator 3D medical image registration pipeline.

---

## 📂 Documentation Structure

### [1. Synthetic Training Tutorial](./synthetic-training/)

A step-by-step guide to training a registration model using the built-in synthetic ellipsoid dataset. Start here if you are new to the project.

| Document | Description |
|----------|-------------|
| [Getting Started](./synthetic-training/01_getting_started.md) | Environment setup, dependencies, and first run |
| [Training Tutorial](./synthetic-training/02_training_tutorial.md) | End-to-end walkthrough: config → training → evaluation |
| [Creating Custom Models](./synthetic-training/03_custom_models.md) | How to implement and register your own registration network |
| [Creating Custom Losses](./synthetic-training/04_custom_losses.md) | How to implement and register your own loss functions |
| [Creating Custom Metrics](./synthetic-training/05_custom_metrics.md) | How to implement and register your own evaluation metrics |
| [Configuration Reference](./synthetic-training/06_configuration_reference.md) | Full reference for every YAML config option |

### [2. DeepReg Pipeline Tutorial](./deepreg-pipeline/)

A guide to using the DeepReg-style synthetic DVF (Displacement Vector Field) pipeline for supervised registration training.

| Document | Description |
|----------|-------------|
| [DeepReg Overview](./deepreg-pipeline/01_deepreg_overview.md) | What DeepReg is, how it relates to USRegistrator, and the DVF-supervised approach |
| [Setup & Configuration](./deepreg-pipeline/02_setup_and_configuration.md) | How to configure and run the DeepReg-style pipeline |
| [Running & Evaluating](./deepreg-pipeline/03_running_and_evaluating.md) | Training, monitoring, and interpreting results with DVF supervision |

---

## Quick Links

- **[README (project root)](../README.md)** — Project overview and quick-start
- **[Config Template](../configs/config_template.yaml)** — Basic synthetic config
- **[DeepReg Config](../configs/deepreg_synth.yaml)** — DeepReg-style config
- **[Requirements](../requirements.txt)** — Python dependencies
