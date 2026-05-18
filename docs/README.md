# USRegistrator Documentation

Welcome to the **USRegistrator** documentation. This folder contains tutorials, reference guides, and how-to documents for working with the USRegistrator 3D medical image registration pipeline.

---

## Choose Your Path

| If you want to… | Start here |
|-----------------|------------|
| Learn the pipeline on synthetic data | [Synthetic Training Tutorial](./synthetic-training/) |
| Train with DeepReg-style DVF supervision | [DeepReg Pipeline Tutorial](./deepreg-pipeline/) |
| Train on your own medical dataset via JSON manifest | [Custom Dataset Training](./custom-training/) |

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

### [3. Custom Dataset Training](./custom-training/)

Train and validate using your own data by setting `name: custom_dataset` and providing `train`/`val` entries in a JSON manifest.

| Document | Description |
|----------|-------------|
| [Custom Dataset Guide](./custom-training/custom_dataset.md) | Manifest schema, multigradicon preprocessing, YAML setup, training, and troubleshooting |
| [TRUSTED Manifest Example](../configs/trusted_manifest.json) | Example `train`/`val` JSON manifest used by custom dataset configs |

Custom-dataset-specific options (`json_file`, `preprocess_style`, `ct_window`, `quantile_range`, etc.) are documented in the custom dataset guide. Shared model/loss/optimizer/training options are documented in [Configuration Reference](./synthetic-training/06_configuration_reference.md).

---

## Quick Links

- **[README (project root)](../README.md)** — Project overview and quick-start
- **[Docs Hub](./README.md)** — Overview of all documentation tracks
- **[Config Template](../configs/config_template.yaml)** — Basic synthetic config
- **[DeepReg Config](../configs/deepreg_synth.yaml)** — DeepReg-style config
- **[Custom Dataset Example Config](../configs/custom_dataset_example.yaml)** — Example config for JSON-manifest training
- **[TRUSTED Manifest Example](../configs/trusted_manifest.json)** — Example manifest format
- **[Requirements](../requirements.txt)** — Python dependencies
