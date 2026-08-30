# Original TransMorph integration

`transmorph_original3d` adapts the [original TransMorph repository](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
to USRegistrator's `(moving, fixed) -> (warped, ddf)` interface. It is **not** the
MONAI SwinUNETR-based TransMorph-style model in [PR #17](https://github.com/dalbenzioG/USRegistrator/pull/17).
The separate registry name allows both implementations to coexist; shared registry
and training-file changes still require merge coordination. Neither is claimed superior.

## Install and run

Use Python 3.12 and a compatible torch/torchvision pair. The local CPU target is
torch 2.8.0 / torchvision 0.23.0; install a CUDA-compatible pair for GPU runs.
Install core dependencies and the two optional dependencies before constructing Tiny:

```bash
python -m pip install -r requirements-validation.txt -r requirements-transmorph.txt
git clone https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git .third_party/TransMorph
git -C .third_party/TransMorph checkout --detach 6357a1d7fc44c36db9b1d1ccaa372409253142cf
python scripts/check_transmorph_source.py
python main.py --config configs/deepreg_synth_transmorph_original_tiny.yaml
```

The example uses one epoch, 4/2 synthetic cases at 64 cubed, batch size 2, model
seed 42, dataset seeds 123/456 and W&B disabled. CUDA/AMP is selected when CUDA is
available; otherwise training uses CPU. Choose a fresh `training.save_dir` to
retain earlier checkpoints. `python train.py --config ...` is also supported.

The two-step smoke explicitly uses `training.amp_init_scale: 128.0`. Initial
calibration at PyTorch's larger default can skip both steps because of nonfinite
scaled gradients. Omitting this optional setting preserves PyTorch's default;
existing configurations are unchanged. The runner requires an actual optimizer
update and records the optimizer step and scaler state, not just a completed epoch.

The source is not vendored. The loader checks normalized SHA-256 hashes of the
two executed upstream Python files against commit `6357a1d7fc44c36db9b1d1ccaa372409253142cf`
before loading, including cached constructions. CRLF/LF conversion is accepted,
code changes are rejected. Source hashes do not pin the entire runtime environment.
Retain upstream licence terms. Use `transmorph_root` in model config or
`USREGISTRATOR_TRANSMORPH_ROOT` for another checkout; it must point to the
`TransMorph` subdirectory containing `models/TransMorph.py`.

External dependencies/imports are lazy. The loader restores the temporary
`models.configs_TransMorph` alias and does not replace this project's `models`
package or modify `sys.path`. It exposes Tiny/Small/Base factories, but only
**Tiny at 64 cubed** has integration validation. No pretrained or historical
checkpoint compatibility is established.

## Necessary shared fixes

The supervised example uses `deepreg_synthetic`, so this PR also fixes:

- Its import-blocking indentation and normalized-displacement scale.
- Image/landmark geometry: voxel-unit `(z,y,x)` fields live on the fixed grid;
  `warped(x_fixed) = moving(x_fixed + ddf(x_fixed))`. Corresponding landmarks use
  that same fixed-grid field and trilinear interpolation with `align_corners=True`.
- Synthesis/registration directions: integrate opposite stationary velocities
  with seven scaling-and-squaring steps instead of negating a displacement.
  Discrete inverse consistency is approximate, not a guarantee of fold-free fields.
- TRE evaluates `fixed + ddf(fixed)` against moving landmarks in float32, including
  half-precision model output. Units are voxels, not millimetres.
- The example's `lncc_dvf` loss calculates LNCC local statistics and supervised
  MSE in float32 outside autocast; the network still runs with AMP. This avoids
  low-precision cancellation in the local variance/correlation calculation.
- `train.run_training(config_path)` makes the existing `main.py` entrypoint usable.

These corrections affect the shared synthetic task and TRE, not only TransMorph.
Old scores cannot be compared directly with corrected scores. Original TransMorph's
warper uses zero padding; the synthetic generator uses border padding. The smoke
compares sampler outputs with matching padding, not an assumed identical border policy.

## Focused scope and validation

This PR deliberately retains upstream **streaming** dataset access, RNG behaviour
and three-checkpoint policy (`best_epe.pt`, `best_val_loss.pt`, `last.pt`). It does
not add indexed samples, caching, seed-range rules, best-mTRE selection or change
existing full-size configs. Those broader experiments are separate work.

```bash
python -m pytest tests -q --junitxml validation-tests.xml
python scripts/validate_transmorph.py --device cpu --output-dir validation_outputs/cpu-new
python scripts/validate_transmorph.py --device cuda --output-dir validation_outputs/gpu-new
```

The runner checks original-model output geometry against MONAI's sampler, finite
nonzero gradients, one-epoch training via the real CLI, optimizer steps, three
checkpoints and strict loading. It then serializes/reloads the trained state and
compares loss/EPE/mTRE on the **same captured fixture cases**.
These captured cases do not alter production dataset policy. The runner's
deterministic cuDNN selection only controls its in-process checks, not the separate
training subprocess; no bitwise CUDA guarantee is made. Because production
validation streams new draws, this is **not** an exact replay of the metrics
recorded during training. Both metric sets are labelled separately in `summary.json`.

The LocalNet option is a small compatibility smoke, not a matched architecture
comparison. Warnings from optional packages are retained with the logs. No full
benchmark, convergence, real-data or clinical claim is made. CI runs only after
publication. See [experiment history](experiment-history.md) for scope decisions
and revision-specific evidence, including earlier work that is not in this PR.
