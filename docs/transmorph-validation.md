# Original TransMorph integration

`transmorph_original3d` adapts the [original TransMorph repository](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
to USRegistrator's `(moving, fixed) -> (warped, ddf)` interface.
The separate registry name allows other implementations to coexist.

## Install and run

Use Python 3.12 and a compatible torch/torchvision pair. The local CPU target is
torch 2.8.0 / torchvision 0.23.0; install a CUDA-compatible pair for GPU runs.
Install core dependencies and the two optional dependencies before constructing Tiny:

```bash
python -m pip install -r requirements.txt -r requirements-transmorph.txt
git clone https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git .third_party/TransMorph
git -C .third_party/TransMorph checkout --detach 6357a1d7fc44c36db9b1d1ccaa372409253142cf
python scripts/check_transmorph_source.py
```

The source is not vendored. The loader checks normalized SHA-256 hashes of the
two executed upstream Python files against commit `6357a1d7fc44c36db9b1d1ccaa372409253142cf`
before loading, including cached constructions. CRLF/LF conversion is accepted,
code changes are rejected. Source hashes do not pin the entire runtime environment.
Retain upstream licence terms. Use `transmorph_root` in model config or
`USREGISTRATOR_TRANSMORPH_ROOT` for another checkout; it must point to the
`TransMorph` subdirectory containing `models/TransMorph.py`.

External dependencies/imports are lazy. The loader restores the temporary
`models.configs_TransMorph` alias and does not replace this project's `models`
package or modify `sys.path`. It exposes Tiny/Small/Base factories.
Only the Tiny variant at 64³ received real-model validation.

## Current CLI limitations

`configs/deepreg_synth_transmorph_original_tiny.yaml` is a configuration template,
not a validated end-to-end training example. At base
`b6645d55bdd65863ab60f7c14c67c61f1d3ae17e`, an indentation error in
`datasets/deepreg_synthetic.py` prevents dataset imports, and `main.py` imports
the absent `train.run_training`. Neither issue is fixed here or by #19.
The focused model tests do not import the dataset or training modules and
require no temporary shared fixes.

## Focused scope and validation

Install pytest separately if needed (`pytest==8.4.2` was used), then run:

```bash
python -m pytest tests/test_transmorph_adapter.py tests/test_transmorph_original_model.py -q
```

CPU validation on 2026-09-13: **17 passed, 0 failed, 0 skipped**, with three
upstream/dependency warnings retained. The source/hash checker returned **PASS**
and reported 244,527 Tiny parameters. No temporary shared fixes were applied.

The nine tests in `test_transmorph_adapter.py` are retained unchanged from #18.
They cover registration, source verification, optional-import isolation and
adapter input/output behavior.

`test_transmorph_original_model.py` adds eight model-specific cases: one real Tiny
forward smoke (parameter count, shapes and finite outputs), one strict state-dict
save/reload check, and six signed original-warper probes.

Reload uses a randomly initialized eval model and compares its outputs after
in-memory serialization. It does not validate optimizer/scaler or full training
resume, or compatibility with pretrained or historical weights.
The warper probes inject constant `-1/+1` voxel fields on z/y/x and compare
interior voxels with analytic shifts. They do not validate spatially varying
fields, boundaries, learned displacement semantics or equivalence to another model.

The real-source tests skip when optional source or packages are absent; hash
mismatches and incompatible installed dependencies fail. A run with skipped
real-source tests does not establish that Tiny completed a forward pass.

The shared registry-driven contract/conformance framework remains in #15.
These tests are specific to Original TransMorph. The real Tiny forward check is
an integration smoke, not automatic contract coverage across registered models;
state-dict reload and signed original-warper probes are model-specific checks.

These results do not establish full training, GPU, convergence, performance,
framework-wide, real-data or clinical validation. Earlier pipeline and cross-test
results are not validation of this PR.

### Merge prerequisites

This PR can be reviewed independently, but it should be merged only after the
known shared repository blockers are resolved and #19 is merged.

The synthetic DDF scaling and TRE fixes are handled separately in #19.

Recommended order: shared repository fixes, then #19, then this PR.
