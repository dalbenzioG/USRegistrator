# Experiment history: original TransMorph integration

This curated history records decisions and meaningful failures, not every repeated
run or private conversation. It accompanies a focused original-TransMorph PR
against USRegistrator base `b6645d55bdd65863ab60f7c14c67c61f1d3ae17e`.

| Stage | Finding / trial | Decision and evidence boundary |
| --- | --- | --- |
| Original exploration | Explore synthetic USRegistrator training and the original external TransMorph implementation. | Integration is the deliverable; no real patient data or clinical claim. |
| Historical continuation | The notebook describes LocalNet/Tiny, seeds 42/43/44, 15 epochs and 128/32 synthetic cases. | Its six-run checkpoint archive was not recovered. Do not present it as independently replayed or replace it with new smoke weights. |
| Geometry audit | DDF scale/direction, landmarks, TRE and the training entrypoint needed corrections. | Retain the corrections needed to train and evaluate the supervised integration example. Analytic tests cover identity, translation, subvoxel sampling and generated landmark consistency. |
| Broader reproducibility work | Indexed samples, RNG isolation, caching, split-range checks and independent best-mTRE checkpointing were explored. | Preserve that work separately; it is deliberately excluded from this focused PR. Streaming behaviour and original checkpoint selection remain. |
| Earlier GPU environment | The installed PyTorch build did not support P100's GPU architecture; a T4 run succeeded. | Record as an environment incompatibility, not an algorithm failure. Use a compatible GPU/runtime. |
| Earlier combined validation | A broader revision passed 42 tests and LocalNet/Tiny CUDA/AMP smokes. A packaging test first failed because an exported ZIP lacks Git metadata; its fixture was corrected. | Retained historical proof, not automatic evidence for this newly reduced branch. Notebook-packaging machinery is not included in this focused PR. |
| PR #17 review | At `5dcd58918cc51e271169f6df48ed08453c6f26a1`, #17 implements a MONAI SwinUNETR-based model. | Use `transmorph_original3d` for the original external model. Credit spellsharp's overlapping indentation, TRE float32 and entrypoint repairs; coordinate shared-file merges. No architecture ranking or priority claim. |
| Focused first PR | Make original TransMorph the first deliverable, not a follow-up behind general benchmark-policy changes. | Branch directly from upstream, retain only necessary shared fixes, add specific tests and a runnable example. Revalidate this exact reduced code independently. |

## What the focused tests establish

- Optional dependency isolation, source pin checking, import-alias restoration,
  registration and correct `(moving,fixed)` input order.
- Fixed-grid voxel geometry, analytic translation/subvoxel TRE, zero velocity,
  landmark correspondence, half-precision TRE and unchanged streaming behaviour.
- Both existing training CLIs import and route to the shared callable entrypoint.
- The integration runner exercises the actual original Tiny network and a small
  LocalNet compatibility configuration; finite/nonzero gradients, training,
  checkpoint creation and strict state restoration are checked.

The roundtrip comparison uses the same captured fixture cases before and after
serialization. It does not replay streaming training-validation draws and must
not be described as reproducing their saved EPE/mTRE values. Those saved metrics
are checked for finiteness and reported separately. The runner retains the
configuration, environment, logs and hashes for each model.

## Evidence status

Focused-branch final CPU/GPU results will be recorded only after execution and
artifact verification. Older 42-test results belong to the broader branch, not
to this PR. Full proof artifacts, notebooks and weights remain outside the source
diff. No credentials or private messages are included.

Historical weights are unnecessary for a software-integration PR, but necessary
for reevaluating those exact trained models. Resuming an old training run also
requires its optimizer/scaler/epoch/RNG state. This work does not establish old
checkpoint compatibility, Small/Base support in practice, or comparative accuracy.
