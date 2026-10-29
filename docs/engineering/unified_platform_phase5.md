# Phase 5 — Contract-first unified research platform (v1)

**Date:** 2026-10-28  
**Status:** Implemented and tested on CPU with synthetic feature data; **no real ESC-50 training or new published model performance claim**.

## Goal and deliberately limited scope

The previously validated training, epoch resume, scheduler, evaluation, and registry were coupled to `load_cache()` and `build_model()`. Rather than rewrite these proven engines and invalidate old run/checkpoint semantics, this release introduces a new **opt-in adapter and experiment specification layer**. Runs initiated using `research_platform.py` travel through the adapters **inside the real training function**; legacy CLI runs continue through their old construction paths. The backend route is recorded in the new run's frozen `manifest.json`.

Implemented:

1. `ModelCatalog` / `ModelAdapter` with explicit capability declarations and a strict batched logit contract. All ten original model names resolve to their unmodified constructors. No JSON-driven arbitrary imports and no implicit model downloads.
2. `FeatureCacheBackend` reads validated NPZ + sidecar, compares *every row* against authoritative official Metadata, and applies strict `src_file` split membership. Accessing held-out feature rows requires `allow_test=True`.
3. `RawAudioBackend` is **inspect-only**. It lazily reads single official WAV files, validates 44.1 kHz / mono / 5 s headers, generates the same 16 kHz log-mel representation and returns raw SHA-256. It intentionally cannot train yet because streaming normalization/shuffling/resume need their own reproducibility contract. Held-out WAV reads require explicit authorization.
4. A frozen `TrainingPlan` JSON contract validates all fields, fold roles, model capabilities, available device and output collisions. The cache is checked before any writes or training. The plan SHA-256 and source Metadata root are included in the new run's configuration and checkpoint provenance. `verify-run` joins a frozen plan and its run record without evaluating held-out predictions.
5. `scripts/research_platform.py` commands: `catalog`, `check`, `run --execute`, `verify-run`. `run` **requires explicit `--execute`**; the default is read-only checking. If preflight fails, execution fails closed.

Not yet implemented: PANNs, AST, CLAP, pretrained fine-tuning, zero-shot inference adapters, streaming WAV training, pluggable cross-domain normalization, remote workers, automated research approval, signed identities or production deployment. Do not label this release a fully general ML platform.

## Architecture

```
Frozen versioned experiment plan (JSON, strict keys)
                 |
                 v
      Platform contract / capability gate
        /                      \
Dataset Backend              Model Adapter
(cache + metadata audit)     (registered local model)
        \                      /
           Verified train-only plan
                    |
             existing training.py
       (optimizer/scheduler/resume/manifest)
                    |
   existing scheduler, evaluation, model registry
```

Adapters are intentionally small and have no network side effects. The training engine remains the authority for checkpoint selection, random state, test prohibition and committed-epoch recovery.

## Data and model interfaces

`InputContract`: modality `log_mel`, per-sample shape `[1,64,313]`, dtype `float32`, hash of the fixed `FRONTEND`. A model's `infer_logits` requires finite `[B,50]` logits. `AudioExample` contains `filename`, `fold`, `target`, `src_file`, and the feature tensor (with raw SHA-256 for the WAV backend).

`FeatureCacheBackend.get('test', idx)` and `RawAudioBackend.get('test', idx)` reject access unless explicitly passed `allow_test=True`. **This API guard is not an operating system security boundary**: an authorized developer can still open files manually, and the existing checkpoint evaluation CLI retains its own separate guard.

The training plan currently supports only `dataset_backend='feature_cache'` and registered `supervised_train` models. A zero-shot model is **not** silently forced into a supervised training interface. Adapter registration exists as an extension point, but new pretrained models require their own data/licensing/checkpoint/protocol implementation and tests before registration.

## Usage (from project root)

```bash
PYTHONPATH=src python scripts/research_platform.py catalog
PYTHONPATH=src python scripts/research_platform.py check --plan examples/platform_cnn_cpu.json
# Only after acquiring the real data and validating the plan:
PYTHONPATH=src python scripts/research_platform.py run --plan examples/platform_cnn_cpu.json --execute
PYTHONPATH=src python scripts/research_platform.py verify-run \
  --plan examples/platform_cnn_cpu.json --run-dir runs/platform_cnn_cpu_new
```

The checked-in example uses 60 epochs and the strict (1,2,3) / 4 / 5 source-excluded development split. No test evaluation is performed by this interface. `check` exits with code 2 until the real features and sidecar exist. Do not run the example into a historical Run directory; the training engine refuses nonempty destinations.

For interrupted runs, continue to use the previously tested `resume-train` command (or Phase 2 scheduler), with the same cached data and exact runtime environment. `platform_plan_sha256` is part of a new run's committed configuration and is retained by resume.

## Verification evidence

- Synthetic-data end-to-end real subprocess (CPU): produced `run.json`, `manifest.json`, `history.json`, `last.pt`, `best.pt` and status `completed` for 1 epoch; run identity includes `platform_plan_sha256`, `platform_metadata_root` and `platform_dataset_backend`. The synthetic dataset is a shape-compatible NPZ with artificial signals and official Metadata: **its accuracy is not a sound classification result**.
- The new `verify-run` read the generated run and confirmed it was bound to the frozen plan; held-out predictions were not read or created.
- Fault-injection and contract tests cover missing cache, unknown model, mode mismatch, schema errors, invalid fold overlap, stale output, unauthorized held-out reads, invalid WAV headers and malformed model inputs.
- Existing model constructors produce the same state dict under the same seed. The 49 archived `.pt` files are copied unchanged.
- 2026-10-28 regression evidence: see `results/historical/infra_v5/` for test group logs. These tests do **not** establish CPU/GPU training determinism across different PyTorch/platform versions.

## Quality gates that remain open

- The user-provided package contains Metadata, but not the complete original 2,000 ESC-50 audio WAVs or real feature cache. Full WAV vs cache recreation/near-duplicate comparison and real five-fold training **cannot be verified here**.
- The earlier Fold 5 and other fold results were previously consulted during research. New experiments must not claim untouched independent confirmation. The strict source exclusion protocol is also not identical to the official 4-fold-training Benchmark.
- MFCC/SVM with real audio and modern pretrained baselines are still pending. The new platform does not manufacture stronger accuracy results.
- Registry `validated` means technical integrity, not model quality, independent scientific verification or deployment clearance.

## Next development iteration

1. Introduce an input-transform adapter for raw waveform pretrained encoders, with explicit per-checkpoint sample rate, windowing and license/data source lineage.
2. Add `FrozenEncoder + Head` and `FineTuneEncoder` adapters that can negotiate optimization groups and checkpointing without altering legacy runs.
3. Add an independent `ZeroShotAudioText` inference protocol; never route it through supervised `train()`.
4. Add real-data conformance tests for cache-vs-raw sample parity, with training-only normalizer isolation and source overlap audit.
5. Integrate plan lineage into the Phase 2 scheduler and Phase 3 evaluation report once those separate contracts are tested.
