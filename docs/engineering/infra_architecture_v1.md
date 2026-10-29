# ESC-50 Research Infrastructure — Architecture Decision Record v1

Date: 2026-10-28. Status: **Phase 0 implemented and locally checked; later phases not implemented.**

## Mission and non-goals

Build a reproducible, auditable *research* platform where datasets, protocol, model,
execution, metrics, selected checkpoint, and reports have explicit provenance.
Do not claim an ordinary supervised classifier is a deployed general-purpose
sound-event recognizer. Do not use this infrastructure to secretly tune against
the already accessed ESC-50 holdout folds. No distributed cluster, service API,
MLflow server, experiment autoscheduler, or exact resume is implemented in v1.

## Present components and boundaries

```
Dataset metadata / original WAVs     Experiment protocol / model choice
       |                                          |
       v                                          v
source-group audit + pinned data       validated CLI / preflight gate
       |                                          |
       v                                          v
 deterministic features / sidecar ---> split + train-only normalization
                                                 |
                                              train
                                                 |
                            run.json + manifest.json + history.json
                                                 |
                                  selected best.pt + validation metrics
                                                 |
                                  read-only run registry and comparison
                                                 |
                            explicit authorized held-out evaluation
                                                 |
                              bounded reports / portable artifact
```

### Ownership contracts

| Owner | Current implementation | Required invariant |
|---|---|---|
| Dataset | `data.py` | No source-ID overlap under strict split; cache sha and frontend match |
| Frontend | `audio.py` | One input representation, documented STFT shape `[1,64,313]` |
| Models | `models.py` | `forward(x)` returns logits `[B,50]`; model names stable |
| Training | `training.py` | No test labels in selection; record actual epochs/updates |
| Records | `records.py` | Code/cache/config/seed fingerprints, no silent historical replacement |
| Evaluation | `evaluation.py` | `--allow-test` required; test access logged |
| Infrastructure v1 | `infra.py` | Read-only, no checkpoint deserialization or test metrics |
| Orchestration | root `runner.py` | Bounded commands and logs; no autonomous provisioner |
| Artifacts | `artifacts.py`, `inference.py` | Manifest/source policy and model card accompany deployment |

### Run lifecycle

`initializing → running → completed | time_limit | failed`; `test_evaluated` is a
separate historical access flag, **not** a model-quality status. A run with
`time_limit` or a partial final epoch is not eligible for equal-epoch comparison.
`best.pt` is selected using minimum validation cross-entropy, not test accuracy.

## Phase 0 commands (implemented)

Readiness check; checks source grouping and cache checksum without fetching data:

```bash
PYTHONPATH=src python -m esc50_sound preflight --root data/ESC-50 --cache data/esc50_features.npz --output results/infra/preflight.json
```

The command exits `2` when data are not ready. Missing WAV is a warning if a
verified cache is present; absence of a cache blocks training. `--quick` skips
cache digest checks and must not be used as the final training gate.

Discover historical records, without opening test predictions or checkpoints:

```bash
PYTHONPATH=src python -m esc50_sound index-runs --root results/historical/final_study/runs --output results/infra/final_index.json --csv results/infra/final_index.csv
```

Compare two completed **validation** runs:

```bash
PYTHONPATH=src python -m esc50_sound compare-runs results/historical/final_study/runs/cnn_freq_seed101_60ep results/historical/final_study/runs/crnn_freq_seed101_60ep
```

`compare-runs` refuses a green approval when training protocol, code hash,
frontend, train/validation/test membership, or cache checksum are unavailable
or inconsistent. Identical recorded protocol does not establish causal
attribution, calibration, or an untouched test. Different model names and seeds
are allowed by design.

The index includes `duplicate_manifest_groups` to reveal copied or potentially
repeated experiment identities. Two paths with the same manifest are **not**
necessarily two independent runs. Check actual history/checkpoint identity
before deduplicating or aggregating.

## Research data policy

- `data/ESC-50/meta/esc50.csv` is not the audio data. Audio downloads are
  separate and should be pinned, verified, and covered by ESC-50's license.
- Fit normalization on the *actual training set* only. Do not tune model or
  policy on the held-out test.
- The source-strict 3-train + 1-validation + 1-test protocol intentionally
  differs from the conventional four-training-fold published benchmark.
- Fold 5 and other folds have already influenced historical exploration;
  subsequent comparisons are **exploratory** unless an independent, source-audited
  external test set has been secured and frozen beforehand.
- ML weights/checkpoint provenance must be treated as claims backed by files,
  not cryptographic proof of which inputs were used during an earlier run.

## Engineering phases and acceptance gates

**Phase 0 (this package): Baseline audit + preflight + run index.**

- [x] Inspect actual code, runs, predictions availability, provenance and split.
- [x] Implement read-only readiness and consistency checks.
- [x] Implement run registry with duplicate identity warning and CSV export.
- [x] Implement conservative validation-comparison gate and unit tests.
- [ ] Obtain official WAVs/cache and verify a full real-data execution.

**Phase 1: Core runtime hardening.** Not yet implemented.

- Atomic run reservation and writes; status transitions and crash-recovery tests.
- Explicit typed versioned experiment config and dry-run plan, separate CLI.
- Exact epoch resume including optimizer, scheduler, RNG, DataLoader generator;
  timeouts mid-epoch must have clearly defined replay semantics.
- Dependency lock for each target CPU/CUDA platform; reproducibility mode,
  seed and system detail including execution constraints.

**Phase 2: Tracking / scheduling / CI.** Partially implemented previously.

- Compare and query versioned experiment families, not only two run directories.
- Opt-in local MLflow backend for cross-run UI and model/artifact registry;
  MLflow should mirror signed-off local records, never replace raw evidence.
- CI smoke gates across data, model, train, evaluation, packing, and failure.
- Runner scheduling based on CPU/RAM/GPU budget, retry policies that do not
  overwrite failed attempts, and recovery from interrupted jobs.

**Phase 3: Evaluation / statistics / model lifecycle.** Not yet implemented.

- Predeclare split roles and model-selection boundaries; nested CV where needed.
- Pairwise predictions and uncertainty that respects source dependence.
- Noise/gain/calibration/OOD robustness as distinct protocols.
- Model cards, test access audit, thresholded promotion gate and artifact hashes.

**Phase 4: Multi-backend infrastructure.** Only if experiment scale requires it.

- Explicit adapters for PANNs/AST/CLAP, model-specific sample rates/frontend.
- GPU worker or managed-job queue, artifact/object store, caching policy and
  resource benchmarking. Avoid Kubernetes until reproducibility and scale
  justify maintenance overhead.

## Design choices and rationale

Use the standard library + current JSON record format first. The project
already records checkpoints/metrics and has a bounded subprocess runner; adding
MLflow, Hydra, DVC, Docker, a database and cloud scheduling simultaneously
would multiply points of failure without fixing missing input data or previous
test exposure. MLflow is a later optional interface when the run schema and
model evaluation protocols are stable. PyTorch explicitly warns that exact
reproducibility across versions, platforms, and CPU/GPU is not guaranteed,
so provenance/independent replication matter beyond `seed_all()`.

## Reference specifications

- ESC-50 official dataset and source folds: https://github.com/karolpiczak/ESC-50
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- MLflow tracking concepts: https://mlflow.org/docs/latest/ml/tracking/

### Phase 0.1 hardening completed

`records.write_json()` now serializes before touching the destination and
atomically replaces a same-directory temporary file after fsync. Failed JSON
serialization leaves prior contents unchanged. This protects run-history
readers from torn JSON writes but does not implement multi-host transactions
or restore optimizer state after a crash.

A small GitHub Actions CPU workflow runs non-runner and subprocess runner tests
separately; it deliberately avoids downloading ESC-50 audio or evaluating a
held-out test. Local validation passes do not imply that CI has been run on
GitHub.
