# Phase 3 — Research Evaluation & Evidence Infrastructure

**Review date:** 2026-10-28 · **Scope:** Existing ESC-50 research artifacts · **Status:** Implementation verified with existing saved prediction files, no new ESC-50 training.

## Why this phase exists

A finished training run, an `accuracy` field, or an apparently complete CSV is **not** enough to substantiate a research claim. Earlier infrastructure checked training-run manifests and allowed validation comparisons, but did not jointly verify: exact held-out membership, per-file ground-truth labels, metrics re-computation, checkpoint association, paired uncertainty, and full five-fold coverage.

The new `src/esc50_sound/research_eval.py` and `scripts/research_evaluation.py` implement a **read-only** layer on top of the already recorded experiments. They never load or execute pickled PyTorch checkpoints and never train or select models.

## What can be verified

1. **Dataset identity:** protocol includes and enforces SHA-256 of the exact `meta/esc50.csv` bytes.
2. **Prediction validity:** exact three-column CSV schema, no duplicate filenames, correct source-strict/official partition membership, correct targets, integer predicted classes in `[0,49]`, exact expected sample count.
3. **Metrics:** recompute accuracy, macro-F1 and confusion matrix from individual predictions, rejecting inconsistent saved metrics.
4. **Training linkage (optional):** compare declared run manifest ID in evaluation metrics with `run.json`/`manifest.json`, match model/fold/seed and optionally SHA-256 of the actual model checkpoint. No checkpoint deserialization.
5. **Paired model comparison:** align samples by filenames irrespective of CSV row order and calculate difference in accuracy, 2×2 disagreement counts and a seeded `src_file`-cluster bootstrap percentile interval.
6. **Five-fold retrospective audit:** verify exactly five distinct test folds, complete 2,000-clip coverage, one model and seed, same code/cache/frontend/training recipe across folds, all checkpoint bytes and run linkage. Report pooled macro-F1, overall accuracy and descriptive fold variation.
7. **Evidence files:** new reports are published with exclusive-create semantics and will not overwrite earlier audit files.

## Historical results independently checked by this code

**Fold 4 development validation:** historical statistical baseline `validation_predictions_l2_0.1.csv` has 398 valid source-strict records.

**Fold 5 held-out historical comparison (seed 101, *retrospective*, previously accessed):**

| Metric | CNN | CNN + BiGRU |
|---|---:|---:|
| Accuracy | 61.50% | 62.00% |

Paired right-minus-left = +0.50 percentage points. Discordant: only CNN correct = 33; only BiGRU correct = 35; both correct = 213; neither = 119. `src_file` cluster-bootstrap 95% percentile interval: approximately **[-3.54, +4.61] percentage points** (2,000 draws, seed 2026, 310 source groups). Interval includes zero. Conditional on fixed predictions, *not* a training-seed or unseen-data CI.

**Retrospective five-fold fixed-recipe results, seed 101:**

| Metric | CNN | CNN + BiGRU |
|---|---:|---:|
| Accuracy, 2,000 predicted clips | 66.50% | 68.30% |
| Macro-F1 (pooled 2,000 predictions) | 66.02% | 67.79% |

The original protocol trained on 3 folds, used another fold for validation, and tested on the fifth. **Do not call this the official standard four-fold-training ESC-50 benchmark or fresh independent confirmation.** Previous research accessed the same folds before this audit. Source-strict exclusion is deliberate and should be disclosed in every comparison.

## Commands (execute from project root)

```
# Validation-only audit (no test access):
PYTHONPATH=src python scripts/research_evaluation.py audit \
  --protocol configs/eval_development_v1.json \
  --predictions results/historical/final_study/results/statistics_baseline/validation_predictions_l2_0.1.csv \
  --metadata-root data/ESC-50 \
  --output results/my_new_dev_audit.json

# Historical tests need explicit consent; results retain exploratory status.
PYTHONPATH=src python scripts/research_evaluation.py compare \
  --protocol configs/eval_historical_holdout_v1.json --allow-holdout \
  --left results/historical/final_study/results/holdout/cnn_freq_seed101_60ep/test_predictions.csv \
  --right results/historical/final_study/results/holdout/crnn_freq_seed101_60ep/test_predictions.csv \
  --left-metrics results/historical/final_study/results/holdout/cnn_freq_seed101_60ep/test.json \
  --right-metrics results/historical/final_study/results/holdout/crnn_freq_seed101_60ep/test.json \
  --left-run results/historical/final_study/runs/cnn_freq_seed101_60ep \
  --right-run results/historical/final_study/runs/crnn_freq_seed101_60ep \
  --left-checkpoint results/historical/final_study/models/cnn_freq_seed101_60ep.pt \
  --right-checkpoint results/historical/final_study/models/crnn_freq_seed101_60ep.pt \
  --output results/my_new_historical_pair.json

PYTHONPATH=src python scripts/research_evaluation.py summarize-cv \
  --manifest configs/eval_historical_cv_cnn_freq_seed101.json \
  --allow-holdout --metadata-root data/ESC-50 \
  --output results/my_new_cnn_cv_audit.json
```

For new studies, create an explicit versioned protocol JSON with the **new metadata SHA-256 and honest selection disclosure**; never retrofit an 'independent_confirmation' label to past outcomes. The local evaluator intentionally rejects attempts to self-certify independent confirmation via JSON.

## Verification and failure tests

- 15 new targeted tests passed: label corruption, duplicate/missing filenames, extra columns, out-of-range class, invalid metrics, wrong metadata hash, holdout consent, forged manifest ID, duplicate CV fold, paired row-order independence, deterministic bootstrap, and no-overwrite evidence.
- 169 non-runner/core tests passed (includes 15 above).
- 21 Pipeline/Study/Resume tests passed.
- 27 Runner/Scheduler tests passed.
- **217 unique tests passed across three independently executed groups.**
- **49 pre-existing `.pt` weights byte-for-byte identical** to the prior Scheduler Full project.
- Real WAV and feature cache still unavailable in this package; training was not rerun, and saved predictions were **not** regenerated by re-running inference.

## Limitations and next work

1. CSV and training-record integrity do not prove that the checkpoint originally generated the stored predictions; that needs independent re-inference from raw WAV, with the exact model code, preprocessing and environment.
2. A file named `test` does not become independent by being declared 'frozen'. External evidence of untouched evaluation and a separate data source are needed for truly independent confirmation.
3. The cluster bootstrap is conditional on fixed model predictions; to quantify training variation, repeat seeds within a predeclared protocol. Overlapping training folds invalidate naive independence assumptions across five folds.
4. `src_file` grouping addresses known source IDs but cannot detect different IDs hiding near-duplicate audio. Perform WAV-level near-duplicate and hash checks once raw audio is available.
5. Compatibility checks compare declared provenance, not execution equivalence on different hardware; training seeds and environment must be recorded and monitored.
6. Future evaluation API should add a first-class model release registry, a real data/label lineage store, external holdout governance, signed artifact provenance if threat model requires it, and a multi-model study manifest for automatic cross-seed reporting.
