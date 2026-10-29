# Independent ESC-50 Code and Evidence Audit — 2026-10-28

Audit basis: current original 846-entry project ZIP, Phase 1 Attention updated
package, direct inspection of `src/esc50_sound`, config/CLI/test source, and
read-only execution of new infra commands. No pretrained/research model was
fit to ESC-50 in this audit. No test predictions were opened by new tools.

## Verified by direct execution in this phase

- Official `data/ESC-50/meta/esc50.csv` is present; raw WAV collection incomplete.
- Source ID collisions across official folds: `131943`, `134049`, `209698`,
  `234879` in this metadata. Source-strict train/validation/test counts under
  folds 1–3 / 4 / 5: **1,200 / 398 / 400**, source-disjoint.
- `data/esc50_features.npz` and `.npz.json` missing from current package.
  `preflight` blocks real training on this environment and reports why.
- `results/historical/final_study/runs` has **21** run.json files, all claim completed
  60-epoch runs, and all pass the read-only internal manifest/history checks.
  This does not constitute re-execution or independently validate predictions.
- `runs/` contains four older self-contained run.json files with **no manifest**.
  New run indexing explicitly marks their provenance incomplete.
- Historical records occur in multiple `results/` subdirectories, including
  copied snapshots. Manifest identity duplicate warnings prevent naive counts.
- No real-data Attention classification results exist; three Attention
  architectures passed synthetic integration tests in the previous package.

## Code architecture: strengths

1. Strict cache sidecar, frontend and shape checks; training-only mel stats.
2. Source-based split repair and explicit test authorization in CLI.
3. Stable shared CNN encoder and isolated heads/attention extensions.
4. History, best/last checkpoints, SHA256 code/cache fingerprints and selected
   validation metrics; simple packaging/inference with explicit model cards.
5. Runner subprocess watchdog and experimental logs already exist; do not
   replace these without compatibility tests.

## Risks and unresolved questions (severity)

**P0 / scientific validity:** Folds 1–4 and Fold 5 have been used in prior
analysis. A new accuracy increase on these folds is exploratory, not a fresh
unbiased confirmation. The 3-fold-training source-strict protocol should not
be described as the published official 4-fold-training benchmark. Audio
near-duplicate and independent pretraining dataset overlap audits remain open.

**P0 / execution blocker:** Full WAV data and feature cache are unavailable
here. No scientific training may be asserted without acquiring and auditing
real samples. Synthetic integration tests are explicitly non-performance tests.

**P1 / lifecycle:** `training.py` cannot exactly resume a stopped run and the
existing completion status does not prove checkpoint-vs-code numerical replay.
`write_json()` currently writes directly rather than atomically. Out-of-space
or process interruption could leave incomplete JSON. Address in Phase 1.

**P1 / comparisons:** Existing records are scattered; duplicate copies of
runs risk pseudo-replication; comparing runs across code/cache/split requires a
hard gate. Added conservative validation-only gate in Phase 0.

**P1 / runtime:** Current source lists broad dependency ranges and does not
pin byte-exact platform environments. Preflight flags missing cache but does
not provision a CPU/GPU or guarantee deterministic execution.

**P2 / performance:** Existing CNN/GRU results show only a small recurrent
mean improvement; serious synthetic noise degradation; Attention benefit has
not been established on ESC-50. Lack of confidence calibration and real-world
independent robustness evidence prevents deployment claims.

**P2 / maintainability:** Several historical root scripts, Colab copies, and
exported snapshots overlap with the canonical `src/` package. Preserve them
for provenance, but keep new code changes in `src/esc50_sound/` and tests.

## Do not infer from this audit

- The 21 prior model fits have *not* been rerun.
- Hashing a manifest does not independently verify the behavior of historical
  software during its run.
- A test passing on synthetic inputs does not provide an ESC-50 accuracy.
- A matching sample seed does not guarantee bitwise determinism across hardware.
- A duplicate manifest ID is not automatically an independently repeated run.

### Infra regression-test observation

A previously existing runner test used a 1-second child deadline and failed
intermittently when this shared CPU environment delayed Python startup; its
log recorded `status=timeout` and `exit_code=-15`, not a model or data error.
The test now uses a 6-second child deadline (10-second global allowance)
while preserving the same registration-before-child and success assertions.
This change is to a test fixture, not the production task runner's timeouts.
An explicit four-step synthetic smoke was too short to satisfy its pre-existing
loss-improvement criterion. The original 40-step smoke passed with decreased
loss in both CNN and CRNN. Neither is a real ESC-50 performance estimate.
