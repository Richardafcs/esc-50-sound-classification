# ESC-50 Foundation Review v2 — Data, Baselines, and Infrastructure

Date: 2026-10-28. Scope: code inspection, saved development evidence and local adversarial tests; **no real-data model training or independent test-set evaluation** in this review.

## Executive decision: gaps were not all closed by Infra Phase 0

Previous Infra Phase 0 added `preflight`, run indexing, comparison and atomic JSON. These established visibility, not proof of good baseline performance or full reproducibility. The next closure gates are deliberately narrower: dataset alignment, baseline evidence, crash-safe checkpoint bytes and orchestration correctness.

## Evidence classification

| Area | Prior state | This iteration | Still open |
|---|---|---|---|
| Metadata | 2,000 entries, 50 labels, official fold checks | Independently verify filename-encoded fold/source/take/target and exact row identity | Original recording quality/near duplicates |
| Split | Source-strict 1,200 / 398 / 400 | Explicit data-role/support contract; no cross-role source overlap | Known Fold 1–5 exposure in historical model development |
| Frontend | Fixed 16 kHz / 512 FFT / 256 hop / 64 log-power Mel, train-only normalization | Preserved exactly; no hidden retraining or feature changes | Comparative 44.1 kHz/16 kHz and feature representation experiments |
| Cache | v2 sidecar hashes cache, frontend and its own metadata records | `preflight` checks **all** cached names, labels, folds, sources, category mapping against raw official metadata; `--quick` cannot approve | Full cached-feature reconstruction requires all raw WAV files |
| Raw WAV provenance | Metadata/rough audio file existence only in preflight | New `prepare` includes per-file WAV hashes; `preflight --verify-audio` validates raw headers and SHA256 | Existing historical cache does not contain all per-file hashes; cannot backfill without WAVs |
| Cache write safety | Direct compressed NPZ destination | Same-directory temporary NPZ + `os.replace`; sidecar remains atomic JSON | Crash between NPZ/sidecar commit means fail-closed mismatch, not all-or-nothing two-file transaction |
| Classical baseline | Logmel mean/std + multinomial logistic | Independent offline hash, optimizer-selection, solver-convergence, 398-row validation audit; additional DCT-logmel cepstral + LinearSVC reference implemented | New SVM has **no real ESC-50 score** until true feature cache exists; not a reproduction of Piczak MFCC/SVM |
| CNN / BiGRU | Historical 60-epoch, multi-seed and fixed-recipe results | Frozen; model math/checkpoints unchanged | Modest accuracy; generalization, pretraining and robustness issues remain |
| Checkpoint durability | `torch.save` directly overwrote `last.pt`/`best.pt` | Atomic same-folder checkpoint replacement, simulated mid-write failure preserves previous bytes | Exact epoch-level resume, optimizer/scheduler/RNG restore and atomic state+history transaction |
| Job Runner | Logging, timeout, process-tree kill | Reap wait after kill no longer reuses an impossibly short grace interval; timeout semantics retained | Resource-aware scheduling and distributed recovery |

## What does `42.21%` actually prove?

The existing historical baseline (`results/historical/final_study/results/statistics_baseline`) is **64-band temporal mean/std + regularized multinomial logistic regression**, not MFCC + SVM. The saved validation metric is 0.42211055276 on **398 validation clips**. The audited search contains three L2 candidates (0.001, 0.01, 0.1), all solver success=True; lowest validation cross-entropy selects 0.001. The model file hash, selected candidate and saved validation predictions agree. **This establishes internal artifact consistency, not independent full training reproduction or state-of-the-art model quality.**

New `cepstral-baseline` uses DCT-II of the project's own log-power Mel bands and temporal mean/std; it trains LinearSVC with C grid (0.01, 0.1, 1, 10), picks validation accuracy with Macro-F1 / smaller-C ties, and records a portable coefficient-only NPZ (no pickle). This is an **MFCC-like, custom frontend cepstral baseline**, not an exact published Piczak MFCC feature implementation. Train-only fit of all standardization statistics. Test evaluation requires a separate explicit command with `--allow-test` and is not a fresh untouched confirmation.

## Remaining scientific constraints

1. The official ESC-50 metadata contains cross-fold source IDs 131943, 134049, 209698 and 234879. Source-strict isolation excludes overlapping identities from train/validation while retaining the test fold; this **is not identical** to standard published four-training-fold ESC-50 benchmarking.
2. Development has already accessed the ESC-50 held-out fold and all five folds through historical exploration. New results here are exploratory or fixed-protocol characterization. Independent confirmation requires new independent data, source-audited and frozen in advance.
3. A complete WAV dataset or original feature cache is absent from this package. Any claims of newly measured ESC-50 performance, full waveform replay, or successful robustness repairs would be unsupported.
4. Existing Log-Mel preprocessing is internally coherent, but this is **not proof of optimal sampling rate, feature representation or calibrated real-world recognition**.
5. Baselines are **good anchors for controlled questions**, not claims of state of the art. Introducing pretraining changes the external data budget and should be a separate study arm.

## Commands

```bash
# Strict check (blocks while cache is absent)
PYTHONPATH=src python -m esc50_sound preflight --root data/ESC-50 --cache data/esc50_features.npz

# Once cache is REGENERATED using updated prepare and all 2000 WAVs available:
PYTHONPATH=src python -m esc50_sound preflight --verify-audio \
  --root data/ESC-50 --cache data/esc50_features.npz

# Independently inspect saved HISTORICAL baseline; only validation predictions read:
PYTHONPATH=src python -m esc50_sound audit-baseline \
  --artifact results/historical/final_study/results/statistics_baseline

# New traditional baseline; requires verified feature cache and optional sklearn extra:
python -m pip install -e '.[baseline]'
PYTHONPATH=src python -m esc50_sound cepstral-baseline \
  --cache data/esc50_features.npz --output runs/cepstral_dev_v1

# Explicit test access only after model selection is fixed; not fresh independent validation:
PYTHONPATH=src python -m esc50_sound cepstral-evaluate \
  --artifact runs/cepstral_dev_v1 --cache data/esc50_features.npz \
  --output runs/cepstral_dev_v1_test --allow-test
```

Do not run `--quick` to approve a training run: it intentionally returns `ready=false` when the cache is not cryptographically checked.

## Next implementation gate (not claimed complete)

**Training Lifecycle v2:** typed versioned config, atomic attempt registration, recoverable epoch commit with model/optimizer/scheduler, Torch/Python/NumPy/DataLoader RNG, and a crash-injection comparison demonstrating resumed vs uninterrupted training. Only after this should scheduling and distributed execution be extended. Model improvements (Attention/PANNs) remain separate from this infrastructure task.

## Local checks performed

- New foundation tests: **9 passed** (real metadata assertions; adversarial label/source checks; provenance WAV tamper; existing historical baseline audit; simulated half-written checkpoint; synthetic linear-SVM fit without test access).
- Non-Runner regression: **167 passed** (isolated process, no real ESC-50 training).
- Runner tests: **16 passed** after correcting termination escalation (isolated process).
- `preflight`: metadata and strict source split passed, **ready=false** because `data/esc50_features.npz` and real WAV collection are missing.
- `audit-baseline`: historical model/selection/validation evidence passed; 398 validation predictions independently counted.

These tests do not establish every combinatorial pipeline behavior or bitwise reproducibility across machines.
