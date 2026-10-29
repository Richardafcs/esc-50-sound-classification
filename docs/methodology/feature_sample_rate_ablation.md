# ESC-50 Feature and Sampling-Rate Ablations — Research Protocol v1

**Status:** implementation and synthetic/unit tests only. Real ESC-50 experiments have **not** been run. All metrics come from *development validation* unless explicitly described otherwise.

## Scope and experimental controls

This is a *separate* lightweight research track; it **does not alter historical CNN/GRU checkpoints**, 16k/64-Mel feature caches, model registry, resume mechanism or historical reports.

Compare four feature representations under three frequency/sample conditions: **12 cases**.

| Condition | Sample rate | Effective analysis bandwidth | STFT FFT | Hop |
|---|---:|---:|---:|---:|
| SR16-F8 | 16 kHz | 0–8 kHz | 512 | 256 |
| SR32-F8 | 32 kHz | 0–8 kHz | 1024 | 512 |
| SR32-F16 | 32 kHz | 0–16 kHz | 1024 | 512 |

All STFT windows are **32 ms**, hops **16 ms**, clip duration **5 seconds**, center padding on, yielding **313 time frames**. Except for intentionally changing the maximum frequency, frequency resolution in Hz is held fixed. These are two *specific* FFT/sample-rate conditions, not evidence for all resampling schemes.

Representations:

1. `logmel64`: 64 HTK-scale triangular log-power Mel bands; the **16k/F8 case exactly equals the historical project frontend**.
2. `logmel32`: 32 log-power Mel bands. Compares *Mel resolution*, same STFT and bandwidth.
3. `mfcc40`: first 40 coefficients of an **orthonormal DCT-II of 64 Log-Mel bands**. No lifter or alternative c0 replacement. This is explicitly a project MFCC-like variant, not a strict reproduction of a particular library's MFCC conventions.
4. `loglinear64`: 64 **linear-frequency triangular** log-power bands. Tests Mel versus linear frequency spacing while preserving output-band count.

Why not 128 Mel bands at 16k/512 FFT? At that resolution, some very narrow triangular filters are empty, producing constant floor values. The prespecified experiment uses 32 vs 64 bands instead, avoiding that artifact.

**Primary model:** `cnn` (the pre-existing frequency-pooled CNN), not `cnn_freq`. For this specific study, global frequency pooling ensures the model has the **same number of trainable parameters** even when there are 32/40/64 input bands. Optional `crnn` is the analogous frequency-pooled recurrent model. Do **not** compare these results directly against historical `cnn_freq`/`crnn_freq` as if they were identical architectures.

All cases: source-strict Train Folds 1–3, Validation Fold 4, **no Fold 5 evaluation**, train-only per-band normalization, same model, initialization seed, AdamW, cosine scheduler, batch size, learning rate, epoch budget and checkpoint selection by **minimum validation cross-entropy**. Original WAV audio is never augmented in this track, to isolate the ablation variables.

## How to run after obtaining the official ESC-50 WAVs

Use the project root and install the project dependencies (`pip install -e .`), or set `PYTHONPATH=src` as below.

```bash
# Review all 12 immutable planned cases
PYTHONPATH=src python scripts/feature_ablation.py plan

# One-time preparation of all 12 caches (requires full official 2,000 WAV files)
PYTHONPATH=src python scripts/feature_ablation.py prepare \
  --raw-root data/ESC-50 \
  --cache-dir data/feature_ablation

# Pilot one configuration / one seed (development validation only)
PYTHONPATH=src python scripts/feature_ablation.py train \
  --raw-root data/ESC-50 \
  --cache-dir data/feature_ablation \
  --results-dir results/feature_ablation \
  --case sr16_fmax8_logmel64 --model cnn --seed 101 \
  --epochs 60 --batch-size 32 --lr 0.001 --device auto

# Once the recipe is fixed, run all cases on seeds 101/202/303:
MODE=cnn EPOCHS=60 DEVICE=auto bash tools/run_feature_grid.sh

# Aggregate individual development validation rows and matched contrasts
PYTHONPATH=src python scripts/feature_ablation.py summary \
  --results-dir results/feature_ablation \
  --output results/feature_ablation_summary.csv
PYTHONPATH=src python scripts/feature_ablation.py contrasts \
  --results-dir results/feature_ablation \
  --output results/feature_ablation_contrasts.csv
```

`run_feature_grid.sh` refuses to silently overwrite a previous run. If any case stopped partway through, inspect or explicitly move its incomplete directory; this **lightweight ablation track does not promise epoch-resume**. The legacy resume-capable training path is unaffected.

## Interpreting contrasts

- **SR16-F8 vs SR32-F8:** sampling-rate-related behavior with matched 8 kHz input bandwidth. Does **not** demonstrate additional high-frequency information.
- **SR32-F8 vs SR32-F16:** adding 8–16 kHz high-frequency content **at fixed 32 kHz sampling**. Because the filterbank is stretched across more bandwidth at a fixed number of Mel bands, this also changes effective per-band spectral resolution; interpret accordingly.
- **Log-Mel 64 vs Log-Mel 32:** Mel filter resolution at fixed sample rate/bandwidth.
- **Log-Mel 64 vs MFCC40:** the effect of replacing time-frequency Mel energies with DCT coefficients (also changes number of input rows; the model parameters stay fixed).
- **Log-Mel 64 vs Log-Linear 64:** Mel frequency warping versus linear triangular bands at fixed band count.

Use **paired seed differences** from the contrast CSV. Three seeds are a stability check, not a strong statistical confidence guarantee. Select a primary recipe using the development validation protocol only; if later evaluating a held-out set, freeze the recipe first and explicitly disclose any prior test-set exposure.

## Deliberate reduction of over-engineered hashing

This **new ablation path** does **not** compute or compare SHA-256 checksums for individual WAVs, metadata, generated NPZ caches, experiment configs, or model outputs. Instead it validates:

- Metadata/filename consistency and source-strict splits;
- every cached filename, label, fold, source ID and category mapping against independent Metadata;
- cache shape, dtype, finite values, absence of duplicate names and explicit frontend settings;
- unique output paths and validation-only predictions.

The **legacy** cache format and saved historical training/checkpoints still use their existing identity/checksum semantics to avoid breaking compatibility and invalidating the historical audit. Those paths were **not** rewritten. This is a *targeted simplification of the new experiment workflow*, not a claim that hashes were removed project-wide.

## Limitations and completion criteria

- Raw WAVs are not bundled; no real ESC-50 ablation measurements are reported in this package.
- This track is intentionally not integrated with the old `cnn_freq` checkpoint/evaluation CLI, which hard-codes the historical frontend. Do not attempt to load a new ablation checkpoint in legacy `esc50 evaluate` or `predict`.
- Generating a full cache reads source WAVs and requires storage space; 12 caches may be several GB (not including original WAVs). For limited storage, prepare/run cases **one at a time** and archive completed validation reports.
- The controlled bandwidth contrast cannot fully isolate sample rate from resampling implementation; the high-frequency contrast changes filter spacing. These caveats belong in the report.
- Complete when each prespecified case has three development runs, paired contrasts have been reviewed, and at most a **predeclared** selected recipe is used in subsequent confirmatory testing.
