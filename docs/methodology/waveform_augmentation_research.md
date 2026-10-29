# ESC-50 Waveform Augmentation & Robustness — research extension

## Scope and evidence

The existing Phase 5 log-mel CNN/CRNN results and historical checkpoints are unchanged. This extension supports **training-time** waveform Gaussian noise and waveform Mixup, together with existing normalized-feature SpecAugment. No new ESC-50 accuracy is claimed: full official WAVs and the generated v2 feature cache are needed for real training. Integration tests use explicitly synthetic waveforms.

## Data and invariants

1. `esc50 prepare` creates a pinned v2 feature cache, with a SHA-256 digest for each raw WAV and for official CSV Metadata.
2. All experiments keep `train`, `validation` and `test` split IDs and source-strict exclusions unchanged. Only selected *training* WAVs are loaded for augmentation. No validation or test waveform receives train-time augmentation; validation still reads the unchanged clean log-mel cache.
3. Before augmented training, original Metadata, train WAV headers/hashes and **each clean training WAV's regenerated log-mel** must match the cache. Fail closed on missing WAV or stale data.
4. Train-only normalization uses the *clean training fold's* cached statistics. Random waveform transformations occur **before** log-mel extraction; SpecAugment is optionally applied after normalization. No waveform augmentations are cached as final ESC-50 features.
5. Noise is zero-mean Gaussian, with per-example probability and uniformly sampled added-noise SNR in dB. Scale is based on the original decoded waveform's RMS. Original ambient noise is unknown. Silence remains silent; generated floats are not artificially clipped. This is a simulated perturbation, not real-world acoustic transfer.
6. Waveform Mixup uses Beta(alpha, alpha) coefficients within the training minibatch. Loss is the per-example weighted sum of two cross-entropies; no hard one-hot label is assigned to a mixed recording.
7. The PyTorch RNG state and DataLoader shuffle generator remain in `last.pt`. Epoch-boundary resume has been tested with both Gaussian noise and Mixup, including the exact final weights and loss history on synthetic data.
8. Training plans use schema v2 to declare augmentation settings explicitly; v1 plans still parse, preserve their original identity and default to **no** waveform augmentation. Runtime and checkpoint provenance record the chosen settings.

### Operations

From the package root, after installing dependencies and restoring complete ESC-50 files:

```bash
PYTHONPATH=src python -m esc50_sound prepare --root data/ESC-50 --cache data/esc50_features.npz

# Check selected plan and pinned data; does not train or read test predictions.
PYTHONPATH=src python scripts/research_platform.py check --plan examples/platform_cnn_noise_v2.json

# Start only after preflight is ready. New and clean run folders only.
PYTHONPATH=src python scripts/research_platform.py run --plan examples/platform_cnn_noise_v2.json --execute
```

Three strictly separate experiment plans are provided (same architecture, folds, epochs, seed):

| Plan | Noise | Mixup | What it tests |
|---|---|---|---|
| `platform_cnn_clean_v2.json` | off | off | Controlled clean baseline under schema v2 |
| `platform_cnn_noise_v2.json` | p=0.5, 10–30 dB | off | Effect of training noise on clean accuracy and stress accuracy |
| `platform_cnn_mixup_v2.json` | off | alpha=0.4 | Effect of waveform mixing independent of Gaussian noise |

For an existing run interrupted **after a complete epoch**, use the existing `resume-train` CLI with its `--output`. This uses recorded settings; it does not accept changed augmentation hyperparameters.

### Clean and corrupted evaluation

Inspect validation results in each run's `validation.json` and prediction CSV. For a controlled *validation-only* robustness check (no test access or holdout peeking), reuse the existing `study.py stress` pipeline:

```bash
python scripts/legacy/study.py stress \
  --checkpoint runs/cnn_noise_snr10_30_seed101/best.pt \
  --cache data/esc50_features.npz \
  --root data/ESC-50 \
  --split validation \
  --conditions clean snr_20 snr_10 snr_0 \
  --seed 2026 \
  --output results/stress_noise_validation_seed101
```

Use a new `--output` per run; the study tool refuses silent replacement. Predeclare seeds and noise conditions, use matching noise realizations for models, and do not optimize against the held-out test fold. Clean accuracy and synthetic-noise degradation are **both** essential; stronger corruption performance might trade off clean performance. The stress tool's waveform/feature parity threshold is stricter than the training preflight and can reject incompatible caches.

### Performance and limitations

Waveform augmentation intentionally caches **only selected training waveforms in RAM** (roughly 384 MB for 1,200 x 80,000 float32 samples), avoiding repeated disk decoding each epoch. It adds STFT/Mel computation per training step; CPU training will be substantially slower than cached-feature training. There is no full real ESC-50 train or stress result for this extension. One noise family (Gaussian), one SNR sampling scheme, a small number of fixed configurations, and only development-validation selection are supported. New production infrastructure or unverified performance claims are not warranted.
