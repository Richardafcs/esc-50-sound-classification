# CPU-only ESC-50 Attention study: execution package

**Status:** CPU tested for model forward/backward with random tensors, but **real ESC-50 training not executed** because this environment lacks full WAV files / the audio feature cache and external DNS is blocked. Do not report synthetic performance as ESC-50 accuracy.

## Resources observed (2026-10-28)

- CPU: 5 logical cores; RAM: ~5.8 GiB; no CUDA; PyTorch 2.10.0+cpu.
- Representative Attention-CRNN model (`crnn_freq_tca`) ran at ~0.28s per 32-sample training step with 3-4 threads on a small synthetic microbenchmark.
- With ~38 train batches per epoch, 60 epochs per run, and 12 runs, **several hours** is a reasonable planning scale, but the measurements do not give a guaranteed wall-clock duration. Do not leave jobs detached expecting results to return in ChatGPT.

## Input needed

Use the **unmodified official ESC-50 WAV audio**, all 2000 files at `data/ESC-50/audio/*.wav`, plus `data/ESC-50/meta/esc50.csv`, OR a provenance-valid `data/esc50_features.npz` alongside its `.npz.json` sidecar created by this project. Merely uploading metadata or saved model weights is not sufficient.

Official dataset: https://github.com/karoldvl/ESC-50 (CC BY-NC).

## Execution once the real audio is available

From the project root, in an environment where the package dependencies have been installed:

```bash
# Not needed if valid cache and sidecar were provided directly.
PYTHONPATH=src python -m esc50_sound prepare --root data/ESC-50 --cache data/esc50_features.npz

# Safely check the actual new model tests.
PYTHONPATH=src python -m pytest -q tests/test_attention.py tests/test_attention_grid.py

# First, optionally run the four-model single-seed pilot (exploratory only):
PYTHONPATH=src python scripts/legacy/runner.py run-grid \
  --config configs/attention_cpu_pilot_v1.json \
  --output results/attention_cpu_pilot_v1 \
  --base-dir .

# Execute the full 12 CPU TRAINING jobs, one after another (no test-set evaluation).
PYTHONPATH=src python scripts/legacy/runner.py run-grid \
  --config configs/attention_cpu_grid_v1.json \
  --output results/attention_cpu_grid_v1 \
  --base-dir .
```

`runner.py` stores command logs and run metadata. Training itself writes validation and checkpoint outputs. The configured 3,500s training and 3,600s per-job runtime caps are protective limits: a run must complete the full 60 Epochs to be comparable; **truncated runs are not equivalent**.

## Scientific safeguards

- Keep the historical `crnn_freq` and CNN baselines unchanged.
- Compare `crnn_freq`, `crnn_freq_ta`, `crnn_freq_ca`, `crnn_freq_tca` under the same source-strict fold policy, training schedule and seeds 101, 202, 303.
- Fold 5 was already inspected historically, so **do not** treat it as a newly blinded test set. This grid trains/validates; it does not run the test evaluator.
- Do not optimize hyperparameters or select models based on the previously viewed Fold 5 test.
- Report partial failure honestly, keep all logs, and explicitly distinguish exploratory results from any newly independent confirmation study.
