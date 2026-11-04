# ESC-50 Research Follow-up — 4 November 2026

The six missing robustness inference tasks are complete: CNN, CRNN and noise-trained CNN at seeds 202 and 303. They add 36 condition-level prediction files to the 18 original seed101 files. No model was retrained and no test-fold WAV was evaluated.

At 10 dB Gaussian added-noise SNR, mean accuracy across the three fixed seeds is **23.37% for clean-trained CNN and 61.31% for noise-trained CNN**. The paired advantage is **+37.94 percentage points**, with a 95% source-cluster bootstrap interval of **[+32.08, +43.88] pp**. This interval conditions on the three fitted models and the fixed noise realizations. Both gain controls cross zero. These results support a Gaussian-noise-specific benefit, rather than a claim about arbitrary real-world noise.

The clean CRNN advantage is **+1.26 pp**, with a conditional source interval of **[−1.81, +4.17] pp**, while its parameter count is about 5.1 times the main CNN. Its useful temporal mechanism remains unresolved. The feature histories confirm 30/36 late best checkpoints; Log-Mel 32's loss advantage appears before the final ten epochs, under the separate pooled-CNN architecture.

Next completed round: [5 November temporal-order probe](../temporal_20261105/README.md).

## Results and implementation

- [Prediction-level results and scope](report_en.md): all 24 clean fits and all 54 robustness condition files; paired errors, source-cluster intervals, and persistent mistakes.
- [Feature convergence analysis](convergence/report.md): the original 36 feature runs, their learning curves, and the limited-budget interpretation.
- [Real-audio error cases](error_cases/report.md): four explicitly selected examples, waveform/spectrogram figures, and the full validation case index. Listening and causal interpretation remain separate from prediction counts.
- [Formal specifications](../../docs/methodology/followup_formal_properties.md) and [solver results](formal_properties.json): nine ideal-arithmetic properties and three counterexamples to incorrect alternatives. These do not prove neural-network robustness.
- [Research discussion record](research_discussions.md): the original user discussion and the two Chrome ChatGPT reviews/code drafts.

## Run from the project root

```sh
# Preview missing inference tasks; no training or inference by default.
python scripts/run_robustness_three_seeds.py

# Run at most one pending task using existing weights and original WAVs.
python scripts/run_robustness_three_seeds.py --execute --max-tasks 1 --audio-root PATH_TO_WAV_DIRECTORY

# Analyze original saved predictions and the completed robustness follow-up.
python scripts/analyze_prediction_errors_v2.py

# Read existing feature histories and inspect selected audio cases.
python scripts/analyze_feature_convergence.py
python scripts/build_error_case_atlas.py --audio-root PATH_TO_WAV_DIRECTORY

# Optional exact-arithmetic specification queries (requires z3-solver).
python scripts/prove_followup_properties.py
```

Full inference requires the existing matrix checkpoints, which remain in the local experiment collection. New code uses original WAVs and each checkpoint's stored training normalization directly; it does not require a feature-cache integrity scan. The original 65-task study is preserved. New scientific conclusions remain development observations on already-exposed Fold4.
