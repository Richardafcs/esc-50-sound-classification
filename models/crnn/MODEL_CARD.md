# Model card: CNN with bidirectional GRU

Version 0.1.0, 28 October 2026. Agent-produced research prototype. This is the retained recurrent comparison model. There is no novel-architecture or production-readiness claim.

## Input, output and intended use

Single environmental-sound classification over the 50 ESC-50 categories. Use for noncommercial research, course experiments and inspecting a small from-scratch model. The complete clip must be available. Input up to five seconds is downmixed to mono, resampled to 16 kHz and right-padded. Longer recordings are rejected. Near-silence (processed waveform RMS < 1e-5) abstains. Malformed/nonfinite input is rejected.

Output is a predicted category and five **uncalibrated softmax scores**. This model does not recognize arbitrary unknown categories, separate simultaneous sources, or establish real-world acoustic monitoring reliability. Silence abstention is an energy rule, not general out-of-distribution detection.

## Training and selection

ESC-50 commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`: 2,000 clips, 50 classes. This checkpoint trained on folds 1–3 (1,200 clips), selected on source-repaired fold 4 (398), and evaluated on untouched-by-selection fold 5 (400). Training-only normalization;16 kHz, 64-band natural-log mel-power frontend; no external pretraining. Source-ID overlaps are disclosed and held-out sources excluded from earlier roles.

AdamW, learning rate 0.001, cosine schedule, batch size 32,60 epochs, no SpecAugment; seed 101 was fixed before confirmation. Checkpoint epoch 58 minimizes validation cross-entropy, earliest tie. Parameters: 251,810. The default CNN was selected before test access because the recurrent three-seed development gain was 1.26 pp, below the 3 pp preference threshold. The separate augmentation pilot failed its development gate.

## Measured performance

This specific checkpoint: validation accuracy 75.13%; fold 5 accuracy 62.00%, macro-F1 59.50%. Across seeds 101/202/303, the model-family fold 5 accuracy is 63.92% ± 1.77 pp (sample SD, three training runs). The deployed checkpoint is seed 101, not the best test seed. This table is one held-out fold under a source-strict three-training-fold protocol, not an official five-fold benchmark score.

All 400 saved fold 5 predictions were reproduced from the original WAVs through the packaged frontend/model in an independent internal check. This is not external human replication. No fold 5 recording triggered silence abstention.

## Failure boundaries

Observed class failure: **airplane and fireworks (0/8 correct in each class)**. Per-class test support is only eight clips, so class-level estimates are imprecise. Accuracy was evaluated on five-second ESC-50 recordings; shorter inputs and new microphones have not been validated.

Synthetic stress accuracies: snr_20: 45.25%; snr_10: 25.00%; snr_0: 7.00%; gain_-6: 60.50%; gain_6: 61.50%. The severe noise degradation means this checkpoint should not be described as noise-robust. Corruptions are one deterministic additive-Gaussian realization per clip/condition and unclipped gain changes; they do not simulate all real microphones or backgrounds. Model decisions should not be treated as a reliable alarm or monitoring system.

## Efficiency and execution

Checkpoint bytes: 1,098,142. On an Apple M4 CPU, two threads, batch size 1, loaded model,10 warmup and 100 measured repeats of one validation WAV: p95 WAV→logits 4.36 ms; p95 cached-feature model inference 1.91 ms. This warms the file cache and excludes model loading, process startup and service overhead. It does not measure CUDA inference or a production service. Training used a Colab TeslaT4 with PyTorch 2.11.0+cu130; portable inference was checked on local PyTorch 2.13.0 CPU.

## Provenance, license and use

Checkpoint SHA256:`037eee005bb092b7c1c7b61cdba55f2fc2d59f0d0cff60845a93fd562cc4f282`. The checkpoint embeds frontend, classes, normalization, training configuration and manifest identity. The bundle provides source and a file manifest. Package/frozen-source model logits were checked for numerical equivalence; retraining on another library/hardware combination is not guaranteed bit-for-bit identical.

ESC-50 is distributed overall under CC BY-NC 3.0. Preserve its individual credits and metadata; this project does not establish commercial-use rights. No dataset recordings are redistributed in the portable bundle. See [the official dataset](https://github.com/karolpiczak/ESC-50).

From the bundle directory, install its requirements if needed and run:

```sh
python predict.py --checkpoint checkpoint.pt --audio YOUR_SHORT_CLIP.wav
```

Read the complete study report for the selection history, controls, uncertainty and broader fold characterization. The package is an offline prototype, not a supported public service.
