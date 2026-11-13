# Noise-trained CNNs

`seed101.pt`, `seed202.pt` and `seed303.pt` are the three frequency-preserving CNN fits trained with Gaussian waveform augmentation. Each model has **49,058 parameters**. The files include weights, training-fitted normalization, frontend settings, class mapping and split roles.

Training uses 60 epochs, with Gaussian noise applied independently to each example with probability 0.5. Added-noise SNR is sampled uniformly from 10–30 dB before log-Mel extraction. Clean development accuracy averages **73.12% ± 0.87 percentage points**.

The original Gaussian robustness panel gives **61.31% accuracy at 10 dB**. The later matched PANNs comparison gives **61.56%** on a different shared noise realization. These are separate panels; within-panel comparisons use the same corrupted waveforms for all models.

The [transfer-head notebook](../../notebooks/02_transfer_heads.ipynb) loads these exports for the matched comparison. For training and augmentation details, see [methods](../../docs/methods.md); for panel definitions and uncertainty, see [evaluation](../../docs/evaluation.md). The measured Gaussian-noise benefit does not establish robustness to arbitrary acoustic interference.
