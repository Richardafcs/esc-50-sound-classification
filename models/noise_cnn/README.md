# Noise-trained CNN checkpoints

Three fixed frequency-preserving CNN fits, seeds 101/202/303, from the completed waveform-augmentation study. Each has 49,058 parameters. Training noise probability0.5, added-noise SNR uniformly10–30dB;60 epochs, train-only normalization. These compact exports contain the model tensors, normalization, frontend, class map and split roles.

Use the transfer-follow-up notebook for matched clean/10 dB inference against the frozen PANNs system. The newly generated seed-sequence-v2 noise is separate from the historical robustness panel. No retraining is needed.
