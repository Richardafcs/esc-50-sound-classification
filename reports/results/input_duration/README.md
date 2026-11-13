# Input duration and padding placement

This diagnostic compares the same frozen PANNs classifier under right-padding, native-duration input and left-padding on short external recordings.

| Input policy | Eligible recordings | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| Right-zero-pad to five seconds | 102 | 28 | 27.45% |
| Native duration | 102 | 34 | 33.33% |
| Left-zero-pad to five seconds | 102 | 27 | 26.47% |

Native minus right-padding is **+5.88 percentage points**, with a conditional 95% recording-bootstrap interval of **[−0.98, 12.75]**. The accuracy gain is uncertain. Native input changes 46 predictions, repairing ten errors and introducing four. Left-padding changes 50 predictions, with one fewer correct recording overall.

All 105 short recordings remain in the right/left comparison. Three recordings shorter than 0.31 seconds are excluded only from native inference: the official encoder requires at least 32 centered-STFT frames before five factor-two temporal pooling stages. At hop 160 and 16 kHz, this requires at least 4,960 samples. The primary three-policy comparison therefore uses the same 102 eligible recordings.

No encoder, scaler or classifier is fitted. The original **45.17% external accuracy on 259 recordings** remains a separate result using its original input policy. The diagnostic was designed after that panel had been examined.

Padding placement changes convolution boundaries, STFT context and pooling inputs together. Its effect does not identify a single internal mechanism. Permuting unchanged encoded tokens is a different intervention from moving zeros in the raw waveform.

[Notebook](../../../notebooks/04_input_duration.ipynb) · [Protocol](protocol.json) · [Metrics](metrics.csv) · [Paired effects](paired_effects.csv) · [Predictions](predictions.csv) · [Exclusions](native_exclusions.csv) · [Short-recording manifest](short_recordings.csv) · [Identical-input controls](negative_controls.csv) · [Summary](summary.json)

Inference completed in 23.98 seconds on Colab Tesla T4, excluding setup/downloads. Intervals use 4,000 recording-level resamples and condition on these recordings and the fitted system.
