# Evaluation on external recordings

This supporting evaluation uses six mapped categories from the manually verified [FSDnoisy18k v1.0](https://zenodo.org/records/2529934) test partition: clapping, engine, fireworks, rain, wind and footsteps. Five recording IDs also present in ESC-50 metadata are excluded, leaving **259 unique recordings**. All 50 ESC-50 output labels remain active.

| Frozen system | Accuracy | Six-class macro recall | Six-class macro-F1 |
| --- | ---: | ---: | ---: |
| PANNs + linear classifier | 45.17% | 46.40% | 58.07% |
| Noise CNN, seed 101 | 18.92% | 18.91% | 28.65% |
| Noise CNN, seed 202 | 21.62% | 21.74% | 33.38% |
| Noise CNN, seed 303 | 19.69% | 20.24% | 30.43% |

The noise-CNN mean is **20.08%**. PANNs' paired advantage is **25.10 percentage points [19.82, 30.63]**, conditional 95% recording-bootstrap interval. The three-CNN mean averages fitted-model metrics and is not an ensemble prediction.

Each recording is resampled to 16 kHz and split into consecutive five-second windows, with only the final fragment zero-padded. Window probabilities are averaged using actual unpadded duration as the weight, then argmax is taken over all 50 labels. Each recording contributes one evaluation row. No fitting or noise injection occurs; 534 windows were evaluated in 28.98 seconds on Colab Tesla T4, excluding setup/downloads.

This is a mapped six-category subset, not a complete independent 50-class benchmark. Recording-ID exclusion does not resolve reuploads, shared uploaders, perceptual duplicates or AudioSet pretraining overlap. Differences from ESC-50 also include category composition, recording duration and aggregation policy.

## Error patterns

PANNs makes 132 predictions outside the six target labels and ten within-six mistakes. It scores 27.62% on 105 recordings shorter than five seconds and 57.14% on 154 longer recordings. Duration, class mix and event content vary together, so these descriptive slices do not identify the cause of the short-recording deficit. The [input-duration diagnostic](../input_duration/README.md) examines a separate paired intervention.

[Notebook](../../../notebooks/03_external_recordings.ipynb) · [Protocol](protocol.json) · [Manifest](manifest.csv) · [Excluded IDs](excluded_recording_ids.csv) · [Metrics](metrics.csv) · [Paired effects](paired_effects.csv) · [Predictions](predictions.csv) · [Per-class results](per_class.csv) · [Duration slices](duration_slices.csv) · [Summary](summary.json)

Dataset reference: Fonseca et al., *Learning Sound Event Classifiers from Web Audio with Noisy Labels*, ICASSP 2019 ([paper](https://arxiv.org/html/1901.01189)). The [dataset license](LICENSE-DATASET) is CC BY 4.0; individual recording licenses are listed in the manifest. Raw audio is not redistributed here.
