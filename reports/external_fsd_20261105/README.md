# External recording transfer evaluation

[Executed notebook](../../notebooks/ESC50_External_Transfer.ipynb) · [Fixed protocol](protocol.json) · [Recording manifest](manifest.csv) · [Excluded recording IDs](excluded_recording_ids.csv)

The official [FSDnoisy18k v1.0](https://zenodo.org/records/2529934) manually verified test partition supplies an external evaluation subset. Six category correspondences are fixed before predictions: Clapping→clapping, Engine→engine, Fireworks→fireworks, Rain→rain, Wind→wind, and Walk_or_footsteps→footsteps. Ambiguous or absent ESC-50 classes are excluded. Five recording IDs found anywhere in ESC-50 metadata are removed; 259 unique recording IDs remain.

Each frozen system receives the same full recording, resampled to 16 kHz and divided into consecutive five-second windows. Only the final fragment is zero-padded. Probabilities are averaged with weights equal to the actual unpadded window duration, followed by argmax over all 50 ESC-50 output classes. Predictions are never restricted to the six target classes. One recording contributes one evaluation row, regardless of its number of windows.

The primary comparison is PANNs accuracy minus the mean accuracy of three existing noise-trained CNNs. The mean is not an ensemble. Secondary metrics include six-class macro recall/F1, per-class recall and the proportion of predictions outside the six target labels. Outside-label predictions remain errors. Four thousand paired recording-ID bootstrap resamples quantify conditional uncertainty; here each ID occurs once, so this is recording-level resampling rather than uploader clustering.

No training, classifier selection, noise injection or ESC-50 WAV evaluation occurs in this phase. Reliable FSD labels do not imply acoustically noise-free recordings. Known recording-ID exclusion does not eliminate reuploads, perceptual duplicates, shared uploaders or possible AudioSet pretraining overlap. Category composition, durations and inference aggregation differ from the earlier ESC-50 panel, preventing causal attribution of changes in absolute accuracy to domain shift alone.

This is neither the original 20-class FSD benchmark nor confirmation across all 50 ESC-50 classes. Retain the original classifier and do not use these outcomes to tune a replacement.

Dataset reference: Fonseca et al., “Learning Sound Event Classifiers from Web Audio with Noisy Labels,” ICASSP 2019, [Section 2](https://arxiv.org/html/1901.01189). Dataset-level [CC BY 4.0 license](LICENSE-DATASET); individual recording licenses are included in the manifest. Raw audio is downloaded from the authors and is not redistributed here.

## Measured results

| Frozen system | Accuracy | Macro recall | Macro-F1 |
| --- | ---: | ---: | ---: |
| panns_lr | 45.17% | 46.40% | 58.07% |
| noise_cnn_s101 | 18.92% | 18.91% | 28.65% |
| noise_cnn_s202 | 21.62% | 21.74% | 33.38% |
| noise_cnn_s303 | 19.69% | 20.24% | 30.43% |

Primary accuracy difference: **+25.10 pp [+19.82, +30.63]**. The noise-CNN mean is 20.08%. This interval is conditional on these fixed models, mapping and recordings.

Completed Colab Tesla T4 inference: 259 recordings, 534 windows, zero fits, 28.98 seconds excluding setup/downloads. [All metrics](metrics.csv), [paired effects](paired_effects.csv), [per-class recall](per_class.csv), [all predictions](predictions.csv), [recording durations/windows](recordings.csv), [execution summary](summary.json).

## Descriptive error follow-up

PANNs makes 132 predictions outside the six target labels and 10 within-six mistakes. The noise-CNN mean has 186.67 outside-six and 20.33 within-six mistakes. These two count differences sum to the mean 65 additional correct PANNs predictions; they are arithmetic accounting, not a causal explanation. Clapping is predicted as can opening on 30 recordings.

PANNs accuracy is 27.62% on 105 recordings under five seconds and 57.14% on 154 recordings of at least five seconds. Duration, event content, category mix and padding vary together. No short recordings are removed and no inference rule is changed. On the same six development categories, each recall denominator is eight clips; development-vs-external differences are descriptive and do not isolate domain shift.

[Error accounting by class](error_accounting_by_class.csv) · [Error totals](error_accounting.csv) · [Duration slices](duration_slices.csv) · [Same-category development context](same_category_development_context.csv) · [PANNs error patterns](panns_error_patterns.csv). These analyses use saved predictions only; no further inference or fitting follows the outcome.
