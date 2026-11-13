# Experiment results

The [course report](../report.md) presents the main comparison. The [results notebook](../../notebooks/01_results.ipynb) combines the tables, analyses and all 40 figures.

| Experiment | Purpose | Main observation |
| --- | --- | --- |
| [Classical references](model_comparison/README.md) | Log-Mel summary logistic regression and cepstral SVM | 42.21% and 40.20% development accuracy |
| [Feature and sampling study](model_comparison/README.md) | Twelve frontends with a separate pooled CNN | Log-Mel32 exceeds64 within this separate track |
| [Model comparison](model_comparison/README.md) | CNN, BiGRU, attention, augmentation and feature settings | CNN 72.61%, BiGRU 73.87%, channel attention 74.37% development accuracy |
| [CNN capacity controls](mlp_controls/README.md) | Approximately parameter-matched MLP alternatives | Pooled/tokenwise MLP 72.28%/69.68% |
| [Errors](error_analysis/README.md) | Paired predictions, class errors and recurring failures | 41 clips are wrong across four model families and all three seeds |
| [Robustness](robustness/README.md) | Gaussian interference and gain changes | Noise-trained CNN 61.31% versus clean-trained CNN 23.37% at 10 dB |
| [Temporal order](temporal_order/README.md) | Interventions on encoded time tokens | BiGRU reversal changes accuracy by −5.03 percentage points |
| [Frozen PANNs](panns/README.md) | Pretrained audio features with a linear classifier | 91.21% accuracy and 91.10% macro-F1 |
| [PANNs reproduction](panns_reproduction/README.md) | A second extraction and classifier fit | The same 363/398 correct development predictions |
| [Transfer heads](transfer_heads/README.md) | Matched noisy inputs and linear/ReLU heads | PANNs has higher noisy accuracy; tested ReLU heads do not improve it |
| [Gaussian adaptation](gaussian_adaptation/README.md) | Noise exposure while fitting the frozen-feature classifier | Gaussian 10-dB accuracy rises from 69.85% to 83.92% |
| [Recorded-noise training](recorded_noise/README.md) | Gaussian versus recorded training interference | Recorded-panel gain of 1.36 pp, with a Gaussian-noise tradeoff |
| [External recordings](external_recordings/README.md) | Six mapped categories from FSDnoisy18k | PANNs 45.17%, noise-CNN mean 20.08% |
| [Input duration](input_duration/README.md) | Padding and native-duration inference | Predictions change; the native-duration accuracy gain is uncertain |
| [Historical baseline](historical_baseline/README.md) | Earlier fold-five comparison | Retained separately from the main development scores |

The [study overview table](../tables/study_overview.csv) records the populations, result sources and interpretation of each study. The latest model experiment is the recorded-background classifier fit completed on 7 November; later organization and analysis refreshes do not add model fits.

Scratch-model means use three fixed training seeds. Most comparisons reuse 398 development clips from 295 recording sources. Historical fold exposure and possible pretraining overlap limit generalization claims; detailed definitions are in [evaluation](../../docs/evaluation.md).
