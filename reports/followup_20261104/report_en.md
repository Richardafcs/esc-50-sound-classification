# ESC-50 Prediction Follow-up — 4 November 2026

Analyzed **24 clean prediction sets**, 398 validation clips and 295 original sources. Available robustness prediction sets: 54/54.

## Paired clean CNN vs CRNN

| Seed | CRNN repairs | New errors | Net clips | Delta accuracy (pp) | 95% source bootstrap interval (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 32 | 22 | 10 | +2.513 | [-1.044, +6.091] |
| 202 | 33 | 27 | 6 | +1.508 | [-2.585, +5.583] |
| 303 | 31 | 32 | -1 | -0.251 | [-4.348, +3.857] |
| fixed_three_seed_mean | — | — | — | +1.256 | [-1.806, +4.167] |

## Noise-trained CNN vs clean-trained CNN

| Condition | Seed | Noise-only correct | Clean-only correct | Net clips | Delta accuracy (pp) | 95% source bootstrap interval (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | 101 | 20 | 20 | 0 | +0.000 | [-3.291, +3.133] |
| clean | 202 | 26 | 19 | 7 | +1.759 | [-1.772, +5.366] |
| clean | 303 | 24 | 25 | -1 | -0.251 | [-3.828, +3.234] |
| clean | fixed_three_seed_mean | — | — | — | +0.503 | [-2.020, +2.884] |
| snr_20 | 101 | 107 | 16 | 91 | +22.864 | [+17.150, +28.536] |
| snr_20 | 202 | 102 | 19 | 83 | +20.854 | [+14.988, +26.583] |
| snr_20 | 303 | 110 | 15 | 95 | +23.869 | [+17.979, +29.880] |
| snr_20 | fixed_three_seed_mean | — | — | — | +22.529 | [+17.354, +27.636] |
| snr_10 | 101 | 166 | 9 | 157 | +39.447 | [+32.978, +45.918] |
| snr_10 | 202 | 151 | 14 | 137 | +34.422 | [+28.085, +40.887] |
| snr_10 | 303 | 169 | 10 | 159 | +39.950 | [+33.333, +46.798] |
| snr_10 | fixed_three_seed_mean | — | — | — | +37.940 | [+32.083, +43.878] |
| snr_0 | 101 | 81 | 4 | 77 | +19.347 | [+15.089, +23.729] |
| snr_0 | 202 | 64 | 7 | 57 | +14.322 | [+10.117, +18.750] |
| snr_0 | 303 | 86 | 3 | 83 | +20.854 | [+16.321, +25.450] |
| snr_0 | fixed_three_seed_mean | — | — | — | +18.174 | [+14.547, +21.872] |
| gain_-6 | 101 | 17 | 18 | -1 | -0.251 | [-3.211, +2.799] |
| gain_-6 | 202 | 24 | 29 | -5 | -1.256 | [-5.182, +2.605] |
| gain_-6 | 303 | 21 | 27 | -6 | -1.508 | [-4.775, +1.777] |
| gain_-6 | fixed_three_seed_mean | — | — | — | -1.005 | [-3.317, +1.368] |
| gain_6 | 101 | 20 | 25 | -5 | -1.256 | [-4.601, +2.211] |
| gain_6 | 202 | 21 | 23 | -2 | -0.503 | [-3.788, +2.771] |
| gain_6 | 303 | 25 | 24 | 1 | +0.251 | [-3.023, +3.655] |
| gain_6 | fixed_three_seed_mean | — | — | — | -0.503 | [-2.799, +1.848] |

## Reduction of corruption-induced accuracy loss

For each fixed fit pair, loss reduction = (noise-CNN corrupted accuracy minus clean-CNN corrupted accuracy) minus their clean-input accuracy difference. Positive values mean noise training reduces the corruption-induced accuracy loss. The table averages the three fixed seeds with the same source draws; per-seed values are in `robustness_loss_reduction.csv`.

| Condition | Mean loss reduction (pp) | 95% conditional source interval (pp) |
| --- | ---: | ---: |
| snr_20 | +22.027 | [+16.926, +27.302] |
| snr_10 | +37.437 | [+31.308, +43.740] |
| snr_0 | +17.672 | [+13.057, +22.371] |
| gain_-6 | -1.508 | [-3.646, +0.672] |
| gain_6 | -1.005 | [-3.256, +1.446] |

## Persistent errors

- cnn: 83 clips from 66 original sources, incorrect in every included seed/model fit.
- crnn: 64 clips from 52 original sources, incorrect in every included seed/model fit.
- shared_four_families: 41 clips from 31 original sources, incorrect in every included seed/model fit.

## Statistical meaning and limits

Tables: `class_accuracy.csv` averages the same unique clips across three fixed fits. `confusions.csv` distinguishes seed-clip error events from unique clips and sources; repeated-fit errors are not independent observations. `persistent_errors.csv` lists concrete samples for subsequent listening and spectrogram review.

Clip-weighted paired accuracy difference. Resample G original sources with replacement and keep every clip of each selected source; each draw uses its actual clip count denominator. Use identical source draws for all paired models and fixed training seeds.

Intervals condition on the fitted models and one fixed corruption realization per clip/condition. Three seeds reuse the same audio and are not 3N independent observations. Fixed-three-seed mean intervals do not quantify the distribution of future training runs.

Sources contain 1–8 clips. Resampled denominators range from 353 to 454 clips. Replacing those denominators with 398 changes the resampling statistic; averaging source accuracies equally changes the estimand.

The bootstrap uses 4,000 resamples by default and an explicit RNG seed (20261104); percentile endpoints can differ from earlier runs that used another seed or a different weighting convention. They are approximate conditional intervals, not a proof of real-world robustness. No family-wise correction is applied to these exploratory contrasts.

Persistent error lists and class differences prioritize listening/spectrogram review. They do not identify annotation errors, causal acoustic mechanisms or information inaccessible to a model. Roughly eight unique clips per class remain roughly eight clips after three training seeds.

This work uses already-exposed development validation audio. Existing reports and the 65 historical training tasks are preserved. No retraining or test-fold inference is performed.
