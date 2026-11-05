# Gaussian-to-recorded-noise transfer — 6 November 2026

[Executed notebook](../../notebooks/ESC50_Noise_Transfer.ipynb) · [Complete study](../../notebooks/ESC50_Results.ipynb) · [Fixed protocol](protocol.json)

**Question:** Does Gaussian training exposure for a linear head on frozen PANNs features transfer to recorded backgrounds excluded from training?

| System | Clean accuracy | Gaussian 10 dB | Recorded mean accuracy | Mean per-condition macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Matched clean LR | 91.21% | 69.85% | 89.07% | 89.15% |
| Gaussian-adapted LR | 91.96% | 83.92% | 90.12% | 90.21% |
| Historical LR reference | 91.21% | 69.85% | 89.07% | 89.15% |

The primary adapted-minus-clean difference is **+1.05 percentage points**, conditional 95% source-bootstrap interval **[-0.54, +2.59]**. The recorded-background accuracy difference remains uncertain under the conditional interval. Clean accuracy changes by **+0.75 pp** and Gaussian accuracy by **+14.07 pp**. These outcomes must be assessed jointly; no acceptable clean-loss margin or clean noninferiority claim is supplied.

## What was held fixed

Two new converged L2 multinomial logistic heads use the same frozen official Cnn14_16k encoder, 2,048-dimensional features, clean-training StandardScaler, C=1, L-BFGS, tolerance 1e-6 and 3,000 maximum iterations. Each original clean training clip has total weight one in either objective. The adapted objective allocates 0.5 to clean and 0.25 to each of two waveform-level Gaussian variants at 10–30 dB. Both total weights remain 1,200, preserving effective regularization. The historical logistic model is unchanged and retained separately.

## Evaluation population and inference

The source-strict development panel contains 398 clips from 295 original recording sources. Six fixed DEMAND conditions use channel 1 of NRIVER, OHALLWAY and TCAR at 30–35 s and 150–155 s, mixed at 10 dB with every target. Each target therefore contributes equally to the average of the six conditions. There are 2,388 target/condition pairs, not 2,388 independent target recordings. Backgrounds were selected for metadata-based diversity before predictions, with no listening-based eligibility claim or outcome-based exclusion.

The task is classifying the designated foreground recording in an additive mixture. Backgrounds may contain other recognizable events. Added-noise SNR uses the original recording's full-window power, not an estimate of event-only SNR. Mixtures are not clipped or peak-normalized.

The paired interval resamples `src_file` 4,000 times after averaging differences over fixed conditions within clips. It conditions on the fitted heads and six selected segments; it does not describe uncertainty over arbitrary future environments, new training exposures or all possible target populations. Environment-specific and other secondary intervals are descriptive and unadjusted. All official target folds have historical exposure, and pretraining overlap remains unresolved. No new-target or field-deployment claim follows.

## Execution and evidence

Colab Tesla T4 completed the fixed comparison in **174.90 seconds**, excluding setup/downloads. New head fits: 2; encoder fits: 0; test WAV reads: 0. Historical reference correctness is 363/398 clean and 278/398 Gaussian. Both fit convergence records are saved.

- [All conditions](metrics.csv), [recorded-condition means](recorded_means.csv), [paired effects](paired_effects.csv), and [error transitions](transitions.csv).
- [Per-example predictions and true-class probabilities/margins](predictions.csv).
- [Training exposures](training_exposure.csv), [head fits](head_fits.csv), [background provenance](background_segments.csv), [realized SNR](realized_snr.csv), and [runtime summary](summary.json).
- All three figures appear in the executed notebook and complete study.

No extra head, noise policy or encoder fine-tuning was selected after inspecting the results. A negative adaptation result would not prove irreversible information loss, and a positive result would not establish an invariant representation. The original default head remains available.

Sources: [official PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), [ESC-50](https://github.com/karolpiczak/ESC-50), [DEMAND](https://zenodo.org/records/1227121). DEMAND author documentation licenses the recordings under CC BY-SA 3.0. Raw audio is not distributed here.
