# Comparing Gaussian and recorded-noise training

With the PANNs encoder, clean-training scaler, classifier objective and augmentation budget fixed, this experiment compares Gaussian and recorded-background training. Recorded training improves performance on the reserved recorded panel, but reduces Gaussian-noise accuracy.

| Linear classifier | Clean accuracy | Gaussian 10 dB | Recorded-grid accuracy | Mean recorded macro-F1 | Mean recorded NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean-trained | 91.21% | 69.85% | 82.88% | 83.11% | 0.791 |
| Gaussian-adapted | 91.96% | 83.92% | 83.43% | 83.78% | 0.772 |
| Recorded-adapted | 91.71% | 73.62% | 84.79% | 85.18% | 0.699 |

Recorded minus Gaussian training gives **+1.36 percentage points [0.41, 2.32]** on the recorded grid. The corresponding Gaussian-10-dB difference is **−10.30 pp [−13.84, −6.95]**; the clean difference is **−0.25 pp [−1.86, 1.45]**. Intervals are conditional 95% foreground-source bootstrap intervals. Secondary comparisons are descriptive and unadjusted.

## Training and evaluation

The recorded classifier uses the same 1,200 training targets, each weighted 0.50 clean and 0.25 for each of two augmented variants. It reuses the Gaussian experiment's per-target SNR assignments from 10–30 dB and the same L2 logistic-regression settings. Total training weight remains 1,200. Two distinct backgrounds are assigned independently of foreground class, with balanced recording exposure and uniformly sampled crop positions.

| DEMAND category | Training recording | Evaluation recording |
| --- | --- | --- |
| Domestic | DKITCHEN | DLIVING |
| Nature | NFIELD | NPARK |
| Office | OMEETING | OOFFICE |
| Public | PRESTO | PCAFETER |
| Street | SPSQUARE | STRAFFIC |
| Transport | TBUS | TMETRO |

Evaluation uses 16-kHz channel one, segments at 60–65 and 210–215 seconds, and SNRs of 0, 10 and 20 dB. The complete grid contains **14,328 mixtures of 398 foreground clips from 295 sources**. Its mean equally weights six recordings, two segments and three SNRs. Macro-F1 is calculated within each condition before averaging.

The original clean and Gaussian heads are retained; one recorded-background head is fitted. It converged in 169 iterations. Extraction, fitting and evaluation took 243.48 seconds on Colab Tesla T4, excluding setup/downloads.

## Interpretation

The result demonstrates a training-distribution tradeoff on this fixed panel. Gaussian and recorded interference differ in spectral, temporal and semantic content, so the comparison does not isolate any single acoustic mechanism. Mixtures use full-window added-noise SNR without clipping or peak normalization; the label remains the designated foreground category even if another event is audible.

The 4,000 paired bootstrap draws resample foreground sources, retaining clips and fixed conditions. They do not estimate uncertainty across future environments, training seeds or augmentation draws. Official target folds have historical exposure, pretraining overlap is unresolved, and the six reserved recordings are not a representative deployment sample.

[Notebook](../../../notebooks/06_recorded_noise.ipynb) · [Protocol](protocol.json) · [Related work](literature_review.md) · [Per-condition metrics](metrics.csv) · [Aggregates](aggregate_metrics.csv) · [Paired effects](paired_effects.csv) · [Predictions](predictions.csv) · [Training exposure](training_exposure.csv) · [Backgrounds](background_segments.csv) · [Summary](summary.json) · [Classifier](../../../models/panns_noise_distribution/recorded_adapted_head.npz)
