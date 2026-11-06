# Training-noise distribution transfer — 7 November 2026

[Executed notebook](../../notebooks/ESC50_Noise_Distribution.ipynb) · [Complete study](../../notebooks/ESC50_Results.ipynb) · [Fixed protocol](protocol.json) · [Primary-source review](literature_review.md)

**Question:** With the frozen PANNs representation, clean-training scaler, classifier objective and augmentation budget fixed, does recorded-background training outperform Gaussian training under reserved recorded backgrounds?

**Main finding:** Recorded-background training improves the fixed recorded panel over Gaussian training, while substantially reducing Gaussian-10-dB accuracy. This is a measured distribution-dependent tradeoff.

## Results

| Readout | Clean accuracy | Gaussian 10 dB | Recorded-grid accuracy | Mean recorded macro-F1 | Mean recorded NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean-trained LR | 91.21% | 69.85% | 82.88% | 83.11% | 0.791 |
| Gaussian-adapted LR | 91.96% | 83.92% | 83.43% | 83.78% | 0.772 |
| Recorded-adapted LR | 91.71% | 73.62% | 84.79% | 85.18% | 0.699 |

The primary recorded-minus-Gaussian accuracy difference is **+1.36 percentage points**, conditional 95% target-source bootstrap interval **[+0.41, +2.32]**. Recorded exposure improves accuracy over Gaussian exposure on the fixed recorded panel. Their clean difference is **-0.25 pp [-1.86, +1.45]**; their Gaussian-10-dB difference is **-10.30 pp [-13.84, -6.95]**. Secondary intervals are descriptive and unadjusted. No practical or clean noninferiority margin was specified, and no operating-condition frequency is assumed.

The recorded-grid mean equally weights six recordings, two windows and three SNRs. Macro-F1 is computed separately within each condition before averaging; it is not a pooled repeated-target F1. The notebook retains full degradation curves, recording-specific differences, clean-to-noisy losses, leave-one-recording-out sensitivity and exploratory class errors. Favorable subsets do not replace the primary result.

## Controlled intervention

The official `Cnn14_16k` encoder remains frozen, producing 2,048-dimensional features. Both existing clean and Gaussian readouts are reused. Exactly one new L2 multinomial logistic head is fitted with C=1, L-BFGS, tolerance 1e-6 and maximum 3,000 iterations in scikit-learn 1.6.1. Training preserves the prior float32 StandardScaler transformation.

Each of the same 1,200 training targets contributes clean weight 0.50 and two waveform-level variants with weight 0.25 each. Total training weight remains 1,200, preserving effective regularization. The recorded head reads the Gaussian study's exact saved per-target SNR assignments; those originally range uniformly over 10–30 dB. No SNR is redrawn to favor the new head.

A seed-20261107 permutation assigns two distinct training recordings to every target. Each recording appears 200 times per variant and every ordered distinct pair appears 40 times. Crop positions use a separate seed, 20261108, and are sampled uniformly from valid five-second positions. Assignment does not depend on foreground class. This is one fixed augmentation realization, not repeated evidence about the training procedure.

## Background separation and task

| DEMAND category | Training | Evaluation |
| --- | --- | --- |
| Domestic | DKITCHEN | DLIVING |
| Nature | NFIELD | NPARK |
| Office | OMEETING | OOFFICE |
| Public | PRESTO | PCAFETER |
| Street | SPSQUARE | STRAFFIC |
| Transport | TBUS | TMETRO |

All inputs use official 16-kHz channel 1. Previously exposed NRIVER, OHALLWAY and TCAR are excluded entirely. Evaluation windows are fixed at 60–65 and 210–215 seconds, crossed with 0, 10 and 20 dB and every development target. The resulting **14,328 target/condition pairs** contain **398 distinct foreground clips from 295 sources**, plus six evaluation background recordings. Channels, windows and SNRs do not create independent environments.

Backgrounds are mean-centered and scaled using full-window foreground/added-noise power. Mixtures are not clipped or peak-normalized. Equal waveform SNR is not equal perceptual difficulty. Backgrounds may contain other ESC-50 events; the task measures recovery of the **designated original foreground label**, not every audible event. No class-neutral or listening-validated background claim is made.

The choices were fixed before this new fit and predictions, informed by earlier exposed results. This was not an externally preregistered or newly blinded foreground benchmark. Required sources were not replaced and evaluation conditions were not screened by their accuracies.

## What the result can establish

Recorded exposure improves accuracy over Gaussian exposure on the fixed recorded panel. The complete training-noise distribution changes, including spectral structure, temporal structure and semantic content. The comparison does not isolate diversity, coverage breadth or a particular acoustic mechanism. Neither an adverse nor an uncertain outcome proves information loss or a need for encoder fine-tuning.

The 4,000 paired bootstrap draws resample foreground `src_file` groups, retain all clips and fixed conditions, and use the sampled clip-count denominator. Intervals condition on the fitted heads, training augmentation draw, six recordings, segments and grid. They do not cover environmental-population or training-seed uncertainty. All official target folds have historical exposure; independence from the encoder's pretraining audio is unresolved. The reserved recordings share the DEMAND collection context and are not a representative deployment sample.

There is no automatic replacement of the default head or additional model search. Clean performance, Gaussian performance and recorded-background performance answer different operating needs. An unweighted average of these three regimes is not a deployment objective.

## Execution and evidence

Colab **Tesla T4** completed the scientific runner in **243.48 seconds**, including audio reads, feature extraction, fitting and evaluation but excluding setup/downloads. The new head converged in **169 iterations**. Encoder fits: 0; new head fits: 1; test WAV reads: 0. A connection failure before script execution was recovered in the same session; it produced no head fit. The fixed comparison was then completed without changing its protocol.

- [Per-condition metrics](metrics.csv), [aggregate metrics](aggregate_metrics.csv), [paired effects](paired_effects.csv), [mixture error transitions](transitions.csv).
- [Per-example predictions](predictions.csv), [training exposures](training_exposure.csv), [fit record](head_fits.csv), [conditions](conditions.csv).
- [Background signal descriptors](background_segments.csv), [realized SNR](realized_snr.csv), [execution summary](summary.json).
- [Recorded-adapted head](../../models/panns_noise_distribution/recorded_adapted_head.npz).

All four new scientific figures are present in this executed notebook and the complete study. Source review records what was inspected and how nearby papers differ. No further head, SNR grid or architecture was selected after seeing these outcomes.
