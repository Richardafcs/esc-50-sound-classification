# Transfer robustness and matched heads - 5 November 2026

[Executed experiment notebook](../../notebooks/ESC50_Transfer_Followup.ipynb) · [Complete project notebook](../../notebooks/ESC50_Results.ipynb) · [Fixed protocol](protocol.json)

| System | Clean accuracy | 10 dB accuracy | Clean macro-F1 | 10 dB macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Frozen PANNs + existing logistic head | 91.21% | 69.85% | 91.10% | 71.53% |
| Three noise-trained CNNs, mean | 73.12% | 61.56% | 71.99% | 59.21% |
| PANNs + factorized-linear head, mean | 91.12% | 68.76% | 91.05% | 69.91% |
| PANNs + ReLU head, mean | 89.78% | 67.34% | 89.70% | 68.19% |

PANNs' primary absolute noisy-accuracy advantage is **+8.29 pp [1.83, 14.74]**, conditional 95% source-bootstrap interval. PANNs loses **21.36 pp** from clean, compared with **11.56 pp** for noise-trained CNN. The difference in losses is **+9.80 pp [3.19, 16.62]**. Higher noisy accuracy and fewer points lost are different outcomes; model capacity, training and frontends also differ.

The ReLU-minus-factorized-linear difference is **−1.34 pp [−2.52, −0.26]** on clean audio and **−1.42 pp [−3.05, +0.24]** at 10 dB. Both heads use 2048→49→50, 102,901 parameters, matched initialization/order, three seeds and exactly 100 fixed epochs. No validation or noisy result selects a checkpoint. The original logistic head is unchanged. This configuration did not improve the observed performance; it is not a universal conclusion about MLPs.

The linear factorization's rank≤49 does not restrict the family of 50-class affine softmax probabilities: a common logit offset is irrelevant, and centered logits have rank≤49. Optimization and regularization still differ from ordinary L-BFGS logistic regression.

## Evidence

- [All 20 model/seed/condition metrics](metrics.csv) and [group summaries](groups.csv).
- [Paired effects](paired_effects.csv), [accuracy losses](accuracy_losses.csv), [all six final head fits](head_fits.csv), and six complete training histories.
- [Clean-to-noisy transition counts](clean_noise_transitions.csv) and [common-clean-correct subsets](common_clean_correct.csv). These subset-specific exploratory comparisons do not replace the primary full-validation result.
- [All class-level recall slices](per_class.csv) and per-model prediction CSVs.
- [Runtime summary](summary.json). All four fixed systems reproduce their earlier clean labels; 10 dB measured SNR ranges from 9.9999988 to 10.0000013 dB.

The fixed experiment took 88.73 seconds on Colab Tesla T4, excluding setup/downloads and original pretraining. It performed six small head fits, no backbone training and no test-WAV evaluation. The primary panel uses newly shared seed-sequence-v2 perturbations; the historical 61.31% noise-CNN value belongs to different draws and remains separate.

All official folds have historical exposure and pretraining overlap is unresolved. Bootstrap intervals condition on fixed fits, observed recording sources and one Gaussian draw per clip. Multiple secondary intervals are descriptive and have no familywise adjustment. Realistic-noise and independent-source generalization remain unestablished. Stop further head/attention tuning on this validation set; independent-source evaluation is the higher-value next question.
