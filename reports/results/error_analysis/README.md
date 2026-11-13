# Prediction errors and paired comparisons

This analysis compares 24 clean prediction sets: eight model variants, each trained with three seeds, on the same 398 development clips from 295 sources. Paired correctness counts distinguish errors repaired by another model from newly introduced errors.

Across all three seeds, the CNN repeatedly misclassifies 83 clips and the BiGRU 64. Four model families—CNN, BiGRU, channel attention and noise-trained CNN—share 41 persistently incorrect clips from 31 sources. These recordings are useful candidates for listening and spectrogram inspection; predictions alone do not establish label errors or acoustic causes.

Class-level comparisons use roughly eight clips per category. Training seeds repeat predictions on the same recordings and do not increase the number of independent examples. Source-bootstrap intervals use 4,000 paired resamples and condition on the fitted models and fixed perturbations.

[Results notebook](../../../notebooks/01_results.ipynb) · [Paired clean results](paired_clean.csv) · [Class results](class_accuracy.csv) · [Recurring errors](persistent_errors.csv) · [Confusions](confusions.csv) · [Example recordings](error_cases/case_index.csv) · [Summary](summary.json)

The `robustness/` subdirectories contain the additional Gaussian/gain predictions used by the [robustness comparison](../robustness/README.md); `convergence/` contains feature-study learning-curve summaries.
