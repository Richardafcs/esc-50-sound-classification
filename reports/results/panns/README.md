# Frozen PANNs with a linear classifier

A frozen AudioSet-pretrained `Cnn14_16k` encoder supplies a 2,048-dimensional embedding for each clip. A training-only StandardScaler and L2 multinomial logistic classifier (`C=1`, L-BFGS) predict the 50 ESC-50 labels.

The classifier correctly identifies **363 of 398 development clips: 91.21% accuracy and 91.10% macro-F1**. The head has 102,450 fitted parameters; the loaded frozen backbone has approximately 81 million parameters. The original head fit converged in 137 iterations.

Compared with the CNN baseline's three-fit mean, accuracy is higher by **18.59 percentage points**. The comparison changes pretraining, architecture, capacity and frontend together, so it does not isolate the effect of pretraining alone. It uses one fixed PANNs head fit, not a three-seed PANNs estimate. All official folds have historical exposure, and pretraining overlap is unresolved.

[Results notebook](../../../notebooks/01_results.ipynb) · [Protocol](protocol.json) · [Run summary](run_summary.json) · [Paired comparisons](comparison_summary.json) · [Predictions](validation_predictions.csv) · [Probabilities](validation_probabilities.csv) · [Per-class results](per_class_comparison.csv) · [Model files](../../../models/panns_linear/README.md)
