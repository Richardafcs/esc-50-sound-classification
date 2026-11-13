# Saved models

The project uses one primary baseline: the frequency-preserving CNN trained from scratch. The other models are architectural, augmentation or pretrained comparisons.

| Directory | Contents | Role |
| --- | --- | --- |
| [cnn/](cnn/README.md) | Seed-101 CNN checkpoint | Primary baseline |
| [crnn/](crnn/README.md) | Seed-101 CNN–BiGRU checkpoint | Recurrent comparison |
| [noise_cnn/](noise_cnn/README.md) | CNN checkpoints for seeds 101, 202 and 303 | Gaussian-augmentation comparison |
| [panns_linear/](panns_linear/README.md) | Clean-trained linear head and label map | Frozen pretrained reference |
| [panns_noise_transfer/](panns_noise_transfer/README.md) | Clean-refit and Gaussian-adapted linear heads | Training-noise comparison |
| [panns_noise_distribution/](panns_noise_distribution/README.md) | Recorded-background-adapted linear head | Recorded-noise comparison |

Each folder describes its model, inputs and usage. The CNN and CRNN exports include their normalization and class mapping. The PANNs exports contain downstream classifier weights; the official encoder must be obtained separately.

Results are summarized in the [course report](../reports/report.md) and [results notebook](../notebooks/01_results.ipynb). A single exported checkpoint is distinct from the three-seed mean reported for a model family. Scores are not calibrated, and the models have no validated general unknown-sound rejection capability. See the [evaluation protocol](../docs/evaluation.md) for split and uncertainty definitions.
