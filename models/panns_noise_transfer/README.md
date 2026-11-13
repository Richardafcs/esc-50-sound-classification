# Gaussian-adapted PANNs classifiers

These two heads share the frozen official PANNs Cnn14_16k encoder and the scaler fitted on clean training embeddings.

| File | Training inputs | Clean accuracy | Gaussian 10-dB accuracy |
| --- | --- | ---: | ---: |
| `clean_refit_head.npz` | Clean recordings | 91.21% | 69.85% |
| `gaussian_adapted_head.npz` | Clean and Gaussian-augmented recordings | 91.96% | 83.92% |

Both use L2 multinomial logistic regression with C=1. Gaussian adaptation uses the same 1,200 foreground training clips: each contributes clean weight 0.50 and two augmented variants of weight 0.25 each. SNR assignments range from 10–30 dB. The total training weight remains 1,200.

Each NPZ stores `mean`, `scale`, `coef`, `intercept` and `classes`. Prediction standardizes a 2,048-dimensional embedding, computes `embedding @ coef.T + intercept`, and maps the maximum through `classes`. The official encoder is required separately.

The [Gaussian-adaptation notebook](../../notebooks/05_gaussian_adaptation.ipynb) contains the executed comparison. [Results and settings](../../reports/results/gaussian_adaptation/README.md) distinguish the Gaussian benefit from uncertain transfer on the initial recorded-noise panel. These heads are different fits from the original classifier in [panns_linear/](../panns_linear/README.md).
