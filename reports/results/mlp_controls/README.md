# CNN classifier-capacity controls

Pooled and tokenwise MLPs provide nonrecurrent comparisons with approximately the same parameter count as CNN–BiGRU. Both share the convolutional encoder and contain **252,036 total parameters**. Each configuration uses seeds 101, 202 and 303 and the same 398 development clips.

| Model | Mean development accuracy |
| --- | ---: |
| CNN baseline | 72.61% |
| CNN–BiGRU | 73.87% |
| CNN + pooled MLP | 72.28% |
| CNN + tokenwise MLP | 69.68% |

Increasing classifier capacity through these MLP configurations does not reproduce the BiGRU result. The comparison does not isolate temporal ordering as the sole cause: optimization and aggregation also differ. These models use scratch CNN features; the separate [PANNs head experiment](../transfer_heads/README.md) evaluates nonlinearity on pretrained embeddings.

CNN and BiGRU reference fits overlap the main comparison and are not additional independent replications.

[Results notebook](../../../notebooks/01_results.ipynb) · [Comparison](comparison.csv) · [Fit metrics](runs.csv) · [Paired differences](paired_differences.csv) · [Predictions](validation_predictions.csv)
