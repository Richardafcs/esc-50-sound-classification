# Recorded-background-adapted PANNs classifier

`recorded_adapted_head.npz` contains a 50-class L2 multinomial logistic classifier on frozen PANNs Cnn14_16k embeddings. It stores the clean-training scaler, coefficients, intercepts and class IDs. The official encoder is required separately.

The same 1,200 training clips contribute clean weight 0.50 and two recorded-background variants of weight 0.25 each. Six DEMAND recordings supply training backgrounds. The comparison reuses the Gaussian experiment's SNR assignments and matches the classifier objective and total training weight.

| Evaluation | Accuracy |
| --- | ---: |
| Clean development clips | 91.71% |
| Gaussian interference at 10 dB | 73.62% |
| Reserved recorded-background grid | 84.79% |

On the reserved grid, the advantage over the Gaussian-adapted head is **+1.36 percentage points**, with a conditional 95% source-bootstrap interval of **[+0.41, +2.32]**. Gaussian 10-dB accuracy is 10.30 points lower, so the result is a distribution-dependent tradeoff.

See the [recorded-noise notebook](../../notebooks/06_recorded_noise.ipynb), [protocol](../../reports/results/recorded_noise/protocol.json) and [results](../../reports/results/recorded_noise/README.md). Inference follows the [existing embedding predictor](../panns_linear/predict_embeddings.py), using this NPZ as `--head`. No DEMAND audio is included.
