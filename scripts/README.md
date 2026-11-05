# Experiment runners

| Experiment | Entry point |
| --- | --- |
| Frozen PANNs extraction and linear classifier | `run_panns_linear_probe.py` |
| Gaussian-noise and gain inference | `run_robustness_three_seeds.py` |
| CNN/CRNN encoder-token interventions | `run_temporal_order_probe.py` |
| Feature and sampling experiments | `feature_ablation.py` |

Use `--help` for arguments. The [results notebook](../notebooks/ESC50_Results.ipynb) contains the executed analysis and figures. Report and PDF generation utilities are excluded from this repository.
