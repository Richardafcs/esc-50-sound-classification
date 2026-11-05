# Experiment runners

| Experiment | Entry point |
| --- | --- |
| Frozen PANNs input-duration diagnostic | `run_duration_probe.py` |
| External mapped FSD recording evaluation | `run_external_fsd.py` |
| Matched transfer robustness and six head fits | `run_transfer_followup.py` |
| Frozen PANNs extraction and linear classifier | `run_panns_linear_probe.py` |
| Gaussian-noise and gain inference | `run_robustness_three_seeds.py` |
| CNN/CRNN encoder-token interventions | `run_temporal_order_probe.py` |
| Feature and sampling experiments | `feature_ablation.py` |

Use `--help` for arguments. The [results notebook](../notebooks/ESC50_Results.ipynb) contains the executed analysis and figures. Report and PDF generation utilities are excluded from this repository.
