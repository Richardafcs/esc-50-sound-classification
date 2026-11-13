# Experiment configurations

| Directory | Purpose |
| --- | --- |
| [studies/](studies/) | Repeated capacity controls and attention study |
| [evaluation/](evaluation/) | Development evaluation settings |
| [examples/](examples/) | Training examples for clean, noisy and Mixup CNNs, including a small CPU configuration |
| [historical/](historical/) | Original model matrix, earlier pilots, selection settings and historical fold evaluations |

The study JSON files record experiment settings. The [original October matrix](historical/model_comparison_20261028.json) contains proposals that were not executable at that stage; its advanced-model status is historical. The current completed studies are indexed in [experiment results](../reports/results/README.md). They are read by the relevant experiment workflows rather than by a single generic configuration loader. Use the [CLI](../scripts/README.md) for training commands and [evaluation notes](../docs/evaluation.md) for the interpretation of their splits.

Main scratch comparisons use seeds 101/202/303, 60 epochs, batch size 32, AdamW with learning rate 0.001 and weight decay 0.0001, cosine scheduling, and minimum development cross-entropy checkpoint selection. Matched PANNs-head fits use a separate fixed 100-epoch protocol, recorded with their [results](../reports/results/transfer_heads/protocol.json).
