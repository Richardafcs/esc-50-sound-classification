# Gaussian-noise and gain robustness

The robustness panel evaluates three fixed model families, each with three training seeds, under clean audio, Gaussian noise at 20, 10 and 0 dB, and gain changes of ±6 dB. Each condition uses the same 398 development clips.

At 10 dB, the clean-trained CNN averages **23.37% accuracy**, while the noise-trained CNN averages **61.31%**. The paired improvement is **37.94 percentage points**, with a conditional 95% source-bootstrap interval of **[32.08, 43.88]**. Noise training's benefit is much larger for the matched Gaussian interference than for the tested gain changes.

The intervals resample original recording sources and condition on the fitted models and perturbation realization. They do not establish robustness to arbitrary noise or variation across future training runs. The later [matched PANNs comparison](../transfer_heads/README.md) uses different shared Gaussian draws and reports 61.56% for the same noise-CNN family; the two panels are kept separate.

The attention comparison finds a 0.50-point channel-attention gain over BiGRU, with interval **[−1.22, 2.17]**. This does not establish a reliable incremental attention benefit.

[Results notebook](../../../notebooks/01_results.ipynb) · [Robustness figure](three_seed_robustness.png) · [Per-run metrics](../error_analysis/robustness_runs.csv) · [Paired robustness effects](../error_analysis/paired_robustness.csv) · [Attention effects](attention_paired_effects.json)
