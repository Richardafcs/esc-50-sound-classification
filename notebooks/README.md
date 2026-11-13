# Notebooks

Start with **[01 — Results](01_results.ipynb)** for the course project. It contains the baseline, architecture and augmentation comparisons, PANNs results, discussion, and all 40 figures. The remaining notebooks provide detailed supplementary experiments.

| Notebook | Question |
| --- | --- |
| [01 — Results](01_results.ipynb) | How do architecture, augmentation and pretrained features affect classification? |
| [02 — Transfer heads](02_transfer_heads.ipynb) | How do the systems compare under matched noise, and does an MLP improve frozen PANNs features? |
| [03 — External recordings](03_external_recordings.ipynb) | How do the existing systems perform on 259 recordings in six mapped external categories? |
| [04 — Input duration](04_input_duration.ipynb) | How do padding and native duration change predictions on short recordings? |
| [05 — Gaussian adaptation](05_gaussian_adaptation.ipynb) | Does Gaussian-augmented head training transfer to recorded background noise? |
| [06 — Recorded noise](06_recorded_noise.ipynb) | How does the training-noise distribution affect performance across noise conditions? |

## Execution

All notebooks include executed analyses of the completed Colab experiments. The default mode reads the result files included in this repository and recomputes the analysis. It does not retrain the scratch models or download model assets.

Install the project with its notebook dependencies before local execution:

```sh
python -m pip install -e '.[baseline,notebook,transfer]'
```

To repeat model computation, enable the relevant `RUN_TRANSFER`, `RUN_EXPERIMENT`, `RUN_EXTERNAL` or `RUN_PROBE` switch and provide the required assets. GPU-dependent branches explain their inputs in the corresponding notebook. Fresh model outputs and regenerated supplementary figures belong in the ignored `results/` directory; the committed result tables describe the completed experiments.

The original 10-dB CNN robustness panel and the later PANNs comparison use different noise draws. Each notebook identifies its panel; their numbers should not be mixed. See the [evaluation protocol](../docs/evaluation.md).
