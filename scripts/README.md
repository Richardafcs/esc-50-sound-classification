# Experiment commands

Core preparation, training and single-clip inference use the `esc50` command-line interface. The scripts below run the supplementary experiments.

## Prepare data and train

From the repository root, install the dependencies and build the log-Mel feature cache:

```sh
python -m pip install -e '.[baseline,notebook,transfer]'
python -m esc50_sound prepare --root data/ESC-50 --cache data/esc50_features.npz
```

A baseline training command using the main study recipe is:

```sh
python -m esc50_sound train --model cnn_freq --cache data/esc50_features.npz --output results/runs/cnn_seed101 --epochs 60 --scheduler cosine --seed 101 --batch-size 32 --lr 0.001 --device cuda --max-seconds 1800
```

The wall-time argument is an execution limit, not a measured training duration. Use `crnn_freq` for CNN–BiGRU and `crnn_freq_ca` for its channel-attention variant. The model matrix records all three seeds and the remaining settings in [configs/studies/](../configs/studies/). See `python -m esc50_sound train --help` for augmentation options.

## Supplementary experiments

| Script | Experiment | Main inputs |
| --- | --- | --- |
| `run_panns_linear_probe.py` | Frozen PANNs extraction and logistic classifier | ESC-50 WAVs, official PANNs source and checkpoint |
| `run_robustness_three_seeds.py` | Gaussian-noise and gain evaluation | Matching feature cache, WAVs and trained CNN/CRNN checkpoints |
| `run_temporal_order_probe.py` | Interventions on encoded time tokens | Trained CNN/CRNN checkpoints and input features |
| `run_transfer_followup.py` | Matched Gaussian inputs and linear/ReLU PANNs heads | Audio, pretrained encoder and saved noise-CNN checkpoints |
| `run_external_fsd.py` | Six-category external recording evaluation | FSD recordings and existing model files |
| `run_duration_probe.py` | Padding and native-duration comparison | Eligible short recordings and the frozen PANNs classifier |
| `run_noise_transfer.py` | Gaussian-adapted linear head | Audio, frozen encoder and recorded-background assets |
| `run_noise_distribution.py` | Recorded-background-adapted linear head | Training/evaluation backgrounds and existing adapted-head settings |
| `feature_ablation.py` | Audio representation and sampling comparisons | Audio and the feature-study settings |

Run each script with `--help` for its arguments. The [numbered notebooks](../notebooks/README.md) provide the corresponding setup and executed analyses. Their default mode reads existing results; GPU model computation is optional. New outputs belong in `results/`, while the completed experiment tables and figures are grouped in [reports/results/](../reports/results/README.md).
