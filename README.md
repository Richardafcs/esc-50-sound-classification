# ESC-50 Environmental Sound Classification

CNN, BiGRU, attention, audio augmentation and frozen PANNs comparisons on ESC-50.

## Results notebooks

Latest: [Transfer robustness and matched MLP heads](notebooks/ESC50_Transfer_Followup.ipynb) · [Run follow-up in Colab](https://colab.research.google.com/github/Richardafcs/esc-50-sound-classification/blob/main/notebooks/ESC50_Transfer_Followup.ipynb) · [Research overview](reports/research_overview.md)

[Open the complete notebook](notebooks/ESC50_Results.ipynb) · [Run in Google Colab](https://colab.research.google.com/github/Richardafcs/esc-50-sound-classification/blob/main/notebooks/ESC50_Results.ipynb)

The notebook displays model metrics, paired comparisons, all 26 scientific figures, noise robustness, temporal interventions, feature experiments, per-class errors, the full experiment ledger and the literature review. It can optionally repeat the original frozen PANNs extraction and fit. The separate follow-up notebook executes the matched robustness and fixed-budget head experiment. Previously completed CNN/CRNN experiments are identified separately.

## Project layout

```text
notebooks/        Executable research and saved outputs
src/esc50_sound/  Models, audio processing, training and evaluation
scripts/          Experiment runners
configs/          Experiment protocols and settings
data/ESC-50/      Official metadata, license and recording credits
models/           Portable CNN/CRNN models and the PANNs linear head
reports/          Compact numerical evidence and original figures
docs/             Methods and model descriptions
```

Both notebooks retain executed Colab Tesla T4 outputs. The earlier PANNs extraction/refit reproduced 363/398 correct. The new fixed study completed matched clean/10 dB inference and six 100-epoch head fits in 88.73 seconds, excluding setup/downloads.

## Main findings

| System | Development accuracy | Macro-F1 |
| --- | ---: | ---: |
| Research CNN | 72.61% ± 0.25 pp | 71.39% ± 0.58 pp |
| CNN + BiGRU | 73.87% ± 1.26 pp | 72.97% ± 1.36 pp |
| BiGRU + channel attention | 74.37% ± 1.15 pp | 73.48% ± 1.00 pp |
| Frozen PANNs + linear classifier | 91.21% | 91.10% |

Scratch-model values are mean ± sample SD over three fixed fits. PANNs uses one fixed head fit. These are development results on 398 clips from 295 sources; all official folds have historical exposure. External pretraining, capacity and frontend differences prevent causal attribution of the PANNs gain to one factor.

The research CNN has 49,058 parameters. It differs from the team's original 1,368,386-parameter CNN, whose score is not inferred here.

## Matched robustness and MLP follow-up

At 10 dB, frozen PANNs + the existing logistic head achieves **69.85%**, versus **61.56%** for three noise-trained CNNs. The paired advantage is **+8.29 pp [1.83, 14.74]**, conditional source-bootstrap 95% interval. PANNs loses more accuracy from clean input (21.36 pp versus 11.56 pp); higher noisy accuracy does not imply better accuracy retention.

The new parameter-matched PANNs heads use 2048→49→50 and 102,901 parameters. Factorized linear averages 91.12% clean/68.76% noisy; ReLU averages 89.78%/67.34%. There is no observed gain from this ReLU configuration. Earlier pooled/tokenwise CNN-MLP controls (72.28%/69.68%) have been restored to the complete comparison.

All original 22 scientific figures and four new follow-up figures are retained. See the [experiment evidence](reports/transfer_followup_20261105/README.md) for all metrics, paired intervals and limitations.

## Run locally

Use Python 3.11 or newer:

```sh
python -m pip install -e '.[baseline,notebook,transfer]'
python -m esc50_sound --help
```

Open the notebook for the complete workflow. Select a Colab GPU runtime for PANNs extraction. Raw audio, feature caches, bulk run weights and document-generation scripts are excluded from Git.

## Data and references

Dataset: [official ESC-50](https://github.com/karolpiczak/ESC-50). Retain the [CC BY-NC 3.0 license](data/ESC-50/LICENSE) and [recording credits](data/ESC-50/meta/esc50.csv).

Pretrained model: [official PANNs implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn). The notebook and [literature review](reports/checkpoint_20261105/literature_review.md) describe the sources and implemented adaptations.
