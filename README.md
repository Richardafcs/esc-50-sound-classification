# ESC-50 Environmental Sound Classification

CNN, BiGRU, attention, audio augmentation and frozen PANNs comparisons on ESC-50.

This project completes the supplied COMP9444 model-improvement checkpoint using an independent CNN baseline, implemented improvements and a pretrained alternative. The course report is the main submission narrative; executed notebooks provide the supporting evidence.

## Course checkpoint — completed

The course deliverable analyzes the CNN baseline, implements CRNN/attention/augmentation improvements, evaluates a pretrained alternative, reports appropriate metrics, and compares positive and negative results. All five supplied checkpoint activities are covered by completed experiments.

**Submission entry:** [English checkpoint report](reports/checkpoint_20261105/Checkpoint_Report_EN.md) · [Executed results notebook](notebooks/ESC50_Results.ipynb) · [Open in Colab](https://colab.research.google.com/github/Richardafcs/esc-50-sound-classification/blob/main/notebooks/ESC50_Results.ipynb)

The core results are CNN 72.61%, CNN–BiGRU 73.87%, channel attention 74.37%, and frozen PANNs plus a linear classifier 91.21% development accuracy. The report explains the shared evaluation, macro-F1, robustness, complexity and limitations. The course checkpoint is complete; no further experiments are planned for this delivery.

Completed noise-transfer, external-recording and input-policy studies remain available as supporting evidence. They are optional extensions, not outstanding course requirements. [Additional completed experiments](reports/research_overview.md).

## Results notebooks

Supporting studies: [Transfer robustness and matched MLP heads](notebooks/ESC50_Transfer_Followup.ipynb) · [External recording evaluation](notebooks/ESC50_External_Transfer.ipynb) · [Short-recording input-policy diagnostic](notebooks/ESC50_Duration_Diagnostic.ipynb) · [Research overview](reports/research_overview.md)

[Open the complete notebook](notebooks/ESC50_Results.ipynb) · [Run in Google Colab](https://colab.research.google.com/github/Richardafcs/esc-50-sound-classification/blob/main/notebooks/ESC50_Results.ipynb)

The notebook displays model metrics, paired comparisons, all 40 scientific figures, noise robustness, temporal interventions, feature experiments, per-class errors, the full experiment ledger and the literature review. It can optionally repeat the original frozen PANNs extraction and fit. The separate follow-up notebook executes the matched robustness and fixed-budget head experiment. Previously completed CNN/CRNN experiments are identified separately.

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

All six notebooks retain executed Colab outputs. The latest scientific comparison used Tesla T4 and completed one converged LR fit plus the full fixed evaluation in 243.48 seconds, excluding setup/downloads. The complete notebook was also executed on Colab from saved evidence. Earlier extraction/refitting and the six fixed-budget head fits are separate stages.

## Main findings

| System | Development accuracy | Macro-F1 |
| --- | ---: | ---: |
| Research CNN | 72.61% ± 0.25 pp | 71.39% ± 0.58 pp |
| CNN + BiGRU | 73.87% ± 1.26 pp | 72.97% ± 1.36 pp |
| BiGRU + channel attention | 74.37% ± 1.15 pp | 73.48% ± 1.00 pp |
| Frozen PANNs + linear classifier | 91.21% | 91.10% |

Scratch-model values are mean ± sample SD over three fixed fits. PANNs uses one fixed head fit. These are development results on 398 clips from 295 sources; all official folds have historical exposure. External pretraining, capacity and frontend differences prevent causal attribution of the PANNs gain to one factor.

The research CNN has 49,058 parameters. The separate team implementation is outside this study.

## Matched robustness and MLP follow-up

At 10 dB, frozen PANNs + the existing logistic head achieves **69.85%**, versus **61.56%** for three noise-trained CNNs. The paired advantage is **+8.29 pp [1.83, 14.74]**, conditional source-bootstrap 95% interval. PANNs loses more accuracy from clean input (21.36 pp versus 11.56 pp); higher noisy accuracy does not imply better accuracy retention.

The new parameter-matched PANNs heads use 2048→49→50 and 102,901 parameters. Factorized linear averages 91.12% clean/68.76% noisy; ReLU averages 89.78%/67.34%. There is no observed gain from this ReLU configuration. Earlier pooled/tokenwise CNN-MLP controls (72.28%/69.68%) have been restored to the complete comparison.

All 40 scientific figures, including the four training-noise-distribution figures, are retained. See the [experiment evidence](reports/transfer_followup_20261105/README.md) for all metrics, paired intervals and limitations.

## External recording evaluation

[Executed external notebook](notebooks/ESC50_External_Transfer.ipynb) · [Run in Colab](https://colab.research.google.com/github/Richardafcs/esc-50-sound-classification/blob/main/notebooks/ESC50_External_Transfer.ipynb)

With all 50 output labels active, frozen PANNs achieves **45.17% accuracy** on 259 FSD recordings in six mapped categories, versus **20.08%** for the three-fit noise-CNN mean. The paired difference is **+25.10 pp [+19.82, +30.63]**, conditional recording-bootstrap 95% interval. Five known ESC-50 recording IDs were excluded before inference. No training or noise injection occurs. This is a mapped external subset, not full-50-class confirmation; perceptual duplicates and pretraining exposure remain unresolved. [Protocol and evidence](reports/external_fsd_20261105/README.md).

## Short-recording input policy

[Executed diagnostic notebook](notebooks/ESC50_Duration_Diagnostic.ipynb) · [Protocol and evidence](reports/duration_probe_20261105/README.md)

On the same 102 eligible short recordings, original right-padding yields **27.45%**, native duration **33.33%**, and left-padding **26.47%** accuracy. Native minus right is **+5.88 pp [−0.98, +12.75]**; its accuracy gain is uncertain. Native and left-padding change 46 and 50 predictions, respectively. This supports input-policy sensitivity without identifying a single internal padding mechanism. The original external benchmark and default inference policy are retained; no model is fitted or selected.

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
