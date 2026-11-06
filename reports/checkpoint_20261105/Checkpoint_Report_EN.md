# Improving Environmental Sound Classification on ESC-50

**COMP9444 — Model Improvement Checkpoint**  
**Completed experiment evidence through 7 November 2026**

## Summary

We implemented a CNN baseline, evaluated recurrent and attention modifications, tested training augmentations, and compared a frozen pretrained PANNs system. The baseline reaches **72.61%** mean development accuracy. CNN–BiGRU reaches **73.87%**, and channel attention reaches **74.37%**. The pretrained alternative reaches **91.21% accuracy and 91.10% macro-F1**. Noise augmentation substantially improves recognition under the tested Gaussian interference, while several other modifications do not improve clean accuracy.

These completed experiments address the checkpoint's five requested activities. This report presents the course argument; the [executed notebook](../../notebooks/ESC50_Results.ipynb) contains the complete results and all 40 scientific figures. Additional experiments are supporting material rather than work required to complete this checkpoint.

## Dataset and common evaluation

ESC-50 contains 2,000 five-second recordings in 50 classes. Our source-strict split uses **1,200 training clips from folds 1–3** and **398 development clips from fold 4**, representing **295 original recording sources**. Metadata source identifiers separate the evaluated training and development recordings. Preprocessing statistics are fitted on training data only.

The scratch models use three fixed training seeds: 101, 202 and 303. Training runs for 60 epochs with batch size 32, AdamW, initial learning rate 0.001, weight decay 0.0001 and cosine scheduling. Checkpoints are selected by minimum development cross-entropy. All official folds have historical exposure, so the scores below are development results rather than an untouched final-test benchmark.

## 1. Analyse baseline performance and identify improvement areas

The baseline converts 16-kHz audio into log-Mel inputs of shape **1 × 64 × 313** using a 512-point FFT and 256-sample hop. Three convolutional stages use 16, 32 and 64 channels, batch normalization, ReLU and pooling. Temporal averaging retains a 512-dimensional frequency-aware representation before the 50-class output. The network has **49,058 trainable parameters**.

Its mean accuracy is **72.61%**, with **71.39% macro-F1**. Three observations motivate improvements:

- Temporal averaging can discard ordering information beyond the local patterns already represented by the CNN. An explicit recurrent stage may help.
- Under fixed Gaussian 10-dB interference, baseline accuracy falls to **23.37%**. Training on noisy examples may reduce this sensitivity.
- Only 1,200 labelled training clips are available. A representation pretrained on a larger audio corpus provides an alternative to learning every feature from scratch.

These are motivations for experiments, not established acoustic explanations for individual errors. Confusion matrices and class-level predictions in the notebook show which categories and clips are confused.

![Three-seed architecture comparison](../figures/matrix_20261028/architecture.png)

## 2. Implement ideas to improve the baseline

**CNN–BiGRU.** The convolutional output is represented as 39 temporal positions with 512 features each. A bidirectional GRU with hidden size 64 produces 128-dimensional outputs for classification. The resulting model has 251,810 parameters and reaches **73.87% accuracy**, a **+1.26 percentage-point** difference from the baseline.

**Attention.** We implemented temporal, channel and combined attention variants. Channel gating reaches **74.37%**, whereas temporal attention reaches **72.28%** and combined attention **72.86%**. Channel attention adds only **0.50 pp** over BiGRU; its conditional source-bootstrap interval spans zero. Therefore, the data do not establish a reliable incremental attention advantage.

**Augmentation.** Gaussian noise is added before feature extraction with probability 0.5 and training SNR sampled uniformly from 10–30 dB. We also implemented time/frequency masking and waveform Mixup. Noise training reaches **73.12% clean accuracy** and substantially improves matched-noise recognition. Masking and Mixup reach **71.86%** and **71.61%** clean accuracy, respectively, and do not improve this baseline.

As capacity controls, pooled and tokenwise MLP alternatives reach **72.28%** and **69.68%**. Increasing classifier capacity alone did not reproduce the observed recurrent-model result in these configurations.

![Augmentation comparison](../figures/matrix_20261028/augmentation.png)

## 3. Experiment with a different approach

We evaluated the official **PANNs Cnn14_16k** encoder pretrained on AudioSet. The encoder remains frozen and returns a 2,048-dimensional embedding for each recording. A StandardScaler fitted on training embeddings precedes an L2 multinomial logistic classifier with C=1 and L-BFGS optimization. The fitted head contains **102,450 parameters**; the loaded frozen backbone contains approximately **81 million parameters**.

This alternative reaches **91.21% accuracy and 91.10% macro-F1**, an **18.59 pp** accuracy difference from the CNN baseline. This is a comparison between complete systems: pretraining data, capacity, architecture and frontend also differ. The experiment does not isolate pretraining as the sole cause of the gain.

![Pretrained alternative and scratch-trained references](../panns_20261105/system_comparison.png)

## 4. Evaluate with appropriate metrics

Accuracy measures the proportion of correctly classified clips. Macro-F1 computes each class's F1 score and averages the 50 class values equally. Confusion matrices and per-class results complement both overall metrics. Scratch results below are **mean ± sample standard deviation over three fixed fits**; the pretrained result uses one fixed head fit. Standard deviations are not confidence intervals.

| System | Development accuracy | Macro-F1 |
| --- | ---: | ---: |
| CNN baseline | 72.61% ± 0.25 pp | 71.39% ± 0.58 pp |
| CNN–BiGRU | 73.87% ± 1.26 pp | 72.97% ± 1.36 pp |
| BiGRU + channel attention | 74.37% ± 1.15 pp | 73.48% ± 1.00 pp |
| CNN + Gaussian augmentation | 73.12% ± 0.87 pp | 71.99% ± 1.23 pp |
| CNN + time/frequency masking | 71.86% ± 1.15 pp | 70.75% ± 1.38 pp |
| CNN + Mixup | 71.61% ± 0.66 pp | 70.65% ± 0.44 pp |
| Frozen PANNs + linear classifier | **91.21%** | **91.10%** |

Source: [model comparison table](model_comparison.csv). The notebook retains learning curves, all-class confusion matrices and error analysis.

For the fixed three-fit Gaussian robustness panel, the clean-trained CNN scores **23.37%** at 10 dB, compared with **61.31%** for the noise-trained CNN. The paired difference is **+37.94 pp**, with a conditional 95% source-bootstrap interval of **[32.08, 43.88]**. The analysis resamples original recording sources and retains their associated clips. This interval conditions on the fitted models and perturbation realization; it does not estimate variation over all possible training runs or noises.

![Gaussian-noise and gain robustness](../final_20261105/three_seed_robustness.png)

## 5. Compare the results and discuss the findings

**Architecture changes produce modest gains.** BiGRU improves the observed mean over the baseline, but also increases parameter count. Channel attention's additional gain is small and uncertain. Other attention configurations perform worse, demonstrating that extra architectural complexity is not automatically beneficial.

**Augmentation depends on the objective.** Noise training improves the tested Gaussian-noise performance much more than clean accuracy. Masking and Mixup do not help the tested clean task. These negative results are part of the comparison, rather than reasons to omit configurations or continue searching until one wins.

**The pretrained alternative gives the largest observed clean improvement.** Its additional training data and much larger frozen encoder must be considered alongside its score. A small trainable head does not make the entire inference system as small as the baseline CNN. Later matched linear/ReLU head experiments also found no gain from the tested added nonlinearity.

**Noise improvements do not transfer uniformly.** As optional supporting evidence, a later fixed-PANNs comparison found that recorded-background training improves a reserved recorded panel over Gaussian training by **+1.36 pp [0.41, 2.32]**, while reducing Gaussian-10-dB accuracy by **10.30 pp** relative to the Gaussian-trained head. This reinforces the course conclusion that an improvement must be evaluated under the conditions it is intended to address. The [separate experiment](../noise_distribution_20261107/README.md) contains its complete protocol and results; it is not required to establish the CNN/RNN checkpoint comparison.

The completed checkpoint therefore demonstrates three outcomes: a modest architectural improvement, a substantial matched-noise augmentation benefit, and a strong pretrained-system alternative. It also documents configurations that fail to improve performance. The conclusions are limited to the stated development evaluation and do not assert a new method or general deployment reliability. **No additional model training or research is needed to complete these five checkpoint activities.**

## Requirement-to-evidence mapping

| Provided checkpoint requirement | Completed evidence |
| --- | --- |
| Analyse performance and identify improvements | Section 1: baseline architecture, clean results and noise sensitivity |
| Implement one or more improvements | Section 2: recurrent processing, attention and augmentation |
| Alternatively compare another approach | Section 3: executed frozen PANNs + linear classifier |
| Evaluate using appropriate metrics | Section 4: accuracy, macro-F1, confusion/error analysis and paired robustness comparison |
| Compare with the baseline and discuss findings | Section 5: gains, negative outcomes, complexity and evaluation limitations |

## Submission evidence and references

- [Complete executed notebook and all 40 scientific figures](../../notebooks/ESC50_Results.ipynb).
- [Model comparison data](model_comparison.csv), [robustness comparison](robustness_comparison.csv) and [baseline identity](baseline_identity.json).
- [Earlier MLP controls](../mlp_controls_20261105/README.md) and [matched pretrained-head comparison](../transfer_followup_20261105/README.md).
- [ESC-50 dataset](https://github.com/karolpiczak/ESC-50); Piczak (2015), *ESC: Dataset for Environmental Sound Classification*.
- [Salamon and Bello (2017)](https://arxiv.org/abs/1608.04363), *Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification*.
- [Kong et al. (2020)](https://arxiv.org/abs/1912.10211), *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition*.
- [Literature decision map](literature_review.md), including the distinction between original methods and the adaptations implemented here.
