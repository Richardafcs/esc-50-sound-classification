# Environmental Sound Classification on ESC-50

**COMP9444 course project** — CNN baseline, recurrent and attention models, audio augmentation, and pretrained audio features.

This project asks which modelling and training choices improve environmental sound classification when labelled data are limited. A small CNN is the primary baseline. We compare a CNN–BiGRU, lightweight attention modules, augmentation strategies, and a frozen PANNs encoder with a linear classifier. The completed work also includes two classical references, a 12-setting frontend study, two different MLP comparisons, encoded-time interventions, external recordings, input duration and controlled training-noise adaptation.

## Read the project

- **[Project report](reports/report.md):** motivation, methods, results, discussion and limitations.
- **[Main results notebook](notebooks/01_results.ipynb):** executed analyses, model comparisons and all 40 research figures. [Open in Colab](https://colab.research.google.com/github/Richardafcs/esc-50-sound-classification/blob/main/notebooks/01_results.ipynb).
- **[Methods](docs/methods.md)** and **[evaluation protocol](docs/evaluation.md):** implementation and experimental settings.
- **[Related work](docs/related_work.md):** literature and its relationship to the implemented methods.

## Main results

| Model | Development accuracy | Macro-F1 |
| --- | ---: | ---: |
| CNN baseline | 72.61% ± 0.25 pp | 71.39% ± 0.58 pp |
| CNN–BiGRU | 73.87% ± 1.26 pp | 72.97% ± 1.36 pp |
| CNN–BiGRU + channel attention | 74.37% ± 1.15 pp | 73.48% ± 1.00 pp |
| CNN + Gaussian noise augmentation | 73.12% ± 0.87 pp | 71.99% ± 1.23 pp |
| Frozen PANNs + linear classifier | **91.21%** | **91.10%** |

Scratch-model values are mean ± sample standard deviation across seeds 101, 202 and 303. PANNs uses one fitted linear classifier. Evaluation uses 398 development clips from 295 recording sources; training uses 1,200 clips. All official folds have historical exposure, so these are development results rather than an untouched test benchmark.

The main findings are:

- Recurrence gives a modest observed gain. Channel attention adds 0.50 percentage points over BiGRU, with an interval that includes zero.
- Gaussian augmentation raises CNN accuracy from 23.37% to 61.31% under the original matched 10-dB Gaussian-noise evaluation. This benefit is specific to the tested conditions.
- PANNs gives the highest clean accuracy, using external AudioSet pretraining and a much larger encoder. Its gain is a complete-system comparison, not an isolated effect of pretraining.
- Mixup, spectrogram masking and the tested additional MLP nonlinearity do not improve their respective clean reference results.

The [experiment index](reports/results/README.md) covers the supplementary studies and keeps their evaluation panels separate.

## Study structure

| Research question | Completed comparisons | Results |
| --- | --- | --- |
| What can be learned from limited labelled audio? | Statistical logistic regression, cepstral SVM, CNN, CNN–BiGRU, attention and scratch MLP controls | [Model comparison](reports/results/model_comparison/README.md), [MLP controls](reports/results/mlp_controls/README.md) |
| How do input representation and sequence processing matter? | Twelve sample-rate/feature settings and interventions on encoded time positions | [Methods](docs/methods.md), [temporal order](reports/results/temporal_order/README.md) |
| Does augmentation help under interference? | Gaussian noise, masking, Mixup and matched noise/gain evaluation | [Robustness](reports/results/robustness/README.md) |
| What does a pretrained representation offer? | Frozen PANNs, shared-noise comparison and matched linear/ReLU heads | [Pretrained reference](reports/results/panns/README.md), [head comparison](reports/results/transfer_heads/README.md) |
| How far do noise-training benefits transfer? | Clean, Gaussian-adapted and recorded-background-adapted readouts with the encoder fixed | [Gaussian adaptation](reports/results/gaussian_adaptation/README.md), [recorded-noise comparison](reports/results/recorded_noise/README.md) |
| What changes outside the five-second development task? | Six-category external recordings and matched duration/padding interventions | [External recordings](reports/results/external_recordings/README.md), [input duration](reports/results/input_duration/README.md) |

The [study overview](reports/tables/study_overview.csv) identifies the source files, evaluation population and interpretation of each study. The primary CNN baseline, earlier classical references, pretrained reference and adapted-head comparators serve different experimental roles.

## Latest completed model experiment

The model experiments extend through **7 November 2026**. The final controlled comparison changes the training-noise distribution while holding the PANNs encoder, clean-training scaler and classifier objective fixed:

| Linear readout | Clean accuracy | Gaussian 10 dB | Reserved recorded-background grid |
| --- | ---: | ---: | ---: |
| Clean-trained | 91.21% | 69.85% | 82.88% |
| Gaussian-adapted | 91.96% | 83.92% | 83.43% |
| Recorded-background-adapted | 91.71% | 73.62% | 84.79% |

Recorded-background training improves the fixed recorded grid over Gaussian training by **1.36 pp [0.41, 2.32]**, but its Gaussian-10-dB score is **10.30 pp lower**. The grid contains six reserved background recordings, two windows and three SNRs applied to the same 398 foreground clips. This is a conditional distribution-dependent tradeoff. The [report](reports/report.md) and [executed notebook](notebooks/06_recorded_noise.ipynb) give the full design and limitations.

## Repository structure

```text
notebooks/                 Main results and five supplementary analyses
src/esc50_sound/            Audio processing, models, training and evaluation
scripts/                   Experiment entry points
configs/
  studies/                 Capacity and attention settings
  evaluation/              Development evaluation settings
  examples/                Runnable training examples
  historical/              Original model matrix and earlier settings
data/                      Dataset metadata and acquisition notes
models/                    Selected checkpoints and trained linear heads
reports/
  report.md                Course project report
  tables/                  Main comparison tables
  figures/                 Shared figures and figure index
  results/                 Results grouped by experiment
docs/                      Methods, evaluation and related work
results/                   Local experiment outputs; excluded from Git
```

## Run the project

Use Python 3.11 or newer. From the repository root:

```sh
python -m pip install -e '.[baseline,notebook,transfer]'
python -m esc50_sound --help
```

Open `notebooks/01_results.ipynb` to inspect and recompute analyses from the included results. All six notebooks include executed analyses of the completed Colab experiments and default to saved-result analysis. Optional model-computation switches are disabled. A fresh PANNs extraction requires the official audio and pretrained checkpoint; use a Colab GPU for that stage.

For single-clip CNN inference:

```sh
python -m esc50_sound predict --checkpoint models/cnn/checkpoint.pt --audio path/to/clip.wav
```

See [data preparation](data/README.md), [model files](models/README.md), [training configurations](configs/README.md) and [experiment commands](scripts/README.md). Raw audio, feature caches, bulk training runs and document-generation utilities are excluded from Git.

## Data and references

ESC-50 is provided by [Karol Piczak](https://github.com/karolpiczak/ESC-50). Its [CC BY-NC 3.0 license](data/ESC-50/LICENSE) and [recording credits](data/ESC-50/meta/esc50.csv) are retained. The pretrained alternative uses the [official PANNs implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn). Full references are in [related work](docs/related_work.md).
