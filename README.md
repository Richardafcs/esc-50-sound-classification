# ESC-50 Environmental Sound Classification

A team research project for environmental sound classification with classical baselines, a frequency-preserving CNN, CNN + BiGRU (CRNN), channel attention, audio augmentation, and feature comparisons.

The completed study contains **65 registered tasks**, including **60 neural runs of 60 epochs**. This repository shares the implementation, protocols, portable CNN/CRNN models, compact result evidence, English PDF reports, all report figures, and the full regression suite. Large local experiment collections are intentionally excluded.

## Start here

| Purpose | Entry point |
| --- | --- |
| Read complete English PDFs | [Latest matrix study](reports/pdf/ESC50_Complete_Study_Report_EN.pdf) · [CNN/CRNN baseline study](reports/pdf/ESC50_CNN_CRNN_Baseline_Report_EN.pdf) |
| Understand the project | [English guide](docs/guides/guide_en.md) · [Chinese guide](docs/guides/guide_zh.md) |
| Inspect findings and figures | [Reports and evidence](reports/README.md) |
| Predict with pretrained models | [Portable CNN and CRNN](models/README.md) |
| Work in Colab | [Notebook guide](notebooks/README.md) |
| Find commands | [Script catalog](scripts/README.md) |

## Install and check

Use Python 3.11 or later. Repository access is required because the project is private.

```sh
git clone https://github.com/Richardafcs/esc-50-sound-classification.git
cd esc-50-sound-classification
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,baseline,report]'
python -m esc50_sound --help
python -m pytest -q
```

The full CPU suite contains **274 tests**. Tests use synthetic inputs and small saved regression fixtures; they do not download audio or launch Colab training. GitHub Actions runs the suite on pushes and pull requests. For Windows, activate the environment with `.venv\Scripts\activate`.

## Shared project layout

```text
src/esc50_sound/       Main Python implementation
scripts/              Experiment, audit, and reporting tools
configs/              Protocols, study matrix, and runnable examples
data/                 Official dataset metadata, licensing, and credits
notebooks/            Editable templates and a compact executed results reader
models/               Portable CNN and CRNN inference bundles
reports/              English PDFs, bilingual reports, all figures, summary evidence
docs/                 Guides, methods, engineering notes, and reviews
tests/                Full unit, integration, and research-contract suite
results/historical/   Only the numerical fixtures required by the test suite
archive/              Only two frozen source references required by tests
```

## Data and a new experiment

ESC-50 contains 2,000 five-second recordings in 50 classes. The dataset revision is pinned to `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. The source-strict development roles are **1,200 training / 398 validation / 400 designated test clips**; two clips are excluded to avoid source overlap. Normalization is fitted on training recordings only.

```sh
python -m esc50_sound prepare
python -m esc50_sound train --model cnn_freq --seed 101 --epochs 60 \
  --scheduler cosine --device cuda --output results/runs/new_cnn
python -m esc50_sound predict --checkpoint results/runs/new_cnn/best.pt \
  --audio YOUR_SHORT_CLIP.wav
```

`prepare` downloads the pinned official audio and builds the validated feature cache. Training requires the selected device; use `--device cpu` if needed. Choose a unique output directory for every run. Review the matrix notebook's resource budget before starting its full study.

## Baselines and findings

| Model family | Validation accuracy | Scope |
| --- | ---: | --- |
| Log-Mel statistics + logistic regression | 42.21% | One selected classical model |
| DCT cepstral statistics + Linear SVM | 40.20% | One selected classical model |
| CNN (`cnn_freq`) | 72.61% ± 0.25 pp | Mean ± sample SD, three training seeds |
| CNN + BiGRU / CRNN (`crnn_freq`) | 73.87% ± 1.26 pp | Mean ± sample SD, three training seeds |
| CRNN + channel attention | 74.37% ± 1.15 pp | Mean ± sample SD, three training seeds |

CNN and CRNN are the deep-learning baselines; logistic regression and SVM are the classical baselines. The classical input representations differ, so their score difference is not a classifier-only comparison. The selected seed101 CRNN checkpoint scores 75.13% on validation; this is distinct from its three-seed mean.

Channel attention adds about 0.50 percentage points over CRNN in this development setting. Synthetic Gaussian-noise augmentation improves the seed101 CNN at 10 dB added-noise SNR from 23.37% to 62.81%, but accuracy at 0 dB remains only 26.13%. Consult the reports for every condition, macro-F1, paired differences, and limitations.

These are **development-validation** results on the same 398 clips. Earlier research accessed all ESC-50 folds, so these results do not establish fresh independent confirmation, field robustness, state of the art, or production readiness. The reports keep historical test results separate from the latest validation study.

## Evidence and repository scope

The [summary tables and verification receipts](reports/evidence/matrix_20261028/) retain the numerical values from the completed study. The recorded waveform replay matched all **24,676 predictions** from 62 fitted models on 398 validation clips each. The retained receipts describe that historical check; this lean checkout does not contain every original model and raw run log needed to repeat the full replay.

| Shared in Git | Local and ignored |
| --- | --- |
| Source, configs, tests, dataset metadata and licenses | Downloaded WAVs and generated feature caches |
| Portable CNN/CRNN checkpoints and model cards | Bulk matrix weights, optimizer states, raw experiment logs |
| Two English PDFs, all 15 PNG figures and 13 SVG versions | Duplicate release ZIPs and standalone package assemblies |
| Curated summary CSVs and verification records | Historical pilot outputs, source archives, bulky training notebooks |
| 45 byte-preserved regression files, about 3.47 MB | New experiment outputs under `results/runs/` |

The [.gitignore](.gitignore) enforces this scope. The [fixture manifest](tests/historical_fixtures.json) identifies the exact historical files needed by tests. The [layout map](docs/project_layout_migration.json) records relocations, including paths printed inside older reports. Some dated engineering notes describe optional local evidence that is intentionally absent from a fresh checkout. Full-history audit and figure-regeneration commands need that original local collection; ordinary development, inference, tests, and reading the reports do not.

## Provenance and licensing

Current implementation lives in `src/`; frozen compatibility utilities live in `scripts/legacy/`. Keep historical evidence unchanged and record new experiments separately. The published checks were produced by the research agents and have not been independently replicated by external reviewers.

Dataset: [official ESC-50](https://github.com/karolpiczak/ESC-50). Retain the [dataset license](data/ESC-50/LICENSE) and [per-recording credits](data/ESC-50/meta/esc50.csv). The dataset is distributed overall under CC BY-NC 3.0; this project does not grant additional commercial-use rights.
