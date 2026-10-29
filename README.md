# ESC-50 Environmental Sound Classification

A reproducible research project for environmental sound classification with classical baselines, CNN and CNN+BiGRU (CRNN), attention, augmentation and feature comparisons.

## 从这里开始 / Start here

| 目的 | 入口 |
| --- | --- |
| 阅读完整英文 PDF | [最新完整研究](reports/pdf/ESC50_Complete_Study_Report_EN.pdf) · [CNN/CRNN baseline](reports/pdf/ESC50_CNN_CRNN_Baseline_Report_EN.pdf) |
| 阅读中文说明与结果 | [内容说明](docs/guides/guide_zh.md) · [最新结果](reports/matrix_20261028/report_zh.md) |
| English guide and results | [Guide](docs/guides/guide_en.md) · [Latest results](reports/matrix_20261028/report_en.md) |
| 使用或检查模型 | [Models](models/README.md) |
| 下载模型与独立包 | [Releases](releases/README.md) |
| 使用 Colab | [Notebook guide](notebooks/README.md) · [已保存的双语 Colab](https://colab.research.google.com/drive/1A3gtdOMYvkezTTnwXzB1ux9wz-vjU_Nf?authuser=1) |
| 了解目录与迁移 | [Project structure](docs/PROJECT_STRUCTURE.md) |

## 项目结构 / Layout

```text
src/esc50_sound/       Main Python implementation
scripts/              Experiment, audit and reporting commands
configs/              Protocols, study matrix and runnable examples
data/                 Dataset metadata, licensing and generated feature caches
notebooks/            Templates, executed evidence and archived notebooks
models/               Portable CNN and CRNN checkpoints
results/              Current study, historical evidence and future run outputs
reports/              English PDFs, bilingual results and research figures
docs/                 Guides, methods, engineering notes and reviews
tests/                Unit, integration and research-contract tests
releases/             Published ZIPs and standalone package source assemblies
archive/              Frozen sources, original downloads and migration records
```

## Install and run

Python 3.11+. Install from the project root:

```sh
python -m pip install -e '.[dev,baseline,report]'
python -m esc50_sound --help
python -m pytest -q
```

Prepare the pinned data and run a new CNN experiment:

```sh
python -m esc50_sound prepare
python -m esc50_sound train --model cnn_freq --seed 101 --epochs 60 \
  --scheduler cosine --device cuda --output results/runs/new_cnn
python -m esc50_sound predict --checkpoint results/runs/new_cnn/best.pt \
  --audio YOUR_SHORT_CLIP.wav
```

The complete study can be audited from saved records with `python scripts/analyze_colab_matrix.py --require-complete`. No new training or test evaluation is needed for that audit. See [script navigation](scripts/README.md) for other tools.

## Recorded research status

The latest matrix executed **65 tasks**, including **60 neural runs of 60 epochs**. Validation accuracy across three seeds was 72.61% for CNN, 73.87% for CRNN and 74.37% for CRNN + channel attention. These are development-validation measurements on the same 398 clips. Historical exposure to every ESC-50 fold prevents a fresh independent-test claim. See the full reports for macro-F1, spread, controls and negative robustness findings.

The source-strict roles contain 1,200 train / 398 validation / 400 designated test clips. Normalization fits training data only. All 62 fitted models reproduced 24,676 validation-WAV predictions in the recorded replay. These are internal agent-produced checks, not external human replication.

## Reproducibility and provenance

Current implementation lives in `src/`; retained compatibility tools live in `scripts/legacy/`. Executed notebooks, run/checkpoint records and existing published ZIP/PDF bytes retain their historical identities. The [migration map](archive/reorganization_20261029/path_map.json) locates their earlier project paths. Working config path bindings were updated; their original bytes remain archived.

Use `results/runs/` for new outputs. `results/historical/` preserves earlier research. Exact resume of a historical training state still requires its original source/runtime; use the frozen release when reproducing that environment. The directory reorganization does not re-run experiments or alter reported scores.

Dataset: [official ESC-50](https://github.com/karolpiczak/ESC-50), pinned commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. Retain the [dataset license](data/ESC-50/LICENSE) and recording credits.
