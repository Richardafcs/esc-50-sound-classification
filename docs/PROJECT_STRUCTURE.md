# 项目结构与迁移 / Project structure and migration

## GitHub collaboration profile / GitHub 协作范围

GitHub shares source, protocols, tests, dataset metadata, portable CNN/CRNN models, both English PDFs, all report figures, and curated evidence. Only 45 historical regression files are allowlisted. Full training records, raw audio, caches, duplicate ZIPs and local archives remain on the original machine and are ignored. The table below describes the full local folder; a fresh clone intentionally contains only the shared subset.

GitHub 只分享开发与协作所需的核心文件。下表描述完整本地目录；大型运行记录、音频、缓存、重复压缩包与历史归档在本地保留，不随仓库上传。

2026-10-29。目录按开发、结果、模型与发布用途分开；数值实验没有重新运行。

| 目录 | 负责内容 | 使用规则 |
| --- | --- | --- |
| `src/esc50_sound/` | 当前实现：音频、数据、模型、训练、推理和研究工具 | 修改实现从这里开始；命令入口为 `python -m esc50_sound` |
| `scripts/` | 命令包装、实验管理、审计与报告导出 | 旧版完整入口集中在 `scripts/legacy/` |
| `configs/` | 数据/评估协议和研究清单；`examples/` 为可运行示例 | 更新后的路径绑定对应本目录；原始协议副本已归档 |
| `data/` | 官方 metadata、许可；下载后生成音频和缓存 | 大型音频和缓存不纳入发布包 |
| `notebooks/templates/` | 当前可运行模板 | 检查预算后再执行训练 |
| `notebooks/executed/` | 已完成研究的原始 notebook 输出 | 作为证据保存，不在原文件中继续执行 |
| `notebooks/archive/` | 历史模板、pilot 和排错过程 | 保留历史身份 |
| `models/cnn/`, `models/crnn/` | 可独立运行的便携模型及模型卡 | 发布模型源码与当前库分开；内部 manifest 不变 |
| `results/matrix_20261028/` | 最新 65 项任务的日志、权重、预测、审计和图 | 原始运行记录与权重保持不变 |
| `results/historical/` | 早期训练、五折结果、审查和性能记录 | 只作为对应阶段的证据 |
| `results/runs/` | 以后新运行的输出 | 每次使用新目录，避免覆盖历史 |
| `reports/` | PDF、最新/历史研究报告、研究图 | 报告的作用范围以其日期与协议为准 |
| `docs/` | `guides/` 使用说明、`methodology/` 方法、`engineering/` 工程、`reviews/` 审查 | 与数值结果区分 |
| `releases/` | 可下载 ZIP、校验值；`packages/` 为独立包源码组装目录 | 已发布 ZIP 保持原样，可按原始结构解压 |
| `archive/` | 冻结源码、原始下载和迁移记录 | 不把历史命令误认为当前入口 |

## 迁移说明

根目录原来的七个 `.py` 文件已移到 `scripts/legacy/`；原 `tools/` 改为 `scripts/`。原 `deliverables/` 分为 `models/` 和 `releases/`。原 `output/pdf/` 的 PDF 放在 `reports/pdf/`，图示合集 ZIP 放在 `releases/`。最新矩阵结果单独存放，其余结果归入历史。

完整逐项映射见 [path_map.json](project_layout_migration.json)。已有 JSON manifest、PDF 内的项目路径和历史日志可能记录旧位置；这些是历史出处，按映射查找即可，不修改它们来伪造当时的环境。已有 PDF 的图示内部跳转仍然自包含。

English: The layout separates current source, operational scripts, protocols, data, notebooks, portable models, experimental evidence, reports and published archives. Historical bytes are retained; old paths printed inside immutable evidence resolve through the relocation map. Working configs retain their scientific parameters while using updated local paths, with original config files archived. New work belongs under `results/runs/`. This is a structural migration, not a new experiment or a change of scientific conclusions.
