# ESC-50 完整检查与 Colab 执行结果报告

报告定稿：2026年10月29日；正式实验执行于10月28日。本报告针对用户提供的 Downloads 项目与独立 Study Matrix。**65/65 项任务实际完成、下载并通过结果审计**；其中 60 次神经网络训练各完成 60 epochs。它是开发验证研究，历史测试集已经暴露，不能称为新的独立确认。

结构组中最高验证均值为 **BiGRU + Channel Attention：74.37%**，相对原 BiGRU 为 +0.50 个百分点。最好的特征组均值来自 `sr32_fmax16_logmel32`，为 57.12%；该组使用不同的 pooled CNN，不能直接当成主结构的单变量改进。0 dB 白噪声下，加噪训练 CNN 相对干净训练 CNN 的变化为 **+19.35 个百分点**；完整条件见后表。

[中文内容说明](../../docs/guides/guide_zh.md) · [English report](report_en.md)

## 1. 检查结论与修复

| 级别 | 已证明的问题 | 已验证修复 |
| --- | --- | --- |
| P1 | 旧目录可能被误认成新实验完成 | 核对任务配置、cache、源码、checkpoint、完整历史与逐样本指标 |
| P1 | JSON 仅核对 task ID，实际字段可能被忽略 | 加载矩阵并校验 schema、划分、预算、模型/seed/增强/条件语义 |
| P1 | 静默子进程可能不受 timeout 约束 | 轮询输出文件与进程，按单任务/会话截止时间终止进程组 |
| P1 | 特征汇总可能接纳未完成或伪造记录 | 验证完整 epochs、选中 epoch、模型、标签/成员与重算指标，最后发布完成指标 |
| P1 | Colab kernel 安装后未刷新模块路径 | 显式加入已验证源码目录；重跑 pilot 成功 |
| P2 | 结果页漏掉 stress.json | 增加三模型 × 六条件的显式结果表 |
| P1 | CI 安装 dev 却漏装 baseline 依赖 | CI 和新指南统一安装 dev、baseline、report extras |

修复后本地与 Colab 均通过 274 项测试；最终 notebook 输出没有 error 记录。首次本地缺少 scikit-learn；首次云端上传缺少历史测试 fixtures；首次 pilot 在 CNN 已训练后发生 kernel 导入错误。它们已修复并保留日志，没有把失败运行当作成功。CI 配置已修正，但没有声称实际触发了 GitHub Actions。

## 2. 实际执行范围与数据协议

| 组 | 正式任务 | 实际内容 |
| --- | --- | --- |
| A | 2 | Log-Mel 统计 + logistic；倒谱统计 + LinearSVC |
| B | 15 | CNN、BiGRU、TA、CA、TCA × 3 seeds |
| C | 9 | SpecAugment、waveform noise、waveform Mixup × 3 seeds |
| D | 36 | 12 种特征/采样条件 × 3 seeds |
| E | 3 | 三种 seed101 checkpoint 各六种验证扰动条件 |

三个 2-epoch pilot 和一个真实数据断点恢复 probe 单独保存，不计入 60 次正式训练或均值。PANNs/AST/BEATs/CLAP/RL 没有实现，因此没有执行或编造分数；本轮没有新增 fold 5 测试或五折确认。软件测试会核对旧 fixture，但这不等于对新模型进行新的测试评价。

数据为固定 commit 的 ESC-50：2,000 段、50 类、每段五秒。实际下载 WAV 并生成 v2 cache；2,000 行 cache 与原始 metadata 逐行对齐。训练 folds 1–3、验证 fold 4、指定测试角色 fold 5；排除跨角色 source ID 后为 **1200 / 398 / 400**。验证 classes 6 与 37 各 7 条，其余各 8 条。归一化只用训练录音拟合。原始音频跨 fold 精确重复检查通过，但没有完成感知近重复审计。

主输入：16 kHz 单声道 → 512 点 Hann STFT、hop 256 → 64 个 HTK Mel 功率带 → natural log → train-only band normalization，shape `[B,1,64,313]`。encoder 输出 `[B,64,8,39]`；GRU 输入 `[B,39,512]`，双向 hidden size 64，输出 128 维。TA 做时间 softmax 加权平均；CA 做 64 → 8 → 64 门控，使用 2 × sigmoid 并初始化为恒等。

神经网络统一训练 60 epochs、batch size 32、AdamW（lr 0.001、weight decay 0.0001）、cosine 调度及 gradient clipping（阈值 5）；主训练每个 epoch 1,200 条、38 次更新，总计 2,280 次更新。所有运行分配同样 60 轮预算，最终权重按最早最低验证 cross-entropy 选择，可能早于第 60 轮。Logistic 选最低验证 cross-entropy；SVM 选最高 accuracy，再按 macro-F1 及较小 C 打破平局。Accuracy 和 50 类等权 Macro-F1 均从逐样本预测重新计算。

## 3. 传统 baseline与结构结果

| Baseline | Accuracy | Macro-F1 | 选择参数 |
| --- | --- | --- | --- |
| logistic | 42.21% | 38.86% | 0.001 |
| cepstral_svm | 40.20% | 37.50% | 1.0 |

Logistic 是128 维 Log-Mel 时间均值/标准差；SVM 是20 个 DCT 系数的均值/标准差，共 40 维。二者改变了特征、分类器和选择规则，不能把差异单独归因于分类器。所有候选拟合均已检查收敛。

| 模型 | Accuracy：101 / 202 / 303 | Accuracy 均值 ± SD（%） | Macro-F1 均值 ± SD（%） | 参数量 |
| --- | --- | --- | --- | --- |
| CNN | 72.61%, 72.36%, 72.86% | 72.61 ± 0.25 | 71.39 ± 0.58 | 49058 |
| CNN + BiGRU | 75.13%, 73.87%, 72.61% | 73.87 ± 1.26 | 72.97 ± 1.36 | 251810 |
| BiGRU + Temporal Attention | 74.87%, 70.10%, 71.86% | 72.28 ± 2.41 | 71.52 ± 2.41 | 251939 |
| BiGRU + Channel Attention | 75.38%, 74.62%, 73.12% | 74.37 ± 1.15 | 73.48 ± 1.00 | 252906 |
| BiGRU + Temporal + Channel Attention | 74.12%, 71.61%, 72.86% | 72.86 ± 1.26 | 71.69 ± 1.35 | 253035 |

![三个 seed 的开发验证结果；SD 不是置信区间。](../figures/matrix_20261028/architecture.png)

三个 seed 的开发验证结果；SD 不是置信区间。 [Data / 数据](../evidence/matrix_20261028/analysis/groups.csv) · [Plot code / 绘图代码](../../scripts/plot_colab_matrix.py)

原 CNN 与 GRU 的六次新运行，在选中 epoch、逐条验证预测及权重 tensor 上与历史开发结果完全一致。这是本次实际重跑的回归验证，不是把旧分数复制成新实验。

## 4. 增强与配对变化

| 训练条件 | Accuracy 均值 ± SD（%） | Macro-F1 均值 ± SD（%） |
| --- | --- | --- |
| Clean CNN | 72.61 ± 0.25 | 71.39 ± 0.58 |
| specaugment | 71.86 ± 1.15 | 70.75 ± 1.38 |
| noise | 73.12 ± 0.87 | 71.99 ± 1.23 |
| mixup | 71.61 ± 0.66 | 70.65 ± 0.44 |

![增强仅用于训练，验证声音始终保持干净。](../figures/matrix_20261028/augmentation.png)

增强仅用于训练，验证声音始终保持干净。 [Data / 数据](../evidence/matrix_20261028/analysis/groups.csv) · [Plot code / 绘图代码](../../scripts/plot_colab_matrix.py)

SpecAugment 遮挡归一化后的时间/频带区域；noise 以 0.5 概率添加 10–30 dB 的 Gaussian noise；Mixup 使用 Beta(0.4, 0.4) 混合波形和两个标签的损失。增强训练 loss 与干净训练 loss 不是相同目标，不用它们的大小直接判断谁更好。

| 配对对照 | 各 seed accuracy 变化（pp） | 均值±SD（pp） | 平均 Macro-F1 变化（pp） |
| --- | --- | --- | --- |
| attention_ca_vs_gru | +0.25, +0.75, +0.50 | +0.50 ± 0.25 | +0.51 |
| attention_interaction | -1.01, +0.75, +0.50 | +0.08 ± 0.95 | -0.33 |
| attention_ta_vs_gru | -0.25, -3.77, -0.75 | -1.59 ± 1.90 | -1.45 |
| attention_tca_vs_gru | -1.01, -2.26, +0.25 | -1.01 ± 1.26 | -1.28 |
| gru_vs_cnn | +2.51, +1.51, -0.25 | +1.26 ± 1.40 | +1.58 |
| mixup_vs_clean_cnn | -0.50, -1.51, -1.01 | -1.01 ± 0.50 | -0.74 |
| noise_vs_clean_cnn | +0.00, +1.76, -0.25 | +0.50 ± 1.10 | +0.60 |
| specaugment_vs_clean_cnn | +0.25, -1.76, -0.75 | -0.75 ± 1.01 | -0.64 |

![同 seed 配对变化；不是显著性检验。](../figures/matrix_20261028/paired_effects.png)

同 seed 配对变化；不是显著性检验。 [Data / 数据](../evidence/matrix_20261028/analysis/paired_contrasts.csv) · [Plot code / 绘图代码](../../scripts/plot_colab_matrix.py)

Attention interaction 为 TCA − TA − CA + 原 GRU，用于描述两模块的组合偏离简单相加的程度，不是无混杂的因果机制证明。三个 seed 重复使用同一 398 条验证录音；它们不是三个独立数据集，也不能用 seed SD 替代数据采样置信区间。

## 5. 十二组特征与采样率对照

| Case | Accuracy 均值 ± SD（%） | Macro-F1 均值 ± SD（%） |
| --- | --- | --- |
| sr16_fmax8_loglinear64 | 51.59 ± 0.81 | 48.58 ± 1.15 |
| sr16_fmax8_logmel32 | 55.61 ± 0.73 | 52.97 ± 0.48 |
| sr16_fmax8_logmel64 | 51.59 ± 2.05 | 48.46 ± 2.20 |
| sr16_fmax8_mfcc40 | 49.25 ± 1.74 | 47.08 ± 1.75 |
| sr32_fmax16_loglinear64 | 48.99 ± 1.33 | 46.09 ± 1.74 |
| sr32_fmax16_logmel32 | 57.12 ± 1.05 | 54.59 ± 1.52 |
| sr32_fmax16_logmel64 | 51.09 ± 0.38 | 48.33 ± 0.48 |
| sr32_fmax16_mfcc40 | 49.58 ± 3.10 | 47.29 ± 2.93 |
| sr32_fmax8_loglinear64 | 50.75 ± 0.66 | 47.58 ± 1.57 |
| sr32_fmax8_logmel32 | 55.70 ± 1.05 | 53.00 ± 1.34 |
| sr32_fmax8_logmel64 | 51.26 ± 1.96 | 48.23 ± 2.16 |
| sr32_fmax8_mfcc40 | 49.83 ± 1.71 | 47.63 ± 1.73 |

![D 组使用 26,658 参数的 pooled CNN；与主结构组分开解释。](../figures/matrix_20261028/feature_sampling.png)

D 组使用 26,658 参数的 pooled CNN；与主结构组分开解释。 [Data / 数据](../evidence/matrix_20261028/analysis/groups.csv) · [Plot code / 绘图代码](../../scripts/plot_colab_matrix.py)

四种表示是 Log-Mel 64、Log-Mel 32、64 维 Log-Mel 的前 40 个正交 DCT-II 系数、Log-linear 64。三种条件为 16 kHz/F8、32 kHz/F8、32 kHz/F16；窗口均 32 ms、hop 均 16 ms、313 帧。16k/F8 → 32k/F8 匹配可用带宽；32k/F8 → 32k/F16 增加高频，也改变滤波器间距。MFCC 改变了行数和相邻坐标意义；相同参数量不等于相同计算量或纯单变量因果实验。全部 17 类 × 3 seeds 的配对差异见 CSV，缺失项没有补成 0。

## 6. 噪声与增益敏感性

| 条件 | 干净 CNN | 干净 BiGRU | 加噪训练 CNN |
| --- | --- | --- | --- |
| clean | 72.61% | 75.13% | 72.61% |
| snr_20 | 47.74% | 54.77% | 70.60% |
| snr_10 | 23.37% | 24.62% | 62.81% |
| snr_0 | 6.78% | 6.28% | 26.13% |
| gain_-6 | 68.84% | 69.85% | 68.59% |
| gain_6 | 71.86% | 73.37% | 70.60% |

![每格 398 条录音；每个模型仅使用一个选中的 seed101 checkpoint。](../figures/matrix_20261028/robustness.png)

每格 398 条录音；每个模型仅使用一个选中的 seed101 checkpoint。 [Data / 数据](../evidence/matrix_20261028/analysis/stress.csv) · [Plot code / 绘图代码](../../scripts/plot_colab_matrix.py)

这些是一次确定性、逐录音 RMS 缩放的 Gaussian noise 与 gain 干预，三个模型使用相同扰动实现；没有峰值裁剪。0 dB 意味着添加噪声与原录音 RMS 功率相同，不意味着真实背景 SNR 已知。即使某些条件改善，也不能声称已覆盖现实噪声、房间、设备或未知类别。E 组仅有 3 个 checkpoint，没有三个 seed 的鲁棒性结论。

## 7. 完整性、重执行与效率证据

独立脚本从 CSV 按 50 类重新计算混淆矩阵、Accuracy 与 Macro-F1，核对全部 65 项任务、60 轮预算、选中 epoch 及配置。随后在 Colab 从原始验证 WAV重新通过项目前端与模型，**62 个拟合模型 × 398 条 = 24,676 条预测全部重现**；这是实际 forward 重执行，不是只读取保存的分数。断点恢复 probe 在真实数据上完成 1 轮、恢复后完成第 2 轮，与不中断 pilot 的权重逐位相同。恢复结论仅覆盖相同 runtime/code/data，不承诺跨版本或跨设备。

| 模型 | CPU WAV→logits p95(ms) | T4 WAV→logits p95(ms) | checkpoint(KiB) |
| --- | --- | --- | --- |
| CNN | 25.787 | 16.045 | 281.1 |
| CNN + BiGRU | 24.477 | 17.860 | 1075.4 |
| BiGRU + Temporal Attention | 32.398 | 16.501 | 1076.5 |
| BiGRU + Channel Attention | 25.846 | 17.290 | 1080.9 |
| BiGRU + Temporal + Channel Attention | 35.366 | 17.276 | 1081.9 |

测量为 Colab `Intel(R) Xeon(R) CPU @ 2.00GHz` / `Tesla T4`，2 CPU threads、batch size 1、float32、10 次 warmup、100 次重复、同一验证 WAV。GPU 计时同步；包括解码、前端与传输，排除模型加载、进程启动与服务排队。Torch 峰值 allocated memory 不是驱动总显存。60 次训练循环记录耗时之和为 72.75 分钟，包括循环内验证与保存；排除初始化、cache/WAV 加载及验证、数据/特征准备、pilot、probe、空闲和导出；不当作整个会话时间。付费支出 0，没有升级或购买资源。

![固定 seed101 验证曲线；checkpoint 按最低 cross-entropy 选择。](../figures/matrix_20261028/learning_curves.png)

固定 seed101 验证曲线；checkpoint 按最低 cross-entropy 选择。 [Data / 数据](../figures/matrix_20261028/learning_curve_data.csv) · [Plot code / 绘图代码](../../scripts/plot_colab_matrix.py)

### 类别错误切片：固定seed101

| 模型 | 零recall类别数 | 最低recall的三类：正确/支持 |
| --- | --- | --- |
| CNN | 0 | sheep 2/8; wind 3/8; clapping 3/8 |
| CNN + BiGRU | 0 | helicopter 1/8; door_wood_creaks 2/8; sheep 3/8 |
| BiGRU + Temporal Attention | 0 | door_wood_creaks 1/8; helicopter 1/8; wind 3/8 |
| BiGRU + Channel Attention | 0 | helicopter 1/8; drinking_sipping 2/8; door_wood_creaks 2/8 |
| BiGRU + Temporal + Channel Attention | 0 | drinking_sipping 1/8; helicopter 1/8; wind 2/8 |

每类仅7或8段验证录音，单段错误会显著改变该类别recall；这些错误切片是探索性诊断。完整混淆矩阵和逐条预测随每个run保存，不能用平均分掩盖具体类别的失败。

## 8. 结论边界与下一项决策

本轮回答了结构、增强、特征与人工扰动在既定开发协议下的表现，没有建立新的 SOTA、算法新颖性或生产部署结论。Attention 并非必然有效；应考察三个 seed 及小幅变化，而不只看最高分。60 轮的分配一致，不等于不同结构参数、FLOPs、延迟完全匹配。BiGRU 是离线双向模型，没有证明实时因果流式识别。Attention 权重也不是可靠因果解释。

**不要把本轮约 70% 以上的 validation 结果，与旧报告约 62% 的 test 结果相减，称为提升。** 它们是不同录音、不同用途的指标。历史默认 CNN的三个 seed 的 fold 5 均值 63.08%、GRU 63.92%，仅作旧证据索引；本轮没有重新确认这些数值。所有 ESC-50 folds 都有既往使用历史，继续挑方案仍是探索。下一步若要确认候选，应先定义新的、来源隔离的评价边界及真实噪声需求；不能通过重命名旧 fold 恢复盲测。

旧 CNN 交付包保留为历史默认；本次所有新模型以研究 checkpoint 交付，完整保留失败与负面结果。未修改考试学习记录，未将 Agent 产物当作用户独立掌握，也未冒称外部人类复现或机构认证。

## 9. 证据入口与复现

[全部逐运行结果](../evidence/matrix_20261028/analysis/runs.csv) · [三个 seed 汇总](../evidence/matrix_20261028/analysis/groups.csv) · [75 个配对变化](../evidence/matrix_20261028/analysis/paired_contrasts.csv) · [鲁棒性](../evidence/matrix_20261028/analysis/stress.csv) · [完整审计](../evidence/matrix_20261028/analysis/audit.json) · [WAV 重执行](../evidence/matrix_20261028/runtime_verification/waveform_replay.json) · [恢复检查](../evidence/matrix_20261028/runtime_verification/resume_probe.json) · [计时定义](../evidence/matrix_20261028/runtime_verification/warm_latency.json)

环境：torch=2.11.0+cu130, numpy=2.1.3, scipy=1.16.3, soundfile=0.14.0, sklearn=1.6.1.

```bash
python -m pip install -e '.[dev,baseline,report]'
python -m esc50_sound prepare
python -m pytest -q
python scripts/analyze_colab_matrix.py --require-complete
python scripts/plot_colab_matrix.py
```

完整矩阵使用 `notebooks/templates/ESC50_Matrix_Full_20261028.ipynb`；它会实际训练所有任务，请先读预算设置。安全预览入口仍为 `ESC50_Colab_Research_Console.ipynb` 的 plan mode。特征实验不能 epoch-resume；主训练恢复需要匹配环境和 last.pt。Colab 磁盘是临时的，本轮逐任务导出并下载；notebook 本身不等于所有权重与音频。

原始数据与许可：[官方 ESC-50](https://github.com/karolpiczak/ESC-50)，固定 commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`。数据整体使用 CC BY-NC 3.0 许可，保留 credits；本项目不授予商业权利。[Colab 资源说明](https://research.google.com/colaboratory/faq.html)。方法原始参考与核验深度见[文献阅读记录](../../docs/methodology/literature_deep_read.md)；本轮结果的直接来源是保存的真实运行及逐样本输出。


## 已保存的 Colab 阅读器与最终审查范围

[在 Colab 打开双语结果](https://colab.research.google.com/drive/1A3gtdOMYvkezTTnwXzB1ux9wz-vjU_Nf?authuser=1)。下载读回后的全部单元源码、文字输出及四幅图均一致，六个代码单元无错误。阅读器展示已审计结果，不重新训练。

[审查覆盖与剩余限制](review_coverage.md) · [计时环境](../evidence/matrix_20261028/runtime_verification/measurement_environment.json)
