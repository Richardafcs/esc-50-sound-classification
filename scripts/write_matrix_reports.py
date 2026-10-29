#!/usr/bin/env python3
"""Write bilingual reports only after complete audit and runtime replay evidence."""
from pathlib import Path
import csv,json,statistics,re
ROOT=Path(__file__).resolve().parents[1]
R=ROOT / 'results/matrix_20261028';D=ROOT/'reports/matrix_20261028'
MODELS=['cnn_freq','crnn_freq','crnn_freq_ta','crnn_freq_ca','crnn_freq_tca']
NAMES=dict(zip(MODELS,['CNN','CNN + BiGRU','BiGRU + Temporal Attention','BiGRU + Channel Attention','BiGRU + Temporal + Channel Attention']))

def readcsv(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

def p(value):return f'{100*float(value):.2f}%'
def spread(mean,sd):return f'{100*float(mean):.2f} ± {100*float(sd):.2f}'
def figure(name,caption):
 sources={'architecture':'analysis/groups.csv','augmentation':'analysis/groups.csv','feature_sampling':'analysis/groups.csv','paired_effects':'analysis/paired_contrasts.csv','robustness':'analysis/stress.csv','learning_curves':'figures/learning_curve_data.csv'}
 return (f'![{caption}](../../results/matrix_20261028/figures/{name}.png)\n\n{caption} '
         f'[Data / 数据](../../results/matrix_20261028/{sources[name]}) · '
         '[Plot code / 绘图代码](../../scripts/plot_colab_matrix.py)')

def main():
 audit=json.loads((R/'analysis/audit.json').read_text());assert audit['complete'] and audit['status_counts']=={'audited':65}
 runs=readcsv(R/'analysis/runs.csv');groups=readcsv(R/'analysis/groups.csv');effects=readcsv(R/'analysis/paired_effects.csv');pairs=readcsv(R/'analysis/paired_contrasts.csv');stress=readcsv(R/'analysis/stress.csv')
 replay=json.loads((R/'runtime_verification/waveform_replay.json').read_text());assert len(replay['runs'])==62 and all(r['clips']==398 and r['changed_predictions']==0 for r in replay['runs'])
 matrix=json.loads((ROOT/'configs/colab_study_matrix_v1.json').read_text())
 expected_replay={t['id'] for t in matrix['tasks'] if t['kind']!='robustness'}
 assert {r['task'] for r in replay['runs']}==expected_replay and len(expected_replay)==62
 for log in ('local_tests_final.txt','cloud_tests_final.txt'):
  assert re.search(r'274 passed', (R/log).read_text()), log
 regression=json.loads((R/'baseline_regression.json').read_text())['runs']
 assert len(regression)==6 and {(r['model'],r['seed']) for r in regression}=={(m,s) for m in ('cnn_freq','crnn_freq') for s in (101,202,303)}
 assert all(r['predictions_equal'] and r['best_epoch_equal'] and r['weights_equal'] for r in regression)
 resume=json.loads((R/'runtime_verification/resume_probe.json').read_text());assert resume['weights_bitwise_equal_to_uninterrupted_pilot']
 timing=json.loads((R/'runtime_verification/warm_latency.json').read_text())
 environment=json.loads((R/'runtime_environment.json').read_text())
 notebook=json.loads((ROOT / 'notebooks/executed/ESC50_Matrix_Full_20261028_output.ipynb').read_text());assert not [o for c in notebook['cells'] if c['cell_type']=='code' for o in c.get('outputs',[]) if o['output_type']=='error']
 assert len([c for c in notebook['cells'] if c['cell_type']=='code'])==9
 execution_log=(R/'full_console.log').read_text(errors='replace')
 assert 'Executing cell 9/9' in execution_log and '65 / 65' in execution_log
 arch={g['model']:g for g in groups if g['group']=='B_architecture'};aug={g['augmentation']:g for g in groups if g['group']=='C_augmentation'};feat=[g for g in groups if g['group']=='D_feature_sampling']
 best=max(arch,key=lambda m:float(arch[m]['accuracy_mean']));bestfeature=max(feat,key=lambda g:float(g['accuracy_mean']))
 delta=100*(float(arch[best]['accuracy_mean'])-float(arch['crnn_freq']['accuracy_mean']))
 def stressval(task,condition):return float(next(r['accuracy'] for r in stress if r['task_id']==task and r['condition']==condition))
 noisegain=100*(stressval('E_noise_cnn','snr_0')-stressval('E_clean_cnn','snr_0'))
 for lang in ('zh','en'):
  zh=lang=='zh'
  header='# ESC-50 完整检查与 Colab 执行结果报告' if zh else '# ESC-50 full review and Colab execution report'
  intro=(f'报告定稿：2026年10月29日；正式实验执行于10月28日。本报告针对用户提供的 Downloads 项目与独立 Study Matrix。**65/65 项任务实际完成、下载并通过结果审计**；其中 60 次神经网络训练各完成 60 epochs。它是开发验证研究，历史测试集已经暴露，不能称为新的独立确认。' if zh else
         'Finalized: 29 October 2026; main experiments ran on 28 October. This report covers the supplied Downloads project and standalone Study Matrix. **65/65 tasks were executed, downloaded and audited**, including 60 neural runs completing 60 epochs each. This is development-validation research; historical test exposure prevents a claim of fresh independent confirmation.')
  highlight=(f'结构组中最高验证均值为 **{NAMES[best]}：{p(arch[best]["accuracy_mean"])}**，相对原 BiGRU 为 {delta:+.2f} 个百分点。最好的特征组均值来自 `{bestfeature["case"]}`，为 {p(bestfeature["accuracy_mean"])}；该组使用不同的 pooled CNN，不能直接当成主结构的单变量改进。0 dB 白噪声下，加噪训练 CNN 相对干净训练 CNN 的变化为 **{noisegain:+.2f} 个百分点**；完整条件见后表。' if zh else
             f'The architecture group with the highest validation mean is **{NAMES[best]}: {p(arch[best]["accuracy_mean"])}**, {delta:+.2f} percentage points relative to the original BiGRU. The highest feature-track mean is `{bestfeature["case"]}` at {p(bestfeature["accuracy_mean"])}; that track uses a different pooled CNN and is not a single-variable replacement for the main architecture. At 0 dB Gaussian-noise SNR, noise-trained versus clean-trained CNN changes accuracy by **{noisegain:+.2f} points**. All conditions are reported below.')
  parts=[header,intro,highlight,('[中文内容说明](../../docs/guides/guide_zh.md) · [English report](report_en.md)' if zh else '[English content guide](../../docs/guides/guide_en.md) · [中文报告](report_zh.md)')]
  parts+=['## '+('1. 检查结论与修复' if zh else '1. Review findings and repairs')]
  repairrows=[
   ('P1','旧目录可能被误认成新实验完成' if zh else 'Stale directories could impersonate completed tasks','核对任务配置、cache、源码、checkpoint、完整历史与逐样本指标' if zh else 'Check task/config/cache/source/checkpoint identity, full history and prediction metrics'),
   ('P1','JSON 仅核对 task ID，实际字段可能被忽略' if zh else 'ID-only matrix checks ignored semantic changes','加载矩阵并校验 schema、划分、预算、模型/seed/增强/条件语义' if zh else 'Consume matrix with schema, split, budget and semantic validation'),
   ('P1','静默子进程可能不受 timeout 约束' if zh else 'Silent subprocesses could evade timeout','轮询输出文件与进程，按单任务/会话截止时间终止进程组' if zh else 'Poll output/process and terminate its group at command/session deadline'),
   ('P1','特征汇总可能接纳未完成或伪造记录' if zh else 'Feature aggregation admitted incomplete/fabricated records','验证完整 epochs、选中 epoch、模型、标签/成员与重算指标，最后发布完成指标' if zh else 'Validate full exposure, selected epoch, weights, membership/labels and metrics; publish completion last'),
   ('P1','Colab kernel 安装后未刷新模块路径' if zh else 'Live Colab kernel did not discover editable installation','显式加入已验证源码目录；重跑 pilot 成功' if zh else 'Add verified source directory to live kernel; pilot rerun succeeded'),
   ('P2','结果页漏掉 stress.json' if zh else 'Results view omitted stress.json','增加三模型 × 六条件的显式结果表' if zh else 'Add explicit three-model × six-condition result table'),
   ('P1','CI 安装 dev 却漏装 baseline 依赖' if zh else 'CI omitted baseline dependencies required by its tests','CI 和新指南统一安装 dev、baseline、report extras' if zh else 'Install dev, baseline and report extras in CI and current guide')]
  parts+=[table(['级别','已证明的问题','已验证修复'] if zh else ['Severity','Demonstrated issue','Verified repair'],repairrows)]
  parts+=[('修复后本地与 Colab 均通过 274 项测试；最终 notebook 输出没有 error 记录。首次本地缺少 scikit-learn；首次云端上传缺少历史测试 fixtures；首次 pilot 在 CNN 已训练后发生 kernel 导入错误。它们已修复并保留日志，没有把失败运行当作成功。CI 配置已修正，但没有声称实际触发了 GitHub Actions。' if zh else
           'The repaired project passed 274 tests locally and 274 in Colab; the final notebook contains no error outputs. Initial issues were a missing local scikit-learn dependency, omitted historical test fixtures in our first cloud upload, and a live-kernel import error after CNN pilot training. Logs are retained; failed execution was not counted as success. CI configuration was repaired, but no GitHub Actions run is claimed.')]
  parts+=['## '+('2. 实际执行范围与数据协议' if zh else '2. Executed scope and data protocol')]
  parts+=[table(['组','正式任务','实际内容'] if zh else ['Group','Tasks','Executed content'],[
   ('A',2,'Log-Mel 统计 + logistic；倒谱统计 + LinearSVC' if zh else 'Log-Mel statistics + logistic; cepstral statistics + LinearSVC'),
   ('B',15,'CNN、BiGRU、TA、CA、TCA × 3 seeds' if zh else 'CNN, BiGRU, TA, CA, TCA × three seeds'),
   ('C',9,'SpecAugment、waveform noise、waveform Mixup × 3 seeds' if zh else 'SpecAugment, waveform noise, waveform Mixup × three seeds'),
   ('D',36,'12 种特征/采样条件 × 3 seeds' if zh else '12 feature/sampling cases × three seeds'),
   ('E',3,'三种 seed101 checkpoint 各六种验证扰动条件' if zh else 'Six validation stress conditions for each of three seed101 checkpoints')])]
  parts+=[('三个 2-epoch pilot 和一个真实数据断点恢复 probe 单独保存，不计入 60 次正式训练或均值。PANNs/AST/BEATs/CLAP/RL 没有实现，因此没有执行或编造分数；本轮没有新增 fold 5 测试或五折确认。软件测试会核对旧 fixture，但这不等于对新模型进行新的测试评价。' if zh else
           'Three two-epoch pilots and one real-data resume probe are stored separately and excluded from the 60 main runs and their means. PANNs/AST/BEATs/CLAP/RL are not implemented; no execution or score is invented. No new fold 5 evaluation or confirmation CV was performed. Software tests check old fixtures, which is distinct from evaluating new models on test audio.')]
  parts+=[('数据为固定 commit 的 ESC-50：2,000 段、50 类、每段五秒。实际下载 WAV 并生成 v2 cache；2,000 行 cache 与原始 metadata 逐行对齐。训练 folds 1–3、验证 fold 4、指定测试角色 fold 5；排除跨角色 source ID 后为 **1200 / 398 / 400**。验证 classes 6 与 37 各 7 条，其余各 8 条。归一化只用训练录音拟合。原始音频跨 fold 精确重复检查通过，但没有完成感知近重复审计。' if zh else
           'The pinned ESC-50 data contains 2,000 five-second clips in 50 classes. Original WAVs were downloaded and a v2 cache prepared; all 2,000 cached identities/labels align with original metadata. Train folds 1–3, validation fold 4 and designated test role fold 5 become **1200 / 398 / 400** after source-strict exclusions. Validation classes 6 and 37 have seven clips each; all others have eight. Normalization is fitted only on training clips. Exact cross-fold duplicate checks passed; perceptual near-duplicate auditing is incomplete.')]
  parts+=[('主输入：16 kHz 单声道 → 512 点 Hann STFT、hop 256 → 64 个 HTK Mel 功率带 → natural log → train-only band normalization，shape `[B,1,64,313]`。encoder 输出 `[B,64,8,39]`；GRU 输入 `[B,39,512]`，双向 hidden size 64，输出 128 维。TA 做时间 softmax 加权平均；CA 做 64 → 8 → 64 门控，使用 2 × sigmoid 并初始化为恒等。' if zh else
           'Main input: 16 kHz mono → 512-sample Hann STFT, hop 256 → 64 HTK Mel-power bands → natural log → train-only band normalization, shape `[B,1,64,313]`. The encoder outputs `[B,64,8,39]`; GRU input is `[B,39,512]`, with 64 hidden units per direction and 128 output features. TA learns softmax time weights; CA uses 64 → 8 → 64 gates with 2 × sigmoid and identity initialization.')]
  parts+=[('神经网络统一训练 60 epochs、batch size 32、AdamW（lr 0.001、weight decay 0.0001）、cosine 调度及 gradient clipping（阈值 5）；主训练每个 epoch 1,200 条、38 次更新，总计 2,280 次更新。所有运行分配同样 60 轮预算，最终权重按最早最低验证 cross-entropy 选择，可能早于第 60 轮。Logistic 选最低验证 cross-entropy；SVM 选最高 accuracy，再按 macro-F1 及较小 C 打破平局。Accuracy 和 50 类等权 Macro-F1 均从逐样本预测重新计算。' if zh else
           'Neural runs use 60 epochs, batch size 32, AdamW (learning rate 0.001, weight decay 0.0001), cosine scheduling and gradient clipping at 5. Main training records 1,200 examples and 38 updates per epoch, or 2,280 updates in total. Every run receives the same 60-epoch budget; selected weights may come from an earlier epoch with the earliest minimum validation cross-entropy. Logistic also selects by validation cross-entropy; SVM selects accuracy, then macro-F1 and smaller C. Accuracy and equally weighted 50-class macro-F1 are recalculated from per-clip predictions.')]
  parts+=['## '+('3. 传统 baseline与结构结果' if zh else '3. Classical baselines and architecture results')]
  base=[r for r in runs if r['kind']=='baseline']
  parts+=[table(['Baseline','Accuracy','Macro-F1','选择参数'] if zh else ['Baseline','Accuracy','Macro-F1','Selected parameter'],[(r['model'],p(r['accuracy']),p(r['macro_f1']),r['selected_hyperparameter']) for r in base])]
  parts+=[('Logistic 是128 维 Log-Mel 时间均值/标准差；SVM 是20 个 DCT 系数的均值/标准差，共 40 维。二者改变了特征、分类器和选择规则，不能把差异单独归因于分类器。所有候选拟合均已检查收敛。' if zh else 'Logistic uses 128 temporal Log-Mel means/standard deviations; SVM uses 40 statistics from 20 DCT coefficients. Features, classifier and selection rule differ, so the score difference cannot be attributed to classifier choice alone. Candidate convergence was checked.')]
  rows=[]
  for m in MODELS:
   g=arch[m];single=[next(r for r in runs if r['id']==f'B_{m}_s{s}') for s in (101,202,303)]
   rows.append((NAMES[m],', '.join(p(r['accuracy']) for r in single),spread(g['accuracy_mean'],g['accuracy_sample_sd']),spread(g['macro_f1_mean'],g['macro_f1_sample_sd']),single[0]['parameters']))
  parts+=[table(['模型','Accuracy：101 / 202 / 303','Accuracy 均值 ± SD（%）','Macro-F1 均值 ± SD（%）','参数量'] if zh else ['Model','Accuracy: 101 / 202 / 303','Accuracy mean ± SD (%)','Macro-F1 mean ± SD (%)','Parameters'],rows)]
  parts+=[figure('architecture','三个 seed 的开发验证结果；SD 不是置信区间。' if zh else 'Three-seed development validation; SD is not a confidence interval.')]
  parts+=[('原 CNN 与 GRU 的六次新运行，在选中 epoch、逐条验证预测及权重 tensor 上与历史开发结果完全一致。这是本次实际重跑的回归验证，不是把旧分数复制成新实验。' if zh else 'All six new CNN/GRU runs match the historical development selected epochs, every validation prediction and weight tensors exactly. This is regression evidence from actual reruns, not reused old scores relabeled as new experiments.')]
  parts+=['## '+('4. 增强与配对变化' if zh else '4. Augmentation and paired changes')]
  rows=[('Clean CNN',spread(arch['cnn_freq']['accuracy_mean'],arch['cnn_freq']['accuracy_sample_sd']),spread(arch['cnn_freq']['macro_f1_mean'],arch['cnn_freq']['macro_f1_sample_sd']))]
  for name in ('specaugment','noise','mixup'):
   g=aug[name];rows.append((name,spread(g['accuracy_mean'],g['accuracy_sample_sd']),spread(g['macro_f1_mean'],g['macro_f1_sample_sd'])))
  parts+=[table(['训练条件','Accuracy 均值 ± SD（%）','Macro-F1 均值 ± SD（%）'] if zh else ['Training condition','Accuracy mean ± SD (%)','Macro-F1 mean ± SD (%)'],rows),figure('augmentation','增强仅用于训练，验证声音始终保持干净。' if zh else 'Augmentation affects training only; validation recordings stay clean.')]
  parts+=[('SpecAugment 遮挡归一化后的时间/频带区域；noise 以 0.5 概率添加 10–30 dB 的 Gaussian noise；Mixup 使用 Beta(0.4, 0.4) 混合波形和两个标签的损失。增强训练 loss 与干净训练 loss 不是相同目标，不用它们的大小直接判断谁更好。' if zh else 'SpecAugment masks normalized time/frequency regions; noise adds Gaussian corruption with probability 0.5 at 10–30 dB; Mixup combines waveforms and both label losses with Beta(0.4, 0.4). Augmented and clean training losses are different objectives and are not directly ranked against one another.')]
  effectrows=[]
  for e in effects:
   if e['effect'].startswith(('sample_rate','added_high','representation')):continue
   vals=[r for r in pairs if r['effect']==e['effect'] and r['left'].rsplit('_s',1)[0]==e['left']]
   effectrows.append((e['effect'],', '.join(f'{float(r["delta_accuracy_pp"]):+.2f}' for r in sorted(vals,key=lambda r:int(r['seed']))),f'{float(e["mean_delta_accuracy_pp"]):+.2f} ± {float(e["sample_sd_delta_accuracy_pp"]):.2f}',f'{float(e["mean_delta_macro_f1_pp"]):+.2f}'))
  parts+=[table(['配对对照','各 seed accuracy 变化（pp）','均值±SD（pp）','平均 Macro-F1 变化（pp）'] if zh else ['Paired contrast','Per-seed accuracy delta (pp)','Mean ± SD (pp)','Mean macro-F1 delta (pp)'],effectrows),figure('paired_effects','同 seed 配对变化；不是显著性检验。' if zh else 'Within-seed deltas; these are not significance tests.')]
  parts+=[('Attention interaction 为 TCA − TA − CA + 原 GRU，用于描述两模块的组合偏离简单相加的程度，不是无混杂的因果机制证明。三个 seed 重复使用同一 398 条验证录音；它们不是三个独立数据集，也不能用 seed SD 替代数据采样置信区间。' if zh else 'Attention interaction is TCA − TA − CA + original GRU, describing deviation from additive module effects rather than proving an unconfounded causal mechanism. Three seeds reuse the same 398 validation clips; they are not independent datasets, and seed SD is not a dataset-sampling confidence interval.')]
  parts+=['## '+('5. 十二组特征与采样率对照' if zh else '5. Twelve feature and sampling-rate conditions')]
  parts+=[table(['Case','Accuracy 均值 ± SD（%）','Macro-F1 均值 ± SD（%）'] if zh else ['Case','Accuracy mean ± SD (%)','Macro-F1 mean ± SD (%)'],[(g['case'],spread(g['accuracy_mean'],g['accuracy_sample_sd']),spread(g['macro_f1_mean'],g['macro_f1_sample_sd'])) for g in sorted(feat,key=lambda g:g['case'])]),figure('feature_sampling','D 组使用 26,658 参数的 pooled CNN；与主结构组分开解释。' if zh else 'Track D uses the 26,658-parameter pooled CNN; interpret separately from the main architecture track.')]
  parts+=[('四种表示是 Log-Mel 64、Log-Mel 32、64 维 Log-Mel 的前 40 个正交 DCT-II 系数、Log-linear 64。三种条件为 16 kHz/F8、32 kHz/F8、32 kHz/F16；窗口均 32 ms、hop 均 16 ms、313 帧。16k/F8 → 32k/F8 匹配可用带宽；32k/F8 → 32k/F16 增加高频，也改变滤波器间距。MFCC 改变了行数和相邻坐标意义；相同参数量不等于相同计算量或纯单变量因果实验。全部 17 类 × 3 seeds 的配对差异见 CSV，缺失项没有补成 0。' if zh else 'Representations are Log-Mel 64, Log-Mel 32, the first 40 orthonormal DCT-II coefficients of 64-band Log-Mel, and Log-linear 64. Conditions are 16k/F8, 32k/F8 and 32k/F16, with 32 ms windows, 16 ms hops and 313 frames. The 16k/F8 → 32k/F8 comparison matches usable bandwidth; 32k/F8 → 32k/F16 adds high frequencies and changes filter spacing. MFCC changes row count and coordinate meaning. Equal parameter count does not imply equal computation or a pure single-variable causal experiment. All 17 × 3 within-track paired contrasts are retained in CSV; missing scores were never imputed as zero.')]
  parts+=['## '+('6. 噪声与增益敏感性' if zh else '6. Noise and gain sensitivity')]
  conds=['clean','snr_20','snr_10','snr_0','gain_-6','gain_6']
  parts+=[table(['条件','干净 CNN','干净 BiGRU','加噪训练 CNN'] if zh else ['Condition','Clean-trained CNN','Clean-trained BiGRU','Noise-trained CNN'],[(c,*(p(stressval(t,c)) for t in ('E_clean_cnn','E_clean_crnn','E_noise_cnn'))) for c in conds]),figure('robustness','每格 398 条录音；每个模型仅使用一个选中的 seed101 checkpoint。' if zh else '398 clips per cell; one selected seed101 checkpoint per model.')]
  parts+=[('这些是一次确定性、逐录音 RMS 缩放的 Gaussian noise 与 gain 干预，三个模型使用相同扰动实现；没有峰值裁剪。0 dB 意味着添加噪声与原录音 RMS 功率相同，不意味着真实背景 SNR 已知。即使某些条件改善，也不能声称已覆盖现实噪声、房间、设备或未知类别。E 组仅有 3 个 checkpoint，没有三个 seed 的鲁棒性结论。' if zh else 'These are deterministic, per-clip RMS-scaled Gaussian-noise and gain interventions, with identical realizations across models and no peak clipping. 0 dB equates added-noise and original-recording RMS power; the original ambient SNR is unknown. Improvements on these conditions do not establish robustness to field noise, rooms, devices or unknown classes. Track E covers three checkpoints, not three-seed robustness.')]
  parts+=['## '+('7. 完整性、重执行与效率证据' if zh else '7. Integrity, replay and efficiency evidence')]
  parts+=[(f'独立脚本从 CSV 按 50 类重新计算混淆矩阵、Accuracy 与 Macro-F1，核对全部 65 项任务、60 轮预算、选中 epoch 及配置。随后在 Colab 从原始验证 WAV重新通过项目前端与模型，**62 个拟合模型 × 398 条 = 24,676 条预测全部重现**；这是实际 forward 重执行，不是只读取保存的分数。断点恢复 probe 在真实数据上完成 1 轮、恢复后完成第 2 轮，与不中断 pilot 的权重逐位相同。恢复结论仅覆盖相同 runtime/code/data，不承诺跨版本或跨设备。' if zh else
           'An independent script recalculates 50-class confusion matrices, accuracy and macro-F1 from CSVs, checking all 65 tasks, full epoch budgets, selected epochs and configurations. Colab then re-executes the project frontends and models from original validation WAVs: **62 fitted models × 398 clips = 24,676 predictions all reproduce**. This is actual forward replay rather than reading saved scores. A real-data probe pauses after one epoch and resumes for epoch 2, matching uninterrupted pilot weights bit-for-bit. Resume evidence applies to the same runtime/code/data, not arbitrary versions or devices.')]
  latencyrows=[]
  for m in MODELS:
   measures=[next(r for r in timing['measurements'] if r['model']==m and r['device']==dev) for dev in ('cpu','cuda')]
   latencyrows.append((NAMES[m],*(f'{r["wav_to_logits"]["p95_ms"]:.3f}' for r in measures),f'{measures[0]["checkpoint_bytes"]/1024:.1f}'))
  parts+=[table(['模型','CPU WAV→logits p95(ms)','T4 WAV→logits p95(ms)','checkpoint(KiB)'] if zh else ['Model','CPU WAV→logits p95(ms)','T4 WAV→logits p95(ms)','Checkpoint(KiB)'],latencyrows)]
  parts+=[(f'测量为 Colab `{timing["cpu_model"]}` / `{timing["gpu"]}`，2 CPU threads、batch size 1、float32、10 次 warmup、100 次重复、同一验证 WAV。GPU 计时同步；包括解码、前端与传输，排除模型加载、进程启动与服务排队。Torch 峰值 allocated memory 不是驱动总显存。60 次训练循环记录耗时之和为 {audit["runtime"]["neural_training_seconds"]/60:.2f} 分钟，包括循环内验证与保存；排除初始化、cache/WAV 加载及验证、数据/特征准备、pilot、probe、空闲和导出；不当作整个会话时间。付费支出 0，没有升级或购买资源。' if zh else
           f'Measured on Colab `{timing["cpu_model"]}` / `{timing["gpu"]}` with 2 CPU threads, batch size 1, float32, 10 warmups, 100 repetitions and one fixed validation WAV. CUDA is synchronized. WAV-to-logits includes decoding, frontend and transfer, excluding model loading, process startup and service queues. Torch peak allocated memory is not total driver memory. Recorded training-loop wall times for 60 runs sum to {audit["runtime"]["neural_training_seconds"]/60:.2f} minutes, including in-loop validation/checkpoint writes but excluding initialization, cache/WAV loading and verification, acquisition/preparation, pilots/probes, idle and export; this is not total session time. Paid spend was 0, with no upgrade or resource purchase.')]
  parts+=[figure('learning_curves','固定 seed101 验证曲线；checkpoint 按最低 cross-entropy 选择。' if zh else 'Fixed-seed101 validation curves; checkpoints use minimum cross-entropy.')]
  error_rows=[]
  category_rows=list(csv.DictReader((ROOT/'data/ESC-50/meta/esc50.csv').open()))
  categories={int(r['target']):r['category'] for r in category_rows}
  for m in MODELS:
   metrics=json.loads((R/'execution/runs/B_architecture'/f'B_{m}_s101/validation.json').read_text())
   cm=metrics['confusion_matrix'];support=[sum(row) for row in cm]
   worst=sorted(range(50),key=lambda k:(cm[k][k]/support[k],k))[:3]
   error_rows.append((NAMES[m],sum(cm[k][k]==0 for k in range(50)), '; '.join(f'{categories[k]} {cm[k][k]}/{support[k]}' for k in worst)))
  parts+=['### '+('类别错误切片：固定seed101' if zh else 'Per-class error slices: fixed seed101')]
  parts+=[table(['模型','零recall类别数','最低recall的三类：正确/支持'] if zh else ['Model','Zero-recall classes','Three lowest-recall classes:correct/support'],error_rows)]
  parts+=[('每类仅7或8段验证录音，单段错误会显著改变该类别recall；这些错误切片是探索性诊断。完整混淆矩阵和逐条预测随每个run保存，不能用平均分掩盖具体类别的失败。' if zh else 'Each class has only 7 or 8 validation clips, so one error substantially changes per-class recall. These slices are exploratory diagnostics. Every run retains its full confusion matrix and per-clip predictions; an aggregate score must not conceal class-specific failures.')]
  parts+=['## '+('8. 结论边界与下一项决策' if zh else '8. Limits and next decision')]
  parts+=[('本轮回答了结构、增强、特征与人工扰动在既定开发协议下的表现，没有建立新的 SOTA、算法新颖性或生产部署结论。Attention 并非必然有效；应考察三个 seed 及小幅变化，而不只看最高分。60 轮的分配一致，不等于不同结构参数、FLOPs、延迟完全匹配。BiGRU 是离线双向模型，没有证明实时因果流式识别。Attention 权重也不是可靠因果解释。' if zh else
           'This round establishes architecture, augmentation, representation and synthetic-stress behavior within the declared development protocol, not SOTA, algorithmic novelty or production readiness. Attention is not guaranteed to help; assess all seeds and small effects instead of the highest single score. Equal epoch budgets do not match parameters, FLOPs and latency simultaneously. The bidirectional GRU is an offline model, not evidence of causal streaming recognition. Attention weights are not reliable causal explanations.')]
  parts+=[('**不要把本轮约 70% 以上的 validation 结果，与旧报告约 62% 的 test 结果相减，称为提升。** 它们是不同录音、不同用途的指标。历史默认 CNN的三个 seed 的 fold 5 均值 63.08%、GRU 63.92%，仅作旧证据索引；本轮没有重新确认这些数值。所有 ESC-50 folds 都有既往使用历史，继续挑方案仍是探索。下一步若要确认候选，应先定义新的、来源隔离的评价边界及真实噪声需求；不能通过重命名旧 fold 恢复盲测。' if zh else
           '**Do not subtract the old roughly 62% test result from the new 70%+ validation results and call the gap improvement.** They measure different recordings and purposes. Historical fold 5 three-seed means are 63.08% for CNN and 63.92% for GRU, referenced only as old evidence; this round does not reconfirm them. All ESC-50 folds have prior exposure, so further selection remains exploratory. Confirmation requires a new source-isolated evaluation boundary and defined real-noise needs; renaming an old fold cannot restore blindness.')]
  parts+=[('旧 CNN 交付包保留为历史默认；本次所有新模型以研究 checkpoint 交付，完整保留失败与负面结果。未修改考试学习记录，未将 Agent 产物当作用户独立掌握，也未冒称外部人类复现或机构认证。' if zh else 'The old CNN bundle remains the historical default. All new models are delivered as research checkpoints, retaining negative findings and failure logs. CV exam learning records were untouched; agent-produced work is not presented as user mastery, external human replication or institutional certification.')]
  parts+=['## '+('9. 证据入口与复现' if zh else '9. Evidence and reproduction')]
  parts+=[('[全部逐运行结果](../../results/matrix_20261028/analysis/runs.csv) · [三个 seed 汇总](../../results/matrix_20261028/analysis/groups.csv) · [75 个配对变化](../../results/matrix_20261028/analysis/paired_contrasts.csv) · [鲁棒性](../../results/matrix_20261028/analysis/stress.csv) · [完整审计](../../results/matrix_20261028/analysis/audit.json) · [WAV 重执行](../../results/matrix_20261028/runtime_verification/waveform_replay.json) · [恢复检查](../../results/matrix_20261028/runtime_verification/resume_probe.json) · [计时定义](../../results/matrix_20261028/runtime_verification/warm_latency.json)' if zh else
           '[Per-run results](../../results/matrix_20261028/analysis/runs.csv) · [Three-seed aggregates](../../results/matrix_20261028/analysis/groups.csv) · [75 paired deltas](../../results/matrix_20261028/analysis/paired_contrasts.csv) · [Stress results](../../results/matrix_20261028/analysis/stress.csv) · [Complete audit](../../results/matrix_20261028/analysis/audit.json) · [WAV replay](../../results/matrix_20261028/runtime_verification/waveform_replay.json) · [Resume check](../../results/matrix_20261028/runtime_verification/resume_probe.json) · [Timing definition](../../results/matrix_20261028/runtime_verification/warm_latency.json)')]
  parts+=[('环境：' if zh else 'Environment: ')+', '.join(f'{k}={environment[k]}' for k in ('torch','numpy','scipy','soundfile','sklearn'))+'.']
  parts+=["```bash\npython -m pip install -e '.[dev,baseline,report]'\npython -m esc50_sound prepare\npython -m pytest -q\npython scripts/analyze_colab_matrix.py --require-complete\npython scripts/plot_colab_matrix.py\n```"]
  parts+=[('完整矩阵使用 `notebooks/templates/ESC50_Matrix_Full_20261028.ipynb`；它会实际训练所有任务，请先读预算设置。安全预览入口仍为 `ESC50_Colab_Research_Console.ipynb` 的 plan mode。特征实验不能 epoch-resume；主训练恢复需要匹配环境和 last.pt。Colab 磁盘是临时的，本轮逐任务导出并下载；notebook 本身不等于所有权重与音频。' if zh else
           'The full matrix uses `notebooks/templates/ESC50_Matrix_Full_20261028.ipynb`; it trains all tasks, so read its budget settings first. The safe preview remains plan mode in `ESC50_Colab_Research_Console.ipynb`. Feature runs cannot resume by epoch; main training requires matching environment and last.pt. Colab disk is temporary; this round exported/downloaded every task. A saved notebook is not a copy of all weights or audio.')]
  parts+=[('原始数据与许可：[官方 ESC-50](https://github.com/karolpiczak/ESC-50)，固定 commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`。数据整体使用 CC BY-NC 3.0 许可，保留 credits；本项目不授予商业权利。[Colab 资源说明](https://research.google.com/colaboratory/faq.html)。方法原始参考与核验深度见[文献阅读记录](../literature_deep_read.md)；本轮结果的直接来源是保存的真实运行及逐样本输出。' if zh else
           'Data and license: [official ESC-50](https://github.com/karolpiczak/ESC-50), pinned commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. Overall dataset license is CC BY-NC 3.0; retain credits. This project grants no commercial rights. See [Colab resource information](https://research.google.com/colaboratory/faq.html). Primary method references and reading depth appear in the [literature reading notes](../literature_deep_read.md); actual saved runs and per-clip outputs directly support this round\'s results.')]
  D.mkdir(exist_ok=True);(D/f'report_{lang}.md').write_text('\n\n'.join(parts)+'\n')
 print('Wrote bilingual result reports with complete audited evidence')

if __name__=='__main__':main()
