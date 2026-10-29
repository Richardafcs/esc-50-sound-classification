#!/usr/bin/env python3
"""Self-contained bilingual results reader; embeds audited tables, never retrains."""
from pathlib import Path
import csv,json,re
import nbformat
P=Path(__file__).resolve().parents[1];R=P / 'results/matrix_20261028'

def csvrows(name):
 with (R/'analysis'/name).open(newline='') as f:return list(csv.DictReader(f))

def main():
 audit=json.loads((R/'analysis/audit.json').read_text());assert audit['complete']
 replay=json.loads((R/'runtime_verification/waveform_replay.json').read_text())
 resume=json.loads((R/'runtime_verification/resume_probe.json').read_text())
 snapshot=dict(groups=audit['groups'],runs=[{k:r[k] for k in ('model','kind','accuracy','macro_f1','selected_hyperparameter')} for r in csvrows('runs.csv') if r['kind']=='baseline'],stress=audit['stress'],effects=audit['paired_effects'],environment=json.loads((R/'runtime_environment.json').read_text()),verification=dict(audited_tasks=65,neural_runs=60,epochs_per_neural_run=60,waveform_replay_models=len(replay['runs']),waveform_replay_clips_each=398,waveform_replay_changes=sum(r['changed_predictions'] for r in replay['runs']),same_runtime_resume_equal=resume['weights_bitwise_equal_to_uninterrupted_pilot']))
 nb=nbformat.v4.new_notebook(metadata=dict(kernelspec=dict(name='python3',display_name='Python 3'),language_info=dict(name='python'),colab=dict(name='ESC50_Bilingual_Results_20261028.ipynb')))
 md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
 nb.cells=[md('''# ESC-50 · 中文与英文结果 / Bilingual Results

**中文：** 本轮在 Colab Tesla T4 实际完成 65 项开发任务：60 次 60-epoch 神经网络训练、两项传统 baseline、三项鲁棒性任务。本 notebook 是**已经审计结果的自包含阅读器**，不会重新训练。完整训练记录保存在 `ESC50_Matrix_Training_20261028_output.ipynb` 和研究交付包中。所有分数是开发验证结果，不是新的独立测试。

**English:** This round executed 65 development tasks on a Colab Tesla T4: 60 neural runs of 60 epochs, two classical baselines and three stress tasks. This notebook is a **self-contained reader of audited results**, not a retraining notebook. Training execution evidence is retained in `ESC50_Matrix_Training_20261028_output.ipynb` and the research package. All scores below are development-validation measurements, not fresh independent test results.

The readable snapshot below is embedded so tables and plots rerun without audio, weights, Drive access or a GPU. / 下方嵌入可读数据，重画表格与图无需音频、权重、Drive 授权或 GPU。''')]
 snapshot_text='{\n'+',\n'.join(json.dumps(k)+': '+('[\n'+',\n'.join(json.dumps(r,ensure_ascii=False) for r in v)+'\n]' if isinstance(v,list) else json.dumps(v,ensure_ascii=False,indent=2)) for k,v in snapshot.items())+'\n}'
 nb.cells+=[code("import json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display\n\n# Saved audited data; no model fitting or evaluation occurs in this notebook.\nSNAPSHOT=json.loads(r'''"+snapshot_text+"''')\ngroups=pd.DataFrame(SNAPSHOT['groups'])\nruns=pd.DataFrame(SNAPSHOT['runs'])\nplt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})\nprint('Audited tasks / 已审计任务:', SNAPSHOT['verification']['audited_tasks'])\nprint('Runtime used for training / 训练环境:', SNAPSHOT['environment'])")]
 nb.cells+=[md('''## 数据与方法 / Data and methods

**中文：** 2,000 段五秒录音、50 类。按原始来源隔离，训练 1,200 段、验证 398 段，指定测试角色的 400 段录音本轮不评价。主输入是 16 kHz 单声道的 64 带 Log-Mel，shape `[B,1,64,313]`。CNN 提取局部模式；BiGRU 沿 39 个时间 token 汇总；Temporal Attention 学习时间权重，Channel Attention 调整特征通道。归一化只从训练录音拟合。神经网络均训练 60 轮，按最早最低验证 cross-entropy 选 checkpoint。

**English:** There are 2,000 five-second recordings and 50 classes. Source-strict roles contain 1,200 training, 398 validation and 400 designated test clips; the latter are not evaluated in this round. Main input is 16 kHz mono 64-band Log-Mel, shape `[B,1,64,313]`. The CNN extracts local patterns; the BiGRU aggregates 39 time tokens. Temporal attention learns time weights; channel attention reweights learned feature maps. Normalization uses training clips only. Neural runs train for 60 epochs and select the earliest minimum-validation-cross-entropy checkpoint.

Three seeds describe training variability on the same 398 clips. Their SD is not a confidence interval for new data. / 三个 seed 反复评价同一 398 条录音，其 SD 不是新数据表现的置信区间。''')]
 nb.cells+=[code('''models=['cnn_freq','crnn_freq','crnn_freq_ta','crnn_freq_ca','crnn_freq_tca']
labels=['CNN','CNN + BiGRU','BiGRU + Temporal Attention','BiGRU + Channel Attention','BiGRU + Both Attentions']
arch=groups[groups.group.eq('B_architecture')].set_index('model').loc[models].reset_index()
view=arch[['model','accuracy_mean','accuracy_sample_sd','macro_f1_mean','macro_f1_sample_sd']].copy()
for key in view.columns[1:]:view[key]=100*view[key]
display(view.round(2))
fig,ax=plt.subplots(figsize=(10,4.8))
y=np.arange(len(arch));values=100*arch.accuracy_mean.to_numpy();sd=100*arch.accuracy_sample_sd.to_numpy()
ax.barh(y,values,xerr=sd,color='#255F91',capsize=4)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,100)
ax.set_xlabel('Validation accuracy (%) — mean ± sample SD, three seeds')
# English chart labels avoid dependence on Chinese fonts in fresh runtimes.
ax.set_title('Architecture comparison — 398 validation clips per seed')
for yy,v,s in zip(y,values,sd):ax.text(v+s+1,yy,f'{v:.2f} ± {s:.2f}',va='center')
plt.tight_layout();plt.show()
base=runs[runs.kind.eq('baseline')][['model','accuracy','macro_f1','selected_hyperparameter']].copy()
for key in ('accuracy','macro_f1'):base[key]=100*base[key].astype(float)
display(base.round(2))''')]
 nb.cells+=[md('''## 训练增强 / Training augmentation

**中文：** SpecAugment 遮挡训练特征；Gaussian noise 以 0.5 概率添加 10–30 dB 噪声；waveform Mixup 同时混合音频和标签损失。验证集始终保持干净。看三个 seed 的均值与代价，而非挑最好一次。

**English:** SpecAugment masks training features; Gaussian noise is added with probability 0.5 at 10–30 dB; waveform Mixup combines audio and label losses. Validation stays clean. Evaluate all seeds and trade-offs rather than selecting the best run.''')]
 nb.cells+=[code('''aug=groups[groups.group.eq('C_augmentation')].copy().set_index('augmentation').loc[['specaugment','noise','mixup']].reset_index()
clean=arch[arch.model.eq('cnn_freq')].copy();clean['augmentation']='clean'
aug=pd.concat([clean,aug],ignore_index=True)
view=aug[['augmentation','accuracy_mean','accuracy_sample_sd','macro_f1_mean']].copy()
for key in view.columns[1:]:view[key]*=100
display(view.round(2))
fig,ax=plt.subplots(figsize=(9,4))
ax.barh(np.arange(4),100*aug.accuracy_mean,xerr=100*aug.accuracy_sample_sd,color=['#255F91','#C77031','#657F45','#AE5478'],capsize=4)
ax.set_yticks(np.arange(4),['Clean CNN','SpecAugment','Waveform noise','Waveform Mixup']);ax.invert_yaxis();ax.set_xlim(0,100)
ax.set_xlabel('Clean validation accuracy (%) — mean ± sample SD')
plt.tight_layout();plt.show()
main_effects=pd.DataFrame(SNAPSHOT['effects'])
display(main_effects[~main_effects.effect.str.startswith(('sample_rate','added_high','representation'))][['effect','n_pairs','mean_delta_accuracy_pp','sample_sd_delta_accuracy_pp','mean_delta_macro_f1_pp']].round(3))''')]
 nb.cells+=[md('''## 特征与采样率 / Features and sampling rates

**中文：** 12 种条件 = 4 种表示 × 3 种采样/带宽条件。D 组统一使用 26,658 参数的频率全局池化 CNN，与主结构组的 `cnn_freq` 不同。16k/F8 → 32k/F8 匹配带宽；32k/F8 → 32k/F16 同时增加高频并改变滤波器间距。参数相同不意味着计算量或物理解释完全相同。

**English:** Twelve cases combine four representations with three sampling/bandwidth conditions. Track D uses the 26,658-parameter frequency-pooled CNN, different from the main `cnn_freq`. The 16k/F8 → 32k/F8 comparison matches bandwidth; 32k/F8 → 32k/F16 adds high frequencies and changes filter spacing. Equal parameters do not establish equal computation or a pure causal interpretation.''')]
 nb.cells+=[code('''feature=groups[groups.group.eq('D_feature_sampling')].copy()
view=feature[['case','accuracy_mean','accuracy_sample_sd','macro_f1_mean']].copy()
for key in view.columns[1:]:view[key]*=100
display(view.round(2))
conditions=['sr16_fmax8','sr32_fmax8','sr32_fmax16'];representations=['logmel64','logmel32','mfcc40','loglinear64']
values=np.array([[100*float(feature[feature['case'].eq(c+'_'+f)].accuracy_mean.iloc[0]) for c in conditions] for f in representations])
fig,ax=plt.subplots(figsize=(8.5,4.8));im=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
ax.set_xticks(range(3),['16k / max8k','32k / max8k','32k / max16k']);ax.set_yticks(range(4),representations)
for i in range(4):
    for j in range(3):ax.text(j,i,f'{values[i,j]:.2f}%',ha='center',va='center',color='white' if values[i,j]>65 else 'black')
fig.colorbar(im,ax=ax,label='Mean validation accuracy (%)')
ax.set_title('Feature track — pooled CNN, three seeds')
plt.tight_layout();plt.show()''')]
 nb.cells+=[md('''## 噪声与增益 / Noise and gain stress

**中文：** 以下每格 398 条验证录音，每个模型只评价一个 seed101 checkpoint；三个模型使用相同的逐录音确定性扰动。0 dB 指添加噪声与录音 RMS 功率相等。它不是对所有真实噪声和设备的鲁棒性证明。

**English:** Each cell covers 398 validation recordings and one seed101 checkpoint per model, using matched deterministic per-clip corruptions. 0 dB equates added-noise and recording RMS power. This is not proof of robustness to all field noise or devices.''')]
 nb.cells+=[code('''stress=pd.DataFrame(SNAPSHOT['stress'])
order=['clean','snr_20','snr_10','snr_0','gain_-6','gain_6']
pivot=stress.pivot(index='condition',columns='task_id',values='accuracy').loc[order,['E_clean_cnn','E_clean_crnn','E_noise_cnn']]*100
display(pivot.round(2))
ax=pivot.plot.bar(figsize=(11,4.8),color=['#255F91','#B37B16','#657F45'],rot=0)
ax.set_ylim(0,100);ax.set_ylabel('Validation accuracy (%)');ax.set_xlabel('Synthetic condition')
ax.set_title('Synthetic stress — seed101 only')
ax.legend(['Clean-trained CNN','Clean-trained BiGRU','Noise-trained CNN'],frameon=False)
plt.tight_layout();plt.show()''')]
 nb.cells+=[md('''## 完整性与解释边界 / Verification and interpretation limits

**中文：** Accuracy 是正确录音比例；Macro-F1 让 50 类等权。所有 65 项任务均从逐样本 CSV 重新核算。随后从原始验证 WAV 通过项目前端重新执行 62 个拟合模型，核对全部 24,676 条预测。真实数据断点恢复 probe 也与不中断 pilot 逐位一致。这是内部执行一致性检查，不是外部人类独立复现。

**English:** Accuracy counts correctly classified clips; macro-F1 weights all 50 classes equally. All 65 tasks were audited from prediction CSVs. Replaying 62 fitted models from original validation WAVs matched 24,676 predictions. A real-data epoch-resume probe also matched an uninterrupted pilot bit-for-bit. These are internal execution-consistency checks, not external human replication.

**中文：** 旧项目已经访问过所有 ESC-50 folds。本轮不评价 fold 5，但也不能把已见过的数据重新称为盲测。不要把约 70% 以上的验证分数与旧约 62% 的测试分数相减，声称模型提升。Attention 不是必然有效，模拟噪声也不等于现实鲁棒性；没有 SOTA、算法新颖性、生产部署或学习者掌握的认证。

**English:** The earlier project already accessed every ESC-50 fold. This round does not evaluate fold 5, but cannot restore blindness to previously used data. Do not subtract the old roughly 62% test scores from the new 70%+ validation scores and call the difference improvement. Attention is not guaranteed to help; synthetic noise is not field robustness. No SOTA, algorithmic-novelty, production-readiness or learner-mastery certification is claimed.''')]
 nb.cells+=[code("display(pd.DataFrame([SNAPSHOT['verification']]).T.rename(columns={0:'Observed result / 实际结果'}))\nassert SNAPSHOT['verification']['audited_tasks']==65\nassert SNAPSHOT['verification']['waveform_replay_models']==62\nassert SNAPSHOT['verification']['waveform_replay_changes']==0\nassert SNAPSHOT['verification']['same_runtime_resume_equal'] is True\nprint('All saved verification checks passed / 保存的核验全部通过')")]
 for lang,title in [('zh','中文详细说明'),('en','Detailed English guide')]:
  text=(P/f'docs/guides/guide_{lang}.md').read_text()
  text=text.split('\n\n![',1)[0]  # The local guides retain the full-resolution illustration.
  text=re.sub(r'\[([^\]]+)\]\((?!https?://)[^)]+\)',r'\1',text)
  nb.cells.append(md('## '+title+'\n\n'+text))
 nbformat.validate(nb)
 path=P / 'notebooks/templates/ESC50_Bilingual_Results_20261028.ipynb';nbformat.write(nb,path)
 print(path)

if __name__=='__main__':main()
