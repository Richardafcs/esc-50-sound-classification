#!/usr/bin/env python3
"""Publication-style figures from audited new validation records (no evaluation)."""
from pathlib import Path
import csv,argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BLUE='#255F91';GOLD='#B37B16';OLIVE='#657F45';PINK='#AE5478';ORANGE='#C77031'
MODELS=['cnn_freq','crnn_freq','crnn_freq_ta','crnn_freq_ca','crnn_freq_tca']
LABELS=['CNN','CNN + BiGRU','BiGRU + temporal attention','BiGRU + channel attention','BiGRU + both attentions']
COLORS=dict(zip(MODELS,[BLUE,GOLD,ORANGE,OLIVE,PINK]))

def load(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)

def bars(groups,runs,labels,colors,title,out):
    if not groups:return
    fig,ax=plt.subplots(figsize=(11,max(3.5,.66*len(groups)+1.5)))
    for i,(g,label,color) in enumerate(zip(groups,labels,colors)):
        mean=100*float(g['accuracy_mean']);sd=100*float(g['accuracy_sample_sd'])
        ax.barh(i,mean,color=color,alpha=.85,height=.6)
        ax.errorbar(mean,i,xerr=sd,color='#161E27',capsize=4,fmt='none',linewidth=1.3)
        values=sorted([r for r in runs if r['group']==g['group'] and r['model']==g['model'] and r['augmentation']==g['augmentation'] and r['case']==g['case'] and r['status']=='audited'],key=lambda r:int(r['seed']))
        for offset,r,marker in zip([-.13,0,.13],values,['o','s','^']):
            ax.scatter(100*float(r['accuracy']),i+offset,marker=marker,facecolors='white',edgecolors='#17202A',s=35,zorder=4)
        ax.text(mean+sd+1.2,i,f'{mean:.2f} ± {sd:.2f}',va='center',fontsize=10)
    ax.set_yticks(range(len(groups)),labels);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('Validation accuracy (%) — 398 clips per seed')
    ax.set_title(title,loc='left',pad=16,fontweight='bold')
    ax.text(0,1.02,'Bars = mean; whiskers = sample SD (not a confidence interval); points: ○101 / □202 / △303.',transform=ax.transAxes,fontsize=9,color='#4C5662')
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);fig.tight_layout();save(fig,out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,default=ROOT / 'results/matrix_20261028/analysis');a=ap.parse_args()
    groups=load(a.analysis/'groups.csv');runs=load(a.analysis/'runs.csv')
    full=[g for g in groups if g['complete_three_seeds']=='True']
    out=a.analysis.parent/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none'})
    arch=[next((g for g in full if g['group']=='B_architecture' and g['model']==m),None) for m in MODELS]
    present=[i for i,g in enumerate(arch) if g]
    bars([arch[i] for i in present],runs,[LABELS[i] for i in present],[COLORS[MODELS[i]] for i in present],'Architecture comparison — three training seeds',out/'architecture')
    aug=[next((g for g in full if g['group']=='B_architecture' and g['model']=='cnn_freq'),None)]+[next((g for g in full if g['group']=='C_augmentation' and g['augmentation']==name),None) for name in ('specaugment','noise','mixup')]
    present=[i for i,g in enumerate(aug) if g]
    bars([aug[i] for i in present],runs,[['Clean CNN','CNN + SpecAugment','CNN + waveform noise','CNN + waveform Mixup'][i] for i in present],[BLUE,ORANGE,OLIVE,PINK][:len(present)],'Training augmentation — clean validation recordings',out/'augmentation')
    feats=['logmel64','logmel32','mfcc40','loglinear64'];conditions=['sr16_fmax8','sr32_fmax8','sr32_fmax16']
    values=np.full((4,3),np.nan);sds=values.copy()
    for i,feat in enumerate(feats):
        for k,cond in enumerate(conditions):
            g=next((g for g in full if g['case']==cond+'_'+feat),None)
            if g:values[i,k]=100*float(g['accuracy_mean']);sds[i,k]=100*float(g['accuracy_sample_sd'])
    if np.isfinite(values).any():
        fig,ax=plt.subplots(figsize=(9,5.5));cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#E9EDF1')
        im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=100,cmap=cmap,aspect='auto')
        for i in range(4):
            for k in range(3):
                text=f'{values[i,k]:.2f}%\n± {sds[i,k]:.2f} pp' if np.isfinite(values[i,k]) else 'Not completed'
                ax.text(k,i,text,ha='center',va='center',color='white' if values[i,k]>65 else '#18222C',fontsize=11)
        ax.set_xticks(range(3),['16 kHz / 8 kHz max','32 kHz / 8 kHz max','32 kHz / 16 kHz max'])
        ax.set_yticks(range(4),['Log-Mel 64','Log-Mel 32','MFCC-like 40','Log-linear 64'])
        ax.set_title('Feature / sample-rate track — pooled CNN, 26,658 parameters',loc='left',pad=18,fontweight='bold')
        fig.colorbar(im,ax=ax,label='Mean validation accuracy (%)')
        fig.text(.02,.01,'Each completed cell: three seeds, 398 validation clips per seed; ± sample SD, not a confidence interval.',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,1));save(fig,out/'feature_sampling')
    stress=load(a.analysis/'stress.csv')
    if stress:
        tasks=['E_clean_cnn','E_clean_crnn','E_noise_cnn'];conds=['clean','snr_20','snr_10','snr_0','gain_-6','gain_6']
        fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(conds))
        for offset,task,color,label in zip([-.24,0,.24],tasks,[BLUE,GOLD,OLIVE],['Clean-trained CNN','Clean-trained BiGRU','Noise-trained CNN']):
            vals=[next((100*float(r['accuracy']) for r in stress if r['task_id']==task and r['condition']==c),np.nan) for c in conds]
            ax.bar(x+offset,vals,width=.23,color=color,label=label)
            for xx,yy in zip(x+offset,vals):
                if np.isfinite(yy):ax.text(xx,yy+1,f'{yy:.1f}',ha='center',fontsize=9)
        ax.set_xticks(x,['Clean','20 dB SNR','10 dB SNR','0 dB SNR','−6 dB gain','+6 dB gain']);ax.set_ylim(0,100)
        ax.set_ylabel('Validation accuracy (%)');ax.set_title('Synthetic stress — seed 101 only, 398 clips per condition',loc='left',pad=32,fontweight='bold')
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,fontsize=10);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        fig.tight_layout();save(fig,out/'robustness')
    pairs=load(a.analysis/'paired_contrasts.csv')
    effects=[('gru_vs_cnn','BiGRU − CNN'),('attention_ta_vs_gru','Temporal attention − BiGRU'),('attention_ca_vs_gru','Channel attention − BiGRU'),('attention_tca_vs_gru','Both attentions − BiGRU'),('specaugment_vs_clean_cnn','SpecAugment − clean CNN'),('noise_vs_clean_cnn','Waveform noise − clean CNN'),('mixup_vs_clean_cnn','Waveform Mixup − clean CNN')]
    effects=[(e,l) for e,l in effects if len([r for r in pairs if r['effect']==e])==3]
    if effects:
        fig,ax=plt.subplots(figsize=(10,max(3.5,.6*len(effects)+1.4)));largest=0
        for i,(effect,label) in enumerate(effects):
            rows=sorted([r for r in pairs if r['effect']==effect],key=lambda r:int(r['seed']));vals=[float(r['delta_accuracy_pp']) for r in rows]
            largest=max(largest,max(abs(v) for v in vals));mean=np.mean(vals)
            for dy,v,marker in zip([-.12,0,.12],vals,['o','s','^']):ax.scatter(v,i+dy,marker=marker,facecolors='white',edgecolors=BLUE,s=45,zorder=3)
            ax.scatter(mean,i,marker='D',color='#18222C',s=35,zorder=4)
        limit=max(3,np.ceil(largest+1));ax.set_xlim(-limit,limit);ax.axvline(0,color='#6D7780',linewidth=1)
        ax.set_yticks(range(len(effects)),[label for _,label in effects]);ax.invert_yaxis();ax.set_xlabel('Validation accuracy difference (percentage points)')
        ax.set_title('Within-seed comparisons — points = seeds, diamond = mean',loc='left',pad=15,fontweight='bold');ax.grid(axis='x',alpha=.18)
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],marker=m,linestyle='',markerfacecolor='white',markeredgecolor=BLUE,label=f'Seed {seed}') for m,seed in zip(['o','s','^'],[101,202,303])]
        handles.append(Line2D([],[],marker='D',linestyle='',color='#18222C',label='Mean'))
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,.07,1,1));save(fig,out/'paired_effects')
    histories=[]
    audited_ids={r['id'] for r in runs if r['status']=='audited'}
    for model in MODELS:
        path=a.analysis.parent/'execution/runs/B_architecture'/('B_'+model+'_s101')/'history.json'
        if path.is_file() and 'B_'+model+'_s101' in audited_ids:histories.append((model,json.loads(path.read_text())))
    if histories:
        with (out/'learning_curve_data.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['model','seed','epoch','validation_cross_entropy','validation_accuracy'])
            for model,history in histories:
                for row in history:writer.writerow([model,101,row['epoch'],row['validation']['loss'],row['validation']['accuracy']])
        fig,axes=plt.subplots(1,2,figsize=(12,4.7))
        for model,h in histories:
            label=LABELS[MODELS.index(model)];epochs=[r['epoch'] for r in h]
            axes[0].plot(epochs,[r['validation']['loss'] for r in h],color=COLORS[model],label=label,linewidth=1.7)
            axes[1].plot(epochs,[100*r['validation']['accuracy'] for r in h],color=COLORS[model],label=label,linewidth=1.7)
        for ax in axes:ax.set_xlabel('Completed epoch');ax.grid(alpha=.18)
        axes[0].set_ylabel('Validation cross-entropy');axes[1].set_ylabel('Validation accuracy (%)');axes[1].set_ylim(0,100)
        fig.suptitle('Learning curves — fixed seed 101, development validation',x=.03,ha='left',fontweight='bold')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,.17,1,.94));save(fig,out/'learning_curves')
    print('Rendered',len(list(out.glob('*.png'))),'figures to',out)

if __name__=='__main__':main()
