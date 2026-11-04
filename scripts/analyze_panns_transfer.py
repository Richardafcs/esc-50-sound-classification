"""Compare a fixed PANNs linear probe with saved development predictions."""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from esc50_sound.prediction_errors import SourceBootstrap, read_predictions, paired_correctness


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'reports/panns_20261105')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/panns_20261105')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    with (ROOT/'data/ESC-50/meta/esc50.csv').open() as f:all_rows=list(csv.DictReader(f))
    forbidden={r['src_file'] for r in all_rows if int(r['fold'])==5}
    rows=sorted([r for r in all_rows if int(r['fold'])==4 and r['src_file'] not in forbidden],key=lambda r:r['filename'])
    targets=np.array([int(r['target']) for r in rows]);names={int(r['target']):r['category'] for r in all_rows}
    new=read_predictions(args.run/'validation_predictions.csv',rows);correct=new==targets
    with (out/'reference_predictions.csv').open() as f:
        reference_records=list(csv.DictReader(f))
    reference_by_name={r['filename']:r for r in reference_records}
    if len(reference_by_name)!=len(rows) or set(reference_by_name)!={r['filename'] for r in rows}:
        raise ValueError('Reference prediction membership differs from validation')
    if any(int(reference_by_name[r['filename']]['target'])!=int(r['target']) for r in rows):
        raise ValueError('Reference labels differ from validation')
    def reference(task_id):
        return np.array([int(reference_by_name[r['filename']][task_id]) for r in rows])
    bp=SourceBootstrap([r['src_file'] for r in rows],4000,20261105)
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    summary={'panns_accuracy':accuracy_score(targets,new),'panns_macro_f1':f1_score(targets,new,labels=np.arange(50),average='macro',zero_division=0),
             'correct':int(correct.sum()),'clips':len(rows),'sources':bp.group_count,
             'scope':'development validation; fixed external-pretrained encoder and deterministic head; historical fold exposure and unresolved pretraining overlap', 'comparisons':{}}
    comparisons=[];paired=[];class_rows=[];old={};seed_preds={}
    for name,task in [('CNN','B_cnn_freq_s{}'),('BiGRU','B_crnn_freq_s{}'),('Channel Attention','B_crnn_freq_ca_s{}')]:
        preds=np.column_stack([reference(task.format(seed)) for seed in (101,202,303)])
        seed_preds[name]=preds;old[name]=preds==targets[:,None]
        effect=bp.summarize(correct[:,None].astype(float)-old[name].astype(float))
        item={'reference':name,'reference_accuracy_mean':float(old[name].mean()),'reference_accuracy_sample_sd':float(np.std(old[name].mean(axis=0),ddof=1)),**effect}
        item['uncertainty_scope']='Source sampling conditional on the single PANNs fit and three fixed reference fits; not independent confirmation or training-seed population uncertainty'
        summary['comparisons'][name]=item;comparisons.append(item)
        for i,seed in enumerate((101,202,303)):
            paired.append({'reference':name,'seed':seed,**paired_correctness(old[name][:,i],correct)})
    for cls in range(50):
        take=targets==cls
        class_rows.append({'target':cls,'category':names[cls],'n':int(take.sum()),'panns_correct':int(correct[take].sum()),'panns_accuracy':float(correct[take].mean()),
           **{name+'_accuracy_mean':float(values[take].mean()) for name,values in old.items()}})
    details=[]
    for i,r in enumerate(rows):
        details.append({**{k:r[k] for k in ('filename','src_file','target','category')},'panns_prediction':int(new[i]),'panns_predicted_category':names[int(new[i])],'panns_correct':int(correct[i]),
            **{f'{name}_correct_fits':int(v[i].sum()) for name,v in old.items()}})
    four=[]
    for task in ('B_cnn_freq_s{}','B_crnn_freq_s{}','B_crnn_freq_ca_s{}'):
        for seed in (101,202,303):four.append(reference(task.format(seed))==targets)
    for seed in (101,202,303):four.append(reference(f'C_noise_s{seed}')==targets)
    persistent=~np.column_stack(four).any(axis=1)
    summary['previous_four_family_persistent_errors']={'total':int(persistent.sum()),'panns_repaired':int((persistent&correct).sum()),'still_wrong':int((persistent&~correct).sum())}
    summary['primary_comparison']='Channel Attention';summary['practical_screen_pass']=summary['comparisons']['Channel Attention']['delta_accuracy_pp']>=3
    (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
    write_csv(out/'paired_differences.csv',comparisons);write_csv(out/'paired_errors.csv',paired)
    write_csv(out/'per_class_comparison.csv',class_rows);write_csv(out/'prediction_comparison.csv',details)
    (out/'validation_predictions.csv').write_bytes((args.run/'validation_predictions.csv').read_bytes())
    summary_path=args.run/'summary.json'
    if not summary_path.exists():summary_path=args.run/'run_summary.json'
    (out/'run_summary.json').write_bytes(summary_path.read_bytes())
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    labels=['CNN','BiGRU','Channel Attention','PANNs + linear']
    values=[100*old[n].mean() for n in labels[:3]]+[100*correct.mean()]
    sd=[100*np.std(old[n].mean(axis=0),ddof=1) for n in labels[:3]]+[0]
    fig,ax=plt.subplots(figsize=(9,4.7));bars=ax.barh(labels,values,color=['#8e9fad']*3+['#168a91'])
    ax.errorbar(values[:3],range(3),xerr=sd[:3],fmt='none',ecolor='black',capsize=3)
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Validation accuracy (%)')
    for bar,v,error in zip(bars,values,sd):ax.text(v+error+1,bar.get_y()+bar.get_height()/2,f'{v:.2f}%',va='center')
    ax.set_title('Fixed pretrained representation versus scratch-trained systems')
    fig.text(.04,.02,'398 development clips / 295 sources. Grey: three-fit mean +/- sample SD. Teal: one deterministic head.\nExternal pretraining, encoder size and frontend differ; no architecture-only causal claim.',fontsize=8)
    fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(out/'system_comparison.png',dpi=180);fig.savefig(out/'system_comparison.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for j,item in enumerate(comparisons):
        v=item['delta_accuracy_pp'];lo=item['source_bootstrap_low_pp'];hi=item['source_bootstrap_high_pp']
        ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color='#168a91')
        ax.text(hi+.4,j,f'{v:+.2f} pp [{lo:+.2f}, {hi:+.2f}]',va='center',fontsize=9)
    ax.set_yticks(range(3),[x['reference'] for x in comparisons]);ax.invert_yaxis();ax.axvline(0,color='grey',lw=1)
    ax.set_xlim(min(0,min(x['source_bootstrap_low_pp'] for x in comparisons))-1,max(x['source_bootstrap_high_pp'] for x in comparisons)+12)
    ax.set_xlabel('PANNs accuracy minus reference three-fit mean (percentage points)')
    ax.set_title('Paired differences with conditional 95% source-bootstrap intervals')
    fig.text(.03,.015,'4,000 source-cluster resamples; retain all clips, varying denominator. Fixed fitted models; not independent confirmation.',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'paired_accuracy_gains.png',dpi=180);fig.savefig(out/'paired_accuracy_gains.svg');plt.close(fig)
    cm=confusion_matrix(targets,new,labels=np.arange(50));np.savetxt(out/'confusion_counts.csv',cm,fmt='%d',delimiter=',')
    fig,ax=plt.subplots(figsize=(11,10));im=ax.imshow(cm/cm.sum(axis=1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(range(50),[names[i] for i in range(50)],rotation=90,fontsize=9);ax.set_yticks(range(50),[names[i] for i in range(50)],fontsize=9)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title('Frozen PANNs + linear head: validation confusion',fontsize=12)
    fig.colorbar(im,ax=ax,shrink=.7,label='Fraction within true class');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=180);fig.savefig(out/'confusion_matrix.svg');plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
