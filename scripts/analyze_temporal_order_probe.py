"""Decompose existing temporal-probe predictions; no model/audio inference."""
from pathlib import Path
import sys,json,csv
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from esc50_sound.robustness import validation_rows,read_predictions
from esc50_sound.prediction_errors import paired_correctness,SourceBootstrap
R=P/'reports/temporal_20261105';rows=validation_rows(P/'data/ESC-50');targets=np.array([int(r['target']) for r in rows]);sources=[r['src_file'] for r in rows]
conditions=('reverse','roll19','shuffle17','shuffle29');seeds=(101,202,303);correct={};raw={};paired=[];classwise=[]
for arch in ('cnn_freq','crnn_freq'):
 for seed in seeds:
  for condition in ('identity',*conditions):
   q=np.array(read_predictions(R/f'{arch}_s{seed}'/f'{condition}.csv',rows));raw[arch,seed,condition]=q;correct[arch,seed,condition]=q==targets
  for condition in conditions:
   a=correct[arch,seed,'identity'];b=correct[arch,seed,condition]
   paired.append(dict(model=arch,seed=seed,condition=condition,**paired_correctness(a,b),both_wrong_prediction_changes=int((~a&~b&(raw[arch,seed,'identity']!=raw[arch,seed,condition])).sum())))
 for category in sorted(set(r['category'] for r in rows)):
  ids=np.array([i for i,r in enumerate(rows) if r['category']==category]);a=np.column_stack([correct[arch,s,'identity'][ids] for s in seeds])
  for condition in conditions:
   b=np.column_stack([correct[arch,s,condition][ids] for s in seeds]);repair=~a&b;introduced=a&~b
   classwise.append(dict(model=arch,category=category,condition=condition,unique_clips=len(ids),unique_sources=len({sources[i] for i in ids}),identity_accuracy=float(a.mean()),condition_accuracy=float(b.mean()),delta_accuracy_pp=float((b.astype(float)-a.astype(float)).mean()*100),repaired_seed_clip_events=int(repair.sum()),introduced_seed_clip_events=int(introduced.sum()),unique_repaired_clips=int(repair.any(axis=1).sum()),unique_introduced_clips=int(introduced.any(axis=1).sum())))
for name,data in [('paired_errors.csv',paired),('class_error_changes.csv',classwise)]:
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
bootstrap=SourceBootstrap(sources,4000,20261105);contrasts={}
for other in ('shuffle17','shuffle29'):
 d=np.column_stack([correct['crnn_freq',s,'reverse'].astype(float)-correct['crnn_freq',s,other].astype(float) for s in seeds]);contrasts['reverse_minus_'+other]=bootstrap.summarize(d)
(R/'exploratory_contrasts.json').write_text(json.dumps(dict(scope='Post hoc comparisons prompted by the observed reversal/shuffle ordering; no family-wise correction or new confirmation.',contrasts=contrasts),indent=2)+'\n')
print('CRNN paired changes')
for q in paired:
 if q['model']=='crnn_freq':print(q)
print('posthoc',contrasts)
print('largest reverse class losses',sorted([x for x in classwise if x['model']=='crnn_freq' and x['condition']=='reverse'],key=lambda x:x['delta_accuracy_pp'])[:5])
