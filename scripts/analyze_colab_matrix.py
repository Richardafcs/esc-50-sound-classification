#!/usr/bin/env python3
"""Read-only audit of the new Colab matrix; writes derived analysis only.

Metrics and strict membership are independently computed here, rather than
using the training package's metric/split helpers. No historical predictions,
feature cache, raw waveform or test-prediction file is read.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected, label):
    require(math.isfinite(float(actual)) and
            math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10),
            f"{label}: {actual} != {expected}")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def metadata_roles(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 2000 and len({r['filename'] for r in rows}) == 2000,
            'Canonical metadata must have 2000 distinct recordings')
    require(all(n == 8 for n in Counter((int(r['fold']),int(r['target'])) for r in rows).values())
            and len({(r['fold'],r['target']) for r in rows}) == 250,
            'Canonical metadata needs eight clips per fold/category')
    test_sources = {r['src_file'] for r in rows if int(r['fold']) == 5}
    validation_sources = {r['src_file'] for r in rows if int(r['fold']) == 4}
    roles = {
        'train': [r for r in rows if int(r['fold']) in (1,2,3) and
                  r['src_file'] not in test_sources | validation_sources],
        'validation': [r for r in rows if int(r['fold']) == 4 and r['src_file'] not in test_sources],
        'test': [r for r in rows if int(r['fold']) == 5]}
    require([len(roles[k]) for k in ('train','validation','test')] == [1200,398,400],
            'Unexpected strict role counts')
    return roles


def prediction_metrics(path, expected):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = [r['filename'] for r in rows]
    expected_names = [r['filename'] for r in expected]
    require(names == expected_names and len(set(names)) == 398,
            f'{path.name}: membership/order differs from canonical 398 validation clips')
    true = np.array([int(r['target']) for r in rows],dtype=np.int64)
    pred = np.array([int(r['prediction']) for r in rows],dtype=np.int64)
    require(np.array_equal(true,[int(r['target']) for r in expected]),'Prediction target mismatch')
    require(bool(((0 <= pred) & (pred < 50)).all()),'Prediction outside 0..49')
    # Independent integer bin-count oracle: rows=true, columns=predicted.
    confusion = np.bincount(50*true+pred,minlength=2500).reshape(50,50)
    correct = np.diag(confusion).astype(float)
    denom = confusion.sum(0)+confusion.sum(1)
    f1 = np.zeros(50,dtype=float)
    np.divide(2*correct,denom,out=f1,where=denom!=0)
    return dict(accuracy=float((true==pred).mean()),macro_f1=float(f1.mean()),n=len(rows),
                correct=int(correct.sum()),confusion_matrix=confusion.tolist())


def reconcile(saved, measured, label):
    for key in ('accuracy','macro_f1','n'):
        same_number(saved[key],measured[key],f'{label}/{key}')
    if 'confusion_matrix' in saved:
        require(saved['confusion_matrix']==measured['confusion_matrix'],label+'/confusion differs')


def check_roles(record, roles):
    require(record['train_folds']==[1,2,3] and record['val_fold']==4 and record['test_fold']==5,
            'Fold protocol differs')
    for role, rows in roles.items():
        require(record['split'][role]['names']==[r['filename'] for r in rows],role+' manifest membership differs')
        require(record['split'][role]['n']==len(rows),role+' manifest denominator differs')


def baseline(folder, task, roles):
    manifest=read_json(folder/'model.json'); search=read_json(folder/'search.json')
    saved=read_json(folder/'validation.json')
    measured=prediction_metrics(folder/'validation_predictions.csv',roles['validation'])
    reconcile(saved,measured,task['id']); check_roles(manifest,roles)
    require(manifest['source_policy']=='source_strict' and manifest['test_evaluated'] is False,
            'Baseline source/test policy differs')
    require(digest(folder/'model.npz')==manifest['model_sha256'],'Baseline coefficient identity differs')
    warnings=[]
    if task['method']=='logistic':
        require([r['l2'] for r in search]==manifest['search_grid']==[.001,.01,.1],'Logistic search grid differs')
        best=min(search,key=lambda r:r['validation']['loss'])
        require(best['l2']==manifest['selected_l2'],'Wrong logistic selected L2')
        require(best['optimization']['success'] is True,'Selected logistic did not converge')
        if not all(r['optimization']['success'] for r in search):warnings.append('nonselected logistic convergence failure')
        for row in search:
            independent=prediction_metrics(folder/f"validation_predictions_l2_{row['l2']}.csv",roles['validation'])
            reconcile(row['validation'],independent,'logistic candidate')
        seconds=sum(r['optimization']['seconds'] for r in search)
        selected=best['l2']
    else:
        require([r['C'] for r in search]==manifest['search_grid']==[.01,.1,1.,10.],'SVM search grid differs')
        best=max(search,key=lambda r:(r['validation']['accuracy'],r['validation']['macro_f1'],-r['C']))
        require(best['C']==manifest['selected_C'],'Wrong SVM selected C')
        require(best['convergence_warning'] is False,'Selected SVM did not converge')
        if any(r['convergence_warning'] for r in search):warnings.append('nonselected SVM convergence warning')
        seconds=None; selected=best['C']
    reconcile(best['validation'],measured,'selected baseline')
    with np.load(folder/'model.npz',allow_pickle=False) as model:
        require(model['class_names'].tolist()==[next(r['category'] for r in roles['train'] if int(r['target'])==i) for i in range(50)],
                'Baseline class ordering differs from canonical metadata')
        require(all(np.isfinite(model[k]).all() for k in model.files if model[k].dtype.kind in 'fiu'),
                'Nonfinite baseline coefficients')
    return measured,dict(seconds=seconds,parameters=manifest.get('parameters'),selected_hyperparameter=selected,
                         convergence='selected and all candidates converged' if not warnings else 'selected converged',warnings=warnings)


def supervised(folder,task,roles,protocol):
    import torch
    record=read_json(folder/'run.json'); history=read_json(folder/'history.json')
    manifest=read_json(folder/'manifest.json'); cfg=record['config']; epochs=protocol['epochs']
    require(record['status']=='completed' and record['epochs_recorded']==epochs,'Incomplete training status/exposure')
    require(len(history)==epochs and [r['epoch'] for r in history]==list(range(1,epochs+1)),
            'Incomplete/nonconsecutive history')
    require(record.get('test_evaluated') is False,'Unexpected new test access recorded')
    require(cfg==manifest['config'],'Run/manifest config differs')
    require(cfg['model']==task['model'] and cfg['seed']==task['seed'] and cfg['epochs']==epochs,
            'Model/seed/epoch plan mismatch')
    require(cfg['batch_size']==protocol['batch_size'] and cfg['lr']==protocol['learning_rate']
            and cfg['scheduler']==protocol['scheduler'] and cfg['train_folds']=='1,2,3'
            and cfg['val_fold']==4 and cfg['test_fold']==5 and not cfg['official_folds'],
            'Training recipe differs from matrix')
    aug=task.get('augmentation','clean')
    require(bool(cfg['specaugment'])==(aug=='specaugment'),'SpecAugment config differs')
    same_number(cfg.get('noise_probability',0),.5 if aug=='noise' else 0,'noise probability')
    same_number(cfg.get('mixup_alpha',0),.4 if aug=='mixup' else 0,'mixup alpha')
    if aug=='noise':
        same_number(cfg['noise_snr_min_db'],10,'noise min SNR'); same_number(cfg['noise_snr_max_db'],30,'noise max SNR')
    check_roles(dict(manifest,train_folds=[1,2,3],val_fold=4,test_fold=5),roles)
    for i,row in enumerate(history,1):
        require(not row.get('partial_epoch') and row['training_examples']==1200
                and row['optimizer_updates']==i*math.ceil(1200/protocol['batch_size']),
                'Training exposure/optimizer updates differ')
        require(row['validation']['n']==398 and row['train_evaluation']['n']==1200,'History denominator differs')
        require(math.isfinite(row['validation']['loss']),'Nonfinite validation loss')
    best=min(history,key=lambda r:r['validation']['loss'])
    require(record['best_epoch']==best['epoch'],'Wrong selected epoch')
    saved=read_json(folder/'validation.json'); measured=prediction_metrics(folder/'validation_predictions.csv',roles['validation'])
    reconcile(saved,measured,task['id']); reconcile(best['validation'],measured,'selected epoch')
    same_number(saved['loss'],best['validation']['loss'],'selected loss')
    same_number(record['best_validation_loss'],best['validation']['loss'],'recorded selected loss')
    checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    expected_id=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(record['manifest_id']==expected_id and checkpoint['manifest_id']==expected_id
            and record['manifest_sha256']==digest(folder/'manifest.json')
            and checkpoint['manifest_sha256']==record['manifest_sha256'],
            'Existing manifest identity binding differs')
    require(checkpoint['class_names']==[next(r['category'] for r in roles['train'] if int(r['target'])==i) for i in range(50)],
            'Checkpoint class ordering differs from metadata')
    require(checkpoint['model_name']==task['model'] and checkpoint['seed']==task['seed']
            and checkpoint['epoch']==best['epoch'] and checkpoint['manifest']==manifest,
            'Checkpoint identity/selection differs')
    require(all(torch.isfinite(v).all() for v in checkpoint['model'].values()),'Nonfinite checkpoint weights')
    from esc50_sound.models import build_model
    with torch.random.fork_rng(devices=[]):
        model=build_model(task['model'])
    model.load_state_dict(checkpoint['model'],strict=True)
    require(sum(p.numel() for p in model.parameters())==record['parameters'],'Parameter count differs')
    require(tuple(checkpoint['mean'].shape)==(1,1,64,1) and tuple(checkpoint['std'].shape)==(1,1,64,1)
            and bool((checkpoint['std']>0).all()),'Checkpoint normalization differs')
    same_number(record['optimizer_updates'],epochs*math.ceil(1200/protocol['batch_size']),'Final updates')
    return measured,dict(seconds=record['seconds'],parameters=record['parameters'],best_epoch=best['epoch'],
                         optimizer_updates=record['optimizer_updates'],peak_cuda_bytes=record.get('peak_cuda_bytes'),
                         warnings=[])


def feature(folder,task,roles,protocol):
    from esc50_sound.feature_ablation import completed_run,FeatureSpec
    config,saved=completed_run(folder)
    require(FeatureSpec(**config['spec']).name==task['case'] and config['model']==task['model']
            and config['seed']==task['seed'] and config['epochs']==protocol['epochs']
            and config['batch_size']==protocol['batch_size'] and config['learning_rate']==protocol['learning_rate'],
            'Feature plan/config differs')
    measured=prediction_metrics(folder/'validation_predictions.csv',roles['validation'])
    reconcile(saved,measured,task['id'])
    return measured,dict(seconds=saved['elapsed_seconds'],parameters=config['parameter_count'],
                         best_epoch=saved['best_epoch'],warnings=[])


def stress(folder,task,roles,runs,tasks):
    report=read_json(folder/'stress.json'); require(report['split']=='validation' and report['seed']==2026,
                                                'Stress protocol differs')
    require(set(report['conditions'])==set(task['conditions']),'Stress conditions differ')
    source=tasks[task['checkpoint_task']]
    checkpoint=runs/source['group']/source['id']/'best.pt'
    require(checkpoint.is_file() and digest(checkpoint)==report['checkpoint_sha256'],
            'Stress checkpoint link differs')
    conditions=[]
    for name in task['conditions']:
        measured=prediction_metrics(folder/f'{name}_predictions.csv',roles['validation'])
        reconcile(report['conditions'][name],measured,task['id']+'/'+name)
        conditions.append(dict(task_id=task['id'],checkpoint_task=task['checkpoint_task'],condition=name,
                               accuracy=measured['accuracy'],macro_f1=measured['macro_f1'],n=398,
                               seconds=report['conditions'][name].get('seconds')))
    clean=next(r for r in conditions if r['condition']=='clean')
    original=prediction_metrics(checkpoint.parent/'validation_predictions.csv',roles['validation'])
    reconcile(clean,original,'Stress clean/source')
    return clean,dict(seconds=sum(r['seconds'] or 0 for r in conditions),warnings=[],conditions=conditions)


def csv_write(path,rows,fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    project=Path(__file__).resolve().parents[1]
    parser.add_argument('--project-root',type=Path,default=project)
    parser.add_argument('--runs',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args(argv); project=args.project_root.resolve()
    sys.path.insert(0,str(project/'src'))
    root=project / 'results/matrix_20261028'
    runs=args.runs or root/'execution/runs'; output=args.output or root/'analysis'
    require(output.resolve().is_relative_to((root/'analysis').resolve()),'Derived output must stay in matrix analysis directory')
    matrix=read_json(project/'configs/colab_study_matrix_v1.json'); tasks={t['id']:t for t in matrix['tasks']}
    require(len(tasks)==65,'Expected 65 unique matrix tasks')
    roles=metadata_roles(project/'data/ESC-50/meta/esc50.csv'); protocol=matrix['train_protocol']
    records=[]; stress_rows=[]; errors=[]
    for task in tasks.values():
        folder=(runs/task['group']/'main'/task['model']/task['case']/f"seed{task['seed']}"
                if task['kind']=='feature_ablation' else runs/task['group']/task['id'])
        row=dict(id=task['id'],group=task['group'],kind=task['kind'],model=task.get('model',task.get('method')),
                 seed=task.get('seed'),augmentation=task.get('augmentation'),case=task.get('case'),path=str(folder),
                 status='missing',accuracy=None,macro_f1=None,n=None,seconds=None)
        if folder.exists():
            try:
                if task['kind']=='baseline': measured,details=baseline(folder,task,roles)
                elif task['kind']=='supervised': measured,details=supervised(folder,task,roles,protocol)
                elif task['kind']=='feature_ablation': measured,details=feature(folder,task,roles,protocol)
                else: measured,details=stress(folder,task,roles,runs,tasks)
                stress_rows.extend(details.pop('conditions',[]))
                row.update({k:measured[k] for k in ('accuracy','macro_f1','n')});row.update(details);row['status']='audited'
            except Exception as exc:
                row.update(status='invalid',error=f'{type(exc).__name__}: {exc}');errors.append(dict(id=task['id'],error=row['error']))
        records.append(row)
    valid={r['id']:r for r in records if r['status']=='audited'}
    grouped=defaultdict(list)
    for row in valid.values():
        if row['kind'] in ('supervised','feature_ablation'):
            grouped[(row['group'],row['model'],row.get('augmentation') or '',row.get('case') or '')].append(row)
    group_rows=[]
    for (group,model,aug,case),items in sorted(grouped.items()):
        group_rows.append(dict(group=group,model=model,augmentation=aug,case=case,seeds=','.join(str(r['seed']) for r in items),
                               n_seeds=len(items),complete_three_seeds={r['seed'] for r in items}==set(protocol['seeds']),
                               accuracy_mean=np.mean([r['accuracy'] for r in items]).item(),
                               accuracy_sample_sd=np.std([r['accuracy'] for r in items],ddof=1).item() if len(items)>1 else None,
                               macro_f1_mean=np.mean([r['macro_f1'] for r in items]).item(),
                               macro_f1_sample_sd=np.std([r['macro_f1'] for r in items],ddof=1).item() if len(items)>1 else None))
    pairs=[]
    def pair(left,right,effect,seed):
        if left in valid and right in valid:
            pairs.append(dict(effect=effect,seed=seed,left=left,right=right,
                              delta_accuracy_pp=100*(valid[right]['accuracy']-valid[left]['accuracy']),
                              delta_macro_f1_pp=100*(valid[right]['macro_f1']-valid[left]['macro_f1'])))
    for seed in protocol['seeds']:
        pair(f'B_cnn_freq_s{seed}',f'B_crnn_freq_s{seed}','gru_vs_cnn',seed)
        interaction_ids=[f'B_crnn_freq{suffix}_s{seed}' for suffix in ('','_ta','_ca','_tca')]
        if all(name in valid for name in interaction_ids):
            base,ta,ca,tca=[valid[name] for name in interaction_ids]
            pairs.append(dict(effect='attention_interaction',seed=seed,
                              left='B_crnn_freq_ta+B_crnn_freq_ca-B_crnn_freq',right=f'B_crnn_freq_tca_s{seed}',
                              delta_accuracy_pp=100*(tca['accuracy']-ta['accuracy']-ca['accuracy']+base['accuracy']),
                              delta_macro_f1_pp=100*(tca['macro_f1']-ta['macro_f1']-ca['macro_f1']+base['macro_f1'])))
        for attention in ('ta','ca','tca'):
            pair(f'B_crnn_freq_s{seed}',f'B_crnn_freq_{attention}_s{seed}','attention_'+attention+'_vs_gru',seed)
        for aug in ('specaugment','noise','mixup'):
            pair(f'B_cnn_freq_s{seed}',f'C_{aug}_s{seed}',aug+'_vs_clean_cnn',seed)
        for feat in ('logmel64','logmel32','mfcc40','loglinear64'):
            pair(f'D_sr16_fmax8_{feat}_s{seed}',f'D_sr32_fmax8_{feat}_s{seed}','sample_rate_matched_bandwidth',seed)
            pair(f'D_sr32_fmax8_{feat}_s{seed}',f'D_sr32_fmax16_{feat}_s{seed}','added_high_frequency_bandwidth',seed)
        for sr,fmax in ((16,8),(32,8),(32,16)):
            for feat in ('logmel32','mfcc40','loglinear64'):
                pair(f'D_sr{sr}_fmax{fmax}_logmel64_s{seed}',f'D_sr{sr}_fmax{fmax}_{feat}_s{seed}',
                     'representation_vs_logmel64',seed)
    paired_groups=defaultdict(list)
    for row in pairs:
        paired_groups[(row['effect'],row['left'].rsplit('_s',1)[0],row['right'].rsplit('_s',1)[0])].append(row)
    paired_effects=[]
    for (effect,left,right),items in sorted(paired_groups.items()):
        paired_effects.append(dict(effect=effect,left=left,right=right,n_pairs=len(items),
                                  complete_three_seeds={r['seed'] for r in items}==set(protocol['seeds']),
                                  mean_delta_accuracy_pp=float(np.mean([r['delta_accuracy_pp'] for r in items])),
                                  sample_sd_delta_accuracy_pp=float(np.std([r['delta_accuracy_pp'] for r in items],ddof=1)) if len(items)>1 else None,
                                  mean_delta_macro_f1_pp=float(np.mean([r['delta_macro_f1_pp'] for r in items]))))
    counts=Counter(r['status'] for r in records)
    report=dict(generated_utc=datetime.now(timezone.utc).isoformat(),scope='new development validation only; no historical/test prediction reads',
                metric_definition='Independent numpy bin-count confusion; 50-class macro-F1, undefined class F1 zero',
                role_sizes={k:len(v) for k,v in roles.items()},task_count=65,status_counts=dict(counts),
                complete=counts['audited']==65 and not errors,errors=errors,
                missing=[r['id'] for r in records if r['status']=='missing'],
                runtime=dict(neural_training_seconds=sum(r.get('seconds') or 0 for r in valid.values()
                                                         if r['kind'] in ('supervised','feature_ablation')),
                             note='Sum of recorded training-loop wall times, including in-loop validation/checkpoint I/O; excludes process start, cache/WAV loading and preparation, initialization, pilots, exports, idle and SVM fitting; not GPU-kernel time'),
                groups=group_rows,paired_contrasts=pairs,paired_effects=paired_effects,stress=stress_rows,
                limitations=['Artifact and arithmetic verification; no independent retraining or waveform replay',
                             'Shared development validation and prior historical test exposure; no fresh confirmation',
                             'Seed SD is training variability, not dataset confidence interval',
                             'D uses pooled CNN; compare within feature track',
                             'Synthetic stress is one seed/checkpoint per model, not field robustness'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    csv_write(output/'runs.csv',records,['id','group','kind','status','model','seed','augmentation','case','accuracy','macro_f1','n',
                                       'best_epoch','parameters','seconds','optimizer_updates','peak_cuda_bytes','selected_hyperparameter','convergence','error','path'])
    csv_write(output/'groups.csv',group_rows,['group','model','augmentation','case','seeds','n_seeds','complete_three_seeds',
                                           'accuracy_mean','accuracy_sample_sd','macro_f1_mean','macro_f1_sample_sd'])
    csv_write(output/'paired_contrasts.csv',pairs,['effect','seed','left','right','delta_accuracy_pp','delta_macro_f1_pp'])
    csv_write(output/'paired_effects.csv',paired_effects,['effect','left','right','n_pairs','complete_three_seeds',
                                                       'mean_delta_accuracy_pp','sample_sd_delta_accuracy_pp','mean_delta_macro_f1_pp'])
    csv_write(output/'stress.csv',stress_rows,['task_id','checkpoint_task','condition','accuracy','macro_f1','n','seconds'])
    print(json.dumps(dict(status_counts=dict(counts),complete=report['complete'],errors=errors,output=str(output)),indent=2))
    return 1 if errors or (args.require_complete and not report['complete']) else 0


if __name__=='__main__':
    raise SystemExit(main())
