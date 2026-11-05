"""Matched frozen-system robustness and a fixed-budget PANNs head ablation."""
from pathlib import Path
import argparse, csv, json, sys, time, signal, platform
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from esc50_sound.audio import read_audio, log_mel
from esc50_sound.data import read_metadata, split_rows
from esc50_sound.models import build_model
from esc50_sound.evaluation import compute_metrics
from esc50_sound.prediction_errors import SourceBootstrap
from esc50_sound.robustness import perturb, PROTOCOL
from esc50_sound.transfer_heads import train_matched_heads


def write_csv(path, records):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def load_panns(repository, checkpoint, device):
    sys.path.insert(0,str(Path(repository)/'pytorch'))
    from models import Cnn14_16k
    model=Cnn14_16k(sample_rate=16000,window_size=512,hop_size=160,mel_bins=64,fmin=50,fmax=8000,classes_num=527)
    from numpy.core.multiarray import _reconstruct
    legacy_types=[(_reconstruct,'numpy.core.multiarray._reconstruct'),np.ndarray,np.dtype,
                  *[type(np.dtype(t)) for t in (np.float32,np.float64,np.uint32,np.int64)]]
    with torch.serialization.safe_globals(legacy_types):
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(weights['model'],strict=True)
    model.requires_grad_(False);return model.to(device).eval()


def embeddings(model, waves, device, batch_size=8):
    values=[]
    with torch.inference_mode():
        for begin in range(0,len(waves),batch_size):
            values.append(model(waves[begin:begin+batch_size].to(device))['embedding'].cpu().numpy())
    return np.concatenate(values)


def effect(bootstrap, values):
    """Clip-weighted source resampling, also supporting loss differences in [-2,2]."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim==2:values=values.mean(1)
    totals=np.bincount(bootstrap.inverse,weights=values,minlength=bootstrap.group_count)
    distribution=(bootstrap.multiplicities@totals)/bootstrap.denominators
    low,high=np.percentile(100*distribution,[2.5,97.5])
    return dict(delta_pp=100*float(values.mean()),low_pp=float(low),high_pp=float(high))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output
    if (out/'summary.json').exists():
        print('Completed fixed experiment already exists; reading its saved outputs is sufficient.');return
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    started=time.perf_counter();device=torch.device(args.device)
    metadata=read_metadata(ROOT/'data/ESC-50');roles=split_rows(metadata)
    rows=sorted([metadata[i] for i in roles['validation']],key=lambda r:r['filename'])
    train_rows=sorted([metadata[i] for i in roles['train']],key=lambda r:r['filename'])
    y=np.array([int(r['target']) for r in rows]);sources=[r['src_file'] for r in rows]
    names=[r['filename'] for r in rows]
    print(f'Validation: {len(rows)} clips from {len(set(sources))} sources. No test WAVs evaluated.',flush=True)
    waves=torch.stack([read_audio(args.assets/'ESC-50/audio'/r['filename']) for r in rows])
    panns=load_panns(args.assets/'audioset_tagging_cnn',args.assets/'Cnn14_16k_mAP=0.438.pth',device)
    with np.load(ROOT/'models/panns_linear/linear_head.npz',allow_pickle=False) as archive:
        head={key:archive[key].copy() for key in archive.files}
    noise_models=[]
    for seed in (101,202,303):
        saved=torch.load(ROOT/f'models/noise_cnn/seed{seed}.pt',map_location='cpu',weights_only=True)
        model=build_model(saved['model_name']).to(device).eval();model.requires_grad_(False)
        model.load_state_dict(saved['model'],strict=True)
        noise_models.append((seed,model,saved['mean'],saved['std']))
    metrics=[];predictions={};feature_sets={};snr_summary={}
    def record(model,seed,condition,pred,parameters):
        key=(model,seed,condition);predictions[key]=np.asarray(pred)
        result=compute_metrics(y,pred)
        metrics.append(dict(model=model,seed=seed,condition=condition,accuracy=result['accuracy'],macro_f1=result['macro_f1'],correct=int((pred==y).sum()),clips=len(y),parameters=parameters))
        write_csv(out/f'{model}_s{seed}_{condition}.csv',[dict(filename=name,target=int(target),prediction=int(value)) for name,target,value in zip(names,y,pred)])
    for condition in ('clean','snr_10'):
        changed=waves if condition=='clean' else torch.stack([perturb(wave,condition,name,2026) for wave,name in zip(waves,names)])
        if condition!='clean':
            signal_power=waves.double().square().mean(1)
            noise_power=(changed.double()-waves.double()).square().mean(1)
            measured=10*torch.log10(signal_power/noise_power)
            snr_summary=dict(min_db=float(measured.min()),max_db=float(measured.max()),mean_db=float(measured.mean()))
        features=embeddings(panns,changed,device);feature_sets[condition]=features
        z=(features-head['mean'])/head['scale'];scores=z@head['coef'].T+head['intercept']
        record('panns_lr',0,condition,head['classes'][scores.argmax(1)],102450)
        cnn_features=torch.stack([log_mel(wave) for wave in changed])
        for seed,model,mean,std in noise_models:
            values=[]
            with torch.inference_mode():
                for begin in range(0,len(rows),32):
                    x=((cnn_features[begin:begin+32]-mean)/std).to(device)
                    values.extend(model(x).argmax(1).cpu().tolist())
            record('noise_cnn',seed,condition,np.array(values),49058)
        print(f'Completed matched {condition} inference for PANNs and all three noiseCNNs.',flush=True)
    # Existing clean training embeddings are reused; a fresh checkout can extract them once.
    cache=ROOT/'results/embedding_cache/train_embeddings.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as archive:
            position={name:i for i,name in enumerate(archive['filenames'].tolist())}
            indices=[position[r['filename']] for r in train_rows]
            x_train=archive['embeddings'][indices].copy();y_train=archive['targets'][indices].copy()
        embedding_origin='Existing clean training embeddings, aligned by filename'
    else:
        train_waves=torch.stack([read_audio(args.assets/'ESC-50/audio'/r['filename']) for r in train_rows])
        x_train=embeddings(panns,train_waves,device);y_train=np.array([int(r['target']) for r in train_rows])
        embedding_origin='New clean training embedding extraction with the same frozen encoder'
    if not np.array_equal(y_train,[int(r['target']) for r in train_rows]):
        raise ValueError('Training embedding labels differ from source-strict metadata')
    del panns,noise_models,waves
    if device.type=='cuda':torch.cuda.empty_cache()
    trained,head_rows=train_matched_heads(x_train,y_train,feature_sets['clean'],feature_sets['snr_10'],y,out/'heads',str(device))
    for (model,seed,condition),pred in trained.items():
        record(model,seed,'snr_10' if condition=='noise10db' else condition,pred,102901)
    write_csv(out/'metrics.csv',metrics)
    # Bootstrap differences keep clips, fitted seeds and perturbations paired.
    bp=SourceBootstrap(sources,4000,20261105)
    def correct(model,condition):
        seeds=[0] if model=='panns_lr' else [101,202,303]
        return np.column_stack([predictions[model,s,condition]==y for s in seeds]).astype(float)
    effects=[]
    for condition in ('clean','snr_10'):
        effects.append(dict(comparison='panns_lr_minus_noise_cnn',condition=condition,**effect(bp,correct('panns_lr',condition)-correct('noise_cnn',condition))))
        effects.append(dict(comparison='mlp_relu_minus_factorized_linear',condition=condition,**effect(bp,correct('mlp_relu',condition)-correct('factorized_linear',condition))))
        for model in ['factorized_linear','mlp_relu']:
            effects.append(dict(comparison=f'{model}_minus_panns_lr',condition=condition,**effect(bp,correct(model,condition)-correct('panns_lr',condition))))
    losses=[]
    for model in ['panns_lr','noise_cnn','factorized_linear','mlp_relu']:
        losses.append(dict(model=model,**effect(bp,correct(model,'clean')-correct(model,'snr_10'))))
    effects.append(dict(comparison='panns_loss_minus_noise_cnn_loss',condition='clean_to_snr10',**effect(bp,(correct('panns_lr','clean')-correct('panns_lr','snr_10'))-(correct('noise_cnn','clean')-correct('noise_cnn','snr_10')))))
    write_csv(out/'paired_effects.csv',effects);write_csv(out/'accuracy_losses.csv',losses)
    # Runtime differences are described, not tuned away to recover earlier scores.
    reference={}
    with (ROOT/'reports/panns_20261105/reference_predictions.csv').open() as stream:
        ref={r['filename']:r for r in csv.DictReader(stream)}
    for seed in (101,202,303):
        old=np.array([int(ref[name][f'C_noise_s{seed}']) for name in names])
        reference[f'noise_cnn_s{seed}']=int((old!=predictions['noise_cnn',seed,'clean']).sum())
    with (ROOT/'reports/colab_notebook_20261105/validation_predictions.csv').open() as stream:
        ref={r['filename']:int(r['prediction']) for r in csv.DictReader(stream)}
    reference['panns_lr']=int((np.array([ref[name] for name in names])!=predictions['panns_lr',0,'clean']).sum())
    per_class=[]
    for model in ['panns_lr','noise_cnn','factorized_linear','mlp_relu']:
        for condition in ['clean','snr_10']:
            for target in range(50):
                take=y==target
                per_class.append(dict(model=model,condition=condition,target=target,category=next(r['category'] for r in rows if int(r['target'])==target),n=int(take.sum()),mean_recall=float(correct(model,condition)[take].mean())))
    write_csv(out/'per_class.csv',per_class)
    groups=[]
    for model in ['panns_lr','noise_cnn','factorized_linear','mlp_relu']:
        for condition in ['clean','snr_10']:
            selected=[r for r in metrics if r['model']==model and r['condition']==condition]
            groups.append(dict(model=model,condition=condition,fits=len(selected),accuracy_percent=100*np.mean([r['accuracy'] for r in selected]),accuracy_sd_pp=100*np.std([r['accuracy'] for r in selected],ddof=1) if len(selected)>1 else None,macro_f1_percent=100*np.mean([r['macro_f1'] for r in selected]),macro_f1_sd_pp=100*np.std([r['macro_f1'] for r in selected],ddof=1) if len(selected)>1 else None))
    write_csv(out/'groups.csv',groups)
    if device.type=='cuda':torch.cuda.synchronize(device)
    summary=dict(status='completed',protocol=PROTOCOL,clips=len(rows),sources=len(set(sources)),test_waveforms_read=0,head_fits=6,head_epochs=100,head_selection='fixed final epoch; no validation selection',training_embedding_origin=embedding_origin,realized_snr=snr_summary,clean_prediction_changes_from_previous=reference,total_seconds=time.perf_counter()-started,primary_effect=effects[4],effects=effects,losses=losses,environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None),uncertainty='Conditional on these fixed fits, recording sources, historical development selection and one Gaussian draw per clip')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status='completed',groups=groups,primary=effects[4],seconds=summary['total_seconds']),indent=2),flush=True)


if __name__=='__main__':
    def deadline(signum,frame):raise TimeoutError('Fixed 25-minute experiment budget exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,25*60)
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
