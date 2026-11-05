"""Two matched linear readouts on frozen PANNs: Gaussian-to-recorded-noise transfer."""
from pathlib import Path
import argparse, csv, json, os, platform, signal, sys, time, warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from esc50_sound.audio import read_audio
from esc50_sound.data import read_metadata, split_rows
from esc50_sound.robustness import perturb
from esc50_sound.prediction_errors import SourceBootstrap
from run_transfer_followup import load_panns, embeddings


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aligned_cache(path, rows):
    with np.load(path,allow_pickle=False) as a:
        lookup={str(n):i for i,n in enumerate(a['filenames'])}
        index=[lookup[r['filename']] for r in rows]
        y=np.array([int(r['target']) for r in rows])
        if not np.array_equal(a['targets'][index],y):raise ValueError('Feature-cache labels differ')
        return a['embeddings'][index].copy()


def mix(waves, noises, snr):
    """CPU float32 waveforms; exact power scaling without output clipping."""
    noises=noises-noises.mean(1,keepdim=True)
    px=waves.double().square().mean(1);pn=noises.double().square().mean(1)
    if not ((px>0).all() and (pn>0).all()):raise ValueError('Zero-power mixture input')
    scale=(px/(pn*torch.pow(torch.tensor(10.,dtype=torch.float64),snr.double()/10))).sqrt().float()
    mixed=waves+noises*scale[:,None]
    realized=10*torch.log10(px/(mixed.double()-waves.double()).square().mean(1))
    return mixed,realized.numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output
    if (out/'summary.json').exists():
        print('Completed comparison exists; reuse its results.');return
    out.mkdir(parents=True,exist_ok=True)
    feature_cache=ROOT/'results/noise_transfer_embeddings'
    feature_cache.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Fixed 30-minute compute budget reached')))
    signal.alarm(1800);started=time.perf_counter();torch.set_num_threads(2)
    protocol=json.loads((ROOT/'reports/noise_transfer_20261106/protocol.json').read_text())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    metadata=read_metadata(ROOT/'data/ESC-50');splits=split_rows(metadata)
    train=sorted([metadata[i] for i in splits['train']],key=lambda r:r['filename'])
    val=sorted([metadata[i] for i in splits['validation']],key=lambda r:r['filename'])
    if (len(train),len(val),len(set(r['src_file'] for r in val)))!=(1200,398,295):raise ValueError('Unexpected population')
    names=[r['filename'] for r in val];y=np.array([int(r['target']) for r in val]);yt=np.array([int(r['target']) for r in train])
    xc=aligned_cache(ROOT/'results/embedding_cache/train_embeddings.npz',train)
    xv=aligned_cache(ROOT/'results/embedding_cache/val_embeddings.npz',val)
    scaler=StandardScaler().fit(xc)
    with np.load(ROOT/'models/panns_linear/linear_head.npz',allow_pickle=False) as a:historical={k:a[k].copy() for k in a.files}
    scaler_difference=dict(mean_max_abs=float(np.max(abs(scaler.mean_-historical['mean']))),scale_max_abs=float(np.max(abs(scaler.scale_-historical['scale']))))
    train_waves=torch.stack([read_audio(args.assets/'ESC-50/audio'/r['filename']) for r in train])
    val_waves=torch.stack([read_audio(args.assets/'ESC-50/audio'/r['filename']) for r in val])
    print('Loaded 1200 training and 398 development targets; no test WAV reads.',flush=True)
    device=torch.device(args.device)
    model=load_panns(ROOT/'official_panns',args.assets/'Cnn14_16k_mAP=0.438.pth',device)
    rng=np.random.default_rng(protocol['training']['seed']);augment=[];exposure=[];snr_values=[]
    for k in range(2):
        snrs=rng.uniform(10,30,len(train)).astype(np.float32)
        noise=torch.from_numpy(rng.standard_normal((len(train),80000),dtype=np.float32))
        changed,realized=mix(train_waves,noise,torch.from_numpy(snrs))
        augment.append(embeddings(model,changed,device))
        snr_values.extend(realized.tolist())
        exposure.extend(dict(filename=r['filename'],target=int(r['target']),variant=k,snr_db=float(s),realized_snr_db=float(v),weight=0.25) for r,s,v in zip(train,snrs,realized))
        print('Extracted Gaussian training variant',k+1,'of 2.',flush=True)
    write_csv(out/'training_exposure.csv',exposure)
    del train_waves,noise,changed
    xaug=np.concatenate([xc,*augment]);yaug=np.tile(yt,3)
    weights=np.concatenate([np.full(len(train),.5),np.full(len(train)*2,.25)])
    heads={};fit_records=[]
    for name,x,labels,weight in [('clean_refit',xc,yt,np.ones(len(train))),('gaussian_adapted',xaug,yaug,weights)]:
        head=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,tol=1e-6)
        before=time.perf_counter()
        with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            head.fit(scaler.transform(x),labels,sample_weight=weight)
        converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        fit_records.append(dict(model=name,rows=len(x),total_weight=float(weight.sum()),iterations=int(head.n_iter_.max()),converged=converged,seconds=time.perf_counter()-before))
        write_csv(out/'head_fits.csv',fit_records)
        np.savez_compressed(out/(name+'_head.npz'),mean=scaler.mean_,scale=scaler.scale_,coef=head.coef_,intercept=head.intercept_,classes=head.classes_)
        if not converged:raise RuntimeError('Head did not converge; preserve artifacts without a scientific negative conclusion')
        heads[name]=head
        print('Completed head:',name,fit_records[-1],flush=True)
    del xaug,augment,xc
    cache_features={'clean':xv}
    metrics=[];prediction_rows=[];predictions={};background_rows=[];condition_info=[];snr_records=[]
    def evaluate(condition,features):
        transformed=scaler.transform(features)
        scores={name:head.decision_function(transformed) for name,head in heads.items()}
        scores['historical_lr']=((features-historical['mean'])/historical['scale'])@historical['coef'].T+historical['intercept']
        for name,logits in scores.items():
            pred=logits.argmax(1);predictions[name,condition]=pred
            stable=logits-logits.max(1,keepdims=True);prob=np.exp(stable);prob/=prob.sum(1,keepdims=True)
            other=logits.copy();other[np.arange(len(y)),y]=-np.inf
            margin=logits[np.arange(len(y)),y]-other.max(1)
            metrics.append(dict(model=name,condition=condition,accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=np.arange(50),average='macro',zero_division=0),correct=int((y==pred).sum()),clips=len(y)))
            prediction_rows.extend(dict(model=name,condition=condition,filename=r['filename'],src_file=r['src_file'],target=int(t),prediction=int(p),true_probability=float(q),true_margin=float(m)) for r,t,p,q,m in zip(val,y,pred,prob[np.arange(len(y)),y],margin))
        print('Evaluated',condition,flush=True)
    evaluate('clean',xv)
    changed=torch.stack([perturb(w,'snr_10',n,2026) for w,n in zip(val_waves,names)])
    gaussian=embeddings(model,changed,device);evaluate('historical_shared_gaussian10',gaussian)
    del changed
    for environment in protocol['evaluation']['backgrounds']:
        background,rate=sf.read(args.assets/(environment+'_ch01.wav'),dtype='float32')
        if rate!=16000 or background.ndim!=1 or len(background)<155*rate:raise ValueError('Unexpected DEMAND format/duration')
        for start in protocol['evaluation']['start_seconds']:
            segment=torch.from_numpy(background[start*rate:(start+5)*rate].copy())
            centered=segment-segment.mean()
            condition=f'{environment}_{start}s'
            background_rows.append(dict(environment=environment,channel='ch01',start_seconds=start,end_seconds=start+5,total_seconds=len(background)/rate,segment_rms=float(centered.square().mean().sqrt()),segment_peak=float(centered.abs().max()),source_url=protocol['sources']['backgrounds'],license='CC BY-SA 3.0'))
            noises=centered[None].expand(len(val),-1)
            changed,realized=mix(val_waves,noises,torch.full((len(val),),10.))
            snr_records.append(dict(condition=condition,min_db=float(realized.min()),max_db=float(realized.max()),mean_db=float(realized.mean()),mixture_peak=float(changed.abs().max())))
            features=embeddings(model,changed,device);evaluate(condition,features)
            condition_info.append(dict(condition=condition,environment=environment,start_seconds=start))
            # Keep fixed inference evidence; raw audio remains outside the repository.
            np.savez_compressed(feature_cache/(condition+'_embeddings.npz'),embeddings=features,filenames=np.array(names))
    del model,val_waves
    torch.cuda.empty_cache()
    bp=SourceBootstrap([r['src_file'] for r in val],resamples=4000,seed=20261106)
    effects=[];transitions=[]
    conditions=['clean','historical_shared_gaussian10']+[c['condition'] for c in condition_info]
    def effect(name,delta):
        distribution=bp.distribution(delta);lo,hi=np.percentile(100*distribution,[2.5,97.5])
        return dict(condition=name,delta_pp=100*float(np.mean(delta)),low_pp=float(lo),high_pp=float(hi))
    for condition in conditions:
        a=predictions['clean_refit',condition]==y;b=predictions['gaussian_adapted',condition]==y
        effects.append(effect(condition,b.astype(float)-a.astype(float)))
        transitions.append(dict(condition=condition,repairs=int((~a&b).sum()),new_errors=int((a&~b).sum()),both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum())))
    recorded=[c['condition'] for c in condition_info]
    diffs=np.column_stack([(predictions['gaussian_adapted',c]==y).astype(float)-(predictions['clean_refit',c]==y).astype(float) for c in recorded])
    effects.append(effect('recorded_mean_primary',diffs.mean(1)))
    for environment in protocol['evaluation']['backgrounds']:
        indices=[i for i,c in enumerate(condition_info) if c['environment']==environment]
        effects.append(effect(environment+'_mean',diffs[:,indices].mean(1)))
    aggregate=[]
    for name in ('clean_refit','gaussian_adapted','historical_lr'):
        subset=[m for m in metrics if m['model']==name and m['condition'] in recorded]
        aggregate.append(dict(model=name,condition='recorded_mean',accuracy=float(np.mean([m['accuracy'] for m in subset])),macro_f1=float(np.mean([m['macro_f1'] for m in subset])),metric_scope='Equal-weight mean over six conditions; macro-F1 computed separately within each condition'))
    for filename,values in [('metrics.csv',metrics),('recorded_means.csv',aggregate),('paired_effects.csv',effects),('transitions.csv',transitions),('predictions.csv',prediction_rows),('background_segments.csv',background_rows),('realized_snr.csv',snr_records)]:write_csv(out/filename,values)
    np.savez_compressed(feature_cache/'clean_and_gaussian_embeddings.npz',clean=xv,gaussian10=gaussian,filenames=np.array(names))
    summary=dict(status='completed',protocol=protocol,primary=next(e for e in effects if e['condition']=='recorded_mean_primary'),new_head_fits=2,encoder_fits=0,train_clips=len(train),development_clips=len(val),development_sources=bp.group_count,recorded_conditions=len(recorded),target_noise_pairs=len(val)*len(recorded),test_waveforms_read=0,fit_records=fit_records,scaler_difference_from_historical=scaler_difference,historical_clean_correct=int((predictions['historical_lr','clean']==y).sum()),historical_gaussian_correct=int((predictions['historical_lr','historical_shared_gaussian10']==y).sum()),training_realized_snr=dict(min_db=min(snr_values),max_db=max(snr_values)),compute_seconds=time.perf_counter()-started,environment=dict(python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    signal.alarm(0)
    print('COMPARISON_COMPLETE',json.dumps(summary['primary']),flush=True)


if __name__=='__main__':main()
