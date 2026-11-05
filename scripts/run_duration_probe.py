"""A fixed within-recording duration-policy probe for the frozen PANNs system."""
from pathlib import Path
import argparse, csv, json, math, platform, signal, sys, time
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, f1_score
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_transfer_followup import load_panns, write_csv

def predict_waveforms(model, head, waves, policy, device="cuda", batch_size=8):
    import numpy as np
    import torch
    import torch.nn.functional as F

    if policy not in ("right_pad", "left_pad", "native"):
        raise ValueError(f"Unknown policy: {policy}")
    if not waves:
        raise ValueError("waves must be nonempty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    lengths = []
    for i, wave in enumerate(waves):
        if not isinstance(wave, torch.Tensor):
            raise TypeError(f"waves[{i}] must be a torch.Tensor")
        if wave.ndim != 1 or wave.numel() == 0:
            raise ValueError(f"waves[{i}] must be nonempty and 1D")
        if wave.device.type != "cpu" or wave.dtype != torch.float32:
            raise ValueError(f"waves[{i}] must be CPU float32")
        if wave.numel() > 80000:
            raise ValueError(f"waves[{i}] exceeds 80000 samples")
        if not torch.isfinite(wave).all().item():
            raise ValueError(f"waves[{i}] contains nonfinite values")
        if policy == "native" and wave.numel() < 4960:
            raise ValueError(f"waves[{i}] is too short for native inference")
        lengths.append(wave.numel())

    mean = np.asarray(head["mean"], dtype=np.float64).reshape(1, 2048)
    scale = np.asarray(head["scale"], dtype=np.float64).reshape(1, 2048)
    coef = np.asarray(head["coef"], dtype=np.float64).reshape(50, 2048)
    intercept = np.asarray(head["intercept"], dtype=np.float64).reshape(1, 50)
    classes = np.asarray(head["classes"], dtype=np.int64).reshape(50)

    if not np.array_equal(np.sort(classes), np.arange(50)):
        raise ValueError("head classes must be a permutation of ESC IDs 0..49")

    dev = torch.device(device)
    embeddings = []

    model.eval()
    with torch.inference_mode():
        step = 1 if policy == "native" else batch_size

        for start in range(0, len(waves), step):
            chunk = waves[start:start + step]

            if policy == "native":
                batch = chunk[0].unsqueeze(0)
            else:
                padded = []
                for wave in chunk:
                    pad = 80000 - wave.numel()
                    if policy == "right_pad":
                        padded.append(F.pad(wave, (0, pad)))
                    else:
                        padded.append(F.pad(wave, (pad, 0)))
                batch = torch.stack(padded)

            # No resampling, normalization, augmentation, or extra padding.
            embedding = model(batch.to(dev))["embedding"]
            embeddings.append(
                embedding.detach().cpu().numpy().copy()
            )

    embeddings = np.concatenate(embeddings, axis=0)

    # Apply the original frozen scaler and logistic regression head.
    standardized = (embeddings.astype(np.float64) - mean) / scale
    scores = standardized @ coef.T + intercept

    # Numerically stable softmax over all 50 competing classes.
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    head_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Reorder probability columns into canonical ESC-50 class ID order.
    probabilities = np.empty_like(head_probs)
    probabilities[:, classes] = head_probs

    return {
        "probabilities": probabilities,
        "embeddings": embeddings,
        "input_samples": np.asarray(lengths, dtype=np.int64),
    }

def decode(path):
    audio,rate=sf.read(path,dtype='float32',always_2d=True)
    audio=audio.mean(1)
    if rate!=16000:
        divisor=math.gcd(rate,16000)
        audio=resample_poly(audio,16000//divisor,rate//divisor).astype(np.float32)
    return torch.from_numpy(audio.copy())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output
    if (out/'summary.json').exists():
        print('Completed fixed duration probe exists; reading saved results.');return
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    def limit(*_):raise TimeoutError('30-minute inference budget exhausted')
    signal.signal(signal.SIGALRM,limit);signal.alarm(1800);torch.set_num_threads(2)
    data=ROOT/'reports/duration_probe_20261105'
    rows=list(csv.DictReader((data/'short_recordings.csv').open()))
    controls=list(csv.DictReader((data/'full_window_controls.csv').open()))
    waves=[decode(args.assets/'audio'/r['filename']) for r in rows]
    for row,wave in zip(rows,waves):
        if len(wave)!=int(row['samples']):raise ValueError('Decoded length differs from frozen manifest')
    eligible=np.array([len(w)>=4960 for w in waves]);selected=np.flatnonzero(eligible)
    y=np.array([int(r['target']) for r in rows]);targets=sorted(set(y.tolist()))
    model=load_panns(args.assets/'audioset_tagging_cnn',args.assets/'Cnn14_16k_mAP=0.438.pth',args.device)
    with np.load(ROOT/'models/panns_linear/linear_head.npz',allow_pickle=False) as z:head={k:z[k].copy() for k in z.files}
    right=predict_waveforms(model,head,waves,'right_pad',args.device)
    with np.load(data/'baseline_panns_probabilities.npz',allow_pickle=False) as z:
        order={name:i for i,name in enumerate(z['filenames'])}
        reference=z['probabilities'][[order[r['filename']] for r in rows]]
    replay=dict(recordings=len(rows),prediction_changes=int((right['probabilities'].argmax(1)!=reference.argmax(1)).sum()),max_probability_difference=float(np.abs(right['probabilities']-reference).max()),tolerance=.001)
    (out/'baseline_replay.json').write_text(json.dumps(replay,indent=2)+'\n')
    print('Baseline replay:',replay,flush=True)
    if replay['prediction_changes'] or replay['max_probability_difference']>.001:raise RuntimeError('Baseline replay differs; stopping before interventions')
    control_waves=[decode(args.assets/'audio'/r['filename'])[:80000] for r in controls]
    fixed=predict_waveforms(model,head,control_waves,'right_pad',args.device)
    native_control=predict_waveforms(model,head,control_waves,'native',args.device)
    control_rows=[dict(filename=r['filename'],fixed_prediction=int(a.argmax()),native_prediction=int(b.argmax()),max_probability_difference=float(np.abs(a-b).max())) for r,a,b in zip(controls,fixed['probabilities'],native_control['probabilities'])]
    write_csv(out/'negative_controls.csv',control_rows)
    negative=dict(recordings=len(controls),prediction_changes=sum(r['fixed_prediction']!=r['native_prediction'] for r in control_rows),max_probability_difference=max(r['max_probability_difference'] for r in control_rows),tolerance=.001)
    if negative['prediction_changes'] or negative['max_probability_difference']>.001:raise RuntimeError('Identical-input control differs; stopping before interventions')
    native=predict_waveforms(model,head,[waves[i] for i in selected],'native',args.device)
    left=predict_waveforms(model,head,waves,'left_pad',args.device)
    conditions={'right_pad':(np.arange(len(rows)),right),'native':(selected,native),'left_pad':(np.arange(len(rows)),left)}
    predictions=[];metrics=[];per_class=[]
    for policy,(indices,result) in conditions.items():
        prob=result['probabilities'];pred=prob.argmax(1);truth=y[indices]
        for group,mask in [('all_short',np.ones(len(indices),dtype=bool)),('native_eligible',eligible[indices])]:
            if policy=='native' and group=='all_short':continue
            t=truth[mask];v=pred[mask]
            metrics.append(dict(policy=policy,population=group,recordings=len(t),correct=int((t==v).sum()),accuracy=float(accuracy_score(t,v)),macro_recall=float(np.mean([(v[t==c]==c).mean() for c in targets])),macro_f1=float(f1_score(t,v,labels=targets,average='macro',zero_division=0)),outside_six_rate=float((~np.isin(v,targets)).mean())))
        for local,index in enumerate(indices):
            row=rows[index]
            predictions.append(dict(policy=policy,filename=row['filename'],src_file=row['src_file'],category=row['category'],target=int(y[index]),prediction=int(pred[local]),correct=bool(pred[local]==y[index]),native_eligible=bool(eligible[index]),valid_samples=len(waves[index]),model_samples=len(waves[index]) if policy=='native' else 80000,true_class_probability=float(prob[local,y[index]])))
        for c in targets:
            mask=(truth==c)&eligible[indices]
            per_class.append(dict(policy=policy,target=c,category=next(r['category'] for r in rows if int(r['target'])==c),recordings=int(mask.sum()),recall=float((pred[mask]==c).mean())))
        np.savez_compressed(out/f'{policy}_probabilities.npz',probabilities=prob,filenames=np.array([rows[i]['filename'] for i in indices]),targets=truth)
    write_csv(out/'metrics.csv',metrics);write_csv(out/'predictions.csv',predictions);write_csv(out/'per_class.csv',per_class)
    rng=np.random.default_rng(20261107);effects=[];transitions=[]
    for policy,indices in [('native',selected),('left_pad',selected),('left_pad',np.arange(len(rows)))]:
        population='native_eligible' if len(indices)==len(selected) else 'all_short'
        base=right['probabilities'][indices];other=native['probabilities'] if policy=='native' else left['probabilities'][indices]
        a=base.argmax(1);b=other.argmax(1);truth=y[indices];difference=(b==truth).astype(float)-(a==truth).astype(float)
        sample=rng.integers(0,len(indices),size=(4000,len(indices)));low,high=np.percentile(100*difference[sample].mean(1),[2.5,97.5])
        effects.append(dict(comparison=policy+'_minus_right_pad',population=population,recordings=len(indices),delta_pp=100*float(difference.mean()),low_pp=float(low),high_pp=float(high)))
        transitions.append(dict(policy=policy,population=population,recordings=len(indices),prediction_flips=int((a!=b).sum()),both_correct=int(((a==truth)&(b==truth)).sum()),repairs=int(((a!=truth)&(b==truth)).sum()),new_errors=int(((a==truth)&(b!=truth)).sum()),both_wrong=int(((a!=truth)&(b!=truth)).sum()),mean_true_class_probability_change=float((other[np.arange(len(indices)),truth]-base[np.arange(len(indices)),truth]).mean())))
    write_csv(out/'paired_effects.csv',effects);write_csv(out/'transitions.csv',transitions)
    excluded=[dict(filename=r['filename'],category=r['category'],samples=len(w),reason='Below4960-sample architecture minimum; retained in original external benchmark') for r,w in zip(rows,waves) if len(w)<4960]
    write_csv(out/'native_exclusions.csv',excluded)
    summary=dict(status='completed',short_recordings=len(rows),primary_recordings=len(selected),native_excluded=len(excluded),conditions=list(conditions),fits=0,baseline_replay=replay,negative_control=negative,primary_effect=effects[0],metrics=metrics,transitions=transitions,inference_seconds=time.perf_counter()-started,environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');signal.alarm(0)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
