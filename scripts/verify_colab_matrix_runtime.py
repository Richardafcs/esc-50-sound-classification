#!/usr/bin/env python3
"""Colab-only extra validation: replay validation WAVs, epoch resume, warm timing.

Run after the 65-task notebook. Uses no held-out audio and writes a separate
verification directory. This checks execution consistency, not independent
algorithm correctness or out-of-domain generalization.
"""
from pathlib import Path
import argparse,csv,json,time,subprocess,sys,os,statistics
import numpy as np
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--storage',type=Path,default=Path('/content/esc50_results'));a=ap.parse_args()
    p=a.project.resolve();sys.path.insert(0,str(p/'src'))
    from esc50_sound.audio import read_audio,log_mel
    from esc50_sound.models import build_model
    from esc50_sound.feature_ablation import by_name,resample_official_waveform,extract_feature
    from esc50_sound.data import read_metadata,split_rows
    from esc50_sound.baseline import summarize_features
    from esc50_sound.cepstral import cepstral_statistics
    torch.set_num_threads(2)
    out=a.storage/'runtime_verification';out.mkdir(exist_ok=False)
    matrix=json.loads((p/'configs/colab_study_matrix_v1.json').read_text())
    rows=read_metadata(p/'data/ESC-50');ids=split_rows(rows,(1,2,3),4,5,strict=True)['validation'];rows=[rows[i] for i in ids]
    names=[r['filename'] for r in rows];root=p/'data/ESC-50/audio';device=torch.device('cuda');replays=[]
    waveforms=[read_audio(root/name) for name in names]
    base=np.stack([log_mel(w).numpy() for w in waveforms]).astype(np.float32)
    def saved(folder,filename='validation_predictions.csv'):
        records=list(csv.DictReader((folder/filename).open()))
        assert [r['filename'] for r in records]==names
        return [int(r['prediction']) for r in records]
    def forward(checkpoint,features):
        model=build_model(checkpoint['model_name']).to(device).eval();model.load_state_dict(checkpoint['model'])
        x=((features-checkpoint['mean'].numpy())/checkpoint['std'].numpy()).astype(np.float32)
        preds=[]
        with torch.inference_mode():
            for begin in range(0,len(x),32):preds.extend(model(torch.from_numpy(x[begin:begin+32]).to(device)).argmax(1).cpu().tolist())
        return preds
    for t in matrix['tasks']:
        if t['kind'] not in ('baseline','supervised'):continue
        folder=a.storage / 'runs'/t['group']/t['id']
        if t['kind']=='supervised':
            ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);pred=forward(ck,base)
        else:
            with np.load(folder/'model.npz',allow_pickle=False) as b:
                if t['method']=='logistic':
                    x=(summarize_features(base)-b['mean'])/b['std'];pred=(x@b['weights']+b['bias']).argmax(1).tolist()
                else:
                    x=(cepstral_statistics(base)-b['mean'])/b['std'];pred=b['classes'][(x@b['coefficients'].T+b['intercept']).argmax(1)].tolist()
        reference=saved(folder);changed=sum(x!=y for x,y in zip(pred,reference));assert changed==0,(t['id'],changed)
        replays.append(dict(task=t['id'],clips=len(rows),changed_predictions=changed,input='original validation WAV through project frontend'))
        print('WAV_REPLAY',t['id'],'398/398',flush=True)
    for case in dict.fromkeys(t['case'] for t in matrix['tasks'] if t['kind']=='feature_ablation'):
        spec=by_name(case)
        features=np.stack([extract_feature(resample_official_waveform(root/name,spec),spec).numpy() for name in names]).astype(np.float32)
        for t in matrix['tasks']:
            if t.get('case')!=case:continue
            folder=a.storage / 'runs'/t['group']/'main'/t['model']/case/('seed'+str(t['seed']))
            ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            pred=forward(ck,features);reference=saved(folder);changed=sum(x!=y for x,y in zip(pred,reference));assert changed==0,(t['id'],changed)
            replays.append(dict(task=t['id'],clips=len(rows),changed_predictions=changed,input='original validation WAV through case-specific frontend'))
            print('WAV_REPLAY',t['id'],'398/398',flush=True)
    (out/'waveform_replay.json').write_text(json.dumps(dict(scope='62 fitted models: original validation WAV replay; no test WAVs',device=str(device),runs=replays),indent=2))
    # One real validation example illustrates the input representation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,5.6),layout='constrained')
    time_axis=np.arange(80000)/16000
    axes[0].plot(time_axis,waveforms[0].numpy(),color='#255F91',linewidth=.6)
    axes[0].set(xlim=(0,5),ylabel='Normalized PCM amplitude',title=f'Validation example: {names[0]} — {rows[0]["category"].replace("_"," ")}')
    image=axes[1].imshow(base[0,0],origin='lower',aspect='auto',extent=(-.008,5.0,-.5,63.5),cmap='magma')
    axes[1].set(xlim=(0,5),xlabel='Time (seconds)',ylabel='Mel-band index',title='64-band natural-log Mel power, before train-fitted normalization')
    fig.colorbar(image,ax=axes[1],label='ln(Mel power), relative units')
    fig.savefig(out/'frontend_example.png',dpi=180)
    fig.savefig(out/'frontend_example.svg')
    plt.close(fig)
    # Real ESC-50 data, two complete epochs: compare paused/resumed run with the
    # already-finished uninterrupted CNN pilot. This is an operational check.
    env=dict(os.environ,PYTHONPATH=str(p/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    resume=out/'resume_probe'
    commands=[[sys.executable,'-m','esc50_sound','train','--model','cnn_freq','--seed','101','--epochs','2','--batch-size','32','--lr','.001','--scheduler','cosine','--threads','2','--device','cuda','--cache',str(p/'data/esc50_features.npz'),'--raw-audio-root',str(p/'data/ESC-50'),'--max-seconds','600','--pause-after-epochs','1','--output',str(resume)],
              [sys.executable,'-m','esc50_sound','resume-train','--output',str(resume),'--max-seconds','600']]
    for i,cmd in enumerate(commands):
        with (out/f'resume_step{i+1}.log').open('w') as log:subprocess.run(cmd,cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    first=json.loads((resume/'run.json').read_text());reference=a.storage / 'runs/B_architecture/B_cnn_freq_s101__pilot2ep'
    if not (reference/'best.pt').is_file():
        # A fresh verification VM may not retain the earlier pilot. Rebuild only
        # this two-epoch operational reference; formal matrix runs stay intact.
        reference=out/'uninterrupted_reference'
        command=list(commands[0]);i=command.index('--pause-after-epochs');del command[i:i+2]
        command[command.index('--output')+1]=str(reference)
        with (out/'reference_pilot.log').open('w') as log:subprocess.run(command,cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    aa=torch.load(resume/'best.pt',map_location='cpu',weights_only=True);bb=torch.load(reference/'best.pt',map_location='cpu',weights_only=True)
    equal=all(torch.equal(aa['model'][k],bb['model'][k]) for k in aa['model'])
    assert equal and saved(resume)==saved(reference) and first['epochs_recorded']==2
    (out/'resume_probe.json').write_text(json.dumps(dict(epochs=2,paused_after=1,final_status=first['status'],weights_bitwise_equal_to_uninterrupted_pilot=equal,validation_predictions_identical=True,scope='same-runtime exact epoch-boundary replay; not cross-device or cross-version resume'),indent=2))
    # Warm batch-one timings, fixed seed101 WAV, identical hardware/thread budget.
    latency=[];sample=root/names[0]
    def measure(fn,sync):
        with torch.inference_mode():
            for _ in range(10):fn();sync()
            values=[]
            for _ in range(100):
                sync();start=time.perf_counter();fn();sync();values.append(1000*(time.perf_counter()-start))
        return dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
    for name in ('cnn_freq','crnn_freq','crnn_freq_ta','crnn_freq_ca','crnn_freq_tca'):
        folder=a.storage / 'runs/B_architecture'/('B_'+name+'_s101');ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        for dev in ('cpu','cuda'):
            model=build_model(name).to(dev).eval();model.load_state_dict(ck['model']);mean=ck['mean'];std=ck['std']
            x=((torch.from_numpy(base[:1])-mean)/std).to(dev)
            sync=torch.cuda.synchronize if dev=='cuda' else lambda:None
            model_only=measure(lambda:model(x),sync)
            def end_to_end():
                feat=log_mel(read_audio(sample)).unsqueeze(0)
                return model(((feat-mean)/std).to(dev))
            e2e=measure(end_to_end,sync)
            latency.append(dict(model=name,device=dev,parameters=sum(v.numel() for v in model.parameters()),checkpoint_bytes=(folder/'best.pt').stat().st_size,model_only=model_only,wav_to_logits=e2e))
    cpu_info=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    cpu_model=next((line.split(':',1)[1].strip() for line in cpu_info.splitlines() if line.startswith('model name')),None)
    (out/'warm_latency.json').write_text(json.dumps(dict(cpu_model=cpu_model,gpu=torch.cuda.get_device_name(0),threads=2,batch_size=1,warmup=10,repetitions=100,wav=names[0],scope='Warm OS file cache, loaded model, float32. Includes decode/frontend/transfer for WAV-to-logits; excludes model load and process startup. CUDA synchronized.',measurements=latency),indent=2))
    print('ALL_RUNTIME_CHECKS_PASSED',flush=True)

if __name__=='__main__':main()
