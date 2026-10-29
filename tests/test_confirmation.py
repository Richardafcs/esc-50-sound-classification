"""Confirmation guards using synthetic WAVs only; no real held-out inference."""
import copy
import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('study_confirmation', ROOT / 'scripts/legacy/study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
esc = study.esc
torch.set_num_threads(2)


@pytest.fixture
def confirmation_fixture(tmp_path):
    root = tmp_path/'raw'
    (root/'audio').mkdir(parents=True)
    (root/'meta').mkdir()
    rows, features = [], []
    for label in range(50):
        fold = label % 5 + 1
        name = f'{fold}-synthetic{label}-A-{label}.wav'
        wave = np.full(80000, .05 + label*.001, dtype=np.float32)
        sf.write(root/'audio'/name, wave, 16000, subtype='FLOAT')
        features.append(esc.log_mel(torch.from_numpy(wave)).numpy())
        rows.append(dict(filename=name,fold=fold,target=label,category=f'class{label}',src_file=f'synthetic{label}'))
    metadata = root/'meta/esc50.csv'
    with metadata.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cache=tmp_path/'cache.npz'
    np.savez_compressed(cache,x=np.stack(features).astype(np.float32),
                        y=np.arange(50,dtype=np.int64),folds=np.array([r['fold'] for r in rows]),
                        sources=np.array([r['src_file'] for r in rows]),names=np.array([r['filename'] for r in rows]),
                        class_names=np.array([r['category'] for r in rows]))
    esc.write_json(str(cache)+'.json',dict(schema_version=2,frontend=copy.deepcopy(esc.FRONTEND),
                   cache_sha256=esc.sha256(cache),metadata_sha256=esc.sha256(metadata),
                   metadata_records_sha256=esc.canonical_hash(esc.metadata_records(rows)),
                   audit=esc.audit_metadata(rows,require_complete=False)))
    config=dict(model='cnn',train_folds='1,2,3',val_fold=4,test_fold=5,official_folds=False)
    classes=[r['category'] for r in rows]
    manifest=dict(config=config,frontend=copy.deepcopy(esc.FRONTEND),class_names=classes,
                  cache_sha256=esc.sha256(cache),source_policy='strict synthetic fixture')
    checkpoint=dict(schema_version=2,model_name='cnn',model=esc.build_model('cnn').state_dict(),
                    frontend=copy.deepcopy(esc.FRONTEND),mean=torch.zeros(1,1,64,1),std=torch.ones(1,1,64,1),
                    class_names=classes,train_folds=(1,2,3),val_fold=4,test_fold=5,
                    cache_sha256=esc.sha256(cache),source_policy=manifest['source_policy'],
                    manifest=manifest,manifest_id=esc.canonical_hash(manifest))
    run=tmp_path/'run';run.mkdir();path=run/'best.pt';torch.save(checkpoint,path)
    esc.write_json(run/'run.json',dict(manifest_id=checkpoint['manifest_id'],test_evaluated=False))
    freeze=tmp_path/'freeze.json'
    frozen=dict(confirmation_schema_version=1,checkpoint_sha256={'cnn':esc.sha256(path)},
                measurement_protocol={'stress':dict(split='test',conditions=['clean','gain_6'],seed=2026),
                                      'order':dict(split='test',permutation_seeds=[101,202])})
    esc.write_json(freeze,frozen)
    args=Namespace(command='stress',checkpoint=str(path),cache=str(cache),root=str(root),
                   output=str(tmp_path/'result'),split='test',allow_test=True,freeze=str(freeze),
                   allow_legacy_checkpoint=False,conditions=['clean','gain_6'],seed=2026)
    return dict(root=root,cache=cache,rows=rows,checkpoint=checkpoint,path=path,freeze=freeze,
                frozen=frozen,args=args)


def audit_records(fixture):
    return [json.loads(line) for line in (fixture['path'].parent/'evaluation_history.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('change',[{'conditions':['clean']},{'conditions':['gain_6','clean']},{'seed':999}])
def test_final_stress_rejects_unregistered_settings_before_access(confirmation_fixture,change,monkeypatch):
    f=confirmation_fixture;args=copy.copy(f['args'])
    for key,value in change.items():setattr(args,key,value)
    monkeypatch.setattr(study,'restore',lambda *a,**k:pytest.fail('No weights or test cache should be accessed'))
    with pytest.raises(ValueError,match='registered exactly'):
        study.stress(args)
    assert not (f['path'].parent/'evaluation_history.jsonl').exists()
    assert not Path(args.output).exists()


def test_order_protocol_binds_seeds_and_measurement_type(confirmation_fixture):
    f=confirmation_fixture;args=copy.copy(f['args']);args.command='order';args.permutation_seeds=[101,202]
    observed=study.check_holdout(args,f['path'])
    assert observed['protocol']==dict(split='test',permutation_seeds=[101,202])
    args.permutation_seeds=[101,303]
    with pytest.raises(ValueError,match='registered exactly'):study.check_holdout(args,f['path'])
    args.command='stress'
    assert study.check_holdout(args,f['path'])['measurement']=='stress'


def test_checkpoint_only_freeze_is_insufficient(confirmation_fixture):
    f=confirmation_fixture;esc.write_json(f['freeze'],dict(checkpoint_sha256={'cnn':esc.sha256(f['path'])}))
    with pytest.raises(ValueError,match='confirmation_schema_version'):
        study.check_holdout(f['args'],f['path'])


def test_audit_starts_before_loading_heldout_cache(confirmation_fixture,monkeypatch):
    f=confirmation_fixture;original=esc.load_cache
    def observed(path):
        assert [r['status'] for r in audit_records(f)]==['started']
        return original(path)
    monkeypatch.setattr(esc,'load_cache',observed)
    monkeypatch.setattr(study,'_stress',lambda *a,**k:None)
    study.stress(f['args'])
    assert [r['status'] for r in audit_records(f)]==['started','completed']


def test_wrong_raw_metadata_rejected_before_any_forward(confirmation_fixture,monkeypatch):
    f=confirmation_fixture
    with (f['root']/'meta/esc50.csv').open('a') as stream:stream.write('\n')
    class Forbidden(torch.nn.Module):
        def forward(self,*args,**kwargs):pytest.fail('Metadata mismatch must reject before model forward')
    monkeypatch.setattr(study,'restore',lambda *a,**k:(f['checkpoint'],Forbidden()))
    with pytest.raises(ValueError,match='root metadata'):
        study.stress(f['args'])
    assert [r['status'] for r in audit_records(f)]==['started','failed']
    assert json.loads((f['path'].parent/'run.json').read_text())['test_evaluated'] is True


def test_changed_waveform_rejected_before_any_forward(confirmation_fixture,monkeypatch):
    f=confirmation_fixture;row=next(r for r in f['rows'] if r['fold']==5)
    sf.write(f['root']/'audio'/row['filename'],np.full(80000,.4,dtype=np.float32),16000,subtype='FLOAT')
    class Forbidden(torch.nn.Module):
        def forward(self,*args,**kwargs):pytest.fail('All reference features must pass before any forward')
    monkeypatch.setattr(study,'restore',lambda *a,**k:(f['checkpoint'],Forbidden()))
    with pytest.raises(ValueError,match='features differ from cache'):
        study.stress(f['args'])
    assert [r['status'] for r in audit_records(f)]==['started','failed']


def test_provenance_complete_before_forward_and_audio_decoded_once(confirmation_fixture,monkeypatch):
    f=confirmation_fixture;calls=[];original=esc.read_audio
    def observed(path):calls.append(path);return original(path)
    monkeypatch.setattr(esc,'read_audio',observed)
    class Check(torch.nn.Module):
        def forward(self,x):
            provenance=json.loads((Path(f['args'].output)/'raw_audio_provenance.json').read_text())
            assert len(provenance['audio_sha256'])==10
            assert provenance['clean_feature_equivalence']['atol']==1e-5
            assert provenance['clean_feature_equivalence']['rtol']==0
            assert provenance['clean_feature_equivalence']['max_absolute_difference']==0
            assert len(calls)==10
            return torch.zeros(len(x),50)
    monkeypatch.setattr(study,'restore',lambda *a,**k:(f['checkpoint'],Check()))
    study.stress(f['args'])
    assert len(calls)==10
    records=audit_records(f)
    assert [r['status'] for r in records]==['started','completed']
    assert records[0]['freeze']['protocol']['conditions']==['clean','gain_6']
    snapshot=json.loads((Path(f['args'].output)/'freeze_snapshot.json').read_text())
    assert snapshot['document']==f['frozen']


@pytest.mark.parametrize('conditions',[[],['clean','clean'],['snr_nan'],['snr_inf'],['unknown'],['gain_1000']])
def test_invalid_conditions_rejected_before_measurement(conditions):
    with pytest.raises(ValueError):study.validate_stress_conditions(conditions)


def test_replaced_checkpoint_cannot_bypass_frozen_hash(confirmation_fixture,monkeypatch):
    f=confirmation_fixture;original=study.check_holdout
    def replaced_after_gate(args,path):
        freeze=original(args,path)
        checkpoint=esc.load_checkpoint(path)
        first=next(iter(checkpoint['model']))
        checkpoint['model'][first]=checkpoint['model'][first]+.1
        torch.save(checkpoint,path)
        return freeze
    monkeypatch.setattr(study,'check_holdout',replaced_after_gate)
    with pytest.raises(ValueError,match='Checkpoint changed'):
        study.stress(f['args'])
    assert not Path(f['args'].output).exists()
    assert not (f['path'].parent/'evaluation_history.jsonl').exists()
