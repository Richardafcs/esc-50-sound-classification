"""Adversarial epoch-resume equivalence on synthetic cached features only.

No ESC-50 test data and no external network calls are used.
"""
from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from esc50_sound.cli import build_parser
from esc50_sound.checkpoints import load_checkpoint, validate_checkpoint
from esc50_sound.training import train, resume_train
from esc50_sound.runtime_state import capture_rng, restore_rng


def synthetic_cache():
    rng = np.random.default_rng(37)
    n = 20
    return dict(x=rng.normal(0, 1, (n, 1, 64, 313)).astype('float32'),
                y=np.arange(n) % 5, folds=np.repeat(np.arange(1, 6), 4),
                sources=np.array([f's{i}' for i in range(n)]),
                names=np.array([f'a{i:03}.wav' for i in range(n)]),
                class_names=np.array([f'class{i}' for i in range(50)]))


def parsed(*arguments):
    a = build_parser().parse_args(arguments)
    del a.func
    return a


def fixture(tmp_path, monkeypatch):
    file = tmp_path / 'cache.fake'
    file.write_bytes(b'synthetic feature fixture, NOT ESC-50')
    monkeypatch.setattr('esc50_sound.training.load_cache', lambda path: synthetic_cache())
    return file


def _training(cache, target, *, model='cnn_freq', epochs=4, pause=None, augment=True):
    args = ['train','--cache',str(cache),'--model',model,'--output',str(target),
            '--epochs',str(epochs),'--threads','2','--device','cpu',
            '--batch-size','4','--seed','107','--max-seconds','120',
            '--scheduler','cosine']
    if pause is not None:
        args += ['--pause-after-epochs',str(pause)]
    if augment: args += ['--specaugment']
    train(parsed(*args))


def _compare(reference, recovered):
    a, b = (load_checkpoint(root / 'last.pt') for root in (reference, recovered))
    assert a['epoch'] == b['epoch'] == 4
    assert a['best_epoch'] == b['best_epoch']
    assert a['best_validation_loss'] == b['best_validation_loss']
    assert a['resume_state']['updates'] == b['resume_state']['updates']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    for key in a['resume_state']['best_model']:
        torch.testing.assert_close(a['resume_state']['best_model'][key], b['resume_state']['best_model'][key], rtol=0, atol=0)
    assert [r['validation'] for r in a['resume_state']['history']] == [r['validation'] for r in b['resume_state']['history']]
    assert [r['training_loss'] for r in a['resume_state']['history']] == [r['training_loss'] for r in b['resume_state']['history']]
    assert [r['next_learning_rate'] for r in a['resume_state']['history']] == [r['next_learning_rate'] for r in b['resume_state']['history']]
    for name in ('validation.json','validation_predictions.csv','training_evaluation.json'):
        assert (reference / name).read_bytes() == (recovered / name).read_bytes(), name
    assert json.loads((recovered/'run.json').read_text())['status'] == 'completed'
    assert 'recovered_from_epoch' in json.loads((recovered/'run.json').read_text())


@pytest.mark.parametrize('model', ['cnn_freq', 'crnn_freq_ta'])
def test_uninterrupted_equals_resumed_with_dropout_augment_and_cosine(tmp_path, monkeypatch, model):
    data = fixture(tmp_path, monkeypatch)
    ref, interrupted = tmp_path/'reference', tmp_path/'interrupted'
    _training(data, ref, model=model)
    _training(data, interrupted, model=model, pause=2)
    assert json.loads((interrupted/'run.json').read_text())['status'] == 'time_limit'
    resume_train(parsed('resume-train','--output',str(interrupted),'--max-seconds','120'))
    _compare(ref, interrupted)


def test_crash_between_last_checkpoint_and_json_sidecars_is_repaired(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    target = tmp_path/'interrupted'
    _training(data, target, pause=2)
    (target/'history.json').write_text('{damaged')
    (target/'validation.json').unlink()
    (target/'best.pt').unlink()
    (target/'run.json').write_text(json.dumps({**json.loads((target/'run.json').read_text()),'status':'failed'}))
    resume_train(parsed('resume-train','--output',str(target),'--max-seconds','120'))
    assert len(json.loads((target/'history.json').read_text())) == 4
    assert (target/'best.pt').exists() and (target/'validation.json').exists()


def test_refuses_resume_after_test_access_or_tampering(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    target = tmp_path/'interrupted'
    _training(data, target, pause=1)
    record = json.loads((target/'run.json').read_text())
    record['test_evaluated'] = True
    (target/'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='test access'):
        resume_train(parsed('resume-train','--output',str(target)))
    record['test_evaluated'] = False
    (target/'run.json').write_text(json.dumps(record))
    data.write_bytes(b'tampered synthetic fixture')
    with pytest.raises(ValueError, match='digest mismatch'):
        resume_train(parsed('resume-train','--output',str(target)))


def test_refuses_legacy_checkpoint_and_completed_run(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    target = tmp_path/'interrupted'
    _training(data, target, pause=1)
    original = load_checkpoint(target/'last.pt')
    del original['resume_state']
    torch.save(original, target/'last.pt')
    with pytest.raises(ValueError, match='missing resume'):
        resume_train(parsed('resume-train','--output',str(target)))
    record = json.loads((target/'run.json').read_text())
    record['status']='completed'
    (target/'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Only interrupted'):
        resume_train(parsed('resume-train','--output',str(target)))


def test_rng_snapshots_are_weights_only_loadable(tmp_path):
    import numpy as np, random
    import torch
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    generator = torch.Generator().manual_seed(23)
    rng = capture_rng(generator)
    torch.save(rng, tmp_path/'rng.pt')
    rng = torch.load(tmp_path/'rng.pt', weights_only=True)
    expected = [torch.rand(1).item(), float(np.random.random()), random.random(),
                torch.randperm(10, generator=generator).tolist()]
    restore_rng(rng, generator)
    actual = [torch.rand(1).item(), float(np.random.random()), random.random(),
              torch.randperm(10, generator=generator).tolist()]
    assert actual == expected


def test_mid_epoch_failure_rolls_back_all_partial_updates(tmp_path, monkeypatch):
    import esc50_sound.training as training
    data = fixture(tmp_path, monkeypatch)
    reference, interrupted = tmp_path/'ref', tmp_path/'interrupted'
    _training(data, reference, epochs=4)
    original = training.spec_augment
    seen = {'batches': 0}
    def abort_during_third_epoch(x):
        seen['batches'] += 1
        # 3 train batches/epoch: abort after one or more updates in epoch 3.
        if seen['batches'] == 8:
            raise RuntimeError('synthetic mid-epoch interruption')
        return original(x)
    monkeypatch.setattr(training, 'spec_augment', abort_during_third_epoch)
    with pytest.raises(RuntimeError, match='synthetic mid-epoch interruption'):
        _training(data, interrupted, epochs=4)
    checkpoint = load_checkpoint(interrupted/'last.pt')
    assert checkpoint['epoch'] == 2
    assert json.loads((interrupted/'run.json').read_text())['status'] == 'failed'
    monkeypatch.setattr(training, 'spec_augment', original)
    resume_train(parsed('resume-train','--output',str(interrupted),'--max-seconds','120'))
    _compare(reference, interrupted)
    assert json.loads((interrupted/'run.json').read_text())['training_attempts'][-1]['start_epoch'] == 2


def test_best_checkpoint_is_inference_only_without_stale_optimizer(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    target = tmp_path/'run'
    _training(data, target, epochs=3)
    best = load_checkpoint(target/'best.pt')
    assert best['checkpoint_role'] == 'validation-selected inference only'
    assert 'optimizer' not in best and 'scheduler' not in best
    assert best['epoch'] == best['best_epoch']
    assert validate_checkpoint(best) == 'verified embedded manifest'
