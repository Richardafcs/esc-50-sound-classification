"""Waveform augmentation correctness and fail-closed provenance; no test access."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import TensorDataset

from esc50_sound.audio import log_mel
from esc50_sound.augmentation import (add_gaussian_noise, augmentation_config,
    load_training_waveforms, log_mel_batch, mix_waveforms, validate_augmentation,
    verify_waveform_sources)
from esc50_sound.platform import TrainingPlan, ContractError
from esc50_sound.records import sha256
from esc50_sound.cli import build_parser
from esc50_sound.checkpoints import load_checkpoint
from esc50_sound.training import train, resume_train


def test_logmel_batch_matches_pinned_frontend():
    torch.set_num_threads(2)
    t = torch.arange(80000) / 16000
    waves = torch.stack([torch.sin(2 * torch.pi * 320 * t), torch.sin(2 * torch.pi * 450 * t)])
    batched = log_mel_batch(waves)
    torch.testing.assert_close(batched[0], log_mel(waves[0]), atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(batched[1], log_mel(waves[1]), atol=1e-5, rtol=1e-6)
    assert batched.shape == (2, 1, 64, 313)


def test_gaussian_noise_snr_and_reproducibility():
    t = torch.arange(80000) / 16000
    x = torch.sin(2 * torch.pi * 310 * t).unsqueeze(0)
    torch.manual_seed(91)
    y = add_gaussian_noise(x, 1.0, 20.0, 20.0)
    power_signal = x.square().mean()
    power_added = (y-x).square().mean()
    snr = 10 * torch.log10(power_signal / power_added).item()
    assert snr == pytest.approx(20.0, abs=0.015)
    torch.manual_seed(91)
    torch.testing.assert_close(y, add_gaussian_noise(x, 1.0, 20.0, 20.0), rtol=0, atol=0)
    assert torch.equal(add_gaussian_noise(x, 0.0, 10, 30), x)
    silence = torch.zeros_like(x)
    assert torch.count_nonzero(add_gaussian_noise(silence, 1.0, 20, 20)) == 0


def test_waveform_mixup_soft_label_consistency():
    x = torch.stack([torch.zeros(80000), torch.ones(80000), 2*torch.ones(80000)])
    y = torch.tensor([2, 5, 7])
    torch.manual_seed(17)
    mixed, partners, lam = mix_waveforms(x, y, .4)
    assert lam.shape == (3,)
    assert torch.all(lam > 0) and torch.all(lam < 1)
    indices = torch.tensor([int((y == yy).nonzero()[0]) for yy in partners])
    torch.testing.assert_close(mixed, lam[:,None] * x + (1-lam[:,None])*x[indices])
    assert mix_waveforms(x, y, 0)[2] is None


def test_invalid_config_fails_closed():
    for args in ((-0.1, 10, 20, 0), (1.1, 10, 20, 0), (.5, 25, 10, 0),
                 (.5, float('nan'), 30, 0), (.5, 10, 30, -1), (0, 10, 30, 11)):
        with pytest.raises(ValueError):
            validate_augmentation(*args)


def test_plan_v1_identity_unchanged_and_v2_enforces_augmentation(tmp_path):
    original = json.loads((Path(__file__).resolve().parents[1] / 'configs/examples/platform_cnn_cpu.json').read_text())
    plan = TrainingPlan.from_dict(original)
    from esc50_sound.records import canonical_hash
    assert plan.identity() == canonical_hash({**original, 'train_folds': tuple(original['train_folds'])})
    raw = json.loads((Path(__file__).resolve().parents[1] / 'configs/examples/platform_cnn_noise_v2.json').read_text())
    v2 = TrainingPlan.from_dict(raw)
    assert v2.noise_probability == .5
    assert v2.to_legacy_args().noise_probability == .5
    assert v2.identity() != TrainingPlan.from_dict({**raw, 'noise_probability': .25}).identity()
    with pytest.raises(ContractError):
        TrainingPlan.from_dict({**original, 'noise_probability': .5})
    with pytest.raises(ValueError):
        TrainingPlan.from_dict({**raw, 'mixup_alpha': float('nan')})


def test_only_training_wav_is_loaded_and_mismatched_bytes_rejected(tmp_path):
    root = tmp_path / 'ESC-50'
    (root/'meta').mkdir(parents=True)
    (root/'audio').mkdir()
    (root/'meta/esc50.csv').write_text('trusted synthetic metadata')
    t = np.arange(220500, dtype=np.float32) / 44100
    path = root/'audio'/'1-111-0-0.wav'
    sf.write(path, .3 * np.sin(2*np.pi*320*t), 44100, subtype='PCM_16')
    from esc50_sound.audio import read_audio
    clean = log_mel(read_audio(path)).numpy()
    cache = dict(x=clean[None], y=np.array([0]),
                 names=np.array([path.name]), sources=np.array(['111']),
                 folds=np.array([1]), class_names=np.array([str(i) for i in range(50)]))
    cache_path = tmp_path/'cache.npz'
    cache_path.write_bytes(b'test sidecar anchor')
    (tmp_path/'cache.npz.json').write_text(json.dumps(dict(schema_version=2,
        metadata_sha256=sha256(root/'meta/esc50.csv'), audio_file_sha256={path.name:sha256(path)})))
    rows = [dict(filename=path.name, fold=1, target=0, src_file='111', category='0')]
    with patch('esc50_sound.augmentation.read_metadata', return_value=rows), \
         patch('esc50_sound.augmentation.metadata_contract', return_value={'valid':True}), \
         patch('esc50_sound.augmentation.cache_metadata_contract', return_value={'valid':True}):
        result = load_training_waveforms(cache, cache_path, [0], root)
        assert len(result) == 1 and result[0][0].shape == (80000,)
        assert verify_waveform_sources(cache, cache_path, [0], root)['checked_training_files'] == 1
        sf.write(path, .2 * np.sin(2*np.pi*320*t), 44100)
        with pytest.raises(ValueError, match='modified training WAV'):
            load_training_waveforms(cache, cache_path, [0], root)


def _synthetic_cache():
    rng = np.random.default_rng(3)
    n = 20
    return dict(x=rng.normal(size=(n,1,64,313)).astype('float32'),
                y=np.arange(n)%5, folds=np.repeat(np.arange(1,6),4),
                sources=np.array([str(i) for i in range(n)]),
                names=np.array([f'file{i}.wav' for i in range(n)]),
                class_names=np.array([f'cls{i}' for i in range(50)]))


def _args(cache, output, pause=None):
    raw = ['train','--cache',str(cache),'--model','cnn_freq', '--output',str(output),
           '--epochs','3','--max-seconds','100','--batch-size','4','--threads','2',
           '--seed','111','--scheduler','cosine','--noise-probability','0.5',
           '--noise-snr-min-db','10','--noise-snr-max-db','30',
           '--mixup-alpha','.4']
    if pause is not None:
        raw += ['--pause-after-epochs',str(pause)]
    args = build_parser().parse_args(raw)
    del args.func
    return args


def test_noise_mixup_training_is_deterministic_after_epoch_resume(tmp_path, monkeypatch):
    cache = tmp_path/'synthetic.npz'
    cache.write_bytes(b'not real esc50 data')
    fake = _synthetic_cache()
    wave = torch.sin(2 * torch.pi * torch.arange(80000)/50).float()
    monkeypatch.setattr('esc50_sound.training.load_cache', lambda path: fake)
    monkeypatch.setattr('esc50_sound.training.load_training_waveforms',
                        lambda c, p, ids, root: TensorDataset(
                            torch.stack([wave*(1 + .02*i) for i in ids]),
                            torch.tensor(fake['y'][ids]).long()))
    ref, paused = tmp_path/'continuous', tmp_path/'paused'
    train(_args(cache,ref))
    train(_args(cache,paused,pause=1))
    resume = build_parser().parse_args(['resume-train','--output',str(paused),'--max-seconds','100'])
    del resume.func
    resume_train(resume)
    a,b = [load_checkpoint(folder/'last.pt') for folder in (ref,paused)]
    assert a['epoch'] == b['epoch'] == 3
    assert [r['training_loss'] for r in a['resume_state']['history']] == [r['training_loss'] for r in b['resume_state']['history']]
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    assert a['manifest']['config']['noise_probability'] == .5
    assert a['manifest']['config']['mixup_alpha'] == .4
