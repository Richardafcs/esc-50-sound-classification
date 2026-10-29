"""Independent adversarial foundation checks; no test fold predictions or WAVs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import copy

import numpy as np
import pytest
import torch

from esc50_sound.checkpoints import load_checkpoint, save_checkpoint_atomic
from esc50_sound.data import read_metadata
from esc50_sound.infra import preflight_report
from esc50_sound.quality import (metadata_contract, cache_metadata_contract,
                                  split_contract, baseline_artifact_contract)

ROOT = Path(__file__).resolve().parents[1]


def test_real_esc50_metadata_and_strict_split_contract():
    rows = read_metadata(ROOT / "data/ESC-50")
    assert metadata_contract(rows)["valid"]
    split = split_contract(rows)
    assert split["valid"] and split["source_disjoint"]
    assert split["sizes"] == {"train": 1200, "validation": 398, "test": 400}
    assert split["excluded_from_requested_folds"] == 2
    assert metadata_contract(rows)["audit"]["source_fold_leakage"] is True


def test_metadata_filename_contract_rejects_silent_label_and_source_errors():
    rows = [dict(filename="1-100032-A-0.wav", fold="1", target="0", category="dog", src_file="100032", take="A")]
    assert metadata_contract(rows, require_official=False)["valid"]
    for field, bad in (("target", "1"), ("src_file", "100033"), ("fold", "2"), ("take", "B")):
        altered = copy.deepcopy(rows)
        altered[0][field] = bad
        assert not metadata_contract(altered, require_official=False)["valid"], field


def test_raw_metadata_alignment_detects_every_cache_identity_mismatch():
    rows = [dict(filename="1-100032-A-0.wav", fold="1", target="0", category="dog", src_file="100032")]
    cache = dict(names=np.array([rows[0]['filename']]), y=np.array([0]), folds=np.array([1]),
                 sources=np.array(['100032']), class_names=np.array(['dog'] + [f'c{i}' for i in range(1, 50)]))
    assert cache_metadata_contract(cache, rows)["valid"]
    for key, other in (("names", "1-100032-A-1.wav"), ("y", 1), ("folds", 2), ("sources", "000000")):
        wrong = copy.deepcopy(cache)
        wrong[key][0] = other
        assert not cache_metadata_contract(wrong, rows)["valid"], key
    wrong = copy.deepcopy(cache)
    wrong['class_names'][0] = 'cow'
    assert not cache_metadata_contract(wrong, rows)["valid"]


def test_preflight_detects_forged_cache_labels_even_if_cache_self_hashes_pass(tmp_path, monkeypatch):
    rows = read_metadata(ROOT / 'data/ESC-50')
    data_root = tmp_path / 'ESC-50'
    (data_root / 'meta').mkdir(parents=True)
    (data_root / 'meta' / 'esc50.csv').write_bytes((ROOT / 'data/ESC-50/meta/esc50.csv').read_bytes())
    cache_file = tmp_path / 'features.npz'
    cache_file.write_bytes(b'fake cache marker; mock loader validates its own digest')
    Path(str(cache_file) + '.json').write_text('{}')
    good = dict(names=np.array([r['filename'] for r in rows]),
                y=np.array([int(r['target']) for r in rows]),
                folds=np.array([int(r['fold']) for r in rows]),
                sources=np.array([r['src_file'] for r in rows]),
                class_names=np.array([next(r['category'] for r in rows if int(r['target']) == i) for i in range(50)]))
    monkeypatch.setattr('esc50_sound.infra.load_cache', lambda path: good)
    report = preflight_report(data_root, cache_file)
    assert report['ready'] and report['checks']['cache']['raw_metadata_alignment']['valid']
    wrong = copy.deepcopy(good)
    wrong['y'][0] = 1
    monkeypatch.setattr('esc50_sound.infra.load_cache', lambda path: wrong)
    report = preflight_report(data_root, cache_file)
    assert not report['ready'] and any('Cached labels' in b for b in report['blockers'])


def test_saved_historical_baseline_validates_without_test_files(tmp_path):
    source = ROOT / 'results/historical/final_study/results/statistics_baseline'
    assert baseline_artifact_contract(source)['valid']
    folder = tmp_path / 'baseline'
    folder.mkdir()
    for name in ('model.json', 'model.npz', 'search.json', 'validation.json', 'validation_predictions.csv'):
        (folder / name).write_bytes((source / name).read_bytes())
    # The auditor must never parse test metrics even if they are malformed.
    (folder / 'test_predictions.csv').write_text('garbage---')
    assert baseline_artifact_contract(folder)['valid']
    manifest = json.loads((folder / 'model.json').read_text())
    manifest['selected_l2'] = 99
    (folder / 'model.json').write_text(json.dumps(manifest))
    assert not baseline_artifact_contract(folder)['valid']


def test_checkpoint_atomic_replace_and_serialization_failure(tmp_path, monkeypatch):
    path = tmp_path / 'last.pt'
    save_checkpoint_atomic(path, {'epoch': 1, 'model': {'w': torch.arange(3)}})
    old_bytes = path.read_bytes()
    assert load_checkpoint(path)['epoch'] == 1
    original = torch.save
    def simulate_partial_write(value, stream):
        stream.write(b'partial checkpoint')
        raise OSError('simulated full disk')
    monkeypatch.setattr(torch, 'save', simulate_partial_write)
    with pytest.raises(OSError, match='full disk'):
        save_checkpoint_atomic(path, {'epoch': 2})
    assert path.read_bytes() == old_bytes
    assert not list(tmp_path.glob('*.tmp'))
    monkeypatch.setattr(torch, 'save', original)
    save_checkpoint_atomic(path, {'epoch': 2, 'model': {'w': torch.ones(3)}})
    assert load_checkpoint(path)['epoch'] == 2


def test_quick_mode_never_returns_verified_readiness(tmp_path, monkeypatch):
    rows = read_metadata(ROOT / 'data/ESC-50')
    root = tmp_path / 'ESC-50'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/esc50.csv').write_bytes((ROOT / 'data/ESC-50/meta/esc50.csv').read_bytes())
    cache = tmp_path / 'features.npz'
    cache.write_bytes(b'unverified')
    Path(str(cache) + '.json').write_text('{}')
    report = preflight_report(root, cache, verify_cache=False)
    assert not report['ready'] and any('Quick mode' in x for x in report['blockers'])


def test_raw_provenance_rejects_changed_wav(tmp_path):
    import soundfile as sf
    from esc50_sound.quality import raw_audio_contract
    from esc50_sound.records import sha256
    audio_dir = tmp_path / 'audio'
    audio_dir.mkdir()
    file = audio_dir / '1-123-A-0.wav'
    sf.write(file, np.zeros(44100 * 5, dtype=np.float32), 44100)
    rows = [dict(filename=file.name, fold=1, src_file='123', target=0, category='dog')]
    manifest = {file.name: sha256(file)}
    assert raw_audio_contract(tmp_path, rows, manifest)['valid']
    sf.write(file, np.ones(44100 * 5, dtype=np.float32), 44100)
    assert not raw_audio_contract(tmp_path, rows, manifest)['valid']
    assert not raw_audio_contract(tmp_path, rows, None)['valid']


def test_cepstral_baseline_trains_only_on_training_folds(tmp_path):
    """Synthetic data validates math/packaging only, not real ESC-50 accuracy."""
    from argparse import Namespace
    from esc50_sound.audio import FRONTEND
    from esc50_sound.data import audit_metadata, metadata_records
    from esc50_sound.records import canonical_hash, sha256, write_json
    from esc50_sound.cepstral import cepstral_statistics, fit_cepstral_svm
    labels, folds = np.repeat(np.arange(50), 5), np.tile(np.arange(1, 6), 50)
    names = np.array([f'{fold}-{5000+c}-{chr(65+fold-1)}-{c}.wav' for c, fold in zip(labels, folds)])
    sources = np.array([f'{5000+c}' for c in labels])
    # Same source intentionally appears in all five folds; strict filtering
    # would drop training classes, so this fixture uses the explicit official flag.
    classes = np.array([f'c{i}' for i in range(50)])
    rng = np.random.default_rng(1)
    x = rng.normal(-3, 0.05, (250, 1, 64, 313)).astype(np.float32)
    for i, label in enumerate(labels):
        x[i, 0, label % 64, :] += 4
    assert cepstral_statistics(x[:2]).shape == (2, 40)
    cache = tmp_path / 'synthetic.npz'
    np.savez_compressed(cache, x=x, y=labels, folds=folds, names=names,
                        sources=sources, class_names=classes)
    rows = [dict(filename=str(n), fold=int(f), target=int(y), src_file=str(s), category=str(classes[y]))
            for n,f,y,s in zip(names,folds,labels,sources)]
    sidecar = dict(schema_version=2,frontend=FRONTEND,cache_sha256=sha256(cache),
                   audit=audit_metadata(rows,require_complete=False),
                   metadata_records_sha256=canonical_hash(metadata_records(rows)))
    write_json(str(cache)+'.json', sidecar)
    out = tmp_path / 'result'
    fit_cepstral_svm(Namespace(cache=str(cache), output=str(out),train_folds='1,2,3',
                               val_fold=4,test_fold=5,official_folds=True,maxiter=100))
    assert (out / 'model.npz').is_file()
    assert (out / 'validation_predictions.csv').is_file()
    assert not (out / 'test.json').exists()
    assert not (out / 'test_predictions.csv').exists()
    manifest = json.loads((out / 'model.json').read_text())
    assert manifest['feature_shape'] == 40 and manifest['source_policy'] == 'official'
    from esc50_sound.quality import cepstral_artifact_contract
    assert cepstral_artifact_contract(out)['valid']
    with np.load(out / 'model.npz', allow_pickle=False) as data:
        assert data['coefficients'].shape == (50, 40)
        assert data['mean'].shape == (40,)
        assert np.all(data['std'] > 0)
