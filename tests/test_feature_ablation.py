"""Ablation contract tests: frequency/sample controls, no test leakage, no SHA dependency."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from esc50_sound.audio import log_mel
from esc50_sound.feature_ablation import (FeatureSpec, GRID, by_name, extract_feature,
                                          prepare_case, load_case, run_case, summary, paired_contrasts, completed_run)
from esc50_sound.models import build_model


def test_grid_is_factorial_and_sampling_is_controlled():
    assert len(GRID) == 12
    assert len({s.name for s in GRID}) == 12
    assert {(s.sample_rate, s.max_frequency) for s in GRID} == {
        (16000, 8000), (32000, 8000), (32000, 16000)}
    assert {s.feature for s in GRID} == {'logmel32', 'logmel64', 'mfcc40', 'loglinear64'}
    assert {s.hop * 1000 // s.sample_rate for s in GRID} == {16}
    assert {s.n_fft * 1000 // s.sample_rate for s in GRID} == {32}
    with pytest.raises(ValueError):
        FeatureSpec(16000, 16000, 'logmel64')
    with pytest.raises(ValueError):
        by_name('not-a-case')


def test_all_features_finite_and_parameter_matched():
    torch.set_num_threads(2)
    for s in GRID:
        t = torch.arange(s.sample_rate * 5) / s.sample_rate
        wave = .2 * torch.sin(2 * torch.pi * 400 * t)
        x = extract_feature(wave, s)
        assert x.shape == (1, s.bands, 313)
        assert torch.isfinite(x).all()
        assert build_model('cnn')(x.unsqueeze(0)).shape == (1, 50)
    params = [sum(p.numel() for p in build_model('cnn').parameters()) for _ in GRID]
    assert len(set(params)) == 1


def test_16k_logmel64_is_legacy_exact():
    wave = torch.randn(80000, generator=torch.Generator().manual_seed(8))
    assert torch.equal(extract_feature(wave, FeatureSpec(16000,8000,'logmel64')), log_mel(wave))


def _partial_fixture(tmp_path):
    root = tmp_path / 'ESC-50'
    (root / 'meta').mkdir(parents=True)
    (root / 'audio').mkdir()
    rows = []
    for i, fold in enumerate([1,1,2,2,3,3,4,4,5,5]):
        target = i % 2
        filename = f'{fold}-{100 + i}-0-{target}.wav'
        t = np.arange(220500, dtype=np.float64) / 44100
        wave = .2 * np.sin(2*np.pi * (300 + 300 * target) * t)
        sf.write(root / 'audio' / filename, wave, 44100, subtype='PCM_16')
        rows.append(dict(filename=filename, fold=fold, target=target, category=f'class{target}',
                         src_file=str(100+i), take='0'))
    with (root / 'meta/esc50.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_cache_checks_data_without_sha_and_rejects_label_swaps(tmp_path):
    root = _partial_fixture(tmp_path)
    spec = FeatureSpec(16000,8000,'logmel64')
    cachedir = tmp_path / 'cache'
    prepare_case(root, cachedir, spec, require_complete=False)
    cache, rows = load_case(root, cachedir, spec, require_complete=False)
    assert cache['x'].shape == (10, 1, 64, 313)
    sidecar = json.loads((cachedir / f'{spec.name}.json').read_text())
    assert 'sha256' not in json.dumps(sidecar).lower()
    with pytest.raises(FileExistsError):
        prepare_case(root,cachedir,spec,require_complete=False)
    with (root/'meta/esc50.csv').open() as f:
        content = f.read()
    (root/'meta/esc50.csv').write_text(content.replace('class0','wrong0'))
    with pytest.raises(ValueError, match='category mapping'):
        load_case(root,cachedir,spec, require_complete=False)
    (root/'meta/esc50.csv').write_text(content)
    bad = content.replace('1-100-0-0.wav,1,0', '1-100-0-0.wav,1,1')
    (root/'meta/esc50.csv').write_text(bad)
    with pytest.raises(ValueError):
        load_case(root,cachedir,spec,require_complete=False)


def test_one_epoch_training_validation_only_and_summary(tmp_path):
    torch.set_num_threads(2)
    root = _partial_fixture(tmp_path)
    spec = FeatureSpec(16000,8000,'logmel32')
    cachedir = tmp_path / 'cache'
    prepare_case(root,cachedir,spec,require_complete=False)
    results = tmp_path/'results'
    run_dir = results/'cnn'/spec.name/'seed101'
    metrics = run_case(root,cachedir,run_dir,spec,epochs=1,batch_size=2,
                       require_complete=False,threads=2)
    assert 0 <= metrics['accuracy'] <= 1
    assert (run_dir/'best.pt').exists()
    assert (run_dir/'validation_predictions.csv').exists()
    assert not (run_dir/'test_predictions.csv').exists()
    with (run_dir/'validation_predictions.csv').open() as f:
        preds=list(csv.DictReader(f))
    assert len(preds) == 2 and all(p['filename'].startswith('4-') for p in preds)
    with pytest.raises(FileExistsError):
        run_case(root,cachedir,run_dir,spec,epochs=1,require_complete=False)
    rows=summary(results,tmp_path/'summary.csv')
    assert len(rows)==1 and rows[0]['accuracy']==metrics['accuracy']


def test_paired_contrasts_identify_sample_rate_effect_and_reject_mixed_recipe(tmp_path):
    root = _partial_fixture(tmp_path)
    cachedir = tmp_path/'cache'
    results = tmp_path / 'results/historical/local_runs'
    specs = [FeatureSpec(16000,8000,'logmel64'), FeatureSpec(32000,8000,'logmel64')]
    for spec in specs:
        prepare_case(root,cachedir,spec,require_complete=False)
        run_case(root,cachedir,results/'cnn'/spec.name/'seed101',spec,
                 seed=101,epochs=1,batch_size=2,threads=2,require_complete=False)
    contrasts=paired_contrasts(results,tmp_path/'contrasts.csv')
    match=[x for x in contrasts if x['effect']=='sample_rate_matched_bandwidth']
    assert len(match)==1
    assert match[0]['left']==specs[0].name and match[0]['right']==specs[1].name
    assert not any('test' in p.name for p in results.rglob('*.csv'))
    config=results/'cnn'/specs[1].name/'seed101'/'config.json'
    cfg=json.loads(config.read_text())
    cfg['epochs']=2
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Incomplete ablation exposure'):
        paired_contrasts(results,tmp_path/'reject.csv')
    cfg['epochs']=1
    cfg['learning_rate']=.002
    config.write_text(json.dumps(cfg))
    checkpoint_path=config.parent/'best.pt'
    checkpoint=torch.load(checkpoint_path,weights_only=True)
    checkpoint['training_config']=cfg
    torch.save(checkpoint,checkpoint_path)
    with pytest.raises(ValueError,match='Incompatible training condition'):
        paired_contrasts(results,tmp_path/'reject.csv')


def test_feature_dimensions_do_not_change_model_parameter_count():
    # CNN global frequency pooling avoids parameter-count confounding.
    network = build_model('cnn')
    network.eval()
    with torch.no_grad():
        for num_bands in (32,40,64):
            assert network(torch.zeros(2,1,num_bands,313)).shape==(2,50)
    assert sum(p.numel() for p in network.parameters()) == sum(
        p.numel() for p in build_model('cnn').parameters())


@pytest.fixture
def completed_ablation(tmp_path):
    root = _partial_fixture(tmp_path)
    spec = FeatureSpec(16000, 8000, "logmel64")
    prepare_case(root, tmp_path / "cache", spec, require_complete=False)
    run_dir = tmp_path / 'results/historical/local_runs/one'
    run_case(root, tmp_path / "cache", run_dir, spec, epochs=2, batch_size=2,
             threads=2, require_complete=False)
    return run_dir


@pytest.mark.parametrize("corruption", ["missing_history", "short_history", "wrong_epoch",
                                      "prediction", "wrong_checkpoint", "heldout_identity"])
def test_summary_and_contrasts_reject_inconsistent_evidence(completed_ablation, tmp_path, corruption):
    run = completed_ablation
    if corruption == "missing_history":
        (run / "history.json").unlink()
    elif corruption == "short_history":
        path = run / "history.json"
        path.write_text(json.dumps(json.loads(path.read_text())[:1]))
    elif corruption == "wrong_epoch":
        path = run / "validation.json"
        value = json.loads(path.read_text())
        value["best_epoch"] = 99
        path.write_text(json.dumps(value))
    elif corruption == "prediction":
        path = run / "validation_predictions.csv"
        rows = list(csv.DictReader(path.open()))
        rows[0]["prediction"] = str((int(rows[0]["prediction"]) + 1) % 50)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    elif corruption == "wrong_checkpoint":
        path = run / "best.pt"
        value = torch.load(path, weights_only=True)
        value["seed"] = 999
        torch.save(value, path)
    else:
        # A mutually consistent CSV/config still cannot relabel holdout as validation.
        path = run / "validation_predictions.csv"
        rows = list(csv.DictReader(path.open()))
        rows[0]["filename"] = "5" + rows[0]["filename"][1:]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        path = run / "config.json"
        value = json.loads(path.read_text())
        value["validation_names"][0] = rows[0]["filename"]
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        summary(run.parent, tmp_path / "bad_summary.csv")
    with pytest.raises(ValueError):
        paired_contrasts(run.parent, tmp_path / "bad_contrasts.csv")
    assert not (tmp_path / "bad_summary.csv").exists()
    assert not (tmp_path / "bad_contrasts.csv").exists()


def test_completed_run_validation_preserves_rng(completed_ablation):
    torch.manual_seed(234)
    before = torch.get_rng_state().clone()
    completed_run(completed_ablation)
    assert torch.equal(before, torch.get_rng_state())


def test_summary_rejects_duplicate_experimental_units(completed_ablation, tmp_path):
    import shutil
    shutil.copytree(completed_ablation, completed_ablation.parent / "copy")
    with pytest.raises(ValueError, match="Duplicate ablation evidence"):
        summary(completed_ablation.parent, tmp_path / "duplicate.csv")
