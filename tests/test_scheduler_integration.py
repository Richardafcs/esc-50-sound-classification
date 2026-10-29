"""Subprocess-level synthetic-cache integration; NOT real ESC-50 performance."""
import json
from pathlib import Path

import numpy as np

from esc50_sound.audio import FRONTEND
from esc50_sound.data import audit_metadata, metadata_records
from esc50_sound.records import canonical_hash, sha256, write_json
from esc50_sound.scheduler import initialize, dispatch


def test_scheduler_real_training_subprocess_one_epoch(tmp_path):
    rng = np.random.default_rng(123)
    n = 30
    folds = np.array([i % 5 + 1 for i in range(n)])
    y = np.array([i % 6 for i in range(n)])
    names = np.array([f'{folds[i]}-{100000+i}-0-{y[i]}.wav' for i in range(n)])
    sources = np.array([str(100000+i) for i in range(n)])
    classes = np.array([f'class_{i}' for i in range(50)])
    cache = tmp_path / 'synthetic_features.npz'
    np.savez_compressed(cache, x=rng.normal(0, 1, (n, 1, 64, 313)).astype(np.float32),
                        y=y, folds=folds, names=names, sources=sources, class_names=classes)
    rows = [dict(filename=str(names[i]), fold=int(folds[i]), target=int(y[i]),
                 category=str(classes[y[i]]), src_file=str(sources[i])) for i in range(n)]
    audit = audit_metadata(rows, require_complete=False)
    write_json(str(cache)+'.json', dict(schema_version=2, frontend=FRONTEND,
             cache_sha256=sha256(cache), audit=audit,
             metadata_records_sha256=canonical_hash(metadata_records(rows)),
             provenance={'type': 'synthetic test only; not official ESC-50'}))
    plan = dict(schema_version=1, max_cpu_threads=2,
                jobs=[dict(id='synthetic_train', kind='esc50_train', timeout_seconds=90, cpu_threads=2,
                           train_args=['--cache', str(cache), '--model', 'cnn_freq', '--epochs', '1',
                                       '--seed', '101', '--batch-size', '6'])])
    ws = tmp_path / 'queue'
    initialize(ws, plan)
    result = dispatch(ws, max_jobs=1, base_dir=str(Path(__file__).resolve().parents[1]))
    assert result['jobs'][0]['state'] == 'COMPLETED', (ws / 'jobs/synthetic_train/attempt_001.log').read_text()
    out = ws / 'jobs/synthetic_train/experiment'
    assert (out / 'last.pt').is_file() and (out / 'best.pt').is_file()
    assert json.loads((out / 'run.json').read_text())['status'] == 'completed'
    assert not (out / 'test_predictions.csv').exists()
