"""Strict, offline dataset and historical baseline evidence checks.

These checks deliberately do not inspect test predictions or load torch checkpoints.
A passing audit validates consistency of recorded evidence, not model quality.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .data import audit_metadata, metadata_records, split_rows
from .evaluation import compute_metrics
from .records import canonical_hash, sha256


def metadata_contract(rows: list[dict], *, require_official: bool = True) -> dict:
    """Validate exact ESC-50 filename/metadata relationship and split membership.

    Same-source across official folds is *reported*, not treated as corruption:
    source-strict splits may explicitly exclude those rows.
    """
    issues = []
    try:
        audit = audit_metadata(rows, require_complete=require_official)
    except (KeyError, ValueError, TypeError) as exc:
        return dict(valid=False, issues=[f"metadata structure: {exc}"], audit=None)
    for i, row in enumerate(rows):
        name = str(row['filename'])
        components = name.removesuffix('.wav').split('-')
        if len(components) != 4:
            issues.append(f"row {i}: unexpected ESC-50 filename format: {name}")
            continue
        fold, source, take, target = components
        if (fold != str(int(row['fold'])) or source != str(row['src_file']) or
                take != str(row.get('take', take)) or target != str(int(row['target']))):
            issues.append(f"row {i}: filename conflicts with fold/source/take/target: {name}")
    return dict(valid=not issues, issues=issues[:30], issue_count=len(issues), audit=audit,
                metadata_records_sha256=canonical_hash(metadata_records(rows)))


def cache_metadata_contract(cache: dict, rows: list[dict]) -> dict:
    """Match cached row identity *and labels* with the trusted raw metadata.

    A sidecar hash alone is insufficient if a cache and sidecar were generated
    from the same wrong labels. This check anchors them to independent metadata.
    """
    issues = []
    n = len(rows)
    for key in ('names', 'y', 'folds', 'sources', 'class_names'):
        if key not in cache:
            return dict(valid=False, issues=[f"missing cached field: {key}"])
    if len(cache['y']) != n:
        return dict(valid=False, issues=[f"cache rows {len(cache['y'])} != metadata rows {n}"])
    fields = [('names', 'filename', str), ('y', 'target', int),
              ('folds', 'fold', int), ('sources', 'src_file', str)]
    for cache_key, source_key, transform in fields:
        expected = [transform(row[source_key]) for row in rows]
        actual = [transform(value) for value in cache[cache_key].tolist()]
        mismatches = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        if mismatches:
            issues.append(f"{cache_key} differs from raw metadata at {len(mismatches)} rows; examples {mismatches[:5]}")
    by_target = {int(row['target']): str(row['category']) for row in rows}
    if cache['class_names'].shape != (50,) or any(
            str(cache['class_names'][i]) != name for i, name in by_target.items()):
        issues.append('class_names mapping differs from raw metadata')
    return dict(valid=not issues, issues=issues, checked_rows=n,
                evidence='full row-level alignment against independent raw metadata')


def split_contract(rows: list[dict], *, train_folds=(1, 2, 3), val_fold=4,
                   test_fold=5, strict=True) -> dict:
    indices = split_rows(rows, train_folds=train_folds, val_fold=val_fold,
                         test_fold=test_fold, strict=strict)
    issues = []
    names = {key: {rows[i]['filename'] for i in ids} for key, ids in indices.items()}
    sources = {key: {str(rows[i]['src_file']) for i in ids} for key, ids in indices.items()}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if names[a] & names[b]:
            issues.append(f'{a}/{b} repeated filenames')
        if strict and sources[a] & sources[b]:
            issues.append(f'{a}/{b} repeated source IDs')
    class_support = {key: dict(sorted(Counter(int(rows[i]['target']) for i in ids).items()))
                     for key, ids in indices.items()}
    for role, ids in indices.items():
        if not ids:
            issues.append(f'empty {role} role')
        if len(class_support[role]) != 50:
            issues.append(f'{role} has only {len(class_support[role])}/50 classes')
    return dict(valid=not issues, issues=issues, policy='source_strict' if strict else 'official',
                sizes={key: len(ids) for key, ids in indices.items()},
                source_disjoint=all(not (sources[a] & sources[b]) for a, b in
                                    (('train', 'validation'), ('train', 'test'), ('validation', 'test'))),
                excluded_from_requested_folds=sum(int(r['fold']) in train_folds or int(r['fold']) == val_fold
                                                  for r in rows) - len(indices['train']) - len(indices['validation']),
                class_support={key: {str(i): int(class_support[key].get(i, 0)) for i in range(50)}
                               for key in indices},
                split_fingerprint=canonical_hash({key: [rows[i]['filename'] for i in ids]
                                                  for key, ids in indices.items()}))


def baseline_artifact_contract(folder: str | Path) -> dict:
    """Read-only baseline audit; refuses to read test data or test_predictions."""
    folder = Path(folder)
    errors, warnings = [], []
    required = ['model.json', 'model.npz', 'search.json', 'validation.json', 'validation_predictions.csv']
    for filename in required:
        if not (folder / filename).is_file():
            errors.append(f'missing {filename}')
    if errors:
        return dict(valid=False, issues=errors, warnings=warnings, evidence='validation-only')
    try:
        manifest = json.loads((folder / 'model.json').read_text())
        search = json.loads((folder / 'search.json').read_text())
        validation = json.loads((folder / 'validation.json').read_text())
        if not isinstance(search, list) or not search:
            raise ValueError('empty or invalid hyperparameter search')
        if sha256(folder / 'model.npz') != manifest.get('model_sha256'):
            errors.append('model.npz SHA256 does not match manifest')
        if manifest.get('baseline_version') != 1:
            errors.append('unexpected baseline schema')
        best = min(search, key=lambda item: item['validation']['loss'])
        if best['l2'] != manifest.get('selected_l2'):
            errors.append('selected L2 differs from min-validation-loss candidate')
        if manifest.get('selection') != 'minimum validation cross-entropy; earliest grid tie':
            warnings.append('unknown model selection policy')
        if any(not entry.get('optimization', {}).get('success', False) for entry in search):
            warnings.append('one or more solver candidates did not report successful convergence')
        if sorted(float(x['l2']) for x in search) != sorted(float(x) for x in manifest['search_grid']):
            errors.append('search candidates do not match recorded grid')
        if any(abs(float(best['validation'][key]) - float(validation[key])) > 1e-7
               for key in ('accuracy', 'macro_f1', 'loss', 'n')):
            errors.append('selected search validation metrics disagree with validation.json')
        with np.load(folder / 'model.npz', allow_pickle=False) as weights:
            if (weights['weights'].shape != (128, 50) or weights['bias'].shape != (50,) or
                    weights['mean'].shape != (128,) or weights['std'].shape != (128,)):
                errors.append('baseline model array shapes are incompatible')
            if any(not np.isfinite(weights[k]).all() for k in ('weights','bias','mean','std')):
                errors.append('nonfinite model or standardization arrays')
            if (weights['std'] <= 0).any():
                errors.append('nonpositive feature standardization scale')
            if 'class_names' not in weights or weights['class_names'].shape != (50,):
                errors.append('class mapping missing in model')
        predictions = []
        with (folder / 'validation_predictions.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                predictions.append(row)
        if not predictions or len(predictions) != len({p['filename'] for p in predictions}):
            errors.append('empty or duplicate validation predictions')
        else:
            targets = np.asarray([int(p['target']) for p in predictions], dtype=np.int64)
            predicted = np.asarray([int(p['prediction']) for p in predictions], dtype=np.int64)
            measured = compute_metrics(targets, predicted)
            if any(abs(float(measured[k]) - float(validation[k])) > 1e-10
                   for k in ('accuracy', 'macro_f1', 'n')):
                errors.append('saved validation predictions disagree with reported metrics')
            expected = (manifest.get('split', {}).get('validation', {}).get('names') or [])
            if [p['filename'] for p in predictions] != expected:
                errors.append('saved validation prediction membership differs from manifest')
        return dict(valid=not errors, issues=errors, warnings=warnings,
                    model='logmel statistics + multinomial logistic (not MFCC+SVM)',
                    selected_l2=manifest.get('selected_l2'),
                    reported_validation_accuracy=validation.get('accuracy'),
                    prediction_rows=len(predictions),
                    convergence={str(row['l2']): bool(row['optimization']['success']) for row in search},
                    evidence='read-only model/hash/hyperparameter/validation-row audit; no test files read; no refit')
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f'baseline artifact unreadable or inconsistent: {type(exc).__name__}: {exc}')
        return dict(valid=False, issues=errors, warnings=warnings, evidence='validation-only')


def raw_audio_contract(root: str | Path, rows: list[dict], expected_hashes: dict | None) -> dict:
    """Check all real WAV headers and hashes against feature-cache acquisition manifest.

    Missing per-file provenance cannot be treated as verified. Uses headers and
    SHA256 only; regenerating STFT/features still requires a separate replay.
    """
    root = Path(root)
    errors = []
    if not isinstance(expected_hashes, dict) or len(expected_hashes) != len(rows):
        return dict(valid=False, checked=0, issues=[
            'Missing complete per-WAV SHA256 manifest: regenerate cache with updated prepare'])
    try:
        import soundfile as sf
    except ImportError:
        return dict(valid=False, checked=0, issues=['soundfile unavailable'])
    checked = 0
    for row in rows:
        path = root / 'audio' / str(row['filename'])
        if not path.is_file():
            errors.append(f'missing WAV: {path.name}')
            continue
        try:
            info = sf.info(path)
            if (info.samplerate != 44100 or info.channels != 1 or
                    abs(info.duration - 5.0) > 0.001):
                errors.append(f'WAV header mismatch: {path.name}')
            if sha256(path) != expected_hashes.get(path.name):
                errors.append(f'WAV SHA256 mismatch: {path.name}')
        except (ValueError, OSError) as exc:
            errors.append(f'WAV unreadable: {path.name}: {exc}')
        checked += 1
    return dict(valid=not errors, checked=checked, issues=errors[:30], issue_count=len(errors),
                evidence='raw WAV format and byte hashes match the recorded cache acquisition manifest')


def cepstral_artifact_contract(folder: str | Path) -> dict:
    """Independent validation-only audit of a saved cepstral LinearSVC baseline."""
    folder = Path(folder)
    issues, warnings = [], []
    required = ('model.json', 'model.npz', 'search.json', 'validation.json', 'validation_predictions.csv')
    missing = [k for k in required if not (folder / k).is_file()]
    if missing:
        return dict(valid=False, issues=[f'missing {m}' for m in missing], evidence='validation-only')
    try:
        manifest = json.loads((folder / 'model.json').read_text())
        history = json.loads((folder / 'search.json').read_text())
        metrics = json.loads((folder / 'validation.json').read_text())
        if manifest.get('schema_version') != 1 or 'LinearSVC' not in manifest.get('model', ''):
            issues.append('incorrect cepstral SVM schema')
        if sha256(folder / 'model.npz') != manifest.get('model_sha256'):
            issues.append('saved SVM coefficients do not match model hash')
        expected = max(history, key=lambda r: (r['validation']['accuracy'],
                                               r['validation']['macro_f1'], -r['C']))
        if expected['C'] != manifest.get('selected_C'):
            issues.append('selected C differs from validation ranking')
        if any(abs(expected['validation'][k] - metrics[k]) > 1e-10 for k in ('accuracy','macro_f1','n')):
            issues.append('selected validation metrics differ from search result')
        if [float(r['C']) for r in history] != [float(x) for x in manifest.get('search_grid', [])]:
            issues.append('recorded hyperparameter search grid differs from trials')
        if any(r.get('convergence_warning') for r in history):
            warnings.append('at least one solver configuration warned about convergence')
        with np.load(folder / 'model.npz', allow_pickle=False) as model:
            if (model['coefficients'].shape != (50, 40) or
                    model['intercept'].shape != (50,) or
                    model['mean'].shape != (40,) or model['std'].shape != (40,) or
                    model['classes'].tolist() != list(range(50)) or
                    model['class_names'].shape != (50,)):
                issues.append('model array or 50-category schema mismatch')
            if any(not np.isfinite(model[k]).all() for k in ('coefficients','intercept','mean','std')):
                issues.append('nonfinite SVM coefficients or statistics')
            if (model['std'] <= 0).any():
                issues.append('nonpositive training standardization scale')
        with (folder / 'validation_predictions.csv').open(newline='') as stream:
            predictions = list(csv.DictReader(stream))
        if not predictions or len(predictions) != len({r['filename'] for r in predictions}):
            issues.append('missing or duplicated validation predictions')
        else:
            measured = compute_metrics(np.array([int(r['target']) for r in predictions]),
                                       np.array([int(r['prediction']) for r in predictions]))
            if any(abs(measured[k] - metrics[k]) > 1e-10 for k in ('accuracy','macro_f1','n')):
                issues.append('validation prediction metrics do not match reported scores')
            if [r['filename'] for r in predictions] != manifest.get('split', {}).get('validation', {}).get('names'):
                issues.append('prediction filenames differ from frozen validation split')
        return dict(valid=not issues, issues=issues, warnings=warnings,
                    selected_C=manifest.get('selected_C'), reported_validation_accuracy=metrics.get('accuracy'),
                    prediction_rows=len(predictions),
                    evidence='validation-only; no checkpoint execution, no test metrics, no refitting')
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        return dict(valid=False, issues=[f'{type(exc).__name__}: {exc}'],warnings=warnings,evidence='validation-only')
