"""Additional classical baseline: DCT-of-logmel cepstral statistics + linear SVM.

Explicitly not a reproduction of Piczak's MFCC/SVM implementation. This uses
our pinned HTK log-power Mel frontend and SciPy DCT-II (20 coefficients), then
train-only StandardScaler-like statistics and a 50-way linear SVM.

Only training and validation labels are accessed during model selection.
Persist coefficients as npz; do not write executable pickle/joblib artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
from scipy.fft import dct

from .baseline import new_output, source_split, validated_cache
from .evaluation import compute_metrics, save_predictions
from .records import append_evaluation_audit, sha256, write_json

CEPS_VERSION = 1
C_GRID = (0.01, 0.1, 1.0, 10.0)


def cepstral_statistics(x, n_coefficients=20):
    """[N,1,64,T] -> [N,2*K] DCT-II mean/std; no fitted parameters."""
    x = np.asarray(x)
    if (x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != 64 or
            not np.isfinite(x).all() or not 1 <= n_coefficients <= 64):
        raise ValueError('Expected finite [N,1,64,T] log-power Mel feature input')
    cep = dct(x[:, 0].astype(np.float64), type=2, axis=1, norm='ortho')[:, :n_coefficients]
    return np.concatenate((cep.mean(axis=2), cep.std(axis=2)), axis=1)


def _features(cache, ids, mean, std):
    return (cepstral_statistics(cache['x'][ids]) - mean) / std


def _scores(y, prediction):
    score = compute_metrics(y, prediction)
    return {k: score[k] for k in ('accuracy', 'macro_f1', 'n')}


def fit_cepstral_svm(args):
    from sklearn.svm import LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    cache, digest, sidecar = validated_cache(args.cache)
    indices = source_split(cache, tuple(int(v) for v in args.train_folds.split(',')),
                           args.val_fold, args.test_fold, strict=not args.official_folds)
    train_ids, val_ids = indices['train'], indices['validation']
    x_train_raw = cepstral_statistics(cache['x'][train_ids])
    mean = x_train_raw.mean(axis=0)
    std = np.maximum(x_train_raw.std(axis=0), 1e-8)
    train_x = (x_train_raw - mean) / std
    val_x = _features(cache, val_ids, mean, std)
    train_y, val_y = cache['y'][train_ids], cache['y'][val_ids]
    if len(np.unique(train_y)) != 50:
        raise ValueError('All 50 ESC-50 categories must be in the training split')
    # Create new evidence location only AFTER cache and split validation.
    out = new_output(args.output)
    candidates = []
    selected = None
    for c in C_GRID:
        clf = LinearSVC(C=c, dual=False, max_iter=args.maxiter, random_state=0)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always', ConvergenceWarning)
            clf.fit(train_x, train_y)
        convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in observed)
        pred = clf.predict(val_x)
        score = _scores(val_y, pred)
        record = dict(C=c, validation=score, iterations=int(np.max(clf.n_iter_)),
                      convergence_warning=convergence_warning)
        candidates.append(record)
        # Highest validation accuracy; Macro-F1 tie break; prefer smaller C.
        rank = (score['accuracy'], score['macro_f1'], -c)
        if selected is None or rank > selected[0]:
            selected = (rank, clf, record, pred)
    _, clf, winning, predictions = selected
    model_path = out / 'model.npz'
    np.savez_compressed(model_path, coefficients=clf.coef_, intercept=clf.intercept_,
                        classes=clf.classes_, mean=mean, std=std,
                        class_names=cache['class_names'])
    manifest = dict(schema_version=CEPS_VERSION,
                    model='DCT(20) logmel cepstral mean/std + LinearSVC (not Piczak replication)',
                    frontend=sidecar['frontend'], cache_sha256=digest,
                    metadata_sha256=sidecar.get('metadata_sha256'), model_sha256=sha256(model_path),
                    feature_shape=40, n_cepstral=20, dct='DCT-II ortho of 64 natural log-power Mel bands',
                    normalization='training clips only',
                    source_policy='official' if args.official_folds else 'source_strict',
                    train_folds=[int(v) for v in args.train_folds.split(',')],
                    val_fold=args.val_fold, test_fold=args.test_fold,
                    split={k: {'n':len(ids), 'names':cache['names'][ids].tolist()}
                           for k,ids in indices.items()},
                    selection='max validation accuracy, then macro-F1, then smaller C',
                    search_grid=list(C_GRID), selected_C=winning['C'],
                    maxiter=args.maxiter, parameters=int(clf.coef_.size+clf.intercept_.size),
                    environment=dict(python=sys.version, numpy=np.__version__,
                                     sklearn=__import__('sklearn').__version__,platform=platform.platform()),
                    test_evaluated=False, warning='Not official standard four-training-fold CV; do not compare as SOTA')
    write_json(out / 'model.json', manifest)
    write_json(out / 'search.json', candidates)
    write_json(out / 'validation.json', winning['validation'])
    save_predictions(out / 'validation_predictions.csv', cache['names'][val_ids], val_y, predictions)
    print(json.dumps(dict(selected_C=winning['C'], validation=winning['validation'],
                          converged=not winning['convergence_warning']), indent=2))


def evaluate_cepstral_svm(args):
    if not args.allow_test:
        raise ValueError('Test is sealed; explicit --allow-test is required')
    artifact = Path(args.artifact)
    manifest = json.loads((artifact / 'model.json').read_text())
    if (manifest.get('schema_version') != CEPS_VERSION or
            sha256(artifact / 'model.npz') != manifest.get('model_sha256')):
        raise ValueError('Cepstral SVM model integrity failure')
    cache, digest, sidecar = validated_cache(args.cache)
    if digest != manifest['cache_sha256'] or sidecar['frontend'] != manifest['frontend']:
        raise ValueError('Cache/frontend differs from trained baseline')
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Output must be a new directory; no overwrites')
    out.mkdir(parents=True)
    audit_path = artifact / 'evaluation_audit.jsonl'
    audit = dict(event='test_started',timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 output=str(out.resolve()),model_sha256=manifest['model_sha256'])
    append_evaluation_audit(audit_path, audit)
    try:
        with np.load(artifact / 'model.npz', allow_pickle=False) as f:
            ids = np.flatnonzero(cache['folds'] == manifest['test_fold'])
            x = _features(cache, ids, f['mean'], f['std'])
            scores = x @ f['coefficients'].T + f['intercept']
            prediction = f['classes'][scores.argmax(axis=1)]
        result = _scores(cache['y'][ids], prediction)
        result.update(test_fold=manifest['test_fold'], source_policy=manifest['source_policy'],
                      interpretation='one previously exposed ESC-50 test fold; not independent confirmation')
        write_json(out / 'test.json', result)
        save_predictions(out / 'test_predictions.csv', cache['names'][ids], cache['y'][ids], prediction)
        append_evaluation_audit(audit_path, dict(audit,event='test_completed',accuracy=result['accuracy']))
    except BaseException as exc:
        append_evaluation_audit(audit_path, dict(audit,event='test_failed', error_type=type(exc).__name__))
        raise
    print(json.dumps(result, indent=2))
