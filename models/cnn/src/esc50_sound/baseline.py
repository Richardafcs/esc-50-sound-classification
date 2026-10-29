"""Log-mel temporal-statistics multinomial logistic diagnostic baseline.

Optimization and feature definitions preserve the original experiment. Dataset
validation, source splits, record helpers and metrics use their owning modules.
Independent metric-recalculation oracles belong in research analysis tools.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .audio import FRONTEND
from .data import load_cache, split_rows
from .evaluation import compute_metrics, save_predictions
from .records import sha256, write_json

BASELINE_VERSION = 1
L2_GRID = (0.001, 0.01, 0.1)
EXPECTED_FRONTEND = FRONTEND


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output must be empty to preserve existing evidence: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validated_cache(path):
    path = Path(path)
    cache = load_cache(path)
    return cache, sha256(path), json.loads(Path(str(path) + ".json").read_text())


def source_split(cache, train_folds=(1, 2, 3), val_fold=4, test_fold=5, strict=True):
    rows = [dict(fold=int(f), src_file=str(s)) for f, s in zip(cache["folds"], cache["sources"])]
    result = {key: np.asarray(ids, dtype=int) for key, ids in
              split_rows(rows, train_folds, val_fold, test_fold, strict=strict).items()}
    if any(not len(ids) for ids in result.values()):
        raise ValueError("Empty role after source isolation")
    return result


def summarize_features(x):
    """[N,1,mel,time] -> [N,2*mel]: temporal mean then temporal std.

    Population standard deviation (ddof=0), float64 accumulation. This removes
    temporal ordering intentionally; it is a cheap diagnostic, not a CNN/RNN.
    """
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[1] != 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,1,mel,time] input")
    values = x[:, 0].astype(np.float64)
    return np.concatenate((values.mean(axis=-1), values.std(axis=-1)), axis=1)


def fit_standardization(train_features):
    train_features = np.asarray(train_features, dtype=np.float64)
    if train_features.ndim != 2 or not len(train_features) or not np.isfinite(train_features).all():
        raise ValueError("Expected nonempty finite training feature matrix")
    mean = train_features.mean(axis=0)
    std = np.maximum(train_features.std(axis=0), 1e-8)
    return mean, std


def _validate_labels(values, num_classes):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("Labels must be a numeric vector")
    if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or
            ((values < 0) | (values >= num_classes)).any()):
        raise ValueError("Labels must be integer class IDs in range")
    return values.astype(np.int64)


def metrics(targets, predictions, num_classes=50):
    targets = _validate_labels(targets, num_classes)
    predictions = _validate_labels(predictions, num_classes)
    result = compute_metrics(targets, predictions, num_classes)
    confusion = np.asarray(result["confusion_matrix"], dtype=np.int64)
    support, predicted = confusion.sum(1), confusion.sum(0)
    recall = np.divide(confusion.diagonal(), support, out=np.zeros(num_classes, dtype=float), where=support > 0)
    result.update(per_class_recall=recall.tolist(), per_class_support=support.tolist(),
                  zero_recall_classes=int((recall == 0).sum()), predicted_classes=int((predicted > 0).sum()))
    return result


def logistic_objective(parameters, x, y, l2, num_classes):
    """Mean multiclass NLL + l2/2 * ||W||²; intercept is unpenalized."""
    weights = parameters[:x.shape[1] * num_classes].reshape(x.shape[1], num_classes)
    bias = parameters[x.shape[1] * num_classes:]
    logits = x @ weights + bias
    normalizers = logsumexp(logits, axis=1)
    loss = float(np.mean(normalizers - logits[np.arange(len(y)), y]) + l2 * np.square(weights).sum() / 2)
    probabilities = np.exp(logits - normalizers[:, None])
    probabilities[np.arange(len(y)), y] -= 1
    probabilities /= len(y)
    gradient = np.concatenate(((x.T @ probabilities + l2 * weights).ravel(), probabilities.sum(axis=0)))
    return loss, gradient


def fit_logistic(x, y, l2, num_classes=50, maxiter=1000):
    x = np.asarray(x, dtype=np.float64)
    y = _validate_labels(y, num_classes)
    if len(x) != len(y) or not len(y) or l2 <= 0:
        raise ValueError("Invalid logistic training data or regularization")
    started = time.monotonic()
    initial = np.zeros((x.shape[1] + 1) * num_classes, dtype=np.float64)
    result = minimize(logistic_objective, initial, args=(x, y, l2, num_classes),
                      jac=True, method="L-BFGS-B",
                      options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6, maxls=30))
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise FloatingPointError("Logistic optimization became nonfinite")
    d = x.shape[1] * num_classes
    return result.x[:d].reshape(x.shape[1], num_classes), result.x[d:], dict(
        optimizer="scipy L-BFGS-B", iterations=int(result.nit), success=bool(result.success),
        message=str(result.message), objective=float(result.fun), seconds=time.monotonic() - started)


def score_logistic(x, y, weights, bias):
    logits = x @ weights + bias
    predicted = logits.argmax(1)
    result = metrics(y, predicted, weights.shape[1])
    result["loss"] = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]))
    return result, predicted


def baseline(args):
    out = new_output(args.output)
    (out / "baseline_source.py").write_bytes(Path(__file__).read_bytes())
    cache, digest, sidecar = validated_cache(args.cache)
    train_folds = tuple(int(f) for f in args.train_folds.split(","))
    indices = source_split(cache, train_folds, args.val_fold, args.test_fold, not args.official_folds)
    # Compute only train and validation summaries; no test feature summaries/predictions during development.
    train_ids, val_ids = indices["train"], indices["validation"]
    train_raw = summarize_features(cache["x"][train_ids])
    val_raw = summarize_features(cache["x"][val_ids])
    mean, std = fit_standardization(train_raw)
    train_x, val_x = (train_raw - mean) / std, (val_raw - mean) / std
    train_y, val_y = cache["y"][train_ids], cache["y"][val_ids]
    candidates, best, best_loss = [], None, float("inf")
    for l2 in L2_GRID:
        weights, bias, optimization = fit_logistic(train_x, train_y, l2, maxiter=args.maxiter)
        train_metrics, _ = score_logistic(train_x, train_y, weights, bias)
        validation_metrics, predictions = score_logistic(val_x, val_y, weights, bias)
        record = dict(l2=l2, optimization=optimization,
                      train={k: train_metrics[k] for k in ("accuracy", "macro_f1", "loss", "n")},
                      validation={k: validation_metrics[k] for k in ("accuracy", "macro_f1", "loss", "n")})
        candidates.append(record)
        save_predictions(out / f"validation_predictions_l2_{l2}.csv", cache["names"][val_ids], val_y, predictions)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best = (l2, weights, bias, validation_metrics, predictions, train_metrics)
        write_json(out / "search.json", candidates)
    l2, weights, bias, validation_metrics, predictions, train_metrics = best
    np.savez_compressed(out / "model.npz", weights=weights, bias=bias, mean=mean, std=std,
                        class_names=cache["class_names"])
    manifest = dict(baseline_version=BASELINE_VERSION, model="logmel_temporal_statistics_multinomial_logistic",
                    input_features=128, feature_definition="64 temporal means followed by 64 population standard deviations",
                    standardization="training summaries only", frontend=sidecar["frontend"],
                    cache_sha256=digest, model_sha256=sha256(out / "model.npz"),
                    metadata_sha256=sidecar.get("metadata_sha256"), source_policy="official" if args.official_folds else "source_strict",
                    train_folds=list(train_folds), val_fold=args.val_fold, test_fold=args.test_fold,
                    split={key: dict(n=len(ids), names=cache["names"][ids].tolist()) for key, ids in indices.items()},
                    selected_l2=l2, search_grid=list(L2_GRID), maxiter=args.maxiter,
                    selection="minimum validation cross-entropy; earliest grid tie", test_evaluated=False,
                    test_status_scope="snapshot at fitting; subsequent test access is recorded in evaluation_audit.jsonl",
                    parameters=int(weights.size + bias.size), contribution="agent-produced",
                    environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                    code_sha256=sha256(__file__))
    write_json(out / "model.json", manifest)
    write_json(out / "validation.json", validation_metrics)
    write_json(out / "train.json", train_metrics)
    save_predictions(out / "validation_predictions.csv", cache["names"][val_ids], val_y, predictions)
    print(json.dumps(dict(selected_l2=l2, validation={k: validation_metrics[k] for k in ("accuracy", "macro_f1", "loss", "n")},
                          train_accuracy=train_metrics["accuracy"], candidates=candidates), indent=2))


def evaluate_test(args):
    if not args.allow_test:
        raise ValueError("Test predictions require an explicit --allow-test after recipe freeze")
    artifact = Path(args.artifact)
    manifest = json.loads((artifact / "model.json").read_text())
    if manifest.get("baseline_version") != BASELINE_VERSION or manifest.get("model_sha256") != sha256(artifact / "model.npz"):
        raise ValueError("Baseline artifact version/hash mismatch")
    cache, digest, sidecar = validated_cache(args.cache)
    if digest != manifest["cache_sha256"] or sidecar["frontend"] != manifest["frontend"]:
        raise ValueError("Test cache/frontend differs from fitted baseline")
    out = new_output(args.output)
    audit = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), event="test_evaluation_started",
                 output=str(out.resolve()), cache_sha256=digest,
                 model_sha256=manifest["model_sha256"], manifest_sha256=sha256(artifact / "model.json"))
    with open(artifact / "evaluation_audit.jsonl", "a") as stream:
        stream.write(json.dumps(audit) + "\n")
    with np.load(artifact / "model.npz", allow_pickle=False) as stream:
        weights, bias, mean, std = (stream[key] for key in ("weights", "bias", "mean", "std"))
        ids = np.flatnonzero(cache["folds"] == manifest["test_fold"])
        x = (summarize_features(cache["x"][ids]) - mean) / std
        result, predictions = score_logistic(x, cache["y"][ids], weights, bias)
    result.update(cache_sha256=digest, model_sha256=manifest["model_sha256"],
                  baseline_manifest_sha256=sha256(artifact / "model.json"), source_policy=manifest["source_policy"],
                  test_fold=manifest["test_fold"], interpretation="one held-out fold; fixed-model result, not five-fold CV")
    write_json(out / "test.json", result)
    save_predictions(out / "test_predictions.csv", cache["names"][ids], cache["y"][ids], predictions)
    audit.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), event="test_evaluation_completed",
                 accuracy=result["accuracy"], macro_f1=result["macro_f1"], n=result["n"])
    with open(artifact / "evaluation_audit.jsonl", "a") as stream:
        stream.write(json.dumps(audit) + "\n")
    print(json.dumps({key: result[key] for key in ("accuracy", "macro_f1", "n")}, indent=2))


