"""ESC-50 statistical baseline, paired prediction audit, and static research plots.

All fitting defaults to source-isolated development folds. Final test evaluation
is a distinct command with an explicit unlock; development never predicts test.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

BASELINE_VERSION = 1
L2_GRID = (0.001, 0.01, 0.1)
EXPECTED_FRONTEND = dict(sample_rate=16000, seconds=5, n_fft=512, hop_length=256,
                         n_mels=64, mel_scale="htk", center=True, log="natural", floor=1e-10)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output must be empty to preserve existing evidence: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validated_cache(path):
    """Validate the actual cache digest, frontend and aligned array schema."""
    path = Path(path)
    sidecar = json.loads(Path(str(path) + ".json").read_text())
    digest = sha256(path)
    if sidecar.get("cache_sha256") != digest:
        raise ValueError("Cache digest differs from sidecar")
    if sidecar.get("frontend") != EXPECTED_FRONTEND:
        raise ValueError("Baseline requires the declared 16 kHz/64-mel frontend")
    with np.load(path, allow_pickle=False) as stream:
        cache = {key: stream[key].copy() for key in stream.files}
    needed = {"x", "y", "folds", "sources", "names", "class_names"}
    if not needed.issubset(cache):
        raise ValueError("Cache schema is incomplete")
    n = len(cache["y"])
    if cache["x"].shape != (n, 1, 64, 313) or not np.isfinite(cache["x"]).all():
        raise ValueError("Unexpected or nonfinite feature cache")
    for key in ("y", "folds", "sources", "names"):
        if cache[key].shape != (n,):
            raise ValueError(f"Misaligned {key} array")
    if len(set(cache["names"].tolist())) != n or len(cache["class_names"]) != 50:
        raise ValueError("Duplicate filenames or invalid class list")
    _validate_labels(cache["y"], 50)
    if not np.isin(cache["folds"], np.arange(1, 6)).all():
        raise ValueError("Invalid fold IDs")
    # The current dataset cache is accompanied by the original metadata. Verify
    # logical row alignment as well as the byte digest, rather than merely
    # assuming equal-length label/feature arrays refer to the same recordings.
    metadata_path = path.parent / "ESC-50" / "meta" / "esc50.csv"
    if metadata_path.exists():
        if sha256(metadata_path) != sidecar.get("metadata_sha256"):
            raise ValueError("Metadata digest differs from cache provenance")
        with open(metadata_path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != n:
            raise ValueError("Metadata row count differs from cache")
        for i, row in enumerate(rows):
            if (row["filename"] != str(cache["names"][i]) or
                    int(row["target"]) != int(cache["y"][i]) or
                    int(row["fold"]) != int(cache["folds"][i]) or
                    str(row["src_file"]) != str(cache["sources"][i]) or
                    row["category"] != str(cache["class_names"][int(cache["y"][i])])):
                raise ValueError(f"Metadata/cache alignment differs at row {i}")
    return cache, digest, sidecar


def source_split(cache, train_folds=(1, 2, 3), val_fold=4, test_fold=5, strict=True):
    train_folds = tuple(train_folds)
    if (not train_folds or len(set(train_folds)) != len(train_folds) or
            any(f not in range(1, 6) for f in (*train_folds, val_fold, test_fold)) or
            val_fold == test_fold or val_fold in train_folds or test_fold in train_folds):
        raise ValueError("Train/validation/test fold roles must be disjoint in 1..5")
    folds, sources = cache["folds"], cache["sources"].astype(str)
    train = np.flatnonzero(np.isin(folds, train_folds))
    val = np.flatnonzero(folds == val_fold)
    test = np.flatnonzero(folds == test_fold)
    if strict:
        heldout_sources = set(sources[val]) | set(sources[test])
        test_sources = set(sources[test])
        train = np.array([i for i in train if sources[i] not in heldout_sources], dtype=int)
        val = np.array([i for i in val if sources[i] not in test_sources], dtype=int)
    if not len(train) or not len(val) or not len(test):
        raise ValueError("Empty role after source isolation")
    return dict(train=train, validation=val, test=test)


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
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("Metrics need nonempty aligned predictions")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    tp = confusion.diagonal().astype(float)
    predicted = confusion.sum(0)
    support = confusion.sum(1)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=support + predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    return dict(accuracy=float(tp.sum() / len(targets)), macro_f1=float(f1.mean()),
                n=len(targets), confusion_matrix=confusion.tolist(),
                per_class_f1=f1.tolist(), per_class_recall=recall.tolist(),
                per_class_support=support.tolist(), zero_recall_classes=int((recall == 0).sum()),
                predicted_classes=int((predicted > 0).sum()),
                confusion_convention="rows=true, columns=predicted")


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


def save_predictions(path, names, targets, predictions):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "target", "prediction"))
        writer.writerows(zip(names, targets, predictions))


def baseline(args):
    out = new_output(args.output)
    (out / "diagnostics_source.py").write_bytes(Path(__file__).read_bytes())
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


def read_prediction_csv(path):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"filename", "target", "prediction"}.issubset(rows[0]):
        raise ValueError("Expected filename,target,prediction CSV")
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prediction filenames")
    for row in rows:
        row["target"], row["prediction"] = int(row["target"]), int(row["prediction"])
    _validate_labels(np.array([r["target"] for r in rows]), 50)
    _validate_labels(np.array([r["prediction"] for r in rows]), 50)
    return rows


def grouped_bootstrap(left_correct, right_correct, groups, replicates=2000, seed=42):
    """Paired source-group percentile intervals for fixed model predictions.

    Resample groups with replacement, include all clips in each chosen group,
    and use clip-weighted accuracy in every replicate. This is descriptive
    sample-composition uncertainty, not seed/fold/search uncertainty or a test
    of a method selected on the same validation data.
    """
    left, right, groups = np.asarray(left_correct), np.asarray(right_correct), np.asarray(groups)
    if (left.ndim != 1 or left.shape != right.shape or left.shape != groups.shape or
            not len(left) or replicates < 1 or not np.isin(left, (0, 1)).all() or
            not np.isin(right, (0, 1)).all()):
        raise ValueError("Need aligned nonempty binary correctness and group arrays")
    unique, membership = np.unique(groups.astype(str), return_inverse=True)
    counts = np.bincount(membership)
    left_sum = np.bincount(membership, weights=left)
    right_sum = np.bincount(membership, weights=right)
    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 3))
    for i in range(replicates):
        sampled = rng.integers(0, len(unique), size=len(unique))
        denominator = counts[sampled].sum()
        lacc = left_sum[sampled].sum() / denominator
        racc = right_sum[sampled].sum() / denominator
        values[i] = lacc, racc, lacc - racc
    return dict(groups=len(unique), clips=len(left), replicates=replicates, seed=seed,
                sampling_unit="source ID; all its clips retained together", weighting="clip-weighted accuracy",
                interval="percentile 2.5%,97.5%", left_accuracy_interval=np.quantile(values[:, 0], [.025, .975]).tolist(),
                right_accuracy_interval=np.quantile(values[:, 1], [.025, .975]).tolist(),
                left_minus_right_interval=np.quantile(values[:, 2], [.025, .975]).tolist(),
                limitation="fixed-model descriptive sample-composition interval; excludes training, fold and model-selection uncertainty; not a confirmatory test on development data")


def paired_comparison(left_rows, right_rows, metadata, replicates=2000, seed=42):
    left = {row["filename"]: row for row in left_rows}
    right = {row["filename"]: row for row in right_rows}
    if not left or left.keys() != right.keys() or len(left) != len(left_rows) or len(right) != len(right_rows):
        raise ValueError("Prediction sets must contain identical unique filenames")
    metadata = {row["filename"]: row for row in metadata}
    names = sorted(left)
    if any(name not in metadata or left[name]["target"] != right[name]["target"] for name in names):
        raise ValueError("Labels/metadata disagree across predictions")
    if any(int(metadata[name]["target"]) != left[name]["target"] for name in names):
        raise ValueError("Prediction target differs from metadata")
    l = np.array([left[n]["prediction"] == left[n]["target"] for n in names])
    r = np.array([right[n]["prediction"] == right[n]["target"] for n in names])
    groups = np.array([str(metadata[n]["src_file"]) for n in names])
    result = dict(n=len(names), both_correct=int((l & r).sum()), left_only=int((l & ~r).sum()),
                  right_only=int((~l & r).sum()), both_wrong=int((~l & ~r).sum()),
                  left_accuracy=float(l.mean()), right_accuracy=float(r.mean()),
                  left_minus_right=float(l.mean() - r.mean()),
                  bootstrap=grouped_bootstrap(l, r, groups, replicates, seed))
    result["left"] = metrics([left[n]["target"] for n in names], [left[n]["prediction"] for n in names])
    result["right"] = metrics([right[n]["target"] for n in names], [right[n]["prediction"] for n in names])
    return result


def compare(args):
    out = new_output(args.output)
    left, right = read_prediction_csv(args.left), read_prediction_csv(args.right)
    with open(args.metadata, newline="") as stream:
        metadata = list(csv.DictReader(stream))
    result = paired_comparison(left, right, metadata, args.bootstrap, args.seed)
    result.update(left_path=str(args.left), right_path=str(args.right),
                  left_sha256=sha256(args.left), right_sha256=sha256(args.right),
                  metadata_sha256=sha256(args.metadata))
    write_json(out / "comparison.json", result)
    classes = {int(row["target"]): row["category"] for row in metadata}
    with open(out / "per_class.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("target", "category", "support", "left_recall", "right_recall", "left_f1", "right_f1"))
        for i in range(50):
            writer.writerow((i, classes.get(i, str(i)), result["left"]["per_class_support"][i],
                             result["left"]["per_class_recall"][i], result["right"]["per_class_recall"][i],
                             result["left"]["per_class_f1"][i], result["right"]["per_class_f1"][i]))
    plot_confusion(result["left"]["confusion_matrix"], out / "left_confusion.png", "Left model validation confusion")
    plot_confusion(result["right"]["confusion_matrix"], out / "right_confusion.png", "Right model validation confusion")
    print(json.dumps({k: result[k] for k in ("n", "left_accuracy", "right_accuracy", "left_minus_right", "left_only", "right_only", "bootstrap")}, indent=2))


def plot_confusion(confusion, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    confusion = np.asarray(confusion)
    support = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, support, out=np.zeros_like(confusion, dtype=float), where=support > 0)
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    ax.set(title=title, xlabel="Predicted class ID", ylabel="True class ID")
    fig.colorbar(image, ax=ax, label="Fraction within true class")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_histories(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for run in runs:
        run = Path(run)
        records = json.loads((run / "history.json").read_text())
        if not records:
            raise ValueError(f"Empty history: {run}")
        epochs = [r["epoch"] for r in records]
        label = run.name
        axes[0].plot(epochs, [r["training_loss"] for r in records], linestyle="--", label=label + " online train")
        axes[0].plot(epochs, [r["validation"]["loss"] for r in records], label=label + " validation")
        axes[1].plot(epochs, [r["validation"]["accuracy"] * 100 for r in records], label=label)
        axes[2].plot(epochs, [r["validation"]["macro_f1"] * 100 for r in records], label=label)
        key = next((k for k in ("train_evaluation", "training_eval", "training_evaluation") if k in records[0]), None)
        if key:
            axes[0].plot(epochs, [r[key]["loss"] for r in records], linestyle=":", label=label + " eval-mode train")
            axes[1].plot(epochs, [r[key]["accuracy"] * 100 for r in records], linestyle=":", label=label + " eval-mode train")
    for ax, title, ylabel in zip(axes, ("Loss (train modes labeled)", "Validation accuracy", "Validation macro-F1"),
                                ("Cross-entropy", "Accuracy (%)", "Macro-F1 (%)")):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=6)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("baseline")
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--output", default="results/statistics_baseline")
    p.add_argument("--train-folds", default="1,2,3")
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--test-fold", type=int, default=5)
    p.add_argument("--official-folds", action="store_true")
    p.add_argument("--maxiter", type=int, default=1000)
    p.set_defaults(func=baseline)
    p = sub.add_parser("evaluate-test")
    p.add_argument("--artifact", required=True)
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-test", action="store_true")
    p.set_defaults(func=evaluate_test)
    p = sub.add_parser("compare")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--metadata", default="data/ESC-50/meta/esc50.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=compare)
    p = sub.add_parser("plot")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=lambda args: plot_histories(args.runs, args.output))
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
