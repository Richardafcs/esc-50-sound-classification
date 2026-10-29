"""Research prediction comparisons and plots; fitting delegates to the package.

Legacy baseline commands remain available for recorded reproduction commands.
Use ``python -m esc50_sound baseline`` for new fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from esc50_sound.baseline import (
    BASELINE_VERSION, L2_GRID, EXPECTED_FRONTEND, baseline, evaluate_test,
    fit_logistic, fit_standardization, logistic_objective, metrics, new_output,
    score_logistic, source_split, summarize_features, validated_cache, _validate_labels,
)
from esc50_sound.records import sha256, write_json


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
    plot_confusion(result["left"]["confusion_matrix"], out / "left_confusion.png", "Left model confusion")
    plot_confusion(result["right"]["confusion_matrix"], out / "right_confusion.png", "Right model confusion")
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
    p.add_argument("--output", default='results/runs/statistics_baseline')
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
