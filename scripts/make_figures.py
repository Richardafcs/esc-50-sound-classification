"""Render static research figures from saved ESC-50 results, without inference.

Run from any directory: ``python scripts/make_figures.py``. Only add
``--include-cv`` after the saved cross_validation.json contains all ten rows.
The default run never reads that potentially incomplete file or a remote runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MODELS = ("cnn_freq", "crnn_freq", "cnn_freq_matched", "cnn_freq_token")
SEEDS = (101, 202, 303)
NAMES = {"cnn_freq": "CNN", "crnn_freq": "CNN + GRU",
         "cnn_freq_matched": "Pooled MLP", "cnn_freq_token": "Tokenwise MLP"}
BLUE, ORANGE = "#4477AA", "#CC6677"


def read_json(path):
    return json.loads(Path(path).read_text())


def metric(path, expected_n):
    value = read_json(path)
    if value.get("n") != expected_n:
        raise ValueError(f"Unexpected sample count in {path}")
    matrix = np.asarray(value["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (50, 50) or matrix.sum() != expected_n:
        raise ValueError(f"Invalid confusion matrix in {path}")
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(50), where=denominator > 0)
    if not (np.isclose(matrix.trace() / expected_n, value["accuracy"], atol=1e-12) and
            np.isclose(f1.mean(), value["macro_f1"], atol=1e-12)):
        raise ValueError(f"Confusion matrix and reported metrics disagree in {path}")
    return value


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "grid.color": "#D8D8D8",
                         "grid.alpha": .55, "axes.axisbelow": True})


def finish(fig, output, name, footer, svg=False, bottom=.17, top=.86):
    # Let labels determine left/right spacing, then reserve exactly the title
    # and caption bands instead of applying their margins twice.
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=bottom, top=top)
    fig.text(.5, .035, footer, ha="center", va="bottom", fontsize=9, color="#444444", linespacing=1.4)
    fig.savefig(output / f"{name}.png", dpi=180)
    if svg:
        fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def comparison(root, output, svg):
    study = root / 'results/historical/final_study'
    records = []
    for model in MODELS:
        validation, holdout = [], []
        for seed in SEEDS:
            tag = f"{model}_seed{seed}_60ep"
            validation.append(metric(study / f"runs/{tag}/validation.json", 398))
            holdout.append(metric(study / f"results/holdout/{tag}/test.json", 400))
        parameters = read_json(study / f"runs/{model}_seed101_60ep/run.json")["parameters"]
        records.append(dict(model=model, label=NAMES[model], seeds=list(SEEDS), parameters=parameters,
                            validation={key: [r[key] for r in validation] for key in ("accuracy", "macro_f1")},
                            holdout={key: [r[key] for r in holdout] for key in ("accuracy", "macro_f1")}))
    baseline_validation = metric(study / 'results/statistics_baseline/validation.json', 398)
    baseline_holdout = metric(study / "results/holdout/statistics_baseline/test.json", 400)
    baseline_manifest = read_json(study / 'results/statistics_baseline/model.json')
    records.append(dict(model="statistics", label="Statistics", seeds=[], parameters=baseline_manifest["parameters"],
                        validation={key: [baseline_validation[key]] for key in ("accuracy", "macro_f1")},
                        holdout={key: [baseline_holdout[key]] for key in ("accuracy", "macro_f1")}))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("From-scratch classifiers: development and reserved fold 5", fontsize=15, y=.985)
    x = np.arange(len(records)); width = .34
    labels = [f"{r['label']}\n{r['parameters']/1000:.0f}k params" if r["parameters"] >= 10000
              else f"{r['label']}\n{r['parameters']/1000:.1f}k params" for r in records]
    for ax, key, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1 across 50 classes")):
        for offset, split, color in ((-width/2, "validation", BLUE), (width/2, "holdout", ORANGE)):
            values = [np.asarray(r[split][key]) * 100 for r in records]
            means = np.array([v.mean() for v in values])
            # A deterministic single fit has no across-seed SD estimate. NaN
            # omits its whisker/cap instead of depicting a measured zero SD.
            deviations = np.array([v.std(ddof=1) if len(v) > 1 else np.nan for v in values])
            bars = ax.bar(x + offset, means, width, yerr=deviations, color=color,
                          edgecolor="white", linewidth=.6, capsize=3,
                          error_kw={"elinewidth": 1., "capthick": 1.})
            for bar, mean, deviation in zip(bars, means, deviations):
                ax.text(bar.get_x() + bar.get_width()/2, mean + (deviation if np.isfinite(deviation) else 0) + 2, f"{mean:.2f}",
                        ha="center", va="bottom", fontsize=8.4)
        ax.set(title=title, ylabel="Percent", ylim=(0, 100), xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Validation fold 4 (398 clips)"),
                        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Holdout fold 5 (400 clips)")],
               loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, frameon=False)
    finish(fig, output, "model_comparison",
           "Neural bars: mean of seeds 101, 202, 303; whiskers: sample SD across seeds, not confidence intervals. Statistics: one deterministic fitted model.\n"
           "All seeds reuse the same clips. These fold-5 results are not official five-fold benchmark scores.", svg, bottom=.19, top=.80)
    return records


def learning_curves(root, output, svg):
    study = root / 'results/historical/final_study'
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    fig.suptitle("Primary seed 101: training and validation learning curves", fontsize=15, y=.99)
    summary = []
    for row, model in enumerate(("cnn_freq", "crnn_freq")):
        folder = study / f"runs/{model}_seed101_60ep"
        history = read_json(folder / "history.json")
        run = read_json(folder / "run.json")
        selected = min(history, key=lambda r: r["validation"]["loss"])["epoch"]
        if run["best_epoch"] != selected or any("train_evaluation" not in r for r in history):
            raise ValueError(f"Incomplete or inconsistent learning-curve evidence: {folder}")
        epochs = [r["epoch"] for r in history]
        for ax, key in zip(axes[row], ("loss", "accuracy")):
            multiplier = 100 if key == "accuracy" else 1
            ax.plot(epochs, [r["train_evaluation"][key] * multiplier for r in history], color=BLUE, lw=1.8)
            ax.plot(epochs, [r["validation"][key] * multiplier for r in history], color=ORANGE, lw=1.8)
            ax.axvline(selected, color="#777777", linestyle=":", lw=1.4)
            ax.set(title=f"{NAMES[model]} · selected epoch {selected}", xlim=(1, max(epochs)),
                   ylabel="Cross-entropy (nats)" if key == "loss" else "Accuracy (%)",
                   xlabel="Epoch" if row == 1 else "")
            ax.set_ylim(bottom=0)
            if key == "accuracy":
                ax.set_ylim(0, 100)
            ax.grid(alpha=.4)
        summary.append(dict(model=model, selected_epoch=selected, epochs=max(epochs)))
    fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=2, label="Training clips · evaluation mode"),
                        Line2D([0], [0], color=ORANGE, lw=2, label="Validation clips"),
                        Line2D([0], [0], color="#777777", ls=":", label="Selected checkpoint")],
               loc="upper center", bbox_to_anchor=(.5, .95), ncol=3, frameon=False, fontsize=9)
    finish(fig, output, "learning_curves",
           "Both curves are measured at epoch end in evaluation mode; online optimization loss is excluded. Train: 1,200 clips; validation: 398 clips.\n"
           "Checkpoint rule: earliest minimum validation cross-entropy. These are development curves for one seed, not holdout learning curves.",
           svg, bottom=.18, top=.86)
    return summary


def synthetic_stress(root, output, svg):
    study = root / 'results/historical/final_study/results/stress/test'
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    fig.suptitle("Synthetic corruption sensitivity on reserved fold 5", fontsize=15, y=.99)
    summary = {}
    for model, color, marker in (("cnn_freq", BLUE, "o"), ("crnn_freq", ORANGE, "s")):
        data = read_json(study / f"{model}_seed101_60ep/stress.json")
        if data["split"] != "test":
            raise ValueError("Expected frozen holdout stress measurements")
        conditions = data["conditions"]
        if any(v["n"] != 400 for v in conditions.values()):
            raise ValueError("Stress conditions use inconsistent sample counts")
        clean = conditions["clean"]["accuracy"] * 100
        noise = [conditions[f"snr_{snr}"]["accuracy"] * 100 for snr in (0, 10, 20)]
        gain = [conditions["gain_-6"]["accuracy"] * 100, clean, conditions["gain_6"]["accuracy"] * 100]
        axes[0].plot((0, 10, 20), noise, color=color, marker=marker, lw=2, label=NAMES[model])
        axes[0].axhline(clean, color=color, ls="--", lw=1.2, alpha=.75)
        axes[1].plot((-6, 0, 6), gain, color=color, marker=marker, lw=2, label=NAMES[model])
        summary[model] = {key: dict(accuracy=value["accuracy"], macro_f1=value["macro_f1"], n=value["n"])
                          for key, value in conditions.items()}
    axes[0].set(title="White Gaussian noise\nDashed lines show clean accuracy", xlabel="Signal-to-noise ratio (dB; higher = less noise)",
                xticks=(0, 10, 20), ylabel="Accuracy (%)", ylim=(0, 100))
    axes[1].set(title="Amplitude gain without peak clipping", xlabel="Amplitude change (dB)",
                xticks=(-6, 0, 6), xticklabels=("−6", "0 · clean", "+6"), ylabel="Accuracy (%)", ylim=(0, 100))
    for ax in axes:
        ax.grid(alpha=.4)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].legend(frameon=False, loc="upper left")
    finish(fig, output, "synthetic_stress",
           "Seed 101 only; same 400 holdout clips. One deterministic noise realization per clip/condition (noise seed 2026); no uncertainty bars.\n"
           "Synthetic corruption sensitivity does not establish real-world acoustic robustness. Clean reference: CNN 61.5%, CNN + GRU 62.0%.",
           svg, bottom=.23, top=.80)
    return summary


def top_confusions(root, output, svg):
    study = root / 'results/historical/final_study/results/holdout'
    with open(root / "data/ESC-50/meta/esc50.csv", newline="") as stream:
        classes = {int(row["target"]): row["category"].replace("_", " ") for row in csv.DictReader(stream)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.7))
    fig.suptitle("Largest named class confusions in primary holdout predictions", fontsize=15, y=.99)
    summary = {}
    for ax, model, color in zip(axes, ("cnn_freq", "crnn_freq"), (BLUE, ORANGE)):
        data = metric(study / f"{model}_seed101_60ep/test.json", 400)
        matrix = np.asarray(data["confusion_matrix"])
        if not np.equal(matrix.sum(axis=1), 8).all():
            raise ValueError("Expected eight held-out clips per true class")
        pairs = sorted(((int(matrix[i, j]), i, j) for i in range(50) for j in range(50)
                        if i != j and matrix[i, j] > 0), key=lambda item: (-item[0], item[1], item[2]))[:12]
        labels = [f"{classes[i]} → {classes[j]}" for _, i, j in pairs]
        values = [value for value, _, _ in pairs]
        y = np.arange(len(pairs))
        ax.barh(y, values, color=color, height=.68)
        for index, value in enumerate(values):
            ax.text(value + .1, index, f"{value}/8", va="center", fontsize=9)
        zeros = [classes[i] for i in range(50) if matrix[i, i] == 0]
        ax.set(title=f"{NAMES[model]} · seed 101\nZero recall: {', '.join(zeros) if zeros else 'none'}",
               yticks=y, yticklabels=labels, xlim=(0, 8.8), xticks=(0, 2, 4, 6, 8),
               xlabel="Misclassified clips (8 clips per true class)")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=.4)
        summary[model] = dict(zero_recall= zeros,
                             largest_confusions=[dict(true=classes[i], predicted=classes[j], clips=value, true_class_support=8)
                                                 for value, i, j in pairs])
    finish(fig, output, "top_confusions",
           "Largest off-diagonal counts, described after evaluation; they were not used to retune the frozen models. A true class may appear in more than one row.\n"
           "Eight clips per class give coarse estimates. These errors do not characterize every real-world recording condition.",
           svg, bottom=.19, top=.84)
    return summary


def order_probe(root, output, svg):
    study = root / 'results/historical/final_study/results/order/test'
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    fig.suptitle("Encoder-token order probe: primary checkpoints on fold 5", fontsize=15, y=.99)
    summary = {}
    for position, model in enumerate(MODELS):
        data = read_json(study / f"{model}_seed101_60ep/order_ablation.json")
        trials = data["trials"]
        if len(trials) != 5 or data["split"] != "test":
            raise ValueError("Order figure needs five saved holdout permutations per primary checkpoint")
        changes = np.array([(trial["accuracy"] - data["clean"]["accuracy"]) * 100 for trial in trials])
        switched = np.array([trial["changed_predictions"] / 400 * 100 for trial in trials])
        color = ORANGE if model == "crnn_freq" else BLUE
        jitter = np.linspace(-.08, .08, len(trials))
        for ax, values in zip(axes, (changes, switched)):
            ax.scatter(position + jitter, values, color=color, s=30, alpha=.8, zorder=3)
            ax.plot([position - .15, position + .15], [values.mean()] * 2, color=color, lw=3, zorder=4)
        summary[model] = dict(clean_accuracy=data["clean"]["accuracy"],
                             permutation_accuracy_change_pp=changes.tolist(), changed_prediction_percent=switched.tolist())
    axes[0].axhline(0, color="#777777", lw=1, ls="--")
    axes[0].set(title="Accuracy change from each model's clean result", ylabel="Accuracy change (percentage points)")
    axes[1].set(title="Prediction changes despite similar accuracy", ylabel="Changed predictions (% of 400 clips)", ylim=(0, 15))
    for ax in axes:
        ax.set(xticks=np.arange(len(MODELS)), xticklabels=[NAMES[m] for m in MODELS], xlim=(-.45, len(MODELS) - .55))
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=.4)
    finish(fig, output, "order_probe",
           "Points: five fixed encoder-token permutations; horizontal marks: mean. One seed-101 model per architecture, not repeated-training intervals.\n"
           "Permutation probes distribution-shift dependence; it does not prove useful recurrent processing.\n"
           "Nonrecurrent heads preserve exact aggregation invariance.",
           svg, bottom=.2, top=.85)
    return summary


def cross_validation(root, output, svg):
    """Explicit later invocation only; reject partial or duplicate fold results."""
    path = root / 'results/historical/final_study/results/cross_validation.json'
    rows = read_json(path)
    expected = {(model, fold) for model in ("cnn_freq", "crnn_freq") for fold in range(1, 6)}
    if not isinstance(rows, list) or len(rows) != 10 or {(r["model"], r["test_fold"]) for r in rows} != expected:
        raise ValueError("CV figure requires all ten distinct model-by-test-fold rows; no partial figure is produced")
    for row in rows:
        saved = metric(root / f"results/historical/final_study/results/holdout/{row['tag']}/test.json", 400)
        if not all(np.isclose(saved[key], row[key], atol=1e-12) for key in ("accuracy", "macro_f1")):
            raise ValueError("CV summary differs from its underlying saved test metrics")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    fig.suptitle("Fixed-recipe five-fold characterization · seed 101", fontsize=15, y=.99)
    summary = {}
    for model, color, marker in (("cnn_freq", BLUE, "o"), ("crnn_freq", ORANGE, "s")):
        ordered = sorted((r for r in rows if r["model"] == model), key=lambda r: r["test_fold"])
        for ax, key, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1 across 50 classes")):
            values = np.array([r[key] for r in ordered]) * 100
            if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
                raise ValueError("Invalid CV metric")
            ax.plot(range(1, 6), values, color=color, marker=marker, lw=1.8,
                    label=f"{NAMES[model]} (mean {values.mean():.1f}%)")
            ax.set(title=title, xlabel="Outer test fold", ylabel="Percent", xticks=range(1, 6), ylim=(0, 100))
            ax.grid(alpha=.4)
            ax.legend(frameon=False, loc="lower left")
        summary[model] = ordered
    finish(fig, output, "cross_validation",
           "Source-strict rotating folds; one seed. Folds 1–4 already informed development: this is not nested model selection or five untouched confirmations.\n"
           "Fold scores share training data and are dependent; no confidence intervals are shown.", svg, bottom=.25, top=.86)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--svg", action="store_true", help="Also write SVG alongside each PNG")
    parser.add_argument("--include-cv", action="store_true", help="Read completed ten-row CV evidence and render its figure")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / 'reports/figures/baseline_20261028'
    output.mkdir(parents=True, exist_ok=True)
    style()
    data = dict(comparison=comparison(root, output, args.svg), learning_curves=learning_curves(root, output, args.svg),
                synthetic_stress=synthetic_stress(root, output, args.svg), top_confusions=top_confusions(root, output, args.svg),
                order_probe=order_probe(root, output, args.svg),
                scope="Derived from saved artifacts only; no training, inference, tuning, audio loading or remote-runtime access")
    if args.include_cv:
        data["cross_validation"] = cross_validation(root, output, args.svg)
    else:
        data["cross_validation"] = "not read: --include-cv was not requested"
    (output / "figure_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Saved figures and numeric source summaries to {output}")


if __name__ == "__main__":
    main()
