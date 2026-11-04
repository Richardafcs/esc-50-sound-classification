#!/usr/bin/env python3
"""Offline RQ3 feature-convergence analysis. No training or inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONFIGS = {
    "sr16_fmax8": "16 kHz / fmax 8 kHz",
    "sr32_fmax8": "32 kHz / fmax 8 kHz",
    "sr32_fmax16": "32 kHz / fmax 16 kHz",
}
FEATURES = ("logmel64", "logmel32", "mfcc40", "loglinear64")
SEEDS = (101, 202, 303)
KEYS = ("epoch", "train_loss", "validation_accuracy",
        "validation_macro_f1", "validation_loss", "learning_rate")
METRICS = ("best_epoch", "selected_accuracy", "selected_macro_f1",
           "selected_validation_loss", "loss_delta_e60_minus_e50",
           "accuracy_delta_pp_e60_minus_e50")


def load_history(path):
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read valid history JSON: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != 60:
        raise ValueError(f"{path}: expected exactly 60 epoch rows")
    for epoch, row in enumerate(rows, 1):
        if not isinstance(row, dict) or any(k not in row for k in KEYS):
            raise ValueError(f"{path}: missing required keys at epoch {epoch}")
        if row["epoch"] != epoch:
            raise ValueError(f"{path}: expected ordered epoch {epoch}")
        try:
            values = [float(row[k]) for k in KEYS[1:]]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: nonnumeric value at epoch {epoch}") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: nonfinite value at epoch {epoch}")
    return rows


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sample_sd": float(a.std(ddof=1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path,
                        default=Path("results/matrix_20261028/execution/runs/"
                                     "D_feature_sampling/main/cnn"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/followup_20261104/convergence"))
    args = parser.parse_args()
    runs, curves = [], {}

    for config in CONFIGS:
        for feature in FEATURES:
            case = f"{config}_{feature}"
            for seed in SEEDS:
                path = args.runs / case / f"seed{seed}" / "history.json"
                history = load_history(path)
                best = min(history, key=lambda r: r["validation_loss"])
                record = {
                    "case": case, "config": config, "feature": feature,
                    "seed": seed, "best_epoch": best["epoch"],
                    "selected_accuracy": best["validation_accuracy"],
                    "selected_macro_f1": best["validation_macro_f1"],
                    "selected_validation_loss": best["validation_loss"],
                    "latebest": int(best["epoch"] >= 51),
                    "loss_delta_e60_minus_e50":
                        history[59]["validation_loss"] - history[49]["validation_loss"],
                    "accuracy_delta_pp_e60_minus_e50":
                        100 * (history[59]["validation_accuracy"]
                               - history[49]["validation_accuracy"]),
                }
                runs.append(record)
                curves[(config, feature, seed)] = np.array(
                    [r["validation_loss"] for r in history], dtype=float)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)

    lookup = {(r["config"], r["feature"], r["seed"]): r for r in runs}
    summary = {
        "total_runs": len(runs),
        "total_latebest": sum(r["latebest"] for r in runs),
        "latebest_definition": "selected best_epoch >= 51",
        "selection": "minimum validation_loss; earliest epoch on ties",
        "accuracy_unit": "fraction; differences expressed in percentage points",
        "per_case": {},
        "paired_logmel32_minus_logmel64": {},
    }
    for config in CONFIGS:
        for feature in FEATURES:
            case = f"{config}_{feature}"
            subset = [lookup[(config, feature, s)] for s in SEEDS]
            summary["per_case"][case] = {
                "latebest_count": sum(r["latebest"] for r in subset),
                **{m: stats([r[m] for r in subset]) for m in METRICS},
            }
        deltas = {
            str(s): 100 * (lookup[(config, "logmel32", s)]["selected_accuracy"]
                           - lookup[(config, "logmel64", s)]["selected_accuracy"])
            for s in SEEDS
        }
        summary["paired_logmel32_minus_logmel64"][config] = {
            "by_seed_pp": deltas, "delta_pp": stats(list(deltas.values()))
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (config, label) in zip(axes, CONFIGS.items()):
        for feature, name in (("logmel32", "Log-Mel 32 bands"),
                              ("logmel64", "Log-Mel 64 bands")):
            mean = np.mean([curves[(config, feature, s)] for s in SEEDS], axis=0)
            ax.plot(range(1, 61), mean, label=name, linewidth=1.8)
        ax.set(title=label, xlabel="Training epoch (1–60)",
               ylabel="Mean development validation loss (3 fixed seeds)")
        ax.set_xlim(1, 60)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("RQ3: Validation-loss trajectories — no confidence intervals")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"validation_loss_curves.{extension}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# RQ3 — Offline Feature Convergence",
        f"\nObserved late-best runs: **{summary['total_latebest']}/36** "
        "(previous review reported 30/36; actual histories are authoritative).",
        "\nCheckpoint: earliest minimum validation loss. "
        "Changes: epoch 60 minus epoch 50.",
        "\n| Case | Late/3 | Best epoch mean±SD | Selected accuracy % mean±SD "
        "| Loss Δ | Accuracy Δ (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case, item in summary["per_case"].items():
        b, a = item["best_epoch"], item["selected_accuracy"]
        loss = item["loss_delta_e60_minus_e50"]["mean"]
        acc = item["accuracy_delta_pp_e60_minus_e50"]["mean"]
        lines.append(f"| {case} | {item['latebest_count']} | "
                     f"{b['mean']:.1f}±{b['sample_sd']:.1f} | "
                     f"{100*a['mean']:.2f}±{100*a['sample_sd']:.2f} | "
                     f"{loss:+.4f} | {acc:+.2f} |")
    lines += ["\n## Paired selected-accuracy difference: Log-Mel 32 − 64 (pp)",
              "\n| Sampling config | Seed 101 | Seed 202 | Seed 303 | Mean±SD |",
              "|---|---:|---:|---:|---:|"]
    for config, item in summary["paired_logmel32_minus_logmel64"].items():
        d, s = item["by_seed_pp"], item["delta_pp"]
        lines.append(f"| {CONFIGS[config]} | {d['101']:+.2f} | "
                     f"{d['202']:+.2f} | {d['303']:+.2f} | "
                     f"{s['mean']:+.2f}±{s['sample_sd']:.2f} |")
    lines += [
        "\n## Interpretation",
        "Late-best checkpoint selection does not prove non-convergence. "
        "These are fixed 60-epoch budget comparisons, not asymptotic results.",
        "The 26,658-parameter pooled CNN differs from the main "
        "49,058-parameter CNN; architectural effects cannot be ignored.",
        "All validation results are development-fold observations. "
        "Three fixed-seed sample SDs are descriptive, not confidence intervals. "
        "No new training or test-fold evaluation was performed.",
    ]
    lines += ["\n![Development validation-loss curves](validation_loss_curves.png)",
              "\nLog-Mel 32 loss separates from Log-Mel 64 before the final ten epochs; "
              "this does not establish asymptotic convergence or superiority under the main architecture."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


if __name__ == "__main__":
    main()