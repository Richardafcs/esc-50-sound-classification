
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/followup_20261104/robustness_runs.csv"
OUT = ROOT / "reports/final_20261105"

variants = ["cnn", "crnn", "noise_cnn"]
seeds = [101, 202, 303]
conditions = ["clean", "snr_20", "snr_10", "snr_0", "gain_-6", "gain_6"]
labels = {"cnn": "CNN", "crnn": "BiGRU",
          "noise_cnn": "Noise-trained CNN"}
colors = {"cnn": "#2878B5", "crnn": "#D68A29",
          "noise_cnn": "#27946B"}

data = {}
with INPUT.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    required = {"variant", "seed", "condition", "path", "correct", "accuracy"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing CSV fields: {required - set(reader.fieldnames or [])}")
    for row in reader:
        v, s, c = row["variant"], int(row["seed"]), row["condition"]
        if v not in variants or s not in seeds or c not in conditions:
            raise ValueError(f"Unexpected row: {v}, {s}, {c}")
        key = (v, s, c)
        if key in data:
            raise ValueError(f"Duplicate row: {key}")
        correct = int(row["correct"])
        accuracy = float(row["accuracy"])
        if not 0 <= correct <= 398 or not np.isfinite(accuracy):
            raise ValueError(f"Invalid metrics: {key}")
        if not np.isclose(accuracy, correct / 398, atol=0.001):
            raise ValueError(f"Accuracy/correct mismatch: {key}")
        data[key] = 100 * correct / 398

expected = {(v, s, c) for v in variants for s in seeds for c in conditions}
if set(data) != expected:
    raise ValueError(f"Missing combinations: {sorted(expected - set(data))}")

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True)
panels = [
    (["clean", "snr_20", "snr_10", "snr_0"],
     ["Clean", "20 dB", "10 dB", "0 dB"],
     "Added Gaussian noise", "Signal-to-noise ratio (SNR)"),
    (["gain_-6", "clean", "gain_6"],
     ["-6", "0", "+6"],
     "Gain perturbation", "Gain (dB)")
]

for ax, (conds, ticks, title, xlabel) in zip(axes, panels):
    x = np.arange(len(conds))
    for v in variants:
        values = np.array([[data[v, s, c] for c in conds] for s in seeds])
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        ax.errorbar(x, mean, yerr=sd, label=labels[v],
                    color=colors[v], marker="o", markersize=5,
                    linewidth=2, capsize=3, elinewidth=1.3)
    ax.set_xticks(x, ticks)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Accuracy (%)")
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc="upper center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 0.98))
fig.text(0.5, 0.035,
         "All 398 development clips; mean ± SD of three fixed fits, not confidence intervals",
         ha="center", fontsize=9)
fig.subplots_adjust(top=0.85, bottom=0.19, left=0.08, right=0.98, wspace=0.15)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"three_seed_robustness.{ext}",
                dpi=220, bbox_inches="tight")
plt.close(fig)
