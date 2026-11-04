# scripts/build_error_case_atlas.py
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esc50_sound.data import read_metadata, split_rows
from esc50_sound.audio import read_audio, log_mel

SEEDS = (101, 202, 303)
CATEGORIES = ("clapping", "siren", "glass_breaking", "helicopter")


def load_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        name = row["filename"]
        if name in result:
            raise ValueError(f"Duplicate prediction: {name} in {path}")
        result[name] = (int(row["target"]), int(row["prediction"]))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path,
                   default=Path("results/matrix_20261028/execution/runs"))
    p.add_argument("--metadata-root", type=Path, default=Path("data/ESC-50"))
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("reports/followup_20261104/error_cases"))
    args = p.parse_args()

    metadata = read_metadata(args.metadata_root)
    val = [metadata[i] for i in split_rows(metadata)["validation"]]
    class_names = {int(r["target"]): r["category"] for r in metadata}
    val = sorted(val, key=lambda r: r["filename"])
    names = {r["filename"] for r in val}
    if len(val) != 398 or len(names) != 398:
        raise ValueError("Expected 398 distinct Source-Strict Fold4 clips")
    if len({r["src_file"] for r in val}) != 295:
        raise ValueError("Expected 295 validation sources")

    predictions = {}
    for model, prefix in (("CNN", "cnn"), ("CRNN", "crnn")):
        for seed in SEEDS:
            path = (args.runs / "B_architecture" /
                    f"B_{prefix}_freq_s{seed}" / "validation_predictions.csv")
            data = load_csv(path)
            if set(data) != names:
                raise ValueError(f"Prediction membership mismatch: {path}")
            predictions[model, seed] = data

    index = []
    for meta in val:
        name = meta["filename"]
        row = dict(filename=name, src_file=meta["src_file"],
                   category=meta["category"])
        target = int(meta["target"])
        for model in ("CNN", "CRNN"):
            correct = 0
            for seed in SEEDS:
                truth, pred = predictions[model, seed][name]
                if truth != target:
                    raise ValueError(f"Target mismatch: {name}, {model}, {seed}")
                row[f"{model}_s{seed}_pred"] = class_names[pred]
                correct += int(pred == target)
            row[f"{model}correctcount"] = correct
        row["netdifference"] = row["CRNNcorrectcount"] - row["CNNcorrectcount"]
        row["audioabsolutepath"] = str((args.audio_root / name).resolve())
        index.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "src_file", "category", "CNNcorrectcount",
              "CRNNcorrectcount", "netdifference"]
    fields += [f"{m}_s{s}_pred" for m in ("CNN", "CRNN") for s in SEEDS]
    fields += ["audioabsolutepath"]
    with (args.output / "case_index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index)

    selected = []
    for category in CATEGORIES:
        candidates = [r for r in index if r["category"] == category]
        if not candidates:
            raise ValueError(f"Missing category: {category}")
        reverse = category != "helicopter"
        selected.append(sorted(candidates, key=lambda r:
                        ((-1 if reverse else 1) * r["netdifference"],
                         r["filename"]))[0])

    waves = [read_audio(Path(r["audioabsolutepath"])) for r in selected]
    spectra = [log_mel(w).squeeze(0).numpy() for w in waves]
    low, high = min(float(m.min()) for m in spectra), max(float(m.max()) for m in spectra)
    fig, axes = plt.subplots(4, 2, figsize=(15, 13))
    for i, row in enumerate(selected):
        path = Path(row["audioabsolutepath"])
        audio, sr = waves[i].numpy(), 16000
        mel = spectra[i]
        duration = len(audio) / sr
        title = (f'{row["filename"]} | {row["category"]} | '
                 f'CNN {row["CNNcorrectcount"]}/3, '
                 f'CRNN {row["CRNNcorrectcount"]}/3')
        axes[i, 0].plot(np.arange(len(audio)) / sr, audio, lw=0.5)
        axes[i, 0].set(title=title, xlabel="Seconds", ylabel="Amplitude")
        image = axes[i, 1].imshow(mel, origin="lower", aspect="auto",
                          extent=[-.008, duration, -.5, 63.5], vmin=low, vmax=high, cmap="magma")
        axes[i, 1].set_xlim(0, duration)
        fig.colorbar(image, ax=axes[i, 1], label="ln(Mel power), relative units")
        axes[i, 1].set(title="Log-Mel spectrogram",
                        xlabel="Seconds", ylabel="Mel-band index")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"error_case_atlas.{ext}", dpi=160)
    plt.close(fig)

    lines = ["# RQ1: Real-audio error-case inspection", "",
             "![Waveforms and log-Mel spectrograms](error_case_atlas.png)", "",
             "| Category | Filename | CNN /3 | CRNN /3 | Difference | Local WAV |",
             "|---|---|---:|---:|---:|---|"]
    for r in selected:
        lines.append(f'| {r["category"]} | {r["filename"]} | '
                     f'{r["CNNcorrectcount"]} | {r["CRNNcorrectcount"]} | '
                     f'{r["netdifference"]:+d} | `{r["audioabsolutepath"]}` |')
    lines += ["", "Selection uses observed CRNN-minus-CNN correct counts: "
              "maximum for clapping, siren and glass_breaking; minimum for "
              "helicopter; filename breaks ties.",
              "", "These four exploratory selected samples are not representative "
              "class-level proof. No human listening, label-error finding, or "
              "causal time-region interpretation is claimed."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
