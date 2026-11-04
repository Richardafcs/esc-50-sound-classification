# scripts/run_temporal_order_probe.py
"""RQ1: Post-encoder temporal-order intervention, development Fold4 only."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esc50_sound.robustness import validation_rows, read_predictions
from esc50_sound.audio import read_audio, log_mel
from esc50_sound.models import build_model
from esc50_sound.prediction_errors import SourceBootstrap
from esc50_sound.evaluation import compute_metrics

SEEDS = (101, 202, 303)
MODELS = ("cnn_freq", "crnn_freq")
CASES = (
    "4-189830-A-22.wav",
    "4-90014-A-42.wav",
    "4-204121-A-39.wav",
    "4-193480-A-40.wav",
)


def permutations():
    base = torch.arange(39)
    result = {
        "identity": base,
        "reverse": base.flip(0),
        "roll19": torch.roll(base, 19),
    }
    for seed in (17, 29):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result[f"shuffle{seed}"] = torch.randperm(39, generator=generator)
    return result


def explicit_head(model, z, architecture):
    if architecture == "cnn_freq":
        return model.head(z.mean(3).flatten(1))
    h, _ = model.gru(z.permute(0, 3, 1, 2).flatten(2))
    return model.head(h.mean(1))


def write_csv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--runs", type=Path,
                        default=ROOT / "results/matrix_20261028/execution/runs/B_architecture")
    parser.add_argument("--metadata-root", type=Path,
                        default=ROOT / "data/ESC-50")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/temporal_20261105")
    args = parser.parse_args()
    if args.execute and args.audio_root is None:
        parser.error("--audio-root is required with --execute")

    torch.set_num_threads(2)
    perms = permutations()
    conditions = tuple(perms)
    protocol = {
        "study": "RQ1 post-encoder temporal-order probe",
        "population": "Source-Strict development Fold4; 398 clips, 295 sources",
        "architectures": list(MODELS),
        "fixed_training_seeds": list(SEEDS),
        "batch_size": 32,
        "device": "cpu",
        "intervention": "index_select on encoder output's last dimension",
        "encoder_shape": [None, 64, 8, 39],
        "permutations": {k: v.tolist() for k, v in perms.items()},
        "hypotheses": [
            "CNN mean pooling is permutation invariant in real arithmetic. Verify pool values in float64 (atol=rtol=1e-12), preserve and report float32 logit residuals, require identical CNN predictions.",
            "CRNN predictions may depend on post-encoder temporal order.",
            "Reversal and shuffling can reveal post-encoder order dependence.",
        ],
        "limits": [
            "Original encoder embeddings remain unchanged.",
            "Post-encoder permutations are not real-audio perturbations.",
            "Order dependence does not prove useful temporal recognition.",
            "Neither model robustness nor causality is established.",
            "Floating-point checks are not SMT proofs.",
            "Inference is restricted to development Fold4.",
            "All intervals condition on three fixed training seeds.",
        ],
        "bootstrap": {"unit": "src_file", "replicates": 4000,
                      "rng_seed": 20261105, "seed_aggregation": "fixed mean"},
    }

    if not args.execute:
        print("PREVIEW ONLY: no audio decoding or model inference.")
        print(json.dumps(protocol, indent=2))
        return

    rows = validation_rows(args.metadata_root)
    names = [r["filename"] for r in rows]
    sources = [r["src_file"] for r in rows]
    targets = [int(r["target"]) for r in rows]
    if len(rows) != 398 or len(set(names)) != 398:
        raise ValueError("Expected 398 unique validation filenames")
    if len(set(sources)) != 295:
        raise ValueError("Expected 295 original recording sources")
    if not set(CASES).issubset(names):
        raise ValueError(f"Missing selected cases: {set(CASES) - set(names)}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )

    # Decode every validation WAV exactly once.
    features = torch.stack([
        log_mel(read_audio(args.audio_root / name)) for name in names
    ])
    if features.shape != (398, 1, 64, 313):
        raise ValueError(f"Unexpected log-Mel shape: {tuple(features.shape)}")

    protocol["numerical_revision"] = "v2 after preserved float32 absolute-tolerance failure; see numerical_note.json when present"
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    predictions = {}
    metric_rows = []
    identity_receipts = []
    class_names = {int(r["target"]): r["category"] for r in rows}
    for architecture in MODELS:
        for seed in SEEDS:
            run = args.runs / f"B_{architecture}_s{seed}"
            reference = read_predictions(run / "validation_predictions.csv", rows)
            ck = torch.load(run / "best.pt", map_location="cpu",
                            weights_only=True)
            if ck["model_name"] != architecture:
                raise ValueError(f"Checkpoint architecture mismatch: {run}")
            model = build_model(ck["model_name"])
            model.load_state_dict(ck["model"], strict=True)
            model.eval()

            x_all = (features - ck["mean"]) / ck["std"]
            identity_max_error = 0.0
            pool64_max_error = 0.0
            logits = {condition: [] for condition in conditions}
            with torch.inference_mode():
                for start in range(0, 398, 32):
                    x = x_all[start:start + 32]
                    z = model.encoder(x)
                    if tuple(z.shape[1:]) != (64, 8, 39):
                        raise ValueError(f"Unexpected encoder shape: {z.shape}")

                    direct = model(x)
                    identity = explicit_head(model, z, architecture)
                    error = (direct - identity).abs().max().item()
                    identity_max_error = max(identity_max_error, error)
                    if error > 1e-5:
                        raise RuntimeError(
                            f"Identity head mismatch: {architecture} "
                            f"seed={seed}, maxdiff={error}"
                        )

                    for condition, indices in perms.items():
                        changed_z = z.index_select(-1, indices)
                        output = explicit_head(model, changed_z, architecture)
                        if architecture == "cnn_freq":
                            pooled64 = z.double().mean(3)
                            changed64 = changed_z.double().mean(3)
                            pool64_max_error = max(pool64_max_error, float((changed64-pooled64).abs().max()))
                            if not torch.allclose(pooled64, changed64, atol=1e-12, rtol=1e-12):
                                raise RuntimeError("Mean-pooling permutation control failed in float64")
                        logits[condition].append(output.cpu())

            logits = {k: torch.cat(v) for k, v in logits.items()}
            baseline = logits["identity"]
            baseline_preds = baseline.argmax(1).tolist()
            if baseline_preds != list(reference):
                bad = [names[i] for i in range(398)
                       if baseline_preds[i] != reference[i]]
                raise RuntimeError(
                    f"Saved clean prediction mismatch: {architecture}, "
                    f"seed={seed}, first examples={bad[:5]}"
                )

            identity_receipts.append(dict(model=architecture, seed=seed,
                identity_head_max_abs_logit_difference=identity_max_error,
                saved_clean_predictions_matched=len(rows),
                pool64_max_abs_difference=pool64_max_error if architecture == "cnn_freq" else None))
            for condition in conditions:
                output = logits[condition]
                preds = output.argmax(1).tolist()
                predictions[architecture, seed, condition] = preds
                write_csv(
                    args.output / f"{architecture}_s{seed}" / f"{condition}.csv",
                    ["filename", "target", "prediction"],
                    [dict(filename=n, target=t, prediction=p)
                     for n, t, p in zip(names, targets, preds)],
                )
                m = compute_metrics(targets, preds)
                differences = (output - baseline).abs()
                if architecture == "cnn_freq" and preds != baseline_preds:
                    raise RuntimeError("CNN prediction changed under encoder-token permutation")
                top2 = baseline.topk(2, dim=1).values
                minimum_margin = float((top2[:,0]-top2[:,1]).min())
                metric_rows.append({
                    "model": architecture,
                    "seed": seed,
                    "condition": condition,
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "changedpredcount": sum(a != b for a, b in
                                             zip(preds, baseline_preds)),
                    "meanabslogitchange": differences.mean().item(),
                    "maxabslogitchange": differences.max().item(),
                    "minimum_identity_top2_margin": minimum_margin,
                    "all_argmax_margin_bounds_hold": bool(((top2[:,0]-top2[:,1]) > 2*differences.amax(dim=1)).all()),
                })

    metric_fields = ["model", "seed", "condition", "accuracy", "macro_f1",
                     "changedpredcount", "meanabslogitchange",
                     "maxabslogitchange", "minimum_identity_top2_margin",
                     "all_argmax_margin_bounds_hold"]
    write_csv(args.output / "metrics.csv", metric_fields, metric_rows)

    class_rows = []
    for category in sorted(set(r["category"] for r in rows)):
        indices = [i for i, r in enumerate(rows) if r["category"] == category]
        for architecture in MODELS:
            for condition in conditions:
                accuracy = np.mean([
                    predictions[architecture, seed, condition][i] == targets[i]
                    for seed in SEEDS for i in indices
                ])
                class_rows.append({
                    "category": category, "model": architecture,
                    "condition": condition,
                    "meanaccuracy3seeds": float(accuracy),
                    "uniqueclips": len(indices),
                })
    write_csv(args.output / "class_results.csv",
              ["category", "model", "condition",
               "meanaccuracy3seeds", "uniqueclips"], class_rows)

    effects = {}
    bootstrap = SourceBootstrap(sources, 4000, 20261105)
    for architecture in MODELS:
        for condition in conditions[1:]:
            delta = np.column_stack([
                np.array([
                    int(predictions[architecture, seed, condition][i] == targets[i])
                    - int(predictions[architecture, seed, "identity"][i] == targets[i])
                    for i in range(398)
                ], dtype=float)
                for seed in SEEDS
            ])
            effects[f"{architecture}/{condition}"] = bootstrap.summarize(delta)

    summary = {
        "population": {"clips": 398, "sources": 295,
                       "fixed_seeds": list(SEEDS)},
        "effect_definition": "condition minus identity, accuracy pp",
        "bootstrap": protocol["bootstrap"],
        "effects": effects,
        "identity_receipts": identity_receipts,
        "torch_version": str(torch.__version__),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )

    labels = list(conditions[1:])
    crnn = [effects[f"crnn_freq/{c}"] for c in labels]
    means = [e["delta_accuracy_pp"] for e in crnn]
    lows = [e["source_bootstrap_low_pp"] for e in crnn]
    highs = [e["source_bootstrap_high_pp"] for e in crnn]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, means, width=0.65)
    ax.vlines(positions, lows, highs, color="black", linewidth=1.5)
    for position, low, high in zip(positions, lows, highs):
        ax.hlines([low, high], position - 0.12, position + 0.12,
                  color="black", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Accuracy change vs identity (percentage points)")
    ax.set_title("CRNN post-encoder order probe | 3 fixed seeds")
    ax.text(0.01, 0.01, "95% source-bootstrap intervals; conditional on fixed seeds",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"crnn_accuracy_change.{extension}", dpi=180)
    plt.close(fig)

    report = [
        "# RQ1 — Post-encoder temporal-order probe", "",
        "Source-Strict development Fold4: 398 clips, 295 recording sources.",
        "Three fixed training seeds: 101, 202, 303. CPU inference only.",
        "",
        "![CRNN accuracy changes](crnn_accuracy_change.png)", "",
        "## Fixed-seed mean effects and conditional intervals", "",
        "| Model | Condition | Delta (pp) | 95% source-bootstrap CI (pp) |",
        "|---|---|---:|---|",
    ]
    for architecture in MODELS:
        for condition in conditions[1:]:
            e = effects[f"{architecture}/{condition}"]
            report.append(
                f'| {architecture} | {condition} | '
                f'{e["delta_accuracy_pp"]:+.3f} | '
                f'[{e["source_bootstrap_low_pp"]:.3f}, '
                f'{e["source_bootstrap_high_pp"]:.3f}] |'
            )

    report += [
        "", "## Four previously selected exploratory cases", "",
        "Each prediction cell lists category names in seed order 101/202/303.",
        "",
        "| Filename | Model | Identity | Reverse | Roll19 | Shuffle17 | Shuffle29 |",
        "|---|---|---|---|---|---|---|",
    ]
    for filename in CASES:
        i = names.index(filename)
        for architecture in MODELS:
            values = [
                " / ".join(class_names[predictions[architecture, s, c][i]] for s in SEEDS)
                for c in conditions
            ]
            report.append(
                f'| {filename} | {architecture} | ' + " | ".join(values) + " |"
            )

    report += [
        "", "## Interpretation boundaries", "",
        "Permutations modify the order of existing encoder embeddings only.",
        "CNN pooling is compared in float64 (atol=rtol=1e-12); original float32 residuals and prediction margins are retained in metrics.csv. See numerical_note.json for the preserved first-attempt failure.",
        "CRNN changes may reveal sensitivity to post-encoder temporal order,",
        "but do not prove useful temporal recognition or a causal mechanism.",
        "The four cases are exploratory, not representative class-level evidence.",
        "Intervals resample original recording sources and average the same",
        "three fixed seeds; they do not quantify training-seed population uncertainty.",
        "No model robustness, label-error, or causal claim is established.",
        "No retraining, architecture modification, or held-out test inference.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print(f"Completed RQ1 temporal-order probe: {args.output}")


if __name__ == "__main__":
    main()
