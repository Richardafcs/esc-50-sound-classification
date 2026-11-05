"""Evaluate frozen ESC-50 systems on a fixed, recording-ID-disjoint FSD subset."""
from pathlib import Path
import argparse
import csv
import json
import math
import platform
import signal
import sys
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from esc50_sound.audio import log_mel
from esc50_sound.models import build_model
from scripts.run_transfer_followup import load_panns


def write_csv(path, records):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_recordings(rows, audio_root):
    """Partition all audio; retain valid sample counts for weighted aggregation."""
    waves, owners, weights, descriptions = [], [], [], []
    for index, row in enumerate(rows):
        audio, rate = sf.read(audio_root / row["filename"], dtype="float32", always_2d=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Nonfinite or empty audio: {row['filename']}")
        audio = audio.mean(axis=1)
        if rate != 16000:
            common = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // common, rate // common).astype(np.float32)
        count = 0
        for begin in range(0, len(audio), 80000):
            fragment = audio[begin:begin + 80000]
            weights.append(len(fragment))
            waves.append(np.pad(fragment, (0, 80000 - len(fragment))))
            owners.append(index)
            count += 1
        descriptions.append(dict(filename=row["filename"], src_file=row["src_file"],
                                 category=row["category"], seconds=len(audio) / 16000,
                                 windows=count))
    return torch.from_numpy(np.stack(waves)), np.array(owners), np.array(weights), descriptions


def aggregate(probabilities, owners, weights, recordings):
    result = np.zeros((recordings, 50), dtype=np.float64)
    np.add.at(result, owners, probabilities * weights[:, None])
    return result / np.bincount(owners, weights=weights)[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = args.output
    if (out / "summary.json").exists():
        print("Completed external evaluation exists; reusing its outputs.")
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("30-minute inference budget")))
    signal.alarm(1800)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    evidence = ROOT / "reports/external_fsd_20261105"
    rows = list(csv.DictReader((evidence / "manifest.csv").open()))
    y = np.array([int(row["target"]) for row in rows])
    classes = sorted(set(y.tolist()))
    waves, owners, weights, descriptions = load_recordings(rows, args.audio)
    write_csv(out / "recordings.csv", descriptions)
    print(f"External inference: {len(rows)} recordings, {len(waves)} windows, {len(classes)} target classes; all50 outputs retained.", flush=True)
    panns = load_panns(args.assets / "audioset_tagging_cnn",
                       args.assets / "Cnn14_16k_mAP=0.438.pth", device)
    with np.load(ROOT / "models/panns_linear/linear_head.npz", allow_pickle=False) as archive:
        head = {key: archive[key].copy() for key in archive.files}
    probabilities = []
    with torch.inference_mode():
        for begin in range(0, len(waves), 8):
            features = panns(waves[begin:begin + 8].to(device))["embedding"].cpu().numpy()
            logits = ((features - head["mean"]) / head["scale"]) @ head["coef"].T + head["intercept"]
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            ordered = np.zeros_like(prob)
            ordered[:, head["classes"].astype(int)] = prob / prob.sum(axis=1, keepdims=True)
            probabilities.append(ordered)
    outputs = {"panns_lr": aggregate(np.concatenate(probabilities), owners, weights, len(rows))}
    del panns
    if device.type == "cuda":
        torch.cuda.empty_cache()
    features = torch.stack([log_mel(wave) for wave in waves])
    for seed in (101, 202, 303):
        saved = torch.load(ROOT / f"models/noise_cnn/seed{seed}.pt", map_location="cpu", weights_only=True)
        model = build_model(saved["model_name"]).to(device).eval().requires_grad_(False)
        model.load_state_dict(saved["model"], strict=True)
        values = []
        with torch.inference_mode():
            for begin in range(0, len(features), 32):
                batch = (features[begin:begin + 32] - saved["mean"]) / saved["std"]
                values.append(model(batch.to(device)).softmax(1).cpu().numpy())
        outputs[f"noise_cnn_s{seed}"] = aggregate(np.concatenate(values), owners, weights, len(rows))
        del model
    metrics, per_class, predictions, correct = [], [], [], {}
    for model_name, prob in outputs.items():
        pred = prob.argmax(1)
        correct[model_name] = (pred == y).astype(float)
        recalls = [float((pred[y == c] == c).mean()) for c in classes]
        metrics.append(dict(model=model_name, recordings=len(y), correct=int((pred == y).sum()),
                            accuracy=float(accuracy_score(y, pred)), macro_recall=float(np.mean(recalls)),
                            macro_f1=float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
                            outside_target_rate=float((~np.isin(pred, classes)).mean())))
        for target, recall in zip(classes, recalls):
            per_class.append(dict(model=model_name, target=target,
                                  category=next(r["category"] for r in rows if int(r["target"]) == target),
                                  recordings=int((y == target).sum()), recall=recall))
        for row, target, value, probabilities in zip(rows, y, pred, prob):
            predictions.append(dict(model=model_name, filename=row["filename"], src_file=row["src_file"],
                                    target=int(target), prediction=int(value), confidence=float(probabilities[value])))
        np.savez_compressed(out / f"{model_name}_probabilities.npz", probabilities=prob,
                            targets=y, filenames=np.array([r["filename"] for r in rows]))
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "per_class.csv", per_class)
    write_csv(out / "predictions.csv", predictions)
    rng = np.random.default_rng(20261106)
    sample = rng.integers(0, len(y), size=(4000, len(y)))
    noise_correct = np.stack([correct[f"noise_cnn_s{s}"] for s in (101, 202, 303)], axis=1).mean(1)
    comparisons = {"panns_minus_mean_noise_cnn": correct["panns_lr"] - noise_correct}
    comparisons.update({f"panns_minus_noise_cnn_s{s}": correct["panns_lr"] - correct[f"noise_cnn_s{s}"] for s in (101, 202, 303)})
    effects = []
    for name, difference in comparisons.items():
        low, high = np.percentile(100 * difference[sample].mean(1), [2.5, 97.5])
        effects.append(dict(comparison=name, delta_pp=100 * float(difference.mean()),
                            low_pp=float(low), high_pp=float(high)))
    write_csv(out / "paired_effects.csv", effects)
    summary = dict(status="completed", recordings=len(rows), unique_recording_ids=len({r["src_file"] for r in rows}),
                   classes=len(classes), windows=len(waves), fits=0, added_noise=False, esc50_waveforms_read=0,
                   all50_output_classes=True, aggregation="valid-duration-weighted probabilities",
                   primary_effect=effects[0], metrics=metrics, inference_seconds=time.perf_counter() - started,
                   environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                    device=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    signal.alarm(0)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
