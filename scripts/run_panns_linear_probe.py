#!/usr/bin/env python3
# Chrome ChatGPT implementation draft, integrated for the fixed ESC-50 protocol.
import argparse
import csv
import json
import platform
import subprocess
import importlib.metadata
import signal
import sys
import time
import warnings
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LIMIT_SECONDS = 45 * 60
MODEL_CONFIG = dict(
    sample_rate=16000, window_size=512, hop_size=160,
    mel_bins=64, fmin=50, fmax=8000, classes_num=527
)


def deadline_handler(signum, frame):
    raise TimeoutError("45-minute wall-time budget exceeded")


def load_splits(metadata):
    metadata = Path(metadata)
    if metadata.is_dir():
        a = metadata / "meta" / "esc50.csv"
        metadata = a if a.exists() else metadata / "esc50.csv"

    with metadata.open(newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"filename", "fold", "target", "src_file", "category"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Invalid ESC-50 metadata columns")

    for r in rows:
        r["fold"], r["target"] = int(r["fold"]), int(r["target"])

    val_sources = {r["src_file"] for r in rows if r["fold"] == 4}
    test_sources = {r["src_file"] for r in rows if r["fold"] == 5}

    train = [
        r for r in rows
        if r["fold"] in (1, 2, 3)
        and r["src_file"] not in val_sources | test_sources
    ]
    val = [
        r for r in rows
        if r["fold"] == 4 and r["src_file"] not in test_sources
    ]

    train.sort(key=lambda r: r["filename"])
    val.sort(key=lambda r: r["filename"])
    print(f"Source-strict split: train={len(train)}, val={len(val)}",
          flush=True)

    if (len(train), len(val)) != (1200, 398):
        raise ValueError("Unexpected split counts; refusing to proceed")

    return train, val, metadata


def read_wave(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # Mono; no peak normalization
    if sr != 16000:
        d = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // d, sr // d)
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all() or len(audio) == 0:
        raise ValueError(f"Invalid audio: {path}")
    return audio


def extract(model, rows, audio_root, device, split, batch_size=8):
    chunks = []
    total = (len(rows) + batch_size - 1) // batch_size
    started = time.perf_counter()

    with torch.inference_mode():
        for b, start in enumerate(range(0, len(rows), batch_size), 1):
            subset = rows[start:start + batch_size]
            waves = [read_wave(audio_root / r["filename"])
                     for r in subset]
            if any(len(w) != 80000 for w in waves):
                raise ValueError("Expected exactly five seconds at 16 kHz")
            width = max(map(len, waves))
            batch = np.stack([
                np.pad(w, (0, width - len(w))) for w in waves
            ])
            inputs = torch.from_numpy(batch).to(device)
            output = model(inputs)["embedding"]
            features = output.cpu().numpy()

            if features.shape != (len(subset), 2048) or not np.isfinite(features).all():
                raise ValueError(f"Unexpected embedding: {features.shape}")

            chunks.append(features)
            print(f"[{split}] batch {b}/{total} "
                  f"({min(start + batch_size, len(rows))}/{len(rows)}) "
                  f"elapsed={time.perf_counter() - started:.1f}s",
                  flush=True)

    return np.concatenate(chunks), time.perf_counter() - started


def save_features(path, x, rows):
    np.savez_compressed(
        path, embeddings=x,
        filenames=np.array([r["filename"] for r in rows]),
        targets=np.array([r["target"] for r in rows]),
        src_files=np.array([r["src_file"] for r in rows])
    )


def main():
    parser = argparse.ArgumentParser()
    for key in ("audio-root", "metadata", "panns-repo",
                "checkpoint", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.perf_counter()
    torch.set_num_threads(2)
    audio_root = Path(args.audio_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("A completed run already exists; use its saved results")

    train, val, metadata_path = load_splits(args.metadata)
    device = torch.device(args.device)

    # Import the author's original model implementation.
    sys.path.insert(0, str(Path(args.panns_repo) / "pytorch"))
    from models import Cnn14_16k

    model = Cnn14_16k(**MODEL_CONFIG)
    # The official 2019 checkpoint includes legacy NumPy training metadata.
    # Retain weights-only loading and allow only the required NumPy data types.
    from numpy.core.multiarray import _reconstruct
    numpy_types = [(_reconstruct, "numpy.core.multiarray._reconstruct"),
                   np.ndarray, np.dtype,
                   *[type(np.dtype(t)) for t in (np.float32, np.float64, np.uint32, np.int64)]]
    with torch.serialization.safe_globals(numpy_types):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    model.requires_grad_(False)
    model.eval()
    model.to(device)
    loaded_parameters = sum(p.numel() for p in model.parameters())
    trainable_encoder_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PANNs checkpoint loaded strictly; encoder frozen.",
          flush=True)

    x_train, t_train = extract(
        model, train, audio_root, device, "train"
    )
    x_val, t_val = extract(
        model, val, audio_root, device, "val"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    save_features(output / "train_embeddings.npz", x_train, train)
    save_features(output / "val_embeddings.npz", x_val, val)

    y_train = np.array([r["target"] for r in train])
    y_val = np.array([r["target"] for r in val])

    head = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=3000, tol=1e-6
        ))
    ])

    print("Fitting one deterministic linear head...", flush=True)
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        head.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    probabilities = head.predict_proba(x_val)
    classes = head.named_steps["classifier"].classes_
    predictions = classes[np.argmax(probabilities, axis=1)]
    n_iter = head.named_steps["classifier"].n_iter_.tolist()

    joblib.dump(head, output / "linear_head.joblib")
    scaler = head.named_steps["scaler"]
    classifier = head.named_steps["classifier"]
    np.savez_compressed(output / "linear_head.npz", mean=scaler.mean_, scale=scaler.scale_,
                        coef=classifier.coef_, intercept=classifier.intercept_, classes=classes)
    with (output / "validation_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "target", "prediction"])
        writer.writerows((r["filename"], r["target"], int(pred)) for r, pred in zip(val, predictions))

    with (output / "validation_probabilities.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "src_file", "target", "prediction",
            *[f"prob_{c}" for c in classes]
        ])
        for r, pred, probs in zip(val, predictions, probabilities):
            writer.writerow([
                r["filename"], r["src_file"], r["target"],
                int(pred), *probs.tolist()
            ])

    labels = list(range(50))
    names = {}
    for r in train + val:
        k = r["target"]
        if k in names and names[k] != r["category"]:
            raise ValueError(f"Inconsistent category for target {k}")
        names[k] = r["category"]
    target_names = [names[k] for k in labels]
    (output / "class_names.json").write_text(json.dumps(target_names, indent=2))

    report = classification_report(
        y_val, predictions, labels=labels,
        target_names=target_names, zero_division=0
    )
    (output / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_val, predictions, labels=labels)
    with (output / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, cm):
            writer.writerow([label, *row.tolist()])

    result = {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "macro_f1": float(f1_score(
            y_val, predictions, labels=labels,
            average="macro", zero_division=0
        )),
        "loaded_model_parameters": loaded_parameters,
        "trainable_encoder_parameters": trainable_encoder_parameters,
        "linear_head_parameters": int(classifier.coef_.size + classifier.intercept_.size),
        "train_n": len(train),
        "val_n": len(val),
        "validation_sources": len({r["src_file"] for r in val}),
        "test_waveforms_read": 0,
        "classes": classes.tolist(),
        "n_iter": n_iter,
        "converged": not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        ),
        "fit_warnings": [str(w.message) for w in caught],
        "timing_seconds": {
            "train_features": t_train,
            "val_features": t_val,
            "linear_fit": fit_seconds,
            "total": time.perf_counter() - started
        },
        "config": {
            "audio_root": str(audio_root),
            "metadata": str(metadata_path),
            "panns_repo": args.panns_repo,
            "checkpoint": args.checkpoint,
            "device": str(device),
            "model": "Cnn14_16k",
            "model_config": MODEL_CONFIG,
            "batch_size": 8,
            "panns_source_revision": subprocess.check_output(["git", "-C", args.panns_repo, "rev-parse", "HEAD"], text=True).strip(),
            "linear_head": "StandardScaler + LogisticRegression",
            "C": 1.0, "solver": "lbfgs",
            "max_iter": 3000, "tol": 1e-6,
            "wall_time_limit_seconds": LIMIT_SECONDS
        },
        "environment": {
            "utc_finished": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
            "sklearn": sklearn.__version__,
            "torchlibrosa": importlib.metadata.version("torchlibrosa"),
            "librosa": importlib.metadata.version("librosa"),
            "gpu": torch.cuda.get_device_name(device)
                   if device.type == "cuda" else None
        },
        "interpretation": (
            "Development validation only. Potential pretraining "
            "overlap with ESC-50 has not been excluded."
        )
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    print(json.dumps({
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "n_iter": n_iter,
        "converged": result["converged"],
        "total_seconds": result["timing_seconds"]["total"]
    }, indent=2), flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, LIMIT_SECONDS)
    try:
        main()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
