"""Focused sample-rate / audio-feature ablations for ESC-50.

Lightweight study path: validate official metadata, sample alignment, shapes,
train-only normalization and split isolation. No redundant SHA-256 provenance
chains. Historical checkpoints and legacy loaders are intentionally unchanged.

All selection uses the development validation fold; no test examples are
consumed in feature-ablation training or summaries.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .audio import log_mel as legacy_log_mel
from .data import audit_metadata, fit_normalization, read_metadata, split_rows
from .evaluation import compute_metrics, run_eval, save_predictions
from .models import build_model
from .records import seed_all, write_json
from .checkpoints import load_checkpoint, save_checkpoint_atomic

FEATURE_TYPES = ("logmel64", "logmel32", "mfcc40", "loglinear64")


@dataclass(frozen=True)
class FeatureSpec:
    sample_rate: int
    max_frequency: int
    feature: str

    def __post_init__(self):
        if self.sample_rate not in (16000, 32000):
            raise ValueError("Sample rate must be 16000 or 32000")
        if self.max_frequency not in (8000, 16000) or self.max_frequency > self.sample_rate // 2:
            raise ValueError("Max frequency must be 8000 or 16000 and at most Nyquist")
        if self.feature not in FEATURE_TYPES:
            raise ValueError(f"Unknown feature: {self.feature}")

    @property
    def name(self):
        return f"sr{self.sample_rate // 1000}_fmax{self.max_frequency // 1000}_{self.feature}"

    @property
    def n_fft(self):
        return self.sample_rate * 32 // 1000  # 32 ms, matching 512/16k legacy

    @property
    def hop(self):
        return self.sample_rate * 16 // 1000  # 16 ms, matching 256/16k legacy

    @property
    def bands(self):
        return {"logmel64": 64, "logmel32": 32, "mfcc40": 40,
                "loglinear64": 64}[self.feature]


# Full factorial feature comparison within each of three spectral conditions.
# sr16/fmax8 vs sr32/fmax8: sample rate, matched usable bandwidth.
# sr32/fmax8 vs sr32/fmax16: effect of additional high-frequency information.
GRID = tuple(FeatureSpec(sr, fmax, feature)
             for sr, fmax in ((16000, 8000), (32000, 8000), (32000, 16000))
             for feature in FEATURE_TYPES)


def by_name(name: str) -> FeatureSpec:
    for spec in GRID:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown case {name!r}. Use the plan command to list cases")


def resample_official_waveform(path: str | Path, spec: FeatureSpec) -> torch.Tensor:
    import soundfile as sf
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != 44100 or waveform.shape != (5 * 44100, 1):
        raise ValueError(f"Expected official 44.1kHz mono 5s WAV: {path}")
    mono = waveform[:, 0]
    if not np.isfinite(mono).all():
        raise ValueError(f"Nonfinite waveform: {path}")
    factor = math.gcd(sr, spec.sample_rate)
    result = resample_poly(mono, spec.sample_rate // factor, sr // factor).astype(np.float32)
    if result.shape != (5 * spec.sample_rate,):
        raise ValueError("Resampled length differs from 5 seconds")
    return torch.from_numpy(result.copy())


def _triangular_bank(spec: FeatureSpec, *, scale: str, bands: int, device):
    nyquist = spec.sample_rate / 2
    if scale == "mel":
        upper = 2595 * math.log10(1 + spec.max_frequency / 700)
        points = torch.linspace(0, upper, bands + 2, device=device)
        hz = 700 * (10 ** (points / 2595) - 1)
    elif scale == "linear":
        hz = torch.linspace(0, spec.max_frequency, bands + 2, device=device)
    else:
        raise ValueError("Unknown filterbank scale")
    freqs = torch.linspace(0, nyquist, spec.n_fft // 2 + 1, device=device)
    left = (freqs[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
    right = (hz[2:, None] - freqs[None]) / (hz[2:, None] - hz[1:-1, None])
    return torch.minimum(left, right).clamp_min(0)


def extract_feature(waveform: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
    """Returns [1,bands,313]; same 32/16ms analysis window/hop in all cases.

    16k / 8k / logmel64 intentionally matches the historical project frontend.
    MFCC40 means 40 orthonormal DCT-II coefficients of 64 natural-log Mel
    energies, without liftering or an explicit c0 replacement.
    """
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if (waveform.shape != (5 * spec.sample_rate,) or
            not bool(torch.isfinite(waveform).all())):
        raise ValueError("Expected finite mono 5-second waveform")
    if spec == FeatureSpec(16000, 8000, "logmel64"):
        return legacy_log_mel(waveform)
    power = torch.stft(waveform, n_fft=spec.n_fft, hop_length=spec.hop,
                       window=torch.hann_window(spec.n_fft, device=waveform.device),
                       center=True, return_complex=True).abs().square()
    if spec.feature == "loglinear64":
        feat = _triangular_bank(spec, scale="linear", bands=64, device=waveform.device) @ power
        feat = feat.clamp_min(1e-10).log()
    else:
        nmel = 32 if spec.feature == "logmel32" else 64
        logmel = (_triangular_bank(spec, scale="mel", bands=nmel,
                                   device=waveform.device) @ power).clamp_min(1e-10).log()
        if spec.feature == "mfcc40":
            # Orthonormal DCT-II across 64 Mel bands. No extra normalization fit.
            n = torch.arange(64, device=waveform.device, dtype=torch.float32)
            k = torch.arange(40, device=waveform.device, dtype=torch.float32)[:, None]
            basis = (2 / 64) ** 0.5 * torch.cos(math.pi / 64 * (n[None] + 0.5) * k)
            basis[0] *= 2 ** -0.5
            feat = basis @ logmel
        else:
            feat = logmel
    feat = feat.unsqueeze(0).contiguous()
    if feat.shape != (1, spec.bands, 313) or not bool(torch.isfinite(feat).all()):
        raise ValueError(f"Unexpected output: {tuple(feat.shape)}")
    return feat


def _validate_rows(rows, *, require_complete):
    audit_metadata(rows, require_complete=require_complete)
    for idx, row in enumerate(rows):
        stem = row["filename"].removesuffix(".wav").split("-")
        if (len(stem) != 4 or stem[0] != str(int(row["fold"])) or
                stem[1] != str(row["src_file"]) or
                stem[3] != str(int(row["target"]))):
            raise ValueError(f"ESC-50 filename/metadata mismatch at row {idx}")


def _canonical_rows(rows):
    return [(r["filename"], int(r["fold"]), int(r["target"]),
             str(r["src_file"]), str(r["category"])) for r in rows]


def prepare_case(raw_root: str | Path, cache_dir: str | Path, spec: FeatureSpec,
                 *, require_complete=True):
    """Read each WAV, write simple NPZ cache + readable JSON (no file hashes)."""
    root, directory = Path(raw_root), Path(cache_dir)
    rows = read_metadata(root)
    _validate_rows(rows, require_complete=require_complete)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.name}.npz"
    sidecar = directory / f"{spec.name}.json"
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Study cache already exists: {path}")
    features = []
    for row in rows:
        wave = resample_official_waveform(root / "audio" / row["filename"], spec)
        features.append(extract_feature(wave, spec).numpy())
    # Validation before publishing the cache. Partial file is never mistaken
    # for a completed experiment; avoid an expensive post-write byte digest.
    data = np.stack(features).astype(np.float32)
    class_names = {int(r["target"]): str(r["category"]) for r in rows}
    class_labels = np.array([class_names[i] for i in sorted(class_names)])
    import tempfile
    import os
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".features-",
                                         suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            np.savez_compressed(handle, x=data, names=np.array([r[0] for r in _canonical_rows(rows)]),
                                y=np.array([r[2] for r in _canonical_rows(rows)], dtype=np.int64),
                                folds=np.array([r[1] for r in _canonical_rows(rows)], dtype=np.int64),
                                sources=np.array([r[3] for r in _canonical_rows(rows)]),
                                class_names=class_labels)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    write_json(sidecar, dict(schema="feature_ablation_v1", spec=asdict(spec),
                            clips=len(rows), shape=list(data.shape),
                            source="official ESC-50 WAV; no raw-file or cache SHA chain",
                            metadata="meta/esc50.csv", has_test_metrics=False))
    return path


def load_case(raw_root: str | Path, cache_dir: str | Path, spec: FeatureSpec,
              *, require_complete=True):
    root, directory = Path(raw_root), Path(cache_dir)
    sidecar = json.loads((directory / f"{spec.name}.json").read_text())
    if sidecar.get("schema") != "feature_ablation_v1" or sidecar.get("spec") != asdict(spec):
        raise ValueError("Wrong feature-ablation cache config")
    rows = read_metadata(root)
    _validate_rows(rows, require_complete=require_complete)
    with np.load(directory / f"{spec.name}.npz", allow_pickle=False) as archive:
        cache = {k: archive[k].copy() for k in ("x", "names", "y", "folds", "sources", "class_names")}
    if (cache["x"].shape != (len(rows), 1, spec.bands, 313) or
            cache["x"].dtype != np.float32 or not np.isfinite(cache["x"]).all() or
            len(set(cache["names"].tolist())) != len(rows)):
        raise ValueError("Invalid, nonfinite, or duplicate feature cache")
    labels = {int(r["target"]): str(r["category"]) for r in rows}
    if cache["class_names"].tolist() != [labels[i] for i in sorted(labels)]:
        raise ValueError("Feature cache category mapping differs from authoritative metadata")
    for idx, row in enumerate(rows):
        if (str(cache["names"][idx]) != row["filename"] or
                int(cache["y"][idx]) != int(row["target"]) or
                int(cache["folds"][idx]) != int(row["fold"]) or
                str(cache["sources"][idx]) != str(row["src_file"])):
            raise ValueError(f"Feature cache/metadata mismatch at row {idx}")
    return cache, rows


def run_case(raw_root, cache_dir, run_dir, spec, *, model_name="cnn", seed=101,
             epochs=60, batch_size=32, learning_rate=1e-3, threads=2, device="cpu",
             require_complete=True):
    """Train on folds 1-3, validate on 4, NEVER read fold 5 features.

    Exactly the original 3/1/1 source-strict protocol. Each run is new-only.
    For frequency-shape independence use cnn/crnn (frequency pooled, identical
    parameter counts), not the frequency-preserving cnn_freq/crnn_freq heads.
    """
    if model_name not in ("cnn", "crnn"):
        raise ValueError("Ablation requires frequency-pooled cnn or crnn to keep parameters fixed")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    seed_all(seed)
    torch.set_num_threads(threads)
    cache, rows = load_case(raw_root, cache_dir, spec, require_complete=require_complete)
    indices = split_rows(rows, (1, 2, 3), 4, 5, strict=True)
    if not indices["train"] or not indices["validation"]:
        raise ValueError("Train/validation empty")
    # No test features are selected. Fold 5 exists in the cache only as a
    # protected, unused partition, matching prior project practice.
    x, y = cache["x"], cache["y"]
    mean, std = fit_normalization(x[indices["train"]])
    train_x = torch.from_numpy(((x[indices["train"]] - mean) / std).astype(np.float32))
    val_x = torch.from_numpy(((x[indices["validation"]] - mean) / std).astype(np.float32))
    train_y = torch.from_numpy(y[indices["train"]]).long()
    val_y = torch.from_numpy(y[indices["validation"]]).long()
    train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                          shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_dl = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    actual_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                 else "cpu" if device == "auto" else device)
    model = build_model(model_name).to(actual_device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(spec=asdict(spec), model=model_name, seed=seed, epochs=epochs,
                  batch_size=batch_size, learning_rate=learning_rate,
                  train_folds=[1, 2, 3], validation_fold=4, test_fold=5,
                  source_strict=True, parameter_count=param_count,
                  data_source="original ESC-50 WAV feature cache", comparison_scope="development only",
                  byte_hash_checks=False, data_complete=require_complete,
                  training_examples=len(indices["train"]),
                  validation_names=cache["names"][indices["validation"]].tolist(),
                  validation_targets=y[indices["validation"]].tolist())
    write_json(out / "config.json", config)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_predictions = None
    best_metrics = None
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch, targets in train_dl:
            batch, targets = batch.to(actual_device), targets.to(actual_device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(targets)
            seen += len(targets)
        metrics, predictions = run_eval(model, val_dl, actual_device)
        history.append(dict(epoch=epoch, train_loss=total / seen, validation_accuracy=metrics["accuracy"],
                            validation_macro_f1=metrics["macro_f1"], validation_loss=metrics["loss"],
                            learning_rate=optimizer.param_groups[0]["lr"]))
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_metrics = metrics
            best_predictions = predictions
            # Best-only checkpoint. It carries human-readable protocol details,
            # not a file-hash chain. Historical checkpoint files stay unchanged.
            save_checkpoint_atomic(out / "best.pt", dict(model=model.state_dict(),
                                  model_name=model_name, epoch=epoch, seed=seed, epochs=epochs,
                                  training_config=config,
                                  sample_rate=spec.sample_rate, feature_spec=asdict(spec),
                                  mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                                  class_names=cache["class_names"].tolist(),
                                  run_scope="ablation-development-validation-only"))
        scheduler.step()
        write_json(out / "history.json", history)
    save_predictions(out / "validation_predictions.csv", cache["names"][indices["validation"]],
                     y[indices["validation"]], best_predictions)
    # Publish completion metrics last; a crash must not advertise incomplete evidence.
    write_json(out / "validation.json", dict(best_metrics, best_epoch=best_epoch,
                                              elapsed_seconds=time.monotonic() - start))
    print(f"{spec.name} / {model_name} / seed={seed}: "
          f"val accuracy={best_metrics['accuracy']:.4f}, best epoch={best_epoch}")
    return best_metrics



def completed_run(run_dir: str | Path):
    """Validate full-budget, mutually consistent development evidence, without hashes.

    This is an artifact consistency check, not independent waveform replay.
    Rejected records are never silently pooled with completed experiments.
    """
    run_dir = Path(run_dir)
    required = ("config.json", "history.json", "best.pt", "validation.json",
                "validation_predictions.csv")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete ablation evidence at {run_dir}: missing {missing}")
    try:
        config = json.loads((run_dir / "config.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        metrics = json.loads((run_dir / "validation.json").read_text())
        spec = FeatureSpec(**config["spec"])
        epochs = config["epochs"]
        if (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1 or
                not isinstance(history, list) or len(history) != epochs or
                [row["epoch"] for row in history] != list(range(1, epochs + 1))):
            raise ValueError("Incomplete ablation exposure: expected every configured epoch")
        if (config["train_folds"] != [1, 2, 3] or config["validation_fold"] != 4 or
                config["test_fold"] != 5 or config["source_strict"] is not True or
                config["model"] not in ("cnn", "crnn")):
            raise ValueError("Ablation protocol differs from development-only source-strict study")
        for row in history:
            for key in ("train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1"):
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError("Nonfinite or invalid ablation history")
            if row["validation_accuracy"] > 1 or row["validation_macro_f1"] > 1:
                raise ValueError("Invalid ablation metric range")
        best = min(history, key=lambda row: row["validation_loss"])
        if metrics["best_epoch"] != best["epoch"]:
            raise ValueError("Selected epoch differs from earliest minimum validation loss")
        for recorded, historical in (("accuracy", "validation_accuracy"),
                                     ("macro_f1", "validation_macro_f1"),
                                     ("loss", "validation_loss")):
            if not math.isclose(metrics[recorded], best[historical], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Selected validation {recorded} differs from history")
        with (run_dir / "validation_predictions.csv").open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        names = [row["filename"] for row in predictions]
        targets = [int(row["target"]) for row in predictions]
        predicted = [int(row["prediction"]) for row in predictions]
        if (not names or len(names) != len(set(names)) or
                names != config["validation_names"] or targets != config["validation_targets"]):
            raise ValueError("Validation predictions differ from registered membership/labels")
        for name, target in zip(names, targets):
            parts = name.removesuffix(".wav").split("-")
            if len(parts) != 4 or parts[0] != "4" or int(parts[3]) != target:
                raise ValueError("Prediction identity is not a validation-fold filename/target")
        if config["data_complete"] and (len(names) != 398 or config["training_examples"] != 1200):
            raise ValueError("Complete ESC-50 study requires 1200 train / 398 validation examples")
        measured = compute_metrics(np.asarray(targets), np.asarray(predicted))
        for key in ("accuracy", "macro_f1", "n"):
            if not math.isclose(metrics[key], measured[key], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Saved prediction {key} differs from validation metrics")
        if metrics.get("confusion_matrix") != measured["confusion_matrix"]:
            raise ValueError("Saved prediction confusion matrix differs from validation metrics")
        checkpoint = load_checkpoint(run_dir / "best.pt")
        if (checkpoint.get("training_config") != config or
                checkpoint.get("model_name") != config["model"] or
                checkpoint.get("feature_spec") != asdict(spec) or
                checkpoint.get("seed") != config["seed"] or
                checkpoint.get("epochs") != epochs or checkpoint.get("epoch") != best["epoch"] or
                checkpoint.get("run_scope") != "ablation-development-validation-only"):
            raise ValueError("Checkpoint identity differs from selected ablation run")
        for key in ("mean", "std"):
            value = checkpoint[key]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 1, spec.bands, 1)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid checkpoint normalization")
        if not bool((checkpoint["std"] > 0).all()):
            raise ValueError("Invalid checkpoint normalization scale")
        with torch.random.fork_rng(devices=[]):
            model = build_model(config["model"])
        if sum(p.numel() for p in model.parameters()) != config["parameter_count"]:
            raise ValueError("Model parameter count differs from config")
        model.load_state_dict(checkpoint["model"], strict=True)
        if any(not bool(torch.isfinite(value).all()) for value in checkpoint["model"].values()):
            raise ValueError("Nonfinite checkpoint weights")
        return config, metrics
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, EOFError) as error:
        raise ValueError(f"Invalid ablation evidence at {run_dir}: {error}") from error


def summary(results_dir: str | Path, output: str | Path):
    """Summarize only completed validation runs. Never touch held-out test."""
    results = []
    seen = set()
    for file in sorted(Path(results_dir).rglob("validation.json")):
        run_dir = file.parent
        cfg, metrics = completed_run(run_dir)
        key = (FeatureSpec(**cfg["spec"]).name, cfg["model"], cfg["seed"])
        if key in seen:
            raise ValueError(f"Duplicate ablation evidence: {key}")
        seen.add(key)
        results.append(dict(case=by_name(f"sr{cfg['spec']['sample_rate']//1000}_fmax{cfg['spec']['max_frequency']//1000}_{cfg['spec']['feature']}").name,
                            feature=cfg["spec"]["feature"], sample_rate=cfg["spec"]["sample_rate"],
                            max_frequency=cfg["spec"]["max_frequency"], model=cfg["model"],
                            seed=cfg["seed"], accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                            best_epoch=metrics["best_epoch"], parameter_count=cfg["parameter_count"]))
    if not results:
        raise ValueError("No completed validation runs")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Summary destination exists; never overwrite experiment evidence")
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def paired_contrasts(results_dir: str | Path, output: str | Path):
    """Predeclared paired ablations; compare same seed/model/hyperparameters.

    Deltas are development-validation accuracy differences, never p-values or
    held-out test claims. Missing case/seed pairs are omitted, not imputed.
    """
    runs = {}
    for item in sorted(Path(results_dir).rglob("validation.json")):
        config, metrics = completed_run(item.parent)
        spec = FeatureSpec(**config["spec"])
        key = (spec.name, config["model"], config["seed"])
        if key in runs:
            raise ValueError(f"Duplicate ablation evidence: {key}")
        runs[key] = (config, metrics)

    contrasts = []
    for model in ("cnn", "crnn"):
        for seed in sorted({key[2] for key in runs if key[1] == model}):
            def match(left, right, effect):
                a, b = (runs.get((left.name, model, seed)),
                        runs.get((right.name, model, seed)))
                if a is None or b is None:
                    return
                ca, ma = a
                cb, mb = b
                for field in ("epochs", "batch_size", "learning_rate", "train_folds",
                              "validation_fold", "test_fold", "source_strict"):
                    if ca[field] != cb[field]:
                        raise ValueError(f"Incompatible training condition: {field}")
                contrasts.append(dict(effect=effect, feature=left.feature, model=model, seed=seed,
                                      left=left.name, right=right.name,
                                      delta_accuracy_pp=100 * (mb["accuracy"] - ma["accuracy"]),
                                      delta_macro_f1_pp=100 * (mb["macro_f1"] - ma["macro_f1"])))
            for feat in FEATURE_TYPES:
                match(FeatureSpec(16000, 8000, feat), FeatureSpec(32000, 8000, feat),
                      "sample_rate_matched_bandwidth")
                match(FeatureSpec(32000, 8000, feat), FeatureSpec(32000, 16000, feat),
                      "added_high_frequency_bandwidth")
            for sr, fmax in ((16000, 8000), (32000, 8000), (32000, 16000)):
                for feat in FEATURE_TYPES:
                    if feat != "logmel64":
                        match(FeatureSpec(sr, fmax, "logmel64"), FeatureSpec(sr, fmax, feat),
                              "representation_vs_logmel64")
    if not contrasts:
        raise ValueError("No matched case/seed pairs; run more ablations")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("Contrast destination exists")
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    return contrasts
