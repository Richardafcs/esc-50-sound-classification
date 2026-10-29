"""Evaluation-mode metrics, held-out evaluation and one-clip research inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .checkpoints import load_checkpoint, validate_checkpoint
from .data import load_cache
from .models import build_model
from .records import append_evaluation_audit, canonical_hash, sha256, write_json

def compute_metrics(targets, predictions, num_classes=50):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("Metrics require equal, nonempty target and prediction arrays")
    if (targets.ndim != 1 or predictions.ndim != 1 or
            not np.issubdtype(targets.dtype, np.integer) or
            not np.issubdtype(predictions.dtype, np.integer) or
            np.any(targets < 0) or np.any(targets >= num_classes) or
            np.any(predictions < 0) or np.any(predictions >= num_classes)):
        raise ValueError("Metric labels must be one-dimensional integer class IDs in range")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets.astype(int), predictions.astype(int)), 1)
    tp = confusion.diagonal().astype(float)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    return dict(accuracy=float(tp.sum() / len(targets)), macro_f1=float(f1.mean()),
                per_class_f1=f1.tolist(), confusion_matrix=confusion.tolist(),
                n=len(targets), confusion_convention="rows=true, columns=predicted")


def run_eval(model, loader, device, time_shuffle_seed=None):
    model.eval()
    targets, predictions, loss_sum = [], [], 0.0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            permutation = None
            if time_shuffle_seed is not None:
                # The same seed/permutation is applied to every clip.
                length = x.shape[-1] // 8
                permutation = torch.randperm(length, generator=torch.Generator().manual_seed(time_shuffle_seed))
            logits = model(x, time_permutation=permutation)
            loss_sum += F.cross_entropy(logits, y.to(device), reduction="sum").item()
            targets.extend(y.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    metrics = compute_metrics(targets, predictions)
    metrics["loss"] = loss_sum / len(targets)
    return metrics, predictions


def save_predictions(path, names, targets, predictions):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "target", "prediction"])
        writer.writerows(zip(names, targets, predictions))


def evaluate(args):
    if not args.allow_test:
        raise ValueError("Test is sealed. Freeze configuration, then explicitly pass --allow-test")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    verification = validate_checkpoint(checkpoint, getattr(args, "allow_legacy_checkpoint", False))
    if checkpoint["cache_sha256"] != sha256(args.cache):
        raise ValueError("Feature cache differs from training")
    cache = load_cache(args.cache)
    out = Path(args.output)
    # Reserve a new result directory BEFORE inference. Refuse silent replacement.
    out.mkdir(parents=True, exist_ok=False)
    audit_path = checkpoint_path.parent / "evaluation_history.jsonl"
    record = dict(evaluation_id=canonical_hash(dict(checkpoint=sha256(checkpoint_path),
                                                   output=str(out.resolve()), time_ns=time.time_ns())),
                  timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  checkpoint_sha256=sha256(checkpoint_path), manifest_id=checkpoint.get("manifest_id"),
                  source_policy=checkpoint.get("source_policy", "legacy: unverified"),
                  output=str(out.resolve()), status="started", test_fold=checkpoint["test_fold"],
                  provenance_verification=verification)
    append_evaluation_audit(audit_path, record)
    # The Boolean now means test access occurred, not just successful completion.
    run_path = checkpoint_path.parent / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text())
        if checkpoint.get("manifest_id") == run.get("manifest_id") or verification.startswith("legacy"):
            run["test_evaluated"] = True
            run["test_evaluation_audit"] = str(audit_path)
            write_json(run_path, run)
    try:
        ids = np.flatnonzero(cache["folds"] == checkpoint["test_fold"])
        x = (cache["x"][ids] - checkpoint["mean"].numpy()) / checkpoint["std"].numpy()
        model = build_model(checkpoint["model_name"])
        model.load_state_dict(checkpoint["model"])
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(cache["y"][ids]).long()), batch_size=32,
                            generator=torch.Generator().manual_seed(0))
        metrics, predictions = run_eval(model, loader, "cpu")
        metrics.update(test_fold=checkpoint["test_fold"], checkpoint_sha256=sha256(checkpoint_path),
                       manifest_id=checkpoint.get("manifest_id"), source_policy=record["source_policy"],
                       split=checkpoint.get("manifest", {}).get("split"),
                       provenance_verification=verification, evaluation_id=record["evaluation_id"],
                       warning="This is one held-out fold, not official five-fold cross-validation")
        write_json(out / "test.json", metrics)
        save_predictions(out / "test_predictions.csv", cache["names"][ids], cache["y"][ids], predictions)
        append_evaluation_audit(audit_path, dict(record, status="completed", n=len(ids),
                                                accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"]))
    except BaseException as error:
        append_evaluation_audit(audit_path, dict(record, status="failed", error_type=type(error).__name__, error=str(error)))
        raise
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "n")}, indent=2))
