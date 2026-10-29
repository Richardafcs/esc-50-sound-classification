"""Read-only research infrastructure: preflight, run provenance and comparisons.

The module deliberately never imports checkpoints or reads test predictions:
research audit must not turn into a covert test-set inspection. Historical
runs remain immutable. A run record is *evidence*, not proof that code ran.
"""
from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import torch

from .audio import FRONTEND
from .data import audit_metadata, load_cache, read_metadata, split_rows
from .models import MODEL_NAMES
from .quality import metadata_contract, cache_metadata_contract, split_contract, raw_audio_contract
from .records import canonical_hash, sha256, write_json


def _safe_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError, UnicodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _metric(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value, _ = _safe_json(path)
    if not isinstance(value, dict):
        return None
    return {k: value[k] for k in ("accuracy", "macro_f1", "loss", "n") if k in value}


def _split_fingerprint(split: dict[str, Any]) -> str | None:
    if not isinstance(split, dict) or not split:
        return None
    try:
        # We fingerprint exact row membership, not just the number of rows.
        return canonical_hash({part: dict(n=value["n"], names=value["names"],
                                          sources=value["sources"])
                               for part, value in sorted(split.items())})
    except (KeyError, TypeError, ValueError):
        return None


def preflight_report(root: str | Path, cache_path: str | Path, *,
                     train_folds=(1, 2, 3), val_fold=4, test_fold=5,
                     official_folds=False, verify_cache=True, verify_audio=False) -> dict[str, Any]:
    """Report *actual readiness* without changing or downloading the dataset."""
    root, cache_path = Path(root), Path(cache_path)
    blockers, warnings, checks = [], [], {}
    metadata = root / "meta" / "esc50.csv"
    rows = None
    if not metadata.is_file():
        blockers.append(f"Missing metadata: {metadata}")
    else:
        try:
            rows = read_metadata(root)
            checks["metadata"] = audit_metadata(rows)
            checks["metadata_contract"] = metadata_contract(rows)
            if not checks["metadata_contract"]["valid"]:
                blockers.append("Metadata filename/source/label contract failed")
            checks["split_contract"] = split_contract(rows, train_folds=train_folds, val_fold=val_fold,
                                                       test_fold=test_fold, strict=not official_folds)
            if not checks["split_contract"]["valid"]:
                blockers.append("Split contract failed")
            indices = split_rows(rows, train_folds, val_fold, test_fold,
                                 strict=not official_folds)
            if not indices["train"] or not indices["validation"] or not indices["test"]:
                blockers.append("Train/validation/test membership cannot be empty")
            checks["split"] = {k: dict(n=len(v),
                                       sha256=canonical_hash([rows[i]["filename"] for i in v]))
                               for k, v in indices.items()}
            groups = {k: {rows[i]["src_file"] for i in v} for k, v in indices.items()}
            checks["split"]["source_disjoint"] = all(
                not (groups[a] & groups[b]) for a, b in
                (("train", "validation"), ("train", "test"), ("validation", "test")))
            if official_folds and not checks["split"]["source_disjoint"]:
                warnings.append("Official folds include known cross-fold source overlaps")
            elif not checks["split"]["source_disjoint"]:
                blockers.append("Strict split contains shared source IDs")
            checks["split"]["policy"] = ("official" if official_folds else "source-strict")
        except (ValueError, KeyError, OSError) as exc:
            blockers.append(f"Metadata/split validation failed: {exc}")
    audio_dir = root / "audio"
    audio_found = sum((audio_dir / r["filename"]).is_file() for r in rows) if rows is not None else 0
    checks["audio"] = dict(available_files=audio_found,
                           expected_files=len(rows) if rows is not None else 2000,
                           complete=rows is not None and audio_found == len(rows))
    if not checks["audio"]["complete"]:
        warnings.append("Raw audio is incomplete: cannot independently regenerate features")
    if verify_audio:
        sidecar = cache_path.with_name(cache_path.name + ".json")
        try:
            content = json.loads(sidecar.read_text())
            if rows is None:
                blockers.append("Cannot verify WAV provenance without metadata")
            else:
                checks["audio"]["provenance"] = raw_audio_contract(
                    root, rows, content.get("audio_file_sha256"))
                if not checks["audio"]["provenance"]["valid"]:
                    blockers.append("Raw WAV provenance could not be verified")
        except (OSError, ValueError) as exc:
            blockers.append(f"Unable to read WAV provenance manifest: {exc}")
    checks["cache"] = dict(path=str(cache_path), exists=cache_path.is_file(),
                            sidecar=Path(str(cache_path) + ".json").is_file())
    if not cache_path.is_file():
        blockers.append("Feature cache unavailable; train command cannot start")
    elif not checks["cache"]["sidecar"]:
        blockers.append("Feature cache provenance sidecar is missing")
    elif verify_cache:
        try:
            loaded = load_cache(cache_path)
            checks["cache"].update(validated=True, count=len(loaded["y"]),
                                   sha256=sha256(cache_path))
            if rows is not None:
                checks["cache"]["raw_metadata_alignment"] = cache_metadata_contract(loaded, rows)
                if not checks["cache"]["raw_metadata_alignment"]["valid"]:
                    blockers.append("Cached labels/folds/sources/class mapping do not match raw metadata")
        except (ValueError, KeyError, OSError) as exc:
            blockers.append(f"Cache integrity check failed: {exc}")
    else:
        checks["cache"]["validated"] = False
        warnings.append("Cache checksum and contents NOT verified (--quick)")
        blockers.append("Quick mode is diagnostic only; full feature-cache verification is required")
    checks["runtime"] = dict(python=sys.version.split()[0], torch=str(torch.__version__),
                             platform=platform.platform(), cuda_available=torch.cuda.is_available(),
                             available_memory_note="check process memory quotas separately",
                             free_disk_bytes=shutil.disk_usage(Path.cwd()).free)
    return dict(schema_version=1, ready=len(blockers) == 0, blockers=blockers,
                warnings=warnings, checks=checks,
                scope="train-only readiness; no test evaluation and no training",
                frontend=FRONTEND)


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    """Audit a run's files without loading executable checkpoint payloads."""
    folder = Path(run_dir)
    issues, warnings = [], []
    run, run_err = _safe_json(folder / "run.json")
    if run_err or not isinstance(run, dict):
        return dict(path=str(folder), valid=False, issues=[f"run.json unreadable: {run_err}"],
                    warnings=[], evidence="no checkpoint execution")
    identity, identity_err = _safe_json(folder / "manifest.json") if (folder / "manifest.json").is_file() else (None, "missing")
    if identity_err or not isinstance(identity, dict):
        warnings.append("manifest.json missing or unreadable; historical run provenance is partial")
    else:
        if run.get("manifest_id") and canonical_hash(identity) != run["manifest_id"]:
            issues.append("manifest identity hash mismatch")
        if run.get("manifest_sha256") and sha256(folder / "manifest.json") != run["manifest_sha256"]:
            issues.append("manifest byte hash mismatch")
    history, hist_err = _safe_json(folder / "history.json") if (folder / "history.json").is_file() else (None, "missing")
    if isinstance(history, list):
        epochs = [h.get("epoch") for h in history if isinstance(h, dict)]
        if len(epochs) != len(history) or epochs != list(range(1, len(history) + 1)):
            issues.append("epoch history is not consecutive 1..N")
        if run.get("epochs_recorded") is not None and run["epochs_recorded"] != len(history):
            issues.append("epochs_recorded differs from history")
        if history and history[-1].get("partial_epoch"):
            warnings.append("last epoch was partial; exclude from equal-exposure studies")
    else:
        warnings.append(f"history.json unavailable ({hist_err})")
        history = []
    config = (identity or run).get("config", run.get("config", {}))
    if not isinstance(config, dict):
        issues.append("invalid run config")
        config = {}
    status = run.get("status", "unknown")
    if status == "completed":
        if not history or len(history) != config.get("epochs") or any(h.get("partial_epoch") for h in history):
            issues.append("completed status inconsistent with required full epochs")
    if not (folder / "best.pt").is_file():
        warnings.append("best.pt is unavailable (no selected deployable checkpoint)")
    # IMPORTANT: never inspect test.json or test_predictions.csv during discovery.
    protocol = (identity or run)
    split_id = _split_fingerprint(protocol.get("split", {}))
    if not split_id:
        warnings.append("split membership fingerprint unavailable")
    if not protocol.get("cache_sha256"):
        warnings.append("cache fingerprint unavailable")
    if not protocol.get("environment", {}).get("code_sha256"):
        warnings.append("source-code fingerprint unavailable")
    return dict(path=str(folder), valid=not issues, issues=issues, warnings=warnings,
                manifest_id=run.get("manifest_id"), status=status, completed=status == "completed" and not issues,
                model=config.get("model"), seed=config.get("seed"),
                declared_epochs=config.get("epochs"), epochs_recorded=len(history),
                test_accessed=bool(run.get("test_evaluated")),
                validation=_metric(folder / "validation.json"),
                cache_sha256=protocol.get("cache_sha256"),
                code_sha256=protocol.get("environment", {}).get("code_sha256"),
                split_fingerprint=split_id,
                source_policy=protocol.get("source_policy"),
                frontend_fingerprint=(canonical_hash(protocol["frontend"]) if protocol.get("frontend") else None),
                training_config={key: config.get(key) for key in (
                    "epochs", "lr", "batch_size", "scheduler", "specaugment", "official_folds",
                    "train_folds", "val_fold", "test_fold")},
                evidence="read-only, validation-only, no checkpoint execution")


def compatibility(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Conservative eligibility gate for controlled validation comparisons."""
    mismatches = []
    if not a.get("completed") or not b.get("completed"):
        mismatches.append("Both runs must finish their declared epochs")
    if not a.get("valid") or not b.get("valid"):
        mismatches.append("Both run records must pass integrity checks")
    for key in ("cache_sha256", "split_fingerprint", "source_policy", "frontend_fingerprint"):
        if not a.get(key) or a.get(key) != b.get(key):
            mismatches.append(f"Provenance missing or different: {key}")
    for key in ("epochs", "lr", "batch_size", "scheduler", "specaugment", "official_folds",
                "train_folds", "val_fold", "test_fold"):
        av, bv = a.get("training_config", {}).get(key), b.get("training_config", {}).get(key)
        if av is None or bv is None or av != bv:
            mismatches.append(f"Training setting missing or different: {key}")
    if a.get("code_sha256") != b.get("code_sha256"):
        mismatches.append("Different or unknown code versions: review before comparing")
    if a.get("validation") is None or b.get("validation") is None:
        mismatches.append("Missing validation metrics")
    return dict(eligible=len(mismatches) == 0, reasons=mismatches,
                scope="validation comparison; comparable protocol does not prove causality",
                test_metrics_read=False)


def run_index(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {root}")
    discovered = sorted(p for p in root.rglob("run.json") if p.is_file() and not p.is_symlink())
    entries = [inspect_run(p.parent) for p in discovered]
    groups = defaultdict(list)
    for record in entries:
        if record.get("manifest_id"):
            groups[record["manifest_id"]].append(record["path"])
    duplicates = {mid: paths for mid, paths in groups.items() if len(paths) > 1}
    return dict(schema_version=1, scanned=str(root), total=len(entries),
                completed=sum(bool(r.get("completed")) for r in entries),
                incomplete=sum(not r.get("completed") for r in entries),
                distinct_manifest_ids=len(groups),
                duplicate_manifest_groups=duplicates,
                caution="matching manifest ID may be a copied run or a rerun of the same recipe; do not count independently without checking checkpoint/history identity",
                runs=entries, test_metrics_read=False)


def write_index_csv(path: str | Path, index: dict[str, Any]) -> None:
    fields = ["path", "model", "seed", "status", "completed", "valid", "epochs_recorded",
              "declared_epochs", "test_accessed", "validation_accuracy", "validation_macro_f1",
              "cache_sha256", "split_fingerprint", "code_sha256"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for entry in index["runs"]:
            record = {k: entry.get(k) for k in fields}
            record["validation_accuracy"] = (entry.get("validation") or {}).get("accuracy")
            record["validation_macro_f1"] = (entry.get("validation") or {}).get("macro_f1")
            writer.writerow(record)
