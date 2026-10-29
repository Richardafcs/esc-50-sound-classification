"""Audit saved ESC-50 results and aggregate unique-seed outcomes without inference.

The utility reads existing artifacts only. It never trains a model, accesses
audio/features, or generates test predictions. A holdout report is an audit of
already-produced test artifacts and is not authorization to evaluate test data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

NUM_CLASSES = 50


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    return value


def _finite(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _integer(value, name, lower=0):
    _finite(value, name, lower=lower)
    if int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _read_metadata(path):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"filename", "fold", "target", "src_file"}.issubset(rows[0]):
        raise ValueError("Metadata needs filename,fold,target,src_file")
    result = {}
    for row in rows:
        name = row["filename"]
        if name in result:
            raise ValueError("Metadata contains duplicate filenames")
        target, fold = int(row["target"]), int(row["fold"])
        if not 0 <= target < NUM_CLASSES or fold not in range(1, 6):
            raise ValueError("Metadata has an invalid class/fold")
        result[name] = dict(row, target=target, fold=fold, src_file=str(row["src_file"]))
    return result


def _read_predictions(path, metadata):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"filename", "target", "prediction"}.issubset(rows[0]):
        raise ValueError("Predictions need filename,target,prediction")
    result = []
    seen = set()
    for row in rows:
        name = row["filename"]
        if name in seen or name not in metadata:
            raise ValueError(f"Duplicate or unknown prediction filename: {name}")
        seen.add(name)
        target, prediction = int(row["target"]), int(row["prediction"])
        if not 0 <= target < NUM_CLASSES or not 0 <= prediction < NUM_CLASSES:
            raise ValueError("Prediction class outside 0..49")
        if target != metadata[name]["target"]:
            raise ValueError(f"Prediction label differs from official metadata: {name}")
        result.append(dict(filename=name, target=target, prediction=prediction))
    return result


def recompute_metrics(rows):
    if not rows:
        raise ValueError("Cannot score empty predictions")
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for row in rows:
        confusion[row["target"], row["prediction"]] += 1
    tp = confusion.diagonal().astype(float)
    support, predicted = confusion.sum(1), confusion.sum(0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=support + predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    return dict(accuracy=float(tp.sum() / len(rows)), macro_f1=float(f1.mean()), n=len(rows),
                correct=int(tp.sum()), confusion_matrix=confusion.tolist(), per_class_f1=f1.tolist(),
                per_class_recall=recall.tolist(), zero_recall_classes=int((recall == 0).sum()))


def _assert_metric_match(stored, recomputed):
    for key in ("accuracy", "macro_f1"):
        _finite(stored.get(key), key, 0, 1)
        if not math.isclose(stored[key], recomputed[key], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Stored {key} disagrees with saved predictions")
    if stored.get("n") != recomputed["n"]:
        raise ValueError("Stored sample count disagrees with predictions")
    if "confusion_matrix" in stored and stored["confusion_matrix"] != recomputed["confusion_matrix"]:
        raise ValueError("Stored confusion matrix disagrees with saved predictions")
    if "per_class_f1" in stored:
        values = np.asarray(stored["per_class_f1"], dtype=float)
        if values.shape != (NUM_CLASSES,) or not np.allclose(values, recomputed["per_class_f1"], atol=1e-10, rtol=0):
            raise ValueError("Stored per-class F1 disagrees with predictions")
    if "loss" in stored:
        _finite(stored["loss"], "loss", lower=0)


def _training_artifacts(reference):
    """Return immutable identity, mutable status, history, and provenance level."""
    reference = Path(reference)
    if reference.is_file():
        folder = reference.parent
        provided = read_json(reference)
    else:
        folder, provided = reference, None
    run_path = folder / "run.json"
    run = read_json(run_path) if run_path.is_file() else (provided or {})
    manifest_path = folder / "manifest.json"
    identity = read_json(manifest_path) if manifest_path.is_file() else (provided or run)
    if not isinstance(identity, dict) or not isinstance(run, dict) or not isinstance(identity.get("config"), dict):
        raise ValueError("Training identity/config is missing")
    if manifest_path.is_file():
        if run.get("manifest_id") != canonical_hash(identity):
            raise ValueError("Training manifest identity hash mismatch")
        if run.get("manifest_sha256") != sha256(manifest_path):
            raise ValueError("Training manifest file hash mismatch")
        if any(run.get(key) != value for key, value in identity.items()):
            raise ValueError("Mutable run record disagrees with immutable manifest")
        provenance = "immutable manifest/hash verified"
    else:
        provenance = "legacy or supplied config: immutable manifest unavailable"
    history_path = folder / "history.json"
    if not history_path.is_file():
        raise ValueError("Training history is missing")
    history = read_json(history_path)
    if not isinstance(history, list) or not history:
        raise ValueError("Training history is empty")
    return folder, identity, run, history, provenance


def _expected_splits(identity, metadata):
    config = identity["config"]
    try:
        train_folds = tuple(int(f) for f in (config["train_folds"].split(",")
                                             if isinstance(config["train_folds"], str) else config["train_folds"]))
        val_fold, test_fold = int(config["val_fold"]), int(config["test_fold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Training fold-role config is incomplete") from error
    if (not train_folds or len(set(train_folds)) != len(train_folds) or val_fold == test_fold or
            val_fold in train_folds or test_fold in train_folds or
            any(f not in range(1, 6) for f in (*train_folds, val_fold, test_fold))):
        raise ValueError("Training fold roles overlap or are invalid")
    expected = {
        "train": {name for name, row in metadata.items() if row["fold"] in train_folds},
        "validation": {name for name, row in metadata.items() if row["fold"] == val_fold},
        "test": {name for name, row in metadata.items() if row["fold"] == test_fold},
    }
    official = bool(config.get("official_folds", False))
    if not official:
        test_sources = {metadata[name]["src_file"] for name in expected["test"]}
        heldout_sources = test_sources | {metadata[name]["src_file"] for name in expected["validation"]}
        expected["validation"] = {name for name in expected["validation"] if metadata[name]["src_file"] not in test_sources}
        expected["train"] = {name for name in expected["train"] if metadata[name]["src_file"] not in heldout_sources}
    declared = identity.get("split")
    if not isinstance(declared, dict):
        raise ValueError("Training identity lacks split membership")
    for role, names in expected.items():
        member = declared.get(role, {})
        saved_names = member.get("names", [])
        if len(saved_names) != len(set(saved_names)) or set(saved_names) != names or member.get("n") != len(names):
            raise ValueError(f"Manifest {role} membership disagrees with metadata/source policy")
        if "sources" in member and set(member["sources"]) != {metadata[name]["src_file"] for name in names}:
            raise ValueError(f"Manifest {role} source IDs disagree with metadata")
    return expected


def _audit_history(identity, run, history, training_count):
    config = identity["config"]
    planned_epochs = _integer(config.get("epochs"), "planned epochs", 1)
    batch_size = _integer(config.get("batch_size"), "batch size", 1)
    if len(history) > planned_epochs:
        raise ValueError("Recorded epochs exceed the planned budget")
    previous_updates, previous_seconds, examples = 0, 0., 0
    for i, record in enumerate(history):
        if record.get("epoch") != i + 1:
            raise ValueError("History epochs must be contiguous and start at one")
        seen = _integer(record.get("training_examples"), "training examples", 1)
        if seen > training_count or (not record.get("partial_epoch", False) and seen != training_count):
            raise ValueError("History exposure disagrees with actual training split")
        updates = _integer(record.get("optimizer_updates"), "optimizer updates", 1)
        if updates - previous_updates != math.ceil(seen / batch_size):
            raise ValueError("Optimizer updates disagree with observed batch exposure")
        if record.get("partial_epoch", False) and i != len(history) - 1:
            raise ValueError("Only the final recorded epoch may be partial")
        seconds = _finite(record.get("seconds"), "history seconds", lower=0)
        if seconds < previous_seconds:
            raise ValueError("History elapsed time must be nondecreasing")
        _finite(record.get("training_loss"), "training loss", lower=0)
        for key in ("loss", "accuracy", "macro_f1"):
            _finite(record.get("validation", {}).get(key), f"validation {key}", lower=0,
                    upper=1 if key != "loss" else None)
        previous_updates, previous_seconds = updates, seconds
        examples += seen
    selected = min(history, key=lambda r: r["validation"]["loss"])
    expected_epoch = selected["epoch"]
    if "best_epoch" in run and run["best_epoch"] != expected_epoch:
        raise ValueError("run.best_epoch violates earliest minimum validation-loss selection")
    for record in history:
        if "best_epoch" in record:
            expected_at_epoch = min(history[:record["epoch"]], key=lambda r: r["validation"]["loss"])["epoch"]
            if record["best_epoch"] != expected_at_epoch:
                raise ValueError("History best_epoch violates the declared checkpoint rule")
    for key, expected in (("optimizer_updates", previous_updates), ("epochs_recorded", len(history))):
        if key in run and run[key] != expected:
            raise ValueError(f"run.{key} disagrees with actual history")
    status = run.get("status")
    if status not in ("completed", "time_limit", "failed"):
        raise ValueError(f"Run is not terminal/auditable: {status}")
    if status == "completed" and (len(history) != planned_epochs or any(r.get("partial_epoch", False) for r in history)):
        raise ValueError("Completed status contradicts actual epoch exposure")
    if "seconds" in run and _finite(run["seconds"], "run seconds", lower=0) < previous_seconds:
        raise ValueError("Run elapsed time is shorter than its history")
    if "best_validation_loss" in run and not math.isclose(run["best_validation_loss"], selected["validation"]["loss"], abs_tol=1e-10, rel_tol=0):
        raise ValueError("run.best_validation_loss disagrees with selection")
    return selected, dict(status=status, completed=status == "completed", planned_epochs=planned_epochs,
                         epochs_recorded=len(history), training_examples_exposed=examples,
                         optimizer_updates=previous_updates, partial_epoch=bool(history[-1].get("partial_epoch", False)),
                         seconds=run.get("seconds", previous_seconds))


def _parameter_count(model_name):
    import torch
    import esc50
    # Audit construction must not disturb a caller's subsequent training RNG.
    with torch.random.fork_rng(devices=[]):
        model = esc50.build_model(model_name)
    return sum(parameter.numel() for parameter in model.parameters())


def _checkpoint_identity(folder, expected_epoch, identity):
    checkpoint_path = folder / "best.pt"
    # Exported studies retain compact models beside runs/, without optimizer
    # copies. Audit the artifact used for delivery and held-out evaluation.
    if folder.parent.name == "runs":
        exported = folder.parent.parent / "models" / f"{folder.name}.pt"
        if exported.is_file():
            checkpoint_path = exported
    if not checkpoint_path.is_file():
        return dict(verification="weights unavailable: selection checked from saved history/metrics only")
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != expected_epoch:
        raise ValueError("Saved best.pt epoch violates validation-loss checkpoint selection")
    if checkpoint.get("model_name") != identity["config"]["model"]:
        raise ValueError("Saved checkpoint architecture differs from its identity")
    if "manifest_id" in checkpoint and checkpoint["manifest_id"] != canonical_hash(identity):
        raise ValueError("Checkpoint manifest ID differs from the training identity")
    return dict(verification="checkpoint epoch/model/available manifest checked",
                path=str(checkpoint_path.resolve()), sha256=sha256(checkpoint_path))


def audit_run(run_dir, metadata_path, split="validation", manifest_reference=None):
    """Validate an existing development run or a final-result directory.

    For test artifacts, manifest_reference is the originating training directory
    or its manifest.json/run.json config record. It supplies history, selection
    and immutable identity when available.
    This function reads saved predictions only; it never performs inference.
    """
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    result_dir = Path(run_dir)
    if split == "test":
        stored = read_json(result_dir / "test.json")
        if manifest_reference is None:
            if (result_dir / "manifest.json").is_file() or (result_dir / "run.json").is_file():
                manifest_reference = result_dir
            else:
                raise ValueError("Test artifact requires its originating training manifest/run reference")
        training_dir, identity, run, history, provenance = _training_artifacts(manifest_reference)
    else:
        training_dir, identity, run, history, provenance = _training_artifacts(result_dir)
        stored = read_json(result_dir / "validation.json")
    metadata = _read_metadata(metadata_path)
    expected_splits = _expected_splits(identity, metadata)
    predictions_path = result_dir / f"{split}_predictions.csv"
    predictions = _read_predictions(predictions_path, metadata)
    if {row["filename"] for row in predictions} != expected_splits[split]:
        raise ValueError(f"Saved predictions do not exactly match manifest {split} membership")
    result = recompute_metrics(predictions)
    _assert_metric_match(stored, result)
    selected, exposure = _audit_history(identity, run, history, len(expected_splits["train"]))
    checkpoint = _checkpoint_identity(training_dir, selected["epoch"], identity)
    config = identity["config"]
    if split == "validation":
        for key in ("loss", "accuracy", "macro_f1"):
            if key not in stored or not math.isclose(stored[key], selected["validation"][key], abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"Validation artifact {key} does not belong to selected checkpoint epoch")
    else:
        if stored.get("test_fold") != int(config["test_fold"]):
            raise ValueError("Test-result fold differs from the training identity")
        if "manifest_id" in stored and stored["manifest_id"] != run.get("manifest_id", canonical_hash(identity)):
            raise ValueError("Test-result manifest ID differs from training")
        if "source_policy" in stored and stored["source_policy"] != identity.get("source_policy"):
            raise ValueError("Test-result source policy differs from training")
        if "split" in stored and stored["split"] is not None and stored["split"] != identity.get("split"):
            raise ValueError("Test-result split membership differs from training")
        if checkpoint.get("sha256") and stored.get("checkpoint_sha256") != checkpoint["sha256"]:
            raise ValueError("Test result was not produced from the audited best.pt")
    model_name = config["model"]
    parameters = _integer(identity.get("parameters"), "stored parameters", 1)
    actual_parameters = _parameter_count(model_name)
    if parameters != actual_parameters:
        raise ValueError("Stored parameter count disagrees with the declared model architecture")
    if identity.get("seed", config.get("seed")) != config.get("seed"):
        raise ValueError("Seed differs between identity and config")
    seed = _integer(config.get("seed"), "seed", lower=0)
    metadata_digest = sha256(metadata_path)
    cohort = dict(metadata_sha256=metadata_digest, split=split, names=sorted(expected_splits[split]),
                  source_policy=identity.get("source_policy", "legacy: unspecified"))
    ignored = {"command", "seed", "output", "cache", "max_seconds", "threads", "device"}
    recipe = {key: value for key, value in config.items() if key not in ignored}
    recipe.setdefault("scheduler", "none")
    recipe.setdefault("specaugment", False)
    environment = identity.get("environment", {})
    regime = dict(recipe=recipe, frontend=identity.get("frontend"), cache_sha256=identity.get("cache_sha256"),
                  code_sha256=environment.get("code_sha256"), cohort_id=canonical_hash(cohort),
                  runtime={key: environment.get(key) for key in ("platform", "torch", "numpy", "device")})
    hashes = dict(metadata_sha256=metadata_digest, predictions_sha256=sha256(predictions_path),
                  metrics_sha256=sha256(result_dir / f"{split}.json"),
                  history_sha256=sha256(training_dir / "history.json"),
                  run_sha256=sha256(training_dir / "run.json") if (training_dir / "run.json").exists() else None,
                  manifest_id=run.get("manifest_id"), manifest_sha256=run.get("manifest_sha256"),
                  cache_sha256=identity.get("cache_sha256"), code_sha256=environment.get("code_sha256"))
    if "loss" in stored:
        result["loss"] = stored["loss"]
    return dict(audit_status="passed", run_dir=str(result_dir.resolve()), training_dir=str(training_dir.resolve()),
                split=split, model=model_name, seed=seed, parameters=parameters,
                selected_epoch=selected["epoch"], selected_validation_loss=selected["validation"]["loss"],
                metrics=result, exposure=exposure, provenance=provenance, checkpoint=checkpoint,
                regime=regime, regime_id=canonical_hash(regime), cohort_id=canonical_hash(cohort),
                prediction_records_hash=canonical_hash(sorted(predictions, key=lambda r: r["filename"])),
                hashes=hashes, interpretation="fixed-checkpoint saved predictions; no inference performed during audit",
                loss_audit_scope="loss checked against selected history for validation; cannot recompute cross-entropy from hard labels")


def aggregate(audited_rows):
    """Summarize seeds within exactly matching model/data/recipe/runtime regimes.

    Exact repeated-seed predictions are counted once. Inconsistent repeats are
    retained and flagged; their group is withheld from quantitative aggregation
    rather than selecting the most favorable repetition.
    """
    groups = {}
    for row in audited_rows:
        if row.get("audit_status") != "passed":
            raise ValueError("Only passed audits may enter aggregation")
        key = (row["model"], row["regime_id"], row["split"])
        groups.setdefault(key, []).append(row)
    output = []
    for (model, regime_id, split), rows in sorted(groups.items()):
        by_seed = {}
        for row in rows:
            by_seed.setdefault(row["seed"], []).append(row)
        unique, repeated, conflicts = [], [], []
        for seed, copies in sorted(by_seed.items()):
            fingerprints = {(row["prediction_records_hash"], row["selected_epoch"],
                             row["metrics"].get("loss")) for row in copies}
            if len(copies) > 1:
                repeated.append(dict(seed=seed, n=len(copies), paths=[row["run_dir"] for row in copies]))
            if len(fingerprints) > 1:
                conflicts.append(seed)
            unique.append(copies[0])
        summary = {}
        if not conflicts:
            for key in ("accuracy", "macro_f1", "loss"):
                values = [row["metrics"][key] for row in unique if key in row["metrics"]]
                if values:
                    summary[key] = dict(mean=float(np.mean(values)),
                                        sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                        n=len(values))
        exposures = {(row["exposure"]["epochs_recorded"], row["exposure"]["training_examples_exposed"],
                      row["exposure"]["optimizer_updates"], row["exposure"]["partial_epoch"]) for row in rows}
        output.append(dict(model=model, split=split, regime_id=regime_id, cohort_id=rows[0]["cohort_id"],
                           n_runs=len(rows), n_unique_seeds=len(by_seed), seeds=sorted(by_seed),
                           repeated_seeds=repeated, conflicting_repeated_seeds=conflicts,
                           summary_status="withheld: conflicting repeated-seed results" if conflicts else "summarized",
                           metrics=summary, parameters=rows[0]["parameters"],
                           all_runs_completed=all(row["exposure"]["completed"] for row in rows),
                           actual_exposure_consistent=len(exposures) == 1,
                           regime=rows[0]["regime"],
                           statistical_unit="unique configured seed on one shared evaluation cohort",
                           uncertainty_scope="sample SD across unique seeds only; same recordings reused across seeds; no pooling of prediction rows and no independent-dataset claim",
                           warning="N=1 gives no sample SD; repeated seeds do not increase N. Equal recipe does not establish all sources of uncertainty."))
    return output


def _markdown_report(rows, groups, errors, split):
    lines = [f"# Audited ESC-50 {split} results", "", "All numbers below were recalculated from existing saved predictions. No inference was run.", "",
             "| Model | Seed | Accuracy | Macro-F1 | Selected epoch | Recorded epochs | Examples exposed | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        metrics, exposure = row["metrics"], row["exposure"]
        lines.append(f"| {row['model']} | {row['seed']} | {metrics['accuracy']*100:.2f}% | {metrics['macro_f1']*100:.2f}% | {row['selected_epoch']} | {exposure['epochs_recorded']} | {exposure['training_examples_exposed']} | {exposure['status']} |")
    lines += ["", "## Regime-specific seed summaries", "", "Distinct data/code/runtime/recipe regimes are never merged. Standard deviations describe seed variation on the same recordings, not independent datasets.", "",
              "| Model | Regime | Unique seeds | Accuracy mean ± sample SD | Macro-F1 mean ± sample SD | Completed / exposure matched |",
              "|---|---|---:|---:|---:|---|"]
    for group in groups:
        def display(key):
            value = group["metrics"].get(key)
            if value is None:
                return "withheld"
            return f"{value['mean']*100:.2f}% ± " + (f"{value['sample_std']*100:.2f} pp" if value['sample_std'] is not None else "unavailable (N=1)")
        lines.append(f"| {group['model']} | {group['regime_id'][:10]} | {group['n_unique_seeds']} | {display('accuracy')} | {display('macro_f1')} | {group['all_runs_completed']} / {group['actual_exposure_consistent']} |")
    lines += ["", "Failed audits are excluded from summaries and retained below. Missing checkpoint weights weaken selection verification and are explicitly labeled in audit.json. Modified source-isolated results are not automatically comparable with official five-fold scores."]
    if errors:
        lines += ["", "## Failed audits", ""] + [f"- `{error['run_dir']}`: {error['error']}" for error in errors]
    return "\n".join(lines) + "\n"


def create_report(run_dirs, metadata_path, output, split="validation", manifests=None):
    if manifests is not None and len(manifests) != len(run_dirs):
        raise ValueError("Supply one training manifest/run reference per test-result directory")
    rows, errors = [], []
    for i, path in enumerate(run_dirs):
        try:
            rows.append(audit_run(path, metadata_path, split, manifests[i] if manifests else None))
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(dict(run_dir=str(path), error_type=type(error).__name__, error=str(error)))
    groups = aggregate(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(schema_version=1, split=split, audited_runs=rows, groups=groups, audit_errors=errors,
                   analysis_code_sha256=sha256(__file__), metadata_sha256=sha256(metadata_path))
    (output / "audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(_markdown_report(rows, groups, errors, split))
    with open(output / "runs.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "seed", "split", "accuracy", "macro_f1", "n", "parameters", "selected_epoch",
                         "recorded_epochs", "training_examples_exposed", "optimizer_updates", "status", "regime_id", "run_dir"))
        for row in rows:
            writer.writerow((row["model"], row["seed"], row["split"], row["metrics"]["accuracy"], row["metrics"]["macro_f1"],
                             row["metrics"]["n"], row["parameters"], row["selected_epoch"], row["exposure"]["epochs_recorded"],
                             row["exposure"]["training_examples_exposed"], row["exposure"]["optimizer_updates"],
                             row["exposure"]["status"], row["regime_id"], row["run_dir"]))
    with open(output / "groups.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "split", "regime_id", "n_runs", "n_unique_seeds", "accuracy_mean",
                         "accuracy_sample_std", "macro_f1_mean", "macro_f1_sample_std", "summary_status",
                         "all_runs_completed", "actual_exposure_consistent"))
        for group in groups:
            accuracy, f1 = group["metrics"].get("accuracy", {}), group["metrics"].get("macro_f1", {})
            writer.writerow((group["model"], group["split"], group["regime_id"], group["n_runs"], group["n_unique_seeds"],
                             accuracy.get("mean"), accuracy.get("sample_std"), f1.get("mean"), f1.get("sample_std"),
                             group["summary_status"], group["all_runs_completed"], group["actual_exposure_consistent"]))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("development", "holdout"):
        p = sub.add_parser(name)
        p.add_argument("--runs", nargs="+", required=True)
        p.add_argument("--metadata", required=True)
        p.add_argument("--output", required=True)
        if name == "holdout":
            p.add_argument("--manifests", nargs="+")
    args = parser.parse_args(argv)
    try:
        payload = create_report(args.runs, args.metadata, args.output,
                                "validation" if args.command == "development" else "test",
                                getattr(args, "manifests", None))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(dict(audited_runs=len(payload["audited_runs"]), groups=len(payload["groups"]),
                          failed_audits=len(payload["audit_errors"]), output=args.output), indent=2))
    if payload["audit_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
