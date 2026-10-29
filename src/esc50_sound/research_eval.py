"""Read-only, fail-closed prediction evidence and paired research comparison.

Never imports/executes checkpoint payloads, trains a model, or selects a model.
A CLI flag permits auditing *already accessed* held-out predictions; it does not
make those results independent confirmation. Proving independence requires an
external governance process and a genuinely untouched data source.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import audit_metadata, read_metadata, split_rows
from .evaluation import compute_metrics
from .infra import inspect_run, compatibility
from .records import canonical_hash, sha256

STAGES = ("development_validation", "exploratory_holdout", "fixed_protocol_cv", "independent_confirmation", "robustness")


@dataclass(frozen=True)
class Protocol:
    stage: str
    partition: str
    train_folds: tuple[int, ...]
    val_fold: int
    test_fold: int
    source_policy: str
    dataset_identity: str
    metadata_sha256: str
    selection_disclosure: str
    # A user-declared string never establishes independence or preregistration.
    preregistration_ref: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "Protocol":
        document = json.loads(Path(path).read_text())
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ValueError("Unsupported evaluation protocol schema")
        allowed = {"schema_version", *cls.__dataclass_fields__}
        if set(document) - allowed:
            raise ValueError(f"Unknown evaluation protocol keys: {set(document) - allowed}")
        required = set(cls.__dataclass_fields__) - {"preregistration_ref"}
        if not required <= set(document):
            raise ValueError(f"Missing protocol keys: {required - set(document)}")
        obj = cls(**{k: (tuple(v) if k == "train_folds" else v)
                     for k, v in document.items() if k != "schema_version"})
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.stage not in STAGES:
            raise ValueError("Unknown protocol stage")
        if self.partition not in ("validation", "test"):
            raise ValueError("Unknown partition")
        if self.stage == "development_validation" and self.partition != "validation":
            raise ValueError("Development validation must use the validation partition")
        if self.stage in ("exploratory_holdout", "fixed_protocol_cv", "independent_confirmation") and self.partition != "test":
            raise ValueError("Holdout/CV/confirmation stages must use the test partition")
        if self.source_policy not in ("source-strict", "official"):
            raise ValueError("Unknown source policy")
        if not isinstance(self.dataset_identity, str) or len(self.dataset_identity.strip()) < 8:
            raise ValueError("dataset_identity must name a versioned, inspectable dataset")
        if not isinstance(self.metadata_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.metadata_sha256) is None:
            raise ValueError("metadata_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.selection_disclosure, str) or len(self.selection_disclosure.strip()) < 10:
            raise ValueError("Explicit selection-history disclosure is required")
        if self.stage == "independent_confirmation":
            # A JSON file cannot self-certify that test data were untouched.
            raise ValueError("Independent confirmation cannot be self-certified by this local evaluator")
        split_rows([{"fold": 1}], self.train_folds, self.val_fold, self.test_fold)

    def public(self) -> dict[str, Any]:
        return {"schema_version": 1, "stage": self.stage, "partition": self.partition,
                "train_folds": list(self.train_folds), "val_fold": self.val_fold,
                "test_fold": self.test_fold, "source_policy": self.source_policy,
                "dataset_identity": self.dataset_identity, "metadata_sha256": self.metadata_sha256,
                "selection_disclosure": self.selection_disclosure,
                "preregistration_ref": self.preregistration_ref}


def _read_predictions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["filename", "target", "prediction"]:
            raise ValueError("Prediction CSV must have exactly filename,target,prediction columns")
        records = []
        for n, row in enumerate(reader, start=2):
            if None in row or not row["filename"] or not row["target"] or not row["prediction"]:
                raise ValueError(f"Missing or extra prediction field on line {n}")
            if row["filename"] != Path(row["filename"]).name or not row["filename"].endswith(".wav"):
                raise ValueError(f"Unsafe prediction filename on line {n}")
            try:
                target, predicted = int(row["target"]), int(row["prediction"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid target/prediction on line {n}") from exc
            if not (0 <= target < 50 and 0 <= predicted < 50):
                raise ValueError(f"Class ID out of range on line {n}")
            records.append({"filename": row["filename"], "target": target, "prediction": predicted})
    if not records:
        raise ValueError("Empty prediction file")
    return records


def audit_predictions(predictions: str | Path, metadata_root: str | Path, protocol: Protocol,
                      *, metrics_file: str | Path | None = None,
                      allow_heldout: bool = False) -> dict[str, Any]:
    """Validate exact file membership and labels, recompute metrics; no WAV or checkpoint needed."""
    protocol.validate()
    if protocol.partition == "test" and not allow_heldout:
        # Gate runs before opening prediction CSV or stored evaluation metrics.
        raise PermissionError("Test prediction access requires explicit --allow-holdout; it is not independent confirmation")
    real_metadata_hash = sha256(Path(metadata_root) / "meta" / "esc50.csv")
    if real_metadata_hash != protocol.metadata_sha256:
        raise ValueError("Dataset metadata SHA-256 differs from frozen evaluation protocol")
    rows = read_metadata(metadata_root)
    audit_metadata(rows)
    strict = protocol.source_policy == "source-strict"
    indices = split_rows(rows, protocol.train_folds, protocol.val_fold, protocol.test_fold, strict=strict)
    expected = {rows[i]["filename"]: rows[i] for i in indices[protocol.partition]}
    records = _read_predictions(predictions)
    names = [r["filename"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate prediction filenames")
    if set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(f"Prediction membership mismatch: {len(missing)} missing, {len(extra)} extra; first: {missing[:2]}, {extra[:2]}")
    for row in records:
        expected_target = int(expected[row["filename"]]["target"])
        if row["target"] != expected_target:
            raise ValueError(f"Metadata label mismatch: {row['filename']}")
    # Canonical metadata order ensures paired vectors line up even if CSV rows differ in order.
    by_name = {r["filename"]: r for r in records}
    ordered = [by_name[name] for name in sorted(expected)]
    metrics = compute_metrics([r["target"] for r in ordered], [r["prediction"] for r in ordered])
    if metrics_file is not None:
        stored = json.loads(Path(metrics_file).read_text())
        if not isinstance(stored, dict):
            raise ValueError("Stored metrics is not a JSON object")
        for key in ("accuracy", "macro_f1"):
            if key not in stored or not np.isfinite(stored[key]) or not np.isclose(stored[key], metrics[key], atol=1e-10, rtol=0):
                raise ValueError(f"Saved {key} does not match predictions")
        if stored.get("n") != len(ordered):
            raise ValueError("Saved sample count does not match predictions")
        if "confusion_matrix" in stored and stored["confusion_matrix"] != metrics["confusion_matrix"]:
            raise ValueError("Saved confusion matrix does not match predictions")
    return {"schema_version": 1, "valid": True, "protocol": protocol.public(),
            "protocol_sha256": canonical_hash(protocol.public()),
            "metadata_sha256": real_metadata_hash,
            "predictions_sha256": sha256(predictions),
            "metrics_sha256": sha256(metrics_file) if metrics_file else None,
            "sample_identity_sha256": canonical_hash(sorted(expected)),
            "n": len(ordered), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "assessment": "read-only prediction integrity, NOT independent reproduction of inference",
            "confirmation_status": "not independently confirmed",
            "test_accessed": protocol.partition == "test",
            "_ordered": ordered,
            "_sources": [str(expected[r["filename"]]["src_file"]) for r in ordered]}


def audit_run_link(run_dir: str | Path, audit: dict[str, Any], *, metrics_file=None, checkpoint=None) -> dict[str, Any]:
    """Check run identity for a prediction artifact; does NOT execute its model.

    Checkpoint hash requires caller to provide the exact saved checkpoint path.
    A matched manifest is evidence of consistency, not proof that inference ran.
    """
    run = inspect_run(run_dir)
    failures = []
    warnings = []
    if not run.get("valid") or not run.get("completed"):
        failures.append("Training run missing, invalid, or incomplete")
    if not run.get("manifest_id"):
        failures.append("Run manifest ID unavailable")
    if metrics_file is None:
        failures.append("Evaluation metrics with run manifest ID required")
        stored = {}
    else:
        stored = json.loads(Path(metrics_file).read_text())
        if stored.get("manifest_id") != run.get("manifest_id"):
            failures.append("Evaluation manifest ID does not match training run")
        if "test_fold" in stored and stored["test_fold"] != audit["protocol"]["test_fold"]:
            failures.append("Evaluation test fold differs from declared protocol")
    if checkpoint is not None:
        if not stored.get("checkpoint_sha256") or sha256(checkpoint) != stored["checkpoint_sha256"]:
            failures.append("Checkpoint file does not match evaluation metrics digest")
    else:
        warnings.append("Checkpoint bytes not independently verified (optional --checkpoint)")
    return dict(linked=not failures, run=str(run_dir), model=run.get("model"),
                seed=run.get("seed"), reasons=failures, warnings=warnings,
                checkpoint_bytes_verified=checkpoint is not None and not any("Checkpoint" in x for x in failures),
                training_valid=run.get("valid", False),
                scope="static manifest and byte hash consistency; no model execution")


def _public(audit: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in audit.items() if not k.startswith("_")}


def compare_predictions(left: str | Path, right: str | Path, metadata_root: str | Path,
                        protocol: Protocol, *, allow_heldout=False, repetitions=2000,
                        seed=2026, left_metrics=None, right_metrics=None,
                        left_run=None, right_run=None, left_checkpoint=None, right_checkpoint=None) -> dict[str, Any]:
    """Paired cluster bootstrap over original recording IDs, not 400 iid clips.

    CI: percentile interval for paired difference, conditional on the supplied
    fixed predictions; it is not uncertainty across training seeds or datasets.
    """
    if not isinstance(repetitions, int) or not 200 <= repetitions <= 100_000:
        raise ValueError("Bootstrap repetitions must be in [200,100000]")
    a = audit_predictions(left, metadata_root, protocol, metrics_file=left_metrics, allow_heldout=allow_heldout)
    b = audit_predictions(right, metadata_root, protocol, metrics_file=right_metrics, allow_heldout=allow_heldout)
    if a["sample_identity_sha256"] != b["sample_identity_sha256"] or a["metadata_sha256"] != b["metadata_sha256"]:
        raise ValueError("Samples or dataset versions are not identical")
    aa, bb = a["_ordered"], b["_ordered"]
    if [r["target"] for r in aa] != [r["target"] for r in bb]:
        raise ValueError("Targets not aligned")
    correct_a = np.asarray([x["target"] == x["prediction"] for x in aa], dtype=np.float64)
    correct_b = np.asarray([x["target"] == x["prediction"] for x in bb], dtype=np.float64)
    difference = correct_b - correct_a
    if (left_run is None) != (right_run is None):
        raise ValueError("Both run directories are required for a run-level comparison")
    run_gate = None
    run_links = None
    if left_run is not None:
        left_link = audit_run_link(left_run, a, metrics_file=left_metrics, checkpoint=left_checkpoint)
        right_link = audit_run_link(right_run, b, metrics_file=right_metrics, checkpoint=right_checkpoint)
        run_gate = compatibility(inspect_run(left_run), inspect_run(right_run))
        run_links = dict(left=left_link, right=right_link)
        if not left_link["linked"] or not right_link["linked"] or not run_gate["eligible"]:
            raise ValueError("Run provenance or comparable-training protocol check failed: "
                             + str((left_link, right_link, run_gate)))
    groups: dict[str, list[int]] = {}
    for i, source in enumerate(a["_sources"]):
        groups.setdefault(source, []).append(i)
    source_list = list(groups)
    rng = np.random.default_rng(seed)
    sampled = np.empty(repetitions, dtype=np.float64)
    # Draw groups with replacement, keeping each group's original sample count.
    for i in range(repetitions):
        selected = rng.integers(0, len(source_list), size=len(source_list))
        idx = np.concatenate([groups[source_list[j]] for j in selected])
        sampled[i] = float(difference[idx].mean())
    low, high = np.quantile(sampled, [.025, .975])
    return {"schema_version": 1, "valid": True, "protocol": protocol.public(),
            "comparison": "right minus left, paired on exact filenames and labels",
            "left": _public(a), "right": _public(b),
            "run_comparability": run_gate, "run_links": run_links,
            "delta_accuracy": float(difference.mean()),
            "right_only_correct": int(((correct_a == 0) & (correct_b == 1)).sum()),
            "left_only_correct": int(((correct_a == 1) & (correct_b == 0)).sum()),
            "both_correct": int(((correct_a == 1) & (correct_b == 1)).sum()),
            "both_incorrect": int(((correct_a == 0) & (correct_b == 0)).sum()),
            "bootstrap": {"unit": "src_file", "groups": len(groups), "seed": seed,
                          "repetitions": repetitions, "confidence_level": .95,
                          "percentile_ci": [float(low), float(high)],
                          "scope": "fixed paired predictions only; no random-seed or unseen-dataset uncertainty"},
            "caution": "Integrity and pairing do not prove independent confirmation. Run provenance, selection history and training-seed uncertainty still matter. This is not a significance claim."}


def summarize_cv(manifest_path: str | Path, metadata_root: str | Path, *,
                 allow_heldout: bool = False) -> dict[str, Any]:
    """Read-only five-fold *retrospective* audit; does not certify official CV.

    All five folds must belong to one model and seed; reject duplicated
    prediction membership, mixed code/cache versions, or absent checkpoints.
    """
    if not allow_heldout:
        raise PermissionError("CV test predictions require explicit --allow-holdout")
    plan = json.loads(Path(manifest_path).read_text())
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or not all(
            k in plan for k in ("model", "seed", "metadata_sha256", "dataset_identity",
                              "selection_disclosure", "folds")):
        raise ValueError("Incomplete CV plan")
    if not isinstance(plan["folds"], list) or len(plan["folds"]) != 5:
        raise ValueError("CV audit needs exactly 5 fold records")
    if sorted(e.get("test_fold") for e in plan["folds"]) != [1, 2, 3, 4, 5]:
        raise ValueError("CV folds must be unique and exactly 1..5")
    if sha256(Path(metadata_root) / "meta" / "esc50.csv") != plan["metadata_sha256"]:
        raise ValueError("CV metadata hash mismatch")
    source_ids, all_names, fold_results = [], set(), []
    settings, training_signatures = [], set()
    total_confusion = np.zeros((50, 50), dtype=np.int64)
    for entry in sorted(plan["folds"], key=lambda e: e["test_fold"]):
        needed = {"test_fold", "val_fold", "train_folds", "source_policy",
                  "predictions", "metrics", "run_dir", "checkpoint"}
        if needed - set(entry):
            raise ValueError(f"CV fold missing fields: {needed - set(entry)}")
        fold = entry["test_fold"]
        proto = Protocol(stage="fixed_protocol_cv", partition="test",
                         train_folds=tuple(entry["train_folds"]), val_fold=entry["val_fold"],
                         test_fold=fold, source_policy=entry["source_policy"],
                         dataset_identity=plan["dataset_identity"],
                         metadata_sha256=plan["metadata_sha256"],
                         selection_disclosure=plan["selection_disclosure"])
        audited = audit_predictions(entry["predictions"], metadata_root, proto,
                                    metrics_file=entry["metrics"], allow_heldout=True)
        linked = audit_run_link(entry["run_dir"], audited, metrics_file=entry["metrics"],
                                checkpoint=entry["checkpoint"])
        if not linked["linked"] or not linked["checkpoint_bytes_verified"]:
            raise ValueError(f"Fold {fold} failed training/checkpoint association")
        if linked["model"] != plan["model"] or linked["seed"] != plan["seed"]:
            raise ValueError(f"Fold {fold}: model or seed does not match CV plan")
        run = inspect_run(entry["run_dir"])
        if run["training_config"].get("test_fold") != fold:
            raise ValueError(f"Fold {fold}: run declares another test fold")
        if run["training_config"].get("val_fold") != entry["val_fold"]:
            raise ValueError(f"Fold {fold}: run declares another validation fold")
        if run["training_config"].get("train_folds") != ",".join(map(str, entry["train_folds"])):
            raise ValueError(f"Fold {fold}: run declares other training folds")
        expect_official = entry["source_policy"] == "official"
        if run["training_config"].get("official_folds") != expect_official:
            raise ValueError(f"Fold {fold}: source policy differs from training run")
        signatures = (run["cache_sha256"], run["frontend_fingerprint"],
                      run["code_sha256"], run["source_policy"])
        if any(not x for x in signatures):
            raise ValueError("Unverifiable training provenance")
        training_signatures.add(signatures)
        common = tuple((k,run["training_config"].get(k)) for k in ("epochs", "lr", "batch_size", "scheduler", "specaugment"))
        settings.append(common)
        names = {r["filename"] for r in audited["_ordered"]}
        if all_names & names:
            raise ValueError("Same prediction filename appears in multiple test folds")
        all_names |= names
        source_ids.extend(audited["_sources"])
        total_confusion += np.asarray(audited["confusion_matrix"], dtype=np.int64)
        fold_results.append(dict(test_fold=fold, n=audited["n"],
                                 accuracy=audited["accuracy"], macro_f1=audited["macro_f1"],
                                 predictions_sha256=audited["predictions_sha256"],
                                 checkpoint_sha256=sha256(entry["checkpoint"]),
                                 run_manifest_id=run["manifest_id"]))
    if len(training_signatures) != 1 or len(set(settings)) != 1:
        raise ValueError("CV folds have different source data, code or training settings")
    metadata = read_metadata(metadata_root)
    if all_names != {r["filename"] for r in metadata} or len(all_names) != 2000:
        raise ValueError("CV union is not exactly the full 2,000-file dataset")
    per_fold = np.asarray([r["accuracy"] for r in fold_results])
    macro = np.asarray([r["macro_f1"] for r in fold_results])
    tp = np.diag(total_confusion).astype(float)
    denominator = total_confusion.sum(axis=0) + total_confusion.sum(axis=1)
    pooled_f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    return {"schema_version": 1, "valid": True, "model": plan["model"], "seed": plan["seed"],
            "assessment": "retrospective fixed-recipe 5-fold evidence audit, NOT preregistered independent confirmation",
            "protocol": "three training folds + separate validation fold, not ESC-50 standard four training folds",
            "selection_disclosure": plan["selection_disclosure"],
            "metadata_sha256": plan["metadata_sha256"], "n": int(total_confusion.sum()),
            "overall_accuracy": float(tp.sum()/total_confusion.sum()),
            "macro_f1_from_pooled_confusion": float(pooled_f1.mean()),
            "mean_fold_accuracy": float(per_fold.mean()),
            "sd_fold_accuracy_descriptive": float(per_fold.std(ddof=1)),
            "mean_fold_macro_f1": float(macro.mean()),
            "test_fold_results": fold_results,
            "num_unique_test_sources": len(set(source_ids)),
            "warnings": ["CV folds share training data and are not independent replicates",
                         "Historical development accessed folds prior to this retrospective audit",
                         "Some original source IDs may appear in more than one official fold; source-strict training removes overlaps"],
            "test_metrics_read": True}
