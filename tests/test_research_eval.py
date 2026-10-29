"""Read-only evaluation evidence tests; no model checkpoint loads and no WAV decoding."""
import csv
import json
from pathlib import Path

import pytest

from esc50_sound.data import read_metadata, split_rows
from esc50_sound.evaluation import compute_metrics
from esc50_sound.research_eval import Protocol, audit_predictions, compare_predictions, _public
from esc50_sound.records import write_json, sha256

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "ESC-50"


def protocol(*, partition="validation", stage="development_validation", source_policy="source-strict"):
    return Protocol(stage=stage, partition=partition, train_folds=(1,2,3), val_fold=4,
                    test_fold=5, source_policy=source_policy,
                    dataset_identity="ESC-50-test-fixture-metadata-2026",
                    metadata_sha256=sha256(DATA_ROOT / "meta" / "esc50.csv"),
                    selection_disclosure="This dataset was used for previous development and is not independent")


def predictions(tmp_path, part="validation", wrong=3, source_policy="source-strict"):
    metadata = read_metadata(DATA_ROOT)
    idx = split_rows(metadata, (1,2,3), 4, 5, strict=source_policy=="source-strict")
    rows = []
    for j, i in enumerate(idx[part]):
        row = metadata[i]
        label = int(row["target"])
        prediction = (label + 1) % 50 if j < wrong else label
        rows.append(dict(filename=row["filename"], target=label, prediction=prediction))
    p = tmp_path / f"{part}_{wrong}.csv"
    with p.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "target", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    return p, rows


def save(path, rows):
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "target", "prediction"])
        w.writeheader()
        w.writerows(rows)


def test_dev_membership_metrics_and_stable_digest(tmp_path):
    path, rows = predictions(tmp_path)
    result = audit_predictions(path, DATA_ROOT, protocol())
    assert result["n"] == 398
    assert result["accuracy"] == pytest.approx(395 / 398)
    assert not result["test_accessed"]
    assert result["confirmation_status"] == "not independently confirmed"
    assert "_ordered" not in _public(result)
    assert audit_predictions(path, DATA_ROOT, protocol())["sample_identity_sha256"] == result["sample_identity_sha256"]


def test_duplicate_missing_extra_and_wrong_label_rejected(tmp_path):
    path, rows = predictions(tmp_path)
    bad = tmp_path / "bad.csv"
    save(bad, rows[:-1] + rows[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        audit_predictions(bad, DATA_ROOT, protocol())
    save(bad, rows[:-1])
    with pytest.raises(ValueError, match="membership"):
        audit_predictions(bad, DATA_ROOT, protocol())
    swapped = [dict(x) for x in rows]
    swapped[0]["target"] = (swapped[0]["target"] + 1) % 50
    save(bad, swapped)
    with pytest.raises(ValueError, match="label mismatch"):
        audit_predictions(bad, DATA_ROOT, protocol())


def test_bad_header_and_out_of_range_rejected(tmp_path):
    path, rows = predictions(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("filename,target,prediction,confidence\n")
    with pytest.raises(ValueError, match="exactly"):
        audit_predictions(bad, DATA_ROOT, protocol())
    altered = [dict(x) for x in rows]
    altered[0]["prediction"] = 50
    save(bad, altered)
    with pytest.raises(ValueError, match="out of range"):
        audit_predictions(bad, DATA_ROOT, protocol())


def test_saved_metrics_verified_not_trusted(tmp_path):
    path, rows = predictions(tmp_path)
    counts = compute_metrics([r["target"] for r in rows], [r["prediction"] for r in rows])
    report = tmp_path / "metrics.json"
    write_json(report, counts)
    assert audit_predictions(path, DATA_ROOT, protocol(), metrics_file=report)["valid"]
    counts["accuracy"] = 1.0
    write_json(report, counts)
    with pytest.raises(ValueError, match="Saved accuracy"):
        audit_predictions(path, DATA_ROOT, protocol(), metrics_file=report)


def test_heldout_access_fail_closed_before_reading_predictions(tmp_path):
    missing = tmp_path / "nonexistent.csv"
    p = protocol(partition="test", stage="exploratory_holdout")
    with pytest.raises(PermissionError, match="explicit"):
        audit_predictions(missing, DATA_ROOT, p)
    path, _ = predictions(tmp_path, part="test")
    result = audit_predictions(path, DATA_ROOT, p, allow_heldout=True)
    assert result["n"] == 400 and result["test_accessed"]
    assert result["protocol"]["stage"] == "exploratory_holdout"


def test_protocol_rejects_independent_self_certification(tmp_path):
    p = protocol(partition="test", stage="independent_confirmation")
    with pytest.raises(ValueError, match="cannot be self-certified"):
        p.validate()
    bad = tmp_path / "protocol.json"
    write_json(bad, dict(schema_version=1, stage="development_validation", partition="test",
                         train_folds=[1,2,3], val_fold=4, test_fold=5, source_policy="official",
                         dataset_identity="ESC-50-dataset", metadata_sha256=sha256(DATA_ROOT / "meta" / "esc50.csv"), selection_disclosure="previous tuning used fold 5"))
    with pytest.raises(ValueError, match="Development validation"):
        Protocol.from_file(bad)


def test_paired_cluster_bootstrap_deterministic_and_order_independent(tmp_path):
    left, a = predictions(tmp_path, wrong=4)
    right, b = predictions(tmp_path, wrong=2)
    flipped = tmp_path / "flipped.csv"
    save(flipped, list(reversed(b)))
    result = compare_predictions(left, flipped, DATA_ROOT, protocol(), repetitions=300, seed=13)
    assert result["delta_accuracy"] == pytest.approx(2/398)
    assert result["right_only_correct"] == 2
    assert result["left_only_correct"] == 0
    assert result["both_correct"] == 394
    again = compare_predictions(left, flipped, DATA_ROOT, protocol(), repetitions=300, seed=13)
    assert result["bootstrap"] == again["bootstrap"]
    assert result["bootstrap"]["groups"] <= 398


def test_source_policy_controls_exact_expected_validation_membership(tmp_path):
    strict, _ = predictions(tmp_path)
    with pytest.raises(ValueError, match="membership"):
        audit_predictions(strict, DATA_ROOT, protocol(source_policy="official"))
    official, _ = predictions(tmp_path, source_policy="official", wrong=2)
    assert audit_predictions(official, DATA_ROOT, protocol(source_policy="official"))["n"] == 400


def test_real_archived_development_predictions_can_be_audited():
    path = Path(__file__).resolve().parents[1] / 'results/historical/final_study/results/statistics_baseline/validation_predictions_l2_0.1.csv'
    result = audit_predictions(path, DATA_ROOT, protocol())
    assert result["n"] == 398


def test_historical_heldout_pairing_requires_flag():
    root = Path(__file__).resolve().parents[1] / 'results/historical/final_study/results/holdout'
    a = root / "cnn_freq_seed101_60ep"
    b = root / "crnn_freq_seed101_60ep"
    # Fail closed and do not read held-out content as part of ordinary test suite.
    with pytest.raises(PermissionError):
        compare_predictions(a / "test_predictions.csv", b / "test_predictions.csv", DATA_ROOT,
                            protocol(partition="test", stage="exploratory_holdout"))


def test_protocol_fails_on_wrong_dataset_hash(tmp_path):
    from dataclasses import replace
    path, _ = predictions(tmp_path)
    altered = replace(protocol(), metadata_sha256="0"*64)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        audit_predictions(path, DATA_ROOT, altered)


def test_run_link_detects_real_historical_identity_and_forged_metric(tmp_path):
    from esc50_sound.research_eval import audit_run_link
    study = Path(__file__).resolve().parents[1] / 'results/historical/final_study'
    exp = "cnn_freq_seed101_60ep"
    pred = study / "results" / "holdout" / exp / "test_predictions.csv"
    metric = study / "results" / "holdout" / exp / "test.json"
    run = study / 'runs' / exp
    weights = study / "models" / (exp + ".pt")
    audited = audit_predictions(pred, DATA_ROOT, protocol(partition="test", stage="exploratory_holdout"),
                                metrics_file=metric, allow_heldout=True)
    good = audit_run_link(run, audited, metrics_file=metric, checkpoint=weights)
    assert good["linked"] and good["checkpoint_bytes_verified"]
    tampered = json.loads(metric.read_text())
    tampered["manifest_id"] = "0"*64
    altered = tmp_path / "altered.json"
    write_json(altered, tampered)
    bad = audit_run_link(run, audited, metrics_file=altered, checkpoint=weights)
    assert not bad["linked"]


def test_five_fold_cv_is_explicit_and_retrospective():
    from esc50_sound.research_eval import summarize_cv
    root = Path(__file__).resolve().parents[1]
    plan = root / "configs" / "eval_historical_cv_cnn_freq_seed101.json"
    with pytest.raises(PermissionError):
        summarize_cv(plan, DATA_ROOT)
    result = summarize_cv(plan, DATA_ROOT, allow_heldout=True)
    assert result["n"] == 2000
    assert result["mean_fold_accuracy"] == pytest.approx(.665)
    assert len(result["test_fold_results"]) == 5
    assert "NOT preregistered" in result["assessment"]


def test_cv_rejects_duplicate_fold_plan_before_reading_test(tmp_path):
    from esc50_sound.research_eval import summarize_cv
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/"configs"/"eval_historical_cv_cnn_freq_seed101.json").read_text())
    plan["folds"][4]["test_fold"] = 4
    fp = tmp_path / "invalid_cv.json"
    write_json(fp, plan)
    with pytest.raises(ValueError, match="unique"):
        summarize_cv(fp, DATA_ROOT, allow_heldout=True)


def test_new_evidence_writer_never_overwrites(tmp_path):
    from esc50_sound.records import write_json_new
    path = tmp_path/"audit.json"
    write_json_new(path, {"old": 1})
    with pytest.raises(FileExistsError):
        write_json_new(path, {"new": 2})
    assert json.loads(path.read_text()) == {"old": 1}
    assert not list(tmp_path.glob("*.tmp"))
