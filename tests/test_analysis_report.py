"""Saved-artifact audits reject corruption and do not inflate repeated seeds."""
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import analysis_report as a


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@pytest.fixture
def artifact(tmp_path):
    metadata = tmp_path / "metadata.csv"
    rows = [dict(filename=f"{name}.wav", target=i % 2, fold=fold, src_file=name, category=f"class_{i%2}")
            for i, (name, fold) in enumerate(zip("abcdefgh", (1, 2, 3, 3, 4, 4, 5, 5)))]
    with open(metadata, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filename", "target", "fold", "src_file", "category"))
        writer.writeheader(); writer.writerows(rows)
    folder = tmp_path / "run"; folder.mkdir()
    split = {role: dict(n=len(names), names=[f"{n}.wav" for n in names], sources=list(names))
             for role, names in (("train", "abcd"), ("validation", "ef"), ("test", "gh"))}
    identity = dict(schema_version=2,
                    config=dict(model="cnn", seed=7, epochs=2, batch_size=2, lr=.001,
                                train_folds="1,2,3", val_fold=4, test_fold=5, official_folds=False),
                    seed=7, split=split, parameters=26658, cache_sha256="a" * 64,
                    frontend=dict(sample_rate=16000), source_policy="source_strict",
                    environment=dict(code_sha256="b" * 64, device="cpu", torch="fixture", numpy="fixture", platform="fixture"))
    write_json(folder / "manifest.json", identity)
    run = dict(identity, manifest_id=a.canonical_hash(identity), manifest_sha256=a.sha256(folder / "manifest.json"),
               status="completed", best_epoch=2, best_validation_loss=1., optimizer_updates=4, epochs_recorded=2, seconds=2.2)
    write_json(folder / "run.json", run)
    predictions = [dict(filename="e.wav", target=0, prediction=0), dict(filename="f.wav", target=1, prediction=0)]
    with open(folder / "validation_predictions.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filename", "target", "prediction"))
        writer.writeheader(); writer.writerows(predictions)
    metric = a.recompute_metrics(predictions); metric["loss"] = 1.
    write_json(folder / "validation.json", metric)
    history = [dict(epoch=i, training_loss=2./i, training_examples=4, optimizer_updates=2*i,
                    validation=dict(loss=loss, accuracy=.5, macro_f1=(2/3)/50),
                    seconds=float(i), partial_epoch=False, best_epoch=i)
               for i, loss in ((1, 1.5), (2, 1.))]
    write_json(folder / "history.json", history)
    return folder, metadata


def test_hand_calculated_metrics_and_complete_run_audit(artifact):
    folder, metadata = artifact
    result = a.audit_run(folder, metadata)
    assert result["metrics"]["accuracy"] == .5
    assert result["metrics"]["macro_f1"] == pytest.approx((2/3)/50)
    assert result["metrics"]["confusion_matrix"][0][0] == 1
    assert result["metrics"]["confusion_matrix"][1][0] == 1
    assert result["selected_epoch"] == 2
    assert result["parameters"] == 26658
    assert result["exposure"]["training_examples_exposed"] == 8
    assert result["exposure"]["optimizer_updates"] == 4
    assert result["provenance"] == "immutable manifest/hash verified"
    assert result["hashes"]["predictions_sha256"] == a.sha256(folder / "validation_predictions.csv")


def test_missing_prediction_membership_is_rejected(artifact):
    folder, metadata = artifact
    (folder / "validation_predictions.csv").write_text("filename,target,prediction\ne.wav,0,0\n")
    with pytest.raises(ValueError, match="exactly match"):
        a.audit_run(folder, metadata)


def test_metadata_label_mismatch_is_rejected(artifact):
    folder, metadata = artifact
    path = folder / "validation_predictions.csv"
    path.write_text(path.read_text().replace("f.wav,1,0", "f.wav,0,0"))
    with pytest.raises(ValueError, match="official metadata"):
        a.audit_run(folder, metadata)


def test_metric_corruption_is_rejected(artifact):
    folder, metadata = artifact
    metric = a.read_json(folder / "validation.json"); metric["accuracy"] = .75
    write_json(folder / "validation.json", metric)
    with pytest.raises(ValueError, match="accuracy"):
        a.audit_run(folder, metadata)


def test_declared_best_epoch_and_ties_are_checked(artifact):
    folder, metadata = artifact
    history = a.read_json(folder / "history.json")
    history[0]["validation"]["loss"] = 1.  # Earliest exact tie must now win.
    write_json(folder / "history.json", history)
    with pytest.raises(ValueError, match="earliest minimum"):
        a.audit_run(folder, metadata)


def test_actual_checkpoint_epoch_is_checked_when_weights_exist(artifact):
    import torch
    folder, metadata = artifact
    torch.save(dict(epoch=1, model_name="cnn"), folder / "best.pt")
    with pytest.raises(ValueError, match="best.pt epoch"):
        a.audit_run(folder, metadata)


def test_exported_compact_model_is_bound_to_saved_run(artifact):
    import torch
    folder, metadata = artifact
    root = folder.parent
    (root / 'runs').mkdir()
    folder = folder.rename(root / 'runs' / folder.name)
    models = root / "models"
    models.mkdir()
    checkpoint = models / f"{folder.name}.pt"
    identity = a.read_json(folder / "manifest.json")
    torch.save(dict(epoch=2, model_name="cnn", manifest_id=a.canonical_hash(identity)), checkpoint)
    result = a.audit_run(folder, metadata)
    assert result["checkpoint"]["path"] == str(checkpoint.resolve())
    assert result["checkpoint"]["sha256"] == a.sha256(checkpoint)
    torch.save(dict(epoch=1, model_name="cnn"), checkpoint)
    with pytest.raises(ValueError, match="epoch"):
        a.audit_run(folder, metadata)


def test_history_exposure_and_terminal_status_are_checked(artifact):
    folder, metadata = artifact
    history = a.read_json(folder / "history.json"); history[1]["optimizer_updates"] = 3
    write_json(folder / "history.json", history)
    with pytest.raises(ValueError, match="batch exposure"):
        a.audit_run(folder, metadata)


def test_manifest_identity_and_stored_parameter_count_are_checked(artifact):
    folder, metadata = artifact
    identity = a.read_json(folder / "manifest.json"); identity["parameters"] = 42
    write_json(folder / "manifest.json", identity)
    with pytest.raises(ValueError, match="identity hash"):
        a.audit_run(folder, metadata)
    run = a.read_json(folder / "run.json"); run.update(identity)
    run.update(manifest_id=a.canonical_hash(identity), manifest_sha256=a.sha256(folder / "manifest.json"))
    write_json(folder / "run.json", run)
    with pytest.raises(ValueError, match="parameter count"):
        a.audit_run(folder, metadata)


def test_duplicate_seed_does_not_inflate_n_or_fabricate_standard_deviation(artifact):
    folder, metadata = artifact
    row = a.audit_run(folder, metadata)
    duplicate = copy.deepcopy(row); duplicate["run_dir"] = "copied_run"
    group = a.aggregate([row, duplicate])[0]
    assert group["n_runs"] == 2 and group["n_unique_seeds"] == 1
    assert group["metrics"]["accuracy"]["n"] == 1
    assert group["metrics"]["accuracy"]["sample_std"] is None
    assert group["repeated_seeds"][0]["seed"] == 7


def test_conflicting_repeated_seed_results_are_retained_and_summary_withheld(artifact):
    folder, metadata = artifact
    row = a.audit_run(folder, metadata)
    conflict = copy.deepcopy(row); conflict["prediction_records_hash"] = "changed"
    group = a.aggregate([row, conflict])[0]
    assert group["conflicting_repeated_seeds"] == [7]
    assert group["metrics"] == {}
    assert group["summary_status"].startswith("withheld")


def test_unique_seed_summary_uses_sample_sd_without_pooling_predictions(artifact):
    folder, metadata = artifact
    row = a.audit_run(folder, metadata)
    other = copy.deepcopy(row); other["seed"] = 8; other["metrics"]["accuracy"] = .75
    group = a.aggregate([row, other])[0]
    assert group["n_unique_seeds"] == 2
    assert group["metrics"]["accuracy"]["mean"] == .625
    assert group["metrics"]["accuracy"]["sample_std"] == pytest.approx(np.std([.5, .75], ddof=1))
    assert "no pooling" in group["uncertainty_scope"]


def test_distinct_regimes_are_never_merged(artifact):
    folder, metadata = artifact
    row = a.audit_run(folder, metadata)
    other = copy.deepcopy(row); other["regime_id"] = "different"
    assert len(a.aggregate([row, other])) == 2


def test_holdout_reads_existing_artifact_and_checks_parent_manifest(artifact):
    folder, metadata = artifact
    test = folder.parent / "test"; test.mkdir()
    rows = [dict(filename="g.wav", target=0, prediction=0), dict(filename="h.wav", target=1, prediction=1)]
    with open(test / "test_predictions.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filename", "target", "prediction")); writer.writeheader(); writer.writerows(rows)
    metric = a.recompute_metrics(rows)
    metric.update(test_fold=5, manifest_id=a.read_json(folder / "run.json")["manifest_id"], source_policy="source_strict", loss=.5)
    write_json(test / "test.json", metric)
    with pytest.raises(ValueError, match="originating training"):
        a.audit_run(test, metadata, split="test")
    result = a.audit_run(test, metadata, split="test", manifest_reference=folder)
    assert result["split"] == "test" and result["metrics"]["accuracy"] == 1
    from_config_path = a.audit_run(test, metadata, split="test", manifest_reference=folder / "run.json")
    assert from_config_path["metrics"] == result["metrics"]
    metric["manifest_id"] = "wrong"; write_json(test / "test.json", metric)
    with pytest.raises(ValueError, match="manifest ID"):
        a.audit_run(test, metadata, split="test", manifest_reference=folder)


def test_report_retains_failed_audits_and_refuses_output_overwrite(artifact, tmp_path):
    folder, metadata = artifact
    output = tmp_path / "report"
    result = a.create_report([folder, tmp_path / "missing"], metadata, output)
    assert len(result["audited_runs"]) == 1 and len(result["audit_errors"]) == 1
    assert (output / "runs.csv").is_file() and (output / "groups.csv").is_file()
    assert "Failed audits" in (output / "report.md").read_text()
    with pytest.raises(FileExistsError):
        a.create_report([folder], metadata, output)
