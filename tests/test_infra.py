"""No real audio, no test-set access, no untrusted checkpoint execution."""
import json
from pathlib import Path

from esc50_sound.infra import (compatibility, inspect_run, preflight_report,
                               run_index, write_index_csv)
from esc50_sound.records import canonical_hash, sha256, write_json


def fake_run(directory, *, model="cnn_freq", seed=101, lr=.001, status="completed"):
    directory.mkdir(parents=True)
    identity = dict(config=dict(model=model, seed=seed, epochs=2, lr=lr,
                batch_size=32, scheduler="none", specaugment=False,
                official_folds=False, train_folds="1,2,3", val_fold=4, test_fold=5),
        split={part: dict(n=1, names=[f"{part}.wav"], sources=[part])
               for part in ("train", "validation", "test")},
        environment=dict(code_sha256="code"),
        cache_sha256="dataset", frontend={"n_mels": 64}, source_policy="source-strict")
    write_json(directory / "manifest.json", identity)
    write_json(directory / "run.json", dict(config=identity["config"], status=status,
        manifest_id=canonical_hash(identity), manifest_sha256=sha256(directory / "manifest.json"),
        epochs_recorded=2, test_evaluated=False))
    write_json(directory / "history.json", [{"epoch": 1, "partial_epoch": False},
                                               {"epoch": 2, "partial_epoch": False}])
    write_json(directory / "validation.json", dict(accuracy=.55, macro_f1=.5, loss=1.2))
    (directory / "best.pt").write_bytes(b"not a checkpoint: inspection must not load")
    (directory / "test.json").write_text("intentionally invalid json; do not read")
    return directory


def test_inspection_and_pairwise_gate(tmp_path):
    a = inspect_run(fake_run(tmp_path / "one"))
    b = inspect_run(fake_run(tmp_path / "two", model="crnn_freq_ta", seed=202))
    assert a["completed"] and b["valid"] and a["validation"]["accuracy"] == .55
    assert compatibility(a, b)["eligible"]
    assert not compatibility(a, inspect_run(fake_run(tmp_path / "three", lr=.1)))["eligible"]
    assert a["evidence"].startswith("read-only")


def test_detect_tampered_manifest_and_partial_epoch(tmp_path):
    folder = fake_run(tmp_path / "one")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["cache_sha256"] = "other"
    write_json(folder / "manifest.json", manifest)
    assert not inspect_run(folder)["valid"]
    folder = fake_run(tmp_path / "two")
    write_json(folder / "history.json", [{"epoch": 1}, {"epoch": 2, "partial_epoch": True}])
    record = inspect_run(folder)
    assert not record["valid"] and not record["completed"]


def test_index_detects_duplicate_manifest_and_excludes_test(tmp_path):
    fake_run(tmp_path / 'results/historical/local_runs/a')
    fake_run(tmp_path / 'results/historical/local_runs/b')
    report = run_index(tmp_path / 'results/historical/local_runs')
    assert report["total"] == 2 and report["completed"] == 2
    assert len(report["duplicate_manifest_groups"]) == 1
    assert report["test_metrics_read"] is False
    write_index_csv(tmp_path / "index.csv", report)
    assert "validation_accuracy" in (tmp_path / "index.csv").read_text()


def test_preflight_reports_missing_cache_without_running_data_download(tmp_path):
    folder = tmp_path / "dataset"
    (folder / "meta").mkdir(parents=True)
    (folder / "meta" / "esc50.csv").write_text("filename,fold,target,category,src_file\na.wav,1,0,cat,aa\n")
    report = preflight_report(folder, tmp_path / "no_cache.npz")
    assert not report["ready"] and report["blockers"]
    assert report["checks"]["audio"]["complete"] is False
    assert not (tmp_path / "no_cache.npz").exists()


def test_atomic_record_rewrite_and_invalid_json_rejected(tmp_path):
    import math
    import pytest
    file = tmp_path / "run.json"
    write_json(file, {"status": "running"})
    write_json(file, {"status": "completed"})
    assert json.loads(file.read_text()) == {"status": "completed"}
    with pytest.raises(ValueError):
        write_json(file, {"loss": math.nan})
    assert json.loads(file.read_text()) == {"status": "completed"}
    assert not list(tmp_path.glob("*.tmp"))
