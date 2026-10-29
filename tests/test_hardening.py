"""Independent provenance/failure checks. Uses synthetic caches; never real test audio."""
import copy
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("esc50_hardening", ROOT / 'scripts/legacy/esc50.py')
esc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esc)
torch.set_num_threads(2)


@pytest.fixture
def synthetic_cache(tmp_path):
    """One synthetic clip/class: every fold is represented without acoustic data."""
    path = tmp_path / "synthetic.npz"
    labels = np.arange(50, dtype=np.int64)
    folds = labels % 5 + 1
    names = np.array([f"{f}-source{c}-A-{c}.wav" for c, f in zip(labels, folds)])
    sources = np.array([f"source{c}" for c in labels])
    classes = np.array([f"class{c}" for c in labels])
    np.savez_compressed(path, x=np.zeros((50, 1, 64, 313), dtype=np.float32),
                        y=labels, folds=folds, names=names, sources=sources, class_names=classes)
    rows = [dict(filename=str(n), target=int(c), fold=int(f), src_file=str(s), category=str(classes[c]))
            for n, c, f, s in zip(names, labels, folds, sources)]
    sidecar = dict(schema_version=2, frontend=copy.deepcopy(esc.FRONTEND),
                   cache_sha256=esc.sha256(path), audit=esc.audit_metadata(rows, require_complete=False),
                   metadata_records_sha256=esc.canonical_hash(esc.metadata_records(rows)))
    esc.write_json(str(path) + ".json", sidecar)
    return path, sidecar


@pytest.fixture
def portable_checkpoint(tmp_path, synthetic_cache):
    cache_path, _ = synthetic_cache
    cache = esc.load_cache(cache_path)
    cfg = dict(model="cnn", train_folds="1,2,3", val_fold=4, test_fold=5, device="cpu", official_folds=False)
    manifest = dict(schema_version=2, config=cfg, frontend=copy.deepcopy(esc.FRONTEND),
                    cache_sha256=esc.sha256(cache_path), source_policy="source-strict synthetic fixture",
                    split={k: {"names": [], "sources": [], "n": 0} for k in ("train", "validation", "test")})
    # Include optional mapping when the implementation's portable schema binds it.
    manifest["class_names"] = cache["class_names"].tolist()
    checkpoint = dict(schema_version=2, frontend=copy.deepcopy(esc.FRONTEND),
                      model=esc.build_model("cnn").state_dict(), model_name="cnn",
                      mean=torch.zeros(1, 1, 64, 1), std=torch.ones(1, 1, 64, 1),
                      class_names=cache["class_names"].tolist(), cache_sha256=esc.sha256(cache_path),
                      train_folds=(1, 2, 3), val_fold=4, test_fold=5,
                      source_policy=manifest["source_policy"], manifest=manifest,
                      manifest_id=esc.canonical_hash(manifest), manifest_sha256="synthetic-fixture")
    path = tmp_path / "run" / "best.pt"
    path.parent.mkdir()
    torch.save(checkpoint, path)
    esc.write_json(path.parent / "run.json", dict(manifest_id=checkpoint["manifest_id"], test_evaluated=False))
    return path, checkpoint


def test_cache_accepts_consistent_portable_sidecar(synthetic_cache):
    path, _ = synthetic_cache
    assert esc.load_cache(path)["x"].shape == (50, 1, 64, 313)


@pytest.mark.parametrize("corruption", ["missing", "frontend", "hash", "metadata", "schema"])
def test_cache_rejects_incompatible_provenance(synthetic_cache, corruption):
    path, sidecar = synthetic_cache
    provenance = Path(str(path) + ".json")
    if corruption == "missing":
        provenance.unlink()
    else:
        if corruption == "frontend": sidecar["frontend"]["sample_rate"] = 8000
        if corruption == "hash": sidecar["cache_sha256"] = "wrong"
        if corruption == "metadata": sidecar["metadata_records_sha256"] = "wrong"
        if corruption == "schema": sidecar["schema_version"] = 999
        esc.write_json(provenance, sidecar)
    with pytest.raises(ValueError):
        esc.load_cache(path)


def test_checkpoint_provenance_survives_detaching_file(portable_checkpoint, tmp_path):
    _, checkpoint = portable_checkpoint
    detached = tmp_path / "detached.pt"
    torch.save(checkpoint, detached)
    restored = esc.load_checkpoint(detached)
    assert esc.validate_checkpoint(restored) == "verified embedded manifest"
    restored["manifest"]["source_policy"] = "changed after training"
    with pytest.raises(ValueError, match="manifest"):
        esc.validate_checkpoint(restored)


def test_checkpoint_rejects_frontend_mismatch(portable_checkpoint):
    _, checkpoint = portable_checkpoint
    checkpoint["frontend"]["n_mels"] = 80
    with pytest.raises(ValueError, match="frontend"):
        esc.validate_checkpoint(checkpoint)


def test_legacy_checkpoint_requires_explicit_compatibility(portable_checkpoint):
    _, checkpoint = portable_checkpoint
    checkpoint.pop("schema_version")
    with pytest.raises(ValueError, match="Legacy"):
        esc.validate_checkpoint(checkpoint)
    with pytest.warns(RuntimeWarning, match="Legacy"):
        assert "unverified" in esc.validate_checkpoint(checkpoint, allow_legacy=True)


def test_device_record_distinguishes_requested_cpu_from_available_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "available GPU fixture")
    observed = esc.environment(torch.device("cpu"))
    assert observed["device"] == "cpu"
    assert observed["available_cuda_device"] == "available GPU fixture"


@pytest.mark.parametrize("name", ["cnn", "cnn_matched"])
def test_nonrecurrent_control_is_exactly_time_permutation_invariant(name):
    model = esc.build_model(name).eval()
    x = torch.randn(2, 1, 64, 313)
    permutation = torch.randperm(39)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x, time_permutation=permutation), atol=0, rtol=0)


def test_recurrent_hook_validates_permutation_and_changes_order_processing():
    esc.seed_all(6)
    model = esc.build_model("crnn").eval()
    x = torch.randn(2, 1, 64, 313)
    with torch.no_grad():
        assert not torch.equal(model(x), model(x, time_permutation=torch.arange(38, -1, -1)))
    for invalid in (torch.arange(38), torch.zeros(39, dtype=torch.long), torch.arange(39).float()):
        with pytest.raises(ValueError, match="time_permutation"):
            model(x, time_permutation=invalid)


def test_matched_control_parameter_budget_is_within_one_percent():
    sizes = [sum(p.numel() for p in esc.build_model(name).parameters()) for name in ("crnn", "cnn_matched")]
    assert abs(sizes[0] - sizes[1]) / sizes[0] < .01


def test_final_evaluation_denial_has_no_side_effect(tmp_path):
    with pytest.raises(ValueError, match="sealed"):
        esc.evaluate(Namespace(allow_test=False))
    assert list(tmp_path.iterdir()) == []


def test_final_evaluation_refuses_overwrite_before_inference(synthetic_cache, portable_checkpoint, tmp_path, monkeypatch):
    cache_path, _ = synthetic_cache
    checkpoint_path, _ = portable_checkpoint
    out = tmp_path / "already_exists"
    out.mkdir()
    def forbidden(*args, **kwargs):
        pytest.fail("Inference must not occur after output collision")
    monkeypatch.setattr("esc50_sound.evaluation.run_eval", forbidden)
    with pytest.raises(FileExistsError):
        esc.evaluate(Namespace(allow_test=True, checkpoint=str(checkpoint_path), cache=str(cache_path), output=str(out)))
    assert not (checkpoint_path.parent / "evaluation_history.jsonl").exists()


def test_synthetic_final_evaluation_records_access_and_completion(synthetic_cache, portable_checkpoint, tmp_path):
    cache_path, _ = synthetic_cache
    checkpoint_path, checkpoint = portable_checkpoint
    out = tmp_path / "synthetic_eval"
    esc.evaluate(Namespace(allow_test=True, checkpoint=str(checkpoint_path), cache=str(cache_path), output=str(out)))
    records = [json.loads(line) for line in (checkpoint_path.parent / "evaluation_history.jsonl").read_text().splitlines()]
    assert [record["status"] for record in records] == ["started", "completed"]
    assert len({record["evaluation_id"] for record in records}) == 1
    assert records[0]["source_policy"] == checkpoint["source_policy"]
    assert json.loads((checkpoint_path.parent / "run.json").read_text())["test_evaluated"] is True
    result = json.loads((out / "test.json").read_text())
    assert result["manifest_id"] == checkpoint["manifest_id"]
    assert result["n"] == 10


def test_synthetic_evaluation_failure_preserves_access_audit(synthetic_cache, portable_checkpoint, tmp_path, monkeypatch):
    cache_path, _ = synthetic_cache
    checkpoint_path, _ = portable_checkpoint
    def fail_inference(*args, **kwargs):
        raise RuntimeError("injected inference failure")
    monkeypatch.setattr("esc50_sound.evaluation.run_eval", fail_inference)
    with pytest.raises(RuntimeError, match="injected"):
        esc.evaluate(Namespace(allow_test=True, checkpoint=str(checkpoint_path), cache=str(cache_path), output=str(tmp_path / "failed_eval")))
    records = [json.loads(line) for line in (checkpoint_path.parent / "evaluation_history.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records] == ["started", "failed"]
    assert json.loads((checkpoint_path.parent / "run.json").read_text())["test_evaluated"] is True


def test_training_exception_is_terminal_failed_state(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("injected preflight rejection")
    monkeypatch.setattr("esc50_sound.training._train", fail)
    out = tmp_path / "failed_train"
    with pytest.raises(ValueError, match="injected"):
        esc.train(Namespace(output=str(out)))
    assert json.loads((out / "run.json").read_text())["status"] == "failed"


def test_repeated_synthetic_evaluations_append_instead_of_replacing(synthetic_cache, portable_checkpoint, tmp_path):
    cache_path, _ = synthetic_cache
    checkpoint_path, _ = portable_checkpoint
    args = dict(allow_test=True, checkpoint=str(checkpoint_path), cache=str(cache_path))
    esc.evaluate(Namespace(**args, output=str(tmp_path / "first_eval")))
    audit = checkpoint_path.parent / "evaluation_history.jsonl"
    original = audit.read_bytes()
    esc.evaluate(Namespace(**args, output=str(tmp_path / "second_eval")))
    assert audit.read_bytes().startswith(original)
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [r["status"] for r in records] == ["started", "completed", "started", "completed"]
    assert records[0]["evaluation_id"] != records[2]["evaluation_id"]


@pytest.mark.parametrize("field,value", [("class_names", ["incorrect"] * 50),
                                        ("train_folds", (1, 2, 4)), ("val_fold", 5), ("test_fold", 4)])
def test_portable_manifest_binds_mapping_and_holdout_role(portable_checkpoint, field, value):
    _, checkpoint = portable_checkpoint
    checkpoint[field] = value
    with pytest.raises(ValueError):
        esc.validate_checkpoint(checkpoint)
