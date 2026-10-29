"""Evidence survives a saved notebook: exact weights, archive safety and recovery."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("artifacts_verification", ROOT / 'scripts/legacy/artifacts.py')
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)
esc = artifacts.esc


@pytest.fixture
def checkpoint(tmp_path):
    classes = [f"class_{n}" for n in range(50)]
    manifest = dict(config=dict(model="cnn", train_folds="1,2,3", val_fold=4, test_fold=5),
                    frontend=copy.deepcopy(esc.FRONTEND), class_names=classes,
                    source_policy="synthetic strict fixture", cache_sha256="synthetic cache digest")
    value = dict(schema_version=2, model=esc.build_model("cnn").state_dict(), model_name="cnn",
                 mean=torch.zeros(1, 1, 64, 1), std=torch.ones(1, 1, 64, 1), frontend=copy.deepcopy(esc.FRONTEND),
                 class_names=classes, source_policy=manifest["source_policy"], cache_sha256=manifest["cache_sha256"],
                 manifest=manifest, manifest_id=esc.canonical_hash(manifest), manifest_sha256="fixture",
                 train_folds=(1, 2, 3), val_fold=4, test_fold=5,
                 optimizer={"state": {0: {"exp_avg": torch.ones(100000)}}}, scheduler={"last_epoch": 4},
                 torch_rng=torch.get_rng_state(), generator_rng=torch.Generator().get_state(),
                 extra_provenance={"uncatalogued_field": "must survive compaction"})
    path = tmp_path / "full.pt"
    torch.save(value, path)
    return path, value


def notebook(path, markups):
    path.write_text(json.dumps({"nbformat": 4, "cells": [
        {"cell_type": "code", "outputs": [{"output_type": "display_data", "data": {"text/html": markup}}]}
        for markup in markups]}))
    return path


def test_compaction_exactly_preserves_model_and_all_inference_provenance(checkpoint, tmp_path):
    source, before = checkpoint
    target = tmp_path / "compact.pt"
    report = artifacts.compact_checkpoint(source, target)
    after = esc.load_checkpoint(target)
    esc.validate_checkpoint(after)
    for name, tensor in before["model"].items():
        assert tensor.dtype == after["model"][name].dtype
        assert torch.equal(tensor, after["model"][name])
    for name in ("manifest", "frontend", "class_names", "source_policy", "extra_provenance"):
        assert after[name] == before[name]
    assert not artifacts.TRAINING_KEYS.intersection(after)
    assert report["compact_bytes"] < report["original_bytes"]
    assert after["inference_artifact"]["original_checkpoint_sha256"] == esc.sha256(source)
    assert source.exists()


def test_compaction_rejects_inconsistent_source_before_output(checkpoint, tmp_path):
    source, value = checkpoint
    value["frontend"]["sample_rate"] = 8000
    torch.save(value, source)
    target = tmp_path / "invalid.pt"
    with pytest.raises(ValueError, match="frontend"):
        artifacts.compact_checkpoint(source, target)
    assert not target.exists()


def test_compaction_never_overwrites(checkpoint, tmp_path):
    source, _ = checkpoint
    target = tmp_path / "existing.pt"
    target.write_bytes(b"prior")
    with pytest.raises(FileExistsError):
        artifacts.compact_checkpoint(source, target)
    assert target.read_bytes() == b"prior"


def test_capsule_recovers_exact_checkpoint_bytes_from_saved_html(checkpoint, tmp_path):
    source, _ = checkpoint
    artifacts.compact_checkpoint(source, tmp_path / "compact.pt")
    (tmp_path / "notes.txt").write_text("Authored evidence only")
    archive = tmp_path / "capsule.zip"
    made = artifacts.create_capsule(tmp_path, ["compact.pt", "notes.txt"], archive, capsule_id="modelA")
    markup = artifacts.capsule_html(archive)
    assert 'href="data:application/zip;base64,' in markup
    assert f'data-evidence-sha256="{made["sha256"]}"' in markup
    saved = notebook(tmp_path / "executed.ipynb", [markup])
    report = artifacts.extract_capsules(saved, tmp_path / "recovered")
    directory = Path(report["capsules"][0]["directory"])
    assert (directory / "files/compact.pt").read_bytes() == (tmp_path / "compact.pt").read_bytes()
    assert (directory / "files/notes.txt").read_text() == "Authored evidence only"
    esc.validate_checkpoint(esc.load_checkpoint(directory / "files/compact.pt"))


def test_explicit_groups_and_repeated_outputs_do_not_clobber_each_other(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / "b.txt").write_text("two")
    archives = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.zip"
        artifacts.create_capsule(tmp_path, [f"{name}.txt"], out, capsule_id=name)
        archives.append(artifacts.capsule_html(out))
    saved = notebook(tmp_path / "executed.ipynb", [archives[0], archives[0], [archives[1]]])
    report = artifacts.extract_capsules(saved, tmp_path / "recovered")
    assert len(report["capsules"]) == 2 and len(report["output_locations"]) == 3
    assert len({r["directory"] for r in report["capsules"]}) == 2


def test_saved_html_archive_hash_mismatch_rejected_without_extraction(tmp_path):
    (tmp_path / "a").write_text("evidence")
    out = tmp_path / "capsule.zip"
    made = artifacts.create_capsule(tmp_path, ["a"], out)
    markup = artifacts.capsule_html(out).replace(made["sha256"], "0" * 64)
    saved = notebook(tmp_path / "bad.ipynb", [markup])
    with pytest.raises(ValueError, match="SHA256"):
        artifacts.extract_capsules(saved, tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "safe/../../outside", "C:/drive", "back\\slash", "."])
def test_unsafe_zip_member_paths_rejected(name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, b"unsafe")
        archive.writestr(artifacts.MANIFEST_NAME, "{}")
    with pytest.raises(ValueError):
        artifacts.validate_capsule(buffer.getvalue())


def test_inner_file_hash_mismatch_rejected_even_with_valid_outer_zip(tmp_path):
    (tmp_path / "file").write_bytes(b"original")
    out = tmp_path / "capsule.zip"
    artifacts.create_capsule(tmp_path, ["file"], out)
    buffer = io.BytesIO()
    with zipfile.ZipFile(out) as source, zipfile.ZipFile(buffer, "w") as changed:
        for name in source.namelist():
            changed.writestr(name, b"modified" if name == "file" else source.read(name))
    with pytest.raises(ValueError, match="hash/size"):
        artifacts.validate_capsule(buffer.getvalue())


def test_archive_symlinks_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        member = zipfile.ZipInfo("link")
        member.external_attr = (0o120777 << 16)
        archive.writestr(member, "../../outside")
        archive.writestr(artifacts.MANIFEST_NAME, "{}")
    with pytest.raises(ValueError, match="nonregular"):
        artifacts.validate_capsule(buffer.getvalue())


def test_archive_cannot_claim_empty_evidence_or_duplicate_members():
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr(artifacts.MANIFEST_NAME, json.dumps({"schema_version": 1, "capsule_id": "empty", "files": {}}))
    with pytest.raises(ValueError, match="schema"):
        artifacts.validate_capsule(empty.getvalue())
    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("file", b"one")
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("file", b"two")
    with pytest.raises(ValueError, match="Duplicate"):
        artifacts.validate_capsule(duplicate.getvalue())


def test_capsule_byte_cap_and_no_overwrite_preserve_prior_data(tmp_path):
    (tmp_path / "a").write_bytes(b"evidence")
    out = tmp_path / "capsule.zip"
    with pytest.raises(ValueError, match="limit"):
        artifacts.create_capsule(tmp_path, ["a"], out, max_bytes=100)
    assert not out.exists()
    artifacts.create_capsule(tmp_path, ["a"], out)
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        artifacts.create_capsule(tmp_path, ["a"], out)
    assert out.read_bytes() == before
    saved = notebook(tmp_path / "saved.ipynb", [artifacts.capsule_html(out)])
    artifacts.extract_capsules(saved, tmp_path / "recovery")
    with pytest.raises(FileExistsError):
        artifacts.extract_capsules(saved, tmp_path / "recovery")


def test_outside_file_or_symlink_is_not_selected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("not selected project evidence")
    (root / "link").symlink_to(outside)
    for selected in ([str(outside)], ["link"], ["../private.txt"]):
        with pytest.raises(ValueError):
            artifacts.create_capsule(root, selected, root / "capsule.zip")


def test_missing_saved_capsules_cannot_be_mistaken_for_persistence(tmp_path):
    saved = notebook(tmp_path / "empty.ipynb", ["<p>download was attempted</p>"])
    with pytest.raises(ValueError, match="No verified"):
        artifacts.extract_capsules(saved, tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()


def test_display_stores_raw_html_bytes_once(tmp_path, monkeypatch):
    import importlib
    display_module = importlib.import_module("IPython.display")
    captured = []
    monkeypatch.setattr(display_module, "display", lambda value, **kwargs: captured.append((value, kwargs)))
    (tmp_path / "file").write_bytes(b"visible and saved")
    archive = tmp_path / "capsule.zip"
    artifacts.create_capsule(tmp_path, ["file"], archive)
    assert artifacts.display_capsule(archive) is None
    assert len(captured) == 1 and captured[0][1] == {"raw": True}
    assert 'data-evidence-sha256=' in captured[0][0]["text/html"]
