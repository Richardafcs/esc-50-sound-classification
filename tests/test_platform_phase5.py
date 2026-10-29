"""Contract-first Phase 5 tests; no ESC-50 audio download or test evaluation."""
from __future__ import annotations
from dataclasses import replace

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

from esc50_sound.platform import (
    ContractError, FeatureCacheBackend, LOGMEL, ModelAdapter, ModelCatalog,
    RawAudioBackend, SplitPolicy, TrainingPlan, builtin_models,
)
from esc50_sound.models import MODEL_NAMES, build_model
from esc50_sound.records import seed_all


def sample_plan(tmp_path):
    raw = json.loads((Path(__file__).resolve().parents[1] / 'configs/examples/platform_cnn_cpu.json').read_text())
    raw["output"] = str(tmp_path / "new_run")
    raw["cache"] = str(tmp_path / "missing_features.npz")
    raw["metadata_root"] = str(Path(__file__).resolve().parents[1] / "data/ESC-50")
    return raw


def test_all_existing_models_registered_without_change():
    assert {x["name"] for x in builtin_models().describe()} == set(MODEL_NAMES)
    for name in MODEL_NAMES:
        assert builtin_models().resolve(name, "supervised_train").input_contract == LOGMEL
    seed_all(13)
    expected = build_model("crnn_freq")
    seed_all(13)
    actual = builtin_models().resolve("crnn_freq", "supervised_train").build()
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)


def test_model_mode_gate_and_duplicate_registration():
    catalog = ModelCatalog()
    catalog.register(ModelAdapter("mock_zero_shot", LOGMEL, ("zero_shot_inference",),
                                  True, lambda n: torch.nn.Linear(1, n)))
    with pytest.raises(ContractError, match="not supported"):
        catalog.resolve("mock_zero_shot", "supervised_train")
    with pytest.raises(ContractError, match="Duplicate"):
        catalog.register(ModelAdapter("mock_zero_shot", LOGMEL, ("zero_shot_inference",),
                                      True, lambda n: torch.nn.Linear(1, n)))
    with pytest.raises(ContractError, match="not registered"):
        builtin_models().resolve("clap", "zero_shot_inference")


def test_plan_strict_schema_and_unknown_models(tmp_path):
    raw = sample_plan(tmp_path)
    plan = TrainingPlan.from_dict(raw)
    assert plan.model == "cnn_freq"
    assert plan.split().source_strict
    assert plan.to_legacy_args().train_folds == "1,2,3"
    assert plan.to_legacy_args().official_folds is False
    assert plan.identity() == TrainingPlan.from_dict(raw).identity()
    for changed in (
        {**raw, "typo": 42},
        {k: v for k, v in raw.items() if k != "seed"},
        {**raw, "model": "clap"},
        {**raw, "dataset_backend": "raw_wav"},
        {**raw, "source_strict": "yes"},
        {**raw, "epochs": True},
        {**raw, "train_folds": [1, 4]},
        {**raw, "test_fold": 4},
        {**raw, "max_seconds": float("inf")},
        {**raw, "scheduler": "exponential"},
    ):
        with pytest.raises(ContractError):
            TrainingPlan.from_dict(changed)


def test_unavailable_cache_blocks_execution_before_training(tmp_path):
    plan = TrainingPlan.from_dict(sample_plan(tmp_path))
    state = plan.check()
    assert state["ready"] is False
    assert "Dataset not ready" in state["blockers"][0]
    with patch("esc50_sound.training.train") as train:
        with pytest.raises(ContractError, match="Preflight blocked"):
            plan.execute()
        train.assert_not_called()


def test_output_collision_blocks_and_no_test_access(tmp_path):
    raw = sample_plan(tmp_path)
    plan = TrainingPlan.from_dict(raw)
    output = Path(raw["output"])
    output.mkdir(parents=True)
    (output / "run.json").write_text("{}")
    assert any("Output contains artifacts" in b for b in plan.check()["blockers"])


def test_raw_backend_test_guard_and_lazy_feature_read(tmp_path):
    original = Path(__file__).resolve().parents[1] / "data/ESC-50/meta/esc50.csv"
    root = tmp_path / "ESC-50"
    (root / "meta").mkdir(parents=True)
    (root / "meta/esc50.csv").write_bytes(original.read_bytes())
    adapter = RawAudioBackend(root, SplitPolicy())
    with pytest.raises(PermissionError):
        adapter.get("test", 0)
    with pytest.raises(ContractError, match="Unknown split"):
        adapter.get("all", 0)
    with pytest.raises(FileNotFoundError):
        adapter.get("train", 0)
    chosen = adapter.rows[adapter.indices["train"][0]]
    (root / "audio").mkdir()
    samples = np.sin(2 * np.pi * 320 * np.arange(16000, dtype=np.float32) / 16000)
    sf.write(root / "audio" / chosen["filename"], samples, 16000)
    with pytest.raises(ContractError, match="44.1kHz"):
        adapter.get("train", 0)
    t = np.arange(5 * 44100, dtype=np.float32) / 44100
    sf.write(root / "audio" / chosen["filename"], np.sin(2 * np.pi * 320 * t), 44100)
    item = adapter.get("train", 0)
    assert item.features.shape == LOGMEL.shape
    assert torch.isfinite(item.features).all()
    assert item.target == int(chosen["target"])
    assert item.source == str(chosen["src_file"])
    assert item.raw_sha256 and len(item.raw_sha256) == 64


def test_feature_backend_uses_canonical_metadata_and_guards_test(tmp_path):
    original = Path(__file__).resolve().parents[1] / "data/ESC-50/meta/esc50.csv"
    root = tmp_path / "data"
    (root / "meta").mkdir(parents=True)
    (root / "meta/esc50.csv").write_bytes(original.read_bytes())
    rows = __import__("esc50_sound.data", fromlist=["read_metadata"]).read_metadata(root)
    # Real metadata source membership; the cache parser is mocked for this interface test.
    fake = {"x": np.broadcast_to(np.zeros((1, 1, 64, 313), dtype=np.float32),
                                        (len(rows), 1, 64, 313)),
            "y": np.array([int(r["target"]) for r in rows]),
            "folds": np.array([int(r["fold"]) for r in rows]),
            "sources": np.array([r["src_file"] for r in rows]),
            "names": np.array([r["filename"] for r in rows]),
            "class_names": np.array(["" for _ in range(50)])}
    for row in rows:
        fake["class_names"][int(row["target"])] = row["category"]
    fake_cache_path = tmp_path / "fake.npz"
    fake_cache_path.write_bytes(b"interface-only-fake-cache")
    with patch("esc50_sound.platform.load_cache", return_value=fake), \
         patch("esc50_sound.platform.cache_metadata_contract", return_value={"valid": True}):
        dataset = FeatureCacheBackend(fake_cache_path, root, SplitPolicy())
        assert len(dataset.indices["train"]) == 1200
        assert len(dataset.indices["test"]) == 400
        with pytest.raises(PermissionError):
            dataset.get("test", 0)
        assert dataset.get("validation", 0).features.shape == LOGMEL.shape


def test_platform_cli_catalog_check_run_requires_execute(tmp_path, capsys):
    from scripts.research_platform import main
    assert main(["catalog"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert "PANNs fine-tune" in listing["not_implemented"]
    file = tmp_path / "plan.json"
    file.write_text(json.dumps(sample_plan(tmp_path)))
    assert main(["check", "--plan", str(file)]) == 2
    assert "ready" in capsys.readouterr().out
    assert main(["run", "--plan", str(file)]) == 2
    assert "--execute" in capsys.readouterr().err
    assert main(["run", "--plan", str(file), "--execute"]) == 2
    assert "Preflight blocked" in capsys.readouterr().err


def test_model_adapter_enforces_input_and_logits():
    adapter = builtin_models().resolve("cnn_freq", "supervised_train")
    model = adapter.build().eval()
    batch = torch.zeros(2, 1, 64, 313, dtype=torch.float32)
    with torch.no_grad():
        result = adapter.infer_logits(model, batch)
    assert result.shape == (2, 50)
    with pytest.raises(ContractError, match="shape"):
        adapter.infer_logits(model, torch.zeros(2, 1, 64, 312))
    with pytest.raises(ContractError, match="float32"):
        adapter.infer_logits(model, batch.to(torch.float64))
    bad = batch.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ContractError, match="finite"):
        adapter.infer_logits(model, bad)


def test_platform_plan_hash_binds_legacy_training_manifest(tmp_path):
    plan = TrainingPlan.from_dict(sample_plan(tmp_path))
    args = plan.to_legacy_args()
    assert args.platform_plan_sha256 == plan.identity()
    assert args.platform_plan_schema_version == 1
    assert "allow_test" not in vars(args)
    assert plan.identity() != replace(plan, seed=202).identity()


def test_verify_run_requires_bound_identity_and_never_reads_test(tmp_path, capsys):
    from scripts.research_platform import main
    raw = sample_plan(tmp_path)
    plan = TrainingPlan.from_dict(raw)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw))
    out = Path(raw["output"])
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({"config": {"platform_plan_sha256": "wrong"}}))
    assert main(["verify-run", "--plan", str(path), "--run-dir", str(out)]) == 2
    assert "not bound" in capsys.readouterr().err
    assert not (out / "test_predictions.csv").exists()
