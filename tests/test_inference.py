"""Portable inference checks use only synthetic waveforms and checkpoint fixtures."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import esc50
import classify

torch.set_num_threads(2)


def write_checkpoint(path, model="cnn", legacy=False):
    torch.manual_seed(1)
    classes = [f"synthetic_class_{i}" for i in range(50)]
    manifest = dict(config=dict(model=model, train_folds="1,2,3", val_fold=4, test_fold=5),
                    frontend=esc50.FRONTEND, class_names=classes, cache_sha256="a" * 64,
                    source_policy="synthetic test fixture; no empirical performance claim")
    checkpoint = dict(schema_version=2, model_name=model, model=esc50.build_model(model).state_dict(),
                      mean=torch.zeros(1, 1, 64, 1), std=torch.ones(1, 1, 64, 1),
                      frontend=esc50.FRONTEND, class_names=classes,
                      manifest=manifest, manifest_id=esc50.canonical_hash(manifest),
                      cache_sha256=manifest["cache_sha256"], source_policy=manifest["source_policy"],
                      train_folds=(1, 2, 3), val_fold=4, test_fold=5)
    if legacy:
        checkpoint.pop("schema_version")
    torch.save(checkpoint, path)
    return checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "fixture.pt"
    write_checkpoint(path)
    return path


def tone(path, seconds=1, rate=8000):
    t = np.arange(round(seconds * rate)) / rate
    sf.write(path, .1 * np.sin(2 * np.pi * 440 * t), rate)
    return path


def test_short_audio_reports_resampling_padding_and_uncalibrated_scores(checkpoint, tmp_path):
    audio = tone(tmp_path / "tone.wav")
    result = classify.AudioClassifier(checkpoint).classify(audio)
    assert result["status"] == "classified" and len(result["top5"]) == 5
    assert result["device"] == "cpu"
    assert result["provenance_verification"] == "verified embedded manifest"
    assert result["preprocessing"]["raw_duration_seconds"] == 1
    assert result["preprocessing"]["right_padding_samples"] == 64000
    assert result["preprocessing"]["source_sample_rate"] == 8000
    assert "Uncalibrated" in result["score_interpretation"]
    scores = [r["softmax_score"] for r in result["top5"]]
    assert all(0 <= score <= 1 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_silence_abstains_without_emitting_class_scores(checkpoint, tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16000), 16000)
    result = classify.AudioClassifier(checkpoint).classify(path)
    assert result["status"] == "abstained"
    assert result["top5"] == [] and result["predicted_category"] is None
    assert result["abstention_reason"] == "silence_or_near_silence"


def test_nonfinite_unreadable_empty_and_long_audio_are_rejected(checkpoint, tmp_path):
    classifier = classify.AudioClassifier(checkpoint)
    unreadable = tmp_path / "bad.wav"; unreadable.write_text("not audio")
    empty = tmp_path / "empty.wav"; sf.write(empty, np.array([], dtype=float), 16000)
    nonfinite = tmp_path / "nan.wav"; sf.write(nonfinite, np.full(16000, np.nan), 16000, subtype="FLOAT")
    long = tone(tmp_path / "long.wav", seconds=5.2)
    for path in (unreadable, empty, nonfinite, long, tmp_path / "missing.wav"):
        with pytest.raises(ValueError):
            classifier.classify(path)


def test_frontend_mismatch_and_implicit_legacy_are_rejected(checkpoint, tmp_path):
    bad = esc50.load_checkpoint(checkpoint)
    bad["frontend"] = dict(esc50.FRONTEND, sample_rate=8000)
    path = tmp_path / "bad_frontend.pt"; torch.save(bad, path)
    with pytest.raises(ValueError, match="frontend"):
        classify.AudioClassifier(path)
    legacy = tmp_path / "legacy.pt"; write_checkpoint(legacy, legacy=True)
    with pytest.raises(ValueError, match="Legacy"):
        classify.AudioClassifier(legacy)
    with pytest.warns(RuntimeWarning, match="Legacy"):
        allowed = classify.AudioClassifier(legacy, allow_legacy=True)
    assert allowed.verification.startswith("legacy")


def test_cli_and_library_produce_identical_predictions(checkpoint, tmp_path):
    audio = tone(tmp_path / "tone.wav", seconds=5, rate=16000)
    expected = classify.AudioClassifier(checkpoint).classify(audio)
    output = tmp_path / "result.json"
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/legacy/classify.py'), "--checkpoint", str(checkpoint),
                                "--audio", str(audio), "--json-output", str(output)],
                               check=True, capture_output=True, text=True)
    actual = json.loads(completed.stdout)
    assert actual == json.loads(output.read_text())
    assert actual["predicted_category"] == expected["predicted_category"]
    np.testing.assert_allclose([r["softmax_score"] for r in actual["top5"]],
                               [r["softmax_score"] for r in expected["top5"]], atol=1e-7, rtol=1e-6)
    repeated = subprocess.run([sys.executable, str(ROOT / 'scripts/legacy/classify.py'), "--checkpoint", str(checkpoint),
                               "--audio", str(audio), "--json-output", str(output)], capture_output=True, text=True)
    assert repeated.returncode == 2  # Existing evidence is never overwritten.
    assert "Traceback" not in repeated.stderr


@pytest.mark.parametrize("model", ["cnn_freq", "crnn_freq", "cnn_freq_matched"])
def test_frequency_preserving_model_variants_load_and_classify(model, tmp_path):
    path = tmp_path / f"{model}.pt"; write_checkpoint(path, model=model)
    result = classify.AudioClassifier(path).classify(tone(tmp_path / "tone.wav"))
    assert result["model"] == model and result["status"] == "classified"


def test_portable_bundle_has_verified_hashes_and_runs_standalone(checkpoint, tmp_path):
    card = tmp_path / "card.md"; card.write_text("Synthetic fixture only. No empirical performance claim.\n")
    bundle = tmp_path / "bundle"
    manifest = classify.package_bundle(checkpoint, card, bundle)
    assert {"predict.py", "pyproject.toml", "requirements.txt", "checkpoint.pt", "MODEL_CARD.md"}.issubset(manifest["files"])
    assert "src/esc50_sound/inference.py" in manifest["files"]
    for name, record in manifest["files"].items():
        assert esc50.sha256(bundle / name) == record["sha256"]
    audio = tone(tmp_path / "tone.wav")
    result = subprocess.run([sys.executable, str(bundle / "predict.py"), "--checkpoint", str(bundle / "checkpoint.pt"),
                             "--audio", str(audio)], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["status"] == "classified"
    with pytest.raises(FileExistsError):
        classify.package_bundle(checkpoint, card, bundle)


def test_explicit_cuda_does_not_silently_fall_back_to_cpu(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no device fallback"):
        classify.AudioClassifier(checkpoint, device="cuda")
