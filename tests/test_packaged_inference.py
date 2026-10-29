"""Numerical migration equivalence and portable-package inference checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from esc50_sound import audio, baseline, checkpoints, inference, models, records


def legacy_module(name):
    spec = importlib.util.spec_from_file_location(f"legacy_migration_{name}", ROOT / "archive" / "source_before_package" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path):
    records.seed_all(31)
    classes = [f"synthetic_class_{n}" for n in range(50)]
    manifest = dict(config=dict(model="cnn_freq", train_folds="1,2,3", val_fold=4, test_fold=5),
                    frontend=audio.FRONTEND, class_names=classes, cache_sha256="b" * 64,
                    source_policy="synthetic fixture, no empirical performance claim")
    value = dict(schema_version=2, model_name="cnn_freq", model=models.build_model("cnn_freq").state_dict(),
                 mean=torch.zeros(1, 1, 64, 1), std=torch.ones(1, 1, 64, 1),
                 frontend=audio.FRONTEND, class_names=classes, manifest=manifest,
                 manifest_id=records.canonical_hash(manifest), source_policy=manifest["source_policy"],
                 cache_sha256=manifest["cache_sha256"], train_folds=(1, 2, 3), val_fold=4, test_fold=5)
    path = tmp_path / "fixture.pt"
    torch.save(value, path)
    return path


def tone(path, seconds=1, rate=8000):
    times = np.arange(round(seconds * rate)) / rate
    sf.write(path, .1 * np.sin(2 * np.pi * 440 * times), rate)
    return path


def test_packaged_classifier_matches_original_numerical_result(checkpoint, tmp_path):
    old = legacy_module("classify")
    path = tone(tmp_path / "tone.wav")
    torch.set_num_threads(2)
    before = old.AudioClassifier(checkpoint).classify(path)
    after = inference.AudioClassifier(checkpoint).classify(path)
    assert after == before
    assert after["preprocessing"]["right_padding_samples"] == 64000
    assert after["preprocessing"]["source_sample_rate"] == 8000


def test_packaged_silence_and_duration_policies_match_original(checkpoint, tmp_path):
    old = legacy_module("classify")
    silence = tmp_path / "silence.wav"
    sf.write(silence, np.zeros(16000), 16000)
    before, after = old.AudioClassifier(checkpoint), inference.AudioClassifier(checkpoint)
    assert before.classify(silence) == after.classify(silence)
    assert after.classify(silence)["status"] == "abstained"
    long = tone(tmp_path / "long.wav", seconds=5.2)
    for classifier in (before, after):
        with pytest.raises(ValueError, match="maximum is five seconds"):
            classifier.classify(long)


def test_packaged_inference_rejects_malformed_or_mismatched_checkpoint(checkpoint, tmp_path):
    value = checkpoints.load_checkpoint(checkpoint)
    value["frontend"] = dict(value["frontend"], sample_rate=8000)
    bad = tmp_path / "bad.pt"
    torch.save(value, bad)
    with pytest.raises(ValueError, match="frontend"):
        inference.AudioClassifier(bad)


def test_bundle_contains_package_and_runs_without_project_pythonpath(checkpoint, tmp_path):
    card = tmp_path / "card.md"
    card.write_text("Synthetic migration fixture; no measured accuracy.\n")
    destination = tmp_path / "portable"
    manifest = inference.package_bundle(checkpoint, card, destination)
    required = {"src/esc50_sound/audio.py", "src/esc50_sound/inference.py", "src/esc50_sound/models.py",
                "pyproject.toml", "requirements.txt", "MODEL_CARD.md", "README.md", "checkpoint.pt", "predict.py"}
    assert required.issubset(manifest["files"])
    assert not (destination / 'scripts/legacy/esc50.py').exists() and not (destination / 'scripts/legacy/classify.py').exists()
    for name, declared in manifest["files"].items():
        assert records.sha256(destination / name) == declared["sha256"]
    path = tone(tmp_path / "tone.wav")
    env = dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1")
    completed = subprocess.run([sys.executable, str(destination / "predict.py"),
                                "--checkpoint", str(destination / "checkpoint.pt"), "--audio", str(path)],
                               cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    expected = inference.AudioClassifier(checkpoint).classify(path)
    assert result == expected
    # The shim's local source must win even if an editable project exists elsewhere.
    probe = subprocess.run([sys.executable, "-c",
                            f"import sys;sys.path.insert(0,{str(destination/'src')!r});import esc50_sound;print(esc50_sound.__file__)"],
                           cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert str(destination / "src") in probe.stdout
    with pytest.raises(FileExistsError):
        inference.package_bundle(checkpoint, card, destination)


def test_baseline_feature_and_objective_numerics_unchanged():
    old = legacy_module("diagnostics")
    x = np.random.default_rng(31).normal(size=(8, 1, 64, 12)).astype(np.float32)
    np.testing.assert_array_equal(baseline.summarize_features(x), old.summarize_features(x))
    np.testing.assert_array_equal(baseline.fit_standardization(baseline.summarize_features(x)),
                                  old.fit_standardization(old.summarize_features(x)))
    features = np.array([[1., 2.], [-1., .5], [0., -1.]])
    labels = np.array([0, 1, 2])
    parameters = np.linspace(-.2, .3, 9)
    before = old.logistic_objective(parameters, features, labels, .1, 3)
    after = baseline.logistic_objective(parameters, features, labels, .1, 3)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])


def test_baseline_optimization_and_metric_results_unchanged():
    old = legacy_module("diagnostics")
    x = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    y = np.array([0, 0, 1, 1])
    old_weights, old_bias, _ = old.fit_logistic(x, y, .001, num_classes=2)
    weights, bias, _ = baseline.fit_logistic(x, y, .001, num_classes=2)
    np.testing.assert_array_equal(weights, old_weights)
    np.testing.assert_array_equal(bias, old_bias)
    before, old_predictions = old.score_logistic(x, y, old_weights, old_bias)
    after, predictions = baseline.score_logistic(x, y, weights, bias)
    assert before == after
    np.testing.assert_array_equal(predictions, old_predictions)


def test_baseline_source_splits_delegate_without_changing_membership():
    old = legacy_module("diagnostics")
    cache = dict(folds=np.array([1, 2, 4, 4, 5, 4, 5]),
                 sources=np.array(["train", "a", "a", "b", "b", "val", "test"]))
    for strict in (True, False):
        before, after = old.source_split(cache, strict=strict), baseline.source_split(cache, strict=strict)
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])


def test_packaged_baseline_test_guard_runs_before_loading_files():
    from argparse import Namespace
    with pytest.raises(ValueError, match="allow-test"):
        baseline.evaluate_test(Namespace(allow_test=False))
