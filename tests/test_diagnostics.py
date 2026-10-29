"""Mathematical and leakage checks for the statistical baseline and audit tools."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnostics", ROOT / 'scripts/legacy/diagnostics.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_temporal_summary_is_order_invariant_and_has_expected_values():
    x = np.array([[[[0., 2., 4.], [1., 1., 1.]]]])
    np.testing.assert_allclose(d.summarize_features(x), [[2., 1., np.sqrt(8/3), 0.]])
    np.testing.assert_array_equal(d.summarize_features(x), d.summarize_features(x[..., ::-1]))


def test_standardization_fits_only_given_training_matrix():
    mean, std = d.fit_standardization([[0., 4.], [2., 4.]])
    np.testing.assert_array_equal(mean, [1., 4.])
    np.testing.assert_array_equal(std, [1., 1e-8])


def test_logistic_gradient_matches_finite_difference():
    x = np.array([[1., 2.], [-1., .5], [0., -1.]])
    y = np.array([0, 1, 2])
    theta = np.linspace(-.2, .3, 9)
    _, analytic = d.logistic_objective(theta, x, y, .1, 3)
    numerical = []
    for i in range(len(theta)):
        delta = np.zeros_like(theta); delta[i] = 1e-6
        a, _ = d.logistic_objective(theta + delta, x, y, .1, 3)
        b, _ = d.logistic_objective(theta - delta, x, y, .1, 3)
        numerical.append((a - b) / 2e-6)
    np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-7)


def test_logistic_fit_learns_separable_examples():
    x = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    y = np.array([0, 0, 1, 1])
    w, b, info = d.fit_logistic(x, y, .001, num_classes=2)
    result, predicted = d.score_logistic(x, y, w, b)
    assert info["success"]
    assert result["accuracy"] == 1
    np.testing.assert_array_equal(predicted, y)


def test_source_split_keeps_test_and_repairs_both_boundaries():
    cache = dict(folds=np.array([1, 2, 4, 4, 5, 4, 5]),
                 sources=np.array(["train", "a", "a", "b", "b", "val", "test"]))
    result = d.source_split(cache)
    assert result["train"].tolist() == [0]
    assert result["validation"].tolist() == [2, 5]
    assert result["test"].tolist() == [4, 6]


def test_grouped_bootstrap_identical_models_have_exact_zero_difference():
    result = d.grouped_bootstrap([1, 0, 1], [1, 0, 1], ["same", "same", "other"], replicates=50)
    assert result["groups"] == 2
    assert result["left_minus_right_interval"] == [0., 0.]


def test_metrics_reject_negative_and_fractional_labels():
    for invalid in ([-1], [.5], [50]):
        with pytest.raises(ValueError):
            d.metrics([0], invalid)


def test_comparison_aligns_filenames_and_rejects_changed_targets():
    left = [dict(filename="a", target=0, prediction=0), dict(filename="b", target=1, prediction=0)]
    right = [dict(filename="b", target=1, prediction=1), dict(filename="a", target=0, prediction=1)]
    metadata = [dict(filename="a", target="0", src_file="x"), dict(filename="b", target="1", src_file="y")]
    result = d.paired_comparison(left, right, metadata, replicates=20)
    assert result["left_only"] == 1 and result["right_only"] == 1
    right[0]["target"] = 0
    with pytest.raises(ValueError):
        d.paired_comparison(left, right, metadata)


def test_test_command_requires_unlock_before_loading_artifact():
    from argparse import Namespace
    with pytest.raises(ValueError, match="allow-test"):
        d.evaluate_test(Namespace(allow_test=False))


def make_tiny_cache(tmp_path):
    from esc50_sound.data import audit_metadata
    path = tmp_path / "features.npz"
    np.savez(path, x=np.zeros((2, 1, 64, 313), dtype=np.float32),
             y=np.array([0, 1]), folds=np.array([1, 4]), sources=np.array(["a", "b"]),
             names=np.array(["a.wav", "b.wav"]), class_names=np.array([str(i) for i in range(50)]))
    metadata = tmp_path / "ESC-50/meta/esc50.csv"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("filename,target,fold,src_file,category\na.wav,0,1,a,0\nb.wav,1,4,b,1\n")
    rows = [dict(filename="a.wav", target=0, fold=1, src_file="a", category="0"),
            dict(filename="b.wav", target=1, fold=4, src_file="b", category="1")]
    sidecar = dict(cache_sha256=d.sha256(path), frontend=d.EXPECTED_FRONTEND,
                   metadata_sha256=d.sha256(metadata), audit=audit_metadata(rows, require_complete=False))
    Path(str(path) + ".json").write_text(json.dumps(sidecar))
    return path, sidecar


def test_cache_rejects_inconsistent_digest_and_frontend(tmp_path):
    path, sidecar = make_tiny_cache(tmp_path)
    cache, _, _ = d.validated_cache(path)
    assert cache["x"].shape == (2, 1, 64, 313)
    sidecar["cache_sha256"] = "wrong"
    Path(str(path) + ".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="SHA256"):
        d.validated_cache(path)
    sidecar["cache_sha256"] = d.sha256(path)
    sidecar["frontend"] = dict(d.EXPECTED_FRONTEND, sample_rate=8000)
    Path(str(path) + ".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="frontend"):
        d.validated_cache(path)


def test_cache_checks_metadata_alignment_not_only_hash(tmp_path):
    path, sidecar = make_tiny_cache(tmp_path)
    metadata = tmp_path / "ESC-50/meta/esc50.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("filename,target,fold,src_file,category\na.wav,1,1,a,1\nb.wav,0,4,b,0\n")
    sidecar["metadata_sha256"] = d.sha256(metadata)
    Path(str(path) + ".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="rows disagree"):
        d.validated_cache(path)
