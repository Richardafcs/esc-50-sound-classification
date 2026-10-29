"""Frequency-preserving controls and bounded diagnostic calculations; no training."""
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("esc50_frequency_tests", ROOT / 'scripts/legacy/esc50.py')
esc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esc)
sys.path.insert(0, str(ROOT))
study_spec = importlib.util.spec_from_file_location("study_frequency_tests", ROOT / 'scripts/legacy/study.py')
study = importlib.util.module_from_spec(study_spec)
study_spec.loader.exec_module(study)
torch.set_num_threads(2)


@pytest.mark.parametrize("name,expected", [("cnn_freq", 49058), ("crnn_freq", 251810),
                                          ("cnn_freq_matched", 252036)])
def test_frequency_models_parameter_counts_and_output_shape(name, expected):
    model = esc.build_model(name).eval()
    assert sum(p.numel() for p in model.parameters()) == expected
    with torch.no_grad():
        result = model(torch.randn(2, 1, 64, 313))
    assert result.shape == (2, 50) and torch.isfinite(result).all()


def test_frequency_models_start_from_identical_encoder_for_same_seed():
    encoders = []
    for name in ("cnn_freq", "crnn_freq", "cnn_freq_matched"):
        esc.seed_all(123)
        encoders.append(esc.build_model(name).encoder.state_dict())
    for key in encoders[0]:
        for other in encoders[1:]:
            torch.testing.assert_close(encoders[0][key], other[key], atol=0, rtol=0)


def test_frequency_sequence_preserves_channel_frequency_and_time_alignment():
    model = esc.build_model("crnn_freq").eval()
    x = torch.randn(2, 1, 64, 313)
    recorded = []
    hook = model.gru.register_forward_pre_hook(lambda _, inputs: recorded.append(inputs[0].detach().clone()))
    with torch.no_grad():
        features = model.encoder(x)
        model(x)
    hook.remove()
    assert recorded[0].shape == (2, 39, 512)
    for time in (0, 17, 38):
        torch.testing.assert_close(recorded[0][:, time], features[:, :, :, time].flatten(1), atol=0, rtol=0)


@pytest.mark.parametrize("name", ["cnn_freq", "cnn_freq_matched"])
def test_frequency_cnn_control_is_exactly_invariant_to_valid_token_order(name):
    model = esc.build_model(name).eval()
    x = torch.randn(2, 1, 64, 313)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x, time_permutation=torch.arange(38, -1, -1)), atol=0, rtol=0)


def test_frequency_gru_responds_to_order_and_rejects_bad_permutation():
    esc.seed_all(123)
    model = esc.build_model("crnn_freq").eval()
    x = torch.randn(2, 1, 64, 313)
    with torch.no_grad():
        assert not torch.equal(model(x), model(x, time_permutation=torch.arange(38, -1, -1)))
    for bad in (torch.arange(38), torch.arange(39) + 1, torch.zeros(39, dtype=torch.long)):
        with pytest.raises(ValueError, match="time_permutation"):
            model(x, time_permutation=bad)


@pytest.mark.parametrize("snr", [20, 10, 0, -5])
def test_white_noise_has_requested_rms_snr_and_is_repeatable(snr):
    wave = torch.sin(torch.arange(80000) * .1)
    noisy = study.perturb(wave, f"snr_{snr}", seed=123)
    error = noisy - wave
    observed = 10 * torch.log10(wave.square().mean() / error.square().mean())
    assert observed.item() == pytest.approx(snr, abs=1e-4)
    torch.testing.assert_close(noisy, study.perturb(wave, f"snr_{snr}", seed=123), atol=0, rtol=0)


def test_gain_uses_amplitude_decibels_and_silence_snr_is_rejected():
    wave = torch.ones(80000) * .25
    torch.testing.assert_close(study.perturb(wave, "gain_6.020599913279624", 1), wave * 2)
    with pytest.raises(ValueError, match="silent"):
        study.perturb(torch.zeros(80000), "snr_20", 1)
    with pytest.raises(ValueError, match="Unknown"):
        study.perturb(wave, "unrecognized", 1)


def test_holdout_probe_requires_both_unlock_and_registered_frozen_checkpoint(tmp_path):
    checkpoint = tmp_path / "synthetic_checkpoint.bin"
    checkpoint.write_bytes(b"not a model: hash-registration fixture only")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"confirmation_schema_version":1,
                                 "checkpoint_sha256": {"fixture": study.esc.sha256(checkpoint)},
                                 "measurement_protocol":{"order":{"split":"test","permutation_seeds":[101]}}}))
    assert study.check_holdout(Namespace(split="validation"), checkpoint) is None
    for args in (Namespace(split="test", allow_test=False, freeze=str(freeze)),
                 Namespace(split="test", allow_test=True, freeze=None)):
        with pytest.raises(ValueError, match="Holdout"):
            study.check_holdout(args, checkpoint)
    assert study.check_holdout(Namespace(split="test", allow_test=True, freeze=str(freeze), command="order", permutation_seeds=[101]), checkpoint)["sha256"] == study.esc.sha256(freeze)
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="not registered"):
        study.check_holdout(Namespace(split="test", allow_test=True, freeze=str(freeze)), checkpoint)


def test_study_split_preserves_checkpoint_official_or_strict_source_policy():
    cache = {"folds": [1, 4, 4, 5], "sources": ["training", "validation", "shared", "shared"]}
    strict = study.split_ids(cache, {"manifest": {"config": {"official_folds": False}}})
    official = study.split_ids(cache, {"manifest": {"config": {"official_folds": True}}})
    assert strict["validation"] == [1]
    assert official["validation"] == [1, 2]
    assert strict["test"] == official["test"] == [3]


def test_study_holdout_measurement_records_access_before_completion(tmp_path):
    checkpoint_path = tmp_path / "synthetic_checkpoint.bin"
    checkpoint_path.write_bytes(b"audit fixture; no forward pass")
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({"manifest_id": "synthetic", "test_evaluated": False}))
    checkpoint = {"manifest_id": "synthetic", "source_policy": "strict fixture"}
    args = Namespace(split="test", command="order", checkpoint=str(checkpoint_path))
    audit = study.begin_measurement_audit(args, checkpoint, tmp_path / "measurement", {"sha256": "frozen fixture"})
    path, record = audit
    before = path.read_bytes()
    assert json.loads(before)["status"] == "started"
    assert json.loads(run_path.read_text())["test_evaluated"] is True
    study.complete_measurement_audit(audit)
    assert path.read_bytes().startswith(before)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["status"] for r in records] == ["started", "completed"]
    assert records[0]["evaluation_id"] == records[1]["evaluation_id"]


def test_study_validation_measurement_does_not_mark_test_access(tmp_path):
    assert study.begin_measurement_audit(Namespace(split="validation"), {}, tmp_path, None) is None
    assert not (tmp_path / "evaluation_history.jsonl").exists()


def test_cosine_schedule_expected_bounds_and_epoch_boundary_values():
    """No gradients or learning: verify declared epoch schedule's numerical contract."""
    weight = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([weight], lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()  # No gradient: parameter stays unchanged.
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[0] == .001 and values[2] == pytest.approx(.0005) and values[-1] == pytest.approx(0)
    assert all(0 <= value <= .001 for value in values)
    assert all(left >= right for left, right in zip(values, values[1:]))
    assert weight.item() == 0
