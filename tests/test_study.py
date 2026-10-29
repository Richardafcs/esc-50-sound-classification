"""Check stress-test physics and the separate confirmation gate."""
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("esc50_study", ROOT / 'scripts/legacy/study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_white_noise_has_declared_snr_and_is_repeatable():
    wave = torch.sin(torch.arange(80000) * .11) * .1
    noisy = study.perturb(wave, "snr_10", 42)
    noise = noisy - wave
    snr = 10 * torch.log10(wave.square().mean()/noise.square().mean())
    assert snr.item() == pytest.approx(10, abs=1e-4)
    torch.testing.assert_close(noisy, study.perturb(wave, "snr_10", 42), atol=0, rtol=0)


def test_gain_applies_amplitude_db_not_power_db():
    wave = torch.ones(80000) * .1
    torch.testing.assert_close(study.perturb(wave, "gain_6", 0), wave * 10**(.3))


def test_snr_rejects_undefined_silent_reference():
    with pytest.raises(ValueError, match="silent"):
        study.perturb(torch.zeros(80000), "snr_20", 0)


def test_holdout_requires_registered_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test fixture")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"checkpoint_sha256": {"model": "wrong"}}))
    args = Namespace(split="test", allow_test=False, freeze=str(freeze), command="order", permutation_seeds=[101,202])
    with pytest.raises(ValueError, match="Holdout"):
        study.check_holdout(args, checkpoint)
    args.allow_test = True
    with pytest.raises(ValueError, match="registered"):
        study.check_holdout(args, checkpoint)
    freeze.write_text(json.dumps({"confirmation_schema_version":1,
                                 "checkpoint_sha256": {"model": study.esc.sha256(checkpoint)},
                                 "measurement_protocol":{"order":{"split":"test","permutation_seeds":[101,202]}}}))
    assert study.check_holdout(args, checkpoint)["sha256"] == study.esc.sha256(freeze)


def test_measurement_outputs_cannot_overwrite(tmp_path):
    path = tmp_path / "measurement"
    study.new_output(path)
    with pytest.raises(FileExistsError):
        study.new_output(path)
