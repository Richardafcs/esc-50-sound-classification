"""Independent checks of leakage controls, frontend behavior, and metric arithmetic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("esc50", ROOT / 'scripts/legacy/esc50.py')
esc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esc)
torch.set_num_threads(2)


def test_source_isolation_repairs_both_boundaries():
    rows = [dict(fold=f, src_file=s) for f, s in
            [(1, "train"), (2, "train-val"), (4, "train-val"),
             (4, "val-test"), (5, "val-test"), (4, "val"), (5, "test")]]
    split = esc.split_rows(rows)
    assert split["train"] == [0]
    assert split["validation"] == [2, 5]
    assert split["test"] == [4, 6]
    sources = [{rows[i]["src_file"] for i in split[k]}
               for k in ("train", "validation", "test")]
    assert all(not sources[i] & sources[j] for i in range(3) for j in range(i))
    assert esc.split_rows(rows, strict=False)["validation"] == [2, 3, 5]


def test_overlapping_fold_request_is_rejected():
    with pytest.raises(ValueError):
        esc.split_rows([], train_folds=(1, 4))


def test_normalization_uses_only_supplied_training_examples():
    train = np.array([0., 2.], dtype=np.float32).reshape(2, 1, 1, 1)
    mean, std = esc.fit_normalization(train)
    assert mean.item() == 1 and std.item() == 1
    heldout = np.full((1, 1, 1, 1), 1000., dtype=np.float32)
    assert ((heldout - mean) / std).item() == 999


def test_log_mel_silence_and_amplitude_scaling():
    silence = esc.log_mel(torch.zeros(80000))
    assert silence.shape == (1, 64, 313)
    assert torch.isfinite(silence).all()
    t = torch.arange(80000) / 16000
    wave = torch.sin(2 * torch.pi * 1000 * t)
    full, half = esc.log_mel(wave), esc.log_mel(wave / 2)
    active = full > -5
    # Power scales by amplitude squared: ln(P) - ln(P/4) = ln(4).
    torch.testing.assert_close((full - half)[active],
                               torch.full_like(full[active], np.log(4)), atol=1e-4, rtol=1e-4)


def test_metrics_match_hand_computed_confusion():
    m = esc.compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["accuracy"] == .75
    assert m["macro_f1"] == pytest.approx((2/3 + 4/5) / 2)


@pytest.mark.parametrize("name", ["cnn", "crnn"])
def test_checkpoint_roundtrip_preserves_logits(name, tmp_path):
    model = esc.build_model(name).eval()
    x = torch.randn(2, 1, 64, 313)
    with torch.no_grad():
        before = model(x)
    path = tmp_path / "weights.pt"
    torch.save(dict(model=model.state_dict()), path)
    restored = esc.build_model(name).eval()
    restored.load_state_dict(esc.load_checkpoint(path)["model"])
    with torch.no_grad():
        torch.testing.assert_close(before, restored(x), atol=0, rtol=0)


def test_final_test_requires_explicit_unlock():
    from argparse import Namespace
    with pytest.raises(ValueError, match="sealed"):
        esc.evaluate(Namespace(allow_test=False))
