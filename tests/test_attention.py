"""Attention architectural contract and compatibility checks; no ESC-50 files required."""
import torch
import pytest

from esc50_sound.models import (
    ChannelAttention, TemporalAttentionPooling, MODEL_NAMES, build_model,
)
from esc50_sound.records import seed_all

torch.set_num_threads(2)


@pytest.mark.parametrize("name,parameters", [
    ("crnn_freq", 251810),
    ("crnn_freq_ta", 251939),
    ("crnn_freq_ca", 252906),
    ("crnn_freq_tca", 253035),
])
def test_attention_variants_are_registered_and_keep_expected_parameter_counts(name, parameters):
    assert name in MODEL_NAMES
    model = build_model(name).eval()
    assert sum(p.numel() for p in model.parameters()) == parameters
    x = torch.randn(2, 1, 64, 313)
    with torch.no_grad():
        scores, weights = model(x, return_attention=True)
    assert scores.shape == (2, 50)
    assert torch.isfinite(scores).all()
    if "ta" in name or name.endswith("tca"):
        assert weights["temporal"].shape == (2, 39)
        torch.testing.assert_close(weights["temporal"].sum(1), torch.ones(2))
    else:
        assert "temporal" not in weights
    if name.endswith("ca") or name.endswith("tca"):
        assert weights["channel"].shape == (2, 64)
    else:
        assert "channel" not in weights


def test_temporal_pooling_is_uniform_and_matches_mean_at_init():
    pooling = TemporalAttentionPooling(features=128)
    x = torch.randn(3, 39, 128)
    pooled, weights = pooling(x, return_weights=True)
    torch.testing.assert_close(pooled, x.mean(1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 39), atol=1e-7, rtol=1e-7)


def test_channel_attention_initially_preserves_full_feature_map():
    layer = ChannelAttention(channels=64, reduction=8)
    x = torch.randn(2, 64, 8, 39)
    gated, weights = layer(x, return_weights=True)
    torch.testing.assert_close(gated, x, atol=0, rtol=0)
    torch.testing.assert_close(weights, torch.ones_like(weights), atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["crnn_freq_ta", "crnn_freq_ca", "crnn_freq_tca"])
def test_base_model_state_dict_is_unchanged_and_init_predictions_match(variant):
    seed_all(123)
    base = build_model("crnn_freq").eval()
    seed_all(123)
    augmented = build_model(variant).eval()
    original = base.state_dict()
    augmented_dict = augmented.state_dict()
    for key, original_value in original.items():
        assert key in augmented_dict
        torch.testing.assert_close(original_value, augmented_dict[key], atol=0, rtol=0)
    x = torch.randn(2, 1, 64, 313)
    with torch.no_grad():
        torch.testing.assert_close(base(x), augmented(x), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name", ["crnn_freq_ta", "crnn_freq_ca", "crnn_freq_tca"])
def test_attention_model_forward_backward_and_checkpoint_round_trip(name, tmp_path):
    seed_all(101)
    model = build_model(name).train()
    x = torch.randn(2, 1, 64, 313)
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([3, 7]))
    assert torch.isfinite(loss)
    loss.backward()
    for key, layer in [("temporal_attention", getattr(model, "temporal_attention")),
                       ("channel_attention", getattr(model, "channel_attention"))]:
        if layer is not None:
            # Even with zero identity initialization, the output projection learns
            # on the first step (the first SE projection gets gradients later).
            final = layer.score if key == "temporal_attention" else layer.fc2
            assert final.weight.grad is not None
            assert torch.isfinite(final.weight.grad).all()
            assert final.weight.grad.abs().sum().item() > 0
    assert model.encoder[0].weight.grad is not None
    assert model.gru.weight_ih_l0.grad is not None
    path = tmp_path / "attention.pt"
    model.eval()
    with torch.no_grad():
        expected = model(x)
    torch.save({"model": model.state_dict()}, path)
    new = build_model(name).eval()
    new.load_state_dict(torch.load(path, weights_only=True)["model"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(new(x), expected, atol=0, rtol=0)


def test_attention_order_intervention_and_diagnostics_shapes():
    seed_all(7)
    model = build_model("crnn_freq_tca").eval()
    x = torch.randn(2, 1, 64, 313)
    with torch.no_grad():
        baseline, original = model(x, return_attention=True)
        scrambled, changed = model(x, time_permutation=torch.arange(38, -1, -1),
                                   return_attention=True)
    assert not torch.equal(baseline, scrambled)
    assert changed["temporal"].shape == original["temporal"].shape
    assert changed["channel"].shape == original["channel"].shape
    with pytest.raises(ValueError, match="time_permutation"):
        model(x, time_permutation=torch.zeros(39, dtype=torch.long))


def test_attention_shape_validation_and_small_channel_count():
    with pytest.raises(ValueError):
        TemporalAttentionPooling(128)(torch.rand(2, 128))
    with pytest.raises(ValueError):
        ChannelAttention(64)(torch.rand(2, 64, 39))
    with pytest.raises(ValueError):
        ChannelAttention(64)(torch.rand(2, 63, 8, 39))
    with pytest.raises(ValueError):
        ChannelAttention(0)
    assert ChannelAttention(4, reduction=8).fc1.out_features == 1
