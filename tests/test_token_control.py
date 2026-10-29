"""Distinguish nonlinear content aggregation from recurrent temporal ordering."""
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import esc50


def test_token_control_matches_capacity_and_encoder():
    torch.manual_seed(17)
    pooled=esc50.build_model('cnn_freq_matched')
    torch.manual_seed(17)
    token=esc50.build_model('cnn_freq_token')
    assert sum(p.numel() for p in token.parameters())==252036
    for a,b in zip(pooled.state_dict().values(),token.state_dict().values()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_token_control_applies_nonlinearity_before_pooling():
    model=esc50.build_model('cnn_freq_token').eval()
    model.encoder=torch.nn.Identity()
    encoded=torch.randn(2,64,8,39)
    with torch.inference_mode():
        actual=model(encoded)
        expected=model.head(encoded.permute(0,3,1,2).flatten(2)).mean(1)
        pooled=model.head(encoded.mean(3).flatten(1))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert not torch.allclose(actual,pooled)
    torch.testing.assert_close(actual,model(encoded,time_permutation=torch.randperm(39)),atol=0,rtol=0)


def test_token_control_waveform_feature_shape():
    model=esc50.build_model('cnn_freq_token').eval()
    assert model(torch.zeros(2,1,64,313)).shape==(2,50)
