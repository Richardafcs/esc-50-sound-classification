"""Shared convolutional encoder and explicit pooled, tokenwise and recurrent heads.

Model names and state-dict paths are stable so existing tensor checkpoints load
without conversion. Head size matches parameter budgets, not compute or latency.
"""
from __future__ import annotations

import torch
from torch import nn

MODEL_NAMES = (
    "cnn", "crnn", "cnn_matched", "cnn_freq", "crnn_freq",
    "cnn_freq_matched", "cnn_freq_token",
)


class ConvolutionalEncoder(nn.Sequential):
    """[B,1,64,313] -> [B,64,8,39] using three conv/BN/ReLU/pool stages."""

    def __init__(self):
        layers = []
        channels = 1
        for width in (16, 32, 64):
            layers.extend([
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            ])
            channels = width
        super().__init__(*layers)


def linear_head(input_features, num_classes):
    return nn.Sequential(nn.Dropout(0.2), nn.Linear(input_features, num_classes))


def matched_head(input_features, num_classes):
    """Pooled MLP with approximately the corresponding BiGRU's parameter count."""
    gru_parameters = 384 * input_features + 25344  # Bidirectional GRU hidden size64.
    hidden = round((gru_parameters + 128 * num_classes) / (input_features + 1 + num_classes))
    return nn.Sequential(
        nn.Dropout(0.2), nn.Linear(input_features, hidden), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(hidden, num_classes),
    )


def permute_time_tokens(tokens, permutation):
    """Apply a validated intervention after encoding, never to the raw waveform."""
    permutation = torch.as_tensor(permutation, device=tokens.device)
    if (permutation.ndim != 1 or permutation.numel() != tokens.shape[1] or
            permutation.dtype not in (torch.int32, torch.int64) or
            not torch.equal(permutation.sort().values,
                            torch.arange(tokens.shape[1], device=tokens.device))):
        raise ValueError("time_permutation must contain every encoder time index exactly once")
    return tokens.index_select(1, permutation.long())


class PooledClassifier(nn.Module):
    """Order-invariant aggregation, with optional frequency positions and token MLP.

    ``tokenwise`` maps each encoded token before averaging. Otherwise the head
    receives the clip average. Both variants have no recurrent state.
    """

    def __init__(self, num_classes=50, *, preserve_frequency=False, matched=False, tokenwise=False):
        super().__init__()
        self.encoder = ConvolutionalEncoder()
        self.recurrent = False
        self.frequency = preserve_frequency
        self.tokenwise = tokenwise
        features = 512 if preserve_frequency else 64
        self.head = matched_head(features, num_classes) if matched else linear_head(features, num_classes)

    def forward(self, x, time_permutation=None):
        x = self.encoder(x)
        # Ignore the permutation for exact control invariance, avoiding even a
        # change in floating-point summation order. Inference has dropout off.
        if self.tokenwise:
            x = x.permute(0, 3, 1, 2).flatten(2)
            return self.head(x).mean(1)
        x = x.mean(3).flatten(1) if self.frequency else x.mean((2, 3))
        return self.head(x)


class RecurrentClassifier(nn.Module):
    """BiGRU over 39 encoded time tokens, followed by temporal averaging."""

    def __init__(self, num_classes=50, *, preserve_frequency=False):
        super().__init__()
        self.encoder = ConvolutionalEncoder()
        self.recurrent = True
        self.frequency = preserve_frequency
        self.tokenwise = False
        features = 512 if preserve_frequency else 64
        self.gru = nn.GRU(features, 64, batch_first=True, bidirectional=True)
        self.head = linear_head(128, num_classes)

    def forward(self, x, time_permutation=None):
        x = self.encoder(x)
        if self.frequency:
            x = x.permute(0, 3, 1, 2).flatten(2)  # [B,39,512].
        else:
            x = x.mean(2).transpose(1, 2)  # [B,39,64].
        if time_permutation is not None:
            x = permute_time_tokens(x, time_permutation)
        x, _ = self.gru(x)
        return self.head(x.mean(1))


def build_model(name, num_classes=50):
    """Construct one registered experimental architecture; no pretrained weights."""
    if name == "cnn":
        return PooledClassifier(num_classes)
    if name == "crnn":
        return RecurrentClassifier(num_classes)
    if name == "cnn_matched":
        return PooledClassifier(num_classes, matched=True)
    if name == "cnn_freq":
        return PooledClassifier(num_classes, preserve_frequency=True)
    if name == "crnn_freq":
        return RecurrentClassifier(num_classes, preserve_frequency=True)
    if name == "cnn_freq_matched":
        return PooledClassifier(num_classes, preserve_frequency=True, matched=True)
    if name == "cnn_freq_token":
        return PooledClassifier(num_classes, preserve_frequency=True, matched=True, tokenwise=True)
    raise ValueError(f"Unknown model: {name}")
