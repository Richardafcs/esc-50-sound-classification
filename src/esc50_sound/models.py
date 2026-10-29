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
    "crnn_freq_ta", "crnn_freq_ca", "crnn_freq_tca",
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


class TemporalAttentionPooling(nn.Module):
    """Learned clip-level aggregation of [B,T,D] recurrent features.

    The initial softmax distribution is uniform, exactly representing mean
    pooling in real arithmetic. Score weights are shared across time steps.
    """

    def __init__(self, features: int = 128):
        super().__init__()
        self.score = nn.Linear(features, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, tokens: torch.Tensor, *, return_weights: bool = False):
        if tokens.ndim != 3:
            raise ValueError("Temporal attention expects [batch,time,features]")
        weights = self.score(tokens).softmax(dim=1)  # [B,T,1].
        pooled = (tokens * weights).sum(dim=1)  # [B,D].
        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled


class ChannelAttention(nn.Module):
    """SE-style 64-channel reweighting before the recurrent frequency flatten.

    Using 2*sigmoid on the final logits permits identity scaling on init.
    This is a deliberate lightweight adaptation, not a verbatim Zhang (2019)
    reproduction. First MLP gradients start at zero until the last layer moves.
    """

    def __init__(self, channels: int = 64, reduction: int = 8):
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor, *, return_weights: bool = False):
        if features.ndim != 4 or features.shape[1] != self.fc2.out_features:
            raise ValueError("Channel attention expects [batch,channels,freq,time]")
        summary = features.mean(dim=(2, 3))
        gates = 2 * torch.sigmoid(self.fc2(torch.relu(self.fc1(summary))))
        scaled = features * gates[:, :, None, None]
        if return_weights:
            return scaled, gates
        return scaled


class RecurrentClassifier(nn.Module):
    """BiGRU over 39 encoded time tokens, followed by temporal averaging."""

    def __init__(self, num_classes=50, *, preserve_frequency=False,
                 temporal_attention=False, channel_attention=False):
        super().__init__()
        if channel_attention and not preserve_frequency:
            raise ValueError("Channel attention variants require preserve_frequency=True")
        self.encoder = ConvolutionalEncoder()
        self.recurrent = True
        self.frequency = preserve_frequency
        self.tokenwise = False
        features = 512 if preserve_frequency else 64
        self.gru = nn.GRU(features, 64, batch_first=True, bidirectional=True)
        self.head = linear_head(128, num_classes)
        # Add optional modules *after* the legacy layers so the same manual
        # seed leaves encoder, GRU and head initialization unchanged.
        self.temporal_attention = TemporalAttentionPooling(128) if temporal_attention else None
        self.channel_attention = ChannelAttention(64, reduction=8) if channel_attention else None

    def forward(self, x, time_permutation=None, *, return_attention=False):
        x = self.encoder(x)
        diagnostics = {}
        if self.channel_attention is not None:
            if return_attention:
                x, diagnostics["channel"] = self.channel_attention(x, return_weights=True)
            else:
                x = self.channel_attention(x)
        if self.frequency:
            x = x.permute(0, 3, 1, 2).flatten(2)  # [B,39,512].
        else:
            x = x.mean(2).transpose(1, 2)  # [B,39,64].
        if time_permutation is not None:
            x = permute_time_tokens(x, time_permutation)
        x, _ = self.gru(x)
        if self.temporal_attention is not None:
            if return_attention:
                x, diagnostics["temporal"] = self.temporal_attention(x, return_weights=True)
            else:
                x = self.temporal_attention(x)
        else:
            x = x.mean(1)
        logits = self.head(x)
        return (logits, diagnostics) if return_attention else logits


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
    if name in ("crnn_freq_ta", "crnn_freq_ca", "crnn_freq_tca"):
        return RecurrentClassifier(
            num_classes, preserve_frequency=True,
            temporal_attention=name in ("crnn_freq_ta", "crnn_freq_tca"),
            channel_attention=name in ("crnn_freq_ca", "crnn_freq_tca"),
        )
    if name == "cnn_freq_matched":
        return PooledClassifier(num_classes, preserve_frequency=True, matched=True)
    if name == "cnn_freq_token":
        return PooledClassifier(num_classes, preserve_frequency=True, matched=True, tokenwise=True)
    raise ValueError(f"Unknown model: {name}")
