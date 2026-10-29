"""Waveform loading and the fixed, checkpoint-compatible log-mel frontend.

Audio has digital amplitude units; the natural-log power transform is not SPL.
"""
from __future__ import annotations

import math

import numpy as np
import torch

FRONTEND = dict(sample_rate=16000, seconds=5, n_fft=512, hop_length=256,
                n_mels=64, mel_scale="htk", center=True, log="natural", floor=1e-10)


def read_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all() or not len(audio):
        raise ValueError("Audio must be nonempty and finite")
    audio = audio.mean(axis=1)
    target = FRONTEND["sample_rate"]
    if rate != target:
        common = math.gcd(rate, target)
        audio = resample_poly(audio, target // common, rate // common).astype(np.float32)
    length = target * FRONTEND["seconds"]
    # Fixed center crop/zero padding; no random validation or inference transforms.
    if len(audio) > length:
        start = (len(audio) - length) // 2
        audio = audio[start:start + length]
    elif len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    return torch.from_numpy(audio.copy())


def log_mel(waveform):
    """Mono [80000] -> natural-log power [1,64,313], HTK triangles.

    16 kHz / 256 samples = 62.5 frames/s; center padding yields 313 frames.
    Magnitude is squared before triangular mel integration. Silence stays finite.
    """
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim != 1 or waveform.numel() != 80000 or not torch.isfinite(waveform).all():
        raise ValueError("Expected finite mono waveform with exactly 80000 samples")
    window = torch.hann_window(512, device=waveform.device)
    power = torch.stft(waveform, 512, 256, window=window, center=True,
                       return_complex=True).abs().square()
    mel_max = 2595 * math.log10(1 + 8000 / 700)
    points = torch.linspace(0, mel_max, 66, device=waveform.device)
    hz = 700 * (10 ** (points / 2595) - 1)
    freq = torch.linspace(0, 8000, 257, device=waveform.device)
    left = (freq[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
    right = (hz[2:, None] - freq[None]) / (hz[2:, None] - hz[1:-1, None])
    filters = torch.minimum(left, right).clamp_min(0)
    return (filters @ power).clamp_min(1e-10).log().unsqueeze(0)


