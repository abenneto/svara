"""梅尔谱与对数梅尔谱。"""

from __future__ import annotations

import numpy as np

from svara.filterbanks import mel_filterbank
from svara.spectral import spectrogram
from svara.utils import FloatArray


def mel_spectrogram(
    signal: FloatArray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int | None = None,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
    window: str = "hann",
) -> FloatArray:
    """把功率谱通过梅尔滤波器组，得到 ``(n_frames, n_mels)`` 的梅尔谱。"""
    psd = spectrogram(
        signal, n_fft=n_fft, hop_length=hop_length, window=window, power=2.0
    )
    fb = mel_filterbank(n_mels, n_fft, sample_rate, fmin=fmin, fmax=fmax)
    return psd @ fb.T


def log_mel_spectrogram(
    signal: FloatArray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int | None = None,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
    window: str = "hann",
    log_offset: float = 1e-6,
) -> FloatArray:
    """梅尔谱取对数。``log_offset`` 用来避免 ``log(0)``。"""
    mel = mel_spectrogram(
        signal,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window=window,
    )
    return np.log(mel + log_offset)
