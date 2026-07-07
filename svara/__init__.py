"""svara：语音特征提取与声学表示工具箱。

顶层重新导出最常用的接口，因此大多数场景下 ``import svara`` 即可：

>>> import svara
>>> signal, sr = svara.read_wav("hello.wav")
>>> feats = svara.mfcc(signal, sr)
>>> f0 = svara.estimate_f0(signal, sr, method="yin")

更细的功能仍可从各子模块导入。
"""

from __future__ import annotations

from svara.features.cepstral import lfcc, mfcc
from svara.features.deltas import delta, delta_delta
from svara.features.melspec import log_mel_spectrogram, mel_spectrogram
from svara.formant import formants, track_formants
from svara.framing import frame_signal, preemphasis
from svara.io import read_wav, resample, write_wav
from svara.pitch import estimate_f0, voiced_flags
from svara.spectral import spectrogram, stft

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "delta",
    "delta_delta",
    "estimate_f0",
    "formants",
    "frame_signal",
    "lfcc",
    "log_mel_spectrogram",
    "mel_spectrogram",
    "mfcc",
    "preemphasis",
    "read_wav",
    "resample",
    "spectrogram",
    "stft",
    "track_formants",
    "voiced_flags",
    "write_wav",
]
