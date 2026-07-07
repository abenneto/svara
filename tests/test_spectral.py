"""spectral 模块的测试。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from svara.spectral import (
    fft_frequencies,
    magnitude_spectrum,
    power_spectrum,
    spectrogram,
    stft,
)


def test_stft_shape(sine: Callable[..., np.ndarray]) -> None:
    x = sine(freq=440.0, duration=0.5)
    n_fft, hop = 512, 128
    spec = stft(x, n_fft=n_fft, hop_length=hop)
    assert spec.shape[1] == n_fft // 2 + 1
    assert spec.dtype == np.complex128


def test_power_is_magnitude_squared(sine: Callable[..., np.ndarray]) -> None:
    spec = stft(sine(duration=0.2), n_fft=256)
    np.testing.assert_allclose(power_spectrum(spec), magnitude_spectrum(spec) ** 2, rtol=1e-6)


def test_spectrogram_peak_at_tone(sine: Callable[..., np.ndarray], sr: int) -> None:
    freq = 1000.0
    n_fft = 2048
    x = sine(freq=freq, sr=sr, duration=0.5)
    psd = spectrogram(x, n_fft=n_fft, hop_length=512, power=2.0)
    freqs = fft_frequencies(sr, n_fft)
    mean_frame = psd.mean(axis=0)
    peak_hz = freqs[int(np.argmax(mean_frame))]
    # 峰值应落在正弦频率附近（一个频率栅格以内）
    assert abs(peak_hz - freq) <= sr / n_fft


def test_spectrogram_power_one_is_magnitude(sine: Callable[..., np.ndarray]) -> None:
    x = sine(duration=0.2)
    mag = spectrogram(x, n_fft=256, power=1.0)
    assert np.all(mag >= 0.0)
