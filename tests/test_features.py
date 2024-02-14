"""features 子包的测试：倒谱、谱形状、时域、差分。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from svara.features.cepstral import lfcc, mfcc
from svara.features.dynamic import delta, delta_delta
from svara.features.shape import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
)
from svara.features.temporal import rms_energy, zero_crossing_rate
from svara.spectral import fft_frequencies, spectrogram


def test_mfcc_shape(sine: Callable[..., np.ndarray], sr: int) -> None:
    m = mfcc(sine(duration=0.5), sr, n_mfcc=13, n_fft=512, hop_length=160)
    assert m.ndim == 2
    assert m.shape[1] == 13


def test_lfcc_shape(sine: Callable[..., np.ndarray], sr: int) -> None:
    c = lfcc(sine(duration=0.5), sr, n_lfcc=13)
    assert c.shape[1] == 13


def test_delta_of_constant_is_zero() -> None:
    feats = np.tile(np.arange(5.0), (10, 1))
    d = delta(feats, width=2)
    np.testing.assert_allclose(d, 0.0, atol=1e-12)


def test_delta_shapes_match() -> None:
    feats = np.random.default_rng(0).standard_normal((20, 13))
    assert delta(feats).shape == feats.shape
    assert delta_delta(feats).shape == feats.shape


def test_delta_tracks_linear_ramp() -> None:
    # 单列线性上升，一阶差分应为常数斜率
    ramp = np.linspace(0.0, 10.0, 50)[:, None]
    d = delta(ramp, width=2)
    assert np.allclose(d[5:-5], d[10], atol=1e-9)


def test_spectral_centroid_tracks_tone(sine: Callable[..., np.ndarray], sr: int) -> None:
    freq = 2000.0
    mag = spectrogram(sine(freq=freq, duration=0.5), n_fft=2048, hop_length=512, power=1.0)
    freqs = fft_frequencies(sr, 2048)
    centroid = spectral_centroid(mag, freqs).mean()
    assert abs(centroid - freq) < 200.0


def test_flatness_noise_gt_tone(
    sine: Callable[..., np.ndarray], white_noise: Callable[..., np.ndarray]
) -> None:
    tone = spectrogram(sine(freq=1000.0, duration=0.5), n_fft=1024, power=1.0)
    noise = spectrogram(white_noise(duration=0.5), n_fft=1024, power=1.0)
    assert spectral_flatness(noise).mean() > spectral_flatness(tone).mean()


def test_rolloff_and_bandwidth_are_finite(sine: Callable[..., np.ndarray], sr: int) -> None:
    mag = spectrogram(sine(freq=1500.0, duration=0.3), n_fft=1024, power=1.0)
    freqs = fft_frequencies(sr, 1024)
    assert np.all(np.isfinite(spectral_rolloff(mag, freqs)))
    assert np.all(spectral_bandwidth(mag, freqs) >= 0.0)


def test_zcr_higher_for_high_freq(sine: Callable[..., np.ndarray], sr: int) -> None:
    low = zero_crossing_rate(sine(freq=100.0, sr=sr, duration=0.3))
    high = zero_crossing_rate(sine(freq=4000.0, sr=sr, duration=0.3))
    assert high.mean() > low.mean()


def test_rms_scales_with_amplitude(sine: Callable[..., np.ndarray]) -> None:
    quiet = rms_energy(sine(amp=0.1, duration=0.3))
    loud = rms_energy(sine(amp=0.9, duration=0.3))
    assert loud.mean() > quiet.mean()
