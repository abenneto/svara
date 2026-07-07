"""filterbanks 模块的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from svara.exceptions import InvalidParameterError
from svara.filterbanks import hz_to_mel, linear_filterbank, mel_filterbank, mel_to_hz


def test_hz_mel_roundtrip() -> None:
    hz = np.array([0.0, 100.0, 440.0, 1000.0, 8000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(hz)), hz, rtol=1e-6)


def test_mel_filterbank_shape() -> None:
    n_fft, n_filters = 512, 26
    fb = mel_filterbank(n_filters, n_fft, sample_rate=16000)
    assert fb.shape == (n_filters, n_fft // 2 + 1)


def test_mel_filterbank_nonnegative_and_peaks_at_one() -> None:
    fb = mel_filterbank(20, 1024, sample_rate=16000)
    assert np.all(fb >= 0.0)
    # 每个三角滤波器的峰值应接近 1
    assert np.all(fb.max(axis=1) > 0.5)


def test_mel_filters_are_higher_resolution_at_low_freq() -> None:
    # 低频滤波器覆盖的 FFT bin 数应少于高频滤波器
    fb = mel_filterbank(20, 1024, sample_rate=16000)
    active = (fb > 0).sum(axis=1)
    assert active[0] <= active[-1]


def test_linear_filterbank_shape() -> None:
    fb = linear_filterbank(24, 512, sample_rate=16000)
    assert fb.shape == (24, 257)
    assert np.all(fb >= 0.0)


def test_filterbank_rejects_bad_range() -> None:
    with pytest.raises(InvalidParameterError):
        mel_filterbank(10, 512, sample_rate=16000, fmin=4000.0, fmax=1000.0)
