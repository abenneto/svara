"""pitch 模块的测试。

纯正弦用于验证自相关 / YIN；倒谱法依赖谐波结构，因此单独用一个多谐波信号。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from svara.exceptions import InvalidParameterError
from svara.pitch import (
    estimate_f0,
    f0_autocorrelation,
    f0_cepstrum,
    f0_yin,
    voiced_flags,
)


def _harmonic(
    f0: float, sr: int = 16000, duration: float = 0.5, n_harmonics: int = 8
) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        sig += np.sin(2.0 * np.pi * k * f0 * t) / k
    return (sig / np.max(np.abs(sig))).astype(np.float64)


def _voiced_median(f0: np.ndarray) -> float:
    voiced = f0[f0 > 0]
    assert voiced.size > 0, "没有检测到浊音帧"
    return float(np.median(voiced))


def test_yin_matches_known_pitch(sine: Callable[..., np.ndarray], sr: int) -> None:
    f0 = f0_yin(sine(freq=220.0, sr=sr, duration=0.5), sr)
    assert _voiced_median(f0) == pytest.approx(220.0, abs=3.0)


def test_autocorrelation_matches_known_pitch(sine: Callable[..., np.ndarray], sr: int) -> None:
    f0 = f0_autocorrelation(sine(freq=180.0, sr=sr, duration=0.5), sr)
    assert _voiced_median(f0) == pytest.approx(180.0, abs=5.0)


def test_cepstrum_matches_known_pitch(sr: int) -> None:
    f0 = f0_cepstrum(_harmonic(150.0, sr=sr, duration=0.5), sr)
    assert _voiced_median(f0) == pytest.approx(150.0, abs=5.0)


def test_estimate_f0_dispatches(sine: Callable[..., np.ndarray], sr: int) -> None:
    f0 = estimate_f0(sine(freq=200.0, sr=sr, duration=0.4), sr, method="yin")
    assert _voiced_median(f0) == pytest.approx(200.0, abs=3.0)


def test_estimate_f0_rejects_unknown_method(sine: Callable[..., np.ndarray], sr: int) -> None:
    with pytest.raises(InvalidParameterError):
        estimate_f0(sine(duration=0.2), sr, method="nope")


def test_noise_is_mostly_unvoiced(white_noise: Callable[..., np.ndarray], sr: int) -> None:
    f0 = f0_yin(white_noise(duration=0.5, seed=1), sr)
    assert voiced_flags(f0).mean() < 0.5


def test_voiced_flags_dtype() -> None:
    flags = voiced_flags(np.array([0.0, 120.0, 0.0, 200.0]))
    assert flags.tolist() == [False, True, False, True]
