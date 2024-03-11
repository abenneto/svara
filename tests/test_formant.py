"""formant 模块的测试。

用“源–滤波器”模型合成一个已知共振峰的元音：F0 冲激串（声源）经过若干
二阶谐振器（声道），再检验估计出的共振峰是否落在设定值附近。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from svara.formant import formants, levinson, lpc, track_formants

SR = 16000


def _resonator(freq: float, bandwidth: float, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2.0 * np.pi * freq / sr
    a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
    return np.array([1.0]), a


def _synth_vowel(f0: float, resonances: list[float], sr: int = SR, dur: float = 0.5) -> np.ndarray:
    n = int(sr * dur)
    period = int(round(sr / f0))
    source = np.zeros(n)
    source[::period] = 1.0
    sig = source
    for f in resonances:
        b, a = _resonator(f, bandwidth=80.0, sr=sr)
        sig = lfilter(b, a, sig)
    return (sig / np.max(np.abs(sig))).astype(np.float64)


def test_lpc_returns_leading_one() -> None:
    frame = np.random.default_rng(0).standard_normal(400)
    coeffs = lpc(frame, order=12)
    assert coeffs.shape == (13,)
    assert coeffs[0] == 1.0


def test_levinson_on_silence_is_stable() -> None:
    r = np.zeros(9)
    coeffs, err = levinson(r, order=8)
    assert err == 0.0
    assert coeffs[0] == 1.0


def test_formants_recover_synthetic_vowel() -> None:
    targets = [700.0, 1220.0, 2600.0]
    vowel = _synth_vowel(120.0, targets)
    frame = vowel[4000:4512]
    est = formants(frame, SR, max_formants=5)
    assert est.size >= 3
    for target in targets:
        assert np.min(np.abs(est - target)) < 150.0


def test_formants_are_sorted_and_below_nyquist() -> None:
    vowel = _synth_vowel(140.0, [500.0, 1500.0, 2500.0])
    est = formants(vowel[3000:3512], SR)
    assert np.all(np.diff(est) > 0)
    assert np.all(est <= SR / 2)


def test_track_formants_shape_and_padding() -> None:
    vowel = _synth_vowel(130.0, [650.0, 1080.0, 2650.0])
    track = track_formants(vowel, SR, n_formants=4, frame_length=400, hop_length=160)
    assert track.shape[1] == 4
    # 至少前两个共振峰在多数帧里应被检测到（非 NaN）
    detected = np.mean(~np.isnan(track[:, :2]))
    assert detected > 0.5
