"""短时傅里叶变换及其派生的功率 / 幅度谱。"""

from __future__ import annotations

import numpy as np

from svara.framing import apply_window, frame_signal
from svara.utils import FloatArray, as_float_array, check_positive

ComplexArray = np.ndarray


def stft(
    signal: FloatArray,
    n_fft: int = 512,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
) -> ComplexArray:
    """计算短时傅里叶变换，返回 ``(n_frames, n_fft // 2 + 1)`` 的复数谱。

    只保留非负频率（``rfft``）。当 ``win_length < n_fft`` 时，加窗后的帧会
    补零到 ``n_fft`` 再做 FFT，从而在不牺牲时间分辨率的情况下细化频率栅格。
    """
    check_positive("n_fft", n_fft)
    hop = hop_length if hop_length is not None else n_fft // 4
    win = win_length if win_length is not None else n_fft
    if win > n_fft:
        raise ValueError("win_length 不能大于 n_fft")

    x = as_float_array(signal)
    frames = frame_signal(x, frame_length=win, hop_length=hop)
    windowed = apply_window(frames, window=window)
    if win < n_fft:
        pad = n_fft - win
        windowed = np.pad(windowed, ((0, 0), (0, pad)))
    return np.fft.rfft(windowed, n=n_fft, axis=1)


def magnitude_spectrum(complex_spec: ComplexArray) -> FloatArray:
    """复数谱的幅度 ``|X|``。"""
    return np.abs(complex_spec).astype(np.float64)


def power_spectrum(complex_spec: ComplexArray) -> FloatArray:
    """复数谱的功率 ``|X|**2``。"""
    mag = np.abs(complex_spec)
    return (mag * mag).astype(np.float64)
