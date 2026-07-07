"""梅尔 / 线性三角滤波器组，以及相关的频率刻度转换。

约定：所有滤波器组返回形状为 ``(n_filters, n_fft // 2 + 1)`` 的矩阵，
可直接与功率谱做 ``power_spectrum @ fb.T`` 得到各滤波器的能量。
"""

from __future__ import annotations

import numpy as np

from svara.exceptions import InvalidParameterError
from svara.utils import FloatArray, check_positive


def hz_to_mel(hz: FloatArray | float) -> FloatArray:
    """Hz -> 梅尔（HTK 公式 ``2595 * log10(1 + f / 700)``）。"""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: FloatArray | float) -> FloatArray:
    """梅尔 -> Hz（:func:`hz_to_mel` 的逆变换）。"""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _triangular_bank(hz_points: FloatArray, fft_freqs: FloatArray) -> FloatArray:
    """由 ``n_filters + 2`` 个中心频点一次性构造三角滤波器组。

    通过广播把每个滤波器的上升沿 / 下降沿同时算出来，避免逐个滤波器的 Python 循环。
    """
    left = hz_points[:-2][:, None]
    center = hz_points[1:-1][:, None]
    right = hz_points[2:][:, None]
    freqs = fft_freqs[None, :]
    rising = (freqs - left) / np.maximum(center - left, 1e-12)
    falling = (right - freqs) / np.maximum(right - center, 1e-12)
    return np.clip(np.minimum(rising, falling), 0.0, None)


def mel_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> FloatArray:
    """构造梅尔刻度上等间隔的三角滤波器组。

    第 ``m`` 个滤波器在 ``mel[m-1]``、``mel[m]``、``mel[m+1]`` 三个梅尔频点上
    分别取 0、1、0，因此中心点在梅尔轴上均匀分布，落到 Hz 轴上则越往高频越稀疏，
    这与人耳的频率分辨率一致。
    """
    check_positive("n_filters", n_filters)
    check_positive("n_fft", n_fft)
    top = fmax if fmax is not None else sample_rate / 2.0
    if not 0.0 <= fmin < top <= sample_rate / 2.0:
        raise InvalidParameterError("需要满足 0 <= fmin < fmax <= sr/2")

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(top), n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    return _triangular_bank(hz_points, fft_freqs)


def linear_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> FloatArray:
    """与 :func:`mel_filterbank` 结构相同，但中心点在 Hz 轴上等间隔。

    在做 LFCC（线性频率倒谱系数）或需要均匀频率分辨率的分析时使用。
    """
    check_positive("n_filters", n_filters)
    check_positive("n_fft", n_fft)
    top = fmax if fmax is not None else sample_rate / 2.0
    if not 0.0 <= fmin < top <= sample_rate / 2.0:
        raise InvalidParameterError("需要满足 0 <= fmin < fmax <= sr/2")

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    hz_points = np.linspace(fmin, top, n_filters + 2)
    return _triangular_bank(hz_points, fft_freqs)
