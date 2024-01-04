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

    fb = np.zeros((n_filters, fft_freqs.shape[0]), dtype=np.float64)
    for m in range(1, n_filters + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        rising = (fft_freqs - left) / max(center - left, 1e-12)
        falling = (right - fft_freqs) / max(right - center, 1e-12)
        fb[m - 1] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb
