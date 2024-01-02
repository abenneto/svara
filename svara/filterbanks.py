"""梅尔 / 线性三角滤波器组，以及相关的频率刻度转换。

约定：所有滤波器组返回形状为 ``(n_filters, n_fft // 2 + 1)`` 的矩阵，
可直接与功率谱做 ``power_spectrum @ fb.T`` 得到各滤波器的能量。
"""

from __future__ import annotations

import numpy as np

from svara.utils import FloatArray


def hz_to_mel(hz: FloatArray | float) -> FloatArray:
    """Hz -> 梅尔（HTK 公式 ``2595 * log10(1 + f / 700)``）。"""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: FloatArray | float) -> FloatArray:
    """梅尔 -> Hz（:func:`hz_to_mel` 的逆变换）。"""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)
