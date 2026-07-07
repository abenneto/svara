"""谱形状描述子：质心、带宽、滚降点、平坦度。

所有函数都接受形状为 ``(n_frames, n_bins)`` 的（幅度或功率）谱，以及长度
``n_bins`` 的频率轴 ``freqs``，返回每帧一个标量组成的一维数组。
"""

from __future__ import annotations

import numpy as np

from svara.utils import EPS, FloatArray


def spectral_centroid(spectrum: FloatArray, freqs: FloatArray) -> FloatArray:
    """谱质心：以幅度为权重的频率加权平均，反映“亮度”。"""
    energy = spectrum.sum(axis=1) + EPS
    return (spectrum * freqs).sum(axis=1) / energy


def spectral_bandwidth(
    spectrum: FloatArray, freqs: FloatArray, p: float = 2.0
) -> FloatArray:
    """谱带宽：各频点到质心距离的 ``p`` 阶加权矩，反映能量的频域展布。"""
    centroid = spectral_centroid(spectrum, freqs)[:, None]
    energy = spectrum.sum(axis=1) + EPS
    deviation = np.abs(freqs[None, :] - centroid) ** p
    return ((spectrum * deviation).sum(axis=1) / energy) ** (1.0 / p)
