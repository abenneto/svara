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


def spectral_rolloff(
    spectrum: FloatArray, freqs: FloatArray, roll_percent: float = 0.85
) -> FloatArray:
    """谱滚降点：累计能量首次达到总能量 ``roll_percent`` 的频率。"""
    if not 0.0 < roll_percent < 1.0:
        raise ValueError("roll_percent 需在 (0, 1) 之间")
    cumulative = np.cumsum(spectrum, axis=1)
    thresholds = roll_percent * cumulative[:, -1:]
    # 每帧第一个累计能量达到阈值的 bin
    reached = cumulative >= thresholds
    idx = np.argmax(reached, axis=1)
    return freqs[idx]


def spectral_flatness(spectrum: FloatArray) -> FloatArray:
    """谱平坦度：几何均值与算术均值之比。

    接近 1 表示类噪声（能量均匀铺开），接近 0 表示类音调（能量集中在少数谐波）。
    """
    power = spectrum + EPS
    geometric = np.exp(np.mean(np.log(power), axis=1))
    arithmetic = np.mean(power, axis=1)
    return geometric / arithmetic
