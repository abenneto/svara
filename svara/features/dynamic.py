"""差分（动态）特征：一阶 delta 与二阶 delta-delta。

差分刻画特征随时间的变化速率，把静态谱特征扩展成能反映动态演化的表示，
在语音识别里几乎是标配。
"""

from __future__ import annotations

import numpy as np

from svara.exceptions import InvalidParameterError
from svara.utils import FloatArray


def delta(features: FloatArray, width: int = 2) -> FloatArray:
    """用对称回归窗计算一阶差分。

    对每个时刻 ``t``，用 ``[-width, width]`` 范围内的加权差分估计斜率：

    .. math:: d_t = \\frac{\\sum_{n=1}^{W} n (c_{t+n} - c_{t-n})}{2 \\sum_{n=1}^{W} n^2}

    边界处对特征序列做边缘复制补齐，输出形状与输入一致。
    """
    if width < 1:
        raise InvalidParameterError("width 必须 >= 1")
    if features.ndim != 2:
        raise InvalidParameterError("delta 期望 (n_frames, n_features) 的二维输入")

    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, width + 1))
    out = np.zeros_like(features)
    n_frames = features.shape[0]
    for n in range(1, width + 1):
        ahead = padded[width + n : width + n + n_frames]
        behind = padded[width - n : width - n + n_frames]
        out += n * (ahead - behind)
    return out / denom


def delta_delta(features: FloatArray, width: int = 2) -> FloatArray:
    """二阶差分：对一阶差分再求一次 :func:`delta`。"""
    return delta(delta(features, width=width), width=width)
