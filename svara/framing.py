"""分帧、加窗与预加重——所有短时分析的公共前处理。"""

from __future__ import annotations

import numpy as np

from svara.utils import FloatArray, as_float_array


def preemphasis(signal: FloatArray, coef: float = 0.97) -> FloatArray:
    """一阶预加重滤波：``y[n] = x[n] - coef * x[n-1]``。

    它抬高高频、压低低频，能部分抵消语音信号 -6 dB/oct 的谱倾斜，
    是 MFCC / LPC 等流程的常见第一步。``coef=0`` 时相当于不处理。
    """
    x = as_float_array(signal)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]
    return y
