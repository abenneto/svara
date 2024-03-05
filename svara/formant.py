"""线性预测（LPC）与共振峰分析。

用自相关法 + Levinson–Durbin 递推求 LPC 系数，再对预测多项式求根得到
声道共振峰的频率与带宽。
"""

from __future__ import annotations

import numpy as np

from svara.exceptions import InvalidParameterError
from svara.utils import FloatArray, as_float_array, check_positive


def autocorrelate(frame: FloatArray, order: int) -> FloatArray:
    """返回滞后 ``0..order`` 的自相关序列。"""
    check_positive("order", order)
    x = as_float_array(frame)
    full = np.correlate(x, x, mode="full")
    mid = full.size // 2
    return full[mid : mid + order + 1]


def levinson(r: FloatArray, order: int) -> tuple[FloatArray, float]:
    """Levinson–Durbin 递推，求解 Yule–Walker 方程。

    输入是滞后 ``0..order`` 的自相关，返回预测多项式系数
    ``a = [1, a1, ..., a_order]`` 及最终预测误差能量。整个过程是 ``O(order^2)``。
    """
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    err = float(r[0])
    if err <= 0.0:
        # 全零 / 静音帧：没有可辨识的共振结构
        return a, 0.0

    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])
        k = -acc / err
        a[1 : i + 1] = a[1 : i + 1] + k * a[i - 1 :: -1][: i]
        err *= 1.0 - k * k
        if err <= 0.0:
            break
    return a, err


def lpc(frame: FloatArray, order: int) -> FloatArray:
    """由单帧信号估计 ``order`` 阶 LPC 系数。"""
    if order < 1:
        raise InvalidParameterError("LPC 阶数必须 >= 1")
    r = autocorrelate(frame, order)
    coeffs, _ = levinson(r, order)
    return coeffs
