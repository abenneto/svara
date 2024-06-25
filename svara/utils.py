"""通用的数组与数值辅助函数。

这里只放与具体语音算法无关的小工具，保持其它模块的可读性。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from svara.exceptions import InvalidParameterError

#: 库内部统一使用的浮点数组类型别名。
FloatArray = NDArray[np.float64]

#: 避免 log(0) / 除零时使用的极小值。
EPS: float = float(np.finfo(np.float64).eps)


def as_float_array(x: ArrayLike) -> FloatArray:
    """把输入转换成连续的一维 ``float64`` 数组。

    多声道输入（二维）会触发异常——本库只处理单声道信号，
    调用方应自行决定如何混音。
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise InvalidParameterError(f"期望一维信号，得到的是 {arr.ndim} 维数组")
    return np.ascontiguousarray(arr)


def check_positive(name: str, value: int | float) -> None:
    """校验 ``value`` 为正数，否则抛出 :class:`InvalidParameterError`。"""
    if value <= 0:
        raise InvalidParameterError(f"{name} 必须为正数，得到 {value!r}")


def next_power_of_two(n: int) -> int:
    """返回不小于 ``n`` 的最小 2 的幂。"""
    check_positive("n", n)
    return 1 if n == 1 else int(2 ** np.ceil(np.log2(n)))
