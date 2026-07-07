"""分帧、加窗与预加重——所有短时分析的公共前处理。"""

from __future__ import annotations

import numpy as np

from svara.exceptions import InvalidParameterError
from svara.utils import FloatArray, as_float_array, check_positive


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


def frame_signal(
    signal: FloatArray,
    frame_length: int,
    hop_length: int,
    *,
    pad: bool = True,
) -> FloatArray:
    """把一维信号切成 ``(n_frames, frame_length)`` 的二维数组。

    使用 :func:`numpy.lib.stride_tricks.sliding_window_view` 做零拷贝滑窗，
    再按 ``hop_length`` 抽取，因此内存开销与帧数无关。当 ``pad=True`` 时，
    信号末尾会补零以容纳最后一个不完整帧。
    """
    check_positive("frame_length", frame_length)
    check_positive("hop_length", hop_length)
    x = as_float_array(signal)

    if x.size < frame_length:
        if not pad:
            raise InvalidParameterError("信号比一帧还短，且 pad=False")
        x = np.pad(x, (0, frame_length - x.size))
    elif pad:
        remainder = (x.size - frame_length) % hop_length
        if remainder != 0:
            x = np.pad(x, (0, hop_length - remainder))

    windows = np.lib.stride_tricks.sliding_window_view(x, frame_length)
    frames = windows[::hop_length]
    return np.ascontiguousarray(frames)
