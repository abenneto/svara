"""时域特征：短时能量与过零率。

这两个特征直接在分帧后的时域信号上计算，常用于粗粒度的清浊音 / 静音判别。
"""

from __future__ import annotations

import numpy as np

from svara.framing import frame_signal
from svara.utils import FloatArray, as_float_array


def zero_crossing_rate(
    signal: FloatArray, frame_length: int = 400, hop_length: int = 160
) -> FloatArray:
    """每帧的过零率：相邻样点符号变化的比例。

    清音（摩擦音等）过零率高，浊音（元音）过零率低。
    """
    frames = frame_signal(as_float_array(signal), frame_length, hop_length)
    signs = np.signbit(frames)
    crossings = np.abs(np.diff(signs.astype(np.int8), axis=1)).sum(axis=1)
    return crossings / float(frame_length)


def rms_energy(
    signal: FloatArray, frame_length: int = 400, hop_length: int = 160
) -> FloatArray:
    """每帧的均方根能量。"""
    frames = frame_signal(as_float_array(signal), frame_length, hop_length)
    return np.sqrt(np.mean(frames**2, axis=1))
