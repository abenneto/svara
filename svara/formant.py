"""线性预测（LPC）与共振峰分析。

用自相关法 + Levinson–Durbin 递推求 LPC 系数，再对预测多项式求根得到
声道共振峰的频率与带宽。
"""

from __future__ import annotations

import numpy as np

from svara.exceptions import InvalidParameterError
from svara.framing import frame_signal, preemphasis
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
        a[1 : i + 1] = a[1 : i + 1] + k * a[i - 1 :: -1][:i]
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


def _default_order(sample_rate: int) -> int:
    """经验规则：每 1 kHz 采样率约对应一对极点，再加 2 的余量。"""
    return 2 + sample_rate // 1000


def formants(
    frame: FloatArray,
    sample_rate: int,
    order: int | None = None,
    max_formants: int = 5,
    min_freq: float = 90.0,
    max_bandwidth: float = 400.0,
    preemph: float = 0.97,
) -> FloatArray:
    """估计单帧的共振峰频率（Hz，升序）。

    对 LPC 预测多项式求根，每个位于上半平面的复根对应一个谐振：
    其角度给出频率，模长给出带宽。过宽（衰减太快）或频率过低的根会被丢弃，
    通常对应声门或数值噪声而非真正的共振峰。
    """
    order = order if order is not None else _default_order(sample_rate)
    windowed = preemphasis(as_float_array(frame), coef=preemph)
    windowed = windowed * np.hanning(windowed.size)
    coeffs = lpc(windowed, order)

    roots = np.roots(coeffs)
    roots = roots[np.imag(roots) > 0.0]
    if roots.size == 0:
        return np.empty(0, dtype=np.float64)

    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * sample_rate / (2.0 * np.pi)
    bandwidths = -0.5 * (sample_rate / (2.0 * np.pi)) * np.log(np.abs(roots))

    keep = (freqs >= min_freq) & (freqs <= sample_rate / 2.0) & (bandwidths < max_bandwidth)
    selected = np.sort(freqs[keep])
    return selected[:max_formants]


def track_formants(
    signal: FloatArray,
    sample_rate: int,
    n_formants: int = 4,
    order: int | None = None,
    frame_length: int = 400,
    hop_length: int = 160,
    min_freq: float = 90.0,
    max_bandwidth: float = 400.0,
    preemph: float = 0.97,
) -> FloatArray:
    """逐帧跟踪前 ``n_formants`` 个共振峰，返回 ``(n_frames, n_formants)``。

    某帧检测到的共振峰不足时，缺失位置以 ``NaN`` 填充，便于下游用
    :func:`numpy.nanmean` 之类的函数统计。
    """
    check_positive("n_formants", n_formants)
    frames = frame_signal(as_float_array(signal), frame_length, hop_length)
    out = np.full((frames.shape[0], n_formants), np.nan, dtype=np.float64)
    for i, frame in enumerate(frames):
        found = formants(
            frame,
            sample_rate,
            order=order,
            max_formants=n_formants,
            min_freq=min_freq,
            max_bandwidth=max_bandwidth,
            preemph=preemph,
        )
        out[i, : found.size] = found
    return out
