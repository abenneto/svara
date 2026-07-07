"""基频（F0）估计。

提供三种互补的时域 / 倒谱域估计器：自相关法、YIN、以及倒谱法，
并由 :func:`estimate_f0` 统一分派。所有函数都以帧为单位工作，返回长度为
帧数的一维数组，清音帧记为 ``0.0``。
"""

from __future__ import annotations

import numpy as np

from svara.exceptions import InvalidParameterError
from svara.framing import frame_signal
from svara.utils import FloatArray, as_float_array, next_power_of_two

# 语音默认的搜索范围（Hz）。
DEFAULT_FMIN = 50.0
DEFAULT_FMAX = 500.0


def _autocorrelate(frames: FloatArray) -> FloatArray:
    """逐帧的自相关（经 Wiener–Khinchin 用 FFT 加速），只保留非负滞后。"""
    n = frames.shape[1]
    nfft = next_power_of_two(2 * n)
    spec = np.fft.rfft(frames, n=nfft, axis=1)
    power = spec * np.conj(spec)
    ac = np.fft.irfft(power, n=nfft, axis=1)[:, :n]
    return ac.astype(np.float64)


def f0_autocorrelation(
    signal: FloatArray,
    sample_rate: int,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    frame_length: int = 1024,
    hop_length: int = 256,
    voicing_threshold: float = 0.3,
) -> FloatArray:
    """自相关法估计 F0。

    在滞后区间 ``[sr/fmax, sr/fmin]`` 内寻找自相关峰值，峰值滞后即基音周期。
    峰值与零滞后能量之比低于 ``voicing_threshold`` 时判为清音。
    """
    max_lag = int(np.ceil(sample_rate / fmin))
    min_lag = max(1, int(np.floor(sample_rate / fmax)))
    if max_lag >= frame_length:
        raise InvalidParameterError("frame_length 太短，无法覆盖 fmin 对应的周期")

    frames = frame_signal(as_float_array(signal), frame_length, hop_length)
    frames = frames - frames.mean(axis=1, keepdims=True)
    ac = _autocorrelate(frames)

    r0 = ac[:, 0]
    segment = ac[:, min_lag : max_lag + 1]
    best = np.argmax(segment, axis=1) + min_lag
    peak = ac[np.arange(ac.shape[0]), best]

    with np.errstate(divide="ignore", invalid="ignore"):
        f0 = np.where(best > 0, sample_rate / best, 0.0)
    voiced = (r0 > 0) & (peak > voicing_threshold * r0)
    return np.where(voiced, f0, 0.0)


def _difference_function(frame: FloatArray, tau_max: int) -> FloatArray:
    """YIN 的差分函数 ``d(tau) = Σ_j (x[j] - x[j+tau])^2``。

    借助平方和的累加与自相关，在 ``O(N log N)`` 内算出所有滞后，避免二重循环。
    """
    w = frame.size
    tau_max = min(tau_max, w)
    cumsum = np.concatenate(([0.0], np.cumsum(frame * frame)))
    nfft = next_power_of_two(2 * w)
    spec = np.fft.rfft(frame, n=nfft)
    acf = np.fft.irfft(spec * np.conj(spec), n=nfft)[:tau_max]
    tau = np.arange(tau_max)
    return cumsum[w - tau] + (cumsum[w] - cumsum[tau]) - 2.0 * acf


def _cumulative_mean_normalized_difference(diff: FloatArray) -> FloatArray:
    """把差分函数归一化为 CMNDF，使其在小滞后处不再总是偏小。"""
    out = np.empty_like(diff)
    out[0] = 1.0
    tau = np.arange(1, diff.size)
    running = np.cumsum(diff[1:])
    out[1:] = diff[1:] * tau / np.maximum(running, np.finfo(np.float64).tiny)
    return out
