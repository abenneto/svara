"""基频（F0）估计。

提供三种互补的时域 / 倒谱域估计器：自相关法、YIN、以及倒谱法，
并由 :func:`estimate_f0` 统一分派。所有函数都以帧为单位工作，返回长度为
帧数的一维数组，清音帧记为 ``0.0``。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from svara.exceptions import InvalidParameterError
from svara.framing import apply_window, frame_signal
from svara.utils import EPS, FloatArray, as_float_array, next_power_of_two

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


def _parabolic_refine(values: FloatArray, tau: int) -> float:
    """用抛物线插值在整数极小点附近求亚样本精度的最小位置。"""
    if tau <= 0 or tau >= values.size - 1:
        return float(tau)
    a, b, c = values[tau - 1], values[tau], values[tau + 1]
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return float(tau)
    return float(tau) + 0.5 * (a - c) / denom


def _yin_frame(cmndf: FloatArray, min_lag: int, threshold: float) -> float:
    """对单帧的 CMNDF 做绝对阈值搜索，返回基音周期（样点，可能是小数），清音返回 0。"""
    tau = min_lag
    n = cmndf.size
    while tau < n:
        if cmndf[tau] < threshold:
            while tau + 1 < n and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            return _parabolic_refine(cmndf, tau)
        tau += 1
    return 0.0


def f0_yin(
    signal: FloatArray,
    sample_rate: int,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    frame_length: int = 1024,
    hop_length: int = 256,
    threshold: float = 0.1,
) -> FloatArray:
    """YIN 算法估计 F0（de Cheveigné & Kawahara, 2002）。

    相比裸自相关，CMNDF + 绝对阈值能显著减少倍频 / 半频错误，
    抛物线插值进一步提供亚样本精度。找不到低于阈值的滞后即判为清音。
    """
    max_lag = int(np.ceil(sample_rate / fmin))
    min_lag = max(1, int(np.floor(sample_rate / fmax)))
    if max_lag >= frame_length:
        raise InvalidParameterError("frame_length 太短，无法覆盖 fmin 对应的周期")

    frames = frame_signal(as_float_array(signal), frame_length, hop_length)
    f0 = np.zeros(frames.shape[0], dtype=np.float64)
    for i, frame in enumerate(frames):
        diff = _difference_function(frame, max_lag + 1)
        cmndf = _cumulative_mean_normalized_difference(diff)
        period = _yin_frame(cmndf, min_lag, threshold)
        if period > 0.0:
            f0[i] = sample_rate / period
    return f0


def f0_cepstrum(
    signal: FloatArray,
    sample_rate: int,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    frame_length: int = 1024,
    hop_length: int = 256,
    window: str = "hann",
    prominence: float = 3.0,
) -> FloatArray:
    """倒谱法估计 F0。

    实倒谱 ``IFFT(log|FFT(x)|)`` 会把周期性激励的谐波结构折叠成基音周期处的
    一个尖峰。峰值相对于搜索区间内幅度中位数的“突出度”低于 ``prominence`` 时
    判为清音。
    """
    max_q = int(np.ceil(sample_rate / fmin))
    min_q = max(1, int(np.floor(sample_rate / fmax)))
    if max_q >= frame_length:
        raise InvalidParameterError("frame_length 太短，无法覆盖 fmin 对应的周期")

    frames = frame_signal(as_float_array(signal), frame_length, hop_length)
    windowed = apply_window(frames, window=window)
    spec = np.fft.rfft(windowed, n=frame_length, axis=1)
    log_mag = np.log(np.abs(spec) + EPS)
    cepstrum = np.fft.irfft(log_mag, n=frame_length, axis=1)

    segment = cepstrum[:, min_q : max_q + 1]
    best = np.argmax(segment, axis=1) + min_q
    peak = cepstrum[np.arange(cepstrum.shape[0]), best]
    baseline = np.median(np.abs(segment), axis=1) + EPS

    voiced = peak > prominence * baseline
    f0 = np.where(best > 0, sample_rate / best, 0.0)
    return np.where(voiced, f0, 0.0)


#: 方法名 -> 估计函数的映射，供 :func:`estimate_f0` 分派。
_METHODS: dict[str, Callable[..., FloatArray]] = {
    "autocorrelation": f0_autocorrelation,
    "yin": f0_yin,
    "cepstrum": f0_cepstrum,
}


def estimate_f0(
    signal: FloatArray,
    sample_rate: int,
    method: str = "yin",
    **kwargs: object,
) -> FloatArray:
    """按名称选择估计器的统一入口。

    ``method`` 可取 ``"autocorrelation"``、``"yin"``、``"cepstrum"``，其余关键字
    参数原样透传给对应实现。
    """
    try:
        fn = _METHODS[method]
    except KeyError:
        valid = ", ".join(sorted(_METHODS))
        raise InvalidParameterError(f"未知的 F0 估计方法 {method!r}，可选：{valid}") from None
    return fn(signal, sample_rate, **kwargs)


def voiced_flags(f0: FloatArray) -> NDArray[np.bool_]:
    """把 F0 序列转成布尔的清浊音标记（``f0 > 0`` 记为浊音）。"""
    return f0 > 0.0
