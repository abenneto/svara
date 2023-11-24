"""音频读写与重采样。

底层直接使用 :func:`scipy.io.wavfile`，只支持 WAV。整数 PCM 会被归一化到
``[-1, 1)`` 的 ``float64``，多声道默认混成单声道。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from svara.exceptions import AudioIOError
from svara.utils import FloatArray, as_float_array

# 各整数 PCM 位宽对应的归一化除数。
_INT_SCALE = {
    np.dtype(np.int16): 2**15,
    np.dtype(np.int32): 2**31,
    np.dtype(np.uint8): 2**7,  # uint8 会先减去偏置
}


def read_wav(path: str | Path, mono: bool = True) -> tuple[FloatArray, int]:
    """读取 WAV 文件，返回 ``(signal, sample_rate)``。

    信号被转换为 ``float64`` 并归一化到 ``[-1, 1)``。当 ``mono=True`` 且文件为
    多声道时，取各声道的平均值。
    """
    try:
        sample_rate, data = wavfile.read(str(path))
    except (OSError, ValueError) as exc:  # pragma: no cover - 依赖具体文件
        raise AudioIOError(f"无法读取音频文件：{path}") from exc

    signal = _to_float(data)
    if signal.ndim == 2 and mono:
        signal = signal.mean(axis=1)
    return np.ascontiguousarray(signal), int(sample_rate)


def _to_float(data: np.ndarray) -> FloatArray:
    """把原始 PCM 数据转换为归一化的 ``float64``。"""
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float64, copy=False)

    dtype = data.dtype
    if dtype == np.dtype(np.uint8):
        return (data.astype(np.float64) - 128.0) / _INT_SCALE[dtype]

    scale = _INT_SCALE.get(dtype)
    if scale is None:
        raise AudioIOError(f"不支持的 PCM 位宽：{dtype}")
    return data.astype(np.float64) / scale


def write_wav(path: str | Path, signal: FloatArray, sample_rate: int) -> None:
    """把 ``[-1, 1]`` 范围的浮点信号写成 16-bit PCM 的 WAV 文件。

    超出范围的样点会被裁剪，避免整数溢出产生爆音。
    """
    sig = as_float_array(signal)
    clipped = np.clip(sig, -1.0, 1.0)
    pcm = np.round(clipped * (_INT_SCALE[np.dtype(np.int16)] - 1)).astype(np.int16)
    try:
        wavfile.write(str(path), int(sample_rate), pcm)
    except OSError as exc:  # pragma: no cover - 依赖文件系统
        raise AudioIOError(f"无法写入音频文件：{path}") from exc


def resample(signal: FloatArray, orig_sr: int, target_sr: int) -> FloatArray:
    """把信号从 ``orig_sr`` 重采样到 ``target_sr``。

    使用 :func:`scipy.signal.resample_poly` 的多相滤波实现——相比 FFT 方法，
    它对非整数倍率更稳健，也不会在端点引入振铃。倍率会先用最大公约数约分。
    """
    sig = as_float_array(signal)
    if orig_sr == target_sr:
        return sig
    gcd = int(np.gcd(int(orig_sr), int(target_sr)))
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled: FloatArray = resample_poly(sig, up, down).astype(np.float64)
    return resampled
