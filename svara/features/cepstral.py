"""倒谱特征：MFCC、LFCC 及其辅助的 DCT / 倒谱提升。"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from svara.features.melspec import log_mel_spectrogram
from svara.filterbanks import linear_filterbank
from svara.spectral import spectrogram
from svara.utils import FloatArray


def dct_ii(log_energies: FloatArray, n_ceps: int) -> FloatArray:
    """对每一帧的对数能量做正交归一化的 DCT-II，并截取前 ``n_ceps`` 个系数。

    DCT 把相邻滤波器高度相关的对数能量去相关，低阶系数刻画谱包络，
    高阶系数刻画细节。
    """
    coeffs: FloatArray = dct(log_energies, type=2, axis=-1, norm="ortho")
    return coeffs[:, :n_ceps]


def sinusoidal_lifter(cepstra: FloatArray, lift: int = 22) -> FloatArray:
    """正弦倒谱提升，抬高高阶倒谱系数的幅度。

    ``lift <= 0`` 时不做处理，直接返回原数组。
    """
    if lift <= 0:
        return cepstra
    n = np.arange(cepstra.shape[1])
    weights = 1.0 + (lift / 2.0) * np.sin(np.pi * n / lift)
    return cepstra * weights


def mfcc(
    signal: FloatArray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int | None = None,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
    lift: int = 22,
) -> FloatArray:
    """梅尔频率倒谱系数（MFCC）。

    流程：对数梅尔谱 -> DCT-II -> 取前 ``n_mfcc`` 阶 -> 正弦倒谱提升。
    返回 ``(n_frames, n_mfcc)``。
    """
    log_mel = log_mel_spectrogram(
        signal,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    cepstra = dct_ii(log_mel, n_mfcc)
    return sinusoidal_lifter(cepstra, lift=lift)


def lfcc(
    signal: FloatArray,
    sample_rate: int,
    n_lfcc: int = 13,
    n_fft: int = 512,
    hop_length: int | None = None,
    n_filters: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
    lift: int = 22,
    log_offset: float = 1e-6,
) -> FloatArray:
    """线性频率倒谱系数（LFCC）。

    与 MFCC 唯一的区别是使用 Hz 轴等间隔的线性滤波器组，因此对高频细节
    更敏感，常用于反欺骗 / 声纹等任务。
    """
    psd = spectrogram(signal, n_fft=n_fft, hop_length=hop_length, power=2.0)
    fb = linear_filterbank(n_filters, n_fft, sample_rate, fmin=fmin, fmax=fmax)
    log_energy = np.log(psd @ fb.T + log_offset)
    cepstra = dct_ii(log_energy, n_lfcc)
    return sinusoidal_lifter(cepstra, lift=lift)
