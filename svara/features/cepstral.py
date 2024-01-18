"""倒谱特征：MFCC、LFCC 及其辅助的 DCT / 倒谱提升。"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

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
