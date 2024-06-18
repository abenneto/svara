"""提取 MFCC 及其一阶 / 二阶差分的最小示例。

用法：

    python extract_mfcc.py [某段.wav]

不传参数时会合成一个带谐波的元音作为演示，因此可以离线直接运行。
"""

from __future__ import annotations

import sys

import numpy as np

from svara import mfcc, read_wav
from svara.features.deltas import delta, delta_delta


def demo_signal(sr: int = 16000, duration: float = 1.0, f0: float = 130.0) -> np.ndarray:
    """合成一个多谐波信号，近似浊音段。"""
    t = np.arange(int(sr * duration)) / sr
    sig = sum(np.sin(2.0 * np.pi * k * f0 * t) / k for k in range(1, 12))
    return (sig / np.max(np.abs(sig))).astype(np.float64)


def main() -> None:
    if len(sys.argv) > 1:
        signal, sr = read_wav(sys.argv[1])
    else:
        sr = 16000
        signal = demo_signal(sr)

    coeffs = mfcc(signal, sr, n_mfcc=13, hop_length=160)
    d1 = delta(coeffs)
    d2 = delta_delta(coeffs)

    print(f"采样率     : {sr} Hz")
    print(f"MFCC       : {coeffs.shape}  (帧数 x 系数)")
    print(f"一阶差分   : {d1.shape}")
    print(f"二阶差分   : {d2.shape}")
    print(f"首帧 MFCC  : {np.round(coeffs[0], 3)}")


if __name__ == "__main__":
    main()
