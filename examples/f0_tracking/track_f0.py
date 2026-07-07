"""对比三种 F0 估计器的示例。

用法：

    python track_f0.py [某段.wav]

不传参数时合成一个已知基频的信号，方便直接检验各方法的估计是否一致。
"""

from __future__ import annotations

import sys

import numpy as np

from svara import read_wav
from svara.pitch import estimate_f0


def demo_signal(sr: int = 16000, duration: float = 1.0, f0: float = 160.0) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    sig = sum(np.sin(2.0 * np.pi * k * f0 * t) / k for k in range(1, 10))
    return (sig / np.max(np.abs(sig))).astype(np.float64)


def main() -> None:
    if len(sys.argv) > 1:
        signal, sr = read_wav(sys.argv[1])
    else:
        sr = 16000
        signal = demo_signal(sr)

    for method in ("autocorrelation", "yin", "cepstrum"):
        f0 = estimate_f0(signal, sr, method=method, hop_length=256)
        voiced = f0[f0 > 0]
        median = float(np.median(voiced)) if voiced.size else float("nan")
        print(f"{method:>15} : 中位 F0 = {median:6.1f} Hz，浊音占比 {np.mean(f0 > 0):.0%}")


if __name__ == "__main__":
    main()
