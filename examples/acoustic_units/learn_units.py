"""学习离散声学单元并得到句级“单元词袋”表示的示例。

用法：

    python learn_units.py [某段.wav]

流程：提取 MFCC -> CMVN 归一化 -> K-means 学码本 -> 量化成离散单元。
不传参数时用两个不同基频的合成段拼接，演示单元序列会随音段切换而变化。
"""

from __future__ import annotations

import sys

import numpy as np

from svara import mfcc, read_wav
from svara.representations.normalize import cmvn
from svara.representations.units import AcousticUnitizer


def demo_signal(sr: int = 16000) -> np.ndarray:
    def tone(f0: float, dur: float) -> np.ndarray:
        t = np.arange(int(sr * dur)) / sr
        sig = sum(np.sin(2.0 * np.pi * k * f0 * t) / k for k in range(1, 10))
        return sig

    sig = np.concatenate([tone(140.0, 0.6), tone(240.0, 0.6)])
    return (sig / np.max(np.abs(sig))).astype(np.float64)


def main() -> None:
    if len(sys.argv) > 1:
        signal, sr = read_wav(sys.argv[1])
    else:
        sr = 16000
        signal = demo_signal(sr)

    feats = cmvn(mfcc(signal, sr, n_mfcc=13, hop_length=160))
    unitizer = AcousticUnitizer(n_units=8, random_state=0).fit(feats)
    units = unitizer.encode(feats)
    histogram = unitizer.unit_histogram(feats)

    print(f"帧数        : {feats.shape[0]}")
    print(f"单元序列前 20 帧 : {units[:20].tolist()}")
    print(f"单元词袋直方图   : {np.round(histogram, 3).tolist()}")


if __name__ == "__main__":
    main()
