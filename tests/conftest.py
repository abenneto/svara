"""测试共用的合成信号夹具。

全部信号都由确定性公式生成（随机数固定种子），因此测试不依赖任何外部
音频文件，也保证可复现。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

SR = 16000


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def sine() -> Callable[..., np.ndarray]:
    """返回一个生成正弦波的工厂函数。"""

    def _make(
        freq: float = 220.0, sr: int = SR, duration: float = 1.0, amp: float = 0.8
    ) -> np.ndarray:
        t = np.arange(int(sr * duration)) / sr
        return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float64)

    return _make


@pytest.fixture
def chirp() -> Callable[..., np.ndarray]:
    """线性扫频信号，用来检查时频相关的实现。"""

    def _make(
        f0: float = 100.0, f1: float = 4000.0, sr: int = SR, duration: float = 1.0
    ) -> np.ndarray:
        n = int(sr * duration)
        t = np.arange(n) / sr
        k = (f1 - f0) / duration
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        return np.sin(phase).astype(np.float64)

    return _make


@pytest.fixture
def white_noise() -> Callable[..., np.ndarray]:
    """固定种子的白噪声。"""

    def _make(sr: int = SR, duration: float = 1.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(int(sr * duration)).astype(np.float64)

    return _make
