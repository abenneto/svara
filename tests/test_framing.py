"""framing 模块的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from svara.exceptions import InvalidParameterError
from svara.framing import apply_window, frame_signal, preemphasis


def test_preemphasis_first_sample_unchanged() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = preemphasis(x, coef=0.97)
    assert y[0] == pytest.approx(1.0)
    assert y[1] == pytest.approx(2.0 - 0.97 * 1.0)


def test_preemphasis_zero_coef_is_identity() -> None:
    x = np.linspace(-1.0, 1.0, 20)
    np.testing.assert_allclose(preemphasis(x, coef=0.0), x)


def test_preemphasis_empty() -> None:
    assert preemphasis(np.array([])).size == 0


def test_frame_signal_shapes() -> None:
    x = np.arange(100, dtype=np.float64)
    frames = frame_signal(x, frame_length=20, hop_length=10, pad=False)
    assert frames.shape == (9, 20)
    # 第二帧应当从第 hop 个样点开始
    np.testing.assert_allclose(frames[1], np.arange(10, 30))


def test_frame_signal_padding_covers_tail() -> None:
    x = np.arange(25, dtype=np.float64)
    frames = frame_signal(x, frame_length=10, hop_length=10, pad=True)
    assert frames.shape[0] == 3
    assert frames.shape[1] == 10


def test_frame_signal_rejects_bad_params() -> None:
    with pytest.raises(InvalidParameterError):
        frame_signal(np.arange(10.0), frame_length=0, hop_length=5)


def test_apply_window_scales_each_frame() -> None:
    frames = np.ones((4, 16))
    out = apply_window(frames, window="hann")
    assert out.shape == frames.shape
    # Hann 窗端点为 0
    assert out[0, 0] == pytest.approx(0.0, abs=1e-12)
    # 每帧使用同一个窗
    np.testing.assert_allclose(out[0], out[1])
