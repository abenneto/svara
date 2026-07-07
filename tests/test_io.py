"""io 模块的测试：读写往返与重采样。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from svara.io import read_wav, resample, write_wav


def test_wav_roundtrip(sine: Callable[..., np.ndarray], sr: int, tmp_path: Path) -> None:
    original = sine(freq=440.0, sr=sr, duration=0.25) * 0.5
    path = tmp_path / "tone.wav"
    write_wav(path, original, sr)

    loaded, loaded_sr = read_wav(path)
    assert loaded_sr == sr
    assert loaded.shape == original.shape
    # 16-bit 量化误差应在 1e-3 量级
    np.testing.assert_allclose(loaded, original, atol=2e-3)


def test_resample_changes_length(sine: Callable[..., np.ndarray]) -> None:
    x = sine(freq=200.0, sr=16000, duration=1.0)
    y = resample(x, 16000, 8000)
    assert abs(y.size - x.size // 2) <= 2


def test_resample_identity_returns_same(sine: Callable[..., np.ndarray]) -> None:
    x = sine(duration=0.1)
    np.testing.assert_array_equal(resample(x, 16000, 16000), x)


def test_write_clips_out_of_range(tmp_path: Path) -> None:
    loud = np.array([2.0, -2.0, 0.0, 1.5], dtype=np.float64)
    path = tmp_path / "clip.wav"
    write_wav(path, loud, 16000)
    loaded, _ = read_wav(path)
    assert np.all(np.abs(loaded) <= 1.0)
