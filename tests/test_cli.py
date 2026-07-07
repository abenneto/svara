"""cli 模块的测试：在临时 WAV 上跑各子命令。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from svara.cli import main
from svara.io import write_wav


def _make_wav(tmp_path: Path, sine: Callable[..., np.ndarray], sr: int) -> Path:
    path = tmp_path / "in.wav"
    write_wav(path, sine(freq=200.0, sr=sr, duration=0.5) * 0.5, sr)
    return path


def test_version(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.count(".") >= 2


def test_extract_writes_npy(tmp_path: Path, sine: Callable[..., np.ndarray], sr: int) -> None:
    wav = _make_wav(tmp_path, sine, sr)
    out = tmp_path / "mfcc.npy"
    code = main(["extract", str(wav), "-o", str(out), "--feature", "mfcc"])
    assert code == 0
    feats = np.load(out)
    assert feats.ndim == 2 and feats.shape[1] == 13


def test_f0_writes_csv(tmp_path: Path, sine: Callable[..., np.ndarray], sr: int) -> None:
    wav = _make_wav(tmp_path, sine, sr)
    out = tmp_path / "f0.csv"
    code = main(["f0", str(wav), "-o", str(out), "--method", "yin", "--format", "csv"])
    assert code == 0
    values = np.loadtxt(out, delimiter=",")
    assert values.size > 0
    assert np.median(values[values > 0]) > 150.0


def test_no_command_prints_help(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()
