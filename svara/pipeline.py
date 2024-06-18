"""端到端特征提取流水线。

把分帧、频谱、倒谱、F0、共振峰等步骤按一份 :class:`FeatureConfig` 串起来，
一次调用得到一组常用特征。适合快速搭建实验基线，也是 CLI 的底层实现。
"""

from __future__ import annotations

import numpy as np

from svara.config import FeatureConfig
from svara.features.cepstral import mfcc
from svara.features.deltas import delta, delta_delta
from svara.features.melspec import log_mel_spectrogram
from svara.pitch import estimate_f0
from svara.utils import FloatArray, as_float_array


def extract_features(
    signal: FloatArray,
    config: FeatureConfig | None = None,
    with_deltas: bool = True,
    f0_method: str = "yin",
) -> dict[str, FloatArray]:
    """按 ``config`` 提取一组常用特征，返回名称到数组的字典。

    键包括 ``log_mel``、``mfcc``、``f0``；当 ``with_deltas=True`` 时额外给出
    ``mfcc_delta`` 与 ``mfcc_delta2``。各特征共用同一套分帧参数。
    """
    cfg = config if config is not None else FeatureConfig()
    x = as_float_array(signal)

    log_mel = log_mel_spectrogram(
        x,
        cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        window=cfg.window,
    )
    cepstra = mfcc(
        x,
        cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    f0 = estimate_f0(x, cfg.sample_rate, method=f0_method, hop_length=cfg.hop_length)

    out: dict[str, FloatArray] = {
        "log_mel": log_mel,
        "mfcc": cepstra,
        "f0": np.asarray(f0, dtype=np.float64),
    }
    if with_deltas:
        out["mfcc_delta"] = delta(cepstra)
        out["mfcc_delta2"] = delta_delta(cepstra)
    return out
