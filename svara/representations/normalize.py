"""倒谱均值方差归一化（CMVN）。

CMVN 逐维减均值、除标准差，能显著削弱信道 / 录音设备差异对特征的影响，
是说话人无关任务里几乎必做的一步。
"""

from __future__ import annotations

from svara.utils import FloatArray


def cmvn(features: FloatArray, norm_vars: bool = True, eps: float = 1e-8) -> FloatArray:
    """对 ``(n_frames, n_features)`` 沿时间轴做均值（可选方差）归一化。

    ``norm_vars=False`` 时只做减均值（CMN）。``eps`` 避免除以零方差的维度。
    """
    if features.ndim != 2:
        raise ValueError("cmvn 期望 (n_frames, n_features) 的二维输入")
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean
    if not norm_vars:
        return centered
    std = features.std(axis=0, keepdims=True)
    return centered / (std + eps)
