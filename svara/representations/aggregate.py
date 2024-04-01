"""句级统计聚合。

把变长的帧级特征序列沿时间轴压成一个定长向量——每个统计量（均值、标准差、
偏度……）作用在每一维特征上，再首尾拼接。这是最经典、也最稳的“定长表示”，
在情感识别、说话人画像等任务里常作为强基线。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

from svara.exceptions import InvalidParameterError
from svara.utils import FloatArray

DEFAULT_STATISTICS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "skew",
    "kurtosis",
)

_FUNCS: dict[str, Callable[[FloatArray], np.ndarray]] = {
    "mean": lambda x: np.mean(x, axis=0),
    "std": lambda x: np.std(x, axis=0),
    "min": lambda x: np.min(x, axis=0),
    "max": lambda x: np.max(x, axis=0),
    "median": lambda x: np.median(x, axis=0),
    "skew": lambda x: stats.skew(x, axis=0, bias=False),
    "kurtosis": lambda x: stats.kurtosis(x, axis=0, bias=False),
}


def aggregate_statistics(
    features: FloatArray, statistics: Sequence[str] = DEFAULT_STATISTICS
) -> FloatArray:
    """把 ``(n_frames, n_features)`` 聚合成 ``(len(statistics) * n_features,)``。

    统计量的拼接顺序与 ``statistics`` 一致，便于下游解读每一段的含义。
    """
    if features.ndim != 2:
        raise InvalidParameterError("aggregate_statistics 期望二维输入")
    parts = []
    for name in statistics:
        try:
            func = _FUNCS[name]
        except KeyError:
            valid = ", ".join(sorted(_FUNCS))
            raise InvalidParameterError(f"未知统计量 {name!r}，可选：{valid}") from None
        parts.append(np.asarray(func(features), dtype=np.float64).ravel())
    return np.concatenate(parts)
