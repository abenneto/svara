"""离散声学单元编码。

思路借鉴自自监督语音表示里的“离散单元”做法：先在大量帧级特征（如 MFCC）上
用 K-means 学一个码本，再把每帧量化到最近的码字，得到一串离散的伪音素标签。
整个过程无监督、离线、可复现，可作为下游声学建模的紧凑输入。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from svara.exceptions import NotFittedError
from svara.representations.cluster import KMeans
from svara.utils import FloatArray


class AcousticUnitizer:
    """把连续帧级特征量化为离散声学单元。

    参数
    ----
    n_units:
        码本大小（离散单元数）。
    random_state:
        传给底层 K-means，保证可复现。
    """

    def __init__(self, n_units: int = 64, random_state: int = 0, max_iter: int = 100) -> None:
        self.n_units = n_units
        self._kmeans = KMeans(n_units, random_state=random_state, max_iter=max_iter)

    def fit(self, features: FloatArray) -> AcousticUnitizer:
        """在帧级特征上学习码本。"""
        self._kmeans.fit(features)
        return self

    def encode(self, features: FloatArray) -> NDArray[np.int64]:
        """把每帧映射到最近码字的下标，返回单元序列。"""
        return self._kmeans.predict(features)

    def fit_encode(self, features: FloatArray) -> NDArray[np.int64]:
        return self.fit(features).encode(features)

    @property
    def codebook(self) -> FloatArray:
        """学到的码字（簇心）。"""
        if self._kmeans.cluster_centers_ is None:
            raise NotFittedError("请先调用 fit")
        return self._kmeans.cluster_centers_

    def unit_histogram(self, features: FloatArray, normalize: bool = True) -> FloatArray:
        """把一段语音的单元序列汇总成定长的“单元词袋”向量。

        这是一种简单的句级表示：忽略时序，只统计每个单元出现的频次。
        """
        units = self.encode(features)
        counts = np.bincount(units, minlength=self.n_units).astype(np.float64)
        if normalize and counts.sum() > 0:
            counts /= counts.sum()
        return counts
