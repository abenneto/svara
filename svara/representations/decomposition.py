"""主成分分析（PCA）与白化。

用 SVD 求主轴，可把高维帧级特征投影到低维、去相关的子空间，
是可视化与降维的常用第一步。接口刻意模仿 scikit-learn 的 ``fit`` / ``transform``。
"""

from __future__ import annotations

import numpy as np

from svara.exceptions import InvalidParameterError, NotFittedError
from svara.utils import FloatArray


class PCA:
    """基于 SVD 的主成分分析。

    参数
    ----
    n_components:
        保留的主成分个数。
    whiten:
        为真时，把各主成分缩放到单位方差（白化），使输出各维不相关且等尺度。
    """

    def __init__(self, n_components: int, whiten: bool = False) -> None:
        if n_components < 1:
            raise InvalidParameterError("n_components 必须 >= 1")
        self.n_components = n_components
        self.whiten = whiten
        self.mean_: FloatArray | None = None
        self.components_: FloatArray | None = None
        self.explained_variance_: FloatArray | None = None

    def fit(self, x: FloatArray) -> PCA:
        """在 ``(n_samples, n_features)`` 上估计主轴。"""
        if x.ndim != 2:
            raise InvalidParameterError("PCA 期望二维输入")
        n_samples, n_features = x.shape
        if self.n_components > min(n_samples, n_features):
            raise InvalidParameterError("n_components 超过数据的秩")

        self.mean_ = x.mean(axis=0, keepdims=True)
        centered = x - self.mean_
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.n_components]
        variance = (singular**2) / max(n_samples - 1, 1)
        self.explained_variance_ = variance[: self.n_components]
        return self

    def transform(self, x: FloatArray) -> FloatArray:
        """把新数据投影到已学到的主成分上。"""
        if self.components_ is None or self.mean_ is None:
            raise NotFittedError("请先调用 fit")
        projected = (x - self.mean_) @ self.components_.T
        if self.whiten:
            assert self.explained_variance_ is not None
            projected = projected / np.sqrt(self.explained_variance_ + 1e-12)
        return projected

    def fit_transform(self, x: FloatArray) -> FloatArray:
        return self.fit(x).transform(x)
