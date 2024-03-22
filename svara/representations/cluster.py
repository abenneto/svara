"""K-means 聚类（Lloyd 迭代 + k-means++ 初始化）。

固定 ``random_state`` 后结果完全可复现。它既能单独用来做特征聚类，也被
:mod:`svara.representations.units` 用来学习离散声学单元的码本。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from svara.exceptions import InvalidParameterError, NotFittedError
from svara.utils import FloatArray


class KMeans:
    """最小实现的 K-means。

    属性
    ----
    cluster_centers_:
        形状 ``(n_clusters, n_features)`` 的码本。
    inertia_:
        样本到各自簇心的平方距离之和。
    n_iter_:
        实际迭代次数。
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        if n_clusters < 1:
            raise InvalidParameterError("n_clusters 必须 >= 1")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: FloatArray | None = None
        self.inertia_: float = float("nan")
        self.n_iter_: int = 0

    def _distances(self, x: FloatArray, centers: FloatArray) -> FloatArray:
        xx = (x**2).sum(axis=1)[:, None]
        cc = (centers**2).sum(axis=1)[None, :]
        dist: FloatArray = xx + cc - 2.0 * x @ centers.T
        return np.maximum(dist, 0.0)

    def _kmeans_plus_plus(self, x: FloatArray, rng: np.random.Generator) -> FloatArray:
        n_samples = x.shape[0]
        centers = np.empty((self.n_clusters, x.shape[1]), dtype=np.float64)
        centers[0] = x[rng.integers(n_samples)]
        closest = ((x - centers[0]) ** 2).sum(axis=1)
        for c in range(1, self.n_clusters):
            total = closest.sum()
            if total <= 0.0:
                centers[c] = x[rng.integers(n_samples)]
            else:
                idx = rng.choice(n_samples, p=closest / total)
                centers[c] = x[idx]
            closest = np.minimum(closest, ((x - centers[c]) ** 2).sum(axis=1))
        return centers

    def fit(self, x: FloatArray) -> KMeans:
        if x.ndim != 2:
            raise InvalidParameterError("KMeans 期望二维输入")
        if x.shape[0] < self.n_clusters:
            raise InvalidParameterError("样本数少于簇数")

        rng = np.random.default_rng(self.random_state)
        centers = self._kmeans_plus_plus(x, rng)

        labels = np.zeros(x.shape[0], dtype=np.int64)
        for iteration in range(1, self.max_iter + 1):
            distances = self._distances(x, centers)
            labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            for c in range(self.n_clusters):
                members = x[labels == c]
                if members.size == 0:
                    # 空簇：用离当前簇心最远的点重启
                    new_centers[c] = x[np.argmax(distances.min(axis=1))]
                else:
                    new_centers[c] = members.mean(axis=0)
            shift = float(np.sqrt(((new_centers - centers) ** 2).sum(axis=1).max()))
            centers = new_centers
            self.n_iter_ = iteration
            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        final = self._distances(x, centers)
        self.inertia_ = float(final[np.arange(x.shape[0]), labels].sum())
        return self

    def predict(self, x: FloatArray) -> NDArray[np.int64]:
        if self.cluster_centers_ is None:
            raise NotFittedError("请先调用 fit")
        labels: NDArray[np.int64] = np.argmin(self._distances(x, self.cluster_centers_), axis=1)
        return labels

    def fit_predict(self, x: FloatArray) -> NDArray[np.int64]:
        return self.fit(x).predict(x)
