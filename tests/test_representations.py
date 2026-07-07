"""representations 子包的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from svara.exceptions import InvalidParameterError
from svara.representations.aggregate import aggregate_statistics
from svara.representations.cluster import KMeans
from svara.representations.decomposition import PCA
from svara.representations.normalize import cmvn
from svara.representations.units import AcousticUnitizer


def test_cmvn_zero_mean_unit_var() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(200, 8))
    out = cmvn(x, norm_vars=True)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-3)


def test_cmvn_mean_only() -> None:
    x = np.random.default_rng(1).normal(size=(50, 4))
    out = cmvn(x, norm_vars=False)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-8)


def test_pca_reduces_dimension_and_decorrelates() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=(300, 2))
    mixing = np.array([[3.0, 1.0], [1.0, 2.0], [0.5, 0.5]])
    x = base @ mixing.T  # 3 维但本征秩为 2
    pca = PCA(n_components=2, whiten=True)
    y = pca.fit_transform(x)
    assert y.shape == (300, 2)
    # 白化后协方差应接近单位阵
    cov = np.cov(y, rowvar=False)
    np.testing.assert_allclose(cov, np.eye(2), atol=0.1)


def test_kmeans_recovers_separated_blobs() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(loc=[-5.0, -5.0], scale=0.3, size=(100, 2))
    b = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(100, 2))
    x = np.vstack([a, b])
    km = KMeans(n_clusters=2, random_state=0)
    labels = km.fit_predict(x)
    # 同一真簇内标签应一致
    assert len(set(labels[:100])) == 1
    assert len(set(labels[100:])) == 1
    assert labels[0] != labels[-1]


def test_kmeans_is_deterministic() -> None:
    x = np.random.default_rng(4).normal(size=(120, 5))
    a = KMeans(n_clusters=4, random_state=7).fit(x)
    b = KMeans(n_clusters=4, random_state=7).fit(x)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)


def test_acoustic_unitizer_histogram_sums_to_one() -> None:
    x = np.random.default_rng(5).normal(size=(500, 13))
    uz = AcousticUnitizer(n_units=16, random_state=0).fit(x)
    hist = uz.unit_histogram(x, normalize=True)
    assert hist.shape == (16,)
    assert hist.sum() == pytest.approx(1.0)


def test_aggregate_statistics_length() -> None:
    x = np.random.default_rng(6).normal(size=(40, 13))
    vec = aggregate_statistics(x, statistics=("mean", "std", "median"))
    assert vec.shape == (3 * 13,)


def test_aggregate_rejects_unknown_stat() -> None:
    with pytest.raises(InvalidParameterError):
        aggregate_statistics(np.zeros((10, 3)), statistics=("nope",))
