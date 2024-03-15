"""声学表示子包。

这里放的是在帧级特征之上、把语音映射到更紧凑或更利于下游任务的表示的方法，
全部离线、可复现、只依赖 NumPy / SciPy：

- :mod:`svara.representations.normalize`      倒谱均值方差归一化（CMVN）
- :mod:`svara.representations.decomposition`  PCA 与白化
- :mod:`svara.representations.cluster`        K-means 聚类
- :mod:`svara.representations.units`          离散声学单元编码
- :mod:`svara.representations.aggregate`      句级统计聚合
"""

from __future__ import annotations
