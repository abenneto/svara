"""特征提取子包。

各模块按特征族划分：

- :mod:`svara.features.melspec`  梅尔谱 / 对数梅尔谱
- :mod:`svara.features.cepstral` MFCC / LFCC 等倒谱特征
- :mod:`svara.features.shape`    谱形状描述子（质心、带宽、滚降、平坦度）
- :mod:`svara.features.temporal` 时域特征（过零率、能量）
- :mod:`svara.features.deltas`  一阶 / 二阶差分
"""

from __future__ import annotations
