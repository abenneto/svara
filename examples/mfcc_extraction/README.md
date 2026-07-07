# MFCC 提取示例

演示如何用 svara 提取 MFCC 及其一阶 / 二阶差分。

```bash
python extract_mfcc.py            # 用内置合成信号
python extract_mfcc.py your.wav   # 用你自己的 WAV
```

输出为各特征矩阵的形状（帧数 × 系数）以及首帧的 MFCC 值。
