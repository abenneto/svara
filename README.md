# svara

[![CI](https://github.com/abenneto/svara/actions/workflows/ci.yml/badge.svg)](https://github.com/abenneto/svara/actions/workflows/ci.yml)
[![CodeQL](https://github.com/abenneto/svara/actions/workflows/codeql.yml/badge.svg)](https://github.com/abenneto/svara/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**svara**（स्वर，意为“音”）是一个小而完整的语音处理与表示工具箱，聚焦三件事：

1. **语音特征提取** —— 分帧、STFT、梅尔/线性滤波器组、MFCC / LFCC、谱形状与时域特征、差分。
2. **基频（F0）与共振峰分析** —— 自相关 / YIN / 倒谱三种 F0 估计器，以及基于 LPC 的共振峰跟踪。
3. **声学表示学习** —— CMVN、PCA/白化、K-means 离散声学单元、句级统计聚合。

它只依赖 **NumPy / SciPy**，不引入深度学习框架，因此**离线、可复现、易读**——
适合作为歌声/语音合成等下游研究的“语音一侧”特征前端与教学参考实现。

> ⚠️ 早期研究阶段：API 仍可能调整，欢迎在 issue 中讨论。

## 安装

```bash
pip install -e .          # 从源码安装
pip install -e ".[dev]"   # 连同开发/测试依赖
```

要求 Python ≥ 3.9。

## 快速上手

```python
import svara

signal, sr = svara.read_wav("hello.wav")

# 1) 特征
mfcc = svara.mfcc(signal, sr, n_mfcc=13)          # (帧数, 13)
logmel = svara.log_mel_spectrogram(signal, sr)    # (帧数, n_mels)

# 2) 基频与共振峰
f0 = svara.estimate_f0(signal, sr, method="yin")  # (帧数,)，清音为 0
formants = svara.track_formants(signal, sr)       # (帧数, n_formants)

# 3) 声学表示
from svara.representations.normalize import cmvn
from svara.representations.units import AcousticUnitizer

feats = cmvn(mfcc)
units = AcousticUnitizer(n_units=64).fit_encode(feats)   # 离散单元序列
```

## 命令行

安装后提供 `svara` 命令：

```bash
svara extract input.wav -o mfcc.npy --feature mfcc
svara f0 input.wav -o f0.csv --method yin --format csv
svara formants input.wav -o formants.npy
```

## 模块速览

| 模块 | 内容 |
| --- | --- |
| `svara.io` | WAV 读写、多相重采样、特征保存 |
| `svara.framing` | 预加重、分帧、加窗 |
| `svara.spectral` | STFT、功率/幅度谱、谱图 |
| `svara.filterbanks` | Hz/梅尔转换、梅尔/线性三角滤波器组 |
| `svara.features` | MFCC、LFCC、谱形状、时域特征、差分 |
| `svara.pitch` | 自相关 / YIN / 倒谱 F0 估计 |
| `svara.formant` | LPC（Levinson–Durbin）、共振峰跟踪 |
| `svara.representations` | CMVN、PCA、K-means、离散单元、统计聚合 |

## 文档

- [架构说明](docs/architecture.md)
- [使用指南](docs/usage.md)
- [设计笔记](docs/design-notes.md)
- [API 参考](docs/api-reference.md)

## 参与贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[MIT](LICENSE) © Rong Shengxue
