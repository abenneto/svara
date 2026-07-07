# 使用指南

本文按典型任务组织，给出可直接复制运行的片段。所有示例都假设已经
`pip install -e .`，并用 16 kHz 单声道信号。

## 读取与预处理

```python
import svara

signal, sr = svara.read_wav("utt.wav")          # 归一化到 [-1, 1] 的 float64
signal = svara.preemphasis(signal, coef=0.97)   # 可选：预加重
```

若采样率不一致，先重采样：

```python
from svara.io import resample
signal = resample(signal, orig_sr=sr, target_sr=16000)
sr = 16000
```

## 帧级特征

```python
mfcc = svara.mfcc(signal, sr, n_mfcc=13, n_fft=512, hop_length=160)
logmel = svara.log_mel_spectrogram(signal, sr, n_mels=40)
```

配合差分得到常见的 39 维特征：

```python
import numpy as np
from svara.features.deltas import delta, delta_delta

feat39 = np.concatenate([mfcc, delta(mfcc), delta_delta(mfcc)], axis=1)
```

谱形状与时域特征需要先自备谱图：

```python
from svara.spectral import spectrogram, fft_frequencies
from svara.features.shape import spectral_centroid, spectral_flatness

mag = spectrogram(signal, n_fft=1024, power=1.0)
freqs = fft_frequencies(sr, 1024)
centroid = spectral_centroid(mag, freqs)
flatness = spectral_flatness(mag)
```

## 基频（F0）

```python
f0_yin = svara.estimate_f0(signal, sr, method="yin", fmin=50, fmax=400)
voiced = svara.voiced_flags(f0_yin)     # 布尔数组，True 为浊音
```

三种方法各有取舍：

| 方法 | 特点 | 适用 |
| --- | --- | --- |
| `autocorrelation` | 快、实现简单，已带亚样本插值 | 干净、单一音高 |
| `yin` | 抗倍频/半频错误，精度高 | 一般语音，推荐默认 |
| `cepstrum` | 依赖谐波结构 | 谐波丰富的浊音段 |

## 共振峰

```python
formant_track = svara.track_formants(signal, sr, n_formants=4)   # (帧数, 4)，缺失为 NaN
mean_formants = np.nanmean(formant_track, axis=0)
```

## 声学表示

```python
from svara.representations.normalize import cmvn
from svara.representations.decomposition import PCA
from svara.representations.units import AcousticUnitizer
from svara.representations.aggregate import aggregate_statistics

feats = cmvn(mfcc)                                    # 归一化
low_dim = PCA(n_components=8, whiten=True).fit_transform(feats)
units = AcousticUnitizer(n_units=64, random_state=0).fit_encode(feats)
utt_vec = aggregate_statistics(feats)                # 定长句级向量
```

## 一步到位的流水线

```python
from svara.config import FeatureConfig
from svara.pipeline import extract_features

cfg = FeatureConfig(sample_rate=sr, n_mfcc=13, hop_length=160)
bundle = extract_features(signal, cfg)     # dict: log_mel / mfcc / f0 / mfcc_delta / mfcc_delta2
```
