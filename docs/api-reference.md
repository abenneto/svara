# API 参考

按模块列出对外接口。签名以关键字参数的默认值为准，完整说明见各函数的
docstring。数组约定：输入信号为 `float64` 单声道；帧级特征为
`(n_frames, n_features)`。

## svara（顶层重导出）

| 名称 | 说明 |
| --- | --- |
| `read_wav(path, mono=True)` | 读取 WAV，返回 `(signal, sr)` |
| `write_wav(path, signal, sr)` | 写 16-bit PCM WAV |
| `resample(signal, orig_sr, target_sr)` | 多相重采样 |
| `preemphasis(signal, coef=0.97)` | 一阶预加重 |
| `frame_signal(signal, frame_length, hop_length)` | 分帧 |
| `stft(signal, n_fft=512, ...)` | 短时傅里叶变换 |
| `spectrogram(signal, n_fft=512, power=2.0, ...)` | 功率/幅度谱图 |
| `mel_spectrogram / log_mel_spectrogram(signal, sr, ...)` | 梅尔谱 |
| `mfcc(signal, sr, n_mfcc=13, ...)` | MFCC |
| `lfcc(signal, sr, n_lfcc=13, ...)` | LFCC |
| `delta(features) / delta_delta(features)` | 差分特征 |
| `estimate_f0(signal, sr, method="yin", ...)` | F0 估计（统一入口） |
| `voiced_flags(f0)` | 浊音布尔掩码 |
| `formants(frame, sr, ...) / track_formants(signal, sr, ...)` | 共振峰 |

## svara.spectral

- `stft(signal, n_fft=512, hop_length=None, win_length=None, window="hann")`
- `magnitude_spectrum(complex_spec)` / `power_spectrum(complex_spec)`
- `spectrogram(signal, ..., power=2.0)`
- `fft_frequencies(sample_rate, n_fft)` → 各频点的 Hz

## svara.filterbanks

- `hz_to_mel(hz)` / `mel_to_hz(mel)`
- `mel_filterbank(n_filters, n_fft, sample_rate, fmin=0.0, fmax=None)` → `(n_filters, n_fft//2+1)`
- `linear_filterbank(n_filters, n_fft, sample_rate, fmin=0.0, fmax=None)`

## svara.features

- `features.melspec.mel_spectrogram / log_mel_spectrogram`
- `features.cepstral.mfcc / lfcc / dct_ii / sinusoidal_lifter`
- `features.shape.spectral_centroid / spectral_bandwidth / spectral_rolloff / spectral_flatness`
- `features.temporal.zero_crossing_rate / rms_energy`
- `features.deltas.delta / delta_delta`

## svara.pitch

- `f0_autocorrelation(signal, sr, fmin=50, fmax=500, ...)`
- `f0_yin(signal, sr, threshold=0.1, ...)`
- `f0_cepstrum(signal, sr, prominence=3.0, ...)`
- `estimate_f0(signal, sr, method="yin", **kwargs)`
- `voiced_flags(f0)`

## svara.formant

- `autocorrelate(frame, order)`
- `levinson(r, order)` → `(coeffs, error)`
- `lpc(frame, order)`
- `formants(frame, sr, order=None, max_formants=5, ...)`
- `track_formants(signal, sr, n_formants=4, ...)`

## svara.representations

- `normalize.cmvn(features, norm_vars=True)`
- `decomposition.PCA(n_components, whiten=False)` —— `fit / transform / fit_transform`
- `cluster.KMeans(n_clusters, random_state=0)` —— `fit / predict / fit_predict`；属性 `cluster_centers_ / inertia_ / n_iter_`
- `units.AcousticUnitizer(n_units=64)` —— `fit / encode / fit_encode / unit_histogram`；属性 `codebook`
- `aggregate.aggregate_statistics(features, statistics=DEFAULT_STATISTICS)`

## svara.config / svara.pipeline

- `config.FeatureConfig(sample_rate=16000, n_fft=512, hop_length=160, ...)`
- `pipeline.extract_features(signal, config=None, with_deltas=True, f0_method="yin")` → dict

## 异常

所有异常继承自 `svara.SvaraError`：`InvalidParameterError`、`AudioIOError`、`NotFittedError`。
