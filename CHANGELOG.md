# 更新日志

本文件记录值得注意的变更，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.0] - 2026-07-08

### Added
- MFCC / LFCC、谱形状与时域特征、一阶/二阶差分。
- 自相关 / YIN / 倒谱三种 F0 估计器及统一入口 `estimate_f0`。
- 基于 LPC（Levinson–Durbin）的共振峰估计与逐帧跟踪。
- 声学表示：CMVN、PCA/白化、K-means、离散声学单元、统计聚合。
- `svara` 命令行：`extract` / `f0` / `formants` 子命令。
- 端到端特征流水线 `extract_features` 与 `FeatureConfig`。

### Changed
- 自相关 F0 增加抛物线插值，提升高音区精度。
- 梅尔与线性滤波器组改为向量化构造。

## [0.0.1]

### Added
- 项目骨架：包结构、打包配置、CI、lint/format/类型检查与测试脚手架。
- WAV 读写与多相重采样。
- 分帧、加窗、STFT 与谱图等基础短时分析原语。

[Unreleased]: https://github.com/abenneto/svara/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/abenneto/svara/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/abenneto/svara/releases/tag/v0.0.1
