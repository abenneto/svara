"""集中管理短时分析与特征提取的公共参数。

把散落在各函数里的 ``n_fft`` / ``hop_length`` / ``n_mels`` 等收拢到一个不可变的
配置对象，既方便复用，也保证同一次实验里各特征使用完全一致的分帧设置。
"""

from __future__ import annotations

from dataclasses import dataclass

from svara.exceptions import InvalidParameterError


@dataclass(frozen=True)
class FeatureConfig:
    """特征提取的公共配置。

    所有时长相关字段以样点为单位；``fmax=None`` 表示奈奎斯特频率。
    """

    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int | None = None
    window: str = "hann"
    n_mels: int = 40
    n_mfcc: int = 13
    fmin: float = 0.0
    fmax: float | None = None
    preemphasis: float = 0.97

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise InvalidParameterError("sample_rate 必须为正")
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise InvalidParameterError("n_fft 与 hop_length 必须为正")
        if self.win_length is not None and self.win_length > self.n_fft:
            raise InvalidParameterError("win_length 不能大于 n_fft")
        if self.n_mfcc > self.n_mels:
            raise InvalidParameterError("n_mfcc 不能大于 n_mels")
        top = self.fmax if self.fmax is not None else self.sample_rate / 2.0
        if not 0.0 <= self.fmin < top <= self.sample_rate / 2.0:
            raise InvalidParameterError("需要满足 0 <= fmin < fmax <= sr/2")

    @property
    def effective_fmax(self) -> float:
        """把 ``None`` 解析成具体的奈奎斯特频率。"""
        return self.fmax if self.fmax is not None else self.sample_rate / 2.0
