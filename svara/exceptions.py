"""svara 使用的异常层次。

所有对外抛出的异常都继承自 :class:`SvaraError`，方便调用方用一个
``except svara.SvaraError`` 捕获本库产生的全部错误。
"""

from __future__ import annotations


class SvaraError(Exception):
    """本库所有异常的基类。"""


class InvalidParameterError(SvaraError, ValueError):
    """参数取值非法（例如负的帧长、越界的频率范围）。"""


class AudioIOError(SvaraError, OSError):
    """读写音频文件时发生的错误。"""


class NotFittedError(SvaraError, RuntimeError):
    """在调用 ``transform`` 之前没有先 ``fit`` 的估计器。"""
