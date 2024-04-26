"""``svara`` 命令行入口。

子命令：

- ``extract``   提取 MFCC / log-mel 等特征并保存
- ``f0``        估计基频轨迹
- ``formants``  跟踪共振峰

设计上只做参数解析与 I/O，真正的计算都委托给库函数。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """构造顶层参数解析器（子命令在后续提交里逐个挂上）。"""
    parser = argparse.ArgumentParser(
        prog="svara",
        description="语音特征提取与声学表示工具箱",
    )
    parser.add_argument("--version", action="store_true", help="打印版本号后退出")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主函数，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from svara import __version__

        print(__version__)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
