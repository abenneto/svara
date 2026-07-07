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

from svara.config import FeatureConfig
from svara.io import read_wav, save_features
from svara.pipeline import extract_features
from svara.pitch import estimate_f0


def _add_common_io(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("input", help="输入 WAV 文件路径")
    sub.add_argument("-o", "--output", required=True, help="输出文件路径")


def _cmd_extract(args: argparse.Namespace) -> int:
    signal, sr = read_wav(args.input)
    config = FeatureConfig(sample_rate=sr, n_mfcc=args.n_mfcc, n_mels=args.n_mels)
    feats = extract_features(signal, config=config, f0_method=args.f0_method)
    save_features(args.output, feats[args.feature], fmt=args.format)
    print(f"已写出 {args.feature}，形状 {feats[args.feature].shape} -> {args.output}")
    return 0


def _cmd_f0(args: argparse.Namespace) -> int:
    signal, sr = read_wav(args.input)
    f0 = estimate_f0(
        signal, sr, method=args.method, fmin=args.fmin, fmax=args.fmax, hop_length=args.hop
    )
    save_features(args.output, f0.reshape(-1, 1), fmt=args.format)
    voiced = float((f0 > 0).mean())
    print(f"F0 帧数 {f0.size}，浊音占比 {voiced:.1%} -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造顶层参数解析器及各子命令。"""
    parser = argparse.ArgumentParser(
        prog="svara",
        description="语音特征提取与声学表示工具箱",
    )
    parser.add_argument("--version", action="store_true", help="打印版本号后退出")
    subparsers = parser.add_subparsers(dest="command")

    extract = subparsers.add_parser("extract", help="提取帧级特征并保存")
    _add_common_io(extract)
    extract.add_argument(
        "--feature",
        default="mfcc",
        choices=["mfcc", "log_mel", "mfcc_delta", "mfcc_delta2"],
        help="要导出的特征名",
    )
    extract.add_argument("--n-mfcc", type=int, default=13, dest="n_mfcc")
    extract.add_argument("--n-mels", type=int, default=40, dest="n_mels")
    extract.add_argument("--f0-method", default="yin", dest="f0_method")
    extract.add_argument("--format", default="npy", choices=["npy", "csv", "json"])
    extract.set_defaults(handler=_cmd_extract)

    f0 = subparsers.add_parser("f0", help="估计基频轨迹")
    _add_common_io(f0)
    f0.add_argument("--method", default="yin", choices=["yin", "autocorrelation", "cepstrum"])
    f0.add_argument("--fmin", type=float, default=50.0)
    f0.add_argument("--fmax", type=float, default=500.0)
    f0.add_argument("--hop", type=int, default=256)
    f0.add_argument("--format", default="npy", choices=["npy", "csv", "json"])
    f0.set_defaults(handler=_cmd_f0)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主函数，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from svara import __version__

        print(__version__)
        return 0

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    result: int = handler(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
