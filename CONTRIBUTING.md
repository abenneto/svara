# 贡献指南

感谢你对 svara 的关注！这是一个处于早期阶段的研究向工具箱，欢迎 issue、
讨论与 PR。

## 开发环境

推荐使用 [uv](https://github.com/astral-sh/uv)：

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

## 提交前检查

CI 会跑以下命令，本地请先自查一遍：

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=svara
```

## 约定

- 保持每个 PR 聚焦一件事，commit 粒度尽量小。
- 新增的公开函数要带类型标注和 docstring（中文即可）。
- 涉及数值算法时，请附上能复现结果的合成信号测试，不要依赖外部音频文件。
- 文档以中文为主，术语可保留英文。

## 行为准则

参与本项目即表示你同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
