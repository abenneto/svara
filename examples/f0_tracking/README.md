# F0 轨迹估计示例

并排比较自相关、YIN、倒谱三种基频估计方法。

```bash
python track_f0.py            # 用内置合成信号（已知 F0 ≈ 160 Hz）
python track_f0.py your.wav
```

每种方法会打印浊音帧的中位 F0 与浊音占比，便于快速对比。
