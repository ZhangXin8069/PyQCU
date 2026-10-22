# AGENTS.md — pyqcu/testing/benchmark

对比 PyTorch、TileLang 与 C++ CUDA 后端实现的性能基准。

## 文件

| 文件 | 用途 |
|---|---|
| `qcu_six_way.py` | Torch/CUDA/KUPL Legacy/Strict、QUDA 对照与 CUDA Legacy fine-only 消融的统一正式编排器 |
| `qcu_six_way_worker.py` | 单后端隔离 worker、独立真残差检查与统一 side JSON schema |
| `test_qcu_six_way.py` | CUDA trace、KUPL correction 判定和 summary schema 的 CPU 回归 |
| `conftest.py` | 基准入口 |
| `env.py` | 基准环境配置 |

## 用法

```bash
python pyqcu/testing/benchmark/conftest.py
```

六版本正式基准：

```bash
python pyqcu/testing/benchmark/qcu_six_way.py \
  --warmups 2 --repeats 5 \
  --out-dir logs/qcu_six_way_20261023/formal-release
```

正式 profile 至少要求 3 次 steady 重复。不要把 CUDA Legacy 的 `n_vcycles=0`、KUPL
Legacy 被回滚的 correction 或 QUDA 自生成 null vector 的路径混写成同一算法口径。
Legacy-MultiGrid 是 QCU 默认开发/优化对象；本基准中的 Strict 行均是显式对照路径。
