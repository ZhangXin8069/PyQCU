# AGENTS.md — pyqcu

顶层 Python 包：QCU 的纯 Python 实现 + 各 GPU 后端 Cython 桥接。实现 Wilson/Clover 狄拉克算子、BiStabCG 与多重网格求解器、stout smearing、规范场生成，全部 MPI 分布于 4D 进程网格。

## 架构

1. **纯 Python**（`dslash/`、`solver/`、`smear/`）— PyTorch 实现，CPU/CUDA/昇腾 NPU（经 `pyqcu.cann`）可用；开发、测试与 NPU 部署路径。
2. **C++ CUDA 后端**（`cuda/` → `cpp/cuda/qcu/`）— 手工调优 CUDA 内核 + MPI halo 交换，经 Cython 桥（`pyqcu.cuda.qcu`）以 `long long` 裸指针传数据。

多重网格求解器可混合两层：最细层用 C++ 后端平滑（`with_cuda_qcu=True`），粗层纯 Python。

## 目录

| 路径 | 内容 |
|---|---|
| `lattice/` | gamma/Gell-Mann 矩阵、SU(3) 检查、规范场生成；Ward 负索引约定 |
| `dslash/` | Wilson/Clover 狄拉克算子（hopping/sitting/operator） |
| `solver/` | BiCGStab(l)、多重网格 V-cycle（`_gmres.py` 为占位） |
| `smear/` | stout smearing |
| `tools/` | MPI 网格、奇偶分割、HDF5 I/O、linalg、multigrid 转移、TileLang（可选） |
| `cuda/` | Cython 桥（qcu.pyx/pxd/pyi + define.py） |
| `dtk/`、`maca/`、`cann/qcu/` | 占位 PASS（未实现） |
| `cann/` | 昇腾 NPU torch 兼容层（复数分解） |
| `testing/` | 集成测试（`test_*`，由 `examples/*/conftest.py` 引用） |

## 核心惯例

- **复数张量运算必须用 `pyqcu.cann as _torch`**（NPU 复数分解），不要直接 `torch.*`；例外：`torch.linalg.det`、`torch.matrix_exp`。
- `pyqcu.cann.force_use_npu = True` 可在 CPU 上强制 NPU 路径测试；部分模块另有逐模块 `force_use_npu` 标志。
- Cython 桥：`.pxd` 必须与 `cpp/cuda/qcu/python/pyqcu.h` 完全一致，否则静默内存损坏；`applyInitQcu`/`applyEndQcu` 之间每次调用后必须 `params[define._SET_INDEX_] += 1`（粗网格 dslash 例外，重置为 0）。
- 入参一律 NumPy 数组/内建标量；禁止字节串入参（Cython `const char*` 签名需显式声明，避免 argtypes 冲突）。
- Cython 文件由 `macro_gen.py` 生成，改签名后需重新生成并重编后端。
- 反模式：`torch.linalg.inv` 禁止逐 site 循环（用批量逆）；禁止在阻塞 `Sendrecv` 前后加 `MPI.Barrier()`；禁止用 `self.sitting` 对象做 truthy 判断。

## 测试

- 每模块同目录测试文件，函数名 `test_*` 对应 pyqcu 符号；`pytest pyqcu/testing/` 或 `cd examples && pytest .`。
- 参考 HDF5 数据在 `examples/data/`；测试失败须 `assert` 显式抛出，禁止静默。

## Skills

`skills/` 下各子目录的完整生成 skill（原 CLAUDE.md 内容）：`cann`、`cuda`、`dslash`、`lattice`、`smear`、`solver`、`testing`、`tools`。
