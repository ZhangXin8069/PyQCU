---
name: pyqcu-testing-cpu
description: 当任务涉及 pyqcu/testing/cpu 或纯 Python CPU 测试时使用。
---
# pyqcu/testing/cpu

CPU-only tests for the pure-Python backend.

## Test Files

| File | What it tests |
|------|---------------|
| `conftest.py` | Main entry — imports from `pyqcu.testing` |
| `conftest.bistabcg.py` | BiStabCG solver on CPU |
| `conftest.mpi.py` | MPI-distributed solver on CPU |

## Usage

```bash
mpirun -np 4 python pyqcu/testing/cpu/conftest.py
```

若在本机执行 KUPL CTest，`kupl_qcu_kunpeng_profile_contract` 需要单一 NUMA
cpuset 的进程 affinity；宽 affinity 跨两个 NUMA 节点时，生产代码必须按设计
fail-closed。验证时应先 `taskset -c <node-cpulist>`，不得为通过测试删除该检查。

## KUPL Strict-MG 性能门

KUPL Strict-MG 的 1/2/4 MPI 测速使用真实 gauge/RHS、c64、2 levels、block
`2x2x2x2`、`nvec=12`、`coarse_spin=2`、target parity 1，并记录每 rank
threads、NUMA/cpuset、cache load、setup、warmup/steady、外层 iterations 和每层
V-cycle。4-rank 不能在 `1x4x1x1` 使用 `8x8x8x16`：local coarse y 会变成 1 并被
Strict 几何 fail-closed；应改用 `2x2x1x1`，使 local fine/coarse 在每个维度保持
偶数。

测后独立 full Wilson/Clover residual 必须在 MPI solve 退出后的独立进程计算。
在仍持有同一 `COMM_WORLD` 的 rank 0 上重入 distributed dslash 会形成 busy-wait
死锁；该失败不计入性能样本。

共享登录节点必须记录 load average 和用户数。近期 `load average≈91/48 CPUs`、
106 用户的窗口只能作为功能/容量参考，不能据此声明鲲鹏强扩展。当前真实
`8x8x8x16` Strict 数据为 1/2/4 rank median
`1.222731/1.101605/1.658620 s`，对应 speedup
`1.000/1.110/0.737x`，所有独立残差低于 `1e-6`。

## Cache 兼容

KUPL odd-Schur Legacy solve 应直接加载统一 CUDA/Torch/KUPL cache，不重建 payload。
回归入口为 `pyqcu/testing/kupl/test_legacy_mg_cache_compat.py`：同一 gauge、同一
CUDA `lonv` 下检查 `hnn/hdg/sit` 相对误差 `<2e-5`。当前实测均低于 `3e-7`。
