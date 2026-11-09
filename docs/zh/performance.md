# 性能

## 测量口径

下表的每个数字都绑定到明确的平台、格点、精度、后端、求解器与迭代数。
跨机器的绝对秒数**不可比**；CPU 与 GPU 行从不合并成单一加速比。历史行保留
下来，是为了记录实现在当前 checkout 无法获得的硬件上的表现；其原始 JSON
不在当前工作树中，来源报告与 git 标签就是 provenance。

本机 `8x8x16x16` 用例使用 Gaussian SU(3) 规范场（`sigma=0.1`、`seed=42`）、
确定性 complex64 源（`seed=42`）、`mass=0.17`、`csw=1.0`、`atol=1e-6`。
Clover BiStabCG 驱动会独立计算全格点真残差。

## 历史跨平台结果

| 用例 | 平台 | 格点/精度 | 后端与求解器 | 结果 | 对照 | 来源 | 可复现性 |
|---|---|---|---|---|---|---|---|
| A | 1× Tesla V100-SXM2-32GB, sm_70 | 24³×64, complex64 | CUDA `applyCloverBistabCgQcu` | 0.147801 s，18 次迭代，8.211 ms/迭代，真残差 6.709e-7 | QUDA Clover BiCGStab 0.156680 s / 19 次 → 1.060× | `docs/three_way_benchmark_20261022.md:82-84` | 历史数据；需 V100 与 `dev103` 时期 checkout |
| B | Intel Xeon E5-2696 v3, 18 核, 1 NUMA | 24³×64, complex64 | KUPL parity-Schur BiStabCG（`--restart 0`） | 8 ranks×2 threads, grid 1×4×2×1：7.532 s / 19 次 | 1 rank×4 threads 19.229 s → 2.553×；4 ranks×4 threads 7.842 s | `docs/three_way_benchmark_20261022.md:85-86,180-185` | 历史数据；绝对秒数需同级别 Xeon |
| C | P100-PCIE-16GB (sm_60) vs V100-SXM2-32GB (sm_70) | 8×8×16×16, complex64 | CUDA `applyCloverBistabCgQcu`，含加载的 solve wall | P100 0.0425 s；V100 0.0149 s；残差 2.846712e-7 / 2.846721e-7 | 架构 A/B，不是加速比声明 | `docs/three_way_benchmark_20261022.md:321-323` | 历史数据；含加载计时，原始 JSON 缺失 |
| D | 1× Tesla V100-SXM2-32GB | 8×8×8×16, complex64, 真实 LIME 前缀 | CUDA Strict-MG `applyMultigridStrictFgmresQcu` | 0.021693 s steady 中位，10 次迭代，真残差 3.348e-7 | QUDA aligned MG 0.144755 s / 8 outer GCR → 6.673× | `docs/qcu_six_way_benchmark_20261023.md:128-131` | 时点证据；工作树不是 clean snapshot |
| E | 1× A100-PCIE-40GB | 32³×64, complex64 | CUDA Legacy-MG `applyCloverMultigridQcu`, restart 20, E12, dof [12,12] | 1/2/4 ranks：0.214456 / 0.219506 / 0.218343 s，20 次迭代 | QUDA GCR-MG 0.482426 / 0.314393 / 0.287465 s → 2.2495× / 1.4323× / 1.3166× | `docs/qcu_mg_performance_ledger_20261029.md:92-110` | 历史数据；需 A100、L32 LIME 与 E12 cache |
| F | 1× A100-PCIE-40GB | 32³×64, complex64 | CUDA native BiStabCG | 1 rank 0.122422 s vs QUDA 0.130358 s；2 ranks 0.115411 s vs QUDA 0.096089 s | 1 rank 1.065×；**2 ranks 0.833×** | `docs/qcu_mg_performance_ledger_20261029.md:48-59` | 历史数据；说明 native 求解器并非普遍更快 |
| G | 1× DCU Pre-Wukong D160A1, gfx906, 16 GB, DTK 23.10 | 24³×64, complex64 | CUDA(HIP) 兼容 QCU Legacy-MG | 1/2/4 ranks：0.301690 / 0.461908 / 1.024669 s，20 次迭代，残差 ≈2.58e-7 | QUDA `BLOCKED-RUNTIME`，无加速比 | `docs/qcu_quda_solver_dcu_l24_20261029.md:99-103` | 历史数据；需同型号 DCU 与 DTK |
| H | 1× Tesla V100, 标签 `test27`/`stab55` | 16×32×32×48, complex64, 3 层 | PyQCU Strict-MG, trace-off | 0.808893 s / 14 次迭代 | QUDA aligned MG 1.640769 s / 39 次 → 2.0284× | `docs/profiler_solver_performance_history.md:261,643` | 历史标签；当前重构可能改变绝对时间 |

由此有两条必须如实说明的限制：

* 用例 F 专门保留，避免只挑有利结果：native BiStabCG 在 1 rank 更快，
  在 2 ranks 反而慢于 QUDA。
* 用例 C、D、G 的格点较小或机器共享，固定开销占主导，不能当作大体积
  kernel 吞吐的证据。

### 历史数据的代码改动审计

`git diff --stat dev103..HEAD -- cpp/kupl/qcu/src cpp/cuda/qcu/src` 显示
18 个文件、4,130 行新增、318 行删除，其中包括 HEAD（`8847f84`）的 KUPL
周期几何修复。用例 A、B 是 `dev103` 时期的数据，只能在对应 tag 上复现，
不能从 `pyqcu==0.1.1` 直接复现。用例 E、F 的账本写
`dev106=2c8fccc8`，而本地 tag `dev106` 解析为 `0a632f4`，因此精确 commit
未核实。用例 H 的 `test27` 报告在 tag `test27` 中存在；对应的 `stab55`
文档路径缺失，因此以 `test27` 为主要 provenance。

## 本机验证：8×8×16×16, complex64

本机构建：`pyqcu==0.1.1`，WSL2，Intel Core i7-13620H（8 逻辑 CPU、1 个
NUMA 节点），NVIDIA GeForce RTX 4060 Laptop GPU（8 GB、sm_89、驱动
617.42），CUDA 12.4，Torch 2.15.0+cu132。安装库 SHA256：
KUPL `793970c6c9903cf4fdb649666604437ef5dd19541dfe9335df486f6041963b4a`，
CUDA `bfd1f8b91be9137c621c4ebb2f30556f07ca9e5b19ad775f0231f9cfe811229e`，
Cython 桥 `b00a84f0a3b73e907e93f6983122694bd3d545f19e876ad6720dd99913e119df`。

0.1.1 是纯元数据版本，0.1.2 增加 CLI 自检，0.1.3 进一步硬化，0.1.4 让 CLI 感知 MPI；求解器代码与 0.1.0 相同，所以下表
的 8x8x16x16 数字仍然有效，它们是在干净的 PyPI 0.1.1 venv 上测量的。仓库驱动
及其自带 NumPy 全格点残差报告 2.21e-7 / 0.134 s。

| 后端 | 设备 | 迭代 | 真残差 | 验收线 | 求解墙钟 | 备注 |
|---|---|---|---|---|---|---|
| KUPL | i7-13620H，1 rank × 4 OpenMP 线程 | 26 | 2.28e-7 | 2.44e-4 | 0.127 s | 干净 PyPI venv；重装 0.143 s |
| CUDA | RTX 4060 Laptop，sm_89 | 26 | 2.96e-7 | 2.44e-4 | 0.039 s | 干净 PyPI venv；一个进程、一个 rank、一张设备 |

安装包在源码树外验证：KUPL 报告指向
`site-packages/pyqcu/kupl/libqcu.so`，CUDA 桥从 `site-packages` 导入，
两次求解都通过同一验收线。本机上 CUDA 整次求解墙钟比 KUPL 短 2.6–3.0×；
这是不同硬件路径的整次求解对比，不是 kernel-only 比值。

命令与验收判据为：

```bash
python pyqcu/testing/simple/kupl_bistabcg.py \
  --lat 8 8 16 16 --dtype c64 \
  --mass 0.17 --csw 1.0 --atol 1e-6 --max-iter 1000 --seed 42 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5 \
  --output data/release_8x8x16x16/solution_kupl_8x8x16x16_c64.h5 \
  --report data/release_8x8x16x16/report_kupl.json
```

验收：`status == "PASS"`、`final_true_residual <= max(atol, 2048*eps32)`
≈ `2.4414e-4`、`0 < iterations <= max_iter`。报告中的 `wall_seconds` 是
求解墙钟；独立残差在之后用 NumPy 计算。

## 什么情况走哪条路径

| 输入 | 执行路径 | 备注 |
|---|---|---|
| Torch CPU 张量 | `pyqcu.torch` 算子 | 总是可用 |
| Torch CUDA 张量 | `pyqcu.torch` CUDA 路径，或构建后的 `pyqcu.cuda` | 桥需要匹配的 `libqcu.so` |
| NumPy 主机数组 | `pyqcu.kupl.qcu` | 需要 KUPL 库与精确 ABI 布局 |
| 无原生库 | 仅 Torch 后端 | `python -m pyqcu` 会报告缺失 |

## 内存

`8×8×16×16` complex64（`16384` sites）：

| 对象 | 元素数 | 字节 |
|---|---|---|
| 规范场 `[3,3,4,X,Y,Z,T]` | 589,824 | 4.72 MB |
| 费米子场 `[4,3,X,Y,Z,T]` | 196,608 | 1.57 MB |
| 单 parity Clover 项 `[4,3,4,3,X,Y,Z,T/2]` | 1,179,648 | 9.44 MB |

KUPL 单 rank 驱动持有四份 Clover 张量，常驻工作集约 52 MB；独立残差还要
一份 complex128 临时拷贝（约 33 MB）。CUDA 设备端工作集同量级；这么小的
格点上 CUDA context 与 Torch allocator 反而占主导。

## 正确性

KUPL 原生化路径由源码仓库测试
`pyqcu/testing/kupl/test_kupl_python.py` 覆盖：

* 40 个导出符号与 C ABI 签名；
* Wilson dslash 对 Torch 参考，`rtol=2e-4`、`atol=2e-5`；
* Clover dslash `rtol=5e-4`、`atol=5e-5`；
* Clover BiStabCG 的独立全残差低于 `2.0e-4`；
* `applyEndQcu` 后的 `set_ptrs` 槽位生命周期。

`8×8×16×16` 发布用例额外用 NumPy 重建全格点 Wilson-Clover 算子，因此报告的
残差没有复用 native 求解器内部的递推量。

## 复现

```bash
# 1) build the native KUPL library
bash ./install.sh kupl

# 2) generate the release gauge field
python pyqcu/testing/simple/give_gauss_gauge.py \
  --lat 8 8 16 16 --seed 42 --sigma 0.1 --backend torch \
  --output data/release_8x8x16x16/gauge_8x8x16x16_c64.h5

# 3) KUPL single-rank solve with independent residual
python pyqcu/testing/simple/kupl_bistabcg.py \
  --lat 8 8 16 16 --dtype c64 \
  --mass 0.17 --csw 1.0 --atol 1e-6 --max-iter 1000 --seed 42 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5 \
  --report data/release_8x8x16x16/report_kupl.json

# 4) CUDA solve on a machine with the CUDA bridge
python pyqcu/testing/simple/qcu_bistabcg.py \
  --lat 8 8 16 16 --mass 0.17 --csw 1.0 --warmup 0 --repeats 1 \
  --gauge data/release_8x8x16x16/gauge_8x8x16x16_c64.h5 \
  --device cuda:0 \
  --report data/release_8x8x16x16/report_cuda.json
```

## 局限

* 历史报告产自当前无法获得的硬件与 checkout；其绝对秒数不是规格。
* P100/V100 行包含库加载，不是 kernel-only 计时。
* CPU 与 GPU 行回答不同问题，不合并。
* 为不同架构构建的原生库可能可以导入但第一次调用就失败；任何新测量都应
  记录库路径与 SHA256。
