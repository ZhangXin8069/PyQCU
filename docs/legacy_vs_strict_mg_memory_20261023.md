# CUDA QCU Legacy-MG 与 Strict-MG 显存峰值对比（2026-10-23）

状态：基于现有独立进程 A/B benchmark JSON 的静态核对；本次未重新运行 GPU、
CUDA 或数值 benchmark。本文只比较当前记录覆盖的 `c64`、两级、`nvec=12`
路径，不把结论外推到未测精度、层级、粗自由度或硬件。

## 1. 结论

在相同 gauge/RHS、同一类 32 GiB 设备、`c64`、两级和 `nvec=12` 的可比记录中，
**Legacy-MG 的设备级运行时显存峰值低于 Strict-MG**。三个成功尺寸的 Strict
峰值分别比 Legacy 高 `32.46%`、`51.72%` 和 `63.25%`。

| 格点 | Legacy-MG solve | Strict-MG solve | Strict / Legacy |
|---|---:|---:|---:|
| `8x8x8x16` | 639,631,360 B = 0.596 GiB | 847,249,408 B = 0.789 GiB | 1.3246 |
| `12x12x12x24` | 1,646,264,320 B = 1.533 GiB | 2,497,708,032 B = 2.326 GiB | 1.5172 |
| `16x16x16x32` | 4,416,602,112 B = 4.113 GiB | 7,210,008,576 B = 6.715 GiB | 1.6325 |
| `24x24x24x64` | 有 Legacy 记录 | 无 Strict solve JSON | 不可比较 |

表中的数值是 `device.peak_delta_over_baseline_bytes`，即设备级
`cudaMemGetInfo` 采样峰值相对进程基线的增量。该口径包含 CUDA context、
PyTorch allocator 以及 C++ QCU 通过 `cudaMallocAsync` 等原生接口产生的分配，
不只是 PyTorch allocator 统计。

## 2. 测量边界

- benchmark 的 Legacy 与 Strict 分处独立进程，避免两边 CUDA context、allocator
  和缓存互相污染。
- Legacy solve 加载预先构建的 33-tensor cache；其 asset-build 记录是另一条独立
  记录，不混入 solve wall time。
- Strict solve 记录从 fine prepare、hierarchy setup、runtime binding 到 warmup 和
  steady solve 的完整进程阶段，因此其峰值包含 hierarchy setup 与首次 FGMRES
  workspace 分配。
- 两侧使用相同 gauge/source SHA256、mass `-0.277`、Csw `1.160920226`、相对
  容差 `1e-6`、2 次 warmup 和 5 次稳态求解。

因此，该表首先回答“现有实际运行入口的设备级端到端峰值谁更低”，而不是严格
同阶段的纯算法常驻集大小。Legacy asset-build 的独立峰值在 `8x8x8x16`、
`12x12x12x24`、`16x16x16x32` 上仍低于对应 Strict 全程峰值，分别为
`432,013,312`、`1,591,738,368`、`4,722,786,304` bytes。

## 3. 差异来源

Legacy 在 compact odd 粗格点上直接保存 33-tensor coarse operator，矩阵主体约为

```text
33 * E^2 * Vc_odd
```

Strict 保持 full-coarse geometry，运行期保存预条件 links 和 onsite 对，矩阵主体约为

```text
(8 Yhat + 2 onsite) * E^2 * Vc_full + V
```

标准每轴 block=2 时，`Vc_full` 通常接近 `2 * Vc_odd`。此外，Strict 还常驻 fine
transfer/null-vector 资产，并在首次 solve 分配精确大小为
`(2*restart + 5)*B_fine + 2*B_coarse` 的 fused FGMRES arena。

在 `16x16x16x32` 记录中：

| 项目 | 字节数 |
|---|---:|
| Legacy 33-tensor cache | 231,215,104 |
| Strict runtime-owned 总量 | 972,822,540 |
| Strict fine transfer | 301,989,888 |
| Strict coarse assets | 377,487,360 |
| Strict fused FGMRES workspace | 286,261,248 |
| Strict coarse hierarchy workspace | 7,084,044 |

因此，Strict 的 full-coarse links、fine transfer 与首次 Krylov arena 共同造成更高
峰值，不能只比较“33 个 block”和“10 类 link/onsite block”的名义数量。

## 4. 正确性状态与限制

- Legacy 四个尺寸的 solve 均通过其 C++ full-operator true residual 门，但 stdout
  同时记录 `n_vcycles=0`、`coarse_iters=0`。因此本文的 Legacy 峰值是当前实际
  fine-level 路径峰值，不是已经执行过 coarse V-cycle 的算法级公平结论。
- Strict `8x8x8x16`、`12x12x12x24`、`16x16x16x32` 通过独立 Python full Wilson/Clover
  真残差门；`24x24x24x64` 只有 0 字节 stdout，没有 JSON，不能判断失败原因，也
  不能补写峰值。
- benchmark JSON 没有 GPU 型号字段；只能确认总显存为 `34,359,345,152` bytes。
  不能仅凭总量确认所有记录来自同一张物理设备。
- Legacy 的 `n_vcycles=0` 与 Strict 的 cycle 语义不同，表内 wall/显存差异都不是
  MG 算法加速比。

## 5. 证据入口

- harness：`pyqcu/testing/qcu/multigrid/benchmarks/legacy_vs_strict.py`
- Legacy solve：`data/benchmark_legacy_vs_strict_20261022/reports/legacy_solve_*.json`
- Legacy build：`data/benchmark_legacy_vs_strict_20261022/reports/legacy_build_*.json`
- Strict solve：`data/benchmark_legacy_vs_strict_20261022/reports/strict_solve_*.json`
- 状态与可比性：[`legacy_vs_strict_mg_status_20261022.md`](legacy_vs_strict_mg_status_20261022.md)
- Strict workspace 公式：[`_strict_multigrid.py`](../pyqcu/cuda/_strict_multigrid.py)
- 粗层资产语义：[`qcu_quda_multigrid_coarse_operator.md`](qcu_quda_multigrid_coarse_operator.md)

## 6. 使用建议

- 容量受限且 Legacy 当前实际只运行 fine-level 路径时，优先选择 Legacy，不要为
  “名义上是 MG”而引入 Strict 的额外常驻资产。
- 需要 full-coarse、MATPC、`X/Y/Yhat` 和 fused right-FGMRES 时使用 Strict，并按
  setup、runtime asset、first solve workspace 和 steady solve 分阶段记录显存。
- 若要把本文升级为算法的永久结论，必须补齐实际执行 coarse V-cycle 的 Legacy
  记录，以及 `24x24x24x64` Strict 的完整分阶段设备峰值。
