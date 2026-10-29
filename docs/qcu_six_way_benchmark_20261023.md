# QCU 六版本求解器性能评定（QUDA 对照，2026-10-23）

状态：当前工作区正式批次结果，未提交、未打 tag、未推送。本文只解释本轮同窗口数据；
工作树在运行前已包含其他未提交改动，完整快照见
历史路径 `logs/qcu_six_way_20261023/formal-release/environment.json`；该 JSON 未保留在
当前工作树。

后续 16⁴ 统一验收已替代本文作为当前快速矩阵主口径，见
[`qcu_mg_refactor_validation_20261023.md`](qcu_mg_refactor_validation_20261023.md)。
本文的 `8×8×8×16` 数据和 trace 诊断仅作为历史时点证据保留。

## 1. 运行环境

本轮固定使用一个 Python 进程、一个 MPI rank。Torch、CUDA QCU 与 QUDA 共用
`Tesla V100-SXM2-32GB`，CPU 后端使用单节点 Intel Xeon E5-2696 v3 的 16 个 OpenMP 线程。
因此 CUDA/Torch/QUDA 的 GPU 绝对时间可同窗比较；KUPL 是 CPU 路径，只能与另一条 KUPL
路径比较，不能把 CPU/GPU 比值解释为算法优劣。

| 项目 | 值 | 证据 |
|---|---|---|
| Git HEAD | `f20d81c437da365f6067d979b38c0e2617039ab8` | 历史 `environment.json`；当前未保留 |
| GPU | V100-SXM2-32GB, UUID `GPU-be23deb4-29b1-7bb2-29ef-c4ab7b34f0a8`, 32 GiB | 同上 |
| GPU 驱动 / CUDA runtime | 551.78 / 12.8（PyTorch）；QUDA 报告 driver 12.4 | 历史 `environment.json`、[`quda-strict.stdout`](../logs/qcu_six_way_20261023/formal-release/quda-strict.stdout) |
| CPU | Xeon E5-2696 v3, 18 physical cores, 1 NUMA node | 历史 `environment.json`；当前未保留 |
| Python / PyTorch | 3.10.12 / 2.10.0+cu128 | 同上 |
| PyQUDA / QUDA | 0.10.54 / 1.1.0 | 历史 `environment.json`、[`quda-strict.stdout`](../logs/qcu_six_way_20261023/formal-release/quda-strict.stdout) |
| `libqcu.so` CUDA sm70 | SHA256 `9f2d4b0f5ad06d791b734ae74dbc4a34d04bfc999b6b2664595471e6ed2fd9a0` | 同上 |
| `libqcu.so` KUPL | SHA256 `677a9c42a827b5bb61ab333a5c67be19c218dc34e517449b47da7bdda47f2226` | 同上 |
| `libquda.so` sm70 | SHA256 `92db62c3a24b693eba92aac1feab7b8ce87c49826d931d10b649e61ab6b218b9` | 同上 |

GPU 时钟预热执行 3 s；每个版本再做 2 次不计时 warmup 和 5 次 steady 重复。工作树不是
clean snapshot，因此本报告是特定工作区证据，不是可移植的永久性能承诺。

## 2. 输入证据

选择仓库中可完成七条路径的最小真实 LIME 前缀：`8×8×8×16`。它来自真实组态
`beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime` 的原点前缀，保持原始 link、
`mass=-0.277` 与 `csw=1.160920226`。新切口边界不周期，因此结果只用于求解器比较，
不作为物理论文结果。

Gauge 是真实 LIME link；RHS/source 不是 LIME 中存储的物理源，而是同一 PCG64 seed-42
float64 流生成后转为 c64 的确定性源。七条路径读取的是同一 source HDF5，并逐条校验 SHA256。

| 输入 | 路径 | SHA256 |
|---|---|---|
| 原始 LIME | `/root/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime` | `a49773fa3742d310f4ea162a932cf559253aeab708bac08a8dd1f6847df4294b` |
| Gauge HDF5 | 历史路径 `data/benchmark_legacy_vs_strict_20261022/cases/8x8x8x16/gauge_8x8x8x16_c64.h5`；当前未保留 | `bac143c8daca542c8143997b7abdbe0c151d1aab23375bf6597db4ec54c17ac2` |
| Source HDF5 | 历史路径 `data/benchmark_legacy_vs_strict_20261022/cases/8x8x8x16/source_8x8x8x16_c64.h5`；当前未保留 | `6e7d74ce8095d71ad0eacf6464c131792dbf7ba303bbc2d55958dd834dda5ba4` |
| Canonical full null vectors | 历史路径 `data/benchmark_legacy_vs_strict_20261022/nullvec/8x8x8x16_nvec12_full_c64_csw1p160920226.h5`；当前未保留 | `a2383e8415585c682465b8c15f8764554f6dd0011c35506a4b8bf25aa2d9a8d3` |

完整输入与缓存清单的历史路径为 `logs/qcu_six_way_20261023/formal-release/inputs.json`；
该 JSON 未保留在当前工作树。
七条主路径的 gauge/source SHA256、`c64`、`tol=1e-6`、`max_iter=200`、2 层、
`block=2×2×2×2`、`nvec=12` 均逐项核对一致。Legacy 的 coarse DOF 为 12；Strict 的
`coarse_spin=2`，coarse DOF 为 24。

## 3. 同口径结果

主表数字均为 5 次 steady 的中位数；`first` 是第一次零初值 solve，包含该版本首次 lazy
workspace/初始化开销，不进入 steady 统计。`setup` 与 `cache` 的边界按后端单独注明，
不能只凭绝对秒数推断框架初始化效率。

| 版本 | 真实运行 | 实际 coarse 执行 | setup s | cache load s | first solve s | steady median s | 迭代 | 独立真残差 | 设备峰值 | 原始证据 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Torch Legacy | 是 | 是；每次 3 个 coarse correction | 2.644805 | 0 | 0.613778 | **0.607062** | 22 | `1.649e-7` | 402,653,184 B | [`stdout`](../logs/qcu_six_way_20261023/formal-release/torch-legacy.stdout)；JSON 未保留 |
| Torch Strict | 是 | 是；每次 20 个 V-cycle 调用 | 4.843557 | 0.022790 | 1.803473 | **1.670798** | 11 | `2.759e-7` | 308,281,344 B | [`stdout`](../logs/qcu_six_way_20261023/formal-release/torch-strict.stdout)；JSON 未保留 |
| CUDA Legacy | 是 | 是；每次 1 个初始 deflation V-cycle | 0.003782 | 0.013491 | 0.041019 | **0.009302** | 16 | `2.208e-7` | 648,019,968 B | [`stdout`](../logs/qcu_six_way_20261023/formal-release/cuda-legacy.stdout)；JSON 未保留 |
| CUDA Strict | 是 | 是；trace 直接观测 10 个 `coarse_vcycle` stage | 4.967338 | 0.013578 | 0.038042 | **0.021693** | 10 | `3.348e-7` | 782,237,696 B | [`stdout`](../logs/qcu_six_way_20261023/formal-release/cuda-strict.stdout)、[`trace stdout`](../logs/qcu_six_way_20261023/formal-release/cuda-strict-trace.stdout) |
| KUPL Legacy | 是 | 尝试 1 次/solve，7/7 被真残差门拒绝，后续 fine-only | 0.013240 | 0.015308 | 0.260122 | **0.256530** | 48 | `6.199e-7` | CPU，N/A | [`stdout`](../logs/qcu_six_way_20261023/formal-release/kupl-legacy.stdout)；JSON 未保留 |
| KUPL Strict | 是，经原始 C ABI | 是；10 个外层迭代对应至少 10 个递归 V-cycle | 0.013145 | 0.022495 | 0.309197 | **0.268440** | 10 | `2.930e-7` | CPU，N/A | [`stdout`](../logs/qcu_six_way_20261023/formal-release/kupl-strict.stdout)；JSON 未保留 |
| QUDA aligned MG | 是 | 是；8 个右预条件 GCR 外层迭代 | 0.535602 | N/A | 0.197704 | **0.144755** | 8 | `5.067e-7` | 138,412,032 B | [`stdout`](../logs/qcu_six_way_20261023/formal-release/quda-strict.stdout)；JSON 未保留 |

说明：

- Torch Legacy 的 `setup` 包含 null-vector 和层级构建；CUDA Legacy 的 `setup` 只含
  `applyInitQcu`，33-tensor cache 已预先构建并单独记录 `cache load`；两者 setup 不可直接相减。
- CUDA Legacy 在同一 `LatticeSet` 上连续求解会出现跨 solve 状态污染：单次求解独立残差正常，
  但同对象末次解对原始 RHS 的真残差可升至 `O(0.58)`。正式样本因此在每次计时前重建运行时，
  重建耗时不计入 solve；原始行为与边界见
  [`cuda-legacy.stdout`](../logs/qcu_six_way_20261023/formal-release/cuda-legacy.stdout)。
- KUPL Strict 的 C++ hierarchy init 只有 `0.013145 s`，但它消费的是由 Torch Strict 在同一
  gauge/null-vector 上导出的 41 MiB strict assets，资产构建时间没有重复计入。其 setup 不能
  与 Torch/CUDA/QUDA 的完整层级构建时间直接比较。
- CUDA Strict 的 workspace 规划值为 `17,891,328 B`，KUPL Strict 为同一 `17,891,328 B`；
  Torch 的首次 lazy allocator 增量分别为 Legacy `794,624 B`、Strict `786,432 B`。设备峰值
  使用 `cudaMemGetInfo` 设备级采样，包含原生分配但不包含 host RSS。
- 每条路径均在 side JSON 中记录全 Wilson/Clover 独立残差；CUDA/QUDA 的独立检查
  stdout 见
  [`cuda-legacy-residual.stdout`](../logs/qcu_six_way_20261023/formal-release/cuda-legacy-residual.stdout)、
  [`cuda-strict-residual.stdout`](../logs/qcu_six_way_20261023/formal-release/cuda-strict-residual.stdout) 与
  [`quda-strict-residual.stdout`](../logs/qcu_six_way_20261023/formal-release/quda-strict-residual.stdout)。

### CUDA Legacy fine-only 消融

同一 V100、同一输入、同一参数，只关闭初始 coarse deflation：

| 模式 | steady median s | 迭代 | 独立真残差 | 设备峰值 | 证据 |
|---|---:|---:|---:|---:|---|
| CUDA Legacy，初始 deflation | 0.009302 | 16 | `2.208e-7` | 648,019,968 B | [`stdout`](../logs/qcu_six_way_20261023/formal-release/cuda-legacy.stdout) |
| CUDA Legacy，fine-only | **0.008500** | 16 | `2.220e-7` | 622,854,144 B | [`stdout`](../logs/qcu_six_way_20261023/formal-release/cuda-legacy-fine-only.stdout) |

该输入上 fine-only 比 coarse-enabled 快 `1.094×`，迭代数没有减少。因而 CUDA Legacy 的
coarse 路径收益为负，且 fine-only 的收益只能标为“关闭 coarse 后的 fine-only 收益”，
不能写成 coarse MG 的算法收益。

## 4. 历史边界与后续修复

1. **旧批次中 KUPL Strict 公共 Python 包装不可直接运行（后续已修复）。** `pyqcu.kupl.multigrid.StrictMultigrid`
   向 strict init 传入 tensor 而 C ABI 要求 `unsigned long long *`，并且 FGMRES 包装缺少
   fine/coarse `E` 两个参数。本轮没有改求解器实现，而是通过同一原始 C ABI 完成真实运行；
   证据见 [`kupl-strict.stdout`](../logs/qcu_six_way_20261023/formal-release/kupl-strict.stdout)。
   该限制随后已修复：16⁴ 批次 public/raw 同解相对差
   `8.32e-8`，见 [`qcu_mg_refactor_validation_20261023.md`](qcu_mg_refactor_validation_20261023.md)。
2. **QUDA 不能加载共享 canonical null-vector QIO。** 当前 sm70 `libquda.so` 的
   `QUDA_QIO=OFF`，因此 QUDA 在同一 gauge 上自行生成 null vectors，层数、block、`nvec=12`、
   `coarse_spin=2`、容差和迭代预算相同，但 null-space 资产不是 byte-identical。该对照是
   同配置算法对照，不是完全相同的 null-basis replay。
3. **CUDA Legacy 同一运行时对象的连续 cold solve 不可作为合法 steady 样本。** 发现后已改为
   每次计时前重建运行时；未重建的连续调用只保留为失败/污染边界，不进入主表。

除上述边界外，六版本和 QUDA 均完成真实求解，不存在用随机/伪造组态替代的路径。

## 5. 性能结论

### 5.1 算法收益

- 同一 V100、同一 gauge/source、相同 `c64/tol/max_iter/2 levels/2^4 block/nvec=12` 下，
  CUDA Strict 的 steady median 为 `0.021693 s`，QUDA aligned MG 为 `0.144755 s`，
  CUDA Strict 快 `6.673×`；CUDA Strict 独立真残差为 `3.348e-7`，QUDA 为 `5.067e-7`。
  该倍率只说明本工作区、本小格、本构建下的路径差异；8³×16 的固定开销占比很高，不能外推为
  大格 kernel 吞吐。
- Torch Strict 为 `1.670798 s`，比 QUDA 慢 `11.542×`。这主要是 Python eager 控制流和
  Python 张量开销，不是 GPU kernel 算力差异。
- CUDA Strict 比 CUDA Legacy 初始 deflation 路径慢 `2.332×`（`0.021693 / 0.009302`）；
  两者分别走 recursive full-coarse FGMRES 与 compact Legacy correction，不能把该比值解释成
  同一算法的实现优劣。CUDA Strict 相对 CUDA Legacy fine-only 慢 `2.552×`。
- Torch Strict 比 Torch Legacy 慢 `2.752×`；两者都在 Torch eager 路径上，差异主要来自
  Strict 的 20 次 V-cycle 调用与 11 次外层迭代，而 Legacy 为 3 次 coarse correction。
- KUPL Strict `0.268440 s` 与 KUPL Legacy 的有效 fine-only `0.256530 s` 相差 `1.046×`。
  这是同一 CPU 节点上的后端/算法组合差异；不能与 GPU 秒数混算。

### 5.2 Fine-only 与 coarse 的区分

- CUDA Legacy fine-only：`0.008500 s`，16 次迭代，`n_vcycles=0` 的关闭配置；只能报告为
  fine-only 结果。
- CUDA Legacy 初始 deflation：`0.009302 s`，16 次迭代，实际执行 1 个初始 coarse V-cycle；
  在该小格上没有减少迭代，反而更慢。
- KUPL Legacy：每次 solve 都尝试 1 次 coarse correction，但 7/7 被 true-residual 门拒绝，
  最终有效路径仍是 fine-only；不能把 `0.256530 s` 标成成功 coarse MG 的收益。
- Torch Legacy 与 Strict、CUDA Strict、KUPL Strict 均实际进入 coarse/V-cycle 路径；
  其中 CUDA Strict 有独立 trace 的 10 个 `coarse_vcycle` stage 直接证据。

### 5.3 硬件与环境差异

- CUDA Strict 相对 QUDA 的 `6.673×` 是同一 V100 的 GPU 路径差异；CUDA Legacy 与 Torch
  结果也在同一 V100，但分别使用 C++ CUDA 和 Python eager 数据面。
- KUPL 绝对时间来自 1 rank × 16 OpenMP threads 的 CPU 节点，不能与 GPU 绝对时间合并为
  跨硬件加速比。
- CUDA Strict 设备级峰值增量 `782,237,696 B` 约为 QUDA `138,412,032 B` 的 `5.65×`；
  Torch Strict 为 `308,281,344 B`，CUDA Legacy 为 `648,019,968 B`。这些是设备级
  `cudaMemGetInfo` 增量，不等同于 allocator 峰值或 host RSS。
- QUDA setup `0.535602 s`、CUDA Strict setup `4.967338 s` 的差异还包含不同 null-space
  构建策略和资产所有权，不应单独归因于某个 kernel 或某一后端。

## 6. 未验证项

- 更大真实格点（尤其 `16³×32`、`24³×64`）的六版本同窗复测；本轮只选择可完成的真实前缀。
- c128 主口径、复杂 MPI 分域、多节点、多 GPU 和 NUMA 绑定的统一六版本矩阵。
- QUDA 与 PyQCU 共享 byte-identical canonical QIO null vectors 的严格 replay。
- KUPL Strict 公共 Python 包装修复后的同一 ABI 运行、以及修复前后的数值逐位对照。
- CUDA Legacy 除初始 deflation 外的长期 periodic correction、更多 correction gate 轨迹，
  以及大格上是否会改变迭代数。
- 生产大格上的 GPU 频率锁定、功耗、温度和长时间热稳态；本轮 V100 是共享设备，未锁频。
- Torch Strict 的 20 次 cycle 调用与 11 次外层迭代之间的完整内部事件分解；本轮只记录公开
  cycle 调用计数，不修改求解器插桩。
- 所有绝对秒数的跨机器、跨驱动、跨共享负载窗口可移植性。

## 7. 复现入口

```bash
source ./env.sh
source pyqcu/testing/qcu/strict/quda_comparison/quda_env.sh
export QUDA_ENABLE_TUNING=0
python pyqcu/testing/benchmark/qcu_six_way.py \
  --warmups 2 --repeats 5 --gpu-warmup-seconds 3 \
  --out-dir logs/qcu_six_way_20261023/formal-release
```

机器可读汇总与逐条原始文件：

- [`summary.md`](../logs/qcu_six_way_20261023/formal-release/summary.md)
- 历史 `summary.json`、`summary.csv`、`inputs.json`、`environment.json` 与
  `data/qcu_six_way_20261023/solution_manifest.json` 在当前工作树未保留。
