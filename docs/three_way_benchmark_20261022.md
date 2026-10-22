# CUDA QCU、KUPL QCU 与 QUDA 三方性能测评（2026-10-22）

状态：工作区结果，尚未提交、打 tag 或推送。数据与脚本均可按第 8、9 节复现。

## 1. 结论摘要

在参考 LIME 组态 `beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950`
及其同 mass、同 Csw 的子体积上，以相同的 PCG64 seed-42 源、`atol=1e-6`
相对 Schur 残差和 c64 精度完成三方对照。主要结论：

| 项目 | 结果 |
|---|---|
| CUDA QCU 对 QUDA（全格点 `24³×64`，同窗） | `0.147801 s` 对 `0.156680 s`（QUDA 内部 solver 计时），CUDA 快 `6.01%`；每迭代 `8.211 ms` 对 `8.246 ms`（CUDA 快 `0.43%`），CUDA 少 1 次迭代 |
| CUDA QCU 最佳生产路径 | standalone `applyCloverBistabCgQcu`，18 次迭代；fine-only Legacy MG 需 20 次迭代、`0.173372 s`，不应替代 BiCGStab |
| KUPL QCU 最佳路径 | `applyCloverMultigridQcu --restart 0`（关闭 coarse hierarchy 的 parity-Schur BiCGStab），`24³×64` 上 8 ranks × 2 threads（`1×4×2×1`）为 `7.532 s`、19 次迭代 |
| KUPL coarse correction | 同一组态 `restart=5/10/20` 均需约 47 次迭代、约 `15.6–16.7 s`，比 `restart=0` 慢约 `2.8–3.0×`；当前 E12 coarse space 在该组态上产生负收益 |
| KUPL 并行效率 | 同窗 16 核：1 rank×4 thr `19.229 s` → 4 ranks×4 thr `7.842 s`（`2.45×`，`61.3%`）；8 ranks×2 thr `7.532 s` 比单 rank×16 thr `9.397 s` 快 `24.8%`；16 ranks×1 thr 回升到 `8.3 s` |
| KUPL 构建优化 | 新增 `-flto -fno-semantic-interposition` 构建：8×2 同窗 `7.556 → 7.196 s`（快 `4.8%`），解逐位一致，CTest 29/29 |
| 小体积行为 | 子体积上 QUDA 单次 solve 存在约 `26–43 ms` 的固定开销；`8³×16` 上 CUDA 单次 solve 比 QUDA 快 `3.79×` |
| 正确性 | 所有运行 `PASS`；CUDA 独立真残差 `2.28e-7–6.71e-7`，QUDA `2.16e-7–9.65e-7`，KUPL 独立真残差 `2.01e-7–4.01e-7`（含 2/4/8/16 ranks 分布式全算子） |
| 多 rank 独立残差 | 新增分布式 halo 重构；`8³×16` 1/2/4 ranks 真残差 `2.0022e-7 / 2.0052e-7 / 1.9986e-7`，解相对差 `1.18e-7`；`24³×64` 4/8/16 ranks 均为 `4.01e-7` |
| GPU 架构选择 | CUDA QCU 与 QUDA 各有原生 sm_60、sm_70 两套安装；按 GPU compute capability 自动选择，P100（sm_60）与 V100（sm_70）解逐位一致（CUDA 相对差 `0.0`，QUDA `2.64e-8`） |

`24³×64` 上 CUDA 的 `0.147801 s` 与 QUDA 的 `0.156680 s` 是同一硬件
（单张 V100-32GB）、同一 gauge、同一 RHS、同一 `atol`、同一时间窗口的
口径，可以直接比较；CUDA/QUDA 每个数字是 3 次独立调用的中位数，每次
调用内部再做 1 次 warmup + 3 次计时。KUPL 的五组配置在同一窗口内
（05:27–05:38）紧随 GPU 之后测量，绝对时间可按同窗比较；但它是 CPU
后端，与 GPU 的绝对耗时比较只说明后端差距，不是同一硬件类别。

共享机器存在明显的时间窗口漂移：同一 L24 CUDA 算子在低负载窗口为
`0.121856 s`，在最终同窗为 `0.147801 s`（慢约 `21%`），迭代数与残差
完全不变。因此本文以同窗表为准，不把不同窗口的绝对值混用；第 5 节
KUPL 使用同一最终窗口的数据，第 3.2 节小体积仍标注其原窗口。

GPU 计时前执行 3 s 矩阵乘法预热以抬升 V100 时钟（空闲时 SM 可降至
405 MHz，满载为 1530 MHz）；小体积单次 solve 仅数毫秒，时钟状态对结果
影响显著。即便如此，`16³×32` 及更小体积的 QUDA 结果仍有约
`±10–20%` 的批间波动，表中给三次独立调用的中位数。

## 2. 固定输入与测量口径

| 项目 | 固定值 |
|---|---|
| LIME | `/root/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime` |
| 格点 | `24×24×24×64` |
| mass | `-0.277` |
| csw | `1.160920226`（LIME metadata） |
| seed | `42` |
| 源 | NumPy PCG64 normal，`(4,3,Lx,Ly,Lz,Lt)` complex64，float64 生成后转换 |
| 精度 | c64 |
| 相对残差容差 | `atol = 1e-6`（三个后端一致） |
| 验收阈值 | `2.44140625e-4`（c64 真残差验收） |
| CUDA/QUDA 设备 | 单张 V100-SXM2-32GB；`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=2`，进程内 `cuda:0` |
| KUPL 设备 | 单节点 Intel Xeon E5-2696 v3，18 cores / 1 socket / 1 NUMA node，`OMP_NUM_THREADS=4` |
| 计时 | CUDA：solve 前后 CUDA event（不含 setup）；QUDA：`invert_param.secs` 与外部 wall 同时记录；KUPL：`applyCloverMultigridQcu` 调用前后 `perf_counter`（不含 gauge/Clover setup） |
| 采样 | 1 次 warmup + 3 次计时（小格点为 1 次 warmup + 3 次；表内给中位数） |
| 迭代数 | CUDA `getCloverBistabCgIterationsQcu`；QUDA `invert_param.iter`；KUPL `getCloverBistabCgIterationsQcu` |

### 子体积的来源与限制

`16³×32`、`12³×32`、`8³×16` 是从上述 LIME 的 `x,y,z,t` 下角切出的子体积，
保持相同的 gauge links、mass 和 Csw，再转成 QCU HDF5。子体积在新边界上
**不是周期的**，因此只用于受控的尺寸/并行扩展研究，不能当作新系综或物理结果。
`mass/csw` 元数据随文件保存；源场按子体积的相同 seed 重新生成。

### 重复求解的冷启动

QUDA 的重复 `Dirac.invert` 会分配新的 `LatticeFermion`，而 PyTorch caching
allocator 可能返回上一次解的内存。若不显式关闭，第二次 solve 会以旧解为初值，
表现为迭代数从 14–19 降到 4。本次对 QUDA 显式设置
`use_init_guess = QUDA_USE_INIT_GUESS_NO`、`use_resident_solution = 0`，
保证 warmup 与计时样本都是冷启动。CUDA 和 KUPL 的重复样本迭代数保持恒定，
未观察到热启动。

## 3. 三方性能结果

### 3.1 全格点 `24³×64`

| 后端 / 路径 | solve (s) | 迭代 | 每迭代 (ms) | 真残差 | 相对 QUDA |
|---|---:|---:|---:|---:|---:|
| CUDA QCU `applyCloverBistabCgQcu` | 0.147801 | 18 | 8.211 | 6.709e-7 | 1.060× |
| CUDA QCU fine-only Legacy MG | 0.173372 | 20 | 8.669 | 4.149e-7 | 0.904× |
| QUDA Clover BiCGStab | 0.156680 | 19 | 8.246 | 4.207e-7 | 1.000× |
| KUPL QCU `--restart 0`，4 ranks × 4 thr | 7.842 | 19 | 412.7 | 4.010e-7 | — |
| KUPL QCU `--restart 0`，8 ranks × 2 thr | 7.532 | 19 | 396.4 | 4.011e-7 | — |

本表为同一最终窗口（05:27–05:38）的独立调用中位数：CUDA/QUDA 各 3 次
（每次 1 warmup + 3 计时），KUPL 4 ranks 与 8 ranks 各 5 次、其余 3 次。
CUDA standalone BiCGStab 每迭代 `8.211 ms`，比 QUDA 的 `8.246 ms` 低
`0.43%`；它靠少 1 次迭代把总 wall 拉快 `6.01%`。早期低负载窗口曾测得
每迭代快 `7.17%`，说明 per-iteration 差异受窗口影响，不能当作稳定优势；
总 wall 上 CUDA 在两轮同窗测量中分别快 `5.02%` 和 `6.01%`，方向的
一致性比幅度更可信。fine-only MG 虽真残差略低，但多 2 次迭代且每迭代
多约 `5.1%`，生产路径仍应使用 standalone BiCGStab。

早期低负载窗口的同一组数字为 CUDA `0.121856 s`、MG `0.143909 s`、QUDA
`0.137938 s`、KUPL 4 ranks `5.530 s`；上一轮最终窗口（05:19）为
`0.149015 / 0.171734 / 0.156806 s`。它们保留在
`data/final_window_20261022/` 与 `data/benchmark_three_way_20261022/warm/`
中作为历史对照，不与本表混算。

### 3.2 子体积

| 格点 | CUDA BiCG (s) | CUDA MG1 (s) | QUDA (s) | CUDA/QUDA | KUPL 最优 (s) |
|---|---:|---:|---:|---:|---:|
| `16³×32` | 0.024244 (18) | 0.029848 (20) | 0.042573 (15) | 1.756× | 1.200 (4 ranks) |
| `12³×32` | 0.013212 (18) | 0.015283 (20) | 0.033641 (15) | 2.546× | 0.4239 (2 ranks) |
| `8³×16` | 0.007385 (16) | 0.008947 (16) | 0.028004 (14) | 3.792× | 0.0565 (4 ranks) |

表中 GPU 数字为三次独立调用（每次 1 次 warmup + 3 次计时）的中位数。
QUDA 在小体积上的 `26–43 ms` 下限主要来自单次 solve 的固定开销：
QUDA 自身的 `init`/preamble/workspace 初始化与 host/device 传输，在每次
`invert` 中都会发生（三者中最明显的是 `8³×16` 仍有 `26 ms`）；体积越小，
固定部分占比越高。因此小体积的 CUDA/QUDA 比值不能外推为 kernel 吞吐比。
`24³×64` 是本次唯一适合做 kernel 级吞吐结论的尺寸。

CUDA 在子体积上迭代数几乎不变（16–18），QUDA 为 14–15；两者容差相同，
差异来自 Krylov 轨迹，不是收敛判据不同。

## 4. KUPL 的算法选择：`restart=0` 与 coarse correction

在同一 LIME、同一 4 ranks × 4 threads 配置下：

| KUPL 模式 | 迭代 | solve (s) | 说明 |
|---|---:|---:|---|
| `--restart 0` | 19 | 5.530 | 关闭 coarse hierarchy，optimized parity-Schur BiCGStab |
| `--restart 5` | 47 | 15.60（同窗基线） | 每 5 次 fine 迭代做 coarse correction |
| `--restart 10` | 47 | 15.86 | 每 10 次 |
| `--restart 20` | 47 | 16.71 | 每 20 次 |

该表来自早期低负载窗口（`--restart 0` 的 4×4 后来同窗复测为 `7.842 s`），
只用于比较四种模式的相对关系。各模式在同一窗口内先后运行，迭代数以及
`2.8–3.0×` 的比值不随窗口改变。

三个 coarse-correction 间隔都落在约 47 次迭代，说明回归不是“chunk 太短”
那么简单；当前 E12、1 个 null vector 的粗空间在该组态上改变了 BiCGStab
轨迹并显著增加迭代数。`restart=0` 因此是该组态的 KUPL 生产选择，而不是
一个只用于 benchmark 的开关。

正确性证据：

- 1 rank 独立 NumPy 全 Wilson-Clover 重构真残差：`24³×64` `4.011055e-7`、
  `16³×32` `3.855e-7`、`12³×32` `3.379e-7`、`8³×16` `2.007e-7`，均通过
  c64 验收 `2.441e-4`。
- `restart=0` 与 `restart=5` 的 1-rank 解相对差 `4.3945e-6`、最大绝对差
  `5.78e-5`，证明是同一线性系统的不同 Krylov 路径，不是快速错误解。
- 多 rank 已有独立全算子真残差。`kupl_residual.py` 在每个 rank 用 NumPy
  重构完整 Wilson-Clover 算子，`M psi = C psi - kappa D_bare psi`；通过
  `sendrecv` 交换 ±μ 方向的首/末 site 平面和反向 gauge link，再 Allreduce
  局部残差范数。gauge 的 link 轴与 fermion 的格点轴不同（`LINK_AXES`
  相对 `LATTICE_AXES` 偏移两个 color 轴），这是首个实现得到 `0.708` 错误
  残差的根因，已修正。
- 验证分三层：(1) `4³×8`、单 rank 上分布式实现与原有单 rank 参考相对差
  `1.58e-8`；(2) `8³×16` 上 1/2/4 ranks 的真残差为 `2.0022e-7 /
  2.0052e-7 / 1.9986e-7`，1-rank 与 4-rank 解相对差 `1.176e-7`、最大
  绝对差 `1.45e-6`；(3) `24³×64` 上 4/8/16 ranks 分别为
  `4.0099e-7 / 4.0113e-7 / 4.0105e-7`，4-rank 与 16-rank 解相对差
  `1.403e-7`、最大绝对差 `3.34e-6`。多 rank 报告的 `final_true_residual`
  不再是空值，`residual_source` 明确写为
  `independent distributed NumPy full Wilson-Clover reconstruction with MPI
  halo exchange`。
- 多 rank 解按 `np.unravel_index(rank, grid, order="C")` 拼回完整
  `[spin,color,x,y,z,t]` 后写出 HDF5；4/8/16 ranks 的输出形状一致，
  可直接与 CUDA/QUDA 解比较。

## 5. KUPL 并行性能

### 5.1 `24³×64` 同窗强扩展（4–16 ranks）

`restart=0`、c64、`mass=-0.277`、`csw=1.160920226`；机器有 18 个物理核、
1 个 NUMA node。除 1×4 外，各配置都使用 16 个线程；每组为 3–5 次独立
冷启动调用，表中为 solve wall 中位数（4×4/8×2/8×2-z 各 5 次，其余 3 次）。

| ranks × threads | grid | solve 中位 (s) | 调用样本 (s) | 相对 1r×4 | 相对 1r×16 | 迭代 |
|---|---|---|---:|---|---:|---:|
| 1 × 4 | `1×1×1×1` | 19.229 | 19.187 / 19.229 / 20.296 | 1.000× | — | 19 |
| 1 × 16 | `1×1×1×1` | 9.397 | 9.333 / 9.397 / 9.505 | 2.046× | 1.000× | 19 |
| 2 × 4 | `1×2×1×1` | 11.490 | 11.331 / 11.490 / 11.526 | 1.674× | — | 19 |
| 4 × 4 | `1×4×1×1` | 7.842 | 7.736 / 7.804 / 7.842 / 7.948 / 8.188 | 2.452× | 1.199× | 19 |
| 8 × 2 | `1×4×2×1` | **7.532** | 7.229 / 7.352 / 7.532 / 8.149 / 8.160 | 2.553× | 1.248× | 19 |
| 16 × 1 | `1×4×4×1` | 8.299 | 8.076 / 8.299 / 9.268 | 2.317× | 1.133× | 19 |
| 16 × 1 | `1×4×2×2` | 8.406 | 8.360 / 8.406 / 8.427 | 2.288× | 1.118× | 19 |

以 1×4 为基线的 4 倍并行效率：4×4 为 `61.3%`，8×2 为 `63.8%`，
16×1（`1×4×4×1`）为 `57.9%`。当前最佳生产配置仍是
**8 ranks × 2 threads、`1×4×2×1`**：比单 rank 16 threads 快 `24.8%`，
比 4×4 快 `4.1%`。16 ranks 继续细分后通信/边界占比上升，16×1 比 8×2
慢约 `10–12%`；且网格选择影响变大（`1×4×4×1` 比 `1×4×2×2` 快 `1.3%`），
因此不建议默认上 16 ranks。8×2 的五次样本跨 `7.23–8.16 s`，4×4 跨
`7.74–8.19 s`，两者差距接近批间噪声，实际部署可任选其一；本机倾向 8×2。

分域方向同样有影响：同样 8 ranks，沿 z 分（`1×1×4×2`）中位 `7.803 s`
（5 次：7.496 / 7.591 / 7.803 / 7.860 / 8.376），比沿 y 分
（`1×4×2×1`）慢 `3.6%`。上一窗口测得的 4 ranks 和 16 ranks z 向分域
（`1×1×4×1` `8.157 s`、`1×1×4×4` `8.954 s`）也略慢于 y 向，方向一致：
应优先沿 y 分域并尽量不切 x。

与早期低负载窗口相比，本表绝对值慢约 `40–50%`（例如 4×4 从 `5.530 s`
变为 `7.842 s`），但迭代数和相对排序在两次窗口内基本一致；这是共享
机器负载差异，不应把本表与第 3.2 节的旧窗口数字直接混算。

### 5.2 非 2 幂 rank 数与线程绑定

为了确认 8 ranks 是否只是“2 的幂”带来的假象，又测了 6、9、12、18 ranks
（同一窗口，05:50–05:58，3 次独立调用中位数）：

| ranks × threads | grid（x×y×z×t） | solve 中位 (s) | 使用核数 | 19 次迭代 | 残差校验 |
|---|---|---:|---:|---|---|
| 6 × 3 | `1×2×3×1` | 8.249 | 18 | 是 | 后端门控 PASS |
| 8 × 2 | `1×4×2×1` | 7.532（5.1 同窗） | 16 | 是 | 独立残差 `4.011e-7` |
| 9 × 2 | `1×3×3×1` | 10.640 | 18 | 是 | 后端门控 PASS |
| 12 × 1 | `1×3×2×2` | 7.830 | 12 | 是 | 独立残差 `4.010e-7` |
| 16 × 1 | `1×4×4×1` | 8.299（5.1 同窗） | 16 | 是 | 后端门控 PASS |
| 18 × 1 | `1×3×3×2` | 11.276 | 18 | 是 | 后端门控 PASS |

12-rank 的解也与 8-rank 拼回解逐点比较：`1×3×2×2` 相对差 `1.412e-7`、
最大绝对差 `3.340e-6`；推荐的 `1×4×3×1` 相对差 `1.398e-7`、最大绝对差
`3.346e-6`，与 4 vs 16 ranks 的量级一致。

又测了“更均衡”的网格形状，结果全部不优于沿 y 优先的默认选择：

| ranks | 均衡网格 | solve 中位 (s) | 对比网格 | 对比中位 (s) |
|---:|---|---:|---|---:|
| 4 | `1×2×2×1` | 7.988 | `1×4×1×1` | 7.842 |
| 8 | `1×2×2×2` | 7.956 | `1×4×2×1` | 7.532 |
| 12 | `1×2×3×2` | 7.833 | `1×4×3×1` | 7.796 |
| 12 | `1×12×1×1`（默认启发式） | 8.040 | `1×4×3×1` | 7.796 |
| 12 | `1×3×2×2` | 7.830 | `1×4×3×1` | 7.796 |
| 12 | `1×6×2×1` | 8.776 | `1×4×3×1` | 7.796 |
| 16 | `1×2×2×4` | 8.702 | `1×4×4×1` | 8.299 |
| 16 | `1×2×4×2` | 8.768 | `1×4×4×1` | 8.299 |

切 t 的代价最明显（8 ranks `1×2×2×2` 比 `1×4×2×1` 慢 `5.6%`），因为
local t 减半后半时间索引与边界面都变小，通信次数不变但每次交换更碎。
因此 `default_kupl_grid()` 保持“y → z → t → x、每轴取最大可行因子”的
顺序与测量一致。12 ranks 是唯一的例外：默认 `1×12×1×1`（local y=2）
实测 `8.040 s`，而显式 `1×4×3×1`（local y=6、z=8）为 `7.796 s`、
`1×3×2×2` 为 `7.830 s`，默认比最优慢 `3.1%`。因此 12 ranks 建议显式
`--kupl-grid 1 4 3 1`；其余已测 rank 数沿用默认即可。

结论：rank 数本身不是关键，局部格点形状才是。12 ranks × 1 thread 只使用
12 个核却达到 `7.830 s`，与 8×2 的 `7.532 s` 相差 `4.0%`，适合需要把
部分核留给其它任务的场景；9 ranks（local y=z=8）和 18 ranks（local
y=z=8、t=32）都明显变慢，说明每 rank 局部面太小、边界/通信占比过高时，
增加 rank 数没有收益。8 ranks 的 local `6×12×64` 是当前测得的平衡点。

线程绑定方面有一个必须记录的反直觉结果：在 OpenMPI `--bind-to none`
之外再加 `OMP_PROC_BIND=spread`（或 `close`）与 `OMP_PLACES=cores`，同一
8×2 配置从 `7.2–7.4 s` 恶化到 `65–67 s`（`8.5×`），迭代数与真残差不变。
原因是 KUPL 的求解器由大量短小的 `#pragma omp parallel for` 组成，每次
进入并行区都会新建线程组，设置 `OMP_PROC_BIND` 后 libgomp 在每次建组时
重新绑定线程，短区域的开销被放大到主导程度。生产上应保持默认的
`--bind-to none`、不要设置 `OMP_PROC_BIND`/`OMP_PLACES`；`OMP_WAIT_POLICY=active`
在本机也没有收益（`8.9–9.9 s`）。

### 5.3 子体积（早期窗口）

下表来自早期低负载窗口，仅用于看尺寸趋势，不与 5.1 的绝对值比较：


| 格点 | 1 rank × 4 thr (s) | 多 rank (s) | speedup | efficiency |
|---|---:|---:|---:|---:|
| `16³×32` | 2.719 | 1.200（4 ranks） | 2.266× | 56.6% |
| `12³×32` | 0.777 | 0.4239（2 ranks） | 1.833× | 91.6% |
| `8³×16` | 0.0965 | 0.0565（4 ranks） | 1.708× | 42.7% |

`16³×32` 的 4-rank 效率低于 `12³×32` 的 2-rank，符合通信/边界占比随
局部体积减小而上升的趋势；`8³×16` 的单次 solve 只有几十毫秒，受启动、
barrier 和计时噪声影响最大，效率数字应视为量级而非精确值。

覆盖范围：`24³×64` 上已测 1、2、4、8、16 ranks（单节点、18 核、1 个
NUMA node），并比较了 `1×4×1×1`、`1×4×2×1`、`1×4×2×2`、`1×4×4×1`、
`1×1×4×1`、`1×1×4×2`、`1×1×4×4` 七种分域；完整配置都做了独立分布式
真残差与解拼回一致性检查（第 4、7 节），计时样本则显式跳过残差以缩短
测量时间。仍未覆盖：跨节点 MPI（InfiniBand/以太网）、
多 NUMA node 与 CPU binding、16 ranks 以上，以及 KUPL CHROMA layout
在奇数 local dim 下的分域（该布局要求每个 local dim 为偶数，本机的
rank 网格选择受此约束）。

## 6. CUDA 侧本轮工作与已穷尽方向

本轮对 CUDA 的实质改动是测量方法而非 kernel：

- 给 `qcu_bistabcg.py`、`qcu_multigrid.py` 增加 `--warmup/--repeats`，
  报告改为中位数；这消除了首次调用包含的初始化误差，使 CUDA 的稳定值
  从早期单次 `0.132–0.135 s` 收敛到 `0.121856 s`。
- 与 QUDA 对照时显式关闭 QUDA 热启动（见 2.3），保证两边都是冷启动。
- 结论：CUDA standalone BiCGStab 已是当前最佳路径；fine-only MG 的
  20 次迭代与额外生命周期开销没有收益。

已在前序阶段证伪、本轮未重复的方向：CUDA Graph、双流 overlap、
6×6 编译期特化、`r/x` 与残差范数融合、E12 V-cycle/FGMRES（均慢于
fine-only）、KUPL 5→4 Allreduce、KUPL y-face SIMD、时间轴 4-rank 分域、
错误的核心绑定。

### GPU 架构：原生 sm_60 与 sm_70

本机有三种卡：2×P100-PCIE-16GB（sm_60）、1×V100-SXM2-32GB（sm_70）。
早期发布物是把 32 个 sm_60 cubin 加 sm_60 PTX 打包，V100 每次进程启动都要
JIT PTX，P100 可直接运行；PyTorch 2.10+cu128 只支持 sm_70 及以上，使 P100
在 Python 侧完全不可用。本轮处理：

- 新增原生 sm_70 构建：`data/libqcu-sm70/libqcu.so`，
  `-DQCU_CUDA_ARCHITECTURES="70-real;70-virtual"`，32 个 sm_70 cubin。
- 新增原生 sm_60 构建：`data/libqcu-sm60/libqcu.so`，
  `-DQCU_CUDA_ARCHITECTURES="60-real;60-virtual"`，32 个 sm_60 cubin。
- `_runner.libqcu_environment()` 在启动后端子进程前按
  `nvidia-smi --query-gpu=compute_cap` 选择目录，并把该目录前置到
  `LD_LIBRARY_PATH`；Cython 桥没有 RPATH，因此第一个 `libqcu.so` 生效。
  `--libqcu auto|sm60|sm70|default` 可自动选择或强制 A/B。
- 物理卡到逻辑卡的映射会反向翻译：`cuda_environment()` 先把
  `cuda:2` 映射成 `CUDA_VISIBLE_DEVICES=2`、进程内 `cuda:0`；架构选择器
  必须再通过 `CUDA_VISIBLE_DEVICES` 找回物理 2 号卡，否则会把逻辑 0 当成
  P100。该错误在首次集成时确实发生过，已修复并加了正/负测试。

A/B 口径见 6.1：同一 V100、同一 `24³×64` 输入，分别强制
`--libqcu sm60`（sm_60 PTX 在 V100 上 JIT）与 `--libqcu sm70`（原生
SASS）。若两者差异落在批间噪声内，则说明 PTX JIT 不是该算子的瓶颈，原生
sm_70 的价值主要在消除进程启动 JIT、固定 SASS 和便于后续按架构调优。
P100 上 sm_60 求解 `8×8×16×16`
的 solve wall 为 `0.0425 s`（含加载），V100 sm_70 为 `0.0149 s`；同一窗口
的 A/B 见 6.1。两者残差 `2.846712e-7` 与 `2.846721e-7`，解逐位一致
（相对差 `0.0`）。强制 `--libqcu sm70` 在 P100 上按预期报
`no kernel image is available for execution on the device`，证明选择器
不是“碰巧能跑”。P100 的 Python 栈使用独立 `data/venv-p100`
（torch 2.4.1+cu121，支持 sm_60），不改动系统 PyTorch 2.10。

QUDA 侧同样补齐了 sm_60 构建：从 QUDA 源码
`-DQUDA_GPU_ARCH=sm_60 -DQUDA_MPI=ON -DQUDA_QMP=OFF -DQUDA_QIO=OFF`
构建并安装到 `data/quda-sm60-install`（356 个 sm_60 cubin）。
`quda_environment()` 按 compute capability 选择 `QUDA_INSTALL`
（`data/quda-sm70-install` 或 `data/quda-sm60-install`），显式
`QUDA_INSTALL` 优先，`--quda-variant auto|sm60|sm70|default` 可覆盖。
P100 上 `8×8×16×16` 的 QUDA solver 为 `0.090304 s`（14 次迭代，真残差
`8.578e-7`），V100 同题 `0.073592 s`（`8.617e-7`）；两卡解相对差
`2.643e-8`、最大绝对差 `3.696e-7`。这是 P100/sm_60 的首次可用 QUDA
数值结果，说明 sm_60 安装在 P100 上不仅“能加载”，而且给出与 V100 一致的解。P100 仍需用 `data/venv-p100`
运行 PyQUDA（torch 2.4.1），否则 PyTorch 2.10 本身会在 sm_60 上失败。

### 6.1 同窗 A/B：sm_60+PTX vs 原生 sm_70

在 V100、`24³×64`、同一输入、3 次独立调用（每次 1 warmup + 3 计时）上
强制 `--libqcu sm60` 与 `--libqcu sm70`：

| 变体 | 中位 solve (s) | 三次调用样本 (s) | 迭代 | 真残差 |
|---|---:|---|---:|---:|
| sm_60 库（V100 上 JIT） | 0.146966 | 0.146966 / 0.148645 / 0.146222 | 18 | 6.709e-7 |
| 原生 sm_70 | 0.147531 | 0.147531 / 0.149677 / 0.146775 | 18 | 6.709e-7 |
| 默认 sm_60+PTX 包 | 0.150286 | 单次调用 | 18 | 6.709e-7 |

sm_70 与 sm60 的中位数差 `0.38%`，小于同一变体内三次调用的散布
（约 `±1%`），因此在本算子上没有可测的 SASS 收益。该窗口的绝对时间比
第 3 节早期窗口慢约 `20%`（`0.122 s → 0.147 s`），原因是共享机器的
后台负载不同；A/B 的相对结论不受影响，跨窗口绝对数不应直接混用。

### 6.2 KUPL：LTO 构建

KUPL 默认构建已是 `-O3 -DNDEBUG -march=native`，但各 kernel 分处不同源文件，
跨翻译单元的小函数调用无法内联，且共享库默认走 PLT。新增独立构建：

```bash
cmake -S cpp/kupl/qcu -B cpp/kupl/qcu/build-lto -DCMAKE_BUILD_TYPE=Release \
      -DKUPL_BUILD_TESTS=ON \
      -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -flto -fno-semantic-interposition"
cmake --build cpp/kupl/qcu/build-lto -j 18
ctest --test-dir cpp/kupl/qcu/build-lto -j 4
```

8 ranks × 2 threads、`1×4×2×1`、`24³×64`，默认库与 LTO 库交替运行各 5 次：

| 库 | solve 中位 (s) | 5 次样本 (s) | 迭代 | 真残差 |
|---|---:|---|---:|---:|
| `build/libqcu.so`（默认） | 7.556 | 7.392 / 7.430 / 7.556 / 7.664 / 7.971 | 19 | 4.011e-7 |
| `build-lto/libqcu.so` | **7.196** | 7.042 / 7.067 / 7.196 / 7.285 / 7.515 | 19 | 4.011e-7 |

LTO 快 `4.8%`，且两库的 8-rank 拼回解逐位一致（相对差 `0.0`）、真残差
相同；LTO 构建的 CTest `29/29` 通过。收益来自跨文件内联与去掉库内
PLT 间接调用，不需要改算法或数据布局。生产脚本通过
`--kupl-lib lto`（或直接设 `PYQCU_KUPL_LIB=<path>`）选择该库；
`--kupl-lib default` 保持原构建，默认值不变。

注意 5.1/5.2 的 KUPL 数字都来自默认库，若改用 LTO 预计再快约 `5%`；
本文没有把 LTO 数字混入跨后端总表，以免读者误以为算法或硬件变了。

## 7. 正确性与回归

| 门禁 | 结果 |
|---|---|
| CUDA 独立真残差（L24，BiCG） | 6.709e-7 PASS |
| CUDA 独立真残差（L24，MG1） | 4.149e-7 PASS |
| QUDA 真残差（L24） | 4.180e-7 PASS |
| KUPL 1-rank 独立真残差 | 2.007e-7–4.011e-7 PASS |
| KUPL 2/4 ranks 独立真残差（`8³×16`） | 2.0052e-7 / 1.9986e-7 PASS |
| KUPL 4/8/16 ranks 独立真残差（`24³×64`） | 4.0099e-7 / 4.0113e-7 / 4.0105e-7 PASS |
| KUPL 12 ranks 独立真残差（`24³×64`，`1×3×2×2` / `1×4×3×1`） | 4.010e-7 / 4.011e-7 PASS |
| KUPL 8 vs 12 ranks 解（`24³×64`） | 相对差 `1.412e-7` / `1.398e-7`，最大绝对差 `3.340e-6` / `3.346e-6` |
| `OMP_PROC_BIND` 绑定回归 | 迭代数/收敛状态不变，wall 从 `7.2–7.4 s` 变为 `65–67 s` |
| 分布式残差 vs 单 rank 参考（`4³×8` 单元测试） | 相对差 `1.58e-8` |
| KUPL 1 vs 4 ranks 解（`8³×16`） | 相对差 `1.176e-7`，最大绝对差 `1.45e-6` |
| KUPL 4 vs 16 ranks 解（`24³×64`） | 相对差 `1.403e-7`，最大绝对差 `3.34e-6` |
| CUDA sm_60（P100）vs sm_70（V100）解 | 相对差 `0.0`（逐位一致），残差 `2.846712e-7` / `2.846721e-7` |
| QUDA sm_60（P100）vs sm_70（V100）解（`8×8×16×16`） | 相对差 `2.643e-8`，最大绝对差 `3.696e-7`；残差 `8.578e-7` / `8.617e-7` |
| CUDA/KUPL 跨后端解（`8×8×16×16`，4 ranks） | 相对差 `4.537e-7`，最大绝对差 `5.980e-6` |
| QCU/QUDA 跨后端解（`8×8×16×16`） | 相对差 `1.540e-6`，最大绝对差 `2.009e-5` |
| sm_70 强制运行在 P100 | 按预期 FAIL（`no kernel image`），选择器负测试通过 |
| 所有报告 `status` | PASS |
| KUPL CTest（默认构建） | 29/29 passed |
| KUPL CTest（LTO 构建） | 29/29 passed |
| KUPL 默认 vs LTO 解（8 ranks，`24³×64`） | 相对差 `0.0`（逐位一致），真残差同为 `4.011e-7` |
| 脚本语法 | `python -m py_compile` 通过 |

## 8. 便捷调用脚本

按“生产”和“测量”拆分为四个脚本，共享 `_common.py`、`_runner.py` 和
`prepare_case.py`；原 `bench_three_way.py` 保留为弃用别名，只转发参数。

| 脚本 | 职责 |
|---|---|
| `prepare_case.py` | 从 LIME 生成全格/子体积 gauge 和 seed-42 源，写 `case_manifest.json`（路径、形状、SHA256） |
| `solve_case.py` | 生产单次求解：`--backend cuda/quda/kupl`，写解 HDF5、后端 JSON、`last_solve_*.json` |
| `compare_three_way.py` | 性能测量：GPU 时钟预热、warmup、重复计时、中位数、相对 QUDA 倍率 |
| `bench_three_way.py` | 弃用兼容别名，转发到 `compare_three_way.py` |

### 8.1 生产求解

```bash
source ./env.sh

# 准备输入（可重复执行；已有文件会校验 mass/csw/seed/形状）
python pyqcu/testing/simple/prepare_case.py \
    --lat 24 24 24 64 --out-dir data/production_L24

# CUDA QCU：V100 为 PCI 顺序 2 号卡，自动选 sm_70
python pyqcu/testing/simple/solve_case.py --backend cuda \
    --lat 24 24 24 64 --device cuda:2 --out-dir data/production_L24

# KUPL QCU：8 ranks x 2 OpenMP threads，1x4x2x1 网格
python pyqcu/testing/simple/solve_case.py --backend kupl \
    --lat 24 24 24 64 --kupl-ranks 8 --kupl-threads 2 \
    --out-dir data/production_L24

# QUDA
python pyqcu/testing/simple/solve_case.py --backend quda \
    --lat 24 24 24 64 --device cuda:2 --out-dir data/production_L24
```

默认产物为 `solution_<lat>_<backend>_c64.h5`、`report_<backend>.json`、
`last_solve_<backend>.json`。生产路径不做 GPU 预热、不重复计时；退出码
仅在 `status == PASS` 时为 0。KUPL 默认计算分布式独立真残差，时间不够时
可显式 `--skip-independent-residual`。跨后端比较时三份解布局相同，可直接
逐点作差；`prepare_case` 的 manifest 保证 gauge/source 是同一份输入。

`--device cuda:N` 在 `CUDA_VISIBLE_DEVICES` 未设置时按 PCI 顺序把物理 N 号
卡映射为进程内 `cuda:0`；若用户已设置 `CUDA_VISIBLE_DEVICES`，不再改写。
`--libqcu auto|sm60|sm70|default` 控制原生构建；`auto` 按 compute capability
选择，P100 用 sm_60、V100 用 sm_70。

### 8.2 性能对照

```bash
# 全格点，三后端
python pyqcu/testing/simple/compare_three_way.py \
    --lat 24 24 24 64 --device cuda:2 \
    --kupl-ranks 4 --kupl-threads 4

# 同 mass/Csw 的子体积
python pyqcu/testing/simple/compare_three_way.py \
    --lat 16 16 16 32 --device cuda:2 \
    --kupl-ranks 4 --kupl-threads 4

# 只跑 GPU 两条路径，快速回归
python pyqcu/testing/simple/compare_three_way.py \
    --lat 8 8 8 16 --backends cuda,quda --device cuda:2 \
    --warmup 1 --repeats 2
```

输出 `data/benchmark_three_way/{summary.json,summary.md}`，逐后端 JSON 在
同一目录。计时模式下 KUPL 跳过独立残差以缩短样本，解仍经过后端
fail-closed 门控；需要独立残差时用 `solve_case.py`。

底层脚本仍可单独调用：

```bash
python pyqcu/testing/simple/qcu_bistabcg.py --lat 24 24 24 64 ...
python pyqcu/testing/simple/qcu_multigrid.py --levels 1 ...
python pyqcu/testing/simple/quda_bistabcg.py --warmup 1 --repeats 3 ...
OMP_NUM_THREADS=4 mpirun -np 4 ... kupl_multigrid.py \
    --levels 1 --restart 0 --output data/sol.h5 ...
```

## 9. 复现数据

| 路径 | 内容 |
|---|---|
| `data/benchmark_three_way_20261022/warm_reps/{lat}/{r1,r2,r3}/` | 子体积三次独立调用原始 JSON；表 3.2 的中位数来源 |
| `data/benchmark_three_way_20261022/warm/` | 全格点及子体积的预热后单次调用，含 `24³×64` 的 CUDA/QUDA 结果 |
| `data/benchmark_three_way_20261022/cuda/` | CUDA BiCG / MG1 的 JSON（含 warmup 与 3 次样本） |
| `data/benchmark_three_way_20261022/quda_bicg_*.json` | QUDA JSON（含 solver/wall 样本、真残差） |
| `data/benchmark_three_way_20261022/kupl/sub2/` | KUPL 子体积与 L24 并行样本（含迭代数） |
| `data/benchmark_three_way_20261022/kupl/scaling_restart0/` | 早期 `restart=0` 强扩展样本 |
| `data/benchmark_three_way_20261022/gauge_*.h5` | 子体积 QCU gauge |
| `data/benchmark_three_way_20261022/source_*.h5` | 共享 PCG64 源 |
| `data/benchmark_three_way_smoke/summary.*` | 统一脚本早期端到端 smoke 结果 |
| `data/benchmark_three_way_final_smoke/summary.*` | 定稿脚本 `--backends all` 在 `8³×16` 上的最终 smoke，四条路径均 PASS |
| `data/bench_split_test/summary.*` | 拆分后 `compare_three_way.py` 的 `8³×16` smoke，四条路径均 PASS |
| `data/prod_split_test/` | 拆分后生产脚本 smoke（`8×8×16×16`）：`prepare_case` manifest、CUDA sm_70/P100 sm_60 解、QUDA sm_70/sm_60 解、KUPL 4-rank 解，以及 sm_70 强制跑 P100 的负测试报告 |
| `data/build-qcu-sm60/`、`data/libqcu-sm60/` | 原生 sm_60 构建与可加载库 |
| `data/build-qcu-sm70/`、`data/libqcu-sm70/` | 原生 sm_70 构建与可加载库 |
| `data/venv-p100/` | P100 专用 Python 环境（torch 2.4.1+cu121，支持 sm_60） |
| `data/benchmark_three_way_20261022/kupl/parallel_*.json` | 4/8/16-rank KUPL 并行与分布式残差报告 |
| `data/benchmark_three_way_20261022/kupl/parallel_sol_*r.h5` | 4/8/16-rank 拼回的完整解，用于 rank 间一致性检查 |
| `data/window2_20261022/gpu/`、`data/window2_20261022/kupl/` | 第 3.1、5.1 节最终同窗（05:27–05:38）的 CUDA/QUDA 三次调用与 KUPL 1–16 ranks 全部原始 JSON（含 4×4/8×2 的残差校验运行） |
| `data/window3_binding_20261022/` | 第 5.2 节 `OMP_PROC_BIND`/`OMP_PLACES`/`OMP_WAIT_POLICY` 绑定实验原始 JSON |
| `data/window4_ranks_20261022/` | 第 5.2 节 6/9/12/18 ranks 扩展、12-rank 独立残差，以及 8/12-rank 拼回解 |
| `data/window5_lto_20261022/` | 第 6.2 节默认库 vs LTO 库交替 5 次的原始 JSON，以及 LTO 残差/拼回解 |
| `data/window6_three_way_lto/` | `compare_three_way.py --backends all --kupl-lib lto` 的端到端 smoke（`warmup=0`，各 1 次）：四条路径 PASS；QUDA `0.1997 s` 含首次调用初始化，只作联调证据，不作性能结论 |
| `data/window7_gridshape_20261022/` | 第 5.2 节均衡网格形状的原始 JSON |
| `cpp/kupl/qcu/build-lto/` | KUPL LTO 构建目录与 CTest 产物 |
| `data/final_window_20261022/r{1,2,3}/` | 上一轮最终窗口（05:19）的三次独立调用，作为窗口漂移对照；绝对值不与本表混算 |
| `data/ab_sm_20261022/{sm60,sm70,default}_r{1,2,3}/` | 第 6.1 节 V100 上 sm_60+PTX 与原生 sm_70 的同窗 A/B 原始 JSON |
| `data/quda-sm60-install/`、`data/build-quda-sm60-full/` | QUDA 原生 sm_60 安装（356 cubins）与构建目录 |

所有 `data/` 路径均为本地运行产物（gitignored），不随仓库提交。

当前构建产物 SHA256：

| 产物 | SHA256 |
|---|---|
| `cpp/cuda/qcu/libqcu.so`（sm_60+PTX 默认） | `c78fca294974d799935817e4c3ed993b5a8aa1a37a8b5ca2c1dcd681d2eae964` |
| `data/libqcu-sm60/libqcu.so` | `7dd24ec883dc36204bd6c536eb417ede5d1b385191a63281205fd588a1844712` |
| `data/libqcu-sm70/libqcu.so` | `9f2d4b0f5ad06d791b734ae74dbc4a34d04bfc999b6b2664595471e6ed2fd9a0` |
| `cpp/kupl/qcu/build/libqcu.so`（默认） | `677a9c42a827b5bb61ab333a5c67be19c218dc34e517449b47da7bdda47f2226` |
| `cpp/kupl/qcu/build-lto/libqcu.so`（LTO） | `db00aa785fa136f280bf58709853b1aecd490857d140ea412a21911d24f7cfef` |

## 10. 限制与后续

1. 子体积在接缝处非周期；不能用于物理量或有限体积外推。
2. 多 rank KUPL 已有独立全算子真残差，并覆盖 1、2、4、6、8、9、12、16、
   18 ranks；但只在单节点、
   18 核、1 个 NUMA node 上验证。跨节点 MPI（InfiniBand/以太网）、多 NUMA
   binding、以及 16 ranks 以上的强扩展仍需在多节点机器复测；本机没有第二
   个 NUMA node，无法给出跨 NUMA 结论。KUPL 的 CHROMA layout 还要求每个
   local dim 为偶数，这限制了可用的 rank 网格。
3. CUDA 已能分别在 P100（sm_60）和 V100（sm_70）上运行并逐位一致，但仍
   没有多卡并行数据：本机只有一张 V100，两张 P100 未做多进程/多卡扩展；
   PyQCU 的多 GPU 路径不在本轮范围内。
4. QUDA 的三方对照仍只在 V100（sm_70）上完成。P100 的 sm_60 构建已完成
   （`data/quda-sm60-install`，356 cubins），并在 `8×8×16×16` 上验证了可运行
   且解与 V100 一致（相对差 `2.643e-8`）；但 P100 上的完整三方性能对照
   （含 KUPL 与全格点）未做，因此三方结论仍以 V100 为准。
5. KUPL 的 LTO 构建只在单节点、8 ranks × 2 threads、`24³×64` 上做了完整
   A/B 与 CTest；更小格点、12/16 ranks 与跨节点尚未复测。默认库仍是生产
   默认值，LTO 通过 `--kupl-lib lto` 显式启用。
6. 本轮未提交、未打 tag、未推送；所有结果与脚本保留在工作区。
