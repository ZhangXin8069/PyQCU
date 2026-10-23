# QCU MultiGrid 重构与跨实现验收（2026-10-23）

## 结论

- `Legacy-MultiGrid` 是 QCU 默认求解器，也是默认 debug/优化对象；
  `Strict-MultiGrid` 仅在显式选择时启用。
- Torch CPU/CUDA、CUDA-QCU、KUPL-QCU 的 Legacy/Strict 共用统一
  `setup/solve/close` 控制面；Legacy 与 Strict 继续保留各自合法的 coarse asset
  格式，不互相伪装或静默转换。
- `16^4` 快速矩阵全部通过独立 full Wilson/Clover residual 门。Strict 三种实现均为
  23 次外层迭代；Legacy 为 79/87/73 次，解向量相对差均小于 `1e-5`。
- CUDA Legacy 三层/四层旧失败已定位并修复。`24^3x64` 单 rank 和 2-rank 的 2/3/4
  层均在 full residual `2.45e-7` 到 `2.61e-7` 范围内通过。
- KUPL Legacy 已修复 NaNs、递归 V-cycle 的粗层尺寸错误、odd-Schur RHS 构造错误和
  correction 失败后的状态污染。correction 变差时回滚并显式 fallback 到已验证的
  parity-Schur BiCGStab。

## 固定验收口径

- c64；`mass=0.05`、`csw=1.0`、`tol=1e-6`、2 levels、block `2^4`、`nvec=12`；
  大格控制组使用对应真实 LIME 的 mass/csw。
- 所有正式计时为 trace-off；QUDA 在进程启动前设置 `QUDA_ENABLE_TUNING=0`。
- MPI rank 数只允许 2 的正整数次幂。
- 数值接受同时要求 backend/native gate 与独立 full residual `<=1e-6`。
- Legacy 跨实现还要求解向量 `relative L2 <=1e-4`；其 c64 Krylov 迭代差允许不超过
  `max(2, ceil(0.20 * min(iterations)))`。Strict 要求绝对迭代差 `<=2`。

## 16^4 QCU 矩阵

输入 hash 与 cache identity 记录在
[`summary.json`](../logs/qcu_mg_matrix_16cubed_20261023/summary.json) 和
[`parity_contract.json`](../logs/qcu_16cubed_quick_20261023/parity/parity_contract.json)。

| ID | backend / variant | iterations | steady median | independent full residual | status |
|---|---|---:|---:|---:|---|
| TL | Torch Legacy (CUDA) | 79 | `2.5032 s` | `7.406e-7` | PASS |
| TS | Torch Strict (CUDA) | 23 | `12.4956 s` | `6.577e-7` | PASS |
| CL | CUDA Legacy | 87 | `0.0564 s` | `3.051e-7` | PASS |
| CS | CUDA Strict | 23 | `0.1950 s` | `6.543e-7` | PASS |
| KL | KUPL Legacy | 73 (5 MG + 68 fallback) | `4.0008 s` | `3.362e-7` | PASS |
| KS | KUPL Strict | 23 | `13.9101 s` | `6.535e-7` | PASS |
| QUDA | aligned MG | 12 | `0.4041 s` | `4.744e-7` | PASS |

Torch CPU reference: Legacy 79 iterations、`39.625 s`、`7.099e-7`；Strict 23
iterations、`793.699 s`、`6.584e-7`。CPU run 不用于 GPU 加速比。

相对 Torch 解向量的 `relative L2`：

- Legacy：CUDA `8.07e-6`；KUPL `8.06e-6`。
- Strict：CUDA `2.82e-7`；KUPL `3.35e-7`。
- QUDA 对 Torch Strict：`1.28e-5`。

同一 `16^4` 输入的 KUPL Legacy 2-rank smoke 也通过：wall `4.5765 s`，独立
distributed full residual `3.179e-7`。该入口不暴露 C++ 内部 `final iterations`，
因此只记录完成状态、时间和残差，不把 `null` 迭代字段解释为零。

## BiCGStab 边界对照

BiCGStab 是判断 MG 是否值得使用的边界控制，不进入 MG-to-MG speedup。

| backend | iterations | effective seconds | independent full residual | status |
|---|---:|---:|---:|---|
| Torch odd-Schur (CUDA) | 67 | `1.5945 s` | `7.245e-7` | PASS |
| CUDA QCU | 68 | `0.0441 s` | `4.175e-7` | PASS |
| KUPL QCU, 4 ranks | 68 | `1.7306 s` | `3.276e-7` | PASS |
| QUDA | 70 | `0.1059 s` | `4.726e-7` | PASS |

在本 `16^4` c64 输入上，各 MG 路径的 steady solve 均慢于同侧 BiCGStab：
Torch `2.5032/1.5945 = 1.57x`、CUDA `0.0564/0.0441 = 1.28x`、
KUPL `4.0008/1.7306 = 2.31x`、QUDA `0.4041/0.1059 = 3.82x`（MG/BiCGStab）。
Strict 虽把外层迭代降到 23 次，仍不足以抵消递归 V-cycle 和层级成本。该小格批次是
统一性与正确性矩阵，不构成“MG 应替代 BiCGStab”的生产性能结论，也不得外推到 L24/L32。

## L24 Legacy 深度与 MPI

`24^3x64` c64、`nvec=12`、block `2^4`：

| ranks / levels | iterations | solve seconds | native full residual | independent full residual | status |
|---|---:|---:|---:|---:|---|
| 1 / 2 | 20 | `0.2151` | `2.336648e-7` | `2.604745e-7` | PASS |
| 1 / 3 | 20 | `0.9324` | `2.335826e-7` | `2.605036e-7` | PASS |
| 1 / 4 | 20 | `0.9222` | `2.336648e-7` | `2.604566e-7` | PASS |
| 2 / 2 | 20 | `1.0505` | `2.342344e-7` | `2.453464e-7` | PASS |
| 2 / 3 | 20 | `6.9275` | `2.342240e-7` | `2.453817e-7` | PASS |
| 2 / 4 | 20 | `8.0346` | `2.342462e-7` | `2.453665e-7` | PASS |

3/4 层修复了正确性，但没有迭代收益；`MultigridDepthWarning` 对 L24 3L/4L 和
`16^4` 2L 的几何风险会明确告警。Legacy 生产默认仍为 2 层。

## 关键根因

1. CUDA Legacy 递归 V-cycle 返回父粗层后恢复最细层 volume，父层 post-smoother
   使用了错误站点数；已改为按父层 geometry 工作。
2. CUDA Legacy 原先把 Schur residual 当停机判据；现在 Schur 候选必须再过 collective
   full residual gate，近阈值带逐迭代检查。
3. adaptive slope gate 对 c64 归约噪声敏感；已替换为确定性的 post-residual rollback。
4. KUPL Legacy 递归 `cycle()` 曾把 coarse-sized correction 当 fine correction 累加；
   已加入显式临时 coarse correction 和 `prolong()`。
5. KUPL odd-Schur 路径曾直接复制 full RHS 的 odd block；现在调用
   `clover_bistabcg_prepare` 构造正确 Schur RHS。
6. KUPL correction 拒绝后曾继续使用受污染 Krylov 状态；现在回滚并 fallback 到已验证的
   fine BiCGStab，报告 `fallback=fine_bicgstab`。
7. Torch 内部 `_MultigridHierarchy` 的统一结果报告原先假定存在 `residual_scope`；
   现在按 strict/compact/full 自动推导。

## 非 root 与输入可移植性

- 本次数值输入、cache、null vectors 与 strict assets 均为 `0644`。
- 输出目录、报告、解文件均可通过 CLI 指定；临时文件原子发布且使用进程 umask。
- `MPI --allow-run-as-root` 只在调用者 euid=0 时自动添加，普通用户路径不依赖它。
- LIME provenance 默认解析为 `~/data/<name>`，并允许 `--full-lime` 显式覆盖；不再在
  benchmark 代码中硬编码 `/root`。

本容器无法执行完整的非 root 进程级 smoke：`/root` 权限为 `0700`，普通用户无法进入
当前工作树。已完成的替代证据是代码/CLI 路径审计、所有输入/输出 `0644` 检查、原子
发布和 root-option 条件化；部署到普通用户可读写的安装目录后仍需执行一次真实 smoke。

## 大格边界与未验证项

- 大格 Strict 行继续 `SKIPPED`，原因是
  `strict_initialization_time_unacceptable`，不是数值失败。
- L32 两级 Legacy cache 构建、单 rank 与 2-rank 结果在本文件对应批次完成后追加。
- 本报告不宣称 Torch Python 实现适合生产热路径；其作用是跨平台 algebra oracle。

## 证据

- 六项矩阵：[`summary.json`](../logs/qcu_mg_matrix_16cubed_20261023/summary.json)
- 跨实现 parity：[`summary.json`](../logs/qcu_16cubed_quick_20261023/parity/summary.json)
- L24 单 rank：[`qcu_mg_final_20261023`](../logs/qcu_mg_final_20261023)
- L24 2-rank：[`qcu_mg_final_20261023`](../data/logs/qcu_mg_final_20261023)
- BiCGStab：[`qcu_bicgstab_final_20261023`](../logs/qcu_bicgstab_final_20261023)
- Strict skip：[`strict-skip.json`](../logs/qcu_l24x64_mpi_20261023/strict-skip.json)
