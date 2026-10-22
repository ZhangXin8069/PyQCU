# Legacy CUDA、Strict CUDA 与 KUPL 多重网格状态核对（2026-10-22）

状态：基于现有 JSON/stdout 的静态核查完成；本次未运行 GPU、CUDA、MPI 或新的数值 benchmark。

相关弃用审计与历史测试入口见
[MultiGrid 弃用与历史残留审计](multigrid_deprecation_audit_20261022.md)。

## 1. 结论

| 项目 | 结论 | 证据类型 |
|---|---|---|
| Legacy CUDA build | `8³×16`、`12³×24`、`16³×32`、`24³×64` 共 4/4 `PASS` | 实测 |
| Legacy CUDA solve | 同一 4 个尺寸共 4/4 `PASS`，稳态真残差均低于 `1e-6` | 实测 |
| Strict CUDA null-vector prep | 同一 4 个尺寸共 4/4 `status=created` | 实测 |
| Strict CUDA solve | `8³×16`、`12³×24`、`16³×32` 共 3/4 `PASS`；`24³×64` 为空 stdout、无 JSON，属于未完成或缺失，原因未记录 | 实测与推导 |
| KUPL build/solve/null-prep | 目标目录没有任何 KUPL JSON 或 stdout 记录，不能报告或比较 KUPL 数值结果 | 实测目录清单后的缺失判定 |
| Legacy coarse correction | 四组 solve 的 stdout 全部记录 `n_vcycles=0`、`coarse_iters=0`、`coarse_solve=0.0 ms`；只能证明使用了 Legacy MG 入口和 fine-level 求解，不能证明实际执行了 coarse V-cycle | 实测字段与语义推导 |
| Strict cycle count | JSON 记录 `restart=20`、hierarchy setup 成功和独立真残差，但没有 V-cycle 计数器；不能把 `restart=20` 解释为 20 次 V-cycle | 实测字段与语义边界 |
| KUPL `restart=0` | 该配置关闭 coarse hierarchy，是 fine-level parity-Schur BiCGStab，不是 coarse V-cycle；本次没有对应 KUPL 记录 | 代码/文档语义与缺失记录 |

本报告只讨论 `c64`。没有把不同精度、不同 coarse 自由度或不同求解算法合并成一个“平均性能”或“MG 加速比”。

## 2. 证据分级

- **实测**：直接来自 JSON 字段或 stdout 行，例如状态、耗时、迭代数、残差、哈希和内存。
- **由实测推出**：由已记录值计算的比值、由缺失文件/空文件判定的完成状态，或由多个已记录字段共同确定的执行边界。
- **未验证**：现有记录没有覆盖的数值质量、硬件身份、库版本、MPI rank、失败原因或算法内部计数。报告不会用外部运行结果补齐这些项。

## 3. 已解析输入

目标目录共有 29 个 JSON/stdout 记录，全部纳入静态解析：

| 记录族 | 文件数 | 解析结果 |
|---|---:|---|
| `legacy_build_*.json` / `.stdout` | 4 + 4 | 4 个 JSON 均合法；每个 stdout 的最终 JSON 与同名 sidecar 完全一致 |
| `legacy_solve_*.json` / `.stdout` | 4 + 4 | 4 个 JSON 均合法；每个 stdout 的最终 JSON 与同名 sidecar 完全一致，并额外提供 `PROF_SECTIONS` 执行计数 |
| `null_prep_*.json` | 4 | 文件是“3658 行 C++ 控制台日志 + 第 3659 行开始的 JSON 对象”，不是整文件合法 JSON；四个尾段 JSON 均可解析 |
| `strict_solve_*.json` / `.stdout` | 3 + 4 | 3 个 JSON 均合法且与 stdout 内嵌 JSON 一致；`strict_solve_24x24x24x64.stdout` 为 0 字节且无 JSON |
| `qcu_mg_8x8x8x16_reference.json` / `.stdout` | 1 + 1 | stdout 末尾内嵌 JSON 与 sidecar 完全一致；作为辅助 Legacy 记录单列，不并入四尺寸主表 |

解析边界：

- `qcu_mg_8x8x8x16_solution.h5` 与目录中的其他 HDF5 文件未作为本报告的数值输入。
- 四个 `null_prep_*.log` 均不在任务指定的 JSON/stdout 范围内，且均为 0 字节，未用于判断成功。
- `qcu_mg_8x8x8x16.stdout` 的最终 JSON 与一行 C++ 日志粘连在同一行，但 JSON 子串和 sidecar 可完整解析且内容一致。

## 4. 完成状态矩阵

| 格点 | Legacy build | Strict null-prep | Legacy solve | Strict solve | KUPL |
|---|---|---|---|---|---|
| `8³×16` | `PASS` | `created` | `PASS` | `PASS` | 无记录 |
| `12³×24` | `PASS` | `created` | `PASS` | `PASS` | 无记录 |
| `16³×32` | `PASS` | `created` | `PASS` | `PASS` | 无记录 |
| `24³×64` | `PASS` | `created` | `PASS` | **缺失**：空 stdout，无 JSON | 无记录 |

Strict 没有独立的“build JSON”。其 hierarchy/setup 计时包含在 `strict_solve_*.json` 的 `hierarchy_setup_seconds` 中，并明确排除外部 null-vector 文件创建。KUPL 的 build、null-prep 和 solve 则均无任何目标目录记录。

## 5. 固定输入与 provenance

Legacy 四尺寸 build/solve 与 Strict 三尺寸成功 solve 的记录使用同一组 mass、Csw、精度和收敛设置；Strict `24³×64` 没有 solve 记录可核对：

| 项目 | 值 |
|---|---|
| precision | `c64` |
| mass | `-0.277` |
| csw | `1.160920226` |
| kappa（由 mass 推出） | `0.13430029546065003` |
| solver relative tolerance | `1e-6` |
| max iteration | `200` |
| Legacy/Strict source seed | `42` |
| source rule | `first-prefix slice of full PCG64 source` |
| gauge provenance | `same-LIME prefix subvolume; nonperiodic seams` |

同一尺寸的 Legacy build、Legacy solve、Strict solve 记录中，gauge/source 对象逐字段相同。SHA256 前缀如下，完整值保留在对应 JSON：

| 格点 | gauge SHA256 前缀 | source SHA256 前缀 |
|---|---|---|
| `8³×16` | `bac143c8daca` | `6e7d74ce809d` |
| `12³×24` | `73120268e7df` | `01e740b1e740` |
| `16³×32` | `99a7d0c334dd` | `18f0585bb470` |
| `24³×64` | `2a22d6730daf` | `47ec61ba5024` |

`qcu_mg_8x8x8x16_reference.json` 没有记录 gauge/source 哈希，只有相同格点、mass、Csw 和 `comparison_valid=true`。因此不能仅凭该辅助 JSON 证明它使用了与主表完全相同的外部输入文件。

## 6. Legacy CUDA build

四组 build 均为 `PASS`。`mg_asset_build_seconds` 与 `mg_init_seconds` 相同，范围为 null vectors 加 33-tensor Galerkin stencil 构建，不包含前面的 fine prepare：

| 格点 | fine prepare (s) | MG asset build (s) | `nv_iters` | coarse 配置 | cache bytes | device-wide peak delta (bytes) | cache SHA256 前缀 |
|---|---:|---:|---:|---|---:|---:|---|
| `8³×16` | 0.3077126460 | 5.5693930390 | 1 | E12, `2×2×2×2` | 14,454,784 | 432,013,312 | `b397295bcfb8` |
| `12³×24` | 0.4395924130 | 32.6015038910 | 1 | E12, `2×2×2×2` | 73,160,704 | 1,591,738,368 | `46e3cfc4ddd0` |
| `16³×32` | 0.5477590490 | 316.9439288570 | 1 | E12, `2×2×2×2` | 231,215,104 | 4,722,786,304 | `68a549574e64` |
| `24³×64` | 2.4054449510 | 323.9064836890 | 1 | E12, `2×2×2×2` | 1,560,678,400 | 30,400,966,656 | `e1b9a613af6f` |

这些结果证明 cache 和 33-tensor asset 已生成并记录哈希。它们不证明 null vectors 的数值质量，也没有给出独立的 null-vector residual。

## 7. Strict CUDA null-vector preparation

四个尾段 JSON 均为 `status=created`。输出布局为 `[nvec,spin,color,x,y,z,t]`，`nvec=12`，dtype 为 `complex64`：

| 格点 | 创建时间 UTC | 输出 shape | identity SHA256 前缀 | output logical SHA256 前缀 |
|---|---|---|---|---|
| `8³×16` | `2026-10-22T11:19:48Z` | `[12,4,3,8,8,8,16]` | `e59a2a6c5d9f` | `411d2a6dc93a` |
| `12³×24` | `2026-10-22T11:20:13Z` | `[12,4,3,12,12,12,24]` | `986013a4f67e` | `63a6f610bfd8` |
| `16³×32` | `2026-10-22T11:17:26Z` | `[12,4,3,16,16,16,32]` | `a8e4d1a8d1b8` | `6db76d7566e0` |
| `24³×64` | `2026-10-22T11:28:27Z` | `[12,4,3,24,24,24,64]` | `2355a5871615` | `ce68d0ed4472` |

共同 metadata：

- `schema=pyqcu.canonical-full-nullvec/v1`。
- odd blocked `lonv` 来自对应尺寸的 Legacy cache，经 Clover zero-rhs block elimination 重构 full vector。
- `reconstruction.version=2`，`csw=1.160920226`，`sigma=0.1`。
- 内存策略为 `stream-one-vector`，每个 blocked/full vector 只驻留 1 个。
- provenance 记录 Git revision `ed19695db997fa668e8febf5d8c87fb4b830ee0c`、Python `3.10.12` 和工具 `prepare_fair_nullvec.py`。

这里的 `output_sha256` 是逻辑 HDF5 dataset 的摘要；Strict solve 中的 `null_vector_sha256` 是整文件摘要，两者作用域不同，不能直接比较数值是否相等。三组成功 Strict solve 的 null-vector 路径也分别与对应 prep 输出路径一致。

`status=created` 只证明文件创建协议完成并记录了输出摘要，不证明 null vectors 满足任何粗空间质量门槛。现有 JSON/stdout 没有 null-vector residual、Galerkin 误差或 coarse-level 收敛诊断。null-prep 总耗时也未记录；输入指纹中的 `elapsed_seconds` 只是计算哈希的时间，不能当作 preparation 运行时间。

## 8. Legacy CUDA solve

四组 solve 均先加载已有 cache，再执行 2 次 warmup 和 5 次计时求解。表中稳态值为最后 5 次的 median/MAD；`cache_load_seconds` 和运行时 `mg_init_seconds` 不计入该 steady solve：

| 格点 | 状态 | cache load (s) | runtime init (s) | 迭代数（2 warmup + 5 steady） | steady median (s) | steady MAD (s) | true residual | 实际 coarse 计数 |
|---|---|---:|---:|---|---:|---:|---:|---|
| `8³×16` | `PASS` | 0.0121954570 | 0.0034591150 | `16, 28×6` | 0.0115697170 | 0.0006167369 | `8.357704e-8` | `n_vcycles=0`, `coarse_iters=0` |
| `12³×24` | `PASS` | 0.0877928000 | 0.0034384840 | `20, 12×6` | 0.0115448180 | 0.0004783260 | `1.763145e-9` | `n_vcycles=0`, `coarse_iters=0` |
| `16³×32` | `PASS` | 0.1982781420 | 0.0063193370 | `20, 12×6` | 0.0233167160 | 0.0001331100 | `6.101726e-9` | `n_vcycles=0`, `coarse_iters=0` |
| `24³×64` | `PASS` | 1.3113937420 | 0.0223907800 | `20, 12×6` | 0.1111170500 | 0.0010069610 | `2.706802e-8` | `n_vcycles=0`, `coarse_iters=0` |

残差来源是 Legacy C++ MG 日志中的 full-operator true residual。其相对值均低于 `1e-6`，因此现有记录支持 `PASS`。

关键边界：JSON 带有 `coarse_dof=12`、`coarse_restart=5` 和两级配置，但每次 solve 的 stdout 都显示 `PROF_SECTIONS ... vcycle=0.0ms n_vcycles=0 coarse_solve=0.0ms ... coarse_iters=0`。这说明配置存在不等于执行了 coarse correction。现有记录不能支持“Legacy coarse V-cycle 已完成”或“V-cycle 带来加速”的结论。

## 9. Strict CUDA solve

三组成功记录均使用 12 个 full null vectors，Strict coarse dof 为 `2*nvec=24`。求解执行 2 次 warmup 和 5 次计时；表中 steady 为最后 5 次：

| 格点 | 状态 | hierarchy setup (s) | runtime init (s) | iterations | backend residual（7 次相同） | independent true residual | steady median (s) | steady MAD (s) | null file SHA256 前缀 |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| `8³×16` | `PASS` | 5.6466774420 | 0.0066785000 | `10×7` | `1.3611337636e-4` | `3.3461674320e-7` | 0.0240973420 | 0.0002530150 | `a2383e841558` |
| `12³×24` | `PASS` | 27.1686966390 | 0.0086314180 | `11×7` | `3.0913823866e-4` | `3.3157110124e-7` | 0.0461964710 | 0.0007987480 | `76f1a8a524cc` |
| `16³×32` | `PASS` | 80.1018036520 | 0.0143250260 | `11×7` | `7.4568769196e-4` | `4.2494960439e-7` | 0.1042182120 | 0.0010211600 | `5d67299ac077` |
| `24³×64` | 缺失 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |

`backend_final_residuals` 与 `final_true_residual` 的归一化/测量路径不同，不能直接互相替代。最终状态按 Python full Wilson/Clover reconstruction 的独立 true residual 判定，三组均低于 `1e-6`。`restart=20` 是 Strict 求解器的外层 Krylov restart 参数；JSON 没有 V-cycle 次数，因此不能把它解释为 coarse-cycle 计数。

Strict `mg_init_seconds` 是 hierarchy setup 加 runtime binding，并明确排除 null-vector 文件创建。它不能与 Legacy build 的“null vectors + 33-tensor stencil”总时间直接相加或相减为公平 cold-start 结论。

## 10. 辅助 `8³×16` Legacy 参考记录

`qcu_mg_8x8x8x16_reference.json` 是另一套 schema/驱动产生的辅助记录，不并入主四尺寸表：

| 字段 | 值 |
|---|---|
| backend / solver | `qcu` / `applyCloverMultigridQcu` |
| device | `cuda:0`, `Tesla V100-SXM2-32GB`, 34,359,345,152 bytes |
| levels / dof / MG grid | `2` / `[12,12]` / `[2,2,2,2]` |
| iterations | `16` |
| warmup / samples | `0` / `1` |
| wall | `0.0434222412109375 s` |
| independent true residual | `2.2189919945958536e-7` |
| estimated throughput | `20.8641847757 GFLOP/s`，lower-bound estimate |
| profile | `fine_iter=9.6 ms`, `n_vcycles=0`, `coarse_iters=0`, `vcycle=0.0 ms` |
| status | `PASS`, `converged=true`, `comparison_valid=true` |

该记录也显示“配置了 2 级 V-cycle 入口但本次执行没有粗层活动”。不过它只有一个 0-warmup 样本，schema 和主表不同，不能与主 Legacy 8³ solve 的 2-warmup/5-steady 数据合并，也不能用它替代主表。

## 11. 可比性

主表内 Legacy 与 Strict 的共同点：

- 同一尺寸的 gauge/source 路径、SHA256、mass 和 Csw 相同。
- 都是 `c64`，tol 都是 `1e-6`，max iteration 都是 `200`。
- 都记录了 2 次 warmup 和 5 次 steady solve。
- steady 计时都排除了各自的前置 build/setup。

主 Legacy/Strict 的直接 elapsed 对照如下。比值只是“已记录路径的稳态 wall 差异”，不是算法效率、单位算力或 coarse-cycle 加速比：

| 格点 | Legacy steady median (s) | Strict steady median (s) | Strict / Legacy |
|---|---:|---:|---:|
| `8³×16` | 0.0115697170 | 0.0240973420 | 2.083 |
| `12³×24` | 0.0115448180 | 0.0461964710 | 4.001 |
| `16³×32` | 0.0233167160 | 0.1042182120 | 4.470 |
| `24³×64` | 0.1111170500 | 缺失 | 不可计算 |

这组比值不能写成 Strict-MG 相对 Legacy-MG 的算法加速/减速倍率，原因如下：

- Legacy 使用 E12 配置但实际 `n_vcycles=0`，没有观察到 coarse correction。
- Strict 使用 `nvec=12`，coarse dof 为 24，且 JSON 未记录实际 cycle count。
- 两者迭代数语义不同；Legacy 计时包含 fine-level BiCGStab 轨迹，Strict 计时包含 Strict hierarchy 的求解路径。
- 两套记录的运行时间戳大多缺失，不能证明绝对耗时来自同一低负载窗口。

`24³×64` Strict 没有可比较数据。KUPL 没有 build/null-prep/solve 记录，因此也不存在可计算的 Legacy/KUPL 或 Strict/KUPL 数值比较。

## 12. 硬件与环境 provenance

已实测：

- 主 JSON 均记录逻辑设备 `cuda:0`。
- Legacy/Strict 四尺寸记录中的设备总量均为 `34,359,345,152 bytes`，空闲量会随工作负载变化。
- 辅助 `qcu_mg_8x8x8x16` 明确记录设备名 `Tesla V100-SXM2-32GB`。
- null-prep 记录绑定 Git revision `ed19695db997fa668e8febf5d8c87fb4b830ee0c` 和 Python `3.10.12`。

未验证：

- 四尺寸 Legacy/Strict 记录没有 `device_name`，所以不能仅凭相同显存总量证明所有记录来自同一张物理 V100。
- 没有驱动版本、CUDA toolkit、编译器、`libqcu.so` SHA256、PyTorch 版本或主机 CPU/NUMA 信息。
- Legacy/Strict 主记录没有 Git revision 或运行时间戳；无法确认四个尺寸是否来自同一代码二进制或同一时间窗口。
- Legacy/Strict JSON 没有 MPI rank 字段。辅助 qcu stdout 的 grid 为 `1×1×1×1`，Strict 主记录的 rank 数仍需以调用命令或其他外部证据确认。
- 没有 GPU 时钟状态、预热策略、功耗、温度或共享负载记录；小格点绝对耗时尤其可能受这些因素影响。

## 13. 明确未验证项

1. **KUPL 全链路**：目标目录没有 KUPL build、null-prep 或 solve JSON/stdout。不能声称 KUPL 在任一尺寸完成、失败或达到某个残差。`restart=0` 也不能填补 coarse V-cycle 数据。
2. **Strict `24³×64`**：只有 0 字节 stdout，没有 JSON；无法判断是未启动、被终止、资源不足还是求解失败。
3. **Legacy coarse V-cycle**：四组主 solve 和辅助 8³ 记录都显示 `n_vcycles=0`；没有一次实际 coarse correction 的数值证据。
4. **Strict coarse cycle count**：JSON 没有 cycle 计数器，只有 `restart=20` 和收敛结果。
5. **null-vector 质量**：没有 residual、Galerkin 误差、coarse convergence 或跨 rank 质量验证。
6. **cold-start 总时间**：Legacy asset build、Strict null prep、Strict hierarchy setup 的计时边界不同，且 null-prep 总时间未记录。
7. **二进制和环境身份**：缺少库哈希、编译器、CUDA、驱动和硬件模型字段。
8. **分布式行为**：没有可核对的 rank 数或 MPI 网格，因此这些记录不能作为多 rank 正确性/性能结论。
9. **文件内容复验**：本任务没有重新读取 HDF5/gauge/source/cache/null-vector 的 dataset 内容；哈希和 shape 仅核对为记录所声明的一致元数据。

## 14. 最终判定

现有证据能够支持：

- Legacy CUDA 四尺寸的 build 和 fine-level solve 已成功完成。
- Strict CUDA 四尺寸的 null-vector 文件已创建，三尺寸 solve 通过独立真残差门。
- 在同一输入和 c64 设置下，已成功的 Strict 路径稳态 wall 高于 Legacy 路径；该差异不可解释为 coarse-MG 算法加速比。

现有证据不能支持：

- Strict `24³×64` 已完成。
- Legacy 主表实际执行了 coarse V-cycle。
- KUPL 任一 build、null-prep 或 solve 已完成。
- KUPL `restart=0` 是 coarse V-cycle。
- 任一跨后端加速比、扩展效率或物理结论。
