# PyQCU dev100 至当前工作区进度总结

状态：`userinterlude`（长期任务进行中，尚未形成新 tag）

生成日期：2026-10-22（Asia/Shanghai）

## 总结范围

上一标签为 `dev100`，peeled commit `1640349d`。本文覆盖：

| 范围 | Git 对象 | 说明 |
|---|---|---|
| 起始 | `dev100` / `1640349d` | 最近本地 tag |
| 已提交 | `04bb23ba` | 保存 dev99 至 dev100 的进度总结与 tag 日志 |
| 已提交 | `5a21a5e6` | 第一轮生产 MG、CUDA/KUPL 优化与 benchmark 工具整合 |
| 当前 | `5a21a5e6` + 未提交工作区 | CUDA/KUPL 后续迭代、真实生产报告与负结果 |

`5a21a5e6` 与 `origin/main` 对齐。当前没有暂存内容，也没有 tag、commit 或 push 覆盖本轮未提交优化。

## 结论摘要

本阶段以真实 LIME 组态上的 Legacy Clover-MG 为主、BiCGStab 为辅，完成了 CUDA QCU 与 KUPL QCU 的并行调试和优化。结论如下：

| 项目 | 当前结论 |
|---|---|
| CUDA BiCGStab | 已稳定快于 QUDA；Stage25 中位 `0.120462 s`、18 iterations，独立真残差 `6.694679e-7` |
| CUDA fine-only Legacy MG | 当前生产最优路径；Stage22 中位 `0.13295 s`、20 iterations、真残差 `4.1257200e-7` |
| CUDA 真实 E12 V-cycle/MG | 当前粗空间不足，所有实测模式均慢于 fine-only；`n_vcycles=0` 仍是合理生产选择 |
| KUPL Legacy MG | 4 ranks × 4 threads 最新稳定中位 `15.072466 s`、47 iterations、native residual 约 `9.138e-7` |
| KUPL 并行瓶颈 | 4-rank 的 fine Wilson-Clover rank-local apply 占 `73.2%`；通信与 canonical 转换不是首要瓶颈 |
| 正确性 | CUDA strict tier1、Clover c64/c128、ABI 与 KUPL CTest 均通过；最新 KUPL CTest 为 `29/29` |
| 当前缺口 | CUDA 更大的 coarse space 受 33-tensor 二次显存增长限制；KUPL fine scalar boundary 仍需继续优化 |

## 固定生产输入与设备口径

| 项目 | 固定值 |
|---|---|
| LIME | `/root/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime` |
| LIME SHA256 | `a49773fa3742d310f4ea162a932cf559253aeab708bac08a8dd1f6847df4294b` |
| 格点 | `24×24×24×64` |
| mass | `-0.277` |
| csw | `1.160920226` |
| seed | `42` |
| RHS SHA256 | `24da8b13665dc2a6e0503142bbb9f3baa0f4264d86541b6fca40a297b7c38a5b` |
| CUDA 正式设备 | 单 V100；`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=2`，进程内为 `cuda:0` |
| KUPL 正式设备 | 单节点，4 ranks × 4 OpenMP threads，y-only `1×4×1×1` 分域 |

禁止并发运行 CUDA 正式 benchmark。P100 不被当前 PyTorch 构建支持，不能作为正式 CUDA benchmark 设备。

## 代码与工程改动

### CUDA Clover 算子与 BiCGStab

- 在 Legacy Clover BiCGStab 单 rank 路径中，将 dslash、标量更新与向量更新固定在 main stream，去掉每轮多流同步开销。
- 融合 `r = s - ωt` 与 `x += αp + ωs`，减少完整向量遍历和 kernel launch。
- 将 `t^H s` 与 `t^H t` 合并为确定性 `bistabcg_dot_pair_complete`；最后完成 block 按固定 partial 顺序归约，不使用非确定性浮点 atomicAdd。
- 新增 `bistabcg_dot_complete`，将 `rho`、`rtv`、检查范数从 cuBLAS 调用迁移到同构的确定性分块树形归约。
- 单 rank 收敛检查改为每 2 次迭代同步一次；18 次生产迭代保持，奇数迭代收敛时最多多执行一次。
- 对 `clover_ee_inv` 与 `clover_oo` 做一次 6×6 结构探测；只有全部 off-block 严格为零时才启用 6×6 内核，否则回退完整 12×12，保留通用外部 Clover 支持。
- 融合 `D_ee^-1` 与 Wilson hopping 的若干访存步骤，减少整向量 D2D copy 和中间 kernel。

### CUDA Legacy MG

- fine-level MG 与 Clover BiCGStab 共享 6×6 Clover 快速路径和确定性点积实现。
- fine Schur fast path 现使用：
  - 45 次单点积 `bistabcg_dot_complete`；
  - 20 次成对点积 `bistabcg_dot_pair_complete`（20 次外层迭代）。
- 保留 adaptive correction gate。当前真实 E12 单 null-vector 粗空间在收敛前不会产生净 wall 收益，因此生产 solve 为 fine-only Schur BiCGStab。
- local stencil gather 使用一次展平周期窗口和批量 `torch.gather`；生产窗口从 `W=10` 收紧到经边界回归验证的 `W=8`。
- 大格 33-tensor stencil 使用 `BatchedLocalSchur + build_stencil_local`，并保留 `PYQCU_MG_LOCAL_STENCIL=0/1` 与 `PYQCU_MG_BUILD_PROGRESS=1` 诊断入口。
- 增加 coarse level 的成对归约、single-dot、结构门控、残差检查和完整指标输出。

### KUPL QCU

- Clover onsite 矩阵与逆矩阵按当前 gamma 基的两个稠密 `6×6` block 计算，同时 ABI 仍写回完整 `12×12` 并显式清零 off-block。
- Legacy MG full-site 向量进入 Clover/Schur 前转换为 parity-canonical 布局，避免近单位测试无法发现的非平凡 κ 错误。
- coarse dslash 使用 512-site tile 预解析 32 个邻接流，并按 site-major coefficient stream 连续累加。
- canonical/full 转换改为 half-site 遍历并同时写出两个时间 parity。
- coarse correction 采用 fail-closed 真残差验收；correction 不变好、变差或非有限时撤销，并以单个长 BiCGStab chunk 完成。
- BiCGStab 将相邻 `(v,v)` 与 `(v,s)` 合并为一个四实数组 `MPI_Allreduce`，将每轮集合归约次数从 6 降到 5。
- 新增/扩展 1/2/4-rank、c64/c128、legacy MG、strict MG、failure-mode 与显式 OpenMP 线程测试；最终 CTest 为 `29/29`。

### Benchmark、缓存与诊断工具

- CUDA benchmark 支持真实 LIME/HDF5、固定 RHS、cache-hit、参考解开关、真残差验证及 setup/solve 分段计时。
- KUPL simple MG benchmark 支持 coarse cache、独立 residual 开关、固定网格与 JSON/MAD 报告。
- 新增 null-vector 批量尺寸、stencil local batch、coarse cache 往返和 MG 几何回归。
- 新增 33-tensor cache 与 Strict cache 的数学、布局、内存和生命周期说明，并同步到 solver skill。

## 性能结果

### 当前三方主结果

| 路径 | 中位耗时 | iterations | 真/验收残差 | 口径 |
|---|---:|---:|---:|---|
| CUDA QCU Clover BiCGStab Stage25 | `0.120462 s` | 18 | `6.694679e-7` | ABBA，2 warmup + 10 steady/轮 |
| QUDA Clover BiCGStab | `0.134009 s` | 19 | `5.6942744e-7` | 同 LIME、mass/csw、RHS |
| CUDA QCU Legacy MG Stage22 | `0.13295 s` | 20 | `4.1257200e-7` | cache-hit，fine-only |
| CUDA QCU MG lifecycle Stage22 | `0.356707 s` | 20 | `4.1257200e-7` | 含 init/end 与全算子残差探针 |
| KUPL Legacy MG | `15.072466 s` | 47 | native 约 `9.138e-7` | 4 ranks × 4 threads，restart=5、deflate |
| QUDA MG nvec24 | `28.007541 s` | 3 outer | `5.1234262e-7` | 与 QCU/KUPL dof12 语义不等价 |

CUDA BiCGStab 相对 QUDA wall 快 `10.11%`；每迭代 `6.692 ms` 对 `7.053 ms`，快 `5.12%`。QUDA MG nvec24 与 dof12 的内存、null-space 和算法语义不同，不能计算公平加速倍率。

### 关键收益链

| 阶段 | 优化 | 结果 |
|---|---|---|
| Stage1-7 | CUDA D2D copy、`rx`、lazy workspace、Clover/hopping 融合 | 多轮生产 MG 与 BiCGStab 降低，累计从 `0.1872 s` 级 BiCG 降至约 `0.133 s` |
| Stage13 | CUDA local stencil gather + `W=8` | cold stencil `598.107 -> 323.745 s`，总 setup 降低 46.85% |
| Stage13 | KUPL 512-site coarse tile 等 | 4-rank MG `94.108 -> 29.942 s`，降低 68.18% |
| Stage16 | CUDA 成对点积；KUPL 成对 Allreduce | CUDA BiCG 降低 2.18%；KUPL 4-rank MG 降低 4.19% |
| Stage17 | CUDA 6×6 Clover 结构门控（MG） | MG backend `0.175150 -> 0.147050 s`，降低 16.04% |
| Stage18 | 6×6 门控扩展到 standalone BiCG | BiCG `0.159155 -> 0.132760 s`，降低 16.58% |
| Stage20 | 单 rank 每 2 次迭代检查 | BiCG `0.127819 -> 0.125522 s`，降低 1.80% |
| Stage21 | MG fine 成对归约 | MG backend `0.14125 -> 0.13700 s`，降低 3.01% |
| Stage22 | MG fine 单点积 | MG backend `0.13760 -> 0.13295 s`，再降低 3.38%；相对 Stage20 累计降低 5.88% |
| Stage25 | standalone BiCG 单点积 | BiCG `0.123998 -> 0.120462 s`，降低 2.85% |

### KUPL 强扩展与 c128

固定 c64 生产配置的三次强扩展：

| ranks | grid | 中位 | speedup | efficiency | iterations |
|---:|---|---:|---:|---:|---:|
| 1 | `1×1×1×1` | `46.407 s` | 1.000x | 100.0% | 45 |
| 2 | `1×2×1×1` | `27.790 s` | 1.670x | 83.5% | 47 |
| 4 | `1×4×1×1` | `15.581 s` | 2.978x | 74.5% | 47 |

最新 4-rank 同窗 A/B 基线进一步为 `15.072466 s`，但未在同窗重跑 1/2-rank，因此不能与新基线混算 efficiency。

c128 结果：

| 配置 | 范围 | 中位 | iterations | 残差 |
|---|---|---:|---:|---:|
| `applyCloverMultigridQcu --restart 0` | 全格点 `24³×64`，4 ranks × 4 threads | `10.282 s` | 21 | native recomputed `6.079e-8` |
| `applyCloverBistabCgQcu` | 全格点 `24³×64`，4 ranks × 4 threads | `11.311 s` | 23 | 独立 `6.96241e-8` |

`restart=0` 关闭 coarse hierarchy，是全格点 c128 parity-Schur BiCGStab，不是 c128 coarse V-cycle 性能。

### 真实 CUDA MG 模式筛选

同一进程、同一 LIME/cache 的结果：

| mode | lifecycle 中位 | backend | iterations | 真残差 |
|---|---:|---:|---:|---:|
| fine-only baseline | `0.365086 s` | `0.132445 s` | 20 | `4.125720e-7` |
| BiCG + deflate V-cycle | `0.401633 s` | `0.172463 s` | 20 | `2.666153e-7` |
| FGMRES+MG, CG | `1.160491 s` | `0.93003 s` | 16 FGMRES | `7.181618e-7` |
| FGMRES+MG, MR | `0.832060 s` | `0.60251 s` | 10 FGMRES | `4.818365e-7` |
| FGMRES+MG, Chebyshev | `1.366174 s` | `1.12978 s` | 19 FGMRES | `4.743437e-7` |
| CA-GCR+MR | `1.911062 s` | `1.68461 s` | 28 CA-GCR | `8.093534e-7` |

deflate 的 wall 增加约 `9.96%`，却仍执行 20 次 fine iterations。单次 V-cycle 约 `40 ms`，其中 coarse residual check 约 `30 ms`。

### 冷构建

| 配置 | setup | null-space | stencil |
|---|---:|---:|---:|
| Stage22 baseline | `356.179 s` | `26.524 s` | 主导，约 `352.962 s` 累计阶段 |
| Stage25 single-dot | `353.635 s` | `26.677 s` | 主导，约 `350.328 s` 累计阶段 |

总 setup 降低 `0.71%`，但 null-space 差异处于噪声，不能宣称 null-vector 构建显著提速。

## 正确性与回归

| 门禁 | 最新结果 |
|---|---|
| CUDA strict tier1 | CPU `20 passed`、cuda-strict `11 passed`、fused-FGMRES `3 passed` |
| 直接 Clover | c64/c128 共 `8 passed` |
| QCU ABI | `header_to_pxd=[]`、`header_to_pyx=[]`、`pxd_extra=[]`、`coverage_missing=[]` |
| Legacy MG breakdown | c64 1 passed；c128 single/pair smoke finite |
| KUPL CTest | `29/29 passed` |
| c64 MG 真残差 | `4.1257200e-7` |
| c64 BiCG 独立真残差 | `6.694679e-7` |
| c128 KUPL 独立真残差 | `6.96241e-8` |
| `git diff --check` | PASS |
| form audit | 仅既有 `docs/progress_dev99_to_1640349d_20261021.md` 的 `TAG_NAME_IN_PATH` finding |

## 已撤销或证伪的方向

| 方向 | 结果 | 处理 |
|---|---|---|
| Stage19 `r/x` 与残差范数融合 | 仅约 `0.23%`，第二样本回到噪声上界 | 完整撤销 |
| CUDA Graph 4-iteration segment | capture/instantiate 无法跨 solve 摊销，生产点变慢 | 完整撤销 |
| `D_oo` 与 hop-chain 双流 overlap | 受带宽竞争和额外 combine kernel 限制，慢约 `2.7%` | 完整撤销 |
| CUDA 6×6 编译期资源特化 | kernel 仍为 88 registers；30 steady 仅 `0.19%`，lifecycle 反慢 | 完整撤销 |
| E12 V-cycle/FGMRES 模式 | 全部慢于 fine-only | 保留 adaptive gate，当前不启用 |
| KUPL 5→4 Allreduce | 4-rank 慢 `10.67%` | 完整撤销 |
| KUPL y-face SIMD | 4-rank 慢 `4.60%`，MAD 扩大约 `11.6x` | 完整撤销 |

## 风险、限制与未完成项

1. CUDA 生产 MG 实际为 fine-only Schur BiCGStab，`n_vcycles=0`。不能把 Stage21/22 的收益写成 coarse V-cycle 加速。
2. E12 单 null vector 的粗空间不足。按当前 33-tensor 精确形状，E24 cache 约 `5.22 GB`、E48 约 `18.86 GB`；E12 local-stencil setup 已有约 `29.75 GB` peak，32 GiB V100 上直接扩大 E 不可行。
3. CUDA 正式性能只在单 V100 上完成；当前机器没有可用于正式基准的第二张兼容 GPU。
4. KUPL 最新 `15.072466 s` 与 `15.581 s` 强扩展数据来自不同时间窗口/脚本配置，不能混算倍率。
5. KUPL 尚无全格点 c128 coarse V-cycle；`restart=0` 结果不能替代 coarse cache 性能。
6. KUPL 跨节点 InfiniBand、跨 NUMA 大规模 rank 放置尚未覆盖。
7. 当前长期任务未提交、未打新 tag、未 push；未提交工作区包含代码、测试、技能、日志和文档。
8. 既存 form finding 与当前任务无关，但会使 form audit 整体返回非零。

## 下一阶段建议

1. CUDA：实现 compressed 或 matrix-free coarse operator，先解决 null-vector 数量增加后的二次显存和 setup workspace，再重新评估真实 V-cycle。
2. CUDA：评估 multi-rank 点积/归约合并，但必须以实际延迟、18/47 次迭代轨迹和生产真残差为门禁，不能仅按集合调用次数推断收益。
3. KUPL：分析 scalar boundary 的 rank-face 分布、页局部性和内存访问；不要再直接扩大 SIMD lane。
4. KUPL：在相同脚本和同时间窗口重跑 1/2/4-rank，将强扩展基线统一到 `15.072466 s` 这一代。
5. 文档与归档：若用户要求形成稳定里程碑，再显式提交并创建新 tag；本轮不做未经授权的 Git 操作。

## 主要证据入口

### 阶段日志

- `logs/v202610210909.txt` 至 `logs/v202610211129.txt`：第一轮 CUDA/KUPL MG 优化。
- `logs/v202610211808.txt`：null-vector BiCGStab 与 KUPL onsite。
- `logs/v202610211925.txt`：CUDA gather/W8 与 KUPL coarse tile。
- `logs/v202610212107.txt`：CUDA/KUPL 成对归约。
- `logs/v202610212129.txt` 至 `logs/v202610212151.txt`：6×6 门控与撤销融合。
- `logs/v202610212220.txt` 至 `logs/v202610212355.txt`：Stage20-22 收敛分块与 MG 单/成对归约。
- `logs/v202610220021.txt`：Stage23 并行瓶颈诊断。
- `logs/v202610220044.txt`：Stage24 真实 MG 与 KUPL fine hotspot。
- `logs/v202610220104.txt`：Stage25 BiCG 单点积与冷构建。

### 性能与测试数据

- `data/production_mg_20261021/`：CUDA/QUDA 生产 JSON、ABBA、strict 和冷构建证据。
- `data/cpp/kupl/qcu/build/kupl_scaling_20261021/`：KUPL c64 强扩展与 c128 生产验证。
- `data/cpp/kupl/qcu/build/kupl_efficiency_20261021/`：2/4-rank 归因与 Allreduce A/B。
- `data/cpp/kupl/qcu/build/fine_hotspot_20261022/`：fine dslash 热点与 SIMD A/B。
- `data/logs/lime_mg_cache_stage14_w8/`：CUDA 生产 33-tensor cache。
- `data/logs/kupl_mg_cache_20261021/full_wilson_clover_L24x24x24x64_E12_nv1_v2.h5`：KUPL 生产 cache。

### 当前构建产物

| 产物 | SHA256 |
|---|---|
| CUDA `cpp/cuda/qcu/libqcu.so` | `c78fca294974d799935817e4c3ed993b5a8aa1a37a8b5ca2c1dcd681d2eae964` |
| KUPL `cpp/kupl/qcu/build/libqcu.so` | `677a9c42a827b5bb61ab333a5c67be19c218dc34e517449b47da7bdda47f2226` |

## 复现命令摘要

```bash
source ./env.sh

# CUDA
bash ./build.sh
cd pyqcu/testing
pytest .

# KUPL
cmake -S ../cpp/kupl/qcu -B ../cpp/kupl/qcu/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../cpp/kupl/qcu/build -j
ctest --test-dir ../cpp/kupl/qcu/build --output-on-failure
```

正式 24³×64 benchmark 必须固定本文“生产输入与设备口径”中的 LIME、mass、csw、seed 和 RHS；报告必须同时保留 iterations、独立/native 真残差、warmup/steady 样本数和库 SHA256。

## 交接结论

当前可继续沿两条独立方向推进：

1. CUDA 的下一项实质工作不是再次微调 E12 fine path，而是降低 coarse operator 内存表示成本并恢复真实 V-cycle 的竞争力。
2. KUPL 的下一项实质工作是围绕 fine scalar boundary 和 NUMA-local memory access 做可解释的 A/B，而不是继续合并集合调用或扩大 SIMD lane。

所有未提交改动和未归档日志均保留在工作区，未执行提交、tag、push 或删除操作。
