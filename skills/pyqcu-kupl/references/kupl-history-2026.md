# KUPL history (2026-10 – 2026-11)

## 2026-10-29 MPI coarse-cache builder

Legacy 33-tensor cache 的规范 MPI 入口是
`pyqcu.testing.simple.mpi.build_full_mg_cache`。它不能退回父目录的 rank-0
串行脚本。实现边界如下：

- 每个 rank 用全局 process grid 生成自己的 fine gauge/Clover 和 null-vector slab；
- 所有 rank 用 HDF5 MPI-IO 把 blocked `lonv=[E,12,X,bx,Y,by,Z,bz,T/2,bt]`
  owned slab 写入暂存 cache，再按 `(2,4,6,8)` 四个 coarse 轴读取本地周期 patch；
  不得把多 GiB `lonv` 经 root gather/scatter，也不能把它当成 axes-last-4 张量；
- KUPL local-stencil API 在全局 MPI 作业内需要 `1x1x1x1` 私有 context；C++ geometry
  以 `params[_NODE_SIZE_]=1` 显式创建单 rank communicator，不能强制用
  `MPI_COMM_WORLD` 解释 patch grid；
- 每个 rank 只输出自己的 coarse slab，四个张量均经 HDF5 MPI-IO 写 owned slab；
  rank 0 只生成属性和 manifest，再由所有 rank 回读各自 hyperslab 与本地 slab 比对；
- `--build-device cuda` 的所有 rank 均可使用 GPU patch stencil；patch LatticeSet 的
  private singleton communicator 必须贯穿 Clover/coarse halo。报告应要求
  `stencil_device=cuda:*`，出现 `cpu` 视为未启用当前 GPU-native 路径；
- `lonv` 时间轴已做 odd-Schur 压缩，物理 fine 到 cache coarse 的 stride 是
  `2*mg_grid[t]`；写盘后释放 local slab，root 只汇总较小的 `hnn/hdg/sit` 并流式
  计算已有 dataset digest；
- 1/2/4-rank 不要求 SHA 位级相同，但同一物理输入的功能资产应报告相对误差。
  2026-10-29 `8x8x8x16` c64 E2 实测 `lonv/hnn/hdg/sit` 的 2/4-rank 对 1-rank
  Frobenius 相对差约 `1.3e-5–2.4e-5`，符合 MPI 归约顺序差异。

禁止把 overlap patch 做成每 rank 重建完整 hierarchy；它是仅覆盖
`2*mg_grid` coarse block 的 halo 工作区。

Legacy CUDA-to-KUPL cache 复用已在 2026-10-29 打通。此前约 `8.9%` 的
Galerkin 和 L8 `hnn=1.35e-2` 失配不是 KUPL slot 顺序，而是 CUDA raw lifecycle
未同步更新后的 parity，导致 CUDA even/odd Clover 相同。修复 CUDA
`sync_parity()` 后，同一 gauge 与同一 `lonv` 下 `hnn/hdg/sit` 相对误差为
`2.48e-7/7.34e-8/2.52e-7`；KUPL 直接加载 CUDA canonical E2 cache 的
`verify_status=0`，独立 true residual `2.016388e-7`。因此不得再对 Legacy payload
做方向转置或私有重排来绕过该问题。

多 rank builder 的 overlap patch 现在由 CUDA GPU 直接生成，不再因
`MPI_COMM_WORLD` halo 串序而降级到 CPU。当前 L8 4-rank、`2x2x1x1`、E2 cache 已由
同 rank/grid 的 KUPL 直接加载，`verify_status=0`，独立 distributed NumPy 真残差
`4.110297e-7`。跨后端验证必须同时记录 cache identity、rank/grid、独立残差和
`stencil_device=cuda:*`，不能只检查 cache 能否打开。

## 2026-10-21 Clover 6×6 块与 MG 几何

KUPL 的 Clover onsite 矩阵及逆矩阵在当前 gamma 基下按自旋 `(0,1)` 与
`(2,3)` 分成两个稠密 `6×6` 块，不是 `(0,3)/(1,2)`。求解热路径使用
`apply_clover_onsite`，逆矩阵也按两个 `6×6` 块计算，但外部 ABI 仍写回完整
`12×12` 矩阵并显式清零另外两块。29 项 CTest 必须全绿后才能保留该优化。

真实 `24³×64`、24 线程/rank、5 次稳态中位数：

| 精度 | MPI | solve 基线 | solve 优化后 | 真残差 |
|---|---:|---:|---:|---:|
| c64 | 1 | 9.858 s | 7.770 s | `4.12e-7` |
| c64 | 2 | 7.961 s | 6.352 s | `4.13e-7` |
| c128 | 1 | 20.196 s | 15.593 s | `6.96e-8` |
| c128 | 2 | 14.450 s | 11.473 s | `6.96e-8` |

双 MPI 相对单 MPI 仅约 `1.22x/1.36x`，该 workload 已受 NUMA/内存带宽和集合
归约限制，不能把增加 rank 数等同于线性加速。两 rank 分域优先切最长时间轴
`(1,1,1,2)`；x 轴二分实测慢约 9%。

`simple/kupl_multigrid.py` 的旧 builder 曾把 `mg_grid` 误作粗格尺寸，导致
`prolong` reshape 失败；正确值恒为 `lat[i] // mg_grid[i]`。非退化回归见
`pyqcu/testing/kupl/test_simple_mg_geometry.py`，不可只用 `4⁴/mg_grid=2`
这种粗格与 block 恰好相等的退化工况验证。

## 2026-10-21 legacy-MG coarse tile 与 rank 扩展

先区分当前选路和历史优化：dev103 同窗 `24^3x64` c64 的当前最佳开发路径是
`restart=0`、8 ranks × 2 threads、`1x4x2x1`，中位 `7.532 s / 19 it`，
并有独立全算子 MPI 真残差 `4.011e-7`。下面 `restart=5` 的 coarse 优化记录
说明该实现如何演进，但不是当前最快路径，也不应把其 native residual 写成独立
真残差。

KUPL LTO 的单点 A/B 为 `7.556 -> 7.196 s`（`4.8%`），解逐位一致、当时 CTest
`29/29`。该收益只覆盖 8 ranks × 2 threads 的 `24^3x64` c64 点，LTO 需通过
`--kupl-lib lto` 或 `PYQCU_KUPL_LIB` 显式选择，默认构建未切换。

`apply_multigrid_coarse_dslash_wide` 以 512-site tile 预解析 32 个邻接流，
随后按 site-major coefficient stream 连续累加；canonical/full 转换改为
half-site 遍历并一次写两个时间 parity，coarse correction 复用刚形成的
true residual。生产 `24³×64` c64、`restart=5`、deflate、4 threads/rank：

| ranks | 中位 | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 85.204 s | 1.00x | 100% |
| 2 | 46.752 s | 1.822x | 91.1% |
| 4 | 29.942 s | 2.846x | 71.1% |

4-rank 相对优化前 `94.108 s` 降低 68.18%，60 iterations，native relative
residual `9.70132e-7`，verifier=0。跨 rank 仍必须使用对称集合操作；该结果只覆盖
单节点 y-only 分域，不外推到跨节点 InfiniBand。

coarse correction 现在采用 fail-closed 验收：每次 correction 后比较 true residual；
若不变、变差或非有限，则撤销 correction 并关闭本 solve 的后续 correction，剩余
BiCGStab 在单个长 chunk 中完成。相同生产 `restart=5, deflate=true` 参数下，
solve 中位进一步降至 `16.771 s`，47 iterations，relative residual
`9.13844e-7`，verifier=0；相对最初的 `94.108 s` 降低 82.18%。

### 2026-10-21 BiCGStab 成对归约

`bicgstab_chunk` 将相邻的 `(v,v)` 与 `(v,s)` 点积合并为一次四实数组
`MPI_Allreduce`，并在同一 OpenMP 遍历中计算两个局部和。这样每轮全流程的集合
归约次数由 6 降到 5，同时保持更新公式、actual iteration counting、停机判据和
adaptive correction 不变。生产 `24³×64` c64、4 ranks x 4 threads、`restart=5`、
deflate：中位 `18.501 -> 17.725 s`（4.19%，1.0438x），均 47 iterations，
native relative residual `9.138e-7`，verifier=0。当时 CTest `29/29` 通过；
独立 NumPy 真残差未在该多 rank 脚本中启用，不能把 native residual 写成独立验证。

## 2026-10-22 2/4-rank efficiency decomposition

固定生产 c64、4 threads/rank 的 profile 分解显示，4-rank 的 fine
Wilson-Clover rank-local apply 占 solve `73.2%`；Allreduce `7.7%`、canonical
conversion `6.4%`、coarse dslash/halo `1.3%`、vector/other `11.4%`。相对
1-rank 完美扩展的额外耗时中，fine apply 占 `52.5%`，Allreduce `16.1%`，
canonical `9.7%`，other `21.1%`，coarse 仅 `0.6%`；4-rank fine apply 内 halo
仅 `7.55%`。因此 2/4-rank efficiency 损失首先是 rank 内计算和内存带宽问题。

把每轮 5 次 Allreduce 降到 4 次的候选在 4-rank 三轮 A/B 中为
`15.072466 -> 16.681220 s`，慢 `10.67%`；虽然均 47 iterations、verifier=0、
native residual 约 `9.138e-7`，候选已撤销。减少集合调用次数不保证降低关键路径。

## 2026-10-22 fine dslash hotspot and rejected face SIMD

固定 local `24×6×24×64`、4 ranks × 4 threads、c64：dslash 内点+周期面占
`53.22%`，标量边界占 `37.85%`，halo `3.29%`，MPI wait `5.65%`。SIMD block
对 Wilson/Clover/Schur 比 scalar 快 `13.98%/6.95%/2.04%`；Clover onsite 占
Clover 算子 `39.36%`，canonical 往返约为每迭代墙钟 `1.73%`。

y-face 连续时间 lane SIMD 候选使 4-rank 中位 `15.446281 -> 16.157466 s`
（慢 `4.60%`），MAD 从 `0.061435` 扩到 `0.715516 s`；六轮均 47 iterations、
verifier=0、PASS，已撤销。下一步应分析 scalar boundary 的 rank-face 分布和页
局部性，不能只因标量边界占比大就盲目扩大 SIMD lane。

## 2026-10-27 鲲鹏运行时与初始化收敛

- KUPL 现从 Linux sysfs 读取 CPU NUMA cpulist，`KUPL_NUMA_BIND=1` 不再按连续 CPU
  编号猜测节点；支持 `KUPL_CPU_NODE`、`KUPL_MEMORY_NODE=local|N|local+N` 与
  preferred/bind memory policy。`PYQCU_KUPL_PROFILE=kunpeng-2numa` 和
  `KUPL_REQUIRE_TOPOLOGY=1` 对跨 NUMA/拓扑缺失 fail closed；鲲鹏 profile 还会
  拒绝非鲲鹏主机，rank 间不一致会立即失败而不是挂起。
- 合成双 NUMA CTest 实际启动 2 rank，得到互斥 cpusets 和不同 cpu_node。
  `test_kunpeng_mpi_binding.py --synthetic-two-numa` 是通用 CI 的拓扑门；真实鲲鹏
  仍须用 `KUPL_TOPOLOGY_REPORT=1` 和 `mpirun --report-bindings` 复核。
- Clover cache 先收敛到目标 parity 惰性 `term + inverse`，本轮再把 term 原位求逆，
  每个 parity 常驻从 `2B` 降到 `B`（`B=144*half_sites*sizeof(complex)`）。缓存版
  standalone Clover dslash 同时改走已有 6×6 spin-block onsite，而外部 supplied
  12×12 ABI 仍保留通用路径。相邻 A/B 中 c64 `1.074 -> 0.957 s`（约 10.9%），
  c128 `1.881 -> 1.451 s`（约 22.9%）；只适用于 standalone `operator=clover`，
  求解器原本已使用 6×6 helper。
- 求解器 9 块全场级 scratch 与 3 块 parity scratch 改为首次求解惰性分配。
  `24³×64`、4 threads、plan=Clover 的新口径中，c64 init `0.547 -> 0.099 s`，
  c128 `1.066 -> 0.180 s`；由于 runtime benchmark 同时改为 init 后再分配输入，
  新报告的 `after_inputs` 必须与旧报告分离。组合 tile 后的最终 c64/c128 首次
  Clover 后 RSS 为 `2.079/3.359 GB`，首次 Clover 为 `3.757/4.690 s`。
- `compute_clover` 改为 site-tile 计算和 component 连续写回。两组相邻 A/B 中 c64
  首次 Clover 分别快 18.1%/23.2%，c128 快 8.4%/8.2%；该收益只属于 Clover setup，
  不得写成 steady solve 加速。
- 6×6 求逆也改为 site-tile 连续加载/写回；相对仅构造 tile 的基线，两组 c64 分别
  快 12.6%/17.4%，c128 快 12.1%/10.8%，同样只归因于 setup。
- face/corner halo 现在只为实际分解维度分配。1-rank plan=Clover 的 init 从
  c64/c128 `0.140/0.236 s` 降到 `0.023/0.021 s`，init RSS delta 约
  `207/411 MB -> 2.3 MB`；2/4-rank x/y/z/t 与对角路径仍须全测。
- `KUPL_FUSED_BICG_DOTS=1` 将 `(t,t)`/`(t,s)` 合并为一次遍历/归约。两个相邻
  `24³×64` c64、2 ranks × 4 threads 进程对分别得到 `19.039 -> 17.434 s` 和
  `19.509 -> 19.143 s`；最终交付二进制两次相邻复核为 `17.968 -> 17.778 s`
  和 `24.551 -> 23.713 s`（1.05% / 3.41%）。均 20 iterations、独立真残差约
  `4.1e-7`。共享机存在大幅漂移，只可引用相邻对，稳定承诺写约 1%–3.4%，
  不可跨窗口拼接或选最大收益。
- 默认 Release、`KUPL_ENABLE_NATIVE=OFF`、`KUPL_ENABLE_LTO=ON` 和 UBSAN 构建
  曾完成 33/33 CTest；新增 profile contract 与 pure-KUPL contract 后，当前
  Release 并行构建为 35/35。
  LTO 选项只是把历史 `4.8%` 构建方式产品化，不把共享机短样本升级为新倍率。
- boundary 预过滤与“全部 Irecv 提前于 face pack”两个候选在 2/4-rank 相邻复测
  中收益不一致，已完整撤销。不能仅凭访存或通信直觉保留。
- 完整指南：`docs/kupl_kunpeng_guide.md`；当前阶段证据：
  `logs/kupl_kunpeng_opt_20261027.md`。当前开发机为单 NUMA x86，容器禁止
  `set_mempolicy`，因此不能据此宣称鲲鹏 NUMA/片上内存性能。

  Strict-MG `8x8x8x16` c64, 2 levels, block `2^4`, `nvec=12`,
  `coarse_spin=2`, `restart=20`, 12 threads/rank currently measures
  `1.222731/1.101605/1.658620 s` at 1/2/4 ranks (2 warmups + 5 steady), with
  independent residuals below `1e-6`. The 4-rank point must use `2x2x1x1`; the
  shared login-node load average makes the negative 4-rank scaling unsuitable as a
  Kunpeng production claim. Record NUMA/cpuset, MPI ranks/grid, cache load and actual
  correction counts with every result.

## 2026-11-06 Kunpeng remote recovery and submission guardrails

- The remote running tree may be reorganized independently of the Git worktree.
  On the recovered LX2 environment, `~/PyQCU` pointed to `PyQCU-main` with source,
  `data`, `logs`, `docs`, `refer`, `skills`, and `pyqcu` present but no `.git`
  metadata. Inputs were in `~/data`, job scripts in `~/sh`, captured stdout/stderr
  in `~/out`/`~/err`, and uploaded helper code in `~/py`. Discover the live paths
  from the authenticated session; do not infer the layout from historical
  `data/production_L24/yNN` or repository-relative paths.
- An authenticated interactive screen/SSH session may be the only usable remote
  control path. In the recovered environment, batch SSH from the workstation was
  rejected and the external address landed on `login03`; the recorded `login04`
  host had a different internal address and rejected public-key batch login.
  Preserve an auditable screen hardcopy, but never claim that a fresh SSH or
  `screen` survived a host/container restart.
- `dsub` may reject a syntactically valid job with only `job was refused by hook`.
  This is an account/submission-policy signal, not permission to bypass the hook.
  Check the official read-only status path (`dqueue`, `djob`, `duser`, `dacct`) and
  re-test with a minimal job using the existing short `qcu_yNN` naming convention.
  Do not read privileged hook configuration, spoof another submit host, reuse
  another user's credentials, disable host-key checking, or submit through an
  undocumented scheduler endpoint. If the official path remains blocked, continue
  login-node builds/tests and mark compute-node execution blocked.
- On the recovered LX2 scheduler, the CLI queue flag mattered: a script containing
  only `#DSUB -q q_hpcapp` was refused, while `dsub -q q_hpcapp <script>` was
  accepted. A 32-rank follow-up then stalled after the host pre-hook with no
  stdout/report and no documented scheduler cancel command. Treat that as a
  pre-hook/runtime failure, not a solver result; do not submit another solver
  matrix until a short writable-output probe completes.
- Before any remote performance run, verify the source blobs, actual shared-library
  path and binary identity. Historical reports are context only; the live source
  and job output are authoritative.
- Long compute jobs follow the 5/10-minute rule: inspect output, stderr, scheduler
  state and progress at five minutes; terminate and audit at ten minutes if the job
  is not demonstrably healthy.
- Roofline first for the `24^3x64` c64 production case. The `levels=1,restart=0`
  control is parity-Schur BiCGStab, not multigrid. On the current 32-rank x
  12-thread t-split layout, local `ht=1`, so every production dslash site goes
  through the scalar boundary kernels; merely lowering the block-dslash site
  threshold does not activate the `ht>=3` block path.
- The recovered 2026-11-06 data measured `0.43456 s / 16 it / PASS` but only
  modeled about `149.5 GB/s` and `199.4 GFLOP/s`. Against conservative
  same-shape ceilings this is roughly `2.8-3.4%` of HBM bandwidth and
  `0.4-0.5%` of FP32 SVE peak. Profile shares were approximately scalar
  boundary `30%`, MPI wait `22%`, halo `7.5%`, and everything outside dslash
  `38.5%`. Do not respond to this result by adding bandwidth pressure; attack
  scalar issue rate, small-message collective latency, and iteration count.
- A `0.1 s` target in the same traffic model needs only about `650 GB/s` and
  `0.87 TFLOP/s`. The hardware can support that working point, so an
  unreachability claim requires a measured post-vectorization/post-collective
  wall, not a roofline argument.
- `ht=1` spatial-lane vectorization is not automatically profitable. The
  first opt-in z-lane implementation passed `2rank_t` and strict one-rank
  correctness, but a norm-checked `A-B-B-A` login-node test was about `4.8%`
  slower. Keep such kernels opt-in until compute-node A/B and compiler
  vectorization evidence both approve them; do not promote the best-looking
  single sample.
- BiCGStab and Legacy-MG have different grid optima. On real L16 c64,
  BiCGStab improved through 32 ranks, while MG was best at 16 ranks
  (`grid=1 1 2 8`) and was only tied with BiCGStab at 32 ranks because coarse
  work and synchronization erased the iteration advantage. Screen both modes
  separately instead of inheriting one grid.
- For KUPL Legacy-MG, start parameter screens with `restart=1/2`, deflate on,
  and coarse settings at their defaults. On the real L8 fixture, disabling
  deflate, reducing coarse max-iter, or loosening coarse tolerance all
  regressed. E=4/E8 and `nv_iters=4` did not produce a stable runtime gain over
  E2/nv2 while increasing cache build time and size by 2x-7x.
- MG cache layout imposes hard geometry: an L8 odd-Schur cache has coarse time
  extent 8, so `grid_t>4` must fail before launch. Local-stencil block extents
  must also satisfy `3*block+2 <= fine_extent`; do not label expected geometry
  rejection as a numerical failure.
- Cache construction is separate from solve time. The real L16 E2/nv2 cache
  took about `186.5 s`, and `local-stencil-batch=16` did not improve it. L24
  extrapolates beyond the 10-minute login-node gate on the current scalar
  stencil loop. Optimize coarse-site reuse there, not OpenMP oversubscription;
  binding the OpenMP master before KUPL reads affinity is an anti-pattern.
- `KUPL_ENABLE_MG_STENCIL_PARALLEL=1` parallelizes the previously serial
  coarse-site cache-build loop with per-worker `source/even/schur` buffers.
  Retain the opt-in guard during compute-node qualification. On the 2026-11-06
  login-node fixtures it reduced L8 `8.617 -> 0.669 s`, L16
  `186.52 -> 10.62 s`, and L24 to `80.70 s` with unchanged L8/L16
  gram/Galerkin values. This changes MG setup economics; do not describe the
  setup-cache solve as a cold-start result.
- Historical L24 Legacy-MG numbers such as `1.60861 s` and `1.51917 s` are
  invalid as true-MG evidence: `PYQCU_KUPL_MG_PROFILE=1` showed
  `coarse_calls=0` followed by `fallback=fine_bicgstab` for the old coarse
  tolerance setting. The corresponding reports must be read as fine-BiCG
  fallback, not multigrid.
- `KUPL_REQUIRE_MG_CORRECTION=1` is the formal true-MG gate. It rejects
  `levels<=1`/`restart=0`, fine-BiCG fallback, and a solve that completes
  without any accepted coarse correction. Every accepted MG result must also
  show `coarse_calls>0`; iteration count alone is not evidence.
- Full-step coarse correction can increase the fine true residual. The current
  implementation builds one coarse correction and searches damping
  `1, 0.5, 0.25, 0.125`, accepting only a step that reduces the true residual.
  A failed search restores the previous solution and residual before fallback.
  L24 one-rank damping gate passed at `7.52818155894056 s`, with
  `fine_calls=49`, `coarse_calls=168`, and `verify_status=0`; the independent
  Python residual did not finish within the 180 s diagnostic timeout.
- On the 2026-11-06 shared login node, ten legal L24 true-MG grids all passed
  the gate. The best observed wall was 16 ranks x 8 threads,
  `grid=1 1 1 16`, at `1.686342053115368 s`; 32-rank grids did not improve it.
  Valid grid axes must divide the cache's coarse extent: L8 tblock2 has coarse
  time extent 4, so `grid_t>4` is a geometry rejection, not a numerical failure.
- Same-window L24 A/B at 16 ranks x 8 threads, `grid=1 1 1 16`, gave BiCGStab
  median `1.2545952125 s` (16 iterations) and true-MG median `1.3541084125 s`
  (`fine_calls=37`, `coarse_calls=29`). The median BiCGStab/MG ratio was
  `0.926510`, i.e. MG was about 7.9% slower in this window. Shared-login load
  drift was large, so this is a negative screen, not a portable production
  speedup; compute-node A/B remains mandatory.
- L32x64 c64 input was prepared from
  `beta6.20_mu-0.2770_ms-0.2400_L32x64_cfg_11000.lime`. A one-rank E1/tblock8
  cache build and a 32-rank L32 BiCGStab run both exceeded the 10-minute gate
  without a report and were terminated. No L32 true-MG or L32 speedup claim is
  valid yet. The next setup attempt must reduce null-vector/stencil cost or use
  a qualified distributed builder, then rerun on the compute node.
- Keep `logical_array_sha256_chunked` / `logical_dataset_sha256` C-order-safe:
  fix prefix coordinates and split only one axis while keeping trailing axes
  complete. Cartesian multi-axis chunks can make a correct E4 `lonv` manifest
  appear to mismatch during validation.
- A subsequent same-window L24 `restart=10` A/B at 16 ranks x 8 threads,
  `grid=1 1 1 16`, produced BiCGStab median `1.39724 s` and true-MG median
  `0.78976 s`, a `1.7692x` median speedup (`fine_calls=32`,
  `coarse_calls=19`). The full-residual companion run passed with
  `wall=1.04379 s`, `verify_status=0`, and independent residual `6.88873e-5`
  below the `2.44140625e-4` gate. This is the current positive login-node
  candidate; it still needs compute-node repeats and is not the 0.1 s target.
- Larger correction intervals were also screened. `restart=20` gave median
  BiCGStab/true-MG `1.230783/0.705146 s` (`1.7454x`), while `restart=50`
  gave `1.091866/0.704241 s` (`1.5504x`). Both used `fine_calls=29` and
  `coarse_calls=9`. At `restart=50` only the initial deflation correction was
  applied, so call it deflation-assisted fine BiCGStab rather than a periodic
  V-cycle. Its full-residual run passed at `0.947332 s` with independent
  residual `5.5308248713e-5`. This is the lowest absolute L24 login-node
  result observed, still about 9.5x above 0.1 s.
- The rebuilt KUPL suite passed `35/35` CTest tests, including 1/2/4-rank MG,
  legacy MG, failure modes, runtime benchmark, NUMA and profile contracts.

## 2026-11-06 contiguous rank slabs and grid-specific MG screening

- On the recovered compute node, jobs with 16 ranks could remain without a
  report for ten minutes even after the HDF5 rank-local slab path was used.
  The stderr eventually contained only the expected topology records, with
  unique cpusets, `memory_policy=bind`, and matching HBM nodes. Do not infer a
  kernel failure or a performance result from a job that never produced a case
  report.
- The remote Python environment has `h5py 3.7.0` with
  `get_config().mpi == False`. The optional `KUPL_HDF5_MPIO_READ=1` path must
  fail closed there; do not claim it was exercised. The fallback is to prepare
  contiguous per-rank slabs with
  `pyqcu/testing/simple/prepare_mpi_slab.py` and pass
  `--gauge-slab-dir`/`--source-slab-dir` to `kupl_multigrid.py`.
- For L24 c64 and `grid=1 1 2 8`, the prepared gauge slabs are 15,931,608
  bytes each and source slabs are 5,314,776 bytes each. On login02 the driver
  measured gauge read `0.0962 s` and source read `0.0096 s`, confirming that the
  slab path itself is fast; it does not guarantee compute-node scheduler or
  MPI startup health.
- Login-node grid screening with 16 ranks x 8 threads and the same cache:

  | grid | BiCGStab (s) | true-MG restart=20 (s) | MG/BiCG |
  |---|---|---:|---:|
  | `1 1 1 16` | 1.116278 median | 1.037458 median | 1.076x median-of-medians |
  | `1 1 2 8` | 1.844325 | 1.840493 | 1.002x |
  | `1 2 2 4` | 1.084889 | 1.250455 | 0.868x |

  The t-only grid is the only one that retained a positive MG signal; grid
  choice must be screened separately for BiCGStab and MG. The t-only samples
  were noisy across shared-login windows, so report medians and per-pair
  dispersion, not only the best pair.
- The restart=20 full-residual companion passed at `1.316182 s` with
  `verify_status=0`, `fine_calls=29`, `coarse_calls=9`, and independent
  residual `5.5308248713e-5` below `2.44140625e-4`. The `restart=50` run was
  `1.010575 s`, so initial deflation did not beat restart=20 in this window.
- Use the slab helper only with a fixed `(lattice, grid)` pair and verify the
  rank file trailing shape. Never reuse a slab directory across a different
  process grid.
- `KUPL_PROFILE_DSLASH=1` on the t-only 16×8 BiCG path measured halo
  `0.0436 s`, wait `0.1554 s`, boundary `0.3252 s`, and interior
  `0.000038 s` over 70 calls. The opt-in
  `KUPL_ENABLE_HT2_TBOUNDARY_SPLIT=1` candidate improved the BiCG median only
  slightly (`1.0996 -> 1.0376 s`) but regressed true-MG
  (`0.9027 -> 1.0411 s`). Keep it opt-in; do not promote it based on BiCG
  alone.

## 2026-11-06 HMPI Cartesian topology and periodic geometry

- On the Kunpeng compute node, the legacy `MPI_Cart_create` /
  `MPI_Cart_shift` path could stop after `bind_logical_numa()` with no
  `GEOMETRY_PROFILE` or `LATTICE_SET_PROFILE` output. A 240 s profile job timed
  out with `rc=124`; this is an HMPI topology-creation hang, not a numerical
  failure or an HBM page-placement result.
- The replacement builds the lattice communicator with `MPI_Comm_dup` and
  computes row-major Cartesian coordinates and neighbor ranks explicitly. The
  lattice uses periodic faces, so edge neighbors must wrap around; replacing
  them with `MPI_PROC_NULL` breaks 2/4-rank Wilson and Laplacian halos.
- `LatticeGeometry::cartesian_rank()` is the single rank formula for diagonal
  Clover and coarse-MG peers. Do not call `MPI_Cart_rank` on the duplicated
  non-topology communicator.
- `KUPL_PROFILE_INIT` and `PYQCU_KUPL_PROFILE_INIT` are aliases for the same
  initialization profiler. The useful stages are `geometry`,
  `comm-duplicate`, `coordinates`, `threads`, `halo-scratch`,
  `boundary-sites`, `register` and `erase`.
- Compute-node profile `2050850` completed with `END rc=0`; 15 ranks took
  `6.02-7.43 s` in `applyInit/applyEnd`, one rank took `34.81 s`, and
  `register` reached `19.48 s`. A real 16×8 t-only HBM BiCGStab run then
  measured `wall_seconds=213.4409 s / 16 it / PASS` with independent residual
  `3.5410827e-6`. Its HBM-bound setup showed `register=41.75-46.30 s`, so setup
  and solve must be reported separately.
- The paired true-MG case had no report at the 10-minute gate and was terminated
  with exit code 143. Do not reuse it as an MG speedup result. The next
  comparison must hold rank/thread/grid and manual geometry fixed on both
  BiCGStab and MG.

## 2026-11-12 L24 HBM-node scan recovery and high-performance configuration

- Remote recovery used the LX2 cluster through `login06` (`ssh 10.44.9.4`,
  user `nsls_yangyb`). The running tree was `/home/share/nsls_yangyb/PyQCU`
  (`PyQCU-main`), cases were under
  `data/kupl_hbm_opt_20261106/scan8_20261112/`, and job scripts under `~/sh`.
- The scheduler is `dsub`/`djob`/`djob -ll`/`dkill` on queue `q_hpcapp`;
  exclusive jobs use `-x job`. Parameters in `#DSUB` headers such as
  `-N/-rpn/-R/-nl` may be ignored, so write scheduling parameters on the
  `dsub` command line and verify `execNodes` with `djob -ll <id>` after
  submission. A 32-rank job needs explicit `-N 32 -rpn 32`. Exclude `cn12038`
  and `cn10579` (`data/kupl_hbm_opt_20261106/excluded_nodes.txt`). A 64-task
  job was refused by the account cap `testusers_rebuild`
  (`mainStateReasonMsg: The limit <testusers_rebuild> has been exceeded`).
  A node preference such as `-nl` can also be ignored when the requested node
  has no resources (`No node is available to meet node preferences`), so
  confirm placement or resubmit on an idle node.
- A 95 s per-case timeout is too short: 16 ranks take about `108-117 s`
  end-to-end (Python import plus initialization about `100 s`) and 32 ranks
  about `233 s`. `RC=124` is only a timeout signal, not proof of a deadlock;
  use `400-600 s` before deciding a case is stuck.
- Do not replace initialization statements with ad-hoc debug prints. On the
  remote tree, `pyqcu/testing/simple/kupl_multigrid.py:847` had
  `verify_status = None` replaced by `print("BEFORE_INIT")`; with `levels=1`
  (`use_coarse_assets=False`) this necessarily raised `UnboundLocalError`.
  The line was restored, and the remote md5
  `c7756171a3619cb93456639dab9d3239` then matched the local file. The debug
  variant is preserved as `kupl_multigrid.py.2026-11-12-20-10-00.bak`
  (`46552 B`).
- Sysfs probing on `cn19701` showed 32 NUMA nodes: `node0-15` at about
  `32 GB` each and `node16-31` at `4 GB` each (on-chip HBM). KUPL reported
  `numa_nodes=16`, `memory_node=16..31`, per-rank cpusets of 8-12 CPUs,
  `memory_policy=bind`, `bound_threads=threads`, and `source=sysfs`; all ranks
  reported `memory_bound=true`. The matching configuration is
  `KUPL_MEMORY_NODE=local+16`.
- Reproduction configuration (identical for every case below unless noted):
  `--lat 24 24 24 64 --mass -0.277 --csw 1.160920226 --atol 1e-4
  --max-iter 50 --dtype c64 --levels 1 --restart 0`; `levels=1,restart=0` is
  parity-Schur BiCGStab, not multigrid. MPI used
  `mpirun --oversubscribe -np NP --bind-to none --map-by ppr:PPR:numa
  --allow-run-as-root`. The fixed control was true residual about `3.5412e-6`
  against the `2.44140625e-4` acceptance threshold, `PASS`.

Measured results (exclusive `-x job`, solve wall `wall_seconds`):

| case | node | ranks x threads | grid | ppr | wall (s) | iter | true residual | status |
|---|---|---|---:|---:|---:|---:|---|---|
| r32t12b | cn19699 | 32r x 12t | 1 1 1 32 | 2 | **0.8410** | 16 | 3.54117e-6 | PASS |
| r32t18b | cn19699 | 32r x 18t | 1 1 1 32 | 2 | 1.5979 | 16 | 3.54122e-6 | PASS |
| r16t8b | cn19699 | 16r x 8t | 1 1 1 16 | 1 | 1.1442 | 16 | 3.54128e-6 | PASS |
| hbm8c | cn19699 | 8r x 16t | 1 1 1 8 | 1 | 2.0595 | 16 | 3.54132e-6 | PASS |
| hbm16d | cn19698 | 16r x 8t | 1 1 1 16 | 1 | 1.0946 | 16 | 3.54128e-6 | PASS |
| hbm4 | cn19698 | 4r x 8t | 1 1 1 4 | 1 | 4.7462 | 16 | 3.54113e-6 | PASS |
| y76r32t12 | cn19699 | 32r x 12t | 1 1 1 32 | 2 | **0.5193** | 16 | 3.54122e-6 | PASS (y76 recipe) |

Same-node (`cn19699`) ordering: 32r x 12t `0.8410 s` < 16r x 8t `1.1442 s`
< 32r x 18t `1.5979 s` < 8r x 16t `2.0595 s`; at 32 ranks, 12 threads clearly
beat 18 threads. On the HBM nodes, 16 ranks x 8 threads beat 8 ranks x
16 threads.

DSLASH profile (`KUPL_PROFILE_DSLASH=1`, 70 calls):

| case | halo | interior | wait | boundary | total |
|---|---:|---:|---:|---:|---:|
| r32t12b | 0.0675 | 0.0095 | 0.2111 | 0.3132 | 0.6013 |
| hbm16d | 0.0306 | 0.000265 | 0.1627 | 0.6164 | 0.8100 |

Cross-node and flag caveat: the x_case_dbg recipe (with
`KUPL_PROFILE_INIT=1`, `KUPL_PROFILE_DSLASH=1`, `KUPL_FUSED_BICG_DOTS=1`)
measured `0.8410 s` on `cn19699`. Re-running the y76 recipe on the same node
gave `0.5193 s` (16 it, residual `3.5412238955e-6`, PASS), so about
`1.62x` of that gap is flag/wrapper overhead. Against historical y76 on
`cn10579` (`0.4346 s`, `docs/qcu_l24_solver_timing_20261111.md`, grade B)
the same-recipe node difference is only about `1.20x`. Always state the node
and the flags together. The y76 recipe did not set
`KUPL_SKIP_INPUT_PREBIND=1`, so its `input_prebind_seconds=15.006 s` is setup
cost and is excluded from `wall_seconds`.

Pending, not conclusions: comparing grid variants (e.g. `1 1 2 16`,
`1 2 2 8`) on `cn19699`, collecting multiple samples for 32r x 12t
(currently one sample per case), and documenting the semantics of
`KUPL_ENABLE_T2_ZBLOCK_DSLASH` (only value `0` was used).

Optimization A/B on the rebuilt library (16r x 8t, grid `1 1 1 16`, y76
recipe, cn19650 with `preferred` memory policy, one sample each):
`opt16p_base` = `0.6814 s` (12 it), `opt16p_halo` =
`0.5483 s` (12 it, `KUPL_SKIP_FINAL_RESIDUAL=1`, `KUPL_HALO_PACK6=1`,
`KUPL_HALO_DEDUP_PACK=1`, `KUPL_T_LOCAL_IN_PART1=1`), and `opt16p_all` =
`0.3900 s` (12 it, additionally `KUPL_ENABLE_BOUNDARY_LANES=1`). All PASS
with true residual about `5.385e-5`. The lane kernel alone is
`1.406x` over the halo group and `1.747x` over base. Old library at the
same shape was `1.1442 s` (different node and memory policy, reference
only). Single samples; treat as a first signal.

On a shared node, `KUPL_MEMORY_POLICY=bind` with
`KUPL_MEMORY_NODE=local+16` was repeatedly killed with `RC=137`
(SIGKILL) after `87-110 s`, while `preferred` or an exclusive node
completed. This is a co-tenancy OOM on the 4 GB on-package HBM domains,
not a KUPL code defect.

## 2026-11-13 L24 单节点性能与带宽剖析

数据来源：远端 `data/kupl_opt_20261113/remote/*/report.json` 与 `meta.txt`
（`/home/share/nsls_yangyb/PyQCU`，本机未挂载，数值经 screen 捕获的
`grep wall_seconds/iterations/final_true` 转抄）、`logs/v202611130900.txt`、
`logs/v202611130926.txt`、`logs/v202611130946.txt`。**C 级时点证据、未提交**；
除注明外每用例 1 个样本。

- 用例：L24 `24^3x64`、c64、点源 `seed 42`、`mass=-0.277`、
  `csw=1.160920226`、`atol=1e-4`、`levels=1`、`restart=0`（parity-Schur
  BiCGStab，不是 multigrid）、grid `1 1 1 32`；`KUPL_NUMA_BIND=1`、
  `KUPL_MEMORY_POLICY=preferred`、`KUPL_REQUIRE_MEMORY_BIND=0`、
  `mpirun --bind-to none --map-by ppr:2:numa`，另加
  `KUPL_HALO_PACK6=1`、`KUPL_HALO_DEDUP_PACK=1`、`KUPL_T_LOCAL_IN_PART1=1`、
  `KUPL_ENABLE_BOUNDARY_LANES=1`、`KUPL_SKIP_FINAL_RESIDUAL=1`。验收为外层
  独立分布式 FULL Wilson-Clover 真残差门，全部 `PASS`、全部 12 it。

| 用例 | 节点 | 规模 | 关键配置 | wall (s) | 真残差 |
|---|---|---|---:|---:|---|
| `v7_t16` | cn19699 | 32r×16t（`-R cpu=16`） | ACTIVE + t2 | **0.188526** | 5.384853e-05 |
| `v7_t18` | cn19699 | 32r×18t（`-R cpu=18`） | ACTIVE + t2 | 0.183044 | 5.384866e-05 |
| `t18rep1–5` | cn19699 | 32r×18t（`-R cpu=18`） | ACTIVE + t2，同配置 5 次 | **中位 0.186230**（min 0.180422 / max 0.193310） | 5.384866e-05 |
| `v7_prof4` | cn19699 | 32r×12t | ACTIVE + t2 | 0.207077 | 5.38e-05 量级 |
| `v7_fast` | cn19699 | 32r×12t | ACTIVE + t2，`build-fast`（`-ffast-math`） | 0.204288 | 5.384889e-05 |
| `v7_base3_hbm` | cn19699 | 32r×12t | PASSIVE，无 t2 | 0.281166 | 5.384894e-05 |
| `v7_t2` | cn19699 | 32r×12t | PASSIVE + t2 | 0.266416 | 未单列，`PASS` |
| `v7_c50_base` | cn19650 | 32r×12t | PASSIVE + t2 | 0.319600 | 5.384831e-05 |
| `v7_c50_act` | cn19650 | 32r×12t | ACTIVE + t2 | 0.249223 | 未单列，`PASS` |
| `v7_blk` | cn19650 | 32r×12t | ACTIVE，`KUPL_BLOCK_DSLASH_MIN_HALF_SITES=8192` | 0.244864 | 未单列，`PASS` |

`v7_t18` 是同一远端捕获中更晚完成的样本（`END=2026-11-13T09:50:32`），未进入
`logs/v202611130946.txt`（该日志以 `v7_t16` `0.188526 s` 为当前最优）；其迭代数
未单独抓取。`v7_t18` 之后提交的 `v7_t19`（32r×19t）、`v7_r16t38`/`v7_r16t36`
均无可用报告：前者只到 `Submit job`，后者 `meta.txt` 不存在且已被 `dkill`。

- `t18` 重复性（2026-11-13 11:43–12:00，作业 `2113436/2113439/2113445/2113447/
  2113452`，cn19699，32r×18t，ACTIVE + t2，`KUPL_MEMORY_POLICY=preferred`）：
  5 次墙钟 `0.180422`、`0.186732`、`0.181537`、`0.193310`、`0.186230 s`；
  中位 `0.186230 s`、均值 `0.185646 s`、MAD `0.004693 s`、极差 `6.9%`。
  5 次均 12 it、`PASS`、真残差 `5.384866e-05`。S11 单点 `0.183044 s` 落在
  样本区间内，但比中位低 `1.74%`，只能作为区间内一点引用。

- `OMP_WAIT_POLICY` 是本次唯一受控单变量：`cn19650` 同 lane 由 PASSIVE
  `0.319600 s` 降至 ACTIVE `0.249223 s`（`1.28×`）。逐 rank 剖析给出的机制是
  小并行区（全局归约、向量更新）的线程唤醒开销：reduce 约 `1.02 ms/次`
  （PASSIVE 批次 rank 26，`0.052268 s/51`）降到约 `0.45 ms/次`（ACTIVE 批次
  rank 17，`0.023190 s/51`）。`1.055×`（t2 对 base）与 `1.287×`（prof4 对 t2）
  是跨批次派生值，不是同窗 A/B。
- 新增 `KUPL_PROFILE_OPT=1`（默认关闭、仅计时、不改变算法）：每 rank 输出
  dslash/clover/reduce 秒数与调用次数、dslash/clover 字节与 `GB/s`
  （rank 与聚合），以及 prepare/solve/reconstruct 阶段。ACTIVE、cn19699、
  rank 17：dslash `0.104099 s/52`、`828.113 MB`、`7.96 GB/s·rank`
  （聚合 `254.56 GB/s`）；clover `0.013720 s/52`、`552.075 MB`、
  `40.24 GB/s·rank`（聚合 `1287.63 GB/s`）；reduce `0.023190 s/51`；
  phases `prepare=0.003573`、`solve=0.156956`、`reconstruct=0.002327`，
  solve 外 setup 约 `44 ms`。
- 带宽峰值：同节点 STREAM triad 探针（8M double × 3 数组、5 次取最优、
  12 线程/rank、`--bind-to numa --map-by ppr:2:numa`）为
  `18.3–20.1 GB/s·rank`（逐轮 `18.267/18.327/18.522/18.648/18.654/18.797/
  18.964/20.120`），32 rank 聚合约 `600 GB/s`。dslash 因此约为峰值的 `40%`；
  clover 的 `40.24 GB/s·rank` 高于探针值，来自 clover cache 复用，不得当作
  超峰值或用作 clover 的带宽分母。
- 跨节点：同为 PASSIVE 基线，`cn19650` `0.319600 s` 对 `cn19699`
  `0.266416 s`，约 `20%` 差异；两节点硬件/负载/cpuset 身份未逐项对齐，
  不得据此宣称节点性能差，也不得用单一节点绝对值外推。
- 噪声级候选：`-ffast-math` 约 `1.4%`（0.204288 对 0.207077）、block-dslash
  约 `1.8%`（0.244864 对 0.249223），均不作为默认构建或默认门控。
- 阻塞：真实 MG（`levels=2`）多 rank 需要预置全局 `--coarse-cache`，
  `kupl_multigrid.py` 的多 rank 导出以
  `multi-rank coarse-cache export is not implemented` 拒绝；1-rank L24 缓存
  构建（作业 2112385）约 25 分钟未产出 `cache_l24_v1.h5`，已 `dkill` 终止。
  属工具链限制，不是算法结论。
- **剖析口径修正**（作业 `2114373` → `2114720`，cn19699、32r×18t、
  ACTIVE + t2 lane、L24 c64）：旧 `KUPL_PROFILE_DSLASH` 用 `!profile` 条件让
  该变量绕过生产 split-boundary 路径，`2114373` 的 halo `0.007089`、
  interior `0.004421`、wait `0.020300`、boundary `0.020990`、
  total `0.052800 s`（28 次）属于通用 interior/boundary 路径，不是生产分解。
  修复后 `2114720` 的生产分解为 halo `0.007918`、interior `0.018824`、
  wait `0.022757`、boundary `0.008081`、total `0.057580 s`（28 次），
  `wall_seconds=0.18316925200633705`；同作业 rank 0 的 `OPT_PROFILE` 为
  dslash `0.101344 s/52`（`8.17 GB/s·rank`）、clover `0.012212 s/52`、
  reduce `0.012635 s/39`、phases `prepare=0.002618`、`solve=0.137568`、
  `reconstruct=0.002382`。`2114373` 的 `OPT_PROFILE`（与路径无关）为
  dslash `0.094981 s/52`（`8.72 GB/s·rank`）、clover `0.013244 s/52`、
  reduce `0.012464 s/39`、`wall_seconds=0.1797297460725531`。
- **修复与验证**：`cpp/kupl/qcu/src/lattice_wilson_dslash.cpp` 的 `bare_dslash`
  已把 split-boundary 与 t2 z-block 两条生产路径就地纳入
  halo/interior/wait/boundary 计时；本地 CTest `37/37` 通过（`114.55 s` 与
  `82.74 s` 两次），本地 2-rank 剖析冒烟 `PASS`（c64 `calls=91`、
  c128 `calls=111`）。远端重建后复测 `2114720` 已取得生产分解；wait 仍约占
  生产 dslash 的 40%，part 1 本地计算只覆盖其约 83%。
- **t2 z-block 缺陷修复（提交 `1ad6ed7e`）**：
  `compute_dslash_t2_spatial_zblocks` 有四处缺陷：z 内点块未加 lane 偏移，
  统一从 lane 0 累加；z 上界周期 wrap 误写 lane 0，应写 `lanes-1`；多 block
  （`z_dim > kSiteLanes`）时 `wrapped_site` 应为 `site - z0`（本地 z=0），
  原实现指向当前 block 起点；`compute_dslash_t2_time_zblocks` 的 backward
  gauge halo/local 条件写反。修复后本地 ctypes 探针（c64、2 rank）相对差：
  lattice `8 4 4 2` 为 `4.59e-08`，`8 4 24 2`（两 block）为 `4.35e-08`；
  退化 gauge 隔离显示修复前仅空间跳差 `33–39%`，仅时间跳差值为 0。全量
  CTest `38/38`，新增 `kupl_qcu_t2_zblock_lanes`（`4 2 24 2`）。远端 A/B
  （cn19699、32r×18t、ACTIVE + t2 lane、`preferred`）：z-block 中位
  `0.18324761 s`（0.18324761/0.17664439/0.18536645），基线中位
  `0.18410848 s`（0.18410848/0.18148361/0.18796321），全部 `converged=true`，
  真残差 `5.3848760834813754e-05`，中位差 `0.47%`。结论：噪声级，z-block
  保持 opt-in，不设为默认；错误路径的 `0.1395 s` 无效。
- **参数与 rank 扫描（2026-11-13 下午，负结果）**：`GOMP_SPINCOUNT=1000`
  中位 `0.183795 s` 对对照 `0.184282 s`（0.26%）；`GOMP_SPINCOUNT=30000`
  中位 `0.185177 s`；`KUPL_HALO_DEDUP_PACK=0` 中位 `0.186914 s`；
  `KUPL_MEMORY_POLICY=bind` 中位 `0.187453 s`（对照 `0.186068 s`）。同节点
  抖动约 ±2%，均判为噪声或略差，不改默认。rank 扫描：8 rank 样本
  `0.437795`/`0.449878`/`1.731601 s`，16 rank `3.054642`/`0.479433`/
  `3.533757 s`，32 rank `0.183248`/`0.176644`/`0.185366 s`；更少 rank 时
  local time extent 变为 4/8，不再走 `local[3]==2` 的 t2 lane 路径，墙钟
  显著变差且不稳定，32 rank 仍是最优点。
- **稳定性与 wait 归因（2026-11-13 傍晚）**：同配置 9 个有效样本中位
  `0.184694 s`、均值 `0.184138 s`、σ `0.002176 s`、极差 `3.8%`（一个样本
  600 s 超时 `RC=124` 作废）。`KUPL_PROFILE_OPT=1` 的逐 rank `SKEW` 诊断
  （作业 `2117260`）：arrival 均值 `0.912 ms`（sd `0.110`）、wait 均值
  `0.762 ms`（sd `0.168`）、相关 `+0.26`；wait 主要是通信而非 rank 偏斜，
  共享内存 halo 的理论上限约 `17–19%`。`MPI_Win_allocate_shared` 原型在修正
  面判定与 `MPI_Win_sync` 后与 MPI 路径逐位一致（rel `0`），但全局 barrier 对
  rank 非对称调用不安全（该开关下 ctest 600 s 超时），已回退；继续需要逐邻居
  序号或双缓冲收发协议。
- **路径开关 A/B（2026-11-13 晚）**：`KUPL_ENABLE_BOUNDARY_LANES=0` 中位
  `0.225429 s`，比默认慢 22%，确认 boundary lane 内核必要。`KUPL_T_LOCAL_IN_PART1=0`
  的 5+5 交替 A/B 为 `0.184149` 对 `0.185412 s`（0.68%），但候选总在对照前提交、
  对照随时间下漂（`0.191470`→`0.183342`），差异不可归因，保持默认 `1`。
- **HALO_PACK6 必要性（2026-11-13 晚，均衡顺序 3+3）**：
  `KUPL_HALO_PACK6=1` 的 dslash 中位 `0.096633 s`、solve 中位 `0.139138 s`；
  `=0` 时为 `0.213639 s` / `0.266636 s`，即 dslash 慢 `2.21×`、solve 慢 `1.92×`。
  作业 `2118031/2118035/2118039/2118042/2118049/2118056`，全部带
  `KUPL_PROFILE_OPT=1`。pack6 把线上 12 分量压到 6 分量，收益远超投影成本。
- **共享内存 halo 原型（2026-11-13 夜，opt-in）**：`KUPL_SHM_HALO=1` 且同节点
  时用 `MPI_Win_allocate_shared` 交换时间面，每面 `published/consumed` 计数 +
  双缓冲，无全局 barrier。本地 c64、2 rank、`8 4 24 2` 与 MPI 路径逐位一致
  （rel `0`），2/4 rank 求解 `PASS`；新增 `kupl_qcu_shm_halo`（时间面）与
  `kupl_qcu_shm_halo_4rank`（对角分解）两个对称回归门，全量 CTest `40/40`；
  `KUPL_OMP_THREADS=4` 下 2/4 rank 也 `PASS`。
  旋转等待有 30 s 上限，超时抛出 `KUPL shared halo handshake timeout`，
  只适用于对称调用序列。远端 3+3 A/B（`2119917/2119919/2119921/2119925/
  2119926/2119927`）因 SSH 停滞未读取，收益未验证，不得宣称加速。
