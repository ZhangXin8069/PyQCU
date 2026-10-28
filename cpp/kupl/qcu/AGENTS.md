# KUPL QCU Backend

- Independent C++17 MPI+OpenMP implementation. Never include or link
  `cpp/cuda/qcu`.
- Keep the same backend layering as CUDA QCU: `include/kupl/` declarations,
  `src/apply_*.cpp` thin C ABI adapters, `src/lattice_*.cpp` numerical
  domains, and `backend.cpp` as the `LatticeSet` facade.
- Public ABI: `params int32[58]`, real `argv[8]`, `set_ptrs int64[100]`.
  `argv[7]` is `csw`; keep indices synchronized with `include/kupl/define.h`.
- Canonical storage is C-order with parity-compressed time:
  gauge `[2,3,3,4,X,Y,Z,T/2]`,
  fermion `[2,4,3,X,Y,Z,T/2]`,
  onsite `[4,3,4,3,X,Y,Z,T/2]`.
- Fermion external indices are `(parity*12+component)*half_sites+site`;
  Clover indices are `(row*12+column)*half_sites+site`. Halo buffers may be
  face-site-major but must be converted through a vector view.
- Use MPI for four-dimensional rank halos and OpenMP only inside a rank.
  Global reductions must use `MPI_Allreduce`; do not introduce rank-dependent
  collectives.
- Clover onsite coefficient is `-kappa*csw/8`, with
  `kappa=1/(2*mass+8)`. The public Clover ABI returns `A=I+T`, never bare
  `T`; `applyCloversQcu` returns `A^-1`, and `csw=0` must produce identity.
- Wilson solver semantics are parity Schur based:
  `S_W=I-kappa^2 D_bare,pq D_bare,qp`. BiStabCG solves `S_W`; CG solves
  `S_W^dagger S_W`. They solve the same full Wilson matrix but do not share an
  iteration trajectory.
- New solve paths check a reconstructed full-field true residual. The c64
  relative floor is `max(atol,2048*epsilon)` and the c128 floor is
  `max(atol,128*epsilon)`.
- The standalone Laplacian uses `gauge[3,3,3,X,Y,Z]`,
  `field[3,X,Y,Z]`, periodic three-dimensional MPI halos, and ignores `T` and
  `grid_t`. Never route it through the parity-compressed Wilson geometry.
- Clover prepare/reconstruct returns `0` on success and nonzero on failure;
  do not turn a failed Schur setup into a plausible output buffer.
- Gaussian gauge, recursive legacy Clover-MG, strict coarse/NULL/MATPC
  primitives, persistent strict V-cycle/FGMRES and MG verify are implemented.
  Keep homogeneous c64/c128 precision and fail closed on unsupported mixed
  precision or CUDA-only mode bits.
- Build and validate all changes with:

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```

- `qcu.map` exposes only the 40 CUDA-compatible `*Qcu` symbols. Keep
  `pyqcu/testing/kupl/check_abi.py` in the CTest suite when changing ABI
  declarations; it compares both headers and the exported dynamic surface.
- `KUPL_NUMA_BIND=1` owns CPU placement; launch MPI with binding disabled
  (`--bind-to none` for OpenMPI). If the scheduler already provides a
  NUMA-local cpuset per rank, keep `KUPL_NUMA_BIND=0`.

- Do not add GPU, LIME/HDF5 I/O, or PyQCU Python changes to this backend
  without a separate interface and validation task.

## 2026-10-21 legacy-MG coarse tile 与 canonical conversion

- `apply_multigrid_coarse_dslash_wide` 按 512 sites 分块，先为整块解析 32 个
  neighbor view，再逐 coefficient stream 连续扫描；避免每个 site 重复方向查找，
  并让 site-major stencil 页面保持热状态。
- `canonical_to_full`/`full_to_canonical` 改为遍历 half-site 并同时发出两个时间
  parity，消除 component-major 索引中的重复四维除法。
- coarse correction 直接返回刚形成的 fine residual norm，避免立即再应用一次全算子。
- 真实 `24³×64` c64、每 rank 4 threads、4-rank y-only、`restart=5`：中位
  `29.942 s`（原 `94.108 s`，3.143x），60 iterations，native relative residual
  `9.70132e-7`，verifier=0；rank0 resident RSS 约 `2.16 GB`。1/2/4-rank 中位为
  `85.204/46.752/29.942 s`，效率 `100%/91.1%/71.1%`。
- coarse correction 必须自适应验收：若 correction 后 true residual 不变、变大或
  非有限，撤销 `parent.correction` 并关闭本 solve 后续 correction，剩余迭代以单个
  长 BiCGStab chunk 完成。生产 `restart=5, deflate=true` 因此从 `29.942 s` 降到
  中位 `16.771 s`，47 iterations，relative residual `9.13844e-7`，verifier=0。
  `bicgstab_chunk` 必须记录实际完成次数，不能用请求 chunk 大小代替。
- `bicgstab_chunk` 的 `(v,v)` 与 `(v,s)` 使用 `mg_dot_pair` 合并为一个四实数组
  `MPI_Allreduce`；每轮全流程集合归约从 6 次降为 5 次，局部点积也在同一 OpenMP
  遍历中完成。真实 4-rank c64 中位 `18.501 -> 17.725 s`（4.19%，1.0438x），
  47 iterations 不变，native relative residual `9.138e-7`，verifier=0；29 项
  CTest 全绿。不得把该通信收益外推为粗算子计算量下降。

## 2026-10-22 2/4-rank efficiency decomposition

固定生产 c64、4 threads/rank 的分段归因显示，4-rank 的 fine Wilson-Clover
rank-local apply 占 solve `73.2%`；Allreduce `7.7%`、canonical conversion
`6.4%`、coarse dslash/halo `1.3%`、vector/other `11.4%`。相对 1-rank 完美扩展
的额外耗时中，fine apply 占 `52.5%`，Allreduce `16.1%`，canonical `9.7%`，
other `21.1%`，coarse 仅 `0.6%`；fine apply 内的 halo 在 4-rank 仅占 `7.55%`。

把每轮 5 次 Allreduce 降到 4 次的候选在 4-rank 三轮 A/B 中为
`15.072 -> 16.681 s`，慢 `10.67%`，虽然仍为 47 iterations、verifier=0 且
native residual 不变，已撤销。不要用减少集合调用次数替代关键路径实测。

## 2026-10-22 fine dslash hotspot and rejected face SIMD

固定 `24×6×24×64` local、4 ranks × 4 threads、c64 的微基准：dslash 内点+周期面
`53.22%`，标量边界 `37.85%`，halo `3.29%`，MPI wait `5.65%`。SIMD block 对
Wilson/Clover/Schur 分别比 scalar 快 `13.98%/6.95%/2.04%`；Clover onsite 占
Clover 算子 `39.36%`，canonical 往返仅约每迭代墙钟 `1.73%`。

仅尝试 y 分域面的连续时间 lane SIMD。三轮 `B-C-C-B-B-C`：baseline 中位
`15.446281 s`、MAD `0.061435 s`；candidate 中位 `16.157466 s`、MAD
`0.715516 s`，慢 `4.60%` 且抖动扩大。六轮均 47 iterations、verifier=0、
PASS，候选已完整撤销。标量边界虽大，y-face SIMD 不是正确切法；后续应优先
检查 scalar boundary 的 rank-face 分布和内存访问，而非继续扩大 SIMD lane。
