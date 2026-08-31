---
name: src
description: cpp/cuda/qcu/src 目录的完整生成 skill：.cu 内核源文件（include 对应头 + 模板实例化 + 启动封装）。
---
# cpp/cuda/qcu/src

CUDA kernel source files. Each `.cu` file `#include`s the corresponding header from `../include/` and provides template instantiations and kernel launch wrappers.

## Files

| File | Purpose |
|------|---------|
| `apply_init.cu` / `apply_end.cu` | Memory allocation/free lifecycle |
| `apply_dslash.cu` | Dslash dispatch (Wilson or Clover based on plan) |
| `wilson_dslash.cu` | Wilson dslash kernel |
| `clover_dslash_single.cu` / `clover_dslash_multi.cu` / `clover_dslash_comm.cu` | Clover dslash: single-GPU, multi-GPU, halo exchange |
| `apply_wilson_bistabcg.cu` / `apply_wilson_bistabcg_dslash.cu` | Wilson BiStabCG solver + its dslash |
| `apply_wilson_cg.cu` / `apply_wilson_cg_dslash.cu` | Wilson CG solver + its dslash |
| `apply_clover_bistabcg.cu` / `apply_clover_bistabcg_dslash.cu` | Clover BiStabCG solver + its dslash |
| `apply_multigrid.cu` | MG restrict/prolong/coarse-dslash |
| `apply_clover_multigrid.cu` | Clover multigrid solver entry (C API bridge) |
| `apply_multigrid_strict.cu` | Strict full-coarse `X/Y/Yhat`, MATPC、prepare/reconstruct、逐层 full-coarse R/P、持久递归 V-cycle 与 fused right-FGMRES |
| `lattice_mpi.cu` | MPI halo exchange helpers |
| `lattice_cuda.cu` | CUDA utility functions |

2026-08-24 存档：gauss_gauge 非 verbose 路径 OOB write（分配 `_LAT_S_` 却写 32 元素）与
device_random_8dtzyx 显存泄漏（cudaFreeAsync）均已修复（`logs/fix-report-2026-07-28.md` §7.1/§7.4）。

## Strict source contract

`apply_multigrid_strict.cu` 直接包含 `../include/qcu.h` 与 `../python/pyqcu.h`，
并导出 `pyqcu.h` 中的 Strict C ABI。它验证 c64/c128、单 rank、偶数 fine geometry
和可整除 block；`target_parity=1`、`start_level=1` 是 fused fine solve 的固定值。
Fine 端用物理 Gauge 与 Clover even/odd/inverse 做 odd-Schur/MATPC；每个 coarse
level 用 full-lattice `X/Yhat` 算子，R/P 只跨 compact fine parity 与 full coarse
field，不对 coarse geometry 再做 checkerboard。

Strict init 分配并持有递归 hierarchy/V-cycle arena；fused FGMRES 的外层 arena
在首次 solve 懒分配、按相同 geometry/restart 复用，end 释放两者。其 exact 外层
字节数为 `(2*m+5)*B_f + 2*B_c`；返回的 `allocated_bytes` 只涵盖 C++ 自有 arena，
不能替代全设备显存账本。
