---
name: pyqcu-cpp-qcu-src
description: 当任务涉及 cpp/cuda/qcu/src 的 .cu 内核源、模板实例化或启动封装时使用。
---
# cpp/cuda/qcu/src

CUDA kernel source files. Each `.cu` file `#include`s the corresponding header from `../include/` and provides template instantiations and kernel launch wrappers.

CUDA/GPGPU production topology is one Python process = one MPI rank = one
device. Do not add or invoke a single-process multi-device production driver;
`MultiGpuMultigrid` is test-only. See
[`pyqcu-cuda/SKILL.md`](../pyqcu-cuda/SKILL.md#cuda-production-topology).

## C API 入口层（定义 `python/pyqcu.h` 的 `extern "C"` 符号）

| File | Purpose |
|------|---------|
| `apply_init.cu` / `apply_end.cu` | Memory allocation/free lifecycle（`applyInitQcu`/`applyEndQcu`） |
| `apply_dslash.cu` | Dslash dispatch (Wilson or Clover based on plan，`applyDslashQcu`) |
| `apply_test.cu` | 正确性自测入口 `testWilsonDslashQcu`/`testCloverDslashQcu` |
| `apply_wilson_dslash.cu` | `applyWilsonDslashQcu` |
| `apply_clover_dslash.cu` | `applyCloverDslashQcu` |
| `apply_wilson_bistabcg.cu` / `apply_wilson_bistabcg_dslash.cu` | Wilson BiStabCG solver + its dslash |
| `apply_wilson_cg.cu` / `apply_wilson_cg_dslash.cu` | Wilson CG solver + its dslash |
| `apply_clover_bistabcg.cu` / `apply_clover_bistabcg_dslash.cu` | Clover BiStabCG solver + its dslash |
| `apply_clover.cu` / `apply_clovers.cu` | Clover 项构造与逆（`applyCloverQcu`/`applyCloversQcu`） |
| `apply_gauss_gauge.cu` | `applyGaussGaugeQcu` 高斯规范场生成入口 |
| `apply_laplacian.cu` | `applyLaplacianQcu` |
| `apply_multigrid.cu` | Legacy MG restrict/prolong/coarse-dslash |
| `apply_clover_multigrid.cu` | Legacy Clover multigrid 求解入口（`applyCloverMultigridQcu` C API 桥） |
| `apply_clover_multigrid_verify.cu` | Legacy MG 与参考实现对照（`verifyCloverMultigridQcu`） |
| `apply_multigrid_strict.cu` | Strict full-coarse `X/Y/Yhat`、MATPC、prepare/reconstruct、逐层 full-coarse R/P、持久递归 V-cycle 与 fused right-FGMRES |

## 内核实现层（被上面的入口/头文件调用，不直接导出 C 符号）

| File | Purpose |
|------|---------|
| `wilson_dslash.cu` | Wilson dslash kernel |
| `clover_dslash_single.cu` / `clover_dslash_multi.cu` / `clover_dslash_comm.cu` | Clover dslash: single-GPU, multi-GPU, halo exchange |
| `bistabcg.cu` | BiCGStab 标量/向量更新内核（`bistabcg_give_1beta/_1rho_prev/_1alpha/_1omega/_1diff2`…） |
| `cg.cu` | CG 标量/向量更新内核（`cg_give_*`） |
| `multigrid.cu` | Legacy MG 内核：restrict/prolong(+cast)、coarse pack/unpack halo |
| `gauss_gauge.cu` | `give_random_8dtzyx`、`_make_gauss_gauge` |
| `laplacian.cu` | Laplacian 内核 + 逐维 send/recv |
| `lattice_set.cu` | `give_param`（params → 设备常量的模板实例化） |
| `lattice_mpi.cu` | MPI halo exchange helpers |
| `lattice_cuda.cu` | CUDA utility functions |

2026-08-24 存档：gauss_gauge 非 verbose 路径 OOB write（分配 `_LAT_S_` 却写 32 元素）与
device_random_8dtzyx 显存泄漏（cudaFreeAsync）均已修复（`logs/fix-report-2026-07-28.md` §7.1/§7.4）。

## Strict source contract

`apply_multigrid_strict.cu` 直接包含 `../include/qcu.h` 与 `../python/pyqcu.h`，
并导出 `pyqcu.h` 中的 Strict C ABI。它验证 c64/c128、MPI rank 对称几何、偶数
fine geometry 和可整除 block；`target_parity=1`、`start_level=1` 是 fused fine
solve 的固定值，多 rank 通过 fine/coarse halo 与全局归约协同。
Fine 端用物理 Gauge 与 Clover even/odd/inverse 做 odd-Schur/MATPC；每个 coarse
level 用 full-lattice `X/Yhat` 算子，R/P 只跨 compact fine parity 与 full coarse
field，不对 coarse geometry 再做 checkerboard。

Strict init 分配并持有递归 hierarchy/V-cycle arena；fused FGMRES 的外层 arena
在首次 solve 懒分配、按相同 geometry/restart 复用，end 释放两者。其 exact 外层
字节数为 `(2*m+5)*B_f + 2*B_c`；返回的 `allocated_bytes` 只涵盖 C++ 自有 arena，
不能替代全设备显存账本。
