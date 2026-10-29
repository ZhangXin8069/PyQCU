# Multi-MPI Legacy-MG cache builder（2026-10-29）

本文记录 `pyqcu.testing.simple.mpi.build_full_mg_cache` 的多 MPI 实现、
`8x8x8x16` 正确性门和当前兼容边界。本文属于当前工作树阶段证据，尚未打 tag，
不能替代最终正式性能和跨平台结论。

## 1. 执行模型

规范入口：

```bash
mpirun --allow-run-as-root --oversubscribe --bind-to none -np 4 \
  python -B -m pyqcu.testing.simple.mpi.build_full_mg_cache \
  --gauge gauge.h5 --lat X Y Z T --mass M --csw C --dtype c64 \
  --dof 12 E --mg-grid 2 2 2 2 --grid GX GY GZ GT \
  --output cache.h5 --report cache.json
```

父目录的 `pyqcu/testing/simple/build_full_mg_cache.py` 保留串行兼容入口；多 MPI
路径必须使用 `simple/mpi/build_full_mg_cache.py`，不能只让 rank 0 构建后广播。

每个 rank 执行以下步骤：

1. 按全局 process grid 读取自己的 gauge hyperslab；
2. 用分布式 Schur/Clover 算子生成本 rank 的 null-vector slab；
3. rank 0 汇总 blocked `lonv`，按 coarse slab 给各 rank 分发周期 overlap patch；
4. 每个 rank 只对 patch 运行 local stencil，输出自己的 `lonv/hnn/hdg/sit` slab；
5. rank 0 组装全局 33-tensor HDF5、统一 manifest 与 SHA256；
6. 所有 rank 回读自己的 HDF5 hyperslab，与本地 slab 逐 tensor 比对后发布。

overlap patch 的 coarse halo 为每个方向两个 coarse block；当窄轴无法容纳该 halo
时，该轴使用完整周期范围，但仍保持 coarse-block 对齐。patch 不是完整 hierarchy，
不得退化为每个 rank 重建全题。

## 2. 布局与通信域修复

- `lonv` 的 coarse 轴是交错布局 `(2,4,6,8)`：
  `[E,12,X,bx,Y,by,Z,bz,T/2,bt]`。不能按普通 lattice-axes-last-4 使用
  `(6,7,8,9)` 切片。
- `hnn`、`hdg`、`sit` 的 coarse 轴分别为 `(4,5,6,7)`、`(5,6,7,8)`、
  `(2,3,4,5)`。
- KUPL patch stencil 在全局 MPI 作业内部需要独立的 `1x1x1x1` LatticeSet context。
  `make_geometry()` 现在仅在 `params[NODE_SIZE]=1` 且 grid product 为 1 时创建
  单 rank 子通信域；正常生产 solve 仍使用完整 `MPI_COMM_WORLD` process grid。
- 所有粗层组装和 slab 回读均通过同一 `_legacy_mg_cache.py` manifest/shape/digest
  契约完成，不引入第二套后端私有 cache 格式。

## 3. 正确性结果

输入为真实 L24 组态前缀：

- lattice：`8x8x8x16`
- dtype：c64
- mass：`-0.277`
- kappa：`0.13430029546065003`
- CSW：`1.160920226`
- `dof=[12,2]`
- `mg_grid=[2,2,2,2]`
- `nv_iters=1`, seed `42`
- gauge file SHA256：`93f3153673f6846b8256d98f6bd4361880aee83eb35f7a00c36389ae6c847479`
- gauge canonical SHA256：`01baadbfa2dcab104126cb53b13c5d3627ec313b657ab7261f7fde9b5b33786d`

| ranks / grid | builder | cache bytes | slab reconstruction |
|---:|---|---:|---|
| 1 / `1x1x1x1` | PASS | 1,064,960 | PASS |
| 2 / `1x2x1x1` | PASS | 1,064,960 | PASS, 2/2 ranks |
| 4 / `1x4x1x1` | PASS | 1,064,960 | PASS, 4/4 ranks |

1-rank setup wall 为 `11.900925 s`；2-rank 为 `11.527726 s`；4-rank 为
`11.219679 s`。这些数字来自共享登录节点的短样本，且窄 y 轴使用完整 overlap，
只用于构建链路验收，不得作为生产并行效率或加速比。

2/4-rank 与 1-rank 的功能资产相对差如下（Frobenius norm）：

| ranks | `lonv` | `hnn` | `hdg` | `sit` |
|---:|---:|---:|---:|---:|
| 2 | `2.376222e-5` | `1.510794e-5` | `2.245565e-5` | `2.397456e-6` |
| 4 | `2.065720e-5` | `1.288437e-5` | `1.948163e-5` | `2.006236e-6` |

差异来自 MPI 全局归约顺序；不能要求不同 rank 数的 SHA 位级相同，也不能用
`SHA 不同` 单独判定物理错误。结构性错误应表现为远大于该量级的 tensor 差异、
Galerkin 失配或独立真残差失败。

CUDA builder 同样完成 1/2/4 MPI。CUDA distributed null-vector 在每 rank 生成局部
正交基，因此不同 rank 数的原始 tensor 不要求数值近似等价；验收改为对应 rank 数的
KUPL solve 直接加载 cache 并检查 verifier 与独立真残差：

| ranks / grid | CUDA setup | KUPL verify | independent true residual |
|---:|---:|---:|---:|
| 1 / `1x1x1x1` | 0.659315 s | 0 | `2.016388e-7` |
| 2 / `1x2x1x1` | 9.655451 s | 0 | `2.018354e-7` |
| 4 / `1x4x1x1` | 10.264782 s | 0 | `2.020363e-7` |

CUDA rank 1 的 patch stencil 在 GPU 上执行。多 rank CUDA `LatticeSet` 的 Clover
halo 仍绑定 `MPI_COMM_WORLD`，与 overlap patch 的串行语义冲突，实际会在不同
collective 间死锁；因此多 rank 的 null-vector 仍在 GPU 分布式构建，而互相独立的
patch stencil 在 CPU 执行。报告中的 `stencil_device=cpu` 是显式 provenance，
不能把该混合 setup 时间外推成纯 GPU builder 性能。

本轮修复后 CUDA 库 SHA256 为
`e881d9a7d5d12712c869e673c2f0f05f4e8ac041ede922a896122b7db4103746`，
KUPL 库 SHA256 为
`4bd1f56cadd5564aeb9c28a00e84a01a01c9c7e568adb7b563c3358e1f17b3fd`。

## 4. 8x8x8x16 求解门

KUPL 直接加载 1-rank canonical E2 cache：

- `levels=2`, `restart=5`, `max_iter=200`, `atol=1e-6`
- `warmup=1`, `repeat=3`
- median wall：`0.867535 s`
- native relative residual：`2.95185e-7`
- independent NumPy true residual：`2.877276e-7`
- `verify_status=0`
- coarse correction 最终由 fine parity-Schur fallback 完成，不能写成 coarse-only
  收敛。

同格点 E12 独立构建的 Gram 误差为 `1.549721e-6`，Galerkin 误差为
`5.804606e-7`，setup `73.796 s`，独立真残差同为 `2.877276e-7`。

Torch 只执行 CPU 小格正确性：

- lattice `8x8x8x16`, c64, 1 MPI rank, `torch-device=cpu`
- `levels=1`, `restart=0`, `atol=1e-6`, `max_iter=200`
- `omp_threads=4`, warmup `0`, repeat `1`
- 15 iterations
- wall：`1.882129 s`
- independent true residual：`4.453952e-7`
- status：PASS

该结果不进入 GPU 性能比较，也不用于证明 Torch CUDA/NPU 路径性能。

## 5. CUDA-to-KUPL Legacy cache 直接复用

旧 L8 fixture 的 `1.3507681e-2` 失配不是 KUPL 方向槽序错误。追溯发现 CUDA
`applyCloversQcu` 在 `applyInitQcu` 后从 `params` 切换到 odd parity 时，没有把新
parity 同步到 native `LatticeSet` 和 device params，导致 CUDA 的 even/odd Clover
被生成为同一份。修复后：

- CUDA `clover_even` 与 `clover_odd` 的 Frobenius 差异为 `1.9179139e-1`；
- KUPL even/odd Clover 与 CUDA 对应项的误差分别为 `4.898e-8/4.908e-8`；
- 同一 gauge、同一 CUDA `lonv` 下，KUPL stencil 的 `hnn/hdg/sit` 相对误差为
  `2.480e-7/7.341e-8/2.525e-7`；
- KUPL 直接加载 CUDA 生成的 canonical E2 cache 完成 solve，
  `verify_status=0`，独立 true residual `2.016388e-7`。

因此 KUPL 无需转换或复制 Legacy 33-tensor cache payload。旧的无 identity E48
fixture 不再作为跨后端兼容门，规范回归改用同一 gauge 的 CUDA canonical cache。

## 6. 复现与证据

证据目录：

`data/qcu_mg_cache_mpi_validation_20261029/`

关键文件：

- `gauge_8x8x8x16_c64.h5`
- `cache_rank1_E2_v2.h5` / `.json`
- `cache_rank2_E2_v3.h5` / `.json`
- `cache_rank4_E2_v1.h5` / `.json`
- `cache_cuda_rank1_E2_fixed.h5` / `.json`
- `cache_cuda_rank2_E2_fixed.h5` / `.json`
- `cuda_clover_parity.h5`
- `kupl_from_cuda_report.json`
- `torch_cpu_correctness/report.json`

相关回归：

```bash
python -m pytest -q pyqcu/testing/simple/mpi/test_mpi_contract.py
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
python -m pytest -q pyqcu/testing/kupl
```

CUDA parity/CSW 回归为 `9 passed`，KUPL pytest 为 `43 passed`。在单 NUMA
`taskset -c 0-23` cpuset 下 CTest 为 `34/34 PASS`；无 `taskset` 时 Kunpeng
profile 会因双 NUMA cpuset 正确 fail-closed。
