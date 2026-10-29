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
3. 所有 rank 用 HDF5 MPI-IO 把 blocked `lonv` owned slab 写入同一暂存缓存；
4. 每个 rank 按周期 coarse 坐标只读取自己的 overlap patch，不再经 rank 0
   `gather/scatter`；
5. 每个 rank 对 patch 运行 local stencil，计算自己的 `lonv/hnn/hdg/sit` slab；
6. 所有 rank 继续用 HDF5 MPI-IO 写 `hnn/hdg/sit` owned slab，rank 0 只补文件属性、
   流式计算 HDF5 dataset digest 并原子发布；
7. 所有 rank 回读自己的 HDF5 hyperslab，与本地 slab 逐 tensor 比对后发布。

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
  单 rank 子通信域，并将该 `comm` 传入 Legacy halo；正常生产 solve 仍使用完整
  `MPI_COMM_WORLD` process grid。早期固定使用 `MPI_COMM_WORLD` 会让 overlap patch
  与其它 rank 的 halo collective 串序，修复后多 rank patch 可直接在各自 GPU 计算。
- 禁止把多 GiB 的 `local_lonv` 列表经 `comm.gather` 汇聚到 root：这会同时保留
  global tensor、Python object 列表和 pickle 临时副本。`lonv` 的 owned slab 由各
  rank 通过 `h5py.File(..., driver="mpio", comm=comm)` 写入，周期读取按每个轴拆成
  连续 source/destination slices，避免 fancy-index 生成整张临时 patch。
- `lonv` 的时间轴已经过 odd-Schur 压缩。物理 fine 坐标到 cache coarse 的步长是
  `mg_grid[t] * 2`，不是 `mg_grid[t]`；计算 patch center/extent 时必须使用该步长。
- 写入后立即释放 `local_lonv`，避免它与 overlap patch 同时驻留。L48 初次 GPU 化
  试验曾证明只移除 `lonv` root gather 仍不足：`1604062` 在四卡 100% 运行约 530 s
  后于 root 汇总 `hnn/hdg/sit` 时触发 cgroup OOM。现行路径把这三个张量也按 owned
  coarse slab 由 MPI-IO 写入，runner 不再构造任何全局粗算子。
- `build_coarse_assets()` 返回后立即释放 `gauge_patch` 和 `lonv_patch`，再去复制
  rank-local slab。L48 两者分别约 `1.36 GB/rank` 和 `1.53 GB/rank`；延迟到写盘后
  才释放会让四 rank 峰值超过分区默认 54,976 MiB。
- 失败作业留下的仅 `lonv` HDF5 可通过 `--reuse-lonv` 继续：先验证 shape/dtype，
  再 hardlink（跨文件系统才复制）为 staged output。复用路径跳过 null-vector，
  但必须在 patch stencil 前按 shared-memory local rank 设置
  `PYQCU_MPI_DEVICE_ID`，否则 native QCU 会错误绑定 device 0。
- 即使四个 tensor 全走 MPI-IO 并提前释放 staging patch，最终 `1604161` 仍在
  `8m32s` cgroup OOM；batch `MaxRSS=29,410,048 KiB` 但精确进程 RSS 求和在写入时约
  `44.8 GiB`，叠加 HDF5/page cache 后超过 54,976 MiB。分区禁止 4 卡任务超额申请
  memory。因此 L48 在该单节点资源门内仍未发布 cache；不得用首次 `lonv` scratch
  冒充完整 cache。
- 后续内存路径改为逐 tensor 处理：从对应 patch 提取一个 local slab、写入 HDF5、
  计算其 local SHA256，然后立即释放该 patch 和 local slab；重建验签也一次只加载
  一个 dataset。禁止重新一次性构造四份 local slab。
- `1604231` 已完成 L48 GPU stencil（约 `7m38s`），随后暴露并修复了把
  `patch_assets` tuple 当作可变列表的一行错误。该失败发生在写盘开始前，不是 OOM；
  修复后内存路径仍需在下一次作业中完成 cache 发布和独立真残差验证。
- batch 16 将实际 L48 stencil 降到约 `412 s`；`1604374` 使用 chunked HDF5
  write、已完成 `hnn` 复用、staging 提前释放和 page-cache eviction，首次发布
  `18,728,105,984 B` cache，总 setup `499.163295 s`，4/4 rank slab 验签 PASS。
- HDF5 gauge/source 必须由每个 rank 直接读取 own hyperslab；旧 worker 在每个 rank
  全量读取 L48 输入后再切片，是 solve 后 OOM 的重要原因。独立残差还需一次只物化
  一个 Clover parity。修复后的 L8 4-rank cache-hit 独立真残差为 `2.032667e-7`；
  L48 Python residual 仍超过 54,976 MiB，因此只保留 native full-operator gate。
- manifest 必须按固定上限（当前 64 MiB）对 HDF5 dataset 做多维分块摘要，不能只按
  第一维切片；L48 的 `hdg` 单个首维切片本身就可能达到数 GiB。刚由同一进程流式
  生成并记录 digest/file SHA 后，内部结构验签不做第二次无条件全量 digest；
  已发布 cache 的正式外部验签仍保持完整。
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

上表是修复前的历史结果。早期多 rank CUDA `LatticeSet` halo 固定使用
`MPI_COMM_WORLD`，与独立 serial overlap patch 的 collective 顺序冲突，因此当时
暂以 CPU patch 保证正确性。现在 LatticeSet 在 `NODE_SIZE=1`、process-grid product
为 1、world size 大于 1 时创建私有 singleton communicator，Clover 和 coarse halo
均使用 `set_ptr->comm`。修复后的 4-rank `8x8x8x16`、grid `2x2x1x1` 实测：

| phase | seconds |
|---|---:|
| HDF5 MPI-IO lonv write/read | `0.354144` |
| GPU patch stencil | `0.137035` |
| HDF5 MPI-IO hnn/hdg/sit write | `0.884016` |
| total setup | `3.040447` |

报告记录 `stencil_device=cuda:0`，4/4 rank slab 重建验签通过；该 cache 随即由同
进程网格 KUPL 直接加载，`verify_status=0`，独立真残差 `4.110297e-7`。因此当前
多 rank CUDA builder 不再使用 CPU patch stencil。

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
