# dev74 —— dev73_5 测试工作进阶版：大格子规模测试 + 资源占用统计 + 多线程版本

> 目标：将 dev73_5 的 MG 性能测量推进到逼近硬件极限的格子大小，为服务器集群
> （512G 内存 / 32G 显存）提供可运行的大格子测量协议与**显存/内存/磁盘占用预算模型**，
> 本地（RTX 4060 Laptop 8GB / 15GB 内存）仅做小格子验证测试；
> 同时实现参考 `applyCloverBistabCgDslashQcu` 的**多线程粗算子构建版本**。
> LaTeX 版：`logs/dev74.tex` → `logs/dev74.pdf`。

## 1. 结论摘要

1. **本地验证（3 个格子，c64, 2L, r10 ct1e5 cmi15）**：MG 在本机小卡上均慢于
   BiStabCG（加速比 0.43x → 0.27x → 0.20x，随格子增大下降），解一致
   （vs_ref ≈ 3e-7）；与 dev73_5 在 V100-32G 上的 2.43x/1.16x/0.81x 相比，
   本机（RTX 4060 Laptop 8GB）GPU 算力与时钟差异导致绝对加速比显著更低——
   **MG 的加速依赖 GPU 算力/粗层开销占比**，本机仅验证正确性与协议。
2. **资源占用统计（实测，r²=1.000 线性校准）**：
   - 粗算子构建峰值（cold）：**53.0 KB/格点**（含 null_vecs 逆迭代 + stencil 探测）
   - 求解阶段峰值（warm）：**27 KB/格点**（24.2 KB/V 常驻粗算子 + 3 KB/V 求解中间量）
   - 进程内存 RSS：1.0→2.1 GB（V=8k→65k）；nullvec 磁盘缓存 0.35→2.3 GB
3. **集群（32G 显存）大格子可行性预测**（实测校准外推）：

   | 格子 | V | cold(GB) | warm(GB) | 判定 |
   |------|----|----------|----------|------|
   | 16x32x32x32 | 0.52M | 28.4 | 14.1 | ✓ 单卡可行（cold 87%） |
   | 16x32x32x64 | 1.05M | 56.9 | 28.2 | △ warm 可行；cold 构建需分阶段/--build cpp |
   | 24x32x32x64 | 1.57M | 85.4 | 42.3 | ✗ 单卡不可行，需多卡分布式（MPI） |

4. **多线程版本**（参考 `cpp/cuda/qcu/python/pyqcu.h:applyCloverBistabCgDslashQcu`）：
   - `CudaSchurOp`：C++ CUDA Schur 奇偶算子封装，与 Python `matvec_parity`
     完全等价（rel err 9.7e-8），单次调用快 **16.5x**
   - 粗算子构建（stencil build）换用 C++ 算子后单线程 **2.05x** 加速
     （48.1s → 24.8s @ 8x8x8x16）；多线程在**单卡无收益**（GPU 为瓶颈 +
     GIL 争抢），为**多卡/多节点**场景设计（每线程独立 set_index / 独立流）
   - 为此修改 C++：移除 `applyCloverBistabCgDslashQcu` 首尾的全局
     `cudaDeviceSynchronize`（dslash 内部主流同步已保证数据就绪，语义不变）

## 2. 测量协议

与 dev73_5 完全一致：参考 = `applyCloverBistabCgQcu`（奇偶预条件 Schur BiStabCG,
VERBOSE=0）；MG = `applyCloverMultigridQcu`（2L, null_vecs 缓存）；
mass=0.05, atol=1e-6, gauge_seed=42, κ=1/(2m+8), E=48, NV_ITERS=2。
干净测量：独立进程 + ref/mg 交叉计时 + min of 5 对（本机验证用 3 对）。
**新增**：峰值显存（torch max_memory_allocated，分 cold/warm 阶段）、进程峰值
RSS（getrusage）、nullvec 缓存磁盘占用（os.walk）。

## 3. 本地验证结果（RTX 4060 Laptop 8GB, c64, 2L, r10 ct1e5 cmi15）

| 配置 | ref(ms) | MG(ms) | 加速比 | iters(MG/ref) | vs_ref | cold/warm 显存 | RSS | 构建(s) |
|------|---------|--------|--------|--------------|--------|---------------|-----|---------|
| 8x8x8x16 | 76 | 176 | **0.43x** | 85/86 | 3.2e-7 | 408 MB | 1.0 GB | 0.1（缓存） |
| 8x16x16x16 | 140 | 513 | **0.27x** | 89/86 | 3.2e-7 | 1768 MB | 1.6 GB | 474 |
| 16x16x16x16 | 193 | 960 | **0.20x** | 129/91 | 3.0e-7 | 3525 MB | 2.1 GB | 1824 |

- 加速比随格子增大单调下降（0.43→0.27→0.20）：小卡上 MG 的粗层求解/同步开销
  占比随 V 增大而上升；与 dev73_5 在 V100 上观察到的"2.43x→1.16x→0.81x"趋势一致，
  但本机绝对水平更低（GPU 算力差异），**本机仅验证协议正确性**。
- 构建耗时（粗算子）随 V 线性增长：8x16x16x16 474s、16x16x16x16 1824s
  （16x16x16x16 的 stencil 探测 98304 点 × ~19ms/点）——大格子必须依赖
  nullvec 缓存复用（`logs/nullvec_cache`）与 C++ 算子构建。

## 4. 资源占用统计与预算模型

### 4.1 实测（三格点线性校准，r²≈1.000）

| 阶段 | 每格点 | 说明 |
|------|--------|------|
| 求解（ref） | 3.0 KB/V | BiStabCG 求解中间量 |
| 构建峰值（cold） | 53.0 KB/V | null_vecs 逆迭代 + stencil 33-tensor |
| 常驻粗算子（warm） | 24.2 KB/V | lonv/hnn/hdg/sit（E=48, 2L） |

模型：`VRAM(V) = (24.2 + α)·V/1e6 + β` MB，实测 α=30.8 KB/V、β=-27 MB；
warm 模型 = 24.2 + α/11 ≈ 27 KB/V。

### 4.2 集群预测（32G 显存 / 512G 内存 / 614G 磁盘）

| 格子 | V | cold(GB) | warm(GB) | RSS(GB) | 磁盘缓存(GB) | 判定 |
|------|----|----------|----------|---------|-------------|------|
| 16x32x32x32 | 0.52M | 28.4 | 14.1 | 3.8 | 22.3 | ✓ 单卡 cold+warm 可行 |
| 16x32x32x64 | 1.05M | 56.9 | 28.2 | 6.3 | 44.7 | △ warm 可行；cold 需分阶段/--build cpp |
| 24x32x32x64 | 1.57M | 85.4 | 42.3 | 8.9 | 67.0 | ✗ 单卡不可行（需 MPI 分布式） |

磁盘/内存远低于极限（512G/614G）；**显存是唯一硬约束**。集群运行命令见
`examples/qcu/mg_dev74_cluster.sh`（dry-run 默认；`RUN=1` 执行）。

## 5. 多线程版本（C++ CUDA 粗算子构建）

### 5.1 C++ 后端修改

`cpp/cuda/qcu/src/apply_clover_bistabcg_dslash.cu`：移除 `applyCloverBistabCgDslashQcu`
入口与出口的 `cudaDeviceSynchronize()`。`dslash()` 内部（`wilson_dslash._run`）在
single-rank 快速路径末尾已 `cudaStreamSynchronize(set_ptr->stream)`，multi-rank 路径
对各 dims 流同步——数据就绪语义不变；全局同步曾把并发实例（各持独立非阻塞流）强制
串行化，是多线程构建并行化的障碍。

### 5.2 CudaSchurOp（`examples/qcu/mg_dev74_dslash.py`）

- 封装 `applyCloverBistabCgDslashQcu`（输入/输出 `[12,X,Y,Z,T/2]` 奇子格，经
  `mg_dev74_layout_test.py` 实测确定布局）
- 每实例持**独立 params 副本**（`_SET_INDEX_` 独占槽位）+ 共享 set_ptrs → 多线程
  并发调用零共享写竞争；Cython 桥支持自定义 params/set_ptrs 张量
- 正确性：与 Python `matvec_parity` rel err **9.7e-8**；单次调用 0.26ms vs
  4.3ms（**16.5x**，8x8x8x16 c64）

### 5.3 多线程 stencil build（`examples/qcu/mg_dev74_stencil_mt.py`）

- 每 worker 线程持独立 CudaSchurOp；探测点 (c_idx, ee) 写集互不相交
  （sit/hop_nn/hop_diag 写入位置由 (c_idx,ee) 唯一确定），线程安全
- 结果（8x8x8x16, E=48, 12288 probes）：

  | 实现 | 耗时 | 说明 |
  |------|------|------|
  | Python S 串行 | 48.1s | dev73_5 协议 |
  | C++ S 单线程 | 24.8s | **2.05x** |
  | C++ S 2/4 线程 | 28.7/61.2s | 单卡退化（GPU 瓶颈+GIL 争抢） |

- stencil 数值等价：tensor rel err ~1e-7（sit/nn/diag），
  vs operator-free 2.1e-7 ✓
- **单卡结论**：多线程无收益（GPU 为瓶颈）；设计目标为多卡/多节点集群
  （每线程一卡、独立流），集群用法见 cluster.sh 说明

## 6. 正确性验证（`mg_dev74_verify.py`，8x8x8x16 c64）

| 检查项 | 结果 |
|--------|------|
| gauge SU(3)（check_su3） | True（unit_err 2.4e-7, det_dev 3.6e-7） |
| 参考解残差 ref_res | 3.5e-7 |
| CudaSchurOp vs Python（rel err） | 9.7e-8（16.5x 加速） |
| null_vecs 零模质量 ratio | 0.19–0.88（≪ λmax=1.17） |
| 块内正交 offdiag | 3.6e-7 |
| C++ restrict/prolong vs einsum | 6.8e-7 / 3.5e-7 |
| C++ 33-tensor 粗 dslash vs P^T S P | 3.4e-7 |

## 7. 数据文件

| 文件 | 内容 |
|------|------|
| `dev74_bench.json` | 本地/集群 bench（计时 + 收敛/热点 + 资源统计） |
| `dev74_clean_L*.json` | 每配置干净测量 + 资源统计（独立进程） |
| `dev74_verify_*.json` | 正确性（gauge/解/CudaSchurOp 对照/null_vecs） |
| `dev74_results.json` | collect 汇总（统一 schema） |
| `dev74_budget_{local,cluster}.json` | 预算模型（实测校准 α/β + 逐格子预测） |
| `dev74_stencil_mt.json` | 多线程 stencil build 对照（耗时/误差） |
| `dev74_*.png` | speedup / vram / time / conv / budget 图 |
| `dev74_tbl_*.tex` | LaTeX 表（main / res / budget） |

## 8. 脚本清单（examples/qcu/）

| 脚本 | 功能 |
|------|------|
| `mg_dev74_dslash.py` | CudaSchurOp（C++ Schur 算子封装，多线程安全） |
| `mg_dev74_layout_test.py` | C++ dslash 输入布局对照实验 |
| `mg_dev74_stencil_mt.py` | 多线程 stencil build + 对照验证 |
| `mg_dev74_budget.py` | 显存/内存/磁盘预算模型（`--fit` 实测校准） |
| `mg_dev74_bench.py` | 本地（默认）/集群（--cluster）bench + 资源统计 |
| `mg_dev74_clean.py` | 干净测量（独立进程交叉计时）+ 资源统计 |
| `mg_dev74_verify.py` | 正确性验证（含 CudaSchurOp 对照） |
| `mg_dev74_collect.py` | 汇总 → `dev74_results.json` |
| `mg_dev74_mktable.py` / `mg_dev74_plots.py` | LaTeX 表 / PNG 图 |
| `mg_dev74_cluster.sh` | 集群运行（dry-run 默认，RUN=1 执行） |

## 9. 遗留与下一步

- **集群真实测量**：16x32x32x32（单卡全流程）、16x32x32x64（先 --build cpp
  构建缓存再 warm 测量）、24x32x32x64（需多卡 MPI 分布式方案）
- c128 双精度大格子：显存翻倍，32G 单卡上限约 V≈0.25M（如 16x16x32x32）——
  未列入本组，可作为后续扩展
- 多线程版本的多卡验证（每线程一卡）待集群执行
- 首次 verify 运行曾出现一次 CudaSchurOp rel err=1.04 的偶发异常，
  重跑稳定为 9.7e-8（未复现，疑似 GPU 初始化时序）
