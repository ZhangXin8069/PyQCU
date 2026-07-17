# CUDA C++ 版 MultiGrid 编写与优化解析文档

> 基于 PyQCU 完整代码库分析 | 2026-07-17

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [Python 层 MultiGrid 完整实现](#2-python-层-multigrid-完整实现)
3. [C++ CUDA 后端现有基础设施](#3-c-cuda-后端现有基础设施)
4. [CUDA C++ MultiGrid 的设计方案](#4-cuda-c-multigrid-的设计方案)
5. [CUDA C++ MultiGrid 的优化策略](#5-cuda-c-multigrid-的优化策略)
6. [实现路线图](#6-实现路线图)

---

## 1. 总体架构概览

### 1.1 项目结构

PyQCU 是一个 Python/Cython 封装的格点 QCD 库，核心计算涉及 Wilson/Clover Dirac 算子在 4D 格点上的作用。MultiGrid 求解器横跨 Python 和 C++ CUDA 两层：

```
pyqcu/
├── solver/_multigrid.py    ← Python 层 MultiGrid 主逻辑 (336行)
├── solver/_bistabcg.py     ← Python 层 BiStabCG 平滑器 (73行)
├── dslash/_operator.py     ← Dirac 算子 (hopping + sitting) (335行)
├── tools/_multigrid.py     ← 延拓/限制/正交化工具 (253行)
└── cuda/
    ├── define.py           ← C++/Python 参数索引镜像
    └── qcu/qcu.pyx         ← Cython 桥接层

cpp/cuda/qcu/
├── src/multigrid.cu        ← **空文件** — CUDA MultiGrid 待实现
├── src/bistabcg.cu         ← BiStabCG 标量更新的 CUDA kernel
├── src/wilson_dslash.cu    ← Wilson dslash 及 MPI 通信 kernel (~1400行)
├── src/clover_dslash_*.cu  ← Clover dslash kernel (single/multi/comm)
├── src/apply_*.cu          ← Cython 入口函数 (init/end/dslash/bistabcg)
├── include/define.h        ← 参数枚举定义 (含 MG 参数预留)
├── include/lattice_set.h   ← 格点几何、MPI、CUDA 资源管理
├── include/lattice_wilson_bistabcg.h  ← Wilson BiStabCG 完整实现
└── include/lattice_clover_bistabcg.h  ← Clover BiStabCG 完整实现
```

### 1.2 当前 MultiGrid 状态

| 组件 | 状态 | 位置 |
|------|------|------|
| Python MultiGrid 求解器 | ✅ 完整 | `pyqcu/solver/_multigrid.py` |
| 格点层级构建 | ✅ Python | `multigrid.__init__` |
| 近零空间向量生成 | ✅ Python | `tools/_multigrid.py:give_null_vecs` |
| 局部正交化 | ✅ Python (QR) | `tools/_multigrid.py:local_orthogonalize` |
| 延拓 (Prolongation) | ✅ Python (einsum) | `tools/_multigrid.py:prolong` |
| 限制 (Restriction) | ✅ Python (einsum) | `tools/_multigrid.py:restrict` |
| 粗网格算子构建 | ✅ Python | `dslash/_operator.py:operator.__init__` (Galerkin) |
| BiStabCG 平滑器 | ✅ Python / ✅ CUDA C++ | `_bistabcg.py` / `lattice_clover_bistabcg.h` |
| Wilson/Clover dslash | ✅ CUDA C++ | `wilson_dslash.cu` / `clover_dslash_*.cu` |
| MPI 通信 | ✅ CUDA C++ | `lattice_mpi.cu` |
| CUDA MultiGrid 主循环 | ❌ **空文件** | `cpp/cuda/qcu/src/multigrid.cu` |
| CUDA 延拓/限制 kernel | ❌ 不存在 | — |
| CUDA 粗网格算子构造 | ❌ 不存在 | — |
| CUDA 层级管理 | ❌ 不存在 | — |

**关键发现**: `define.h` 中已经预留了 MultiGrid 的全部参数索引 (`_MG_NUM_LEVEL_` 到 `_MG_LEVEL4_NUM_RESTART_`)，但 `multigrid.cu` 是空文件。MultiGrid 完全在 Python 层运行，仅最细层 BiStabCG dslash 可调用 CUDA 后端加速。

---

## 2. Python 层 MultiGrid 完整实现

### 2.1 类结构与初始化 (`multigrid.__init__`)

```python
class multigrid:
    def __init__(self, dtype_list, device_list, U, clover_term,
                 kappa, u_0, clover_ee_inv, clover_oo_inv,
                 min_size=4, max_level=4, mg_grid_size=[2,2,2,2],
                 num_convergence_sample=50, dof_list, tol, max_iter,
                 num_restart, root=0, support_parity=False, verbose=True):
```

**关键参数**:
- `dtype_list/device_list`: 每层可独立选择精度(complex64/128)和设备(CPU/GPU)
- `mg_grid_size`: 粗化因子，默认 `[2,2,2,2]` (每方向减半 → 体积 ×1/16)
- `dof_list`: 每层自由度，如 `[12, 24, 24, 24]` (精细层 spin×color=12，粗层可用更多)
- `min_size=4`: 最小格点尺寸，决定层级数量
- `num_restart=5`: 每 5 次平滑迭代后触发一次 coarse-grid correction
- `with_cuda_qcu`: 当提供 `clover_ee_inv` 和 `clover_oo_inv` 时启用 CUDA 加速

**层级构建**:
```python
self.lat_size_list = []
_lat_size = self.lat_size  # [Lx, Ly, Lz, Lt]
while all(_ >= self.min_size for _ in _lat_size) and len(self.lat_size_list) < self.max_level:
    self.lat_size_list.append(_lat_size)
    _lat_size = [_lat_size[d] // self.mg_grid_size[d] for d in range(4)]
self.num_level = len(self.lat_size_list)
```

### 2.2 初始化流程 (`multigrid.init`)

```
Level 0: 已有 fine operator (Wilson/Clover Dirac)
         ↓
For i = 1 to num_level-1:
  1. 生成随机近零向量 null_vecs[i]  [dof[i], dof[i-1], Lx, Ly, Lz, Lt]
  2. 用 BiStabCG 光滑化: v = v - A^{-1} A v
     (最细层可选 CUDA BiStabCG 加速)
  3. 局部正交化: lonv[i] = local_orthogonalize(null_vecs[i])
     在每个粗格点做 QR 分解 → 正交归一基
  4. Galerkin 投影构建粗网格算子:
     op[i] = dslash.operator(
         fine_hopping=op[i-1].hopping,
         fine_sitting=op[i-1].sitting,
         local_ortho_null_vecs=lonv[i]
     )
```

**Galerkin 粗网格算子构建详解** (`dslash/_operator.py` 第 153-217 行):

对粗网格自由度的每个基向量 e:
1. 构造只在 e 分量、特定 ward 方向的粗网格源 `_src_c`
2. 用 `prolong(lonv, _src_c)` 延拓到精细网格
3. 用精细 hopping 算子作用 (plus/minus 方向)
4. 用 `restrict(lonv, _dest_f)` 限制回粗网格
5. 同理用精细 sitting 算子作用并限制

最终得到粗网格上的 hopping 矩阵 `M_plus_list/M_minus_list` 和 sitting 矩阵 `M`。

**数学**: 若精细算子为 D_f，近零空间基为 V，则粗网格算子 D_c = V^† D_f V。

### 2.3 V-Cycle 主循环 (`multigrid.cycle`)

```
function cycle(level):
    if level == 0 and support_parity:
        做 even-odd 预处理: b = b_o - κ D_oe A_ee^{-1} b_e
        matvec = A_oo - κ² D_oe A_ee^{-1} D_eo  (Schur 补)

    x = 0, r = b - A·x

    for i in range(max_iter):
        # --- BiStabCG 平滑 (pre-smoothing + post-smoothing) ---
        ρ   = ⟨r̃, r⟩
        β   = (ρ/ρ_prev) · (α/ω)
        p   = r + β(p - ω·v)
        v   = A·p
        α   = ρ / ⟨r̃, v⟩
        s   = r - α·v
        t   = A·s
        ω   = ⟨t, s⟩ / ⟨t, t⟩
        x   = x + α·p + ω·s
        r   = s - ω·t

        # --- Coarse-Grid Correction (每 num_restart 次) ---
        if count_restart > num_restart and level < num_level-1:
            r_coarse = restrict(lonv[level], r)
            e_coarse = cycle(level+1)     # 递归
            e_fine   = prolong(lonv[level], e_coarse)
            x = x + e_fine
            r = b - A·x
            count_restart = 0

        # --- 收敛检查 + 自适应层级缩减 ---
        if r_norm < tol: break
```

### 2.4 自适应层级缩减 (`multigrid.adaptive` + `multigrid.levels_back`)

```python
def adaptive(self, iter):
    if self.convergence_tol > 3:
        self.num_level = 1  # 直接降到最细层 (退化为纯 BiStabCG)
    # 比较当前残差与最近 num_convergence_sample 次残差
    # 如果超过半数历史残差更小 → 收敛停滞 → convergence_tol += 1

def levels_back(self):
    self.convergence_tol = 0
    self.num_level = min(len(self.op_list), self.max_level)
```

**策略**: 当收敛停滞超过阈值时，自动减少 V-cycle 深度，极端情况下直接退化为纯 BiStabCG。这是一种实用的自适应策略，避免在粗网格修正无效时浪费计算。

### 2.5 延拓/限制算子 (`tools/_multigrid.py`)

**延拓 (Prolongation)**:
```python
# 数学: e_fine(x) = Σ_E V_E(x) · e_coarse_E
# 实现: einsum("EeXxYyZzTt, EXYZT -> eXxYyZzTt", lonv, coarse_vec)
```

- `lonv`: shape `[E, e, X, x, Y, y, Z, z, T, t]`
  - E: 粗网格自由度
  - e: 精细网格自由度 (spin×color=12)
  - X,x: 粗/细格点索引
- 将粗网格向量 `[E, X, Y, Z, T]` 扩展到精细网格 `[e, X*x, Y*y, Z*z, T*t]`

**限制 (Restriction)**:
```python
# 数学: r_coarse_E = Σ_x V*_E(x) · r_fine(x)  (共轭转置)
# 实现: einsum("EeXxYyZzTt, eXxYyZzTt -> EXYZT", lonv.conj(), fine_vec)
```

**NPU 兼容**: 由于 Ascend NPU 不支持超过 8 维的 tensor，代码额外提供了 `_npu` 版本的函数，通过 reshape/permute 将 10 维操作降为 ≤8 维。

### 2.6 Python BiStabCG 平滑器 (`_bistabcg.py`)

```python
def bistabcg(b, matvec, tol=1e-6, max_iter=1000, x0=None):
    x = x0 or randn_like(b)
    r = b - matvec(x)
    r_tilde = r.clone()   # 影子残差 (BiCG 部分)
    p = v = s = t = 0
    ρ_prev = α = ω = 1.0

    for i in range(max_iter):
        ρ   = vdot(r_tilde, r)
        β   = (ρ/ρ_prev) * (α/ω)
        p   = r + β*(p - ω*v)          # BiCG 方向更新
        v   = matvec(p)                 # 矩阵向量乘
        α   = ρ / vdot(r_tilde, v)
        s   = r - α*v                   # BiCG 残差
        t   = matvec(s)                 # 第二个矩阵向量乘
        ω   = vdot(t,s) / vdot(t,t)     # GCR(1) 步
        x   = x + α*p + ω*s             # 解更新
        r   = s - ω*t
        if norm(r) < tol: break
    return x
```

**算法特点**: BiStabCG = BiCG + GCR(1) 稳定化。每次迭代需要 **2 次矩阵向量乘**。

---

## 3. C++ CUDA 后端现有基础设施

### 3.1 参数系统 (`define.h`)

C++ 和 Python 通过两个平坦数组传递参数，索引必须严格同步:

| 数组 | 类型 | 长度 | 用途 |
|------|------|------|------|
| `params` | `int32[54]` | `_PARAMS_SIZE_=54` | 格点尺寸、网格、数据类型、迭代参数、**MG 层级参数** |
| `argv` | `float/double[7]` | `_ARGV_SIZE_=7` | 物理参数 mass、atol、sigma、**MG 层级 atol** |
| `set_ptrs` | `int64[100]` | `_SET_PTRS_SIZE_=100` | 指向 `LatticeSet` 实例的指针数组 |

**MG 相关 params 索引** (已预留):

```
_MG_NUM_LEVEL_        = 17   # 总层级数
_MG_LEVEL_INDEX_      = 18   # 当前层级
_MG_LEVEL1_E_  ... _MG_LEVEL1_T_       = 19-23
_MG_LEVEL1_MAX_ITER_                   = 24
_MG_LEVEL1_DATA_TYPE_                  = 25
_MG_LEVEL1_NUM_RESTART_                = 26
# ... 同理 LEVEL2/3/4 (27-50)
_MG_PARAMS_SIZE_      = 8     # 每层 8 个参数
```

**MG 相关 argv 索引** (已预留):

```
_MG_LEVEL1_ATOL_  = 3
_MG_LEVEL2_ATOL_  = 4
_MG_LEVEL3_ATOL_  = 5
_MG_LEVEL4_ATOL_  = 6
```

### 3.2 资源管理层 (`LatticeSet<T>`)

`LatticeSet` 是 C++ 端的核心资源管理器，一次性分配所有需要的 CUDA/MPI 资源:

**几何计算**:
- 4D 格点索引分解: `(x,y,z,t)` ↔ 线性索引
- 邻居 rank 计算: `move_wards[8]` 用于 4 方向的 backward/forward 通信
- 2D 面通信 rank: `move_wards[24]` 用于 Clover dslash 的棱/角通信

**CUDA 资源**:
- `stream`: 主计算流
- `streams[4]`: 4 个独立流 (用于 BiStabCG 内积流水线)
- `stream_dims[4]`: 4 个独立流 (用于 dslash 4 维通信并行)
- `cublasH`: 主 cuBLAS 句柄
- `cublasHs[4]`: 4 个 cuBLAS 句柄 (每流一个)

**内存分配** (按 plan 层级递增):
| Plan | 分配内容 |
|------|----------|
| `_SET_PLAN_N_2_` (-2, Laplacian) | 4 维 send/recv buffer (3 色) |
| `_SET_PLAN0_` (0, Wilson dslash) | 4 维 send/recv buffer (spin×color/2) |
| `_SET_PLAN1_` (1, BiStabCG/CG) | + vec0/vec1/vec2/device_vals (工作向量) |
| `_SET_PLAN2_` (2, Clover dslash) | + gauge 1D/2D send/recv buffer (parity×color) |

**接口函数** (Cython 暴露):
```c
applyInitQcu(set_ptrs, params, argv)    // 创建 LatticeSet
applyEndQcu(set_ptrs, params)           // 销毁 LatticeSet
applyWilsonDslashQcu(...)               // Wilson dslash
applyCloverDslashQcu(...)               // Clover dslash
applyWilsonBistabCgQcu(...)             // Wilson BiStabCG
applyCloverBistabCgQcu(...)             // Clover BiStabCG (完整求解)
applyCloverBistabCgDslashQcu(...)       // Clover BiStabCG dslash (仅算子)
```

### 3.3 Wilson Dslash Kernel (`wilson_dslash.cu`)

**核心 kernel `wilson_dslash`** — 单 GPU 内 (非边界) 的 Wilson dslash:

```cuda
// 每线程处理一个时空点
// 对 4 个方向 (X,Y,Z,T) × 2 个朝向 (forward/backward):
//   backward: load U(x-μ, μ), src(x-μ) → U^† × (1±γ_μ) × src
//   forward:  load U(x, μ),   src(x+μ) → U   × (1∓γ_μ) × src
```

**分步实现** (分离 MPI 通信和本地计算):

1. `wilson_dslash_x_send/recv` — X 方向通信
   - send: 边界面数据打包 (带 gamma 矩阵投影)
   - recv: 接收邻居面数据，完成 SU(3) 矩阵乘
2. `wilson_dslash_y_send/recv` — Y 方向
3. `wilson_dslash_z_send/recv` — Z 方向
4. `wilson_dslash_t_send/recv` — T 方向 (带 even-odd 处理)
5. `wilson_dslash_inside` — 内部点 (不涉及通信)

**MPI 通信流水线** (`lattice_mpi.cu`):
```
for each direction:
    // 异步: 启动 GPU kernel 准备发送数据
    wilson_dslash_x_send<<<stream_dim[x]>>>()
    // 异步: GPU → CPU memcpy
    cudaMemcpyAsync(host_send, device_send, ...)
    // 同步: MPI Sendrecv
    MPI_Sendrecv(host_send → neighbor_plus, host_recv ← neighbor_minus)
    // 异步: CPU → GPU memcpy
    cudaMemcpyAsync(device_recv, host_recv, ...)
    // 异步: GPU kernel 处理接收数据
    wilson_dslash_x_recv<<<stream_dim[x]>>>()
```

4 个方向可以使用 4 个独立 CUDA stream 部分重叠。

### 3.4 Clover BiStabCG (`lattice_clover_bistabcg.h`)

**Clover dslash 算子** (`LatticeCloverBistabCg::dslash`):

```cpp
void dslash(void *fermion_out, void *fermion_in) {
    // A_oo · src_o  -  κ² · D_oe · A_ee⁻¹ · D_eo · src_o
    // =======          ====   ============================
    //  (1)              (2)            (3)

    // (3) D_eo(src_o) → device_vec0
    wilson_dslash.run_eo(device_vec0, fermion_in, gauge);
    // A_ee⁻¹ · (3)  → device_vec2
    clover_dslash_ee_inv.give(device_vec2 ← device_vec0);
    // (2) D_oe(A_ee⁻¹ D_eo src_o) → device_vec1
    wilson_dslash.run_oe(device_vec1, device_vec2, gauge);
    // (1) A_oo(src_o) → device_vec2
    clover_dslash_oo.give(device_vec2 ← fermion_in);
    // dest = (1) - κ² · (2)
    bistabcg_give_dest_o<<<>>>(fermion_out, device_vec2, device_vec1, κ);
}
```

**BiStabCG 主循环** (`_run`):

```cpp
for (int loop = 0; loop < max_iter; loop++) {
    // 流水线化: 4 个独立 cublas stream 并行内积
    _dot(r_tilde, r, _rho_,      stream_a);  // stream_a: ρ = ⟨r̃, r⟩
    // ... stream_b: 同步等待
    bistabcg_give_1beta(stream_a);           // β = (ρ/ρ_prev)·(α/ω)
    bistabcg_give_1rho_prev(stream_b);       // ρ_prev = ρ
    bistabcg_give_p(stream_a);               // p = r + β(p - ωv)
    _dot(r, r, _norm2_tmp_,      stream_c);  // stream_c: ||r||²  (残差监控)
    dslash(v, p);                            // v = A·p (主计算)
    _dot(r_tilde, v, _tmp0_,    stream_d);   // stream_d: ⟨r̃, v⟩
    bistabcg_give_1alpha(stream_d);          // α = ρ/⟨r̃, v⟩
    bistabcg_give_s(stream_a);               // s = r - αv
    dslash(t, s);                            // t = A·s (主计算)
    _dot(t, s, _tmp0_,          stream_c);   // stream_c: ⟨t, s⟩
    _dot(t, t, _tmp1_,          stream_d);   // stream_d: ⟨t, t⟩
    bistabcg_give_1omega(stream_d);          // ω = ⟨t,s⟩/⟨t,t⟩
    bistabcg_give_r(stream_a);               // r = s - ωt
    bistabcg_give_x_o(stream_b);             // x = x + αp + ωs
    if (|r||² < atol²) break;
}
```

**关键优化**:
- 4 个 cuBLAS 句柄 + 4 个独立 CUDA stream 实现内积流水线
- `_dot_mpi` 将 cuBLAS dot 结果通过 MPI_Allreduce 做跨节点归约
- 标量计算用单线程 kernel (`<<<1,1>>>`) 而非 CPU 端串行
- 向量更新 kernel 使用 `gridDim = lat_4dim / BLOCK_SIZE` 覆盖全部格点

### 3.5 Clover Dslash Kernel (`clover_dslash_*.cu`)

Clover dslash 由三部分组成:

1. **`clover_dslash_single.cu`**: 单 GPU 内部计算 `A_ee · src` 或 `A_ee⁻¹ · src`
   - 本地 12×12 矩阵向量乘 (12 = 4 spin × 3 color)
   - 使用 `einsum("EeXYZT, eXYZT -> EXYZT")` 模式

2. **`clover_dslash_multi.cu`**: 多 GPU 面/棱通信 (gauge 场的 halo exchange)
   - 1D 面通信 (gauge 场在边界面的交换，4 方向 × 2 朝向)
   - 2D 棱通信 (6 对方向组合 × 4 种 B/F 组合 = 24 个 buffer)
   - 使用 `memcpy` stream 实现异步 GPU↔CPU 数据传输

3. **`clover_dslash_comm.cu`**: MPI 通信协调
   - 编排 1D/2D send/recv 的顺序
   - 保证通信边界的数据正确性

---

## 4. CUDA C++ MultiGrid 的设计方案

### 4.1 总体思路

将 Python 层的 MultiGrid V-cycle 主循环下沉到 C++ CUDA，每层级使用已有的 CUDA kernel (Wilson/Clover dslash + BiStabCG)，同时在 GPU 上实现延拓/限制算子。Python 层仅负责初始化 (近零空间构造、粗网格算子构建) 和顶层调用。

### 4.2 新增 CUDA Kernel

#### 4.2.1 延拓 Kernel (`prolong_coarse_to_fine`)

```cuda
// 数学: ψ_fine(e, Xx, Yy, Zz, Tt) = Σ_E V(E, e, Xx, Yy, Zz, Tt) · ψ_coarse(E, X, Y, Z, T)
// 对精细网格的每个点，累加所有粗网格自由度的贡献
template <typename T>
__global__ void prolong_kernel(
    void *device_fine_vec,     // [e, Xx, Yy, Zz, Tt]  (输出)
    void *device_coarse_vec,   // [E, X, Y, Z, T]      (输入)
    void *device_lonv,         // [E, e, X, x, Y, y, Z, z, T, t]  (延拓算子)
    void *device_params)       // 含 coarse_E, fine_e, coarse/fine 格点尺寸
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 解析精细网格坐标
    int fine_xyzt = ...;
    int e = ...;  // 精细 dof (0..11)
    
    LatticeComplex<T> sum(0.0, 0.0);
    for (int E = 0; E < coarse_dof; E++) {
        // 从 lonv 中查表获取 V(E, e, fine_xyz, coarse_xyz)
        // 从 coarse_vec 中获取对应粗格点的值
        sum += V[E*stride_E + e*stride_e + ...] * coarse_vec[E*stride_cE + ...];
    }
    fine_vec[idx] = sum;
}
```

**优化要点**:
- `lonv` 是非稀疏的稠密张量，需全部存入 GPU 显存
- 对 `E` 的循环是主要计算量，可用共享内存缓存 `coarse_vec` 的值
- 如果 `coarse_dof` 较小 (如 12-24)，循环开销可控

#### 4.2.2 限制 Kernel (`restrict_fine_to_coarse`)

```cuda
// 数学: ψ_coarse(E, X, Y, Z, T) = Σ_{e,x,y,z,t} V*(E, e, Xx, Yy, Zz, Tt) · ψ_fine(e, Xx, Yy, Zz, Tt)
// 对每个粗格点，累加其内部所有精细格点的贡献
template <typename T>
__global__ void restrict_kernel(
    void *device_coarse_vec,   // [E, X, Y, Z, T]      (输出)
    void *device_fine_vec,     // [e, Xx, Yy, Zz, Tt]  (输入)
    void *device_lonv,         // [E, e, X, x, Y, y, Z, z, T, t]
    void *device_params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 解析粗网格坐标
    int coarse_xyzt = ...;     // 粗格点线性索引
    
    LatticeComplex<T> sum[_MAX_COARSE_DOF_] = {0};
    for (int x = 0; x < coarse_x_size; x++)
    for (int y = 0; y < coarse_y_size; y++)
    for (int z = 0; z < coarse_z_size; z++)
    for (int t = 0; t < coarse_t_size; t++) {
        for (int e = 0; e < fine_dof; e++) {
            for (int E = 0; E < coarse_dof; E++) {
                sum[E] += conj(V[E*stride_E + e*stride_e + ...]) 
                        * fine_vec[e*stride_fe + ...];
            }
        }
    }
    for (int E = 0; E < coarse_dof; E++)
        coarse_vec[E * stride_c + coarse_xyzt] = sum[E];
}
```

**优化要点**:
- 这是计算密集型 kernel: 每个粗格点需要 `coarse_dof × fine_dof × fine_vol_per_coarse` 次乘加
- 对 `(e, x, y, z, t)` 的 5 重循环可用 warp-level 归约加速
- `sum[E]` 可放在寄存器中

#### 4.2.3 MultiGrid 管理层 (`LatticeMultiGrid<T>`)

仿照 `LatticeCloverBistabCg` 的设计模式:

```cpp
template <typename T>
struct LatticeMultiGrid {
    LatticeSet<T> *set_ptr;  // 最细层 LatticeSet
    
    // 每层的求解器
    LatticeCloverBistabCg<T> *level_solvers;  // 每层一个 BiStabCG
    
    // 层级数据
    int num_levels;
    int *level_lat_sizes;     // 每层格点尺寸
    int *level_dofs;          // 每层自由度
    void **level_b;           // 每层的 b 向量
    void **level_x;           // 每层的 x 向量
    void **level_lonv;        // 每层的延拓/限制算子
    
    // 粗网格 dslash (粗格点上的 hopping + sitting)
    LatticeWilsonDslash<T> *coarse_dslash;
    
    void give(LatticeSet<T> *_set_ptr);
    void init(...);           // 从 Python 接收层级数据
    void cycle(int level);    // V-cycle 递归
    void solve();             // 顶层入口
    void end();
};
```

### 4.3 V-Cycle C++ 实现

```cpp
template <typename T>
void LatticeMultiGrid<T>::cycle(int level) {
    if (level == num_levels - 1) {
        // 最粗层: 直接精确求解 (小矩阵可用 BiStabCG 或直接求逆)
        level_solvers[level].run();
        return;
    }
    
    // --- Pre-smoothing: BiStabCG 迭代 ---
    for (int i = 0; i < num_pre_smooth; i++) {
        level_solvers[level].dslash(v, p);  // 2次 dslash
        // ... BiStabCG 标量更新
    }
    
    // --- 计算残差 ---
    level_solvers[level].dslash(Ax, x);
    // r = b - Ax  (element-wise on GPU)
    
    // --- 限制残差到粗网格 ---
    restrict_kernel<<<grid, block>>>(
        level_b[level+1],          // 粗网格 b
        level_r[level],            // 精细残差 r
        level_lonv[level],         // V
        level_params[level]
    );
    
    // --- 递归: 粗网格求解 ---
    cycle(level + 1);
    
    // --- 延拓修正到精细网格 ---
    prolong_kernel<<<grid, block>>>(
        level_e[level],            // 精细修正 e
        level_x[level+1],          // 粗网格解
        level_lonv[level],         // V
        level_params[level]
    );
    
    // --- 修正 ---
    // x = x + e  (element-wise on GPU)
    
    // --- Post-smoothing: BiStabCG 迭代 ---
    for (int i = 0; i < num_post_smooth; i++) {
        level_solvers[level].dslash(v, p);
        // ... BiStabCG 标量更新
    }
}
```

### 4.4 Cython 接口

在 `apply_multigrid.cu` 中新增:

```cpp
void applyMultiGridQcu(
    long long _fermion_out,      // 解向量
    long long _fermion_in,       // 右端向量  
    long long _gauge,            // 规范场
    long long _clover_ee,        // Clover 偶-偶
    long long _clover_oo,        // Clover 奇-奇
    long long _clover_ee_inv,    // Clover 逆
    long long _clover_oo_inv,    // Clover 逆
    long long _set_ptrs,         // LatticeSet 指针数组
    long long _params,           // 参数 (含 MG 层级信息)
    long long _level_lonvs,      // 各层延拓算子 (指针数组)
    long long _level_hoppings,   // 各层粗 hopping (指针数组)
    long long _level_sittings    // 各层粗 sitting (指针数组)
) {
    // 创建 LatticeMultiGrid
    // 初始化各层 solver
    // 调用 cycle(0) → solve()
    // 写回 fermion_out
}
```

---

## 5. CUDA C++ MultiGrid 的优化策略

### 5.1 延拓/限制算子的 GPU 优化

#### 5.1.1 内存布局优化

当前 Python 的 `local_ortho_null_vecs` 形状为 `[E, e, X, x, Y, y, Z, z, T, t]` (10 维)。GPU 上需要压平为连续内存。

**推荐布局**: 将 `lonv` 重新排列为 `[X*Y*Z*T, E, e*x*y*z*t]`:
- 每个粗格点块连续存储 → 更好的 L2 cache 局部性
- 每线程处理一个粗格点块 → 线程束内无分支

#### 5.1.2 使用 Tensor Core

对于延拓操作 `fine = V · coarse`，本质是矩阵乘:
- 将 `V` 视为 `[fine_dof * fine_vol_per_coarse, coarse_dof]` 矩阵
- 用 cuBLAS `gemv` 或手写 warp-level 矩阵乘

对于限制操作 `coarse = V† · fine`，同理。

```cpp
// 使用 cuBLAS batched gemv
for (int block = 0; block < num_coarse_blocks; block++) {
    cublasGemmEx(..., 
        V_block[block],  // [local_dim, coarse_dof]
        coarse_vec,      // [coarse_dof, 1]  
        fine_vec);       // [local_dim, 1]
}
```

#### 5.1.3 共享内存优化

对于 `coarse_dof ≤ 24` 的小型矩阵，可将粗网格向量全部载入共享内存:

```cuda
__shared__ LatticeComplex<T> coarse_shared[MAX_COARSE_DOF];
// 每个 coarse_dof 元素由一个线程载入
if (threadIdx.x < coarse_dof)
    coarse_shared[threadIdx.x] = coarse_vec[threadIdx.x];
__syncthreads();
// 然后所有线程读取共享内存中的 coarse_shared
```

### 5.2 BiStabCG 平滑器的 GPU 优化

#### 5.2.1 减少 Kernel Launch 开销

当前 `LatticeCloverBistabCg::_run()` 每次迭代启动 12+ kernel。优化方向:

1. **Kernel 融合**: 将 `give_p` + `give_s` + `give_x_o` + `give_r` 融合为单 kernel
   - 减少 4 次 kernel launch → 1 次
   - 在寄存器中完成所有标量读取和向量更新
   
2. **标量 kernel 消除**: 当前 beta/alpha/omega 用 `<<<1,1>>>` kernel 计算
   - 改为 CPU 端 cudaMemcpy 标量值 → CPU 计算 → cudaMemcpy 回 GPU
   - 或直接用 constant memory / kernel 参数传递标量

#### 5.2.2 通信-计算重叠

当前 dslash 的 4 维通信是顺序执行的。改进:

```
Stream 0: X方向 send → MPI_X → X方向 recv → X本地计算
Stream 1: Y方向 send → MPI_Y → Y方向 recv → Y本地计算
Stream 2: Z方向 send → MPI_Z → Z方向 recv → Z本地计算
Stream 3: T方向 send → MPI_T → T方向 recv → T本地计算
(所有 stream 并发执行)
```

**前提**: 需要将 gauge/source/dest buffer 分区，避免 4 个方向的写冲突。

#### 5.2.3 混合精度

| 层级 | 精度 | 理由 |
|------|------|------|
| Level 0 (最细) | complex64 (float) | 计算量最大，减少显存带宽 |
| Level 1-2 | complex64 | 中间层 |
| Level 3+ (最粗) | complex128 (double) | 小格点，追求精度 |

当前 `LatticeSet` 按 `_DATA_TYPE_` 选择 float/double 模板，MG 需要在不同层级使用不同的 `LatticeSet` 实例。

### 5.3 粗网格算子的 GPU 构造

当前 Python 中粗网格算子通过 Galerkin 投影逐基向量构造，涉及大量精细网格操作。GPU 加速方案:

```cuda
// 并行策略: 每个粗网格 dof 的每个方向独立计算
template <typename T>
__global__ void build_coarse_hopping(
    void *device_coarse_M_plus,   // [E, E, X, Y, Z, T]  per direction
    void *device_lonv,            // 延拓算子
    void *device_fine_hopping,    // 精细 hopping 矩阵
    void *device_params)
{
    int E_dest = blockIdx.x;  // 输出 dof
    int E_src  = blockIdx.y;  // 输入 dof
    int ward   = blockIdx.z;  // 方向
    
    // 对每个粗格点:
    //   src_c = δ_{E_src} (只在对应粗格点和方向非零)
    //   src_f = prolong(V, src_c)
    //   dest_f = fine_hopping.apply(ward, src_f)
    //   dest_c = restrict(V†, dest_f)
    //   coarse_M[E_dest, E_src, ...] = dest_c[E_dest, ...]
}
```

**优化**: 将粗网格算子构造从 Python 的逐基向量串行改为 GPU 上各 (dest_dof, src_dof, 方向, 粗格点) 完全并行。

### 5.4 内存管理优化

#### 5.4.1 显存预估

以 `32^4` 精细格点, 2^4 粗化因子为例:

| 数据 | 精细层 (FP32) | 粗层 (FP32) |
|------|---------------|-------------|
| Gauge field U | 3×3×4×32^4 = 48 MB | — |
| Fermion vector | 12×32^4×8B = 96 MB | 24×16^4×8B = 12 MB |
| lonv (per level) | — | 24×12×2^4×16^4×8B = ~6 MB |
| 总计 (4 levels) | ~150 MB | ~50 MB |

总显存需求约 **200 MB** (单精度)，完全适合单 GPU。

#### 5.4.2 内存池

使用 `cudaMallocAsync` + 内存池减少分配开销:

```cpp
// 预分配所有层级的 buffer
void *mg_memory_pool;
cudaMallocAsync(&mg_memory_pool, total_size, stream);
// 各级 buffer 通过偏移量访问
```

### 5.5 自适应层级管理

将 Python 的 `adaptive()` 和 `levels_back()` 逻辑移植到 C++:

```cpp
template <typename T>
void LatticeMultiGrid<T>::adaptive(int iter, T current_residual) {
    if (convergence_stall_count > 3) {
        effective_num_levels = 1;  // 退化为纯 BiStabCG
        return;
    }
    // 比较当前残差与历史残差采样
    int worse_count = 0;
    for (int i = 0; i < num_convergence_sample; i++) {
        if (residual_history[i] < current_residual)
            worse_count++;
    }
    if (worse_count >= num_convergence_sample / 2)
        convergence_stall_count++;
}
```

### 5.6 V-Cycle 变体

| 变体 | 描述 | 适用场景 |
|------|------|----------|
| V(1,1) | 1次前平滑 + 1次后平滑 | 标准配置 |
| V(2,2) | 2次前平滑 + 2次后平滑 | 收敛困难时 |
| W-cycle | 粗网格求解两次 | 近零空间质量差时 |
| K-cycle | 粗网格用 Krylov 加速 | 最稳定，开销大 |
| Full MG | 从最粗层开始逐层细化 | 初始猜测好 |

**推荐**: 先用 V(1,1)，根据收敛情况自适应切换到 V(2,2) 或 K-cycle。

### 5.7 与 Python 层的接口设计

```python
# Python 端调用
class multigrid:
    def init(self):
        # 1. 构建近零空间 (Python)
        # 2. 局部正交化 (Python → 可下沉到 GPU)
        # 3. 构建粗网格算子 (Python → 可下沉到 GPU)
        # 4. 将所有层级数据传到 GPU
        
    def solve(self, b, x0):
        # 调用 C++ MultiGrid
        qcu.applyMultiGridQcu(
            fermion_out, fermion_in,
            gauge, clover_ee, clover_oo,
            clover_ee_inv, clover_oo_inv,
            set_ptrs, params,
            level_lonvs_ptrs,       # 各层延拓算子 GPU 指针
            level_coarse_hoppings,  # 各层粗 hopping GPU 指针
            level_coarse_sittings   # 各层粗 sitting GPU 指针
        )
```

**职责划分**:
- **Python**: 近零空间构造 (需要 BiStabCG 迭代)、QR 分解、Galerkin 投影 (逻辑复杂但只执行一次)
- **C++ CUDA**: V-cycle 主循环、延拓/限制、BiStabCG 平滑 (高频调用，需 GPU 加速)

---

## 6. 实现路线图

### Phase 1: 基础设施 (预计 3-5 天)

1. **完善 `multigrid.cu`**: 从空文件变为 `applyMultiGridQcu` 入口函数
2. **实现延拓 kernel**: `prolong_kernel` (需 ~100 行 CUDA)
3. **实现限制 kernel**: `restrict_kernel` (需 ~100 行 CUDA)
4. **单元测试**: 单 GPU 上对比 Python 延拓/限制的数值结果

### Phase 2: MG 管理结构 (预计 3-5 天)

5. **实现 `LatticeMultiGrid<T>`**: 仿照 `LatticeCloverBistabCg` 模式
   - `give/init/cycle/solve/end` 方法
   - 层级数据管理 (b, x, r, lonv 指针数组)
6. **最细层 CUDA BiStabCG 集成**: 复用 `LatticeCloverBistabCg`
7. **粗层 BiStabCG**: 复用 `LatticeWilsonBistabCg` (粗网格无 Clover 项时)
   - 或实现通用粗网格 dslash: 粗 hopping + 粗 sitting

### Phase 3: V-Cycle 集成 (预计 3-5 天)

8. **V(1,1) cycle 实现**: 递归调用 GPU kernel
9. **MPI 通信验证**: 多 GPU 测试 V-cycle 正确性
10. **与 Python MG 对比**: 残差收敛曲线对比

### Phase 4: 优化 (预计 5-7 天)

11. **Kernel 融合**: 将 BiStabCG 的多个标量 kernel 融合
12. **通信-计算重叠**: 4 维 dslash 通信流并行
13. **内存池**: 减少 cudaMalloc/cudaFree 开销
14. **自适应层级**: 移植 `adaptive()` 逻辑
15. **混合精度**: 不同层级使用不同精度
16. **Profile**: 用 Perfetto/Nsight 分析瓶颈

### Phase 5: 高级特性 (可选)

17. **W-cycle / K-cycle**: 更稳定的粗网格求解
18. **粗网格算子 GPU 构造**: 将 `operator.__init__` 的 Galerkin 投影 GPU 化
19. **局部正交化 GPU 化**: QR 分解的 GPU 实现

---

## 附录: 关键代码引用索引

| 文件 | 行数 | 内容 |
|------|------|------|
| `pyqcu/solver/_multigrid.py` | 1-336 | Python MultiGrid 完整实现 |
| `pyqcu/solver/_bistabcg.py` | 1-73 | Python BiStabCG |
| `pyqcu/dslash/_operator.py` | 146-217 | 粗网格算子 Galerkin 构造 |
| `pyqcu/dslash/_operator.py` | 219-335 | 完整 Dirac 算子 (含 even-odd) |
| `pyqcu/tools/_multigrid.py` | 7-55 | 近零空间向量生成 |
| `pyqcu/tools/_multigrid.py` | 58-97 | 局部正交化 (QR) |
| `pyqcu/tools/_multigrid.py` | 100-128 | restrict/prolong (einsum) |
| `pyqcu/cuda/define.py` | 1-161 | C++/Python 参数镜像 |
| `cpp/cuda/qcu/include/define.h` | 65-98 | MG 参数枚举预留 |
| `cpp/cuda/qcu/include/lattice_set.h` | 1-742 | 格点几何 & CUDA 资源管理 |
| `cpp/cuda/qcu/include/lattice_clover_bistabcg.h` | 1-442 | Clover BiStabCG 完整实现 |
| `cpp/cuda/qcu/include/lattice_wilson_bistabcg.h` | 1-395 | Wilson BiStabCG 完整实现 |
| `cpp/cuda/qcu/src/wilson_dslash.cu` | 1-1406 | Wilson dslash + MPI 通信 |
| `cpp/cuda/qcu/src/bistabcg.cu` | 1-258 | BiStabCG 标量 kernel |
| `cpp/cuda/qcu/src/multigrid.cu` | **空文件** | **待实现** |
