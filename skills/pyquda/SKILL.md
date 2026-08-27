---
name: pyquda
description: examples/pyquda 目录的完整生成 skill：PyQCU 与 PyQuda-0.3.2（QUDA 1.1.0）双进程隔离对比套件——Wilson/Clover dslash、BiCGStab/CG 求解的结果与性能对比（残差/逐迭代残差/耗时/作图），含维度排布转换（pyqcu 切 t vs pyquda 切 x）与归一化锚定（m+4=1/(2κ)）。
---
# CLAUDE.md — examples/pyquda

PyQCU（纯 Python 后端）与 PyQuda-0.3.2（QUDA 1.1.0）的结果/性能对比套件。
双进程隔离（dev87 F2：libqcu.so 与 libquda.so 同进程加载会 cudart 上下文冲突），
经 h5 交换数据（数据优先 `examples/data/pyquda_cmp/<lat>/`，h5py 读写）。

## 数据维度排布（2026-08-28 以实际代码实测核对）

| 量 | pyqcu（Python 后端） | pyquda 0.3.2（QUDA） |
|---|---|---|
| gauge | `[c,c,d,x,y,z,t]`=(3,3,4,Lx,Ly,Lz,Lt) | `[d,q,t,z,y,x/2,c,c]`=(4,2,Lt,Lz,Ly,Lx//2,3,3) |
| fermion | `[s,c,x,y,z,t]`=(4,3,Lx,Ly,Lz,Lt) | `[q,t,z,y,x/2,s,c]`=(2,Lt,Lz,Ly,Lx//2,4,3) |
| 时空轴序 | xyzt 正序 | tzyx 倒序 |
| 奇偶切分维 | **t（最后轴）**：`[p,prefix...,Lx,Ly,Lz,Lt//2]`（oooxyzt2poooxyzt） | **x（最后轴）**：q=0 偶奇偶 (x+y+z+t)%2==0 |

- 棋盘格约定两侧一致：(x+y+z+t)%2==0 → q=0 / p=0（even parity）。
- 转换函数（common.py，纯 numpy）：`pyqcu_gauge_to_quda` / `quda_gauge_to_pyqcu` /
  `pyqcu_fermion_to_quda` / `quda_fermion_to_pyqcu`（内部 evenodd_split_x/merge_x 按 x 切分）。

## 归一化锚定（dev87 G2 + 本套件实测复核）

- pyqcu κ = 1/(2m+8)（κ 归一化 D = I − κH）；pyquda getDslash 内部 κ' = 1/(2(m+1))（勿用其默认输出）。
- **对齐方式**：`invert_param.mass_normalization = QUDA_MASS_NORMALIZATION` 且**绕过
  general.invert 的 ×2κ'**（直接调 `invertQuda(x.data_ptr, b.data_ptr, invert_param)`）。
- 结果：pyqcu 解 x_p = (m+4) · x_quda（实测回归 c = 4.050000，rel 1.5e-6，8³×16）。
- dslash 单步（跳跃部分）：QUDA `dslashQuda` 输出 = +H_hop·b（不含 κ、无负号），
  pyqcu `give_wilson(with_I=False)` = −κ·H_hop·b ⇒ 回归 c = −κ_pyqcu（实测 −0.123457，rel 1.5e-8）。
- Clover（csw=1）：pyqcu 侧 `D_cl ψ = give_wilson(ψ,U,κ) + give_clover(ψ, make_clover(U,κ))`；
  quda 侧 `getDslash(..., clover_coeff_t=1.0)`，解同样满足 x_p = (m+4)·x_q（实测 rel 1.6e-6）。

## 文件清单

| 文件 | 职责 |
|---|---|
| `common.py` | 维度转换（numpy）、h5 I/O（save_h5/load_h5 独立句柄）、rel_diff/linreg_scale、CG 日志解析 |
| `run_pyqcu.py` | 进程 A：生成 U/b（seed42/σ0.1/randn）→ input.h5；give_wilson、BiCGStab（Wilson/Clover）→ pyqcu*.h5/json |
| `run_pyquda.py` | 进程 B：读 h5 → `pyquda.init(grid_size=[1,1,1,1])` → CG（mass 归一化，VERBOSE 捕获逐迭代残差）→ pyquda*.h5/json；干净计时二次运行 |
| `compare.py` | 聚合：解互比（回归+rel_diff）、hop 中间量、残差表、逐迭代残差图+耗时柱状图 → out/compare_*.{json,md,png} |
| `run_all.sh` | 一键三阶段：`bash run_all.sh 8x8x8x16 [tol]` |

## 关键 API（PyQuda-0.3.2）

- 初始化：`pyquda.init(grid_size=[1,1,1,1])`（QUDA 编译带 MPI 时必须，否则 MPI_Comm_rank 崩溃）。
- 字段：`LatticeGauge(lat, cupy_array)` / `LatticeFermion(lat, cupy_array)`——**value 必须 device
  （cupy）且 complex128**（QUDA cpu/cuda_prec=DOUBLE；传 numpy host 数组 → NaN 发散实测）。
- 算子：`core.getDslash(lat, mass, tol, maxiter, clover_coeff_t=None, anti_periodic_t=False)`。
- 求解：`invertQuda`（对齐需手动调 + 改 mass_normalization）；逐迭代残差读 `invert_param.iter`
  （int，注意不是 iterations）、`secs`（QUDA 内部）、`true_res`。
- 单步：`dslashQuda(out半场_ptr, in半场_ptr, invert_param, QudaParity.QUDA_*_PARITY)`（子块算子）。
- 逐迭代残差：`invert_param.verbosity = QUDA_VERBOSE`，stdout 行 `CG: <N> iterations, <r,r> = ..., |r|/|b| = ...`，
  regex：`CG:\s*(\d+)\s+iterations.*?\|r\|/\|b\|\s*=\s*([0-9.eE+-]+)`。

## 实测基线（2026-08-28，RTX 4060，m=0.05，T 周期边界，tol=1e-10）

| 指标 | 4×4×4×4 | 8×8×8×16 |
|---|---|---|
| pyqcu / pyquda 迭代（Wilson） | 63 / 46 | 109 / 86 |
| pyqcu / pyquda 迭代（Clover） | 65 / 48 | 129 / 93 |
| pyqcu 真实残差 | 5.9e-7 | 4.6e-7（tol=1e-10 仍受 fp32 递推漂移限制） |
| pyquda true_res | 5.9e-9 | 8.4e-11（reliable-update） |
| 解互比（缩放 (m+4)） | 4.5e-7 | 3.7e-7 |
| hop 定标 | −κ_p 精确 | −κ_p 精确（rel 1.4e-8） |

## Key Anti-Patterns（本套件实测踩坑）

- **T 边界默认不匹配**：pyquda getDslash 默认 anti_periodic_t=True（反周期 T），pyqcu 全周期
  （torch.roll）⇒ 解差 ~6%（回归 4.31 vs 4.05）。必须 `anti_periodic_t=False`。
- **pyqcu Python BiCGStab fp32 递推残差漂移**（dev87 F3 同源）：停机判据用递推 r（s−ωt），
  ‖D x − b‖ 实际比递推残差大 300×（8³×16 上 tol=1e-10 只到 4.6e-7）。对比迭代数时需知此差异。
- **dslashQuda 是奇偶子块算子**：out/in 都传**半场指针**（`odd_ptr`/`even_ptr`），两次调用分别
  计算 D_oe/D_eo；传全场指针会数据错位。
- **cupy 数组取 numpy 必须 `.get()`**（np.asarray 抛 TypeError）。
- pyquda 顶层不导出 LatticeGauge/getDslash：`from pyquda.field import ...`、`from pyquda.core import ...`。
- egg 内 pyqcu.py 缺失（仅 .pyi）→ `import pyquda` 打印循环导入警告，被 try/except 吞掉，**不影响运行**。
- `invert_param.residue` 是数组（list）不能 float()。

## 已知局限

- PyQuda-0.3.2 无 LatticeGauge mat()/nullvec 导出 API ⇒ 中间量对比限于 dslash 跳跃输出
  （hop）；MG nullvec 对比不可行（0.3.2 未暴露），需上游支持。
- 性能对比中 pyqcu 侧为纯 Python einsum 实现（非公平基线），且 QUDA 侧含 tuning 首轮开销；
  报告以迭代数与 QUDA 内部 secs 为准。