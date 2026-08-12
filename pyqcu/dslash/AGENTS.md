# AGENTS.md — pyqcu.dslash

Wilson 与 Clover Dirac 算子 — 格点 QCD 的核心线性算子。

## 文件

| 文件 | 用途 |
|---|---|
| `_wilson.py` | Wilson hopping 项 D_w：γ_μ 矩阵 + 平行移动的空间导数。有每模块 `force_use_npu` 标志 |
| `_clover.py` | Clover 项（色磁场强 F_{μν} 贡献）。四plaquette clover 构造 + MPI 边界交换 |
| `_operator.py` | 组合 Dirac 算子 = hopping + sitting，含奇偶预处理、粗网格 Galerkin 投影、MPI halo 交换 |

## 导出 API

### Wilson（`_wilson.py`）

- `give_wilson(src, U, kappa, u_0, with_I, verbose)` — 全 Wilson 算子 D_w = I − κ/u_0 · Σ_μ [(1−γ_μ)U_μ δ_{x+μ,y} + (1+γ_μ)U^†_{x−μ,μ} δ_{x−μ,y}]
- `give_wilson_eo(src_o, U_eo, kappa, u_0, verbose)` / `give_wilson_oe(...)` — 奇偶 Wilson（eo: 偶 dest 自奇 src）
- `give_hopping_plus(ward, U, kappa, u_0, verbose)` — 方向 hopping M_μ^+ = −κ/u_0 · (I−γ_μ) ⊗ U_μ，形状 `[12, 12, Lx, Ly, Lz, Lt]`
- `give_hopping_minus(ward, U, U_head, kappa, u_0, verbose)` — M_μ^- = −κ/u_0 · (I+γ_μ) ⊗ U^†_{x−μ,μ}，经 `U_head` halo 交换
- `give_wilson_plus(ward, src, hopping, src_tail, parity, verbose)` / `give_wilson_minus(...)` — 应用 hopping：einsum("Eexyzt,exyzt→Exyzt")，处理 MPI `src_tail`/`src_head` 边界与奇偶掩码

eo/oe 变体用 `ward_p_keys`（x, y, z, t_p）— `t_p` 方向处理奇偶拆分的时间 hopping（奇偶掩码）。

### Clover（`_clover.py`）

- `make_clover(U, kappa, u_0, support_parallel, verbose)` — 四plaquette F_{μν} 构造（每 μν 对 12 个平移 link），返回 `[4,3,4,3,Lx,Ly,Lz,Lt]`
- `add_I(clover_term, ...)` / `cut_I(...)` — M = I + clover_term / M = clover_term − I
- `inverse(clover_term, ...)` — 批量 12×12 矩阵求逆（`torch.linalg.inv`，**禁止**逐点 for 循环！）
- `give_clover(src, clover_term, ...)` — einsum("SCscxyzt,scxyzt→SCxyzt")
- `give_clover_ee` / `give_clover_oo` — 偶偶/奇奇 clover 应用（委托 `give_clover`）

**Clover 系数注意：** 用 `_clover_factor = −0.125 · κ/u_0`。标准约定 c_sw=1 给出 −κ/(16·u_0)。因子 2 可能来自反厄米部分约定。改动前需与 QUDA/Chroma 交叉验证。

`support_parallel=True` 时 `make_clover` 对所有 μν plaquette 的 4 角 4 边做 MPI halo 交换。

### Operator（`_operator.py`）

- **`hopping` 类**：init 时预计算 `M_plus_list[4]`/`M_minus_list[4]`（并做规范场边界 MPI 交换，`support_parity=True` 时再拆奇偶块）。`matvec(src)` = Σ_μ (matvec_plus + matvec_minus)，带费米子边界 halo 交换
- **`sitting` 类**：M = I + T；`matvec(src)` — clover_term 为 None 时原样返回 src
- **`operator` 类**：`matvec`（自动检测 `[4,3,...]` vs `[12,...]` 布局）、`matvec_eo/oe/ee/oo`、`matvec_ee_inv/oo_inv`、`matvec_parity`（M_oo − M_oe·M_ee⁻¹·M_eo）、`give_b_parity`、`give_x_e`、`matvec_all`。给定 `fine_hopping`+`fine_sitting`+`local_ortho_null_vecs` 时构建 Galerkin 粗网格算子 P^T D_fine P

**MPI halo 交换**由 `grid_size[ward] != 1` 守护 — 单进程方向无 MPI 开销。

## 反模式

- **绝不**用 `self.sitting`（对象）做 truthy 判断；用 `self.sitting.clover_term is not None`
- **绝不**逐点 for 循环 `torch.linalg.inv`；用批量求逆（permute 到 batch 维、一次求逆、permute 回来）
- **绝不**在阻塞 `Sendrecv` 前后加 `MPI.Barrier()` — 冗余且拖慢执行
