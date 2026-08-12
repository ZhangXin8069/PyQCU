# AGENTS.md — pyqcu.dslash

Wilson 与 Clover 狄拉克算子 — 格点 QCD 的核心线性算子。

## 文件

| 文件 | 用途 |
|---|---|
| `_wilson.py` | Wilson 跳跃项 D_w（γ_μ 矩阵 + 平行输运）；逐模块 `force_use_npu` 标志 |
| `_clover.py` | Clover 项（色磁场 F_{μν}，四 plaquette 构造，MPI 边界交换） |
| `_operator.py` | 组合 Dirac 算子 = hopping + sitting，含奇偶预处理、Galerkin 粗网格投影、MPI halo 交换 |

## Wilson API（关键函数）

- `give_wilson(src, U, kappa, u_0, with_I, verbose)` — 完整算子 D_w = I − κ/u_0·Σ_μ[(1−γ_μ)U_μ δ + (1+γ_μ)U† δ]
- `give_wilson_eo/oe` — 奇偶 Wilson（eo/oe 用 `ward_p_keys`，t_p 方向处理奇偶分裂的时间跳跃）
- `give_hopping_plus/minus` — 方向跳跃矩阵 M_μ^±，形状 `[12,12,Lx,Ly,Lz,Lt]`；minus 经 `U_head` halo 交换
- `give_wilson_plus/minus` — einsum("Eexyzt,exyzt→Exyzt") 应用，处理 MPI 边界与奇偶掩码

## Clover API

- `make_clover` — 四 plaquette F_{μν} 构造，`support_parallel=True` 时交换 4 角/边 halo；返回 `[4,3,4,3,Lx,Ly,Lz,Lt]`
- `add_I`/`cut_I`、`inverse`（批量 12×12 求逆，**禁止逐 site 循环**）、`give_clover` 及 ee/oo 变体
- **系数注意**：`_clover_factor = −0.125·κ/u_0`；标准约定（c_sw=1）为 −κ/(16·u_0)。改动前须与 QUDA/Chroma 交叉验证。

## operator 类

- `hopping`：init 预计算 M_plus/minus_list[4] + 规范场 MPI halo；`matvec_plus/minus` 做费米子 halo 交换；`matvec` 四方向求和
- `sitting`：M = I + clover_term；clover 为 None 时 `matvec` 原样返回
- `operator`：`matvec`（自动识别 `[4,3,...]` vs `[12,...]` 布局）、`matvec_eo/oe/ee/oo`(+`_inv`)、`matvec_parity`（M_oo − M_oe·M_ee⁻¹·M_eo）、`give_b_parity`、`give_x_e`、`matvec_eeo/oeo`、`matvec_all`
- halo 交换由 `grid_size[ward] != 1` 守卫

## 粗网格算子（Galerkin 投影）

`fine_hopping`+`fine_sitting`+`local_ortho_null_vecs` 齐备时：对每个 null 基矢 e 与方向 ward：delta 源延长 → 细层 hopping → 限制回粗层；sitting 同样投影。奇偶分离沿当前方向 step=2。

## 反模式（禁止）

- 用 `self.sitting.clover_term is not None` 判断，**不要**用 `self.sitting` 对象做 truthy 检查
- 禁止逐 site `torch.linalg.inv`；用批量逆（permute 到 batch 维一次逆完）
- 禁止在阻塞 `Sendrecv` 前后加 `MPI.Barrier()`
