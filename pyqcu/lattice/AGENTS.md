# AGENTS.md — pyqcu.lattice

格点 QCD 基础：gamma 矩阵、Gell-Mann 矩阵、SU(3) 工具与规范场生成。

## 模块级数据（import 时 CPU complex64 计算）

- `gamma` — Dirac-Pauli 表示的 γ₀…γ₃（γ₀ 反厄米、γ_i 厄米），形状 `[4,4,4]`
- `gamma_5`、`gamma_gamma`（六个 γ_μγ_ν 乘积，作 clover 的 σ_{μν}）、`I`、`minus_I`
- `gell_mann` — 8 个 Gell-Mann 矩阵，形状 `[8,3,3]`；λ₁,₄,₆ 实，λ₂,₅,₇ 为 i×实

## Ward 负索引约定（时空维总是最后 4 轴，负索引保证前缀维度无关）

```python
wards['x']=-4; wards['y']=-3; wards['z']=-2; wards['t']=-1; wards['t_p']=-1
```

- `ward_keys` = `['x','y','z','t']`；`ward_p_keys` 含 `t_p`（奇偶分裂）；`ward_ward_keys` = 6 平面方向
- `ward_wards['xy'] = {'mu':-4, 'nu':-3, 'ward':-6}` — gamma_gamma 索引映射

## 导出函数

- `check_su3(U, tol=1e-3)` — 幺正性 U^H U≈I + det(U)≈1（裸 `torch.linalg.det`）+ 列叉积 minor 恒等式；三查全过才 True
- `generate_gauge_field(U, sigma=0.1, seed=None)` — 每 site/方向采 8 个高斯系数 → H = Σ c_a λ_a → U = exp(i·σ·H)（`torch.matrix_exp`），原地写入，返回 U
- `give_support_multi()` — `MPI.COMM_WORLD.size > 1`

## 数据布局

规范场 U：`[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`
