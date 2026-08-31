# AGENTS.md — pyqcu.lattice

格点 QCD 基础：gamma 矩阵、Gell-Mann 矩阵、SU(3) 群工具、规范场生成。

## 模块级数据（import 时计算，CPU，complex64）

- **`gamma`** — 4×4×4 Euclidean gamma 矩阵，逐元素匹配 QUDA `QUDA_DEGRAND_ROSSI_GAMMA_BASIS`（四个 γ_μ 均厄米）。形状 `[4, 4, 4]`
- **`gamma_5`** — γ₅ = γ₀γ₁γ₂γ₃。形状 `[4, 4]`
- **`gamma_gamma`** — 六个 γ_μ γ_ν 积：[γ_x,γ_y]…[γ_z,γ_t]。形状 `[6, 4, 4]`，用作 clover 项的 σ_{μν}
- **`I`** / **`minus_I`** — 4×4 单位阵（complex64）/−I（预计算）
- **`gell_mann`** — 八个 Gell-Mann 矩阵 λ₁…λ₈（SU(3) 生成元，无迹厄米）。形状 `[8, 3, 3]`。λ₁,λ₄,λ₆ 实；λ₂,λ₅,λ₇ 为 i×实

## Ward 索引约定

时空维永远是最后 4 轴（`...xyzt`），所以用负索引：

```python
wards['x'] = -4; wards['y'] = -3; wards['z'] = -2; wards['t'] = -1
wards['t_p'] = -1  # 奇偶拆分的时间（与 t 同索引）
```

- **`ward_keys`** = `['x','y','z','t']`；**`ward_p_keys`** = `['x','y','z','t_p']`；**`ward_ward_keys`** = `['xy','xz','xt','yz','yt','zt']`
- `ward_wards['xy'] = {'mu': -4, 'nu': -3, 'ward': -6}` 等映射 gamma_gamma 索引

## 导出函数

### `check_su3(U, tol=1e-3, verbose=True) → bool`
1. 幺正性：U^H U ≈ I（`_torch.allclose`，`atol=tol`）
2. 行列式：det(U) ≈ 1（原始 `torch.linalg.det` — 无需 NPU 等价）
3. 幺模恒等式：每列是另两列（共轭后）的叉积

### `generate_gauge_field(U, sigma=0.1, seed=None, verbose=False)`
1. 每格点每方向采样 8 个高斯系数
2. H = Σ_a c_a λ_a（厄米矩阵）
3. U = exp(i·σ·H)（`torch.matrix_exp`）
4. 重排为 `[3, 3, 4, Lx, Ly, Lz, Lt]` 布局

### `give_support_multi() → bool`
`MPI.COMM_WORLD.size > 1`（多进程运行）。

## 数据布局

规范场 U：`[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`

### 源场构造（`_source.py`，参考 PyQUDA pyquda_utils/source.py）

布局 `[4, 3, Lx, Ly, Lz, Lt]`；spin/color 为 None 时该自由度维全填。

- `point_source(latt_size, t_srce=[x,y,z,t], spin=None, color=None)` — 时空 δ 源
- `wall_source(latt_size, t_srce=int, ...)` — 固定时间片全空间
- `volume_source(latt_size, ...)` — 全时空
- `z2_source(latt_size, seed=None)` — Z2 噪声 ±1（随机源平均）
- `momentum_source(latt_size, mode=[nx,ny,nz,nt], t_srce=None|int)` — 平面波 exp(i2π n·x/L)
- `fermion_source(latt_size, kind, ...)` — 统一分派：kind ∈ {point, wall, volume, momentum, z2}

dtype/device 可选参数，默认 complex128/cpu。momentum 的 wall 型与体积型仅差时间片支撑。
