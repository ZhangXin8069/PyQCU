# AGENTS.md — pyqcu.smear

规范场 smearing — 空间平滑规范 link 以降低紫外噪声。

## 文件

| 文件 | 用途 |
|---|---|
| `_stout.py` | Stout smearing 算法（源自 EasyDistillation 的 elemental generator） |

## 导出 API

### `stout_smear(U, nstep=1, rho=0.12, support_parallel=False)`

应用 nstep 轮 rho 参数的 stout smearing。

**算法（每步）**：
1. **Q_μ = staple 求和**：每个方向 μ 对 ν≠μ 的两个 3-link staple（U_ν U_μ U^†_ν 前向 + U^†_ν U_μ U_ν 后向）
2. **投影到 su(3) 代数**：Q ← ρ · Q · U^†，反厄米化：Q ← i/2 · (Q^† − Q) − (1/3) Tr(Q) · I
3. **SU(3) 投影系数 f₀, f₁, f₂**（Morningstar-Peardon 方法）：c₀ = Re(Tr(Q³))/3，c₁ = Re(Tr(Q²))/2；θ = arccos(c₀/(2(c₁/3)^(3/2)))；u = √(c₁/3)·cos(θ/3)，w = √c₁·sin(θ/3)；f₀,f₁,f₂ 用 e^{iu}、e^{2iu}、cos(w)、sinc(w) 表达
4. **奇偶处理**（c₀ < 0 时）：f₀ → f₀^*，f₁ → −f₁^*，f₂ → f₂^*（标准路径）；NPU 路径用实虚分解
5. **更新 U**：U_new = (f₀·I + f₁·Q + f₂·Q²) · U

**数值稳定性**：c₁ 下限 1e-15（防 c₀_max=0）；arccos 域比值钳制 [−1+1e-15, 1−1e-15]；|w|≤0.05 用 sinc(w) Taylor 展开；分母 9u²−w² 加 1e-15 epsilon 防除零。

**MPI**：`support_parallel=True` 时每步重算边界数据（U_head/U_tail/U_head_tail）— 每步 U 都在变。

## 关键反模式（已修复）

`nstep>1` 循环曾不更新 U — 循环变量正确重绑定但 MPI 边界数据在循环外计算。修复：MPI 交换移入步进循环内。

## 数据布局

规范场 `[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`，返回同形状。

## NPU 支持

每模块 `force_use_npu` 标志。NPU 上 f₀/f₁/f₂ 奇偶符号约定用显式实虚分解：
- f₀：imag = −imag（共轭）
- f₁：real = −real，imag 不变（共轭 + 前导负号抵消）
- f₂：同 f₁
