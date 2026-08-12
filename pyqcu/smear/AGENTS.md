# AGENTS.md — pyqcu.smear

规范场 smearing — 平滑规范链接以压低 UV 噪声。

## 文件

| 文件 | 用途 |
|---|---|
| `_stout.py` | Stout smearing 算法（自 EasyDistillation 的 elemental generator 改编） |

## API

`stout_smear(U, nstep=1, rho=0.12, support_parallel=False)` — 每步：

1. **Q_μ = staple 求和**（ν≠μ 的两条 3-link staple）
2. **投影到 su(3)**：Q ← ρ·Q·U†，反厄米化 Q ← i/2·(Q†−Q) − (1/3)Tr(Q)·I
3. **Morningstar-Peardon 投影系数 f₀,f₁,f₂**（c₀=Re(Tr Q³)/3, c₁=Re(Tr Q²)/2, θ=arccos(...), u/w 含 e^{iu}、sinc(w)）
4. **奇偶处理**（c₀<0）：f₀→f₀*, f₁→−f₁*, f₂→−f₂*；NPU 路径用实/虚分解
5. **更新**：U_new = (f₀I + f₁Q + f₂Q²)·U

## 数值稳定性

- c₁ 下限 1e-15；arccos 比率夹在 [−1+1e-15, 1−1e-15]；sinc 在 |w|≤0.05 用 Taylor 展开；分母 9u²−w² 加 1e-15 保护

## MPI

`support_parallel=True` 时 U 每步变化，MPI 边界数据（U_head/tail/head_tail）必须在**步循环内**重算（已修复的 bug：曾放循环外）。

## NPU

逐模块 `force_use_npu` 标志；f₀/f₁/f₂ 奇偶符号约定用显式实/虚分解。

## 数据布局

`[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`，返回值同形。
