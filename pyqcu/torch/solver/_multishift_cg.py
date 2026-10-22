import math
from time import perf_counter
from typing import Callable, Optional, List
import torch
from pyqcu import tools
import pyqcu.cann as _torch

"""多质量 CG（multi-shift CG）— 一次 matvec 序列同时求解 (A + σ_i)·x_i = b。

参考实现 quda lib/inv_multi_cg_quda.cpp（updateAlphaZeta 递推，Jegerlehner
hep-lat/9612014 多移位 CG）。约定：shifts 升序排列，最低移位 σ_0 为**主链**，
其移位折叠进标量积 pAp = <p,Ap> + σ_0<p,p>（不修改算子）；其余链的 zeta/alpha
相对主链递推。每迭代仅一次 A·p（全部链共享），内存 O(N_shift) 个向量。
适用：A Hermitian 正定（如 D†D、奇偶 Schur 补 M_oo−M_oe·M_ee⁻¹·M_eo 的 γ₅ 变换）。
"""


def multishift_cg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor],
                  shifts: List[float], tol: float = 1e-6, max_iter: int = 1000,
                  x0: Optional[List[Optional[torch.Tensor]]] = None, if_rtol: bool = False,
                  verbose: bool = True) -> List[torch.Tensor]:
    """多质量 CG 求解 (A + σ_i)·x_i = b。

    Args:
        b: 右端项; matvec: 基础算子 A·v（不含移位）
        shifts: 移位列表（内部升序排序；最低者为主链）
        tol/if_rtol: 各移位统一的收敛容差（估计残差 ||r_i|| ≈ zeta_i·||r_0||）
        x0: 初解列表（None 元素取零；当前版本忽略初解差异——各链初值须同为
            b − A·x0 的 Krylov 起点，非零 x0 时自动退化为零初值并告警）
    Returns:
        与升序 shifts 对齐的解列表
    """
    n = len(shifts)
    if n == 0:
        raise ValueError("PYQCU::SOLVER::MULTISHIFTCG:\n shifts must be non-empty")
    order = sorted(range(n), key=lambda i: shifts[i])
    sig = [float(shifts[i]) for i in order]
    if any(sig[j + 1] - sig[j] < 1e-14 for j in range(n - 1)):
        raise ValueError("PYQCU::SOLVER::MULTISHIFTCG:\n duplicated shifts")
    if x0 is not None and any(x is not None for x in x0):
        if verbose:
            print("PYQCU::SOLVER::MULTISHIFTCG:\n Warning: nonzero x0 unsupported, using zero initial guess")
    xs = [_torch.zeros_like(b) for _ in range(n)]
    r = b.clone()
    ps = [b.clone() for _ in range(n)]
    b_norm = tools.norm(b)
    _tol = b_norm * tol if if_rtol else tol
    stop = float(_tol) ** 2
    zeta = [1.0] * n
    zeta_old = [1.0] * n
    beta = [0.0] * n
    alpha = [1.0] * n
    active = [True] * n
    v0 = tools.vdot(r, r)
    r2 = float(v0.real) if torch.is_tensor(v0) else float(v0)
    start_time = perf_counter()
    if verbose:
        print(f"PYQCU::SOLVER::MULTISHIFTCG:\n Norm of b:{b_norm}")
    converged_count = sum(1 for i in range(n) if zeta[i] * zeta[i] * r2 < stop)
    active = [zeta[i] * zeta[i] * r2 >= stop for i in range(n)]
    k = 0
    while k < max_iter and converged_count < n:
        # 主链移位 σ₀ 就地折入 Ap（quda axpyReDot 语义）——否则内部 r 漂移 σ·x
        Ap = matvec(ps[0])
        if sig[0] != 0.0:
            Ap = Ap + sig[0] * ps[0]
        p0 = ps[0]
        pAp = tools.vdot(p0, Ap).real
        if abs(float(pAp)) < 1e-30:
            raise RuntimeError(
                f"MultiShiftCG breakdown at iter {k}: <p,(A+σ)p> ≈ 0 "
                "(operator not positive definite on Krylov subspace?).")
        alpha_old = list(alpha)
        alpha[0] = r2 / float(pAp)
        for j in range(1, n):
            if not active[j]:
                continue
            c0 = zeta[j] * zeta_old[j] * alpha_old[0]
            c1 = alpha[0] * beta[0] * (zeta_old[j] - zeta[j])
            c2 = zeta_old[j] * alpha_old[0] * (1.0 + (sig[j] - sig[0]) * alpha[0])
            zeta_old[j] = zeta[j]
            denom = c1 + c2
            zeta_new = c0 / denom if denom != 0.0 else 0.0
            # 数值防护：c64 精度地板下 ζ 递推可能爆炸/失真，冻结该移位
            if not math.isfinite(zeta_new) or abs(zeta_new) > 1e15:
                zeta[j] = 0.0
                alpha[j] = 0.0
                continue
            zeta[j] = zeta_new
            alpha[j] = alpha[0] * zeta[j] / zeta_old[j] if zeta[j] != 0.0 else 0.0
        r2_old = r2
        r = r - alpha[0] * Ap
        r2_new = tools.vdot(r, r)
        r2 = float(r2_new.real) if torch.is_tensor(r2_new) else float(r2_new)
        beta[0] = r2 / r2_old
        xs[0] = xs[0] + alpha[0] * p0
        ps[0] = r + beta[0] * p0
        for j in range(1, n):
            if not active[j]:
                continue
            beta[j] = beta[0] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[0])
            xs[j] = xs[j] + alpha[j] * ps[j]
            ps[j] = zeta[j] * r + beta[j] * ps[j]
            est = zeta[j] * zeta[j] * r2
            if zeta[j] == 0.0 or est < stop:
                active[j] = False
                converged_count += 1
                if verbose:
                    print(f"PYQCU::SOLVER::MULTISHIFTCG:\n shift {sig[j]:.6g} converged at iter {k} "
                          f"(est residual {math.sqrt(est):.6e})")
        if active[0] and r2 < stop and converged_count >= n - 1:
            active[0] = False
            converged_count += 1
        # 注：σ₀ 已逐步折入 Ap，内部 r 恒为真残差（舍入级），无需中途重同步；
        # 周期性 r 替换/p 清理会打断 CG 共轭性导致发散（2026-08-22 实测），勿加回。
        if verbose and k % 50 == 0:
            print(f"PYQCU::SOLVER::MULTISHIFTCG:\n Iteration {k}: primary residual = {math.sqrt(r2):.6e}")
        k += 1
    total_time = perf_counter() - start_time
    if verbose:
        if converged_count < n:
            print("PYQCU::SOLVER::MULTISHIFTCG:\n Warning: Maximum iterations reached, may not have converged")
        print(f"PYQCU::SOLVER::MULTISHIFTCG:\n Total time: {total_time:.6f} seconds, iterations: {k}")
    return xs


def multishift_cg_true_residuals(b: torch.Tensor, shifted_matvec: Callable[[float, torch.Tensor], torch.Tensor],
                                 shifts: List[float], xs: List[torch.Tensor]) -> List[float]:
    """逐移位真实残差 ||b − (A+σ)x||（验证用；shifted_matvec(s,v) 返回 (A+σ·I)v）。"""
    res = []
    for s, x in zip(shifts, xs):
        r = b - shifted_matvec(s, x)
        val = tools.norm(r)
        res.append(float(val.item()) if torch.is_tensor(val) else float(val))
    return res
