from time import perf_counter
from typing import Callable, Optional, List
import torch
from pyqcu import tools
import pyqcu.cann as _torch

"""CA-CG（communication-avoiding / s-step CG）— quda lib/inv_ca_cg.cpp 思想移植。

结构：每外迭代构建 m 维 Krylov 基 {r, Ar, ..., A^{m-1}r}（m 次 matvec 仅一次同步块），
MGS 正交化基 W 并以相同线性组合携带像 AW（零额外 matvec），随后以一个 m×m 方程组整体推进：
极小残差系统 (AW†AW)γ = AW†r → x += Wγ。块末做可靠更新（真残差 r = b − Ax）。

与 quda 的差异（有意为之，2026-08-22 实验定案）：
  1. quda 默认 power basis 直接构造 Gram——无归一化时 ‖A^i r‖ 指数膨胀致系统病态、廉价残差
     与真残差脱钩（实测 N=64 即崩）；其 Chebyshev 备选依赖 λ 界且模板实例化在 quda 内已禁用。
     本实现改用 MGS 正交化基，数值稳健。
  2. 跨块 beta 对齐未移植：正交基下等价于重启 GMRES(m)/FOM(m)，收敛率≈GMRES(m)，
     单调性由极小残差系统保证（galerkin/FOM 变体残差可振荡，弃用）。
适用：Hermitian 正定算子（CG 家族约束）。容差建议 c128 ≥1e-12、c64 ≥1e-5。
"""


def cacg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor],
         tol: float = 1e-6, max_iter: int = 1000, x0: Optional[torch.Tensor] = None,
         n_krylov: int = 8, if_rtol: bool = False, verbose: bool = True,
         history: Optional[List[float]] = None) -> torch.Tensor:
    """s-step CG 求解 A·x = b（A Hermitian 正定）。

    Args:
        b: 右端项; matvec: A·v; n_krylov: 每外迭代基维数 m
        tol/if_rtol/verbose/history: 同 solver.bistabcg 约定（history 记真残差，
            每外迭代一个点）
    Returns:
        解张量（形状同 b）
    """
    x = x0.clone() if x0 is not None else _torch.zeros_like(b)
    r = b - matvec(x)
    r_norm = tools.norm(r)
    b_norm = tools.norm(b)
    if history is not None:
        history.append(float(r_norm))
    _tol = float(b_norm * tol if if_rtol else tol)
    if verbose:
        print(f"PYQCU::SOLVER::CACG:\n Norm of b:{b_norm}")
        print(f"PYQCU::SOLVER::CACG:\n Norm of r:{r_norm}")
    if r_norm < _tol:
        print("PYQCU::SOLVER::CACG:\n x0 is just right!")
        return x
    start_time = perf_counter()
    m = max(1, int(n_krylov))
    total_outer = 0
    converged = False
    while not converged and total_outer * m < max_iter:
        # 1) 幂基 raw=[r, Ar, ...] 与像 im=[Ar, A²r, ...]（m 次 matvec）
        raw = [r.clone()]
        im = []
        for _ in range(m):
            nxt = matvec(raw[-1])
            im.append(nxt)
            raw.append(nxt)
        raw = raw[:m]
        # 2) MGS 正交化 raw；像按相同组合变换（零额外 matvec）
        W: List[torch.Tensor] = []
        AW: List[torch.Tensor] = []
        for i in range(len(raw)):
            w = raw[i].clone()
            aw = im[i].clone()
            for u, uw in zip(W, AW):
                c = _torch.vdot(u, w)
                w = w - c * u
                aw = aw - c * uw
            nrm = float(tools.norm(w))
            if nrm < 1e-13:
                break
            w = w / nrm
            aw = aw / nrm
            for u, uw in zip(W, AW):
                c = _torch.vdot(u, w)
                w = w - c * u
                aw = aw - c * uw
            W.append(w)
            AW.append(aw)
        k = len(W)
        if k == 0:
            break
        # 3) 极小残差块解：批量 Gram（单次 matmul 替代 k² 次独立归约）
        Wm = torch.stack([w.reshape(-1) for w in W])
        AWm = torch.stack([a.reshape(-1) for a in AW])
        r_flat = r.reshape(-1)
        Gm = _torch.matmul(AWm.conj(), AWm.T)          # G_ij=<AW_i,AW_j>
        gm = _torch.matmul(AWm.conj(), r_flat)         # g_i=<AW_i,r>
        try:
            # 求解在 c128 域进行（c64 场时上转保精度），回转场精度
            gamma = torch.linalg.solve(Gm.to(torch.complex128),
                                       gm.to(torch.complex128))
        except Exception as e:
            raise RuntimeError(
                f"CA-CG breakdown at outer iter {total_outer}: Gram system singular "
                f"(basis collapse, m={m}).") from e
        x = x + _torch.matmul(gamma.to(W[0].dtype).unsqueeze(0), Wm).reshape(x.shape).to(x.dtype)
        # 4) 可靠更新：真残差
        r = b - matvec(x)
        r_norm = tools.norm(r)
        total_outer += 1
        if history is not None:
            history.append(float(r_norm))
        if verbose and total_outer % 10 == 0:
            print(f"PYQCU::SOLVER::CACG:\n Iteration ~{total_outer * m}: Residual = {float(r_norm):.6e}")
        if float(r_norm) < _tol:
            converged = True
    total_time = perf_counter() - start_time
    if verbose:
        if not converged:
            print("PYQCU::SOLVER::CACG:\n Warning: Maximum iterations reached, may not have converged")
        print(f"PYQCU::SOLVER::CACG:\n Total time: {total_time:.6f} seconds")
        print(f"PYQCU::SOLVER::CACG:\n Final residual: {float(r_norm):.2e}")
    return x
