import torch
from time import perf_counter
from typing import Callable, Optional, List
from pyqcu import tools
import pyqcu.cann as _torch


def _givens_rotation(H: List[List[complex]], g: List[complex],
                     cs: List[complex], sn: List[complex], j: int):
    """复 Givens 旋转：将 Hessenberg 列 j 旋至上三角（DDalphaAMG-SM fgmres.cpp::rotation）。"""
    for i in range(j):
        temp = cs[i].conjugate() * H[i][j] + sn[i].conjugate() * H[i + 1][j]
        H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j]
        H[i][j] = temp
    # The rotation must be unitary in complex arithmetic.  In particular,
    # ``sqrt(abs(a*a + b*b))`` can spuriously vanish when the two squared
    # phases cancel (for example a=1j, b=1).  QUDA's FlexArnoldiProcedure
    # uses sqrt(norm(a) + norm(b)), i.e. the Euclidean norm of the pair.
    den = (abs(H[j][j]) ** 2 + abs(H[j + 1][j]) ** 2) ** 0.5
    if den == 0.0:
        sn_j, cs_j = 0.0 + 0.0j, 1.0 + 0.0j
        H[j][j] = H[j][j]
        H[j + 1][j] = 0.0
    else:
        sn_j = H[j + 1][j] / den
        cs_j = H[j][j] / den
        H[j][j] = cs_j.conjugate() * H[j][j] + sn_j.conjugate() * H[j + 1][j]
        H[j + 1][j] = 0.0
    sn[j] = sn_j
    cs[j] = cs_j
    g[j + 1] = -sn_j * g[j]
    g[j] = cs_j.conjugate() * g[j]


def _solve_upper_triangular(H: List[List[complex]], g: List[complex],
                            n: int) -> List[complex]:
    """上三角回代（DDalphaAMG-SM fgmres.cpp::solve_upper_triangular）。"""
    eta = [0.0 + 0.0j] * n
    for i in range(n - 1, -1, -1):
        s = g[i]
        for jj in range(i + 1, n):
            s -= H[i][jj] * eta[jj]
        # Arnoldi can break down for a singular/zero coarse operator.  Keep
        # the iterate finite and let the true-residual check decide whether
        # a later restart can make progress; do not divide by an exact zero.
        if abs(H[i][i]) == 0.0:
            eta[i] = 0.0 + 0.0j
        else:
            eta[i] = s / H[i][i]
    return eta


def fgmres(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor],
           tol: float = 1e-6, max_iter: int = 1000, restart: int = 30,
           x0: Optional[torch.Tensor] = None,
           precond: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
           if_rtol: bool = False, verbose: bool = True,
           history: Optional[List[float]] = None) -> torch.Tensor:
    """FGMRES(m) — 右预条件广义极小残量求解器（flexible GMRES）。

    算法移植参考：refer/git-rep/DDalphaAMG-SM/{include/fgmres.h, src/fgmres.cpp}
    （Arnoldi 正交化 + 复 Givens 旋转 + 上三角回代 + 重启）；
    右预条件可逐迭代变化（flexible），precond=None 时退化为标准 GMRES(m)。

    Args:
        b: 右端项（任意形状，时空布局 ...xyzt 或展平均可）
        matvec: 算子作用 matvec(src) → dest（形状与 src 相同）
        tol: 收敛容差；if_rtol=True 时为相对容差 tol*||b||
        max_iter: 内迭代总预算（跨重启周期累计）
        restart: 重启长度 m（Krylov 子空间维数上限）
        x0: 初值解（None → 零初始解）
        precond: 右预条件算子（如 SAP/MG smoother），逐迭代可变化
        if_rtol: 容差语义开关
        verbose: 日志输出（PYQCU::SOLVER::FGMRES）
        history: 传入 list 时逐内迭代追加残差估计 |g[j+1]|，
                 每重启周期末尾追加真实残差 ||b-Ax||（画收敛曲线用）

    Returns:
        解张量（形状同 b）
    """
    shp = b.shape
    x = x0.clone() if x0 is not None else _torch.zeros_like(b)
    r = (b - matvec(x)).reshape(-1)
    b_norm = tools.norm(b.reshape(-1))
    _tol = b_norm * tol if if_rtol else tol
    err = tools.norm(r)
    if verbose:
        print(f"PYQCU::SOLVER::FGMRES:\n Norm of b:{b_norm}")
        print(f"PYQCU::SOLVER::FGMRES:\n Norm of r:{err}")
    # A zero coarse residual is a normal event in a V-cycle.  With a
    # relative tolerance its threshold is also zero, so ``err < _tol`` is
    # false and the old code would enter Arnoldi with beta=0 and create NaNs.
    if b_norm == 0.0 or err == 0.0:
        if verbose:
            print("PYQCU::SOLVER::FGMRES:\n zero right-hand side/residual")
        return x
    if err < _tol:
        if verbose:
            print("PYQCU::SOLVER::FGMRES:\n x0 is just right!")
        return x
    m = max(1, int(restart))
    V = [_torch.zeros_like(r) for _ in range(m + 1)]
    Z = [_torch.zeros_like(r) for _ in range(m)]
    start_time = perf_counter()
    n_iter = 0
    converged = False
    while n_iter < max_iter and not converged:
        beta = err
        V[0] = r / beta
        H = [[0.0 + 0.0j] * m for _ in range(m + 1)]
        g = [0.0 + 0.0j] * (m + 1)
        g[0] = complex(beta)
        sn = [0.0 + 0.0j] * m
        cs = [0.0 + 0.0j] * m
        n_inner = min(m, max_iter - n_iter)
        max_inner = 0
        for j in range(n_inner):
            v_j = V[j]
            z_j = precond(v_j.reshape(shp)).reshape(-1) if precond is not None else v_j
            Z[j] = z_j
            w = matvec(z_j.reshape(shp)).reshape(-1)
            for i in range(j + 1):
                h_ij = tools.vdot(V[i], w).item()
                H[i][j] = h_ij
                w = w - h_ij * V[i]
            h_jp1 = float(tools.norm(w))
            H[j + 1][j] = h_jp1
            if h_jp1 > 0.0:
                V[j + 1] = w / h_jp1
            _givens_rotation(H, g, cs, sn, j)
            n_iter += 1
            max_inner = j + 1
            res_est = abs(g[j + 1])
            if history is not None:
                history.append(float(res_est))
            if verbose:
                print(
                    f"PYQCU::SOLVER::FGMRES:\n Iteration {n_iter}: Residual(est) = {res_est:.6e}")
            if res_est < _tol:
                break
        eta = _solve_upper_triangular(H, g, max_inner)
        for jj in range(max_inner):
            x = x + eta[jj] * Z[jj].reshape(shp)
        r = (b - matvec(x)).reshape(-1)
        err = tools.norm(r)
        if history is not None:
            history.append(float(err))
        if verbose:
            print(
                f"PYQCU::SOLVER::FGMRES:\n Restart cycle: Residual(true) = {err:.6e}")
        if err < _tol:
            converged = True
    total_time = perf_counter() - start_time
    if verbose:
        status = "Converged" if converged else "Warning: Maximum iterations reached, may not have converged"
        print(f"PYQCU::SOLVER::FGMRES:\n {status}")
        print(f"PYQCU::SOLVER::FGMRES:\n Total iterations: {n_iter}")
        print(f"PYQCU::SOLVER::FGMRES:\n Total time: {total_time:.6f} seconds")
        print(f"PYQCU::SOLVER::FGMRES:\n Final residual: {err:.2e}")
    return x
