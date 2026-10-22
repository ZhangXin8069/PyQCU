from time import perf_counter
from typing import Callable, Optional, List, Tuple
import torch
from pyqcu import tools
import pyqcu.cann as _torch

"""thick-restart Lanczos（TR-Lanczos, Wu & Simon）— Hermitian 算子最低本征对求解底座。

参考 quda lib/eig_trlm.cpp 思想（TRLM）：正交基 + 显式对称投影矩阵 H（每步以内积填充
一列，全重正交化保稳定），基满后 Rayleigh–Ritz 取 k 个最低 Ritz 向量与残差方向重启，
新 H 按 Wu–Simon arrowhead 公式解析重建（H[i][k]=beta·S[-1,i]，对角=θ_i）。后续步的
内积列填充与 arrowhead 理论值自洽。为 GMRES-DR/eigCG 类 deflation 求解器提供底座。

收敛判据：每个目标对的残差估计 |beta·S[last,i]| < tol·|theta_i|（全达标才返回）。
适用：A Hermitian（如 D†D、Schur 补的 γ₅ 变换、Laplacian）。
"""


def tr_lanczos(matvec: Callable[[torch.Tensor], torch.Tensor],
               v0: torch.Tensor, ncv: int = 32, k: int = 6,
               tol: float = 1e-8, max_iter: int = 400,
               verbose: bool = True) -> Tuple[List[float], List[torch.Tensor]]:
    """求 Hermitian A 的 k 个最低本征对（代数最小）。

    Args:
        matvec: A·v
        v0: 初始向量（形状即算子作用域；随机 Z2 噪声即可，必填）
        ncv: 基容量 m（要求 ≥ 2k+2）
        tol/max_iter: 相对残差容差 / matvec 步数上限
    Returns:
        (evals 升序列表, 对应 Ritz 向量列表)

    Breakdown：不变子空间过早出现且未达判据时注入新随机方向续跑。
    """
    if v0 is None:
        raise ValueError("PYQCU::SOLVER::TRLANCZOS:\n v0 is required (defines the operator domain shape)")
    if ncv < 2 * k + 2:
        raise ValueError("PYQCU::SOLVER::TRLANCZOS:\n ncv must be >= 2*k+2")
    start_time = perf_counter()
    v0_normed = v0 / tools.norm(v0)
    V: List[torch.Tensor] = [v0_normed]
    Vm = v0_normed.reshape(1, -1)          # 堆叠基 [n, N]：批量内积/组合用
    H: List[List[float]] = [[float(_torch.vdot(v0_normed, matvec(v0_normed)).real)]]
    n_mv = 1
    pending = False

    def _ritz():
        T = torch.tensor(H, dtype=torch.float64)
        theta, S = torch.linalg.eigh(T)
        return theta.tolist(), S

    def _form_ritz(S: torch.Tensor, idx: int) -> torch.Tensor:
        coef = S[:len(V), idx].to(V[0].dtype).to(V[0].device).reshape(1, -1)
        return _torch.matmul(coef, Vm).reshape(V[0].shape)

    while n_mv < max_iter:
        j = len(V) - 1
        Av = matvec(V[j])
        n_mv += 1
        col_all = _torch.matmul(Vm, Av.reshape(-1).conj()).real   # 单次批量归约
        col = [float(col_all[l]) for l in range(j + 1)]
        w = Av
        for l in range(j + 1):
            w = w - col[l] * V[l]
        for _ in range(2):
            for u in V:
                c = _torch.vdot(u, w)
                w = w - c * u
        beta = float(tools.norm(w))
        # 填充第 j 列并对称双写（eigh 读下三角——单侧填充会让 H 读到垃圾）
        for l in range(j):
            H[l][j] = col[l]
            H[j][l] = col[l]
        H[j][j] = col[j]
        pending = False
        if len(V) >= k + 2:
            theta, S = _ritz()
            res = [abs(beta * float(S[-1, i])) for i in range(k)]
            if verbose and n_mv % 50 < 3:
                print(f"PYQCU::SOLVER::TRLANCZOS:\n mv={n_mv} theta[:k]={theta[:k]} "
                      f"max_rel_res={max(res[i] / max(abs(theta[i]), 1e-30) for i in range(k)):.3e}")
            if all(r < tol * max(abs(theta[i]), 1e-30) for i, r in enumerate(res)):
                evals = [theta[i] for i in range(k)]
                evecs = [_form_ritz(S, i) for i in range(k)]
                # 真残差核验（防低精度下估计假收敛）
                verified = True
                for lam, y in zip(evals, evecs):
                    rv = float(tools.norm(matvec(y) - lam * y))
                    n_mv += 1
                    if not (rv <= 100.0 * tol * max(abs(lam), 1e-30)):
                        verified = False
                        break
                if verified:
                    if verbose:
                        total_time = perf_counter() - start_time
                        print(f"PYQCU::SOLVER::TRLANCZOS:\n Converged {k} pairs, matvecs={n_mv}, "
                              f"time={total_time:.3f}s")
                    return evals, evecs
        if len(V) == ncv:
            # thick restart：保留 k 个最低 Ritz 向量 + 残差方向；arrowhead 重建 H
            theta, S = _ritz()
            kept = [_form_ritz(S, i) for i in range(k)]
            V = kept + [w / beta]
            Vm = torch.stack([v.reshape(-1) for v in V])
            H = [[0.0] * (k + 1) for _ in range(k + 1)]
            for i in range(k):
                H[i][i] = theta[i]
                H[i][k] = beta * float(S[-1, i])
                H[k][i] = H[i][k]
            pending = True   # 末行/列为零占位，须待下步填充后才可读谱
        elif beta < 1e-14:
            # 不变子空间：注入随机正交方向
            w = _torch.randn_like(V[0])
            for _ in range(2):
                for u in V:
                    w = w - _torch.vdot(u, w) * u
            V.append(w / tools.norm(w))
            Vm = torch.cat([Vm, V[-1].reshape(1, -1)], 0)
            for row in H:
                row.append(0.0)
            H.append([0.0] * len(V))
            pending = True
            pending = True
        else:
            V.append(w / beta)
            Vm = torch.cat([Vm, V[-1].reshape(1, -1)], 0)
            for row in H:
                row.append(0.0)
            H.append([0.0] * len(V))
            pending = True
    if pending:
        # 占位列未填即耗尽预算：补算末列（1 次 matvec）后再取 Ritz，防零占位污染谱
        j = len(V) - 1
        Av = matvec(V[j])
        n_mv += 1
        col_all = _torch.matmul(Vm, Av.reshape(-1).conj()).real   # 单次批量归约
        col = [float(col_all[l]) for l in range(j + 1)]
        for l in range(j):
            H[l][j] = col[l]
            H[j][l] = col[l]
        H[j][j] = col[j]
    theta, S = _ritz()
    n_out = min(k, len(V))
    evals = [theta[i] for i in range(n_out)]
    evecs = [_form_ritz(S, i) for i in range(n_out)]
    if verbose:
        print("PYQCU::SOLVER::TRLANCZOS:\n Warning: Maximum iterations reached, may not have converged")
        print(f"PYQCU::SOLVER::TRLANCZOS:\n Total time: {perf_counter() - start_time:.6f} seconds")
    return evals, evecs
