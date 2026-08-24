import numpy as np
import torch
import mpi4py.MPI as MPI
from pyqcu import tools
import pyqcu.cann as _torch

"""Wuppertal 高斯 smearing（费米子场）— 参考 PyQUDA gaussianSmear / quda wuppertalSmear。

迭代（Chroma 同约定，σ = ρ²/(4·nstep)）：
    x'(x) = (1 − 6σ)·x(x) + σ·Σ_μ [ U_μ(x)·x(x+μ̂) + U_μ†(x−μ̂)·x(x−μ̂) ]

规范场在迭代期间固定 → U 的 halo 每次调用只交换一次；
费米子场每步变化 → src 边界每步重算（stout BUGFIX 2026-07-28 同教训）。
"""

force_use_npu = False


def _exchange_boundaries(field: torch.Tensor, dims_num: int):
    """对张量的最后 4 轴（xyzt）做 ±1 halo 交换。

    Returns:
        (head_list, tail_list)：head_list[w] = 负方向邻居的尾切片，
        tail_list[w] = 正方向邻居的头切片（仅 grid_size>1 的方向非占位）。
    """
    grid_size = tools.give_grid_size()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_plus_list = [tools.give_rank_plus(ward=w) for w in range(4)]
    rank_minus_list = [tools.give_rank_minus(ward=w) for w in range(4)]
    head_list = [torch.zeros([])] * 4
    tail_list = [torch.zeros([])] * 4
    for w in range(4):
        if grid_size[w] != 1:
            tail4send = field[tools.slice_dim(dims_num=dims_num, ward=w, point=-1)].cpu().contiguous().numpy()
            head4recv = np.zeros_like(tail4send)
            comm.Sendrecv(sendbuf=tail4send, dest=rank_plus_list[w], sendtag=rank,
                          recvbuf=head4recv, source=rank_minus_list[w], recvtag=rank_minus_list[w])
            head_list[w] = torch.from_numpy(head4recv).to(device=field.device)
            head4send = field[tools.slice_dim(dims_num=dims_num, ward=w, point=0)].cpu().contiguous().numpy()
            tail4recv = np.zeros_like(head4send)
            comm.Sendrecv(sendbuf=head4send, dest=rank_minus_list[w], sendtag=rank_minus_list[w],
                          recvbuf=tail4recv, source=rank_plus_list[w], recvtag=rank_plus_list[w])
            tail_list[w] = torch.from_numpy(tail4recv).to(device=field.device)
    return head_list, tail_list


def wuppertal_smear(src: torch.Tensor, U: torch.Tensor, rho: float = 4.0,
                    nstep: int = 40, support_parallel: bool = False,
                    verbose: bool = False) -> torch.Tensor:
    """Wuppertal 高斯 smearing。

    Args:
        src: 费米子场 [4, 3, Lx, Ly, Lz, Lt]
        U: 规范场 [3, 3, 4, Lx, Ly, Lz, Lt]，迭代期间固定
        rho: 高斯宽度 ρ；sigma = rho²/(4·nstep)
        nstep: 迭代步数 N
        support_parallel: MPI 多进程支持（每步交换 src 边界）
    Returns:
        smeared 场（形状同 src）
    """
    assert src.ndim == 6 and src.shape[0] == 4 and src.shape[1] == 3, \
        "PYQCU::SMEAR::WUPPERTAL:\n src must be [4, 3, Lx, Ly, Lz, Lt]"
    assert nstep >= 1, \
        "PYQCU::SMEAR::WUPPERTAL:\n nstep must be >= 1 (sigma = rho^2/(4*nstep) divides by nstep)"
    sigma = rho * rho / (4.0 * nstep)
    # 空间三维 (x,y,z) smear，6 邻居 — 与中心系数 (1-6*sigma) 自洽（文献 Wuppertal/quda
    # wuppertalSmear 仅空间；时间切片保持局域）。若含 t(-1) 则为 8 邻居，常数场每步
    # 被放大 x(1+2*sigma)，nstep 步后指数发散（实测 U=I 常数场 dev=44@nstep=10, rho=4）。
    wards = [-4, -3, -2]
    x = src.clone()
    grid_size = tools.give_grid_size()
    u_head_list, u_tail_list = [torch.zeros([])] * 4, [torch.zeros([])] * 4
    if support_parallel:
        u_head_list, u_tail_list = _exchange_boundaries(U, dims_num=7)
    for step in range(nstep):
        s_head_list, s_tail_list = [torch.zeros([])] * 4, [torch.zeros([])] * 4
        if support_parallel:
            s_head_list, s_tail_list = _exchange_boundaries(x, dims_num=x.ndim)
        acc = _torch.zeros_like(x)
        for mu in range(len(wards)):
            w = wards[mu]
            U_mu = U[:, :, mu]
            # 前向：U_μ(x)·x(x+μ̂)
            x_plus = _torch.roll(x, -1, w)
            # 后向：U_μ†(x−μ̂)·x(x−μ̂)；V[a,b](y)=conj(U[b,a](y))
            V = U_mu.conj().permute(1, 0, *range(2, U_mu.ndim))
            V_roll = _torch.roll(V, +1, w)
            x_minus = _torch.roll(x, +1, w)
            if support_parallel and grid_size[mu] != 1:
                x_plus[tools.slice_dim(dims_num=x.ndim, ward=mu, point=-1)] = \
                    s_tail_list[mu].to(x_plus.dtype)
                x_minus[tools.slice_dim(dims_num=x.ndim, ward=mu, point=0)] = \
                    s_head_list[mu].to(x_minus.dtype)
                V_roll[tools.slice_dim(dims_num=V_roll.ndim, ward=mu, point=0)] = \
                    u_head_list[mu].conj().permute(1, 0, *range(2, U_mu.ndim)).to(V_roll.dtype)
            acc = acc + _torch.einsum("abxyzt,mbxyzt->maxyzt", U_mu, x_plus)
            acc = acc + _torch.einsum("abxyzt,mbxyzt->maxyzt", V_roll, x_minus)
        x = (1.0 - 6.0 * sigma) * x + sigma * acc
        if verbose:
            print(f"PYQCU::SMEAR::WUPPERTAL:\n step {step}: norm={_torch.norm(x)}")
    return x
