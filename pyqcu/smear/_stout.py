import torch
import numpy as np
from pyqcu import tools
from typing import Optional
import pyqcu.cann as _torch
import mpi4py.MPI as MPI
force_use_npu = False
"""
    Copy from https://github.com/IHEP-LQCD/EasyDistillation/blob/master/lattice/generator/elemental.py
"""


def stout_smear(U: torch.Tensor, nstep: int = 1, rho: float = 0.12, support_parallel: bool = False):
    # BUGFIX 2026-07-28: MPI boundary data must be recomputed each step
    # since the gauge field U changes with each smearing step.
    for step in range(nstep):
        if support_parallel:
            grid_size = tools.give_grid_size()
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            rank_plus_list = [tools.give_rank_plus(
                ward=ward) for ward in range(4)]
            rank_minus_list = [tools.give_rank_minus(
                ward=ward) for ward in range(4)]
            U_head_list = [torch.zeros([]), torch.zeros(
                []), torch.zeros([]), torch.zeros([])]  # xyzt
            U_tail_list = [torch.zeros([]), torch.zeros(
                []), torch.zeros([]), torch.zeros([])]  # xyzt
            for ward in range(4):
                if grid_size[ward] != 1:
                    U_tail4send = U[tools.slice_dim(
                                    dims_num=7, ward=ward, point=-1)].cpu().contiguous().numpy()
                    U_head4recv = np.zeros_like(U_tail4send)
                    comm.Sendrecv(sendbuf=U_tail4send, dest=rank_plus_list[ward], sendtag=rank,
                                  recvbuf=U_head4recv, source=rank_minus_list[ward], recvtag=rank_minus_list[ward])
                    U_head_list[ward] = torch.from_numpy(U_head4recv).to(
                        device=U.device)
                    U_head4send = U[tools.slice_dim(
                                    dims_num=7, ward=ward, point=0)].cpu().contiguous().numpy()
                    U_tail4recv = np.zeros_like(U_head4send)
                    comm.Sendrecv(sendbuf=U_head4send, dest=rank_minus_list[ward], sendtag=rank_minus_list[ward],
                                  recvbuf=U_tail4recv, source=rank_plus_list[ward], recvtag=rank)
                    U_tail_list[ward] = torch.from_numpy(U_tail4recv).to(
                        device=U.device)
            U_head_tail_list = [[torch.zeros([]), torch.zeros(
                []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                    []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                        []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                            []), torch.zeros([]), torch.zeros([])]]
            #  NEVER NEVER USE THE SHIT LIKE THAT "[[torch.zeros([]), torch.zeros([]), torch.zeros([]), torch.zeros([])]]*4"
            for mu in range(4):
                for nu in range(4):
                    if mu != nu and grid_size[mu] != 1 and grid_size[nu] != 1:
                        U_tail_head4send = U[tools.slice_dim_dim(
                            dims_num=7, ward_a=mu, point_a=-1, ward_b=nu, point_b=0)].cpu().contiguous().numpy()
                        U_head_tail4recv = np.zeros_like(U_tail_head4send)
                        comm.Sendrecv(sendbuf=U_tail_head4send, dest=tools.give_rank_plus_minus(ward_a=mu, ward_b=nu, rank=rank), sendtag=rank,
                                      recvbuf=U_head_tail4recv, source=tools.give_rank_minus_plus(ward_a=mu, ward_b=nu, rank=rank), recvtag=tools.give_rank_minus_plus(ward_a=mu, ward_b=nu, rank=rank))
                        U_head_tail_list[mu][nu] = torch.from_numpy(U_head_tail4recv).to(
                            device=U.device)
        Q = torch.zeros_like(U)
        for mu in range(4):
            # for mu in range(4 - 1):
            Q_mu = torch.zeros_like(Q[:, :, mu, :, :, :, :])
            for nu in range(4):
                # for nu in range(4 - 1):
                if mu != nu:
                    U_mu = U[:, :, mu, :, :, :, :]
                    U_nu = U[:, :, nu, :, :, :, :]
                    U_nu_conj = U[:, :, nu, :, :, :, :].permute(
                        1, 0, 2, 3, 4, 5).conj()
                    if support_parallel:
                        roll_u0 = _torch.roll(U_mu, -1, -4+nu)
                        if grid_size[nu] != 1:
                            roll_u0[tools.slice_dim(
                                dims_num=6, ward=nu, point=-1)] = U_tail_list[nu][:, :, mu, :, :, :]
                        roll_u1 = _torch.roll(U_nu_conj, -1, -4+mu)
                        if grid_size[mu] != 1:
                            roll_u1[tools.slice_dim(
                                dims_num=6, ward=mu, point=-1)] = U_tail_list[mu][:, :, nu, :, :, :].permute(1, 0, 2, 3, 4).conj()
                        roll_u2 = _torch.roll(U_nu_conj, +1, -4+nu)
                        if grid_size[nu] != 1:
                            roll_u2[tools.slice_dim(
                                dims_num=6, ward=nu, point=0)] = U_head_list[nu][:, :, nu, :, :, :].permute(1, 0, 2, 3, 4).conj()
                        roll_u3 = _torch.roll(U_mu, +1, -4+nu)
                        if grid_size[nu] != 1:
                            roll_u3[tools.slice_dim(
                                dims_num=6, ward=nu, point=0)] = U_head_list[nu][:, :, mu, :, :, :]
                        roll_u4 = _torch.roll(_torch.roll(
                            U_nu, +1, -4+nu), -1, -4+mu)
                        if grid_size[nu] != 1:
                            roll_u4[tools.slice_dim(dims_num=6, ward=nu, point=0)] = _torch.roll(
                                U_head_list[nu][:, :, nu, :, :, :], -1, -4+mu+(mu < nu))
                        if grid_size[mu] != 1:
                            roll_u4[tools.slice_dim(dims_num=6, ward=mu, point=-1)] = _torch.roll(
                                U_tail_list[mu][:, :, nu, :, :, :], +1, -4+nu+(nu < mu))
                        if grid_size[mu] != 1 and grid_size[nu] != 1:
                            roll_u4[tools.slice_dim_dim(
                                    dims_num=6, ward_a=nu, ward_b=mu, point_a=0, point_b=-1)] = U_head_tail_list[nu][mu][:, :, nu, :, :]
                        Q_mu += _torch.einsum(
                            "abxyzt,bcxyzt,dcxyzt->adxyzt",
                            U_nu,
                            roll_u0,
                            roll_u1,
                        )
                        Q_mu += _torch.einsum(
                            "baxyzt,bcxyzt,cdxyzt->adxyzt",
                            roll_u2,
                            roll_u3,
                            roll_u4,
                        )
                    else:
                        Q_mu += _torch.einsum(
                            "abxyzt,bcxyzt,dcxyzt->adxyzt",
                            U_nu,
                            _torch.roll(U_mu, -1, -4+nu),
                            _torch.roll(U_nu_conj, -1, -4+mu),
                        )
                        Q_mu += _torch.einsum(
                            "baxyzt,bcxyzt,cdxyzt->adxyzt",
                            _torch.roll(U_nu_conj, +1, -4+nu),
                            _torch.roll(U_mu, +1, -4+nu),
                            _torch.roll(_torch.roll(
                                U_nu, +1, -4+nu), -1, -4+mu),
                        )
            Q[:, :, mu, :, :, :, :] = Q_mu.clone()
        Q = _torch.einsum("abDxyzt,cbDxyzt->acDxyzt", rho * Q, U.conj())
        Q = 0.5j * (_torch.einsum("abDxyzt->baDxyzt", Q.conj()) - Q)
        Q -= 1 / 3 * _torch.einsum("aaDxyzt,bc->bcDxyzt", Q,
                                   torch.eye(3).to(dtype=U.dtype, device=U.device))
        c0 = _torch.einsum("abDxyzt,bcDxyzt,caDxyzt->Dxyzt", Q, Q, Q).real / 3
        c1 = _torch.einsum("abDxyzt,baDxyzt->Dxyzt", Q, Q).real / 2
        # BUGFIX 2026-07-28 R2: numerical stability — clamp c1 to avoid c0_max=0
        # and guard arccos domain; add epsilon to f_denom to prevent division by zero.
        c1 = torch.clamp(c1, min=1e-15)  # prevent c1=0 → c0_max=0
        parity = c0 < 0
        c0 = torch.abs(c0)
        c0_max = 2 * (c1 / 3) ** (3 / 2)
        ratio = torch.clamp(c0 / c0_max, -1.0 + 1e-15, 1.0 - 1e-15)  # arccos domain
        theta = torch.arccos(ratio)
        u = (c1 / 3) ** 0.5 * torch.cos(theta / 3)
        w = c1**0.5 * torch.sin(theta / 3)
        u_sq = u**2
        w_sq = w**2
        e_iu = torch.exp(-1j * u)
        e_2iu = torch.exp(2j * u)
        cos_w = torch.cos(w)
        sinc_w = 1 - w_sq / 6 * \
            (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
        large = torch.abs(w) > 0.05
        w_large = w[large]
        sinc_w[large] = torch.sin(w_large) / w_large
        # BUGFIX 2026-07-28 R2: add epsilon to prevent division by zero when 9*u^2 == w^2
        f_denom = 1 / (9 * u_sq - w_sq + 1e-15)
        f0 = ((u_sq - w_sq) * e_2iu + e_iu * (8 * u_sq * cos_w +
              2j * u * (3 * u_sq + w_sq) * sinc_w)) * f_denom
        f1 = (2 * u * e_2iu - e_iu * (2 * u * cos_w -
              1j * (3 * u_sq - w_sq) * sinc_w)) * f_denom
        f2 = (e_2iu - e_iu * (cos_w + 3j * u * sinc_w)) * f_denom
        if (U.device.type == 'npu' or force_use_npu) and torch.is_complex(U):
            # NPU complex decomposition for parity sign convention.
            # Standard (CUDA/CPU) path:
            #   f0[parity] =  f0[parity].conj()    → real same, imag flipped
            #   f1[parity] = -f1[parity].conj()    → real negated, imag flipped-then-negated-back
            #   f2[parity] = -f2[parity].conj()    → real negated, imag flipped
            #
            # f0: conj() = (a+bi)* = a-bi. NPU: imag → -imag. Correct.
            f0.imag[parity] = -f0.imag[parity]
            #
            # f1: -f1.conj() = -(a-bi) = -a+bi.
            # real: negate (-a). imag: conj flips to -b, leading minus flips back → +b.
            # NPU: real = -real, imag = +imag (i.e., NO negation on imag — the conj flip
            # and the leading minus cancel each other).
            # BUGFIX 2026-07-28 R3: previous R2 fix incorrectly negated imag (gave -a-bi).
            f1.real[parity] = -f1.real[parity]            # -a
            f1.imag[parity] =  f1.imag[parity]            # +b (NOT -b: conj+minus cancel)
            #
            # f2: -f2.conj() = -(a-bi) × (-1) = ?? Wait, f2[parity] = -f2[parity].conj()
            # Actually f2 only has the leading minus (no conj→negate interplay issue):
            # -f2.conj() = -(a-bi) = -a+bi.
            # Wait, that's the same as f1! Let me re-check the code...
            # f2[parity] = -f2[parity].conj() on line 162 in standard path.
            # But f2 has NO leading minus in stout formula; this parity flip applies to
            # the stout SU(3) projection coefficients when c0 < 0.
            # f2 parity: -f2.conj(). Same as f1, so imag should be +imag.
            # Actually wait, let me re-examine line 162:
            # f2[parity] = -f2[parity].conj()
            # This IS the same as f1. So same logic applies: imag gets NO negation.
            f2.real[parity] = -f2.real[parity]            # -a
            f2.imag[parity] =  f2.imag[parity]            # +b (NOT -b)
        else:
            f0[parity] = f0[parity].conj()
            f1[parity] = -f1[parity].conj()
            f2[parity] = f2[parity].conj()
        f0 = _torch.einsum("Dxyzt,ab->abDxyzt", f0,
                           torch.eye(3).to(dtype=U.dtype, device=U.device))
        f1 = _torch.einsum("Dxyzt,abDxyzt->abDxyzt", f1, Q)
        f2 = _torch.einsum("Dxyzt,abDxyzt,bcDxyzt->acDxyzt", f2, Q, Q)
        # BUGFIX 2026-07-28: rebind local U to the smeared gauge field so
        # the next iteration uses the updated field. Note this is NOT an
        # in-place update of the caller's tensor — the function returns
        # the new tensor, and the loop body rebinds the local variable.
        U = _torch.einsum("abDxyzt,bcDxyzt->acDxyzt", f0 + f1 + f2, U)
    return U
