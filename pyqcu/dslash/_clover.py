import torch
import numpy as np
import mpi4py.MPI as MPI
from pyqcu import lattice, tools
import pyqcu.cann as _torch

def make_clover(U: torch.Tensor, kappa: float = 0.1,
                u_0: float = 1.0, support_parallel: bool = False, verbose: bool = False) -> torch.Tensor:
    """
    Give Clover term:
    $$
    \frac{a^2\kappa}{u_0^4}\sum_{\mu<\nu}\sigma_{\mu \nu}F_{\mu \nu}\delta_{x,y}
    $$
    """
    if verbose:
        print("PYQCU::DSLASH::CLOVER:\n Applying Dirac operator...")
        print(f"PYQCU::DSLASH::CLOVER:\n Gauge field shape: {U.shape}")
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
                comm.Barrier()
                comm.Sendrecv(sendbuf=U_tail4send, dest=rank_plus_list[ward], sendtag=rank,
                              recvbuf=U_head4recv, source=rank_minus_list[ward], recvtag=rank_minus_list[ward])
                comm.Barrier()
                U_head_list[ward] = torch.from_numpy(U_head4recv).to(
                    device=U.device)
                U_head4send = U[tools.slice_dim(
                                dims_num=7, ward=ward, point=0)].cpu().contiguous().numpy()
                U_tail4recv = np.zeros_like(U_head4send)
                comm.Barrier()
                comm.Sendrecv(sendbuf=U_head4send, dest=rank_minus_list[ward], sendtag=rank_minus_list[ward],
                              recvbuf=U_tail4recv, source=rank_plus_list[ward], recvtag=rank)
                comm.Barrier()
                U_tail_list[ward] = torch.from_numpy(U_tail4recv).to(
                    device=U.device)
        U_head_tail_list = [[torch.zeros([]), torch.zeros(
            []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                    []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                        []), torch.zeros([]), torch.zeros([])]]
        U_head_head_list = [[torch.zeros([]), torch.zeros(
            []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                    []), torch.zeros([]), torch.zeros([])], [torch.zeros([]), torch.zeros(
                        []), torch.zeros([]), torch.zeros([])]]
        U_tail_tail_list = [[torch.zeros([]), torch.zeros(
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
                    comm.Barrier()
                    comm.Sendrecv(sendbuf=U_tail_head4send, dest=tools.give_rank_plus_minus(ward_a=mu, ward_b=nu, rank=rank), sendtag=rank,
                                  recvbuf=U_head_tail4recv, source=tools.give_rank_minus_plus(ward_a=mu, ward_b=nu, rank=rank), recvtag=tools.give_rank_minus_plus(ward_a=mu, ward_b=nu, rank=rank))
                    comm.Barrier()
                    U_head_tail_list[mu][nu] = torch.from_numpy(U_head_tail4recv).to(
                        device=U.device)

                    U_tail_tail4send = U[tools.slice_dim_dim(
                        dims_num=7, ward_a=mu, point_a=-1, ward_b=nu, point_b=-1)].cpu().contiguous().numpy()
                    U_head_head4recv = np.zeros_like(U_tail_tail4send)
                    comm.Barrier()
                    comm.Sendrecv(sendbuf=U_tail_tail4send, dest=tools.give_rank_plus_plus(ward_a=mu, ward_b=nu, rank=rank), sendtag=rank,
                                  recvbuf=U_head_head4recv, source=tools.give_rank_minus_minus(ward_a=mu, ward_b=nu, rank=rank), recvtag=tools.give_rank_minus_minus(ward_a=mu, ward_b=nu, rank=rank))
                    comm.Barrier()
                    U_head_head_list[mu][nu] = torch.from_numpy(U_head_head4recv).to(
                        device=U.device)

                    U_head_head4send = U[tools.slice_dim_dim(
                        dims_num=7, ward_a=mu, point_a=0, ward_b=nu, point_b=0)].cpu().contiguous().numpy()
                    U_tail_tail4recv = np.zeros_like(U_head_head4send)
                    comm.Barrier()
                    comm.Sendrecv(sendbuf=U_head_head4send, dest=tools.give_rank_minus_minus(ward_a=mu, ward_b=nu, rank=rank), sendtag=tools.give_rank_minus_minus(ward_a=mu, ward_b=nu, rank=rank),
                                  recvbuf=U_tail_tail4recv, source=tools.give_rank_plus_plus(ward_a=mu, ward_b=nu, rank=rank), recvtag=rank)
                    comm.Barrier()
                    U_tail_tail_list[mu][nu] = torch.from_numpy(U_tail_tail4recv).to(
                        device=U.device)
    # Compute adjoint gauge field (dagger conjugate)
    U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj()
    # Initialize clover term tensor
    clover = torch.zeros(
        (4, 3, 4, 3, U.shape[-4], U.shape[-3], U.shape[-2], U.shape[-1]), dtype=U.dtype, device=U.device)
    # Define directions with corresponding axes and gamma_gamma matrices
    ward_ward_keys = ['xy', 'xz', 'xt', 'yz', 'yt', 'zt']
    # Give clover term for each direction
    for ward_ward_key in ward_ward_keys:
        '''
        \begin{align*}
        F_{\mu \nu} = \frac{1}{a^2*8*i}\gamma_{\mu}\gamma_{\nu}[\\
        & u(x,\mu)u(x+\mu,\nu)u^{\dag}(x+\nu,\mu)u^{\dag}(x,\nu)\\
        &+ u(x,\nu)u^{\dag}(x-\mu+\nu,\mu)u^{\dag}(x-\mu,\nu)u(x-\mu,\mu)\\
        &+ u^{\dag}(x-\mu,\mu)u^{\dag}(x-\mu-\nu,\nu)u(x-\mu-\nu,\mu)u(x-\nu,\nu)\\
        &+ u^{\dag}(x-\nu,\nu)u(x-\nu,\mu)u(x-\nu+\mu,\nu)u^{\dag}(x,\mu)-BEFORE^{\dag}]
        \end{align*}
        '''
        ward_ward = lattice.ward_wards[ward_ward_key]
        F = torch.zeros((3, 3, U.shape[-4], U.shape[-3], U.shape[-2],
                        U.shape[-1]), dtype=U.dtype, device=U.device)
        mu = ward_ward['mu']
        nu = ward_ward['nu']
        # $$ \sigma_{\mu,\nu} &= -i/2*(\gamma_{\mu}\gamma_{\nu} - \gamma_{\nu}\gamma_{\mu}) &= -i/2*2\gamma_{\mu}\gamma_{\nu}\\ $$
        sigma = lattice.gamma_gamma[ward_ward['ward']].to(
            U.device).type(U.dtype)
        if verbose:
            print(
                f"PYQCU::DSLASH::CLOVER:\n Processing {ward_ward_key}-direction (mu={mu},nu={nu})...")
        # Extract gauge field for current direction
        U_mu = U[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x]
        U_nu = U[..., nu, :, :, :, :]  # [c1, c2, t, z, y, x]
        U_dag_mu = U_dag[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x]
        U_dag_nu = U_dag[..., nu, :, :, :, :]  # [c1, c2, t, z, y, x]
        if support_parallel:
            roll_u0 = _torch.roll(U_nu, shifts=-1, dims=mu)
            if grid_size[mu] != 1:
                roll_u0[tools.slice_dim(
                    dims_num=6, ward=mu, point=-1)] = U_tail_list[mu][:, :, nu, :, :, :]
            roll_u1 = _torch.roll(U_dag_mu, shifts=-1, dims=nu)
            if grid_size[nu] != 1:
                roll_u1[tools.slice_dim(
                    dims_num=6, ward=nu, point=-1)] = U_tail_list[nu][:, :, mu, :, :, :].permute(1, 0, 2, 3, 4).conj()
            roll_u2 = _torch.roll(_torch.roll(
                U_dag_mu, shifts=1, dims=mu), shifts=-1, dims=nu)
            if grid_size[mu] != 1:
                roll_u2[tools.slice_dim(dims_num=6, ward=mu, point=0)] = _torch.roll(
                    U_head_list[mu][:, :, mu, :, :, :].permute(1, 0, 2, 3, 4).conj(), -1, nu+(nu < mu))
            if grid_size[nu] != 1:
                roll_u2[tools.slice_dim(dims_num=6, ward=nu, point=-1)] = _torch.roll(
                    U_tail_list[nu][:, :, mu, :, :, :].permute(1, 0, 2, 3, 4).conj(), +1, mu+(mu < nu))
            if grid_size[mu] != 1 and grid_size[nu] != 1:
                roll_u2[tools.slice_dim_dim(
                        dims_num=6, ward_a=mu, ward_b=nu, point_a=0, point_b=-1)] = U_head_tail_list[mu][nu][:, :, mu, :, :].permute(1, 0, 2, 3).conj()
            roll_u3 = _torch.roll(U_dag_nu, shifts=1, dims=mu)
            if grid_size[mu] != 1:
                roll_u3[tools.slice_dim(
                    dims_num=6, ward=mu, point=0)] = U_head_list[mu][:, :, nu, :, :, :].permute(1, 0, 2, 3, 4).conj()
            roll_u4 = _torch.roll(U_mu, shifts=1, dims=mu)
            if grid_size[mu] != 1:
                roll_u4[tools.slice_dim(
                    dims_num=6, ward=mu, point=0)] = U_head_list[mu][:, :, mu, :, :, :]
            roll_u5 = _torch.roll(U_dag_mu, shifts=1, dims=mu)
            if grid_size[mu] != 1:
                roll_u5[tools.slice_dim(
                    dims_num=6, ward=mu, point=0)] = U_head_list[mu][:, :, mu, :, :, :].permute(1, 0, 2, 3, 4).conj()
            roll_u6 = _torch.roll(_torch.roll(
                U_dag_nu, shifts=1, dims=mu), shifts=1, dims=nu)
            if grid_size[mu] != 1:
                roll_u6[tools.slice_dim(dims_num=6, ward=mu, point=0)] = _torch.roll(
                    U_head_list[mu][:, :, nu, :, :, :].permute(1, 0, 2, 3, 4).conj(), +1, nu+(nu < mu))
            if grid_size[nu] != 1:
                roll_u6[tools.slice_dim(dims_num=6, ward=nu, point=0)] = _torch.roll(
                    U_head_list[nu][:, :, nu, :, :, :].permute(1, 0, 2, 3, 4).conj(), +1, mu+(mu < nu))
            if grid_size[mu] != 1 and grid_size[nu] != 1:
                roll_u6[tools.slice_dim_dim(
                        dims_num=6, ward_a=mu, ward_b=nu, point_a=0, point_b=0)] = U_head_head_list[mu][nu][:, :, nu, :, :].permute(1, 0, 2, 3).conj()
            roll_u7 = _torch.roll(_torch.roll(
                U_mu, shifts=1, dims=mu), shifts=1, dims=nu)
            if grid_size[mu] != 1:
                roll_u7[tools.slice_dim(dims_num=6, ward=mu, point=0)] = _torch.roll(
                    U_head_list[mu][:, :, mu, :, :, :], +1, nu+(nu < mu))
            if grid_size[nu] != 1:
                roll_u7[tools.slice_dim(dims_num=6, ward=nu, point=0)] = _torch.roll(
                    U_head_list[nu][:, :, mu, :, :, :], +1, mu+(mu < nu))
            if grid_size[mu] != 1 and grid_size[nu] != 1:
                roll_u7[tools.slice_dim_dim(
                        dims_num=6, ward_a=mu, ward_b=nu, point_a=0, point_b=0)] = U_head_head_list[mu][nu][:, :, mu, :, :]
            roll_u8 = _torch.roll(U_nu, shifts=1, dims=nu)
            if grid_size[nu] != 1:
                roll_u8[tools.slice_dim(
                    dims_num=6, ward=nu, point=0)] = U_head_list[nu][:, :, nu, :, :, :]
            roll_u9 = _torch.roll(U_dag_nu, shifts=1, dims=nu)
            if grid_size[nu] != 1:
                roll_u9[tools.slice_dim(
                    dims_num=6, ward=nu, point=0)] = U_head_list[nu][:, :, nu, :, :, :].permute(1, 0, 2, 3, 4).conj()
            roll_u10 = _torch.roll(U_mu, shifts=1, dims=nu)
            if grid_size[nu] != 1:
                roll_u10[tools.slice_dim(
                    dims_num=6, ward=nu, point=0)] = U_head_list[nu][:, :, mu, :, :, :]
            roll_u11 = _torch.roll(_torch.roll(
                U_nu, shifts=-1, dims=mu), shifts=1, dims=nu)
            if grid_size[mu] != 1:
                roll_u11[tools.slice_dim(dims_num=6, ward=mu, point=-1)] = _torch.roll(
                    U_tail_list[mu][:, :, nu, :, :, :], +1, nu+(nu < mu))
            if grid_size[nu] != 1:
                roll_u11[tools.slice_dim(dims_num=6, ward=nu, point=0)] = _torch.roll(
                    U_head_list[nu][:, :, nu, :, :, :], -1, mu+(mu < nu))
            if grid_size[mu] != 1 and grid_size[nu] != 1:
                roll_u11[tools.slice_dim_dim(
                    dims_num=6, ward_a=mu, ward_b=nu, point_a=-1, point_b=0)] = U_head_tail_list[nu][mu][:, :, nu, :, :]
            temp1 = _torch.einsum('abxyzt,bcxyzt->acxyzt', U_mu, roll_u0)
            temp2 = _torch.einsum('abxyzt,bcxyzt->acxyzt', temp1, roll_u1)
            F += _torch.einsum('abxyzt,bcxyzt->acxyzt', temp2, U_dag_nu)
            temp1 = _torch.einsum('abxyzt,bcxyzt->acxyzt', U_nu, roll_u2)
            temp2 = _torch.einsum('abxyzt,bcxyzt->acxyzt', temp1, roll_u3)
            F += _torch.einsum('abxyzt,bcxyzt->acxyzt', temp2, roll_u4)
            temp1 = _torch.einsum('abxyzt,bcxyzt->acxyzt', roll_u5, roll_u6)
            temp2 = _torch.einsum('abxyzt,bcxyzt->acxyzt', temp1, roll_u7)
            F += _torch.einsum('abxyzt,bcxyzt->acxyzt', temp2, roll_u8)
            temp1 = _torch.einsum('abxyzt,bcxyzt->acxyzt', roll_u9, roll_u10)
            temp2 = _torch.einsum('abxyzt,bcxyzt->acxyzt', temp1, roll_u11)
            F += _torch.einsum('abxyzt,bcxyzt->acxyzt', temp2, U_dag_mu)
        else:
            # $$U_1 &= u(x,\mu)u(x+\mu,\nu)u^{\dag}(x+\nu,\mu)u^{\dag}(x,\nu)                \\$$
            temp1 = _torch.einsum('abxyzt,bcxyzt->acxyzt', U_mu,
                                  _torch.roll(U_nu, shifts=-1, dims=mu))
            temp2 = _torch.einsum('abxyzt,bcxyzt->acxyzt', temp1,
                                  _torch.roll(U_dag_mu, shifts=-1, dims=nu))
            F += _torch.einsum('abxyzt,bcxyzt->acxyzt', temp2, U_dag_nu)
            # $$U_2 &= u(x,\nu)u^{\dag}(x-\mu+\nu,\mu)u^{\dag}(x-\mu,\nu)u(x-\mu,\mu)        \\$$
            temp1 = _torch.einsum('abxyzt,bcxyzt->acxyzt', U_nu,
                                  _torch.roll(_torch.roll(U_dag_mu, shifts=1, dims=mu), shifts=-1, dims=nu))
            temp2 = _torch.einsum('abxyzt,bcxyzt->acxyzt', temp1,
                                  _torch.roll(U_dag_nu, shifts=1, dims=mu))
            F += _torch.einsum('abxyzt,bcxyzt->acxyzt', temp2,
                               _torch.roll(U_mu, shifts=1, dims=mu))
            # $$U_3 &= u^{\dag}(x-\mu,\mu)u^{\dag}(x-\mu-\nu,\nu)u(x-\mu-\nu,\mu)u(x-\nu,\nu)\\$$
            temp1 = _torch.einsum('abxyzt,bcxyzt->acxyzt', _torch.roll(U_dag_mu, shifts=1, dims=mu),
                                  _torch.roll(_torch.roll(U_dag_nu, shifts=1, dims=mu), shifts=1, dims=nu))
            temp2 = _torch.einsum('abxyzt,bcxyzt->acxyzt', temp1,
                                  _torch.roll(_torch.roll(U_mu, shifts=1, dims=mu), shifts=1, dims=nu))
            F += _torch.einsum('abxyzt,bcxyzt->acxyzt', temp2,
                               _torch.roll(U_nu, shifts=1, dims=nu))
            # $$U_4 &= u^{\dag}(x-\nu,\nu)u(x-\nu,\mu)u(x-\nu+\mu,\nu)u^{\dag}(x,\mu)        \\$$
            temp1 = _torch.einsum('abxyzt,bcxyzt->acxyzt', _torch.roll(U_dag_nu, shifts=1, dims=nu),
                                  _torch.roll(U_mu, shifts=1, dims=nu))
            temp2 = _torch.einsum('abxyzt,bcxyzt->acxyzt', temp1,
                                  _torch.roll(_torch.roll(U_nu, shifts=-1, dims=mu), shifts=1, dims=nu))
            F += _torch.einsum('abxyzt,bcxyzt->acxyzt', temp2, U_dag_mu)
        # Give whole F
        F -= F.permute(1, 0, 2, 3, 4, 5).conj()  # -BEFORE^{\dag}
        # Multiply F with sigma
        sigmaF = _torch.einsum(
            'Ss,Ccxyzt->SCscxyzt', sigma, F)
        # Make Clover term
        clover += -0.125/u_0*kappa*sigmaF
        if verbose:
            print(
                f"PYQCU::DSLASH::CLOVER:\n sigmaF term norm: {_torch.norm(sigmaF).item()}")
    if verbose:
        print("PYQCU::DSLASH::CLOVER:\n Clover term complete")
        print(
            f"PYQCU::DSLASH::CLOVER:\n clover norm: {_torch.norm(clover).item()}")
    return clover


def add_I(clover_term: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    _clover_term = clover_term.reshape(12, 12, -1).clone()
    if verbose:
        print('PYQCU::DSLASH::CLOVER:\n Clover is adding I......')
        print(
            f"PYQCU::DSLASH::CLOVER:\n _clover_term.shape:{_clover_term.shape}")
    eye = _torch.eye(12, dtype=_clover_term.dtype,
                     device=_clover_term.device)
    _clover_term += eye.unsqueeze(-1)
    dest = _clover_term.reshape(clover_term.shape)
    if verbose:
        print(f"PYQCU::DSLASH::CLOVER:\n dest.shape:{dest.shape}")
    return dest


def inverse(clover_term: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    _clover_term = clover_term.reshape(12, 12, -1).clone()
    if verbose:
        print('PYQCU::DSLASH::CLOVER:\n Clover is inversing......')
        print(
            f"PYQCU::DSLASH::CLOVER:\n _clover_term.shape:{_clover_term.shape}")
    for i in range(_clover_term.shape[-1]):
        _clover_term[:, :, i] = torch.linalg.inv(_clover_term[:, :, i])
    dest = _clover_term.reshape(clover_term.shape)
    if verbose:
        print(f"dest.shape:{dest.shape}")
    return dest


def give_clover(src: torch.Tensor, clover_term: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    if verbose:
        print('PYQCU::DSLASH::CLOVER:\n Clover is giving......')
        print(f"PYQCU::DSLASH::CLOVER:\n src.shape:{src.shape}")
    dest = _torch.einsum('SCscxyzt,scxyzt->SCxyzt', clover_term, src)
    if verbose:
        print(f"PYQCU::DSLASH::CLOVER:\n dest.shape:{dest.shape}")
    return dest


def give_clover_ee(src_e: torch.Tensor, clover_e: torch.Tensor) -> torch.Tensor:
    return give_clover(src=src_e, clover_term=clover_e)


def give_clover_oo(src_o: torch.Tensor, clover_o: torch.Tensor) -> torch.Tensor:
    return give_clover(src=src_o, clover_term=clover_o)
