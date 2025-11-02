import torch
import inspect
import functools
import numpy as np
from time import perf_counter
from typing import Tuple, Callable
from pyqcu.ascend.io import *
from pyqcu.ascend.define import *
from pyqcu.ascend.dslash import *


def cg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, if_rtol: bool = False, if_multi: bool = give_if_multi(), verbose: bool = True) -> torch.Tensor:
    _matvec = functools.partial(matvec, if_multi=if_multi) if 'if_multi' in inspect.signature(
        matvec).parameters else matvec
    _torch_vdot = functools.partial(torch_vdot_, if_multi=if_multi)
    _torch_norm = functools.partial(torch_norm_, if_multi=if_multi)
    x = x0.clone() if x0 is not None else torch_randn_like(b)
    r = b - _matvec(x)
    r_norm = _torch_norm(r)
    if if_rtol:
        _tol = _torch_norm(b)*tol
    else:
        _tol = tol
    if verbose:
        print(f"Norm of b:{_torch_norm(b)}")
        print(f"Norm of r:{r_norm}")
        print(f"Norm of x0:{_torch_norm(x)}")
    if r_norm < _tol:
        print("x0 is just right!")
        return x.clone()
    p = r.clone()
    v = torch.zeros_like(b)
    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho = _torch_vdot(r, r)
    rho_prev = 1.0
    start_time = perf_counter()
    iter_times = []
    for i in range(max_iter):
        iter_start_time = perf_counter()
        v = _matvec(p)
        rho_prev = rho
        alpha = rho / _torch_vdot(p, v)
        r -= alpha * v
        x += alpha * p
        rho = _torch_vdot(r, r)
        beta = rho / rho_prev
        p = r + beta * p
        r_norm = torch_sqrt(rho)
        iter_time = perf_counter() - iter_start_time
        iter_times.append(iter_time)
        if verbose:
            # print(f"alpha,beta,rho:{alpha,beta,rho}\n")
            print(
                f"CG-Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
        if r_norm < _tol:
            if verbose:
                print(
                    f"Converged at iteration {i} with residual {r_norm:.6e}")
            break
    else:
        print("  Warning: Maximum iterations reached, may not have converged")
    total_time = perf_counter() - start_time
    avg_iter_time = sum(iter_times) / len(iter_times)
    print("\nPerformance Statistics:")
    print(f"Total iterations: {len(iter_times)}")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"Average time per iteration: {avg_iter_time:.6f} s")
    print(f"Final residual: {r_norm:.2e}")
    return x.clone()


def bicgstab(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, if_rtol: bool = False, if_multi: bool = give_if_multi(), verbose: bool = True) -> torch.Tensor:
    _matvec = functools.partial(matvec, if_multi=if_multi) if 'if_multi' in inspect.signature(
        matvec).parameters else matvec
    _torch_vdot = functools.partial(torch_vdot_, if_multi=if_multi)
    _torch_norm = functools.partial(torch_norm_, if_multi=if_multi)
    x = x0.clone() if x0 is not None else torch_randn_like(b)
    r = b - _matvec(x)
    r_norm = _torch_norm(r)
    if if_rtol:
        _tol = _torch_norm(b)*tol
    else:
        _tol = tol
    if verbose:
        print(f"Norm of b:{_torch_norm(b)}")
        print(f"Norm of r:{r_norm}")
        print(f"Norm of x0:{_torch_norm(x)}")
    if r_norm < _tol:
        print("x0 is just right!")
        return x.clone()
    r_tilde = r.clone()
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    s = torch.zeros_like(b)
    t = torch.zeros_like(b)
    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    start_time = perf_counter()
    iter_times = []
    for i in range(max_iter):
        iter_start_time = perf_counter()
        rho = _torch_vdot(r_tilde, r)
        beta = (rho / rho_prev) * (alpha / omega)
        rho_prev = rho
        p = r + beta * (p - omega * v)
        v = _matvec(p)
        alpha = rho / _torch_vdot(r_tilde, v)
        s = r - alpha * v
        t = _matvec(s)
        omega = _torch_vdot(t, s) / \
            _torch_vdot(t, t)
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = _torch_norm(r)
        iter_time = perf_counter() - iter_start_time
        iter_times.append(iter_time)
        if verbose:
            # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
            print(
                f"BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
        if r_norm < _tol:
            if verbose:
                print(
                    f"Converged at iteration {i} with residual {r_norm:.6e}")
            break
    else:
        print("  Warning: Maximum iterations reached, may not have converged")
    total_time = perf_counter() - start_time
    avg_iter_time = sum(iter_times) / len(iter_times)
    print("\nPerformance Statistics:")
    print(f"Total iterations: {len(iter_times)}")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"Average time per iteration: {avg_iter_time:.6f} s")
    print(f"Final residual: {r_norm:.2e}")
    return x.clone()


def give_null_vecs(
    null_vecs: torch.Tensor,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    normalize: bool = True, ortho_r: bool = False, ortho_null_vecs: bool = False, verbose: bool = True
) -> torch.Tensor:
    dof = null_vecs.shape[0]
    null_vecs = torch_randn_like(null_vecs)  # [Eetzyx]
    for i in range(dof):
        if ortho_r:
            # The orthogonalization of r
            for j in range(0, i):
                null_vecs[i] -= multi_vdot(null_vecs[j], null_vecs[i])/multi_vdot(
                    null_vecs[j], null_vecs[j])*null_vecs[j]
        # v=r-A^{-1}Ar
        # tol needs to be bigger...
        null_vecs[i] -= bicgstab(b=matvec(null_vecs[i]),
                                 matvec=matvec, tol=5e-5, verbose=verbose)
        if ortho_null_vecs:
            # The orthogonalization of null_vecs
            for j in range(0, i):
                null_vecs[i] -= multi_vdot(null_vecs[j], null_vecs[i])/multi_vdot(
                    null_vecs[j], null_vecs[j])*null_vecs[j]
        if normalize:
            null_vecs[i] /= multi_norm(null_vecs[i])
        if verbose:
            print(
                f"(_matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]:{(matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]}")
    if verbose:
        print(f"Near-null space check:")
        for i in range(dof):
            Av = matvec(null_vecs[i])
            print(
                f"  Vector {i}: ||A*v/v|| = {multi_norm(Av/null_vecs[i]):.6e}")
            print(
                f"  Vector {i}: A*v/v:100 = {(Av/null_vecs[i]).flatten()[:100]}")
            print(
                f"_torch_norm(null_vecs[{i}]):.6e:{multi_norm(null_vecs[i]):.6e}")
            # orthogonalization
            for j in range(0, i+1):
                print(
                    f"multi_vdot(null_vecs[{i}], null_vecs[{j}]):{multi_vdot(null_vecs[i], null_vecs[j])}")
    return null_vecs.clone()


def local_orthogonalize(null_vecs: torch.Tensor,
                        coarse_lat_size: Tuple[int, int,
                                               int, int] = (2, 2, 2, 2),
                        normalize: bool = True,
                        verbose: bool = False) -> torch.Tensor:
    assert null_vecs.ndim == 6, "Expected shape [E, e, T*t, Z*z, Y*y, X*x]"
    E, e, Tt, Zz, Yy, Xx = null_vecs.shape
    T, Z, Y, X = coarse_lat_size[::-1]  # [xyzt]
    # sanity checks
    assert Tt % T == 0 and Zz % Z == 0 and Yy % Y == 0 and Xx % X == 0, \
        "Each lattice extent must be divisible by its coarse_lat_size factor."
    t, z, y, x = Tt // T, Zz // Z, Yy // Y, Xx // X
    local_dim = e * t * z * y * x
    if E > local_dim:
        raise ValueError(f"E={E} exceeds local_dim={local_dim}. "
                         f"Cannot produce {E} orthonormal columns in a {local_dim}-dim space.")
    # Reshape to expose coarse/fine structure: [E, e, T, t, Z, z, Y, y, X, x]
    v = null_vecs.reshape(E, e, T, t, Z, z, Y, y, X, x).clone()
    # Move coarse coords to the front (as batch): [T, Z, Y, X, E, e, t, z, y, x]
    v = v.permute(2, 4, 6, 8, 0, 1, 3, 5, 7, 9).contiguous()
    # Collapse to blocks: [n_blocks, E, local_dim]
    n_blocks = T * Z * Y * X
    v = v.view(n_blocks, E, local_dim)
    # Build A = [n_blocks, local_dim, E] (columns = E vectors at a coarse site)
    A = v.transpose(-2, -1)  # [n_blocks, local_dim, E]
    # Batched QR on each block; Q has orthonormal columns in R^{local_dim}
    # Use reduced mode: Q: [n_blocks, local_dim, E], R: [n_blocks, E, E]
    Q, _ = torch_linalg_qr(A, mode='reduced')
    if normalize:
        # Normalize each column vector explicitly
        Q = Q / torch_norm(Q, dim=-2, keepdim=True)
    # Restore lattice structure: [T, Z, Y, X, e, t, z, y, x, E]
    Q = Q.view(T, Z, Y, X, e, t, z, y, x, E)
    # Permute back to [E, e, T, t, Z, z, Y, y, X, x]
    Q = Q.permute(9, 4, 0, 5, 1, 6, 2, 7, 3, 8).contiguous().clone()
    if verbose:
        print(f"[local_orthogonalize] in={tuple(null_vecs.shape)}, coarse_lat_size(T,Z,Y,X)={coarse_lat_size}, "
              f"(t,z,y,x)=({t},{z},{y},{x}), local_dim={local_dim}, n_blocks={n_blocks}")
    return Q.clone()


def restrict(local_ortho_null_vecs: torch.Tensor, fine_vec: torch.Tensor, verbose: bool = True) -> torch.Tensor:
    dtype = fine_vec.dtype
    device = fine_vec.device
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        fine_vec = fine_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _fine_vec = fine_vec.reshape(shape=shape[1:]).clone()
    return torch_einsum(
        "EeTtZzYyXx,eTtZzYyXx->ETZYX", local_ortho_null_vecs.conj(), _fine_vec).clone().to(dtype=dtype, device=device)


def prolong(local_ortho_null_vecs: torch.Tensor, coarse_vec: torch.Tensor, verbose: bool = True) -> torch.Tensor:
    dtype = coarse_vec.dtype
    device = coarse_vec.device
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        coarse_vec = coarse_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _coarse_vec = coarse_vec.reshape(shape=shape[0:1]+shape[-8:][::2]).clone()
    return torch_einsum(
        "EeTtZzYyXx,ETZYX->eTtZzYyXx", local_ortho_null_vecs, _coarse_vec).reshape([shape[1], shape[-8]*shape[-7], shape[-6]*shape[-5], shape[-4]*shape[-3], shape[-2]*shape[-1]]).clone().to(dtype=dtype, device=device)


class hopping:
    def __init__(self, wilson: wilson_mg = None, U: torch.Tensor = None, if_multi: bool = give_if_multi()):
        self.M_plus_list = [torch.zeros([]), torch.zeros(
            []), torch.zeros([]), torch.zeros([])]  # xyzt
        self.M_minus_list = [torch.zeros([]), torch.zeros(
            []), torch.zeros([]), torch.zeros([])]  # xyzt
        self.wilson = wilson if wilson is not None else wilson_mg(
            verbose=False)
        self.U = U
        self.grid_size = give_grid_size()
        self.grid_index = give_grid_index()
        if self.wilson != None and self.U != None:
            for ward in range(4):  # xyzt
                if if_multi and self.grid_size[ward] != 1:
                    comm = MPI.COMM_WORLD
                    rank = comm.Get_rank()
                    U_tail4send = self.U[slice_dim(
                        dim=7, ward=ward, point=-1)].cpu().contiguous().numpy().copy()
                    U_head4recv = np.zeros_like(U_tail4send).copy()
                    rank_plus = give_rank_plus(ward=ward)
                    rank_minus = give_rank_minus(ward=ward)
                    comm.Sendrecv(sendbuf=U_tail4send, dest=rank_plus, sendtag=rank,
                                  recvbuf=U_head4recv, source=rank_minus, recvtag=rank_minus)
                    comm.Barrier()
                    U_head = torch.from_numpy(U_head4recv).to(
                        device=U.device).clone()
                else:
                    U_head = None
                self.M_plus_list[ward] = wilson.give_hopping_plus(
                    ward=ward, U=self.U)
                self.M_minus_list[ward] = wilson.give_hopping_minus(
                    ward=ward, U=self.U, U_head=U_head)

    def matvec_plus(self, ward: int, src: torch.Tensor, if_multi: bool) -> torch.Tensor:
        dtype = src.dtype
        device = src.device
        _dtype = self.M_plus_list[ward].dtype
        _device = self.M_plus_list[ward].device
        if dtype != _dtype or device != _device:
            src = src.to(dtype=_dtype, device=_device)
        if if_multi and self.grid_size[ward] != 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            src_head4send = src[slice_dim(
                dim=5, ward=ward, point=0)].cpu().contiguous().numpy().copy()
            src_tail4recv = np.zeros_like(src_head4send).copy()
            rank_plus = give_rank_plus(ward=ward)
            rank_minus = give_rank_minus(ward=ward)
            comm.Sendrecv(sendbuf=src_head4send, dest=rank_minus, sendtag=rank_minus,
                          recvbuf=src_tail4recv, source=rank_plus, recvtag=rank)
            comm.Barrier()
            src_tail = torch.from_numpy(src_tail4recv).to(
                device=src.device).clone()
        else:
            src_tail = None
        return self.wilson.give_wilson_plus(ward=ward, src=src, hopping=self.M_plus_list[ward], src_tail=src_tail).to(dtype=dtype, device=device)

    def matvec_minus(self, ward: int, src: torch.Tensor, if_multi: bool) -> torch.Tensor:
        dtype = src.dtype
        device = src.device
        _dtype = self.M_minus_list[ward].dtype
        _device = self.M_minus_list[ward].device
        if dtype != _dtype or device != _device:
            src = src.to(dtype=_dtype, device=_device)
        if if_multi and self.grid_size[ward] != 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            src_tail4send = src[slice_dim(
                dim=5, ward=ward, point=-1)].cpu().contiguous().numpy().copy()
            src_head4recv = np.zeros_like(src_tail4send).copy()
            rank_plus = give_rank_plus(ward=ward)
            rank_minus = give_rank_minus(ward=ward)
            comm.Sendrecv(sendbuf=src_tail4send, dest=rank_plus, sendtag=rank,
                          recvbuf=src_head4recv, source=rank_minus, recvtag=rank_minus)
            comm.Barrier()
            src_head = torch.from_numpy(src_head4recv).to(
                device=src.device).clone()
        else:
            src_head = None
        return self.wilson.give_wilson_minus(ward=ward, src=src, hopping=self.M_minus_list[ward], src_head=src_head).to(dtype=dtype, device=device)

    def matvec(self, src: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
        dest = torch.zeros_like(src)
        for ward in range(4):
            dest += self.matvec_plus(ward=ward, src=src, if_multi=if_multi)
            dest += self.matvec_minus(ward=ward, src=src, if_multi=if_multi)
        return dest.clone()


class sitting:
    def __init__(self, clover: clover = None, clover_term: torch.Tensor = None):
        self.M = torch.zeros([])
        self.clover = clover
        self.clover_term = clover_term
        if self.clover != None and self.clover_term != None:  # remmber to add I
            self.M = self.clover.add_I(clover_term=self.clover_term).reshape(
                [12, 12]+list(self.clover_term.shape[-4:])).clone()  # A = I + T

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        dtype = src.dtype
        device = src.device
        _dtype = self.M.dtype
        _device = self.M.device
        if dtype != _dtype or device != _device:
            src = src.to(dtype=_dtype, device=_device)
        return torch_einsum(
            "EeTZYX, eTZYX->ETZYX", self.M, src).clone().to(dtype=dtype, device=device)


class op:
    def __init__(self, wilson: wilson_mg = None, U: torch.Tensor = None, clover: clover = None, clover_term: torch.Tensor = None, fine_hopping: hopping = None, fine_sitting: sitting = None, local_ortho_null_vecs: torch.Tensor = None, if_multi: bool = give_if_multi(), verbose: bool = True):
        self.if_multi = if_multi
        self.hopping = hopping(wilson=wilson, U=U)
        self.sitting = sitting(clover=clover, clover_term=clover_term)
        self.verbose = verbose
        if fine_hopping != None and fine_sitting != None and local_ortho_null_vecs != None:
            shape = local_ortho_null_vecs.shape  # EeTtZzYyXx
            coarse_shape = [shape[-8], shape[-6],
                            shape[-4], shape[-2]]  # TZYX
            coarse_dof = shape[0]  # E
            self.sitting.M = torch.zeros(
                size=[coarse_dof, coarse_dof]+coarse_shape, dtype=local_ortho_null_vecs.dtype, device=local_ortho_null_vecs.device)  # EETZYX
            for ward in range(4):  # xyzt
                self.hopping.M_plus_list[ward] = torch.zeros_like(
                    self.sitting.M)
                self.hopping.M_minus_list[ward] = torch.zeros_like(
                    self.sitting.M)
            for e in range(coarse_dof):
                for ward in range(4):  # xyzt
                    # give partly sitting.ee and whole hopping.oe
                    _src_c = torch.zeros_like(self.sitting.M[0])
                    _src_c[e][slice_dim(ward=ward, start=0)] = 1.0
                    _src_f = prolong(
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                    _dest_f_plus = fine_hopping.matvec_plus(
                        ward=ward, src=_src_f, if_multi=self.if_multi)
                    _dest_f_minus = fine_hopping.matvec_minus(
                        ward=ward, src=_src_f, if_multi=self.if_multi)
                    _dest_c_plus = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_plus, verbose=self.verbose)
                    _dest_c_minus = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_minus, verbose=self.verbose)
                    self.sitting.M[:, e][slice_dim(dim=5,
                                                   ward=ward, start=0)] += _dest_c_plus[slice_dim(dim=5, ward=ward, start=0)].clone()
                    self.sitting.M[:, e][slice_dim(dim=5,
                                                   ward=ward, start=0)] += _dest_c_minus[slice_dim(dim=5, ward=ward, start=0)].clone()
                    self.hopping.M_plus_list[ward][:, e][slice_dim(dim=5,
                                                                   ward=ward, start=1)] = _dest_c_plus[slice_dim(dim=5, ward=ward, start=1)].clone()
                    self.hopping.M_minus_list[ward][:, e][slice_dim(dim=5,
                                                                    ward=ward, start=1)] = _dest_c_minus[slice_dim(dim=5, ward=ward, start=1)].clone()
                    # give partly sitting.oo and whole hopping.eo
                    _src_c = torch.zeros_like(self.sitting.M[0])
                    _src_c[e][slice_dim(ward=ward, start=1)] = 1.0
                    _src_f = prolong(
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                    _dest_f_plus = fine_hopping.matvec_plus(
                        ward=ward, src=_src_f, if_multi=self.if_multi)
                    _dest_f_minus = fine_hopping.matvec_minus(
                        ward=ward, src=_src_f, if_multi=self.if_multi)
                    _dest_c_plus = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_plus, verbose=self.verbose)
                    _dest_c_minus = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_minus, verbose=self.verbose)
                    self.sitting.M[:, e][slice_dim(dim=5,
                                                   ward=ward, start=1)] += _dest_c_plus[slice_dim(dim=5, ward=ward, start=1)].clone()
                    self.sitting.M[:, e][slice_dim(dim=5,
                                                   ward=ward, start=1)] += _dest_c_minus[slice_dim(dim=5, ward=ward, start=1)].clone()
                    self.hopping.M_plus_list[ward][:, e][slice_dim(dim=5,
                                                                   ward=ward, start=0)] = _dest_c_plus[slice_dim(dim=5, ward=ward, start=0)].clone()
                    self.hopping.M_minus_list[ward][:, e][slice_dim(dim=5,
                                                                    ward=ward, start=0)] = _dest_c_minus[slice_dim(dim=5, ward=ward, start=0)].clone()
                # give aother partly sitting.ee and sitting.oo
                _src_c = torch.zeros_like(self.sitting.M[0])
                _src_c[e] = 1.0
                _src_f = prolong(
                    local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                _dest_f = fine_sitting.matvec(src=_src_f)
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f, verbose=self.verbose)
                self.sitting.M[:, e] += _dest_c.clone()

    def matvec(self, src: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
        if src.shape[0] == 4 and src.shape[1] == 3:
            return (self.hopping.matvec(src=src.reshape([12]+list(src.shape)[2:]), if_multi=self.if_multi and if_multi)+self.sitting.matvec(src=src.reshape([12]+list(src.shape)[2:]))).reshape([4, 3]+list(src.shape)[2:])
        else:
            return self.hopping.matvec(src=src, if_multi=self.if_multi and if_multi)+self.sitting.matvec(src=src)


class mg:
    def __init__(self, lat_size: Tuple[int, int, int, int], dtype: torch.dtype = None, device: torch.device = None, dtype_list: Tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype] = None, device_list: Tuple[torch.device, torch.device, torch.device, torch.device] = None, wilson: wilson_mg = None, U: torch.Tensor = None, clover: clover = None, clover_term: torch.Tensor = None,  min_size: int = 2, max_levels: int = 6, mg_grid_size: Tuple[int, int, int, int] = [2, 2, 2, 2], num_convergence_sample: int = 50, dof_list: Tuple[int, int, int, int] = [12, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24], tol: float = 1e-6, max_iter: int = 1000, root: int = 0, verbose: bool = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.lat_size = list(lat_size)
        self.min_size = min_size
        self.max_levels = max_levels
        self.tol = tol
        self.max_iter = max_iter
        self.root = root
        self.verbose = verbose
        if dtype != None:
            self.dtype_list = [dtype]*max_levels
        else:
            self.dtype_list = dtype_list[:max_levels]
        if device != None:
            self.device_list = [device]*max_levels
        else:
            self.device_list = device_list[:max_levels]
        self.dof_list = dof_list
        if self.rank == self.root:
            print(f"self.dof_list:{self.dof_list}")
            print(f"self.dtype_list:{self.dtype_list}")
            print(f"self.device_list:{self.device_list}")
        for device in self.device_list:
            set_device(device=device)
        self.op_list = [op(wilson=wilson, U=U,
                           clover=clover, clover_term=clover_term, verbose=self.verbose)]
        self.b = torch.randn(size=[12]+self.lat_size[::-1],
                             dtype=self.dtype_list[0], device=self.device_list[0])
        self.x0 = torch.randn(
            size=[12]+self.lat_size[::-1], dtype=self.dtype_list[0], device=self.device_list[0])
        self.b_list = [self.b.clone()]
        self.nv_list = []  # null_vecs_list
        self.lonv_list = []  # local_ortho_null_vecs_list
        self.num_convergence_sample = num_convergence_sample
        self.convergence_history = []
        self.convergence_tol = 0
        # Build grid list
        self.lat_size_list = []
        self.mg_grid_size = mg_grid_size
        _lat_size = self.lat_size
        while all(_ >= self.min_size for _ in _lat_size) and len(self.lat_size_list) < self.max_levels:
            self.lat_size_list.append(_lat_size)
            _lat_size = [_lat_size[d] // self.mg_grid_size[d]
                         for d in range(4)]
        self.num_levels = len(self.lat_size_list)
        if self.rank == self.root:
            print(f"self.lat_size_list:{self.lat_size_list}")
        self.dof_list = self.dof_list[:self.num_levels]

    def init(self):
        # Build local-orthonormal near-null space vectors
        comm = MPI.COMM_WORLD
        comm.Barrier()
        for i in range(1, len(self.lat_size_list)):
            _null_vecs = torch.randn(size=[self.dof_list[i], self.dof_list[i-1]] +
                                     self.lat_size_list[i-1][::-1], dtype=self.dtype_list[i-1], device=self.device_list[i-1])
            _null_vecs = give_null_vecs(
                null_vecs=_null_vecs,
                matvec=self.op_list[i-1].matvec,
                verbose=self.verbose).to(dtype=self.dtype_list[i], device=self.device_list[i])
            self.nv_list.append(_null_vecs)
            _local_ortho_null_vecs = local_orthogonalize(
                null_vecs=_null_vecs,
                coarse_lat_size=self.lat_size_list[i],
                verbose=self.verbose)
            self.lonv_list.append(_local_ortho_null_vecs)
            self.b_list.append(torch.zeros(
                size=[self.dof_list[i]]+self.lat_size_list[i][::-1], dtype=self.dtype_list[i], device=self.device_list[i]))
            self.op_list.append(op(fine_hopping=self.op_list[i-1].hopping, fine_sitting=self.op_list[i -
                                1].sitting, local_ortho_null_vecs=_local_ortho_null_vecs,  verbose=self.verbose))
        comm.Barrier()

    def levels_back(self):
        self.convergence_tol = 0
        self.num_levels = len(self.op_list) if len(
            self.op_list) <= self.max_levels else self.max_levels

    def adaptive(self, iter: int = 0):
        if self.convergence_tol > 3:
            self.num_levels = 1
        if (iter+1)*2 < self.num_convergence_sample+1:
            return
        convergence_now = self.convergence_history[-1]
        convergence_sample = self.convergence_history[-(
            self.num_convergence_sample+1):-1]
        count = 0
        for convergence in convergence_sample:
            if convergence < convergence_now:
                count += 1
        if count >= self.num_convergence_sample//2:
            self.convergence_tol += 1

    def cycle(self, level: int = 0) -> torch.Tensor:
        matvec = self.op_list[level].matvec
        b = self.b_list[level].clone()
        x = torch.zeros_like(b)
        r = b - matvec(x)
        r_norm = multi_norm(r)
        _tol = r_norm*0.5 if level != self.num_levels - 1 else r_norm*0.1
        if self.verbose:
            print(f"MG-{level}:Norm of b:{multi_norm(b)}")
            print(f"MG-{level}:Norm of r:{r_norm}")
            print(f"MG-{level}:Norm of x0:{multi_norm(x)}")
        if level == 0:
            self.convergence_history.append(r_norm)
            _tol = self.tol
        if r_norm < _tol:
            print("x0 is just right!")
            return x.clone()
        r_tilde = r.clone()
        p = torch.zeros_like(b)
        v = torch.zeros_like(b)
        s = torch.zeros_like(b)
        t = torch.zeros_like(b)
        rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        start_time = perf_counter()
        iter_times = []
        for i in range(self.max_iter):
            iter_start_time = perf_counter()
            rho = multi_vdot(r_tilde, r)
            beta = (rho / rho_prev) * (alpha / omega)
            rho_prev = rho
            p = r + beta * (p - omega * v)
            v = matvec(p)
            alpha = rho / multi_vdot(r_tilde, v)
            s = r - alpha * v
            t = matvec(s)
            omega = multi_vdot(t, s) / multi_vdot(t, t)
            x = x + alpha * p + omega * s
            r = s - omega * t
            r_norm = multi_norm(r)
            if level == 0:
                self.convergence_history.append(r_norm)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"B-MG-{level}-BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}")
            # cycle start
            if level < self.num_levels-1:
                r_coarse = restrict(
                    local_ortho_null_vecs=self.lonv_list[level], fine_vec=r, verbose=self.verbose)
                self.b_list[level+1] = r_coarse.clone().to(dtype=self.dtype_list[level+1],
                                                           device=self.device_list[level+1])
                e_coarse = self.cycle(level=level+1).to(dtype=self.dtype_list[level],
                                                        device=self.device_list[level])
                e_fine = prolong(
                    local_ortho_null_vecs=self.lonv_list[level], coarse_vec=e_coarse, verbose=self.verbose)
                x = x + e_fine
                r = b - matvec(x)
            r_norm = multi_norm(r)
            if level == 0:
                self.convergence_history.append(r_norm)
                self.adaptive(iter=i)
            # cycle end
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"F-MG-{level}-BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
            if r_norm < _tol:
                if self.verbose:
                    print(
                        f"Converged at iteration {i} with residual {r_norm:.6e}")
                break
        else:
            print("  Warning: Maximum iterations reached, may not have converged")
        total_time = perf_counter() - start_time
        avg_iter_time = sum(iter_times) / len(iter_times)
        print("\nPerformance Statistics:")
        print(f"Total iterations: {len(iter_times)}")
        print(f"Total time: {total_time:.6f} seconds")
        print(f"Average time per iteration: {avg_iter_time:.6f} s")
        print(f"Final residual: {r_norm:.2e}")
        return x.clone()

    def solve(self, b: torch.Tensor = None, x0: torch.Tensor = None) -> torch.Tensor:
        if b != None:
            self.b = b.reshape([12]+list(b.shape)[2:]).clone()  # sc->e
            self.b_list[0] = self.b.clone()
        if x0 != None:
            self.x0 = x0.reshape([12]+list(x0.shape)[2:]).clone()  # sc->e
        self.levels_back()
        start_time = perf_counter()
        x = self.cycle()
        total_time = perf_counter() - start_time
        print("\nPerformance Statistics:")
        print(f"Total time: {total_time:.6f} seconds")
        print(f"Final residual: {self.convergence_history[-1]:.2e}")
        return x.reshape([4, 3]+list(x.shape[-4:])).clone()

    def plot(self, save_path=None):
        if self.rank == self.root:
            import matplotlib.pyplot as plt
            import numpy as np
            try:
                np.Inf = np.inf
            except Exception as e:
                print(f"Error: {e}")
            plt.figure(figsize=(10, 6))
            plt.title(
                f"(self.lat_size_list:{self.lat_size_list})convergence_history(self.dof_list:{self.dof_list})", fontsize=16)
            plt.semilogy(range(1, len(self.convergence_history) + 1),
                         self.convergence_history, 'b-o', markersize=4, linewidth=2)
            plt.xlabel(
                f"Iteration", fontsize=12)
            plt.ylabel('Residual Norm', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path is None:
                save_path = "convergence_history.png"
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close()
