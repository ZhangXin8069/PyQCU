import torch
import functools
from time import perf_counter
from typing import Tuple, Callable
from pyqcu.ascend.io import *
from pyqcu.ascend.define import *
from pyqcu.ascend.dslash import *


def cg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, if_rtol: bool = False, if_multi: bool = give_if_multi(), verbose: bool = True) -> torch.Tensor:
    """
    Conjugate Gradient (CG) solver for linear systems Ax = b. (Requirement A is a Hermitian matrix).
    Args:
        b: Right-hand side vector (torch.Tensor).
        matvec: Function computing matrix-vector product (A @ x).
        tol: Tolerance for convergence (default: 1e-6).
        max_iter: Maximum iterations (default: 500).
        x0: Initial guess (default: zero vector).
        if_rtol: if use relative tolerance (default: False).
        if_multi: if to use multi-device (default: give_if_multi()).
        verbose: Print convergence progress (default: True).
    Returns:
        x: Approximate solution to Ax = b.
    """
    try:
        _matvec = functools.partial(matvec, if_multi=if_multi)
    except Exception as e:
        _matvec = matvec
        print(f"Error: {e}")
    _torch_vdot = functools.partial(torch_vdot, if_multi=if_multi)
    _torch_norm = functools.partial(torch_norm, if_multi=if_multi)
    x = x0.clone() if x0 is not None else torch.randn_like(b)
    r = b - _matvec(x)
    r_norm = _torch_norm(r).item()
    if if_rtol:
        _tol = _torch_norm(b).item()*tol
    else:
        _tol = tol
    if verbose:
        print(f"Norm of b:{_torch_norm(b).item()}")
        print(f"Norm of r:{r_norm}")
        print(f"Norm of x0:{_torch_norm(x).item()}")
    if r_norm < _tol:
        print("x0 is just right!")
        return x.clone()
    p = r.clone()
    v = torch.zeros_like(b)
    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho = _torch_vdot(r.flatten(), r.flatten())
    rho_prev = 1.0
    start_time = perf_counter()
    iter_times = []
    for i in range(max_iter):
        iter_start_time = perf_counter()
        v = _matvec(p)
        rho_prev = rho
        alpha = rho / _torch_vdot(p.flatten(), v.flatten())
        r -= alpha * v
        x += alpha * p
        rho = _torch_vdot(r.flatten(), r.flatten())
        beta = rho / rho_prev
        p = r + beta * p
        r_norm = torch.sqrt(rho)
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
    """
    BIConjugate Gradient STABilized(BICGSTAB) solver for linear systems Ax = b. (It is not required that A be a Hermitian matrix).
    Args:
        b: Right-hand side vector (torch.Tensor).
        matvec: Function computing matrix-vector product (A @ x).
        tol: Tolerance for convergence (default: 1e-6).
        max_iter: Maximum iterations (default: 500).
        x0: Initial guess (default: zero vector).
        if_rtol: if use relative tolerance (default: False).
        if_multi: if to use multi-device (default: give_if_multi()).
        verbose: Print convergence progress (default: True).
    Returns:
        x: Approximate solution to Ax = b.
    """
    try:
        _matvec = functools.partial(matvec, if_multi=if_multi)
    except Exception as e:
        _matvec = matvec
        print(f"Error: {e}")
    _torch_vdot = functools.partial(torch_vdot, if_multi=if_multi)
    _torch_norm = functools.partial(torch_norm, if_multi=if_multi)
    x = x0.clone() if x0 is not None else torch.randn_like(b)
    r = b - _matvec(x)
    r_norm = _torch_norm(r).item()
    if if_rtol:
        _tol = _torch_norm(b).item()*tol
    else:
        _tol = tol
    if verbose:
        print(f"Norm of b:{_torch_norm(b).item()}")
        print(f"Norm of r:{r_norm}")
        print(f"Norm of x0:{_torch_norm(x).item()}")
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
        rho = _torch_vdot(r_tilde.flatten(), r.flatten())
        beta = (rho / rho_prev) * (alpha / omega)
        rho_prev = rho
        p = r + beta * (p - omega * v)
        v = _matvec(p)
        alpha = rho / _torch_vdot(r_tilde.flatten(), v.flatten())
        s = r - alpha * v
        t = _matvec(s)
        omega = _torch_vdot(t.flatten(), s.flatten()) / \
            _torch_vdot(t.flatten(), t.flatten())
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = _torch_norm(r).item()
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
    normalize: bool = True, ortho_r: bool = False, ortho_null_vecs: bool = False, if_multi: bool = give_if_multi(), verbose: bool = True
) -> torch.Tensor:
    """
    Generates orthonormal near-null space vectors for a linear operator.
    This function refines initial random vectors to become approximate null vectors
    (eigenvectors corresponding to near-zero eigenvalues) through iterative refinement
    and orthogonalization.
    Args:
        null_vecs: Initial random vectors [dof, *dims].
        matvec: Function computing matrix-vector product (A @ x).
        normalize: normalize the null_vecs (default: True).
        ortho_r: orthogonalization of r (default: True).
        ortho_null_vecs: orthogonalization of null_vecs (default: True).
        if_multi: if to use multi-device (default: give_if_multi()).
        verbose: Print progress information (default: True).
    Returns:
        Orthonormal near-null space vectors
    """
    dof = null_vecs.shape[0]  # Number of null space vectors
    null_vecs = torch.randn_like(null_vecs)  # [Eetzyx]
    try:
        _matvec = functools.partial(matvec, if_multi=if_multi)
    except Exception as e:
        _matvec = matvec
        print(f"Error: {e}")
    _torch_vdot = functools.partial(torch_vdot, if_multi=if_multi)
    _torch_norm = functools.partial(torch_norm, if_multi=if_multi)
    for i in range(dof):
        if ortho_r:
            # The orthogonalization of r
            for j in range(0, i):
                null_vecs[i] -= _torch_vdot(null_vecs[j].flatten(), null_vecs[i].flatten())/_torch_vdot(
                    null_vecs[j].flatten(), null_vecs[j].flatten())*null_vecs[j]
        # v=r-A^{-1}Ar
        # tol needs to be bigger...
        null_vecs[i] -= bicgstab(b=_matvec(null_vecs[i]),
                                 matvec=_matvec, tol=5e-5, if_multi=if_multi, verbose=verbose)
        if ortho_null_vecs:
            # The orthogonalization of null_vecs
            for j in range(0, i):
                null_vecs[i] -= _torch_vdot(null_vecs[j].flatten(), null_vecs[i].flatten())/_torch_vdot(
                    null_vecs[j].flatten(), null_vecs[j].flatten())*null_vecs[j]
        if normalize:
            null_vecs[i] /= _torch_norm(null_vecs[i]).item()
        if verbose:
            print(
                f"(_matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]:{(_matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]}")
    if verbose:
        print(f"Near-null space check:")
        for i in range(dof):
            Av = _matvec(null_vecs[i])
            print(
                f"  Vector {i}: ||A*v/v|| = {_torch_norm(Av/null_vecs[i]).item():.6e}")
            print(
                f"  Vector {i}: A*v/v:100 = {(Av/null_vecs[i]).flatten()[:100]}")
            print(
                f"_torch_norm(null_vecs[{i}]).item():.6e:{_torch_norm(null_vecs[i]).item():.6e}")
            # orthogonalization
            for j in range(0, i+1):
                print(
                    f"_torch_vdot(null_vecs[{i}].flatten(), null_vecs[{j}].flatten()):{_torch_vdot(null_vecs[i].flatten(), null_vecs[j].flatten())}")
    return null_vecs.clone()


def local_orthogonalize(null_vecs: torch.Tensor,
                        mg_size: Tuple[int, int, int, int] = (2, 2, 2, 2),
                        normalize: bool = True,
                        verbose: bool = False) -> torch.Tensor:
    """
    Orthogonalize near-null space vectors locally.
    Args:
        null_vecs: Initial random vectors [dof, *dims].
        mg_size: cut latt_size to [latt_size[i]/mg_size[i] for i in range(len(latt_size))] (default: [2, 2, 2, 2]).
        normalize: normalize the null_vecs (default: True).
        verbose: Print progress information (default: True).
    Returns:
        local-orthonormal near-null space vectors.
    """
    assert null_vecs.ndim == 6, "Expected shape [E, e, T*t, Z*z, Y*y, X*x]"
    E, e, Tt, Zz, Yy, Xx = null_vecs.shape
    T, Z, Y, X = mg_size[::-1]  # [xyzt]
    # sanity checks
    assert Tt % T == 0 and Zz % Z == 0 and Yy % Y == 0 and Xx % X == 0, \
        "Each lattice extent must be divisible by its mg_size factor."
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
    Q, _ = torch.linalg.qr(A, mode='reduced')
    if normalize:
        # Normalize each column vector explicitly
        Q = Q / torch.norm(Q, dim=-2, keepdim=True)
    # Restore lattice structure: [T, Z, Y, X, e, t, z, y, x, E]
    Q = Q.view(T, Z, Y, X, e, t, z, y, x, E)
    # Permute back to [E, e, T, t, Z, z, Y, y, X, x]
    Q = Q.permute(9, 4, 0, 5, 1, 6, 2, 7, 3, 8).contiguous().clone()
    if verbose:
        print(f"[local_orthogonalize] in={tuple(null_vecs.shape)}, mg_size(T,Z,Y,X)={mg_size}, "
              f"(t,z,y,x)=({t},{z},{y},{x}), local_dim={local_dim}, n_blocks={n_blocks}")
    return Q.clone()


def restrict(local_ortho_null_vecs: torch.Tensor, fine_vec: torch.Tensor, verbose: bool = True) -> torch.Tensor:  # wilson-mg:restrict_f2c conj()
    """
    Restriction operator: fine -> coarse
    Args:
        local_ortho_null_vecs: local-orthogonalized near-null space vectors [dof, *dims].
        fine_vec: vector in fine grid [*dims].
        verbose: Print progress information (default: True).
    Returns:
        vector in coarse grid.
    """
    shape = local_ortho_null_vecs.shape
    coarse_dof = shape[0]
    if verbose:
        print(f"restrict:shape,coarse_dof:{shape,coarse_dof}")
    _fine_vec = fine_vec.reshape(shape=shape[1:]).clone()
    if verbose:
        print("EeTtZzYyXx,eTtZzYyXx->ETZYX")
    return torch.einsum(
        "EeTtZzYyXx,eTtZzYyXx->ETZYX", local_ortho_null_vecs.conj(), _fine_vec).clone()


def prolong(local_ortho_null_vecs: torch.Tensor, coarse_vec: torch.Tensor, verbose: bool = True) -> torch.Tensor:
    """
    Prolongation operator: coarse -> fine
    Args:
        local_ortho_null_vecs: local-orthogonalized near-null space vectors [dof, *dims].
        coarse_vec: vector in coarse grid [*dims].
        verbose: Print progress information (default: True).
    Returns:
        vector in coarse grid.
    """
    shape = local_ortho_null_vecs.shape
    fine_dof = shape[1]
    if verbose:
        print(f"prolong:shape,fine_dof:{shape,fine_dof}")
    _coarse_vec = coarse_vec.reshape(shape=shape[0:1]+shape[-8:][::2]).clone()
    if verbose:
        print("EeTtZzYyXx,ETZYX->eTtZzYyXx")
    return torch.einsum(
        "EeTtZzYyXx,ETZYX->eTtZzYyXx", local_ortho_null_vecs, _coarse_vec).reshape([fine_dof, shape[-8]*shape[-7], shape[-6]*shape[-5], shape[-4]*shape[-3], shape[-2]*shape[-1]]).clone()


class hopping:
    def __init__(self, wilson: wilson_mg = None, U: torch.Tensor = None):
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
                self.M_plus_list[ward] = wilson.give_hopping_plus(
                    ward=ward, U=self.U)
                self.M_minus_list[ward] = wilson.give_hopping_minus(
                    ward=ward, U=self.U)

    def matvec_plus(self, ward: int, src: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
        if if_multi and self.grid_size[ward] != 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            src_head4send = src[slice_dim(
                dim=5, ward=ward, point=0)].cpu().numpy().copy()
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
        return self.wilson.give_wilson_plus(ward=ward, src=src, hopping=self.M_plus_list[ward], src_tail=src_tail)

    def matvec_minus(self, ward: int, src: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
        if if_multi and self.grid_size[ward] != 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            src_tail4send = src[slice_dim(
                dim=5, ward=ward, point=-1)].cpu().numpy().copy()
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
        return self.wilson.give_wilson_minus(ward=ward, src=src, hopping=self.M_minus_list[ward], src_head=src_head)

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
        return torch.einsum(
            "EeTZYX, eTZYX->ETZYX", self.M, src).clone()


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
            fine_shape = [shape[-8]*shape[-7], shape[-6]*shape[-5],
                          shape[-4]*shape[-3], shape[-2]*shape[-1]]  # (Tt)(Zz)(Yy)(Xx)
            coarse_dof = shape[0]  # E
            fine_dof = shape[1]  # e
            self.sitting.M = torch.zeros(
                size=[coarse_dof, coarse_dof]+coarse_shape, dtype=local_ortho_null_vecs.dtype, device=local_ortho_null_vecs.device)  # EETZYX
            for ward in range(4):  # xyzt
                self.hopping.M_plus_list[ward] = torch.zeros_like(
                    self.sitting.M)
                self.hopping.M_minus_list[ward] = torch.zeros_like(
                    self.sitting.M)
            if self.verbose:
                print(
                    f"local_ortho_null_vecs.shape,coarse_dof,coarse_shape,fine_dof,fine_shape:{local_ortho_null_vecs.shape,coarse_dof,coarse_shape,fine_dof,fine_shape}")
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
    def __init__(self, b: torch.Tensor = None,  wilson: wilson_mg = None, U: torch.Tensor = None, clover: clover = None, clover_term: torch.Tensor = None,  min_size: int = 2, max_levels: int = 5, dof_list: Tuple[int, int, int, int] = [12, 24, 24, 24, 24], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, root: int = 0, verbose: bool = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.min_size = min_size
        self.max_levels = max_levels
        self.dof_list = dof_list
        self.tol = tol
        self.max_iter = max_iter
        self.root = root
        self.verbose = verbose
        self.op_list = [op(wilson=wilson, U=U,
                           clover=clover, clover_term=clover_term, if_multi=give_if_multi(), verbose=self.verbose)]
        self.sub_b = None
        self.sub_matvec = None
        self.nv_list = []  # null_vecs_list
        self.lonv_list = []  # local_ortho_null_vecs_list
        self.convergence_history = []
        if b == None:
            self.b = b
        else:
            self.b = b.reshape([12]+list(b.shape)[2:]).clone()  # sc->e
            self.x0 = x0.clone().reshape(
                [12]+list(x0.shape)[2:]) if x0 is not None else torch.randn_like(self.b)  # sc->e
            # Build grid list
            _Lx = b.shape[-1]
            _Ly = b.shape[-2]
            _Lz = b.shape[-3]
            _Lt = b.shape[-4]
            self.grid_list = []
            self.b_list = [self.b.clone()]
            if self.verbose:
                print(f"Building grid list:")
            while all(_ >= self.min_size for _ in [_Lt, _Lz, _Ly, _Lx]) and len(self.grid_list) < self.max_levels:
                self.grid_list.append([_Lx, _Ly, _Lz, _Lt])
                if self.verbose:
                    print(
                        f"  Level {len(self.grid_list)-1}: {_Lx}x{_Ly}x{_Lz}x{_Lt}")
                # go with hopping and sitting, must be 2->1
                _Lx //= 2
                _Ly //= 2
                _Lz //= 2
                _Lt //= 2
            if self.verbose:
                print(f"self.grid_list:{self.grid_list}")
            self.num_levels = len(self.grid_list)
            self.dof_list = self.dof_list[:self.num_levels]

    def init(self):
        # Build local-orthonormal near-null space vectors
        comm = MPI.COMM_WORLD
        comm.Barrier()
        if give_if_multi():
            i = 1
            grid_size = give_grid_size()
            _null_vecs = torch.randn(self.dof_list[i], self.dof_list[i-1], self.grid_list[i-1][-1]//grid_size[-1], self.grid_list[i-1][-2]//grid_size[-2], self.grid_list[i-1][-3]//grid_size[-3], self.grid_list[i-1][-4]//grid_size[-4],
                                     dtype=self.b.dtype, device=self.b.device)
            _null_vecs = give_null_vecs(
                null_vecs=_null_vecs,
                matvec=self.sub_matvec,
                if_multi=True,
                verbose=False)
            full_null_vecs = local2full_tensor(
                local_tensor=_null_vecs, lat_size=self.b.shape[-4:][::-1], device=self.b.device, root=self.root)
            comm.Barrier()
            if self.rank == self.root:
                self.nv_list.append(full_null_vecs)
                _local_ortho_null_vecs = local_orthogonalize(
                    null_vecs=full_null_vecs,
                    mg_size=self.grid_list[i], verbose=True)
                self.lonv_list.append(_local_ortho_null_vecs)
                self.b_list.append(torch.zeros(
                    size=[self.dof_list[i]]+self.grid_list[i][::-1], dtype=self.b.dtype, device=self.b.device))
                self.op_list.append(op(fine_hopping=self.op_list[i-1].hopping, fine_sitting=self.op_list[i -
                                                                                                         1].sitting, local_ortho_null_vecs=_local_ortho_null_vecs, if_multi=False, verbose=self.verbose))
        if self.rank == self.root:
            for i in range(1+give_if_multi(), len(self.grid_list)):
                _null_vecs = torch.randn(self.dof_list[i], self.dof_list[i-1], self.grid_list[i-1][-1], self.grid_list[i-1][-2], self.grid_list[i-1][-3], self.grid_list[i-1][-4],
                                         dtype=self.b.dtype, device=self.b.device)
                _null_vecs = give_null_vecs(
                    null_vecs=_null_vecs,
                    matvec=self.op_list[i-1].matvec,
                    if_multi=False,
                    verbose=self.verbose)
                self.nv_list.append(_null_vecs)
                _local_ortho_null_vecs = local_orthogonalize(
                    null_vecs=_null_vecs,
                    mg_size=self.grid_list[i], verbose=self.verbose)
                self.lonv_list.append(_local_ortho_null_vecs)
                self.b_list.append(torch.zeros(
                    size=[self.dof_list[i]]+self.grid_list[i][::-1], dtype=self.b.dtype, device=self.b.device))
                self.op_list.append(op(fine_hopping=self.op_list[i-1].hopping, fine_sitting=self.op_list[i -
                                                                                                         1].sitting, local_ortho_null_vecs=_local_ortho_null_vecs, if_multi=False, verbose=self.verbose))
        comm.Barrier()

    def cycle(self, level: int = 0) -> torch.Tensor:
        if_multi = True if level == 0 and give_if_multi() else False
        try:
            _matvec = self.sub_matvec if if_multi else functools.partial(
                self.op_list[level].matvec, if_multi=False)
        except Exception as e:
            _matvec = self.op_list[level].matvec
            print(f"Error: {e}")
        _torch_vdot = functools.partial(torch_vdot, if_multi=if_multi)
        _torch_norm = functools.partial(torch_norm, if_multi=if_multi)
        # init start
        b = self.b_list[level].clone()
        x = torch.zeros_like(b)
        # init end
        r = b - _matvec(x)
        r_norm = _torch_norm(r).item()
        _tol = r_norm*0.5 if level != self.num_levels - 1 else r_norm*0.1
        if self.verbose:
            print(f"MG-{level}:Norm of b:{_torch_norm(b).item()}")
            print(f"MG-{level}:Norm of r:{r_norm}")
            print(f"MG-{level}:Norm of x0:{_torch_norm(x).item()}")
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
            rho = _torch_vdot(r_tilde.flatten(), r.flatten())
            beta = (rho / rho_prev) * (alpha / omega)
            rho_prev = rho
            p = r + beta * (p - omega * v)
            v = _matvec(p)
            alpha = rho / _torch_vdot(r_tilde.flatten(), v.flatten())
            s = r - alpha * v
            t = _matvec(s)
            omega = _torch_vdot(t.flatten(), s.flatten()) / \
                _torch_vdot(t.flatten(), t.flatten())
            x = x + alpha * p + omega * s
            r = s - omega * t
            r_norm = _torch_norm(r).item()
            if level == 0:
                self.convergence_history.append(r_norm)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"B-MG-{level}-BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}")
            # cycle start
            if level != self.num_levels-1:
                if if_multi:
                    r = local2full_tensor(
                        local_tensor=r, lat_size=self.grid_list[0], device=b.device, root=self.root)
                if self.rank == self.root:
                    r_coarse = restrict(
                        local_ortho_null_vecs=self.lonv_list[level], fine_vec=r, verbose=self.verbose)
                    self.b_list[level+1] = r_coarse.clone()
                    e_coarse = self.cycle(level=level+1)
                    e_fine = prolong(
                        local_ortho_null_vecs=self.lonv_list[level], coarse_vec=e_coarse, verbose=self.verbose)
                else:
                    e_fine = None
                if if_multi:
                    e_fine = full2local_tensor(
                        full_tensor=e_fine, lat_size=self.grid_list[0], device=b.device, root=self.root)
                x = x + e_fine
                r = b - _matvec(x)
            r_norm = _torch_norm(r).item()
            if level == 0:
                self.convergence_history.append(r_norm)
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
        """
        Main multigrid solver routine.
        Sets up the multigrid list, performs cycle iterations until
        convergence, and returns the solution.
        """
        if b != None:
            self.b = b.reshape([12]+list(b.shape)[2:]).clone()  # sc->e
            self.b_list[0] = self.b.clone()
        if x0 != None:
            self.x0 = x0.reshape([12]+list(x0.shape)[2:]).clone()  # sc->e
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
            np.Inf = np.inf
            plt.figure(figsize=(10, 6))
            plt.title(
                f"(self.grid_list:{self.grid_list})convergence_history(self.dof_list:{self.dof_list})", fontsize=16)
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
