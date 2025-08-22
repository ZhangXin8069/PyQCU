from typing import Tuple
import torch
from time import perf_counter
from typing import Tuple, Callable
from pyqcu.ascend import dslash
from pyqcu.ascend.include import *

if_laplacian = True  # just for test.
# if_laplacian = False  # just for test.


def cg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, if_rtol: bool = False,
       verbose: bool = True) -> torch.Tensor:
    """
    Conjugate Gradient (CG) solver for linear systems Ax = b. (Requirement A is a Hermitian matrix).
    Args:
        b: Right-hand side vector (torch.Tensor).
        matvec: Function computing matrix-vector product (A @ x).
        tol: Tolerance for convergence (default: 1e-6).
        max_iter: Maximum iterations (default: 500).
        x0: Initial guess (default: zero vector).
        if_rtol: if use relative tolerance (default: False).
        verbose: Print convergence progress (default: True).
    Returns:
        x: Approximate solution to Ax = b.
    """
    x = x0.clone() if x0 is not None else torch.randn_like(b)
    r = b - matvec(x)
    r_norm = torch.norm(r).item()
    if if_rtol:
        _tol = torch.norm(b).item()*tol
    else:
        _tol = tol
    if verbose:
        print(f"Norm of b:{torch.norm(b).item()}")
        print(f"Norm of r:{r_norm}")
        print(f"Norm of x0:{torch.norm(x).item()}")
    if r_norm < _tol:
        print("x0 is just right!")
        return x.clone()
    p = r.clone()
    v = torch.zeros_like(b)
    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho = torch.vdot(r.flatten(), r.flatten())
    rho_prev = 1.0
    start_time = perf_counter()
    iter_times = []
    for i in range(max_iter):
        iter_start_time = perf_counter()
        v = matvec(p)
        rho_prev = rho
        alpha = rho / torch.vdot(p.flatten(), v.flatten())
        r -= alpha * v
        x += alpha * p
        rho = torch.vdot(r.flatten(), r.flatten())
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


def bicgstab(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, if_rtol: bool = False, verbose: bool = True) -> torch.Tensor:
    """
    BIConjugate Gradient STABilized(BICGSTAB) solver for linear systems Ax = b. (It is not required that A be a Hermitian matrix).
    Args:
        b: Right-hand side vector (torch.Tensor).
        matvec: Function computing matrix-vector product (A @ x).
        tol: Tolerance for convergence (default: 1e-6).
        max_iter: Maximum iterations (default: 500).
        x0: Initial guess (default: zero vector).
        if_rtol: if use relative tolerance (default: False).
        verbose: Print convergence progress (default: True).
    Returns:
        x: Approximate solution to Ax = b.
    """
    x = x0.clone() if x0 is not None else torch.randn_like(b)
    r = b - matvec(x)
    r_norm = torch.norm(r).item()
    if if_rtol:
        _tol = torch.norm(b).item()*tol
    else:
        _tol = tol
    if verbose:
        print(f"Norm of b:{torch.norm(b).item()}")
        print(f"Norm of r:{r_norm}")
        print(f"Norm of x0:{torch.norm(x).item()}")
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
        rho = torch.vdot(r_tilde.flatten(), r.flatten())
        beta = (rho / rho_prev) * (alpha / omega)
        rho_prev = rho
        p = r + beta * (p - omega * v)
        v = matvec(p)
        alpha = rho / torch.vdot(r_tilde.flatten(), v.flatten())
        s = r - alpha * v
        t = matvec(s)
        omega = torch.vdot(t.flatten(), s.flatten()) / \
            torch.vdot(t.flatten(), t.flatten())
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = torch.norm(r).item()
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
    normalize: bool = True, ortho_r: bool = True, ortho_null_vecs: bool = True, verbose: bool = True
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
        verbose: Print progress information (default: True).
    Returns:
        Orthonormal near-null space vectors
    """
    dof = null_vecs.shape[0]  # Number of null space vectors
    null_vecs = torch.randn_like(null_vecs)  # [Eetzyx]
    for i in range(dof):
        if ortho_r:
            # The orthogonalization of r
            for j in range(0, i):
                null_vecs[i] -= torch.vdot(null_vecs[j].flatten(), null_vecs[i].flatten())/torch.vdot(
                    null_vecs[j].flatten(), null_vecs[j].flatten())*null_vecs[j]
        # v=r-A^{-1}Ar
        # tol needs to be bigger...
        null_vecs[i] -= bicgstab(b=matvec(null_vecs[i]),
                                 matvec=matvec, tol=5e-5, verbose=True)
        if ortho_null_vecs:
            # The orthogonalization of null_vecs
            for j in range(0, i):
                null_vecs[i] -= torch.vdot(null_vecs[j].flatten(), null_vecs[i].flatten())/torch.vdot(
                    null_vecs[j].flatten(), null_vecs[j].flatten())*null_vecs[j]
        if normalize:
            null_vecs[i] /= torch.norm(null_vecs[i]).item()
        if verbose:
            print(
                f"(matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]:{(matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]}")
    if verbose:
        print(f"Near-null space check:")
        for i in range(dof):
            Av = matvec(null_vecs[i])
            print(
                f"  Vector {i}: ||A*v/v|| = {torch.norm(Av/null_vecs[i]).item():.6e}")
            print(
                f"  Vector {i}: A*v/v:100 = {(Av/null_vecs[i]).flatten()[:100]}")
            print(
                f"torch.norm(null_vecs[{i}]).item():.6e:{torch.norm(null_vecs[i]).item():.6e}")
            # orthogonalization
            for j in range(0, i+1):
                print(
                    f"torch.vdot(null_vecs[{i}].flatten(), null_vecs[{j}].flatten()):{torch.vdot(null_vecs[i].flatten(), null_vecs[j].flatten())}")
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
    T, Z, Y, X = mg_size
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
    def __init__(self, wilson: dslash.wilson_mg = None, U: torch.Tensor = None):
        self.M_plus_list = []  # tzyx
        self.M_minus_list = []  # tzyx
        self.wilson = wilson if wilson is not None else dslash.wilson_mg(
            verbose=False)
        self.U = U
        if self.wilson != None and self.U != None:
            for ward in range(4):
                if if_laplacian:
                    self.M_plus_list.append(torch.eye(n=12, dtype=self.U.dtype, device=self.U.device).repeat(
                        list(self.U.shape[-4:])+[1, 1]).permute(4, 5, 0, 1, 2, 3))
                    self.M_minus_list.append(torch.eye(n=12, dtype=self.U.dtype, device=self.U.device).repeat(
                        list(self.U.shape[-4:])+[1, 1]).permute(4, 5, 0, 1, 2, 3))
                    print(
                        f"self.M_plus_list[{ward}].shape:{self.M_plus_list[ward].shape}")
                else:
                    self.M_plus_list.append(
                        wilson.give_hopping_plus(ward=ward, U=self.U))
                    self.M_minus_list.append(
                        wilson.give_hopping_minus(ward=ward, U=self.U))

    def matvec_plus(self, ward: int, src: torch.Tensor) -> torch.Tensor:
        return self.wilson.give_wilson_plus(ward=ward, src=src, hopping=self.M_plus_list[ward])

    def matvec_minus(self, ward: int, src: torch.Tensor) -> torch.Tensor:
        return self.wilson.give_wilson_minus(ward=ward, src=src, hopping=self.M_minus_list[ward])

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        dest = torch.zeros_like(src)
        for ward in range(4):
            dest += self.matvec_plus(ward=ward, src=src)
            dest += self.matvec_minus(ward=ward, src=src)
        return dest.clone()


class sitting:
    def __init__(self, clover: dslash.clover = None, clover_term: torch.Tensor = None):
        self.M = torch.zeros([])
        self.clover = clover
        self.clover_term = clover_term
        if self.clover != None and self.clover_term != None:  # remmber to add I
            if if_laplacian:
                self.M = -8 * torch.eye(n=12, dtype=self.clover_term.dtype, device=self.clover_term.device).repeat(
                    list(self.clover_term.shape[-4:])+[1, 1]).permute(4, 5, 0, 1, 2, 3)
                print(f"self.M.shape{self.M.shape}")
            else:
                self.M = self.clover_term.reshape(
                    [12, 12]+list(self.clover_term.shape[-4:])).clone()

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "EeTZYX, eTZYX->ETZYX", self.M, src).clone()


class op:
    def __init__(self, wilson: dslash.wilson_mg = None, U: torch.Tensor = None, clover: dslash.clover = None, clover_term: torch.Tensor = None, fine_hopping: hopping = None, fine_sitting: sitting = None, local_ortho_null_vecs: torch.Tensor = None, verbose: bool = True):
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
            for ward in range(4):  # tzyx
                self.hopping.M_plus_list.append(
                    torch.zeros_like(self.sitting.M))
                self.hopping.M_minus_list.append(
                    torch.zeros_like(self.sitting.M))
            if self.verbose:
                print(
                    f"local_ortho_null_vecs.shape,coarse_dof,coarse_shape,fine_dof,fine_shape:{local_ortho_null_vecs.shape,coarse_dof,coarse_shape,fine_dof,fine_shape}")
            for e in range(coarse_dof):
                for ward in range(4):  # tzyx
                    # give partly sitting.ee and whole hopping.oe
                    _src_c = torch.zeros_like(self.sitting.M[0])
                    _src_c[e][slice_dim(ward=ward, start=0)] = 1.0
                    print(f"_src_c.shape:{_src_c.shape}")
                    print(f"torch.sum(_src_c):{torch.sum(_src_c)}")
                    print(
                        f"_src_c[e][slice_dim(ward=ward, start=0)].shape:{_src_c[e][slice_dim(ward=ward, start=0)].shape}")
                    _src_f = prolong(
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                    _dest_f_plus = fine_hopping.matvec_plus(
                        ward=ward, src=_src_f)
                    _dest_f_minus = fine_hopping.matvec_minus(
                        ward=ward, src=_src_f)
                    _dest_c_plus = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_plus, verbose=self.verbose)
                    _dest_c_minus = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_minus, verbose=self.verbose)
                    print(f"e:{e}")
                    print(f"ward:{ward}")
                    print(
                        f"self.sitting.M[:, e][slice_dim(dim=5,ward=ward+1, start=0)].shape:{self.sitting.M[:, e][slice_dim(dim=5,ward=ward+1, start=0)].shape}")
                    print(
                        f"_dest_c_plus[slice_dim(dim=5, ward=ward+1, start=0)].shape:{_dest_c_plus[slice_dim(dim=5, ward=ward+1, start=0)].shape}")
                    self.sitting.M[:, e][slice_dim(dim=5,
                                                   ward=ward+1, start=0)] += _dest_c_plus[slice_dim(dim=5, ward=ward+1, start=0)].clone()
                    self.sitting.M[:, e][slice_dim(dim=5,
                                                   ward=ward+1, start=0)] += _dest_c_minus[slice_dim(dim=5, ward=ward+1, start=0)].clone()
                    self.hopping.M_plus_list[ward][:, e][slice_dim(dim=5,
                                                                   ward=ward+1, start=1)] = _dest_c_plus[slice_dim(dim=5, ward=ward+1, start=1)].clone()
                    self.hopping.M_minus_list[ward][:, e][slice_dim(dim=5,
                                                                    ward=ward+1, start=1)] = _dest_c_minus[slice_dim(dim=5, ward=ward+1, start=1)].clone()
                    # give partly sitting.oo and whole hopping.eo
                    _src_c = torch.zeros_like(self.sitting.M[0])
                    _src_c[e][slice_dim(ward=ward, start=1)] = 1.0
                    _src_f = prolong(
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                    _dest_f_plus = fine_hopping.matvec_plus(
                        ward=ward, src=_src_f)
                    _dest_f_minus = fine_hopping.matvec_minus(
                        ward=ward, src=_src_f)
                    _dest_c_plus = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_plus, verbose=self.verbose)
                    _dest_c_minus = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_minus, verbose=self.verbose)
                    self.sitting.M[:, e][slice_dim(dim=5,
                                                   ward=ward+1, start=1)] += _dest_c_plus[slice_dim(dim=5, ward=ward+1, start=1)].clone()
                    self.sitting.M[:, e][slice_dim(dim=5,
                                                   ward=ward+1, start=1)] += _dest_c_minus[slice_dim(dim=5, ward=ward+1, start=1)].clone()
                    self.hopping.M_plus_list[ward][:, e][slice_dim(dim=5,
                                                                   ward=ward+1, start=0)] = _dest_c_plus[slice_dim(dim=5, ward=ward+1, start=0)].clone()
                    self.hopping.M_minus_list[ward][:, e][slice_dim(dim=5,
                                                                    ward=ward+1, start=0)] = _dest_c_minus[slice_dim(dim=5, ward=ward+1, start=0)].clone()
                    # give aother partly sitting.ee and sitting.oo
                    _src_c = torch.zeros_like(self.sitting.M[0])
                    _src_c[e] = 1.0
                    _src_f = prolong(
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                    _dest_f = fine_sitting.matvec(src=_src_f)
                    _dest_c = restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f, verbose=self.verbose)
                    self.sitting.M[:, e] += _dest_c.clone()

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        if src.shape[0] == 4 and src.shape[1] == 3:
            return (self.hopping.matvec(src=src.reshape([12]+list(src.shape)[2:]))+self.sitting.matvec(src=src.reshape([12]+list(src.shape)[2:]))).reshape([4, 3]+list(src.shape)[2:])
        else:
            return self.hopping.matvec(src=src)+self.sitting.matvec(src=src)


class mg:
    def __init__(self, b: torch.Tensor,  wilson: dslash.wilson_mg, U: torch.Tensor, clover: dslash.clover, clover_term: torch.Tensor,  min_size: int = 2, max_levels: int = 2, dof_list: Tuple[int, int, int, int] = [12, 12, 12, 12, 24, 24, 24, 24, 48, 48, 24, 8, 8, 8, 4, 12, 12, 12, 8, 4, 2, 4, 4, 24, 12, 12, 12, 4, 4, 4, 4, 4], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, verbose: bool = True):
        self.b = b.reshape([12]+list(b.shape)[2:])  # sc->e
        self.min_size = min_size
        self.max_levels = max_levels
        self.dof_list = dof_list
        print(f"self.dof_list:{self.dof_list}")
        self.tol = tol
        self.max_iter = max_iter
        self.x0 = x0.clone().reshape(
            [12]+list(x0.shape)[2:]) if x0 is not None else torch.randn_like(self.b)  # sc->e
        self.verbose = verbose
        self.op_list = [op(wilson=wilson, U=U,
                           clover=clover, clover_term=clover_term, verbose=self.verbose)]
        # Build grid list
        _Lx = b.shape[-1]
        _Ly = b.shape[-2]
        _Lz = b.shape[-3]
        _Lt = b.shape[-4]
        self.grid_list = []
        self.b_list = [self.b.clone()]
        self.u_list = [self.x0.clone()]
        print(f"Building grid list:")
        while all(_ >= self.min_size for _ in [_Lt, _Lz, _Ly, _Lx]) and len(self.grid_list) < self.max_levels:
            self.grid_list.append([_Lt, _Lz, _Ly, _Lx])
            print(
                f"  Level {len(self.grid_list)-1}: {_Lx}x{_Ly}x{_Lz}x{_Lt}")
            # go with hopping and sitting, must be 2->1
            _Lx //= 2
            _Ly //= 2
            _Lz //= 2
            _Lt //= 2
        print(f"self.grid_list:{self.grid_list}")
        self.num_levels = len(self.grid_list)
        self.dof_list = self.dof_list[:self.num_levels]
        self.nv_list = []  # null_vecs_list
        self.lonv_list = []  # local_ortho_null_vecs_list
        # Build local-orthonormal near-null space vectors
        for i in range(1, len(self.grid_list)):
            _null_vecs = torch.randn(self.dof_list[i], self.dof_list[i-1], self.grid_list[i-1][-4], self.grid_list[i-1][-3], self.grid_list[i-1][-2], self.grid_list[i-1][-1],
                                     dtype=b.dtype, device=b.device)
            _null_vecs = give_null_vecs(
                null_vecs=_null_vecs,
                matvec=self.op_list[i-1].matvec,
                verbose=verbose)
            self.nv_list.append(_null_vecs)
            _local_ortho_null_vecs = local_orthogonalize(
                null_vecs=_null_vecs,
                mg_size=self.grid_list[i], verbose=verbose)
            self.lonv_list.append(_local_ortho_null_vecs)
            self.b_list.append(torch.zeros(
                size=[self.dof_list[i]]+self.grid_list[i], dtype=b.dtype, device=b.device))
            self.u_list.append(torch.zeros(
                size=[self.dof_list[i]]+self.grid_list[i], dtype=b.dtype, device=b.device))
            self.op_list.append(op(fine_hopping=self.op_list[i-1].hopping, fine_sitting=self.op_list[i -
                                                                                                     1].sitting, local_ortho_null_vecs=_local_ortho_null_vecs, verbose=self.verbose))
        self.convergence_history = []

    def cycle(self, level: int = 0) -> torch.Tensor:
        # init start
        b = self.b_list[level].clone()
        x = torch.randn_like(b)
        matvec = self.op_list[level].matvec
        max_iter = self.max_iter
        verbose = self.verbose
        # init end
        r = b - matvec(x)
        r_norm = torch.norm(r).item()
        _tol = torch.norm(b).item()*0.01 if level != self.num_levels - \
            1 else torch.norm(b).item()*0.001
        if verbose:
            print(f"MG-{level}:Norm of b:{torch.norm(b).item()}")
            print(f"MG-{level}:Norm of r:{r_norm}")
            print(f"MG-{level}:Norm of x0:{torch.norm(x).item()}")
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
        for i in range(max_iter):
            iter_start_time = perf_counter()
            rho = torch.vdot(r_tilde.flatten(), r.flatten())
            beta = (rho / rho_prev) * (alpha / omega)
            rho_prev = rho
            p = r + beta * (p - omega * v)
            v = matvec(p)
            alpha = rho / torch.vdot(r_tilde.flatten(), v.flatten())
            s = r - alpha * v
            t = matvec(s)
            omega = torch.vdot(t.flatten(), s.flatten()) / \
                torch.vdot(t.flatten(), t.flatten())
            x = x + alpha * p + omega * s
            r = s - omega * t
            # r = b - matvec(x)
            r_norm = torch.norm(r).item()
            if level == 0 and verbose:
                self.convergence_history.append(r_norm)
            if verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"B-MG-{level}-BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}")
            # cycle start
            if level != self.num_levels-1:
                r_coarse = restrict(
                    local_ortho_null_vecs=self.lonv_list[level], fine_vec=r)
                self.b_list[level+1] = r_coarse.clone()
                e_coarse = self.cycle(level=level+1)
                e_fine = prolong(
                    local_ortho_null_vecs=self.lonv_list[level], coarse_vec=e_coarse)
                x = x + e_fine
                r = b - matvec(x)
            r_norm = torch.norm(r).item()
            if level == 0 and verbose:
                self.convergence_history.append(r_norm)
            # cycle end
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)
            if verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"F-MG-{level}-BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
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
        # self.u_list[level] = torch.zeros_like(x)
        # self.u_list[level] = x.clone()
        return x.clone()

    def solve(self) -> torch.Tensor:
        """
        Main multigrid solver routine.
        Sets up the multigrid list, performs cycle iterations until
        convergence, and returns the solution.
        """
        start_time = perf_counter()
        x = self.cycle()
        total_time = perf_counter() - start_time
        print("\nPerformance Statistics:")
        print(f"Total time: {total_time:.6f} seconds")
        print(f"Final residual: {self.convergence_history[-1]:.2e}")
        return x.reshape([4, 3]+list(x.shape[-4:])).clone()

    def plot(self):
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
