import torch
import torch.nn as nn
import time
from time import perf_counter
from typing import Tuple, Optional, Callable
from pyqcu.ascend import dslash
import matplotlib.pyplot as plt


def cg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 500, x0=None, verbose=True) -> torch.Tensor:
    """
    Conjugate Gradient (CG) solver for linear systems Ax = b. (Requirement A is a Hermitian matrix).
    Args:
        b: Right-hand side vector (torch.Tensor).
        matvec: Function computing matrix-vector product (A @ x).
        tol: Tolerance for convergence (default: 1e-6).
        max_iter: Maximum iterations (default: 500).
        x0: Initial guess (default: zero vector).
        verbose: Print convergence progress (default: True).
    Returns:
        x: Approximate solution to Ax = b.
    """
    x = x0.clone() if x0 is not None else torch.randn_like(b)
    r = b - matvec(x)
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
        if rho.abs() < 1e-30:
            if verbose:
                print("Breakdown: rho ≈ 0")
            break
        beta = rho / rho_prev
        p = r + beta * p
        iter_time = perf_counter() - iter_start_time
        iter_times.append(iter_time)
        if verbose:
            print(f"alpha,beta,rho:{alpha,beta,rho}\n")
            print(
                f"Iteration {i}: Residual = {rho.real:.6e}, Time = {iter_time:.6f} s")
        if rho.real < tol:
            if verbose:
                print(
                    f"Converged at iteration {i} with residual {rho.real:.6e}")
            break
    total_time = perf_counter() - start_time
    avg_iter_time = sum(iter_times) / len(iter_times)
    print("\nPerformance Statistics:")
    print(f"Total time: {total_time:.6f} s")
    print(f"Average time per iteration: {avg_iter_time:.6f} s")
    return x


def bicgstab(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 500, x0=None, verbose=True) -> torch.Tensor:
    """
    BIConjugate Gradient STABilized(BICGSTAB) solver for linear systems Ax = b. (It is not required that A be a Hermitian matrix).
    Args:
        b: Right-hand side vector (torch.Tensor).
        matvec: Function computing matrix-vector product (A @ x).
        tol: Tolerance for convergence (default: 1e-6).
        max_iter: Maximum iterations (default: 500).
        x0: Initial guess (default: zero vector).
        verbose: Print convergence progress (default: True).
    Returns:
        x: Approximate solution to Ax = b.
    """
    x = x0.clone() if x0 is not None else torch.randn_like(b)
    r = b - matvec(x)
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
        if rho.abs() < 1e-30:
            if verbose:
                print("Breakdown: rho ≈ 0")
            break
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
        r_norm2 = torch.norm(r).item()
        iter_time = perf_counter() - iter_start_time
        iter_times.append(iter_time)
        if verbose:
            # print(f"alpha,beta,omega,r_norm2:{alpha,beta,omega,r_norm2}\n")
            print(
                f"Iteration {i}: Residual = {r_norm2:.6e}, Time = {iter_time:.6f} s")
        if r_norm2 < tol:
            if verbose:
                print(
                    f"Converged at iteration {i} with residual {r_norm2:.6e}")
            break
    total_time = perf_counter() - start_time
    avg_iter_time = sum(iter_times) / \
        len(iter_times) if iter_times else 0.0
    if verbose:
        print("\nPerformance Statistics:")
        print(f"Total time: {total_time:.6f} s")
        print(f"Average time per iteration: {avg_iter_time:.6f} s")
    return x


def give_null_vecs(
    null_vecs: torch.Tensor,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    tol: float = 1e-6, max_iter: int = 500, normalize: bool = True, ortho_r: bool = True, ortho_null_vecs: bool = False, verbose: bool = True
) -> torch.Tensor:
    """
    Generates orthonormal near-null space vectors for a linear operator.
    This function refines initial random vectors to become approximate null vectors
    (eigenvectors corresponding to near-zero eigenvalues) through iterative refinement
    and orthogonalization.
    Args:
        null_vecs: Initial random vectors [dof, *dims].
        matvec: Function computing matrix-vector product (A @ x).
        tol: Tolerance for convergence (default: 1e-6).
        max_iter: Maximum iterations (default: 500).
        normalize: normalize the null_vecs (default: True).
        ortho_r: orthogonalization of r (default: True).
        ortho_null_vecs: orthogonalization of null_vecs (default: False).
        verbose: Print progress information (default: True).
        verbose: Print progress information (default: True).
    Returns:
        Orthonormal near-null space vectors
    """
    dof = null_vecs.shape[0]  # Number of null space vectors
    null_vecs = torch.randn_like(null_vecs)
    if ortho_r:
        for i in range(dof):
            # The orthogonalization of r
            if normalize:
                null_vecs[i] /= torch.norm(null_vecs[i]).item()
            for j in range(0, i):
                null_vecs[i] -= torch.vdot(null_vecs[j].flatten(), null_vecs[i].flatten())/torch.vdot(
                    null_vecs[j].flatten(), null_vecs[j].flatten())*null_vecs[j]
            if normalize:
                null_vecs[i] /= torch.norm(null_vecs[i]).item()
    for i in range(dof):
        # v=r-A^{-1}Ar
        null_vecs[i] -= bicgstab(b=matvec(null_vecs[i]), matvec=matvec, tol=tol*1000,
                                 max_iter=max_iter, verbose=verbose)  # tol needs to be bigger...
    if ortho_null_vecs:
        for i in range(dof):
            # The orthogonalization of null_vecs
            if normalize:
                null_vecs[i] /= torch.norm(null_vecs[i]).item()
            for j in range(0, i):
                null_vecs[i] -= torch.vdot(null_vecs[j].flatten(), null_vecs[i].flatten())/torch.vdot(
                    null_vecs[j].flatten(), null_vecs[j].flatten())*null_vecs[j]
            if normalize:
                null_vecs[i] /= torch.norm(null_vecs[i]).item()
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
    return null_vecs


def local_orthogonalize(null_vecs: torch.Tensor,
                        mg_size: Tuple[int, int, int, int] = [2, 2, 2, 2],
                        normalize: bool = True, verbose: bool = True
                        ) -> torch.Tensor:
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
    dof = null_vecs.shape[0]  # Number of null space vectors
    latt_size = list(null_vecs.shape[-4:])
    shape = list(null_vecs.shape[:-4])+[mg_size[0], latt_size[0]//mg_size[0], mg_size[1], latt_size[1] //
                                        mg_size[1], mg_size[2], latt_size[2]//mg_size[2], mg_size[3], latt_size[3]//mg_size[3],]
    if verbose:
        print(f"dof,latt_size,mg_size,shape:{dof,latt_size,mg_size,shape}")
    if not all(latt_size[-i-1] == shape[-2*i-1]*shape[-2*i-2] for i in range(4)):
        print(
            'not all(latt_size[-i-1] == shape[-2*i-1]*shape[-2*i-2] for i in range(4))')
    local_null_vecs = null_vecs.reshape(shape=shape)
    local_ortho_null_vecs = torch.zeros_like(local_null_vecs)
    for X in range(mg_size[-1]):
        for Y in range(mg_size[-2]):
            for Z in range(mg_size[-3]):
                for T in range(mg_size[-4]):
                    _local_null_vecs = local_null_vecs[...,
                                                       T, :, Z, :, Y, :, X, :]
                    for i in range(dof):
                        # The orthogonalization of local_null_vecs
                        if normalize:
                            _local_null_vecs[i] /= torch.norm(
                                _local_null_vecs[i]).item()
                        for j in range(0, i):
                            _local_null_vecs[i] -= torch.vdot(_local_null_vecs[j].flatten(), _local_null_vecs[i].flatten())/torch.vdot(
                                _local_null_vecs[j].flatten(), _local_null_vecs[j].flatten())*_local_null_vecs[j]
                        if normalize:
                            _local_null_vecs[i] /= torch.norm(
                                _local_null_vecs[i]).item()
                    local_ortho_null_vecs[..., T, :, Z,
                                          :, Y, :, X, :] = _local_null_vecs
                    if verbose:
                        for i in range(dof):
                            for j in range(0, i+1):
                                print(
                                    f"torch.vdot(local_ortho_null_vecs[..., {T}, :, {Z}, :, {Y}, :, {X}, :][{i}].flatten(), local_ortho_null_vecs[..., {T}, :, {Z}, :, {Y}, :, {X}, :][{j}].flatten()):{torch.vdot(local_ortho_null_vecs[..., T, :, Z, :, Y, :, X, :][i].flatten(), local_ortho_null_vecs[..., T, :, Z, :, Y, :, X, :][j].flatten())}")
    return local_ortho_null_vecs


def restrict(local_ortho_null_vecs: torch.Tensor, fine_vec: torch.Tensor, verbose: bool = True) -> torch.Tensor:
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
    dof = shape[0]
    if verbose:
        print(f"shape,dof:{shape,dof}")
    if fine_vec.shape != shape[1:]:
        print('fine_vec.shape != shape[1:]!!!')
    _fine_vec = fine_vec.reshape(shape=shape[1:]).clone()
    if len(shape) == 10:
        print("EeTtZzYyXx,eTtZzYyXx->ETZYX")
        return torch.einsum(
            "EeTtZzYyXx,eTtZzYyXx->ETZYX", local_ortho_null_vecs, _fine_vec)
    elif len(shape) == 11:
        print("EscTtZzYyXx,scTtZzYyXx->ETZYX")
        return torch.einsum(
            "EscTtZzYyXx,scTtZzYyXx->ETZYX", local_ortho_null_vecs, _fine_vec)
    else:
        print('len(shape) != 10 or 11!!!')
        raise ValueError


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
    dof = shape[0]
    if verbose:
        print(f"shape,dof:{shape,dof}")
    if coarse_vec.shape != shape[1:]:
        print('fine_vec.shape != shape[1:]!!!')
        print(f"shape[0:1]+shape[-8:][::2]:{shape[0:1]+shape[-8:][::2]}")
    _coarse_vec = coarse_vec.reshape(shape=shape[0:1]+shape[-8:][::2]).clone()
    if len(shape) == 10:
        print("EeTtZzYyXx, ETZYX->eTtZzYyXx")
        return torch.einsum(
            "EeTtZzYyXx, ETZYX->eTtZzYyXx", local_ortho_null_vecs.conj(), _coarse_vec)
    elif len(shape) == 11:
        print("EscTtZzYyXx, ETZYX->scTtZzYyXx")
        return torch.einsum(
            "EscTtZzYyXx, ETZYX->scTtZzYyXx", local_ortho_null_vecs.conj(), _coarse_vec)
    else:
        print('len(shape) != 10 or 11!!!')
        raise ValueError


class hopping:
    def __init__(self, wilson: dslash.wilson_parity = None, U_eo: torch.Tensor = None):
        self.M_eo = torch.zeros([])
        self.M_oe = torch.zeros([])
        self.wilson = wilson
        self.U_eo = U_eo

    def matvec_eo(self, src_o: torch.Tensor) -> torch.Tensor:
        if self.wilson != None:
            return self.wilson.give_wilson_eo(
                src_o=src_o, U_eo=self.U_eo)
        else:
            return torch.einsum(
                "EeTZYX, eTZYX->ETZYX", self.M_eo, src_o)

    def matvec_oe(self, src_e: torch.Tensor) -> torch.Tensor:
        if self.wilson != None:
            return self.wilson.give_wilson_oe(
                src_e=src_e, U_eo=self.U_eo)
        else:
            return torch.einsum(
                "EeTZYX, eTZYX->ETZYX", self.M_oe, src_e)


class sitting:
    def __init__(self, clover: dslash.clover_parity = None, clover_eo: torch.Tensor = None):
        self.M_ee = torch.zeros([])
        self.M_oo = torch.zeros([])
        self.clover = clover
        self.clover_eo = clover_eo

    def matvec_ee(self, src_e: torch.Tensor) -> torch.Tensor:
        if self.clover != None:  # remmber to add I
            return self.clover.give_clover_ee(
                src_e=src_e, clover_eo=self.clover_eo)
        else:
            return torch.einsum(
                "EeTZYX, eTZYX->ETZYX", self.M_ee, src_e)

    def matvec_oo(self, src_o: torch.Tensor) -> torch.Tensor:
        if self.clover != None:  # remmber to add I
            return self.clover.give_clover_oo(
                src_o=src_o, clover_eo=self.clover_eo)
        else:
            return torch.einsum(
                "EeTZYX, eTZYX->ETZYX", self.M_oo, src_o)


class _mg:
    def __init__(self, wilson: dslash.wilson_parity = None, U_eo: torch.Tensor = None, clover: dslash.clover_parity = None, clover_eo: torch.Tensor = None, fine_hopping: hopping = None, fine_sitting: sitting = None, local_ortho_null_vecs: torch.Tensor = None, verbose=True):
        self.hopping = hopping(wilson=wilson, U_eo=U_eo)
        self.sitting = sitting(clover=clover, clover_eo=clover_eo)
        if fine_hopping != None and fine_sitting != None and local_ortho_null_vecs != None:
            shape = local_ortho_null_vecs.shape  # EeTtZzYyXx
            dtype = local_ortho_null_vecs.dtype
            device = local_ortho_null_vecs.device
            dof = shape[0]
            self.hopping.M_eo = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # EETZYX
            self.hopping.M_oe = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # EETZYX
            self.sitting.M_ee = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # EETZYX
            self.sitting.M_oo = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # EETZYX
            src_c = torch.zeros(
                size=[dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # ETZYX
            src_c_I = torch.ones(
                size=[shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # TZYX
            dest_f = torch.zeros_like(local_ortho_null_vecs[0])  # eTtZzYyXx
            dest_f_eo = dslash.xxxtzyx2pxxxtzyx(input_array=dest_f.clone())
            for e in range(dof):
                _src_c = src_c.clone()
                _src_c[e] = src_c_I.clone()
                _src_f = prolong(
                    local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=verbose)
                _src_f_eo = dslash.xxxtzyx2pxxxtzyx(input_array=_src_f)
                # give partly sitting.ee and whole hopping.oe
                _dest_f_eo = dest_f_eo.clone()
                _dest_f_eo[0] = fine_hopping.matvec_eo(src=_src_f_eo[1])
                _dest_f = dslash.pxxxtzyx2xxxtzyx(input_array=_dest_f_eo)
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f)
                _dest_c_eo = dslash.xxxtzyx2pxxxtzyx(input_array=_dest_c)
                self.sitting.M_ee[:, e, ...] = _dest_c_eo[0]
                self.hopping.M_oe[:, e, ...] = _dest_c_eo[1]
                # give partly sitting.oo and whole hopping.eo
                _dest_f_eo = dest_f_eo.clone()
                _dest_f_eo[1] = fine_hopping.matvec_oe(src=_src_f_eo[0])
                _dest_f = dslash.pxxxtzyx2xxxtzyx(input_array=_dest_f_eo)
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f)
                _dest_c_eo = dslash.xxxtzyx2pxxxtzyx(input_array=_dest_c)
                self.sitting.M_oo[:, e, ...] = _dest_c_eo[1]
                self.hopping.M_eo[:, e, ...] = _dest_c_eo[0]
                # give aother partly sitting.ee and sitting.oo
                dest_f_eo = dest_f_eo.clone()
                _dest_f_eo[0] = fine_sitting.matvec_ee(src=_src_f_eo[0])
                _dest_f_eo[1] = fine_sitting.matvec_oo(src=_src_f_eo[1])
                _dest_f = dslash.pxxxtzyx2xxxtzyx(input_array=_dest_f_eo)
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f)
                _dest_c_eo = dslash.xxxtzyx2pxxxtzyx(input_array=_dest_c)
                self.sitting.M_ee[:, e, ...] += _dest_c_eo[0]
                self.sitting.M_oo[:, e, ...] += _dest_c_eo[1]

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        src_eo = dslash.xxxtzyx2pxxxtzyx(input_array=src)
        dest_eo = torch.zeros_like(src_eo)
        dest_eo[0] = self.hopping.matvec_eo(
            src_o=src_eo[1])+self.sitting.matvec_ee(src_e=src_eo[0])
        dest_eo[1] = self.hopping.matvec_oe(
            src_e=src_eo[0])+self.sitting.matvec_oo(src_e=src_eo[1])
        return dslash.pxxxtzyx2xxxtzyx(input_array=dest_eo).clone()


class mg:
    def __init__(self, b: torch.Tensor,  wilson: dslash.wilson_parity, U_eo: torch.Tensor, clover: dslash.clover_parity, clover_eo: torch.Tensor,  min_size: int = 2, max_levels: int = 5, tol: float = 1e-6, max_iter: int = 500, x0=None, verbose=True):
        self.b = b
        self._mg_list = [_mg(wilson=wilson, U_eo=U_eo,
                             clover=clover, clover_eo=clover_eo, verbose=verbose)]
        self.min_size = min_size
        self.max_levels = max_levels
        self.tol = tol
        self.max_iter = max_iter
        self.x0 = x0.clone() if x0 is not None else torch.randn_like(b)
        self.verbose = verbose
        # Build grid hierarchy
        _Lx = b.shape[-1]
        _Ly = b.shape[-2]
        _Lz = b.shape[-3]
        _Lt = b.shape[-4]
        self.grid_params = []
        print(f"Building grid hierarchy:")
        while all([_Lx, _Ly, _Lz, _Lt] >= self.min_size) and len(self.grid_params) < self.max_levels:
            self.grid_params.append([_Lx, _Ly, _Lz, _Lt])
            print(
                f"  Level {len(self.grid_params)-1}: {_Lx}x{_Ly}x{_Lz}x{_Lt}")
            _Lx = max(self.min_size, _Lx // 2)
            _Ly = max(self.min_size, _Ly // 2)
            _Lz = max(self.min_size, _Lz // 2)
            _Lt = max(self.min_size, _Lt // 2)

class EllipticPartialDifferentialEquations:
    """
    Elliptic Partial Differential Equation discretization class for 3D problems.

    Solves equations of the form: -α∇²u + βu = f on the unit cube [0,1]³
    using finite difference discretization with Dirichlet boundary conditions.

    Args:
        nx, ny, nz: Grid points in x, y, z directions (default: 32, 32, 12)
        dtype: Data type for computations (default: torch.complex128)
        alpha: Diffusion coefficient (default: 1.0)
        beta: Reaction coefficient (default: 1j)
        device: Device for computations (default: 'cpu')
    """

    def __init__(self, nx: int = 32, ny: int = 32, nz: int = 12,
                 dtype: torch.dtype = torch.complex128, alpha: float = 1.0,
                 beta: complex = 1j, device: str = 'cpu'):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.alpha = alpha
        self.beta = beta
        self.dtype = dtype
        self.device = torch.device(device)
        self.n = nx * ny * nz

        # Grid spacing in each direction
        self.hx = 1.0 / (nx + 1)
        self.hy = 1.0 / (ny + 1)
        self.hz = 1.0 / (nz + 1)

        # Main diagonal coefficient: 2α(1/h²ₓ + 1/h²ᵧ + 1/h²ᵤ) + β
        self.main_diag = (2 * alpha * (1/self.hx**2 + 1/self.hy**2 + 1/self.hz**2) + beta) * \
            torch.ones(self.n, dtype=dtype, device=self.device)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """
        Matrix-vector multiplication for the discretized elliptic operator.

        Applies the finite difference stencil to compute A*v where A represents
        the discretized elliptic operator -α∇² + β.

        Args:
            v: Input vector of size nx*ny*nz

        Returns:
            result: A*v as flattened tensor
        """
        result = torch.zeros_like(v)
        nx, ny, nz = self.nx, self.ny, self.nz

        # Apply main diagonal
        result[:] = self.main_diag * v

        # Reshape to 3D for easier indexing
        v_3d = v.reshape((nz, ny, nx))
        result_3d = result.reshape((nz, ny, nx))

        # Apply 7-point finite difference stencil
        for k in range(nz):
            for i in range(ny):
                for j in range(nx):
                    # X-direction neighbors
                    if j > 0:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hx**2) * v_3d[k, i, j-1]
                    if j < nx - 1:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hx**2) * v_3d[k, i, j+1]

                    # Y-direction neighbors
                    if i > 0:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hy**2) * v_3d[k, i-1, j]
                    if i < ny - 1:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hy**2) * v_3d[k, i+1, j]

                    # Z-direction neighbors
                    if k > 0:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hz**2) * v_3d[k-1, i, j]
                    if k < nz - 1:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hz**2) * v_3d[k+1, i, j]

        return result_3d.flatten()

    def give_b(self, func_type: str = 'sine') -> torch.Tensor:
        """
        Generate right-hand side vector for different test problems.

        Args:
            func_type: Type of test function ('sine', 'exponential', or 'constant')

        Returns:
            b: Right-hand side vector as flattened tensor
        """
        # Create coordinate grids
        x = torch.linspace(self.hx, 1-self.hx, self.nx,
                           dtype=torch.float64, device=self.device)
        y = torch.linspace(self.hy, 1-self.hy, self.ny,
                           dtype=torch.float64, device=self.device)
        z = torch.linspace(self.hz, 1-self.hz, self.nz,
                           dtype=torch.float64, device=self.device)

        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        if func_type == 'sine':
            # Trigonometric test function with complex coefficient
            f = torch.sin(2*torch.pi*X) * torch.sin(2*torch.pi*Y) * \
                torch.sin(2*torch.pi*Z) * (1 + 1j)
        elif func_type == 'exponential':
            # Complex exponential test function
            f = torch.exp(X + 1j*Y + 1j*Z)
        else:
            # Constant function
            f = torch.ones((self.nx, self.ny, self.nz),
                           dtype=torch.complex128, device=self.device)

        return f.flatten().to(self.dtype)


class GMRESSmoother:
    """
    GMRES-based smoother for multigrid methods.

    Implements the Generalized Minimal Residual method with restarts
    as a smoother for the multigrid hierarchy.

    Args:
        max_krylov: Maximum Krylov subspace dimension (default: 5)
        max_restarts: Maximum number of restarts (default: 1)
        tol: Relative tolerance for convergence (default: 0.1)
    """

    def __init__(self, max_krylov: int = 5, max_restarts: int = 1, tol: float = 0.1):
        self.max_krylov = max_krylov
        self.max_restarts = max_restarts
        self.tol = tol

    def smooth(self, op: EllipticPartialDifferentialEquations,
               b: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Apply GMRES smoothing to the linear system op*x = b.

        Args:
            op: Linear operator (EllipticPartialDifferentialEquations instance)
            b: Right-hand side vector
            x0: Initial guess

        Returns:
            x: Smoothed solution
        """
        x = x0.clone()
        r = b - op.matvec(x0)
        r_norm = torch.norm(r).item()

        if r_norm < 1e-12:
            return x0

        # GMRES with restarts
        for _ in range(self.max_restarts):
            Q, H = self._arnoldi(op, r, r_norm)

            # Set up least squares problem
            e1 = torch.zeros(H.shape[1] + 1, dtype=b.dtype, device=b.device)
            e1[0] = r_norm

            # Solve least squares problem
            y = self._solve_least_squares(H, e1)

            # Update solution
            dx = Q[:, :-1] @ y
            x = x + dx

            # Check convergence
            new_r = b - op.matvec(x)
            new_r_norm = torch.norm(new_r).item()

            if new_r_norm < self.tol * r_norm:
                break

            r = new_r
            r_norm = new_r_norm

        return x

    def _arnoldi(self, op: EllipticPartialDifferentialEquations,
                 r0: torch.Tensor, r_norm: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arnoldi process for building orthonormal Krylov subspace.

        Args:
            op: Linear operator
            r0: Initial residual vector
            r_norm: Norm of initial residual

        Returns:
            Q: Orthonormal basis matrix
            H: Upper Hessenberg matrix
        """
        m = self.max_krylov
        n = len(r0)

        Q = torch.zeros((n, m+1), dtype=r0.dtype, device=r0.device)
        H = torch.zeros((m+1, m), dtype=r0.dtype, device=r0.device)

        Q[:, 0] = r0 / r_norm

        for j in range(m):
            w = op.matvec(Q[:, j])

            # Modified Gram-Schmidt orthogonalization
            for i in range(j+1):
                H[i, j] = torch.vdot(Q[:, i], w)
                w = w - H[i, j] * Q[:, i]

            h_norm = torch.norm(w).item()
            H[j+1, j] = h_norm

            if h_norm < 1e-12:
                return Q[:, :j+1], H[:j+1, :j]

            if j < m:
                Q[:, j+1] = w / h_norm

        return Q, H

    def _solve_least_squares(self, H: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
        """
        Solve the least squares problem in GMRES.

        Args:
            H: Upper Hessenberg matrix
            e1: Right-hand side vector

        Returns:
            y: Solution to least squares problem
        """
        m = H.shape[1]
        h_height = H.shape[0]

        # Augment matrix for QR factorization
        R = torch.zeros((h_height, m+1), dtype=H.dtype, device=H.device)
        R[:, :m] = H
        R[:, m] = e1[:h_height]

        # QR factorization and solve
        Q, R_qr = torch.linalg.qr(R, mode='complete')
        y = torch.linalg.solve(R_qr[:m, :m], Q[:m, :m].conj().T @ e1[:m])

        return y


class GeometricMultigrid:
    """
    Geometric Multigrid solver for 3D elliptic PDEs.

    Implements a V-cycle multigrid method with GMRES smoothing and BiCGSTAB
    coarse grid solver for complex-valued elliptic problems.

    Args:
        nx, ny, nz: Fine grid dimensions (default: 32, 32, 12)
        op: Operator class (default: EllipticPartialDifferentialEquations)
        min_size: Minimum grid size for coarsening (default: 4)
        max_levels: Maximum multigrid levels (default: 5)
        tolerance: Convergence tolerance (default: 1e-8)
        max_iterations: Maximum V-cycle iterations (default: 10000)
        dtype: Data type (default: torch.complex128)
        device: Computing device (default: 'cpu')
    """

    def __init__(self, nx: int = 32, ny: int = 32, nz: int = 12,
                 op=EllipticPartialDifferentialEquations, min_size: int = 4,
                 max_levels: int = 5, tolerance: float = 1e-8,
                 max_iterations: int = 10000, dtype: torch.dtype = torch.complex128,
                 device: str = 'cpu'):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.op = op
        self.min_size = min_size
        self.max_levels = max_levels
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.dtype = dtype
        self.device = torch.device(device)
        self.convergence_history = []
        self.level_info = []

        # Initialize GMRES smoother
        self.gmres_smoother = GMRESSmoother(
            max_krylov=5, max_restarts=1, tol=0.1)

    def restrict(self, u_fine: torch.Tensor, nx_fine: int, ny_fine: int, nz_fine: int) -> torch.Tensor:
        """
        Restriction operator: transfer from fine to coarse grid.

        Uses full-weighting restriction with 3x3 stencil in x-y plane,
        no coarsening in z-direction.

        Args:
            u_fine: Fine grid solution
            nx_fine, ny_fine, nz_fine: Fine grid dimensions

        Returns:
            u_coarse: Coarse grid solution
        """
        if nx_fine < 2 or ny_fine < 2:
            return u_fine

        nx_coarse = max(2, nx_fine // 2)
        ny_coarse = max(2, ny_fine // 2)
        nz_coarse = nz_fine

        u_fine_3d = u_fine.reshape((nz_fine, ny_fine, nx_fine))
        u_coarse_3d = torch.zeros((nz_coarse, ny_coarse, nx_coarse),
                                  dtype=self.dtype, device=self.device)

        # Apply full-weighting restriction slice by slice
        for k in range(nz_fine):
            u_fine_slice = u_fine_3d[k, :, :]
            u_coarse_slice = torch.zeros((ny_coarse, nx_coarse),
                                         dtype=self.dtype, device=self.device)

            for i in range(ny_coarse):
                for j in range(nx_coarse):
                    ii, jj = 2*i, 2*j
                    weight_sum = 0
                    value_sum = 0

                    # 3x3 stencil with weights: 1/16, 1/8, 1/4 for corners, edges, center
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = ii + di, jj + dj
                            if 0 <= ni < ny_fine and 0 <= nj < nx_fine:
                                if di == 0 and dj == 0:
                                    weight = 1/4  # Center point
                                elif di == 0 or dj == 0:
                                    weight = 1/8  # Edge points
                                else:
                                    weight = 1/16  # Corner points
                                weight_sum += weight
                                value_sum += weight * u_fine_slice[ni, nj]

                    u_coarse_slice[i, j] = value_sum / \
                        weight_sum if weight_sum > 0 else 0

            u_coarse_3d[k, :, :] = u_coarse_slice

        return u_coarse_3d.flatten()

    def prolongate(self, u_coarse: torch.Tensor, nx_fine: int, ny_fine: int, nz_fine: int) -> torch.Tensor:
        """
        Prolongation operator: transfer from coarse to fine grid.

        Uses bilinear interpolation in x-y plane, direct injection in z-direction.

        Args:
            u_coarse: Coarse grid solution
            nx_fine, ny_fine, nz_fine: Fine grid dimensions

        Returns:
            u_fine: Fine grid solution
        """
        nx_coarse = nx_fine // 2
        ny_coarse = ny_fine // 2
        nz_coarse = nz_fine

        u_coarse_3d = u_coarse.reshape((nz_coarse, ny_coarse, nx_coarse))
        u_fine_3d = torch.zeros((nz_fine, ny_fine, nx_fine),
                                dtype=self.dtype, device=self.device)

        # Apply bilinear interpolation slice by slice
        for k in range(nz_fine):
            u_coarse_slice = u_coarse_3d[k, :, :]
            u_fine_slice = torch.zeros((ny_fine, nx_fine),
                                       dtype=self.dtype, device=self.device)

            for i in range(ny_fine):
                for j in range(nx_fine):
                    # Continuous coordinates in coarse grid
                    i_c = i / 2.0
                    j_c = j / 2.0

                    # Integer coordinates and interpolation weights
                    i0, j0 = int(i_c), int(j_c)
                    i1 = min(i0 + 1, ny_coarse - 1)
                    j1 = min(j0 + 1, nx_coarse - 1)

                    wx = i_c - i0
                    wy = j_c - j0

                    # Bilinear interpolation
                    u_fine_slice[i, j] = (1 - wx) * (1 - wy) * u_coarse_slice[i0, j0] + \
                        (1 - wx) * wy * u_coarse_slice[i0, j1] + \
                        wx * (1 - wy) * u_coarse_slice[i1, j0] + \
                        wx * wy * u_coarse_slice[i1, j1]

            u_fine_3d[k, :, :] = u_fine_slice

        return u_fine_3d.flatten()

    def smooth(self, op: EllipticPartialDifferentialEquations,
               b: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Apply smoothing operation using GMRES.

        Args:
            op: Linear operator
            b: Right-hand side vector
            u: Current solution estimate

        Returns:
            u_smooth: Smoothed solution
        """
        return self.gmres_smoother.smooth(op, b, u)

    def compute_residual(self, op: EllipticPartialDifferentialEquations,
                         b: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute residual r = b - A*u.

        Args:
            op: Linear operator
            b: Right-hand side vector
            u: Current solution estimate

        Returns:
            r: Residual vector
        """
        return b - op.matvec(u)

    def bicgstab_solver(self, op: EllipticPartialDifferentialEquations,
                        b: torch.Tensor, x0: Optional[torch.Tensor] = None,
                        tol: float = 1e-10, maxiter: int = 1000,
                        verbose: bool = False) -> Tuple[torch.Tensor, int]:
        """
        BiCGSTAB solver for coarse grid problems.

        Biconjugate Gradient Stabilized method for solving linear systems Ax = b.
        Adapted from the reference implementation with PyTorch operations.

        Args:
            op: Linear operator
            b: Right-hand side vector
            x0: Initial guess (default: zero vector)
            tol: Convergence tolerance (default: 1e-10)
            maxiter: Maximum iterations (default: 1000)
            verbose: Print convergence info (default: False)

        Returns:
            x: Solution vector
            info: Convergence flag (0: converged, 1: not converged)
        """
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        # Initialize BiCGSTAB vectors
        r = b - op.matvec(x)
        r_tilde = r.clone()
        p = torch.zeros_like(b)
        v = torch.zeros_like(b)
        s = torch.zeros_like(b)
        t = torch.zeros_like(b)

        # Initialize scalars
        rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
        omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)

        start_time = perf_counter()
        iter_times = []

        for i in range(maxiter):
            iter_start_time = perf_counter()

            # BiCGSTAB iteration
            rho = torch.vdot(r_tilde.flatten(), r.flatten())

            if rho.abs() < 1e-30:
                if verbose:
                    print("BiCGSTAB breakdown: rho ≈ 0")
                break

            beta = (rho / rho_prev) * (alpha / omega)
            rho_prev = rho

            p = r + beta * (p - omega * v)
            v = op.matvec(p)
            alpha = rho / torch.vdot(r_tilde.flatten(), v.flatten())
            s = r - alpha * v
            t = op.matvec(s)
            omega = torch.vdot(t.flatten(), s.flatten()) / \
                torch.vdot(t.flatten(), t.flatten())

            x = x + alpha * p + omega * s
            r = s - omega * t

            r_norm2 = torch.norm(r).item()
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)

            if verbose:
                print(
                    f"BiCGSTAB Iteration {i}: Residual = {r_norm2:.6e}, Time = {iter_time:.6f} s")

            if r_norm2 < tol:
                if verbose:
                    print(
                        f"BiCGSTAB converged at iteration {i} with residual {r_norm2:.6e}")
                return x, 0

        total_time = perf_counter() - start_time
        avg_iter_time = sum(iter_times) / \
            len(iter_times) if iter_times else 0.0

        if verbose:
            print("\nBiCGSTAB Performance Statistics:")
            print(f"Total time: {total_time:.6f} s")
            print(f"Average time per iteration: {avg_iter_time:.6f} s")

        return x, 1

    def v_cycle(self, op_hierarchy, b_hierarchy, u_hierarchy, grid_params, level: int = 0) -> torch.Tensor:
        """
        V-cycle multigrid recursion.

        Implements the standard V-cycle: pre-smooth, restrict, recurse, 
        prolongate, post-smooth.

        Args:
            op_hierarchy: List of operators at different levels
            b_hierarchy: List of RHS vectors at different levels
            u_hierarchy: List of solution vectors at different levels
            grid_params: List of grid dimensions at different levels
            level: Current recursion level

        Returns:
            u: Updated solution at current level
        """
        current_level_idx = len(op_hierarchy) - 1 - level
        nx, ny, nz = grid_params[current_level_idx]

        print(
            f"V-cycle level {level}, grid index: {current_level_idx}, grid size: {nx}x{ny}x{nz}")

        op = op_hierarchy[current_level_idx]
        b = b_hierarchy[current_level_idx]
        u = u_hierarchy[current_level_idx]

        # Coarsest grid: direct solve with BiCGSTAB
        if current_level_idx == 0 or level >= self.max_levels - 1:
            residual = self.compute_residual(op, b, u)
            residual_norm = torch.norm(residual).item()
            print(f"    Pre-solve residual norm: {residual_norm:.4e}")
            print(f"    Solving coarsest grid directly...")

            u_coarse, info = self.bicgstab_solver(
                op, b, u, tol=self.tolerance*0.1, maxiter=1000)

            if info != 0:
                print(
                    f"    Warning: Coarse grid solver did not converge! Info: {info}")

            u_hierarchy[current_level_idx] = u_coarse
            residual = self.compute_residual(
                op, b, u_hierarchy[current_level_idx])
            residual_norm = torch.norm(residual).item()
            print(f"    Post-solve residual norm: {residual_norm:.4e}")

            return u_hierarchy[current_level_idx]

        # Pre-smoothing
        residual_before_smooth = self.compute_residual(op, b, u)
        residual_norm_before_smooth = torch.norm(residual_before_smooth).item()
        print(
            f"    Pre-smooth residual norm: {residual_norm_before_smooth:.4e}")
        print(f"    Pre-smoothing...")

        u = self.smooth(op, b, u)
        u_hierarchy[current_level_idx] = u

        residual = self.compute_residual(op, b, u_hierarchy[current_level_idx])
        residual_norm = torch.norm(residual).item()
        print(f"    Post pre-smooth residual norm: {residual_norm:.4e}")

        # Coarse grid correction
        if current_level_idx > 0:
            # Restrict residual to coarse grid
            r_coarse = self.restrict(residual, nx, ny, nz)
            b_hierarchy[current_level_idx - 1] = r_coarse
            u_hierarchy[current_level_idx -
                        1] = torch.zeros_like(r_coarse, dtype=self.dtype)

            # Recursive call to coarser level
            e_coarse = self.v_cycle(
                op_hierarchy, b_hierarchy, u_hierarchy, grid_params, level + 1)

            # Prolongate error correction to fine grid
            nx_fine, ny_fine, nz_fine = grid_params[current_level_idx]
            e_fine = self.prolongate(e_coarse, nx_fine, ny_fine, nz_fine)

            # Apply correction
            u = u + e_fine
            u_hierarchy[current_level_idx] = u

        # Post-smoothing
        residual_before_post_smooth = self.compute_residual(op, b, u)
        residual_norm_before_post_smooth = torch.norm(
            residual_before_post_smooth).item()
        print(
            f"    Pre post-smooth residual norm: {residual_norm_before_post_smooth:.4e}")
        print(f"    Post-smoothing...")

        u = self.smooth(op, b, u)
        u_hierarchy[current_level_idx] = u

        residual = self.compute_residual(op, b, u_hierarchy[current_level_idx])
        residual_norm = torch.norm(residual).item()
        print(f"    Post post-smooth residual norm: {residual_norm:.4e}")

        return u

    def adaptive_criterion(self, residual_norms: list) -> bool:
        """
        Adaptive convergence criterion for multigrid.

        Detects slow convergence based on convergence rate over recent iterations.

        Args:
            residual_norms: List of residual norms from previous iterations

        Returns:
            bool: True if convergence is slow and more levels may be needed
        """
        if len(residual_norms) < 3:
            return False

        conv_rate = residual_norms[-1] / \
            residual_norms[-2] if residual_norms[-2] != 0 else 1
        return conv_rate > 0.8

    def solve(self) -> torch.Tensor:
        """
        Main multigrid solver routine.

        Sets up the multigrid hierarchy, performs V-cycle iterations until
        convergence, and returns the solution.

        Returns:
            solution: 3D solution tensor of shape (nz, ny, nx)
        """
        print(f"\n{'='*60}")
        print("Starting Adaptive Multigrid Complex Solver")
        print(f"{'='*60}")

        # Build grid hierarchy
        grid_params = []
        current_nx, current_ny = self.nx, self.ny
        current_nz = self.nz

        print(f"Building grid hierarchy:")
        while min(current_nx, current_ny) >= self.min_size and len(grid_params) < self.max_levels:
            grid_params.append((current_nx, current_ny, current_nz))
            print(
                f"  Level {len(grid_params)-1}: {current_nx}x{current_ny}x{current_nz}")
            current_nx = max(2, current_nx // 2)
            current_ny = max(2, current_ny // 2)

        num_levels = len(grid_params)
        print(f"Total {num_levels} grid levels")

        # Build operators and initialize vectors for each level
        print(f"\nBuilding system operators for each level:")
        op_hierarchy = []
        b_hierarchy = []
        u_hierarchy = []

        for i, (nx, ny, nz) in enumerate(grid_params):
            print(f"Level {i} ({nx}x{ny}x{nz}):")
            op = self.op(nx, ny, nz, dtype=self.dtype, device=self.device)
            b = op.give_b()
            u = torch.zeros(nx * ny * nz, dtype=self.dtype, device=self.device)

            op_hierarchy.append(op)
            b_hierarchy.append(b)
            u_hierarchy.append(u)

        # Reverse hierarchies for bottom-up indexing
        op_hierarchy.reverse()
        b_hierarchy.reverse()
        u_hierarchy.reverse()
        grid_params.reverse()

        print(f"\nStarting multigrid iterations:")
        print("-" * 30)

        start_time = time.time()

        # Main multigrid iteration loop
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}:")

            # Perform V-cycle
            u_hierarchy[-1] = self.v_cycle(op_hierarchy,
                                           b_hierarchy, u_hierarchy, grid_params)

            # Check convergence on finest grid
            op_finest = op_hierarchy[-1]
            b_finest = b_hierarchy[-1]
            u_finest = u_hierarchy[-1]

            finest_residual = self.compute_residual(
                op_finest, b_finest, u_finest)
            residual_norm = torch.norm(finest_residual).item()
            self.convergence_history.append(residual_norm)

            print(
                f"  Iteration {iteration + 1} completed, residual norm: {residual_norm:.4e}")

            # Check for convergence
            if residual_norm < self.tolerance:
                print(f"  ✓ Converged to tolerance {self.tolerance}")
                break

            # Check adaptive criterion for slow convergence
            if self.adaptive_criterion(self.convergence_history):
                print(f"  Note: Slow convergence detected, may need more grid levels")
        else:
            print("  Warning: Maximum iterations reached, may not have converged")

        solve_time = time.time() - start_time

        print("\n" + "="*60)
        print("Solution completed!")
        print(f"Total iterations: {len(self.convergence_history)}")
        print(f"Final residual: {self.convergence_history[-1]:.2e}")
        print(f"Solve time: {solve_time:.4f} seconds")
        print(f"{'='*60}")

        # Return solution reshaped to 3D
        return u_hierarchy[-1].reshape((self.nz, self.ny, self.nx))

    def verify_solution(self, solution: torch.Tensor) -> Tuple[float, float]:
        """
        Verify the correctness of the computed solution.

        Computes the residual norm and relative error to assess solution quality.

        Args:
            solution: 3D solution tensor

        Returns:
            residual_norm: L2 norm of residual
            relative_error: Relative error compared to RHS norm
        """
        print("\nVerifying solution correctness:")
        print("-" * 30)

        # Create operator and RHS for verification
        op = self.op(self.nx, self.ny, self.nz,
                     dtype=self.dtype, device=self.device)
        b = op.give_b()
        u_flat = solution.flatten()

        # Compute residual: r = A*u - b
        residual = op.matvec(u_flat) - b
        residual_norm = torch.norm(residual).item()
        relative_error = residual_norm / torch.norm(b).item()

        # Print solution statistics
        print(
            f"Solution real part range: [{torch.real(solution).min().item():.4f}, {torch.real(solution).max().item():.4f}]")
        print(
            f"Solution imaginary part range: [{torch.imag(solution).min().item():.4f}, {torch.imag(solution).max().item():.4f}]")
        print(
            f"Solution magnitude range: [{torch.abs(solution).min().item():.4f}, {torch.abs(solution).max().item():.4f}]")
        print(f"Verification residual norm: {residual_norm:.4e}")
        print(f"Relative error: {relative_error:.2e}")

        if relative_error < 1e-6:
            print("✓ Solution verification passed!")
        else:
            print("⚠ Solution may have accuracy issues")

        return residual_norm, relative_error


def demo():
    """
    Demonstration function showing usage of the PyTorch multigrid solver.

    Solves a 3D elliptic PDE with complex coefficients and visualizes convergence.
    """
    print("PyTorch Elliptic PDE Multigrid Solver Demo")
    print("=" * 50)

    # Choose device (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize solver with complex double precision
    solver = GeometricMultigrid(
        nx=32, ny=32, nz=12,
        dtype=torch.complex64,
        tolerance=1e-6,
        device=device
    )

    # Alternative: use complex single precision for faster computation
    # solver = GeometricMultigrid(
    #     nx=32, ny=32, nz=12,
    #     dtype=torch.complex64,
    #     tolerance=1e-5,
    #     device=device
    # )

    # Solve the system
    solution = solver.solve()

    # Verify solution quality
    solver.verify_solution(solution)

    # Print convergence statistics
    print(f"\nConvergence Statistics:")
    print(f"Number of iterations: {len(solver.convergence_history)}")
    print(f"Final residual: {solver.convergence_history[-1]:.2e}")

    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.title('PyTorch Adaptive Multigrid Complex Solution Convergence', fontsize=16)
    plt.semilogy(range(1, len(solver.convergence_history) + 1),
                 solver.convergence_history, 'b-o', markersize=4, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Residual Norm', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save convergence plot
    solve_time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    filename = f"PyTorch_Adaptive_Multigrid_Complex_Convergence_{solve_time_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved as: {filename}")

    # Optional: visualize solution slices
    if solution.shape[0] > 1:  # If we have multiple z-slices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot real and imaginary parts of middle z-slice
        mid_z = solution.shape[0] // 2

        # Real part
        im1 = axes[0, 0].imshow(
            solution[mid_z, :, :].real.cpu().numpy(), cmap='RdBu_r')
        axes[0, 0].set_title(f'Real Part (z-slice {mid_z})')
        plt.colorbar(im1, ax=axes[0, 0])

        # Imaginary part
        im2 = axes[0, 1].imshow(
            solution[mid_z, :, :].imag.cpu().numpy(), cmap='RdBu_r')
        axes[0, 1].set_title(f'Imaginary Part (z-slice {mid_z})')
        plt.colorbar(im2, ax=axes[0, 1])

        # Magnitude
        im3 = axes[1, 0].imshow(
            torch.abs(solution[mid_z, :, :]).cpu().numpy(), cmap='viridis')
        axes[1, 0].set_title(f'Magnitude (z-slice {mid_z})')
        plt.colorbar(im3, ax=axes[1, 0])

        # Phase
        im4 = axes[1, 1].imshow(torch.angle(
            solution[mid_z, :, :]).cpu().numpy(), cmap='hsv')
        axes[1, 1].set_title(f'Phase (z-slice {mid_z})')
        plt.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        solution_filename = f"PyTorch_Multigrid_Solution_Visualization_{solve_time_str}.png"
        plt.savefig(solution_filename, dpi=300, bbox_inches='tight')
        print(f"Solution visualization saved as: {solution_filename}")

    print("\nDemo completed successfully!")
    print(f"{'='*80}")
