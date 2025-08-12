import torch
import torch.nn as nn
import time
from time import perf_counter
from typing import Tuple, Optional, Callable
from pyqcu.ascend import dslash


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
                                        mg_size[1], mg_size[2], latt_size[2]//mg_size[2], mg_size[3], latt_size[3]//mg_size[3]]
    if verbose:
        print(f"null_vecs.shape:{null_vecs.shape}")
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
        print(f"restrict:shape,dof:{shape,dof}")
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
        print(f"prolong:shape,dof:{shape,dof}")
    _coarse_vec = coarse_vec.reshape(shape=shape[0:1]+shape[-8:][::2]).clone()
    if len(shape) == 10:
        print("EeTtZzYyXx, ETZYX->eTtZzYyXx")
        return torch.einsum(
            "EeTtZzYyXx, ETZYX->eTtZzYyXx", local_ortho_null_vecs.conj(), _coarse_vec).reshape([dof, shape[-8]*shape[-7], shape[-6]*shape[-5], shape[-4]*shape[-3], shape[-2]*shape[-1]])
    elif len(shape) == 11:
        print("EscTtZzYyXx, ETZYX->scTtZzYyXx")
        return torch.einsum(
            "EscTtZzYyXx, ETZYX->scTtZzYyXx", local_ortho_null_vecs.conj(), _coarse_vec).reshape([4, 3, shape[-8]*shape[-7], shape[-6]*shape[-5], shape[-4]*shape[-3], shape[-2]*shape[-1]])
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


class op:
    def __init__(self, wilson: dslash.wilson_parity = None, U_eo: torch.Tensor = None, clover: dslash.clover_parity = None, clover_eo: torch.Tensor = None, fine_hopping: hopping = None, fine_sitting: sitting = None, local_ortho_null_vecs: torch.Tensor = None, verbose=True):
        self.hopping = hopping(wilson=wilson, U_eo=U_eo)
        self.sitting = sitting(clover=clover, clover_eo=clover_eo)
        self.verbose = verbose
        if fine_hopping != None and fine_sitting != None and local_ortho_null_vecs != None:
            shape = local_ortho_null_vecs.shape  # EeTtZzYyXx
            dtype = local_ortho_null_vecs.dtype
            device = local_ortho_null_vecs.device
            dof = shape[0]
            self.hopping.M_eo = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]//2], dtype=dtype, device=device)  # EETZYX_p
            self.hopping.M_oe = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]//2], dtype=dtype, device=device)  # EETZYX_p
            self.sitting.M_ee = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]//2], dtype=dtype, device=device)  # EETZYX_p
            self.sitting.M_oo = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]//2], dtype=dtype, device=device)  # EETZYX_p
            src_c = torch.zeros(
                size=[dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)   # ETZYX
            src_c_I = torch.ones(
                size=[shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # TZYX
            dest_f = torch.zeros(size=[dof, shape[-8]*shape[-7], shape[-6]*shape[-5],
                                 shape[-4]*shape[-3], shape[-2]*shape[-1]], dtype=dtype, device=device) if len(shape) == 10 else torch.zeros(size=[4, 3, shape[-8]*shape[-7], shape[-6]*shape[-5],
                                                                                                                                                   shape[-4]*shape[-3], shape[-2]*shape[-1]], dtype=dtype, device=device)  # e(Tt)(Zz)(Yy)(Xx)
            dest_f_eo = dslash.xxxtzyx2pxxxtzyx(input_array=dest_f.clone())
            if self.verbose:
                print(
                    f"local_ortho_null_vecs.shape,src_c.shape,dest_f.shape:{local_ortho_null_vecs.shape,src_c.shape,dest_f.shape}")
            for e in range(dof):
                _src_c = src_c.clone()
                _src_c[e] = src_c_I.clone()
                _src_f = prolong(
                    local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=verbose)
                if self.verbose:
                    print(
                        f"_src_f.shape:{_src_f.shape}")
                _src_f_eo = dslash.xxxtzyx2pxxxtzyx(input_array=_src_f)
                # give partly sitting.ee and whole hopping.oe
                _dest_f_eo = dest_f_eo.clone()
                _dest_f_eo[0] = fine_hopping.matvec_eo(src_o=_src_f_eo[1])
                _dest_f = dslash.pxxxtzyx2xxxtzyx(input_array=_dest_f_eo)
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f)
                _dest_c_eo = dslash.xxxtzyx2pxxxtzyx(input_array=_dest_c)
                self.sitting.M_ee[:, e, ...] = _dest_c_eo[0]
                self.hopping.M_oe[:, e, ...] = _dest_c_eo[1]
                # give partly sitting.oo and whole hopping.eo
                _dest_f_eo = dest_f_eo.clone()
                _dest_f_eo[1] = fine_hopping.matvec_oe(src_e=_src_f_eo[0])
                _dest_f = dslash.pxxxtzyx2xxxtzyx(input_array=_dest_f_eo)
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f)
                _dest_c_eo = dslash.xxxtzyx2pxxxtzyx(input_array=_dest_c)
                self.sitting.M_oo[:, e, ...] = _dest_c_eo[1]
                self.hopping.M_eo[:, e, ...] = _dest_c_eo[0]
                # give aother partly sitting.ee and sitting.oo
                dest_f_eo = dest_f_eo.clone()
                _dest_f_eo[0] = fine_sitting.matvec_ee(src_e=_src_f_eo[0])
                _dest_f_eo[1] = fine_sitting.matvec_oo(src_o=_src_f_eo[1])
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
            src_e=src_eo[0])+self.sitting.matvec_oo(src_o=src_eo[1])
        return dslash.pxxxtzyx2xxxtzyx(input_array=dest_eo).clone()


class GMRESSmoother:
    """
    GMRES-based smoother for multigrid methods.
    Implements the Generalized Minimal Residual method with restarts
    as a smoother for the multigrid list.
    Args:
        max_krylov: Maximum Krylov subspace dimension (default: 5)
        max_restarts: Maximum number of restarts (default: 1)
        tol: Relative tolerance for convergence (default: 0.1)
    """

    def __init__(self, max_krylov: int = 5, max_restarts: int = 1, tol: float = 0.1):
        self.max_krylov = max_krylov
        self.max_restarts = max_restarts
        self.tol = tol

    def smooth(self, matvec: Callable[[torch.Tensor], torch.Tensor],
               b: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Apply GMRES smoothing to the linear system op*x = b.
        Args:
            matvec: Linear operator
            b: Right-hand side vector
            x0: Initial guess
        Returns:
            x: Smoothed solution
        """
        def _matvec(src: torch.Tensor) -> torch.Tensor:
            return matvec(src.reshape(b.shape)).flatten()
        x = x0.clone().flatten()
        r = b.flatten() - _matvec(x)
        r_norm = torch.norm(r).item()
        if r_norm < 1e-12:
            return x0
        # GMRES with restarts
        for _ in range(self.max_restarts):
            Q, H = self._arnoldi(matvec=_matvec, r0=r, r_norm=r_norm)
            # Set up least squares problem
            e1 = torch.zeros(H.shape[1] + 1, dtype=b.dtype, device=b.device)
            e1[0] = r_norm
            # Solve least squares problem
            y = self._solve_least_squares(H, e1)
            # Update solution
            dx = Q[:, :-1] @ y
            x = x + dx
            # Check convergence
            new_r = b.flatten() - _matvec(x)
            new_r_norm = torch.norm(new_r).item()
            if new_r_norm < self.tol * r_norm:
                break
            r = new_r
            r_norm = new_r_norm
        return x.reshape(b.shape)

    def _arnoldi(self, matvec: Callable[[torch.Tensor], torch.Tensor],
                 r0: torch.Tensor, r_norm: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arnoldi process for building orthonormal Krylov subspace.
        Args:
            matvec: Linear operator
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
            w = matvec(Q[:, j])
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


class mg:
    def __init__(self, b: torch.Tensor,  wilson: dslash.wilson_parity, U_eo: torch.Tensor, clover: dslash.clover_parity, clover_eo: torch.Tensor,  min_size: int = 2, max_levels: int = 5, dof: int = 12, tol: float = 1e-6, max_iter: int = 500, x0=None, verbose=True):
        self.b = b
        self.op_list = [op(wilson=wilson, U_eo=U_eo,
                           clover=clover, clover_eo=clover_eo, verbose=verbose)]
        self.min_size = min_size
        self.max_levels = max_levels
        self.dof = dof
        self.tol = tol
        self.max_iter = max_iter
        self.x0 = x0.clone() if x0 is not None else torch.randn_like(b)
        self.verbose = verbose
        # Build grid list
        _Lx = b.shape[-1]
        _Ly = b.shape[-2]
        _Lz = b.shape[-3]
        _Lt = b.shape[-4]
        self.grid_list = []
        self.b_list = [torch.zeros_like(b)]
        self.u_list = [torch.randn_like(b)]
        print(f"Building grid list:")
        while len(self.grid_list) < self.max_levels:
            self.grid_list.append([_Lt, _Lz, _Ly, _Lx])
            print(
                f"  Level {len(self.grid_list)-1}: {_Lx}x{_Ly}x{_Lz}x{_Lt}")
            if all(_ == self.min_size for _ in [_Lx, _Ly, _Lz, _Lt]):
                break
            _Lx = max(self.min_size, _Lx // 2)
            _Ly = max(self.min_size, _Ly // 2)
            _Lz = max(self.min_size, _Lz // 2)
            _Lt = max(self.min_size, _Lt // 2)
        print(f"self.grid_list:{self.grid_list}")
        self.lonv_list = []  # local_ortho_null_vecs_list
        # Build local-orthonormal near-null space vectors
        for i in range(1, len(self.grid_list)):
            if i == 1:
                _null_vecs = torch.randn(dof, 4, 3, b.shape[-4], b.shape[-3], b.shape[-2], b.shape[-1],
                                         dtype=b.dtype, device=b.device)
            else:
                _null_vecs = torch.randn(dof, dof, self.grid_list[i-1][-4], self.grid_list[i-1][-3], self.grid_list[i-1][-2], self.grid_list[i-1][-1],
                                         dtype=b.dtype, device=b.device)
            _null_vecs = give_null_vecs(
                null_vecs=_null_vecs,
                matvec=self.op_list[i-1].matvec,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=self.verbose
            )
            _local_ortho_null_vecs = local_orthogonalize(
                null_vecs=_null_vecs,
                mg_size=self.grid_list[i])
            self.lonv_list.append(_local_ortho_null_vecs)
            self.b_list.append(torch.zeros(
                size=[dof]+self.grid_list[i], dtype=b.dtype, device=b.device))
            self.u_list.append(torch.zeros(
                size=[dof]+self.grid_list[i], dtype=b.dtype, device=b.device))
            self.op_list.append(op(fine_hopping=self.op_list[i-1].hopping, fine_sitting=self.op_list[i -
                                                                                                     1].sitting, local_ortho_null_vecs=_local_ortho_null_vecs, verbose=self.verbose))
        self.num_levels = len(self.grid_list)
        self.convergence_history = []
        self.level_info = []
        # Initialize GMRES smoother
        self.smoother = GMRESSmoother(
            max_krylov=5, max_restarts=1, tol=0.1)

    def smooth(self, level: int = 0) -> torch.Tensor:
        return self.smoother.smooth(matvec=self.op_list[level].matvec, b=self.b_list[level], x0=self.u_list[level])

    def give_residual(self, level: int = 0) -> torch.Tensor:
        return self.b_list[level] - self.op_list[level].matvec(self.u_list[level])

    def give_residual_norm(self, level: int = 0) -> torch.Tensor:
        return torch.norm(self.give_residual(level=level)).item()

    def v_cycle(self, level: int = 0) -> torch.Tensor:
        """
        V-cycle multigrid recursion.
        Implements the standard V-cycle: pre-smooth, restrict, recurse, 
        prolongate, post-smooth.
        Args:
            level: Current recursion level
        Returns:
            u: Updated solution at current level
        """
        if self.verbose:
            print(
                f"V-cycle level {level}, mg_size: {self.grid_list[level]}")
        # Coarsest grid: direct solve with BiCGSTAB
        if level == self.num_levels-1:
            if self.verbose:
                print(
                    f"    Pre-solve residual norm: {self.give_residual_norm(level=level):.4e}")
                print(f"    Solving coarsest grid directly...")
            self.u_list[level] = bicgstab(
                b=self.b_list[level], matvec=self.op_list[level].matvec, x0=self.u_list[level], tol=self.tol*0.1, max_iter=self.max_iter)
            if self.verbose:
                print(
                    f"    Post-solve residual norm: {self.give_residual_norm(level=level):.4e}")
            return self.u_list[level].clone()
        # Pre-smoothing
        if self.verbose:
            print(
                f"    Pre-smooth residual norm: {self.give_residual_norm(level=level):.4e}")
            print(f"    Pre-smoothing...")
        self.u_list[level] = self.smooth(level=level)
        residual = self.give_residual(level=level)
        if self.verbose:
            print(
                f"    Post pre-smooth residual norm: {torch.norm(residual).item():.4e}")
        # Coarse grid correction
        if level != self.num_levels-1:
            # Restrict residual to coarse grid
            r_coarse = restrict(fine_vec=residual,
                                local_ortho_null_vecs=self.lonv_list[level])
            self.b_list[level + 1] = r_coarse
            self.u_list[level + 1] = torch.zeros_like(r_coarse)
            # Recursive call to coarser level
            e_coarse = self.v_cycle(level=level + 1)
            # Prolongate error correction to fine grid
            e_fine = prolong(coarse_vec=e_coarse,
                             local_ortho_null_vecs=self.lonv_list[level])
            # Apply correction
            self.u_list[level] += e_fine
        # Post-smoothing
        if self.verbose:
            print(
                f"    Pre post-smooth residual norm: {self.give_residual_norm(level=level):.4e}")
            print(f"    Post-smoothing...")
        self.u_list[level] = self.smooth(level=level)
        residual = self.give_residual(level=level)
        if self.verbose:
            print(
                f"    Post post-smooth residual norm: {torch.norm(residual).item():.4e}")
        return self.u_list[level].clone()

    def solve(self) -> torch.Tensor:
        """
        Main multigrid solver routine.
        Sets up the multigrid list, performs V-cycle iterations until
        convergence, and returns the solution.
        """
        print(f"\nStarting multigrid iterations:")
        print("-" * 30)
        start_time = time.time()
        # Main multigrid iteration loop
        for iteration in range(self.max_iter):
            print(f"\nIteration {iteration + 1}:")
            # Perform V-cycle
            self.u_list[0] = self.v_cycle(level=0)
            # Check convergence on finest grid
            residual_norm = self.give_residual_norm(level=0)
            self.convergence_history.append(residual_norm)
            print(
                f"  Iteration {iteration + 1} completed, residual norm: {residual_norm:.4e}")
            # Check for convergence
            if residual_norm < self.tol:
                print(f"  ✓ Converged to tolerance {self.tol}")
                break
        else:
            print("  Warning: Maximum iterations reached, may not have converged")
        solve_time = time.time() - start_time
        print("\n" + "="*60)
        print("Solution completed!")
        print(f"Total iterations: {len(self.convergence_history)}")
        print(f"Final residual: {self.convergence_history[-1]:.2e}")
        print(f"Solve time: {solve_time:.4f} seconds")
        print(f"{'='*60}")
        return self.u_list[0].clone()
