import torch
from time import perf_counter
from typing import Tuple, Callable
from pyqcu.ascend import dslash
from pyqcu.ascend.include import *


def cg(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, verbose: bool = True) -> torch.Tensor:
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
    if verbose:
        print(f"Norm of b:{torch.norm(b).item()}")
        print(f"Norm of r:{torch.norm(r).item()}")
        print(f"Norm of x0:{torch.norm(x).item()}")
    r_norm = torch.norm(r).item()
    if r_norm < tol:
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
        if r_norm < tol:
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


def bicgstab(b: torch.Tensor, matvec: Callable[[torch.Tensor], torch.Tensor], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, verbose: bool = True) -> torch.Tensor:
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
    if verbose:
        print(f"Norm of b:{torch.norm(b).item()}")
        print(f"Norm of r:{torch.norm(r).item()}")
        print(f"Norm of x0:{torch.norm(x).item()}")
    r_norm = torch.norm(r).item()
    if r_norm < tol:
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
        if r_norm < tol:
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
    tol: float = 1e-6, max_iter: int = 1000, normalize: bool = True, ortho_r: bool = True, ortho_null_vecs: bool = False, verbose: bool = True
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
    Returns:
        Orthonormal near-null space vectors
    """
    dof = null_vecs.shape[0]  # Number of null space vectors
    null_vecs = torch.randn_like(null_vecs)  # [Eetzyx]
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
        null_vecs[i] -= bicgstab(b=matvec(null_vecs[i]), matvec=matvec, x0=torch.zeros_like(null_vecs[i]), tol=tol*1000,
                                 max_iter=max_iter, verbose=True)  # tol needs to be bigger...
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
    return null_vecs.clone()


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
                                        mg_size[1], mg_size[2], latt_size[2]//mg_size[2], mg_size[3], latt_size[3]//mg_size[3]]  # [EeTtZzYyXx]
    if verbose:
        print(f"null_vecs.shape:{null_vecs.shape}")
        print(f"dof,latt_size,mg_size,shape:{dof,latt_size,mg_size,shape}")
    if not all(latt_size[-i-1] == shape[-2*i-1]*shape[-2*i-2] for i in range(4)):
        print(
            'not all(latt_size[-i-1] == shape[-2*i-1]*shape[-2*i-2] for i in range(4))')
    local_null_vecs = null_vecs.reshape(shape=shape).clone()
    local_ortho_null_vecs = torch.zeros_like(local_null_vecs)
    for X in range(mg_size[-1]):
        for Y in range(mg_size[-2]):
            for Z in range(mg_size[-3]):
                for T in range(mg_size[-4]):
                    _local_null_vecs = local_null_vecs[...,
                                                       T, :, Z, :, Y, :, X, :]  # [Eetzyx]
                    for i in range(dof):  # [E]
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
    return local_ortho_null_vecs.clone()


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
    def __init__(self, wilson: dslash.wilson = None, U: torch.Tensor = None):
        self.M = torch.zeros([])
        self.wilson = wilson
        self.U = U

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        if self.wilson != None:
            return self.wilson.give_wilson(
                src=src.reshape([4, 3]+list(src.shape[1:])), U=self.U, with_I=False).reshape([12]+list(src.shape)[1:]).clone()  # e->sc->e
        else:
            return torch.einsum(
                "EeTZYX, eTZYX->ETZYX", self.M, src)


class sitting:
    def __init__(self, clover: dslash.clover = None, clover_term: torch.Tensor = None):
        self.M = torch.zeros([])
        self.clover = clover
        self.clover_term = clover_term

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        if self.clover != None:  # remmber to add I
            return self.clover.give_clover(
                src=src.reshape([4, 3]+list(src.shape[1:])), clover_term=self.clover_term).reshape([12]+list(src.shape)[1:]).clone()  # e->sc->e
        else:
            return torch.einsum(
                "EeTZYX, eTZYX->ETZYX", self.M, src).clone()


class op:
    def __init__(self, wilson: dslash.wilson = None, U: torch.Tensor = None, clover: dslash.clover = None, clover_term: torch.Tensor = None, fine_hopping: hopping = None, fine_sitting: sitting = None, local_ortho_null_vecs: torch.Tensor = None, verbose: bool = True):
        self.hopping = hopping(wilson=wilson, U=U)
        self.sitting = sitting(clover=clover, clover_term=clover_term)
        self.verbose = verbose
        if fine_hopping != None and fine_sitting != None and local_ortho_null_vecs != None:
            shape = local_ortho_null_vecs.shape  # EeTtZzYyXx
            dtype = local_ortho_null_vecs.dtype
            device = local_ortho_null_vecs.device
            dof = shape[0]
            fine_dof = shape[1]
            self.hopping.M = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # EETZYX
            self.sitting.M = torch.zeros(
                size=[dof, dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)  # EETZYX
            src_c = torch.zeros(
                size=[dof, shape[-4*2], shape[-3*2], shape[-2*2], shape[-1*2]], dtype=dtype, device=device)   # ETZYX
            dest_f = torch.zeros(size=[fine_dof, shape[-8]*shape[-7], shape[-6]*shape[-5],
                                 shape[-4]*shape[-3], shape[-2]*shape[-1]], dtype=dtype, device=device)  # e(Tt)(Zz)(Yy)(Xx)
            if self.verbose:
                print(
                    f"local_ortho_null_vecs.shape,src_c.shape,dest_f.shape:{local_ortho_null_vecs.shape,src_c.shape,dest_f.shape}")
            even_mask_c = give_parity_mask(
                x=src_c.shape[-1], y=src_c.shape[-2], z=src_c.shape[-3], t=src_c.shape[-4], parity=0)
            odd_mask_c = give_parity_mask(
                x=src_c.shape[-1], y=src_c.shape[-2], z=src_c.shape[-3], t=src_c.shape[-4], parity=1)
            even_mask_f = give_parity_mask(
                x=dest_f.shape[-1], y=dest_f.shape[-2], z=dest_f.shape[-3], t=dest_f.shape[-4], parity=0)
            odd_mask_f = give_parity_mask(
                x=dest_f.shape[-1], y=dest_f.shape[-2], z=dest_f.shape[-3], t=dest_f.shape[-4], parity=1)
            for e in range(dof):
                _src_c = src_c.clone()
                _src_c[e] = torch.ones_like(_src_c[e])
                _src_f = prolong(
                    local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                # give partly sitting.ee and whole hopping.oe
                _src_e0_f = _src_f.clone()
                _src_e0_f[..., odd_mask_f] = torch.zeros_like(
                    _src_e0_f[..., odd_mask_f])
                _dest_f = fine_hopping.matvec(src=_src_e0_f)
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f, verbose=self.verbose)
                self.sitting.M[:, e, odd_mask_f] = _dest_c[..., even_mask_c]
                self.hopping.M[:, e, odd_mask_f] = _dest_c[..., even_mask_c]
                # give partly sitting.oo and whole hopping.eo
                _dest_f = dest_f.clone()
                _dest_f[1] = fine_hopping.matvec_oe(src_e=_src_f[0])
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f, verbose=self.verbose)
                self.sitting.M_oo[:, e, ...] = _dest_c[1]
                self.hopping.M[:, e, ...] = _dest_c[0]
                # give aother partly sitting.ee and sitting.oo
                dest_f = dest_f.clone()
                _dest_f[0] = fine_sitting.matvec_ee(src_e=_src_f[0])
                _dest_f[1] = fine_sitting.matvec_oo(src_o=_src_f[1])
                _dest_c = restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f, verbose=self.verbose)
                self.sitting.M_ee[:, e, ...] += _dest_c[0]
                self.sitting.M_oo[:, e, ...] += _dest_c[1]

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        return self.hopping.matvec(src=src)+self.sitting.matvec(src=src)


class mg:
    def __init__(self, b: torch.Tensor,  wilson: dslash.wilson, U: torch.Tensor, clover: dslash.clover, clover_term: torch.Tensor,  min_size: int = 2, max_levels: int = 2, dof_list: Tuple[int, int, int, int] = [12, 12, 12, 12, 8, 8, 4, 12, 12, 12, 8, 4, 2, 4, 4, 24, 12, 12, 12, 4, 4, 4, 4, 4], tol: float = 1e-6, max_iter: int = 1000, x0: torch.Tensor = None, max_restarts: int = 5, pre_smooth: bool = True, post_smooth: bool = False, verbose: bool = True):
        self.b = b.reshape([12]+list(b.shape)[2:])  # sc->e
        self.min_size = min_size
        self.max_levels = max_levels
        self.dof_list = dof_list
        print(f"self.dof_list:{self.dof_list}")
        self.tol = tol
        self.max_iter = max_iter
        self.max_restarts = max_restarts
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
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
        self.b_list = [self.b]
        self.u_list = [self.x0]
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
        self.lonv_list = []  # local_ortho_null_vecs_list
        # Build local-orthonormal near-null space vectors
        for i in range(1, len(self.grid_list)):
            _null_vecs = torch.randn(self.dof_list[i], self.dof_list[i-1], self.grid_list[i-1][-4], self.grid_list[i-1][-3], self.grid_list[i-1][-2], self.grid_list[i-1][-1],
                                     dtype=b.dtype, device=b.device)

            _null_vecs = give_null_vecs(
                null_vecs=_null_vecs,
                matvec=self.op_list[i-1].matvec,
                tol=self.tol,
                verbose=False,
            )
            _local_ortho_null_vecs = local_orthogonalize(
                null_vecs=_null_vecs,
                mg_size=self.grid_list[i], verbose=False)
            self.lonv_list.append(_local_ortho_null_vecs)
            self.b_list.append(torch.randn(
                size=[self.dof_list[i]]+self.grid_list[i], dtype=b.dtype, device=b.device))
            self.u_list.append(torch.randn(
                size=[self.dof_list[i]]+self.grid_list[i], dtype=b.dtype, device=b.device))
            self.op_list.append(op(fine_hopping=self.op_list[i-1].hopping, fine_sitting=self.op_list[i -
                                                                                                     1].sitting, local_ortho_null_vecs=_local_ortho_null_vecs, verbose=self.verbose))
        self.convergence_history = []

    def give_residual(self, level: int = 0) -> torch.Tensor:
        return self.b_list[level] - self.op_list[level].matvec(self.u_list[level])

    def give_residual_norm(self, level: int = 0) -> torch.Tensor:
        return torch.norm(self.give_residual(level=level)).item()

    def cycle(self, level: int = 0) -> torch.Tensor:
        # init start
        x0 = self.u_list[level].clone()
        b = self.b_list[level].clone()
        matvec = self.op_list[level].matvec
        verbose = self.verbose
        max_iter = self.max_iter
        # init end
        x = x0.clone() if x0 is not None else torch.randn_like(b)
        r = b - matvec(x)
        if verbose:
            print(f"Norm of b:{torch.norm(b).item()}")
            print(f"Norm of r:{torch.norm(r).item()}")
            print(f"Norm of x0:{torch.norm(x).item()}")
        r_norm = torch.norm(r).item()
        tol = r_norm*0.25 if level != self.num_levels else r_norm*0.1
        if r_norm < tol:
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
            # cycle start
            if level != self.num_levels:
                r_coarse = restrict(
                    local_ortho_null_vecs=self.lonv_list[level], fine_vec=r)
                self.b_list[level+1] = r_coarse
                e_coarse = self.cycle(level=level+1)
                e_fine = prolong(
                    local_ortho_null_vecs=self.lonv_list[level], coarse_vec=e_coarse)
                x = x + e_fine
            # cycle end
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)
            if verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"MG-{level}-BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
            if r_norm < tol:
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
        self.u_list[level] = x.clone()
        return x.clone()

    def solve(self) -> torch.Tensor:
        """
        Main multigrid solver routine.
        Sets up the multigrid list, performs cycle iterations until
        convergence, and returns the solution.
        """
        start_time = perf_counter()
        iter_times = []
        # Main multigrid iteration loop
        self.convergence_history.append(self.give_residual_norm())
        for i in range(self.max_iter):
            print(f"\nMG:Iteration {i + 1}:")
            iter_start_time = perf_counter()
            # Perform cycle
            self.u_list[0] = self.cycle(level=0)
            # Check convergence on finest grid
            residual_norm = self.give_residual_norm(level=0)
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)
            if self.verbose:
                print(
                    f"MG-Iteration {i + 1} completed, residual norm: {residual_norm:.4e}")
            # Check for convergence
            if residual_norm < self.tol:
                if self.verbose:
                    print(
                        f"Converged at iteration {i} with residual {residual_norm:.6e}")
                break
        else:
            print("  Warning: Maximum iterations reached, may not have converged")
        total_time = perf_counter() - start_time
        avg_iter_time = sum(iter_times) / len(iter_times)
        print("\nPerformance Statistics:")
        print(f"Total iterations: {len(iter_times)}")
        print(f"Total time: {total_time:.6f} seconds")
        print(f"Average time per iteration: {avg_iter_time:.6f} s")
        print(f"Final residual: {self.convergence_history[-1]:.2e}")
        return self.u_list[0].clone().reshape([4, 3]+list(self.u_list[0].shape)[1:])

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
            f"Iteration(self.max_restarts:{self.max_restarts})", fontsize=12)
        plt.ylabel('Residual Norm', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
