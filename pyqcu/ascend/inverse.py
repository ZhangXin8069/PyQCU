import torch
import torch.nn as nn
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
    x = x0.clone() if x0 is not None else torch.rand_like(b)
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
    x = x0.clone() if x0 is not None else torch.rand_like(b)
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
    tol: float = 1e-6, max_iter: int = 500, normalize: bool = True, verbose: bool = True
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
        verbose: Print progress information (default: True).
    Returns:
        Orthonormal near-null space vectors
    """
    dof = null_vecs.shape[0]  # Number of null space vectors
    null_vecs = torch.rand_like(null_vecs)
    for i in range(dof):
        # The orthogonalization of r
        if normalize:
            null_vecs[i] /= torch.norm(null_vecs[i]).item()
        for j in range(0, i):
            null_vecs[i] -= torch.vdot(null_vecs[j].flatten(), null_vecs[i].flatten())/torch.vdot(
                null_vecs[j].flatten(), null_vecs[j].flatten())*null_vecs[j]
        if normalize:
            null_vecs[i] /= torch.norm(null_vecs[i]).item()
        # v=r-A^{-1}Ar
        null_vecs[i] -= bicgstab(b=matvec(null_vecs[i]), matvec=matvec, tol=tol*1000, max_iter=max_iter, x0=torch.zeros_like(null_vecs[i]),
                                 verbose=verbose)  # tol needs to be bigger...
        if verbose:
            print(f"A*v/v check:")
            Av = matvec(null_vecs[i])
            print(f"  Vector {i}: ||A*v|| = {torch.norm(Av).item():.6e}")
            print(
                f"  Vector {i}: v = {null_vecs[i]}")
            print(
                f"  Vector {i}: A*v = {Av}")
            print(
                f"  Vector {i}: A*v/v = {Av/null_vecs[i]}")
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
            print(f"A*v/v check again:")
            Av = matvec(null_vecs[i])
            print(f"  Vector {i}: ||A*v|| = {torch.norm(Av).item():.6e}")
            print(
                f"  Vector {i}: v = {null_vecs[i]}")
            print(
                f"  Vector {i}: A*v = {Av}")
            print(
                f"  Vector {i}: A*v/v = {Av/null_vecs[i]}")
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


class mg(nn.Module):
    def __init__(self,
                 matvec: Callable[[torch.Tensor], torch.Tensor],
                 latt_size: Tuple[int, int, int, int] = (
                     16, 16, 16, 16),  # 4D lattice (x, y, z, t)
                 n_refine: int = 3,  # Multigrid refinement levels
                 kappa: float = 0.1,  # Wilson parameter
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.complex64,
                 verbose: bool = False):
        """
        4D lattice solver with spin=4 and color=3, based on dslash operator (Wilson-Dirac)
        """
        super().__init__()
        self.matvec = matvec
        self.latt_size = latt_size
        self.Lx, self.Ly, self.Lz, self.Lt = latt_size
        self.volume = self.Lx * self.Ly * self.Lz * self.Lt
        self.kappa = kappa
        self.device = device or (torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        self.verbose = verbose
        self.wilson = dslash.wilson(
            latt_size=self.latt_size,
            kappa=self.kappa,
            dtype=self.dtype,
            device=self.device,
            verbose=False
        )
        self.clover = dslash.clover(
            latt_size=self.latt_size,
            kappa=self.kappa,
            dtype=self.dtype,
            device=self.device,
            verbose=False
        )
        self.U = self.wilson.generate_gauge_field(sigma=0.1, seed=42)
        self.clover_term = self.clover.make_clover(U=self.U)
        # Initialize multigrid
        self.mg = self._mg(self, n_refine)
        if self.verbose:
            print(
                f"Initialization complete: Lattice size {self.Lx}x{self.Ly}x{self.Lz}x{self.Lt}, Spin 4, Color 3")
            print(f"Device: {self.device}, Dtype: {self.dtype}")

    class hopping(dslash.wilson):
        def __init__(self,
                     latt_size: Tuple[int, int, int, int],
                     kappa: float = 0.1,
                     u_0: float = 1.0,
                     dtype: torch.dtype = torch.complex128,
                     device: torch.device = None,
                     verbose: bool = False):
            """
            Wilson-Dirac operator on a 4D lattice with SU_eo(3) gauge fields with parity decomposition.
            Args:
                latt_size: Tuple (Lx_p, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3, p(parity)=2.
                kappa: Hopping parameter (controls fermion mass).
                u_0: Wilson parameter (usually 1.0).
                dtype: Data type for tensors.
                device: Device to run on (default: CPU).
                verbose: Enable verbose output for debugging.
            reference:
                [4].
            addition:
            """
            super().__init__(latt_size=latt_size, kappa=kappa,
                             u_0=u_0, dtype=dtype, device=device, verbose=False)
            self.Lx_p = self.Lx // 2
            self.verbose = verbose
            if self.verbose:
                print(f"Initializing Wilson with parity decomposition:")
                print(f"  Lattice size: {latt_size} (x,y,z,t)")
                print(f"  Lattice x with parity: {self.Lx_p}")
                print(f"  Parameters: kappa={kappa}, u_0={u_0}")
                print(
                    f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
                print(f"  Device: {self.device}")

    class sitting(dslash.clover):
        def __init__(self,
                     latt_size: Tuple[int, int, int, int],
                     kappa: float = 0.1,
                     u_0: float = 1.0,
                     dtype: torch.dtype = torch.complex128,
                     device: torch.device = None,
                     verbose: bool = False):
            """
            The Clover term corrected by adding the Wilson-Dirac operator
            Args:
                latt_size: Tuple (Lx_p, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3, p(parity)=2
                kappa: Hopping parameter (controls fermion mass)
                u_0: Wilson parameter (usually 1.0)
                dtype: Data type for tensors
                device: Device to run on (default: CPU)
                verbose: Enable verbose output for debugging
            reference:
                [1](1-60);[1](1-160)
            """
            super().__init__(latt_size=latt_size, kappa=kappa,
                             u_0=u_0, dtype=dtype, device=device, verbose=False)
            self.Lx_p = self.Lx // 2
            self.verbose = verbose
            if self.verbose:
                print(f"Initializing Clover with parity decomposition:")
                print(f"  Lattice size: {latt_size} (x,y,z,t)")
                print(f"  Lattice x with parity: {self.Lx_p}")
                print(f"  Parameters: kappa={kappa}, u_0={u_0}")
                print(
                    f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
                print(f"  Device: {self.device}")

    class _mg:
        def __init__(self, solver, n_refine: int):
            self.solver = solver
            self.n_refine = n_refine
            self.device = solver.device
            self.dtype = solver.dtype
            self.blocksize = [2, 2, 2, 2]  # Compression rate per dimension
            # Degrees of freedom on coarse grids
            self.coarse_dof = [12, 12, 12]
            # Operators per level (share the same gauge field)
            self.mg_ops = [solver]
            self.R_null_vec = []  # Near-null vectors
            self.coarse_map = []  # Coarse-fine grid mapping
            self.fine_sites_per_coarse = []  # Fine sites per coarse block
            self._build_multigrid()

        def _build_multigrid(self):
            if self.solver.verbose:
                print(f"Building multigrid with {self.n_refine} levels...")
            Lx, Ly, Lz, Lt = self.solver.Lx, self.solver.Ly, self.solver.Lz, self.solver.Lt
            for i in range(self.n_refine):
                coarse_dof = self.coarse_dof[i]
                if self.solver.verbose:
                    print(
                        f"Level {i} coarse grid: size {Lx}x{Ly}x{Lz}x{Lt}, DOF {coarse_dof}")
                null_vec = torch.randn(coarse_dof, 4, 3, Lt, Lz, Ly, Lx,
                                       dtype=self.dtype, device=self.device)  # Not real null vectors......
                null_vec = self._orthogonalize_null_vec(null_vec, coarse_dof)
                print(f"null_vec.shape:{null_vec.shape}\n")
                print(
                    f"null_vec.reshape(coarse_dof, -1).shape:{null_vec.reshape(coarse_dof, -1).shape}\n")
                self.R_null_vec.append(null_vec.reshape(coarse_dof, -1))
                self.fine_sites_per_coarse.append(self.blocksize[0] * self.blocksize[1] *
                                                  self.blocksize[2] * self.blocksize[3] * 4 * 3)
                coarse_vol = Lx * Ly * Lz * Lt
                map_shape = (coarse_vol, self.fine_sites_per_coarse[i])
                self.coarse_map.append(torch.zeros(
                    map_shape, dtype=torch.int64, device=self.device))
                self._build_mapping(i, Lx, Ly, Lz, Lt)
                self.mg_ops.append(self.solver)
                Lx //= self.blocksize[0]
                Ly //= self.blocksize[1]
                Lz //= self.blocksize[2]
                Lt //= self.blocksize[3]

        def _orthogonalize_null_vec(self, vec: torch.Tensor, dof: int) -> torch.Tensor:
            """Orthogonalize the near-null space vectors"""
            for i in range(dof):
                for j in range(i):
                    proj = torch.vdot(vec[i].flatten(), vec[j].flatten(
                    )) / torch.vdot(vec[j].flatten(), vec[j].flatten())
                    vec[i] -= proj * vec[j]
                vec[i] /= torch.norm(vec[i])
            return vec

        def _build_mapping(self, level: int, Lx_coarse: int, Ly_coarse: int, Lz_coarse: int, Lt_coarse: int):
            """Build mapping from fine to coarse grid (4D)"""
            block = self.blocksize
            fine_Lx, fine_Ly, fine_Lz, fine_Lt = self.solver.Lx, self.solver.Ly, self.solver.Lz, self.solver.Lt
            sites_per_block = block[0] * block[1] * block[2] * block[3] * 4 * 3
            if self.coarse_map[level].shape[-1] != sites_per_block:
                if self.solver.verbose:
                    print(
                        f"Warning: mismatched mapping size, expected {sites_per_block}, got {self.coarse_map[level].shape[-1]}")
                self.coarse_map[level] = torch.zeros(self.coarse_map[level].shape[0],
                                                     sites_per_block,
                                                     dtype=torch.int64,
                                                     device=self.device)
            map_idx = 0
            for tc in range(Lt_coarse):
                for zc in range(Lz_coarse):
                    for yc in range(Ly_coarse):
                        for xc in range(Lx_coarse):
                            t_start, t_end = tc * block[0], (tc + 1) * block[0]
                            z_start, z_end = zc * block[1], (zc + 1) * block[1]
                            y_start, y_end = yc * block[2], (yc + 1) * block[2]
                            x_start, x_end = xc * block[3], (xc + 1) * block[3]
                            local_idx = 0
                            for s in range(4):
                                for c in range(3):
                                    for t in range(t_start, t_end):
                                        for z in range(z_start, z_end):
                                            for y in range(y_start, y_end):
                                                for x in range(x_start, x_end):
                                                    fine_idx = (s * 3 * fine_Lt * fine_Lz * fine_Ly * fine_Lx +
                                                                c * fine_Lt * fine_Lz * fine_Ly * fine_Lx +
                                                                t * fine_Lz * fine_Ly * fine_Lx +
                                                                z * fine_Ly * fine_Lx +
                                                                y * fine_Lx + x)
                                                    if local_idx < self.coarse_map[level].shape[-1]:
                                                        self.coarse_map[level][map_idx,
                                                                               local_idx] = fine_idx
                                                        local_idx += 1
                                                    else:
                                                        if self.solver.verbose:
                                                            print(
                                                                f"Warning: mapping index out of range, local_idx={local_idx}, max={self.coarse_map[level].shape[-1]-1}")
                            map_idx += 1

        def restrict(self, level: int, fine_vec: torch.Tensor) -> torch.Tensor:
            """Restriction operator: fine -> coarse"""
            coarse_dof = self.coarse_dof[level]
            # print(f"self.mg_ops:{self.mg_ops}\n")
            Lx, Ly, Lz, Lt = self.mg_ops[level + 1].Lx, self.mg_ops[level +
                                                                    1].Ly, self.mg_ops[level + 1].Lz, self.mg_ops[level + 1].Lt
            coarse_vec = torch.zeros(coarse_dof, 3, Lt, Lz, Ly, Lx,
                                     dtype=self.dtype, device=self.device)
            fine_flat = fine_vec.flatten()
            coarse_flat = coarse_vec.flatten()
            print(
                f"coarse_vec.shape,fine_vec.shape:{coarse_vec.shape,fine_vec.shape}\n")
            for i in range(coarse_vec.numel() // coarse_dof):
                for d in range(coarse_dof):
                    idx = i * coarse_dof + d
                    # print(
                    #     f"Lx, Ly, Lz, Lt, i,coarse_vec.numel(),coarse_dof,d:{Lx, Ly, Lz, Lt,i,coarse_vec.numel(),coarse_dof,d}\n")
                    valid_indices = self.coarse_map[level][i] < len(fine_flat)
                    valid_map = self.coarse_map[level][i, valid_indices]
                    if len(valid_map) > 0:
                        coarse_flat[idx] = torch.dot(self.R_null_vec[level][d, valid_map],
                                                     fine_flat[valid_map].conj())
            return coarse_vec

        def prolong(self, level: int, coarse_vec: torch.Tensor) -> torch.Tensor:
            """Prolongation operator: coarse -> fine"""
            fine_level = self.mg_ops[level]
            fine_vec = torch.zeros(4, 3, fine_level.Lt, fine_level.Lz, fine_level.Ly, fine_level.Lx,
                                   dtype=self.dtype, device=self.device)
            coarse_flat = coarse_vec.flatten()
            fine_flat = fine_vec.flatten()
            for i in range(coarse_vec.numel() // self.coarse_dof[level]):
                for d in range(self.coarse_dof[level]):
                    idx = i * self.coarse_dof[level] + d
                    valid_indices = self.coarse_map[level][i] < len(fine_flat)
                    valid_map = self.coarse_map[level][i, valid_indices]
                    if len(valid_map) > 0:
                        fine_flat[valid_map] += self.R_null_vec[level][d,
                                                                       valid_map] * coarse_flat[idx]
            return fine_vec

        def mg_solve(self, b: torch.Tensor, tol: float = 1e-8, max_iter: int = 300) -> torch.Tensor:
            """Multigrid solver (recursive BiCGSTAB)"""
            return self._mg_bicgstab(b, tol, max_iter, level=0)

        def _mg_bicgstab(self, b: torch.Tensor, tol: float, max_iter: int, level: int) -> torch.Tensor:
            solver = self.mg_ops[level]
            x = torch.zeros_like(b)
            r = b - solver.matvec(x)
            r0 = r.clone()
            p = r.clone()
            alpha = 1.0
            omega = 1.0
            count = 0
            while count < max_iter and torch.norm(r) > tol:
                Ap = solver.matvec(p)
                alpha = torch.vdot(r0.flatten(), r.flatten(
                )) / torch.vdot(r0.flatten(), Ap.flatten())
                x += alpha * p
                r1 = r - alpha * Ap
                if torch.norm(r1) < tol:
                    break
                if level < self.n_refine:
                    r_coarse = self.restrict(level, r1)
                    e_coarse = self._mg_bicgstab(
                        r_coarse, tol * 0.25, max_iter // 2, level + 1)
                    x += self.prolong(level, e_coarse)
                    r1 = b - solver.matvec(x)
                t = solver.matvec(r1)
                omega = torch.vdot(t.flatten(), r1.flatten(
                )) / torch.vdot(t.flatten(), t.flatten())
                x += omega * r1
                r = r1 - omega * t
                beta = (torch.vdot(r.flatten(), r0.flatten()) /
                        torch.vdot(r1.flatten(), r0.flatten())) * (alpha / omega)
                p = r + beta * (p - omega * Ap)
                count += 1
            if solver.verbose and level == 0:
                print(
                    f"Multigrid solve finished: {count} iterations, residual {torch.norm(r):.2e}")
            return x
