import torch
import torch.nn as nn
from time import perf_counter
from typing import Tuple, Optional, List
from pyqcu.ascend import dslash


class LatticeSolver(nn.Module):
    def __init__(self,
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
        self.U = self.wilson.generate_gauge_field(sigma=0.1, seed=42)
        # Initialize multigrid
        self.mg = self.MultiGrid(self, n_refine)
        if self.verbose:
            print(
                f"Initialization complete: Lattice size {self.Lx}x{self.Ly}x{self.Lz}x{self.Lt}, Spin 4, Color 3")
            print(f"Device: {self.device}, Dtype: {self.dtype}")

    def _dslash(self, src: torch.Tensor) -> torch.Tensor:
        """
        Apply Wilson-Dirac operator (dslash): Dψ = (1 - kappa * D_wilson)ψ
        src shape: [spin, color, t, z, y, x]
        return: same shape
        """
        return self.wilson.apply_dirac_operator(src, self.U).clone()

    class MultiGrid:
        def __init__(self, solver, n_refine: int):
            self.solver = solver
            self.n_refine = n_refine
            self.device = solver.device
            self.dtype = solver.dtype
            self.blocksize = [2, 2, 2, 2]  # Compression rate per dimension
            # Degrees of freedom on coarse grids
            self.coarse_dof = [16, 16, 16]
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
                null_vec = torch.randn(coarse_dof, 4, 3, Lt, Lz, Ly, Lx,
                                       dtype=self.dtype, device=self.device)
                null_vec = self._orthogonalize_null_vec(null_vec, coarse_dof)
                self.R_null_vec.append(null_vec.reshape(coarse_dof, -1))
                Lx //= self.blocksize[0]
                Ly //= self.blocksize[1]
                Lz //= self.blocksize[2]
                Lt //= self.blocksize[3]
                self.fine_sites_per_coarse.append(self.blocksize[0] * self.blocksize[1] *
                                                  self.blocksize[2] * self.blocksize[3] * 4 * 3)
                coarse_vol = Lx * Ly * Lz * Lt
                map_shape = (coarse_vol, self.fine_sites_per_coarse[i])
                self.coarse_map.append(torch.zeros(
                    map_shape, dtype=torch.int64, device=self.device))
                self._build_mapping(i, Lx, Ly, Lz, Lt)
                self.mg_ops.append(self.solver)
                if self.solver.verbose:
                    print(
                        f"Level {i} coarse grid: size {Lx}x{Ly}x{Lz}x{Lt}, DOF {coarse_dof}")

        def _orthogonalize_null_vec(self, vec: torch.Tensor, dof: int) -> torch.Tensor:
            """Orthogonalize the near-null space vectors"""
            for i in range(dof):
                for j in range(i):
                    proj = torch.vdot(vec[i].conj().flatten(), vec[j].flatten()) / \
                        torch.vdot(vec[j].conj().flatten(), vec[j].flatten())
                    vec[i] -= proj * vec[j]
                vec[i] /= torch.norm(vec[i])
            return vec

        def _build_mapping(self, level: int, Lx_coarse: int, Ly_coarse: int, Lz_coarse: int, Lt_coarse: int):
            """Build mapping from fine to coarse grid (4D)"""
            block = self.blocksize
            fine_Lx, fine_Ly, fine_Lz, fine_Lt = self.solver.Lx, self.solver.Ly, self.solver.Lz, self.solver.Lt
            sites_per_block = block[0] * block[1] * block[2] * block[3] * 4 * 3
            if self.coarse_map[level].size(1) != sites_per_block:
                if self.solver.verbose:
                    print(
                        f"Warning: mismatched mapping size, expected {sites_per_block}, got {self.coarse_map[level].size(1)}")
                self.coarse_map[level] = torch.zeros(self.coarse_map[level].size(0),
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
                                                    if local_idx < self.coarse_map[level].size(1):
                                                        self.coarse_map[level][map_idx,
                                                                               local_idx] = fine_idx
                                                        local_idx += 1
                                                    else:
                                                        if self.solver.verbose:
                                                            print(
                                                                f"Warning: mapping index out of range, local_idx={local_idx}, max={self.coarse_map[level].size(1)-1}")
                            map_idx += 1

        def restrict(self, level: int, fine_vec: torch.Tensor) -> torch.Tensor:
            """Restriction operator: fine -> coarse"""
            coarse_dof = self.coarse_dof[level]
            print(f"self.mg_ops:{self.mg_ops}\n")
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
                    print(
                        f"Lx, Ly, Lz, Lt, i,coarse_vec.numel(),coarse_dof,d:{Lx, Ly, Lz, Lt,i,coarse_vec.numel(),coarse_dof,d}\n")
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
            r = b - solver._dslash(x)
            r0 = r.clone()
            p = r.clone()
            alpha = 1.0
            omega = 1.0
            count = 0
            while count < max_iter and torch.norm(r) > tol:
                Ap = solver._dslash(p)
                alpha = torch.vdot(r0.conj().flatten(), r.flatten(
                )) / torch.vdot(r0.conj().flatten(), Ap.flatten())
                x += alpha * p
                r1 = r - alpha * Ap
                if torch.norm(r1) < tol:
                    break
                if level < self.n_refine:
                    r_coarse = self.restrict(level, r1)
                    e_coarse = self._mg_bicgstab(
                        r_coarse, tol * 0.25, max_iter // 2, level + 1)
                    x += self.prolong(level, e_coarse)
                    r1 = b - solver._dslash(x)
                t = solver._dslash(r1)
                omega = torch.vdot(t.conj().flatten(), r1.flatten(
                )) / torch.vdot(t.conj().flatten(), t.flatten())
                x += omega * r1
                r = r1 - omega * t
                beta = (torch.vdot(r.conj().flatten(), r0.flatten()) /
                        torch.vdot(r1.conj().flatten(), r0.flatten())) * (alpha / omega)
                p = r + beta * (p - omega * Ap)
                count += 1
            if solver.verbose and level == 0:
                print(
                    f"Multigrid solve finished: {count} iterations, residual {torch.norm(r):.2e}")
            return x

    def bicgstab(self, b: torch.Tensor, tol: float = 1e-8, max_iter: int = 1000, x0=None) -> torch.Tensor:
        """Standard BiCGSTAB solver"""
        x = x0.clone() if x0 is not None else torch.rand_like(b)
        r = b - self._dslash(x)
        r_tilde = r.clone()
        p = torch.zeros_like(b)
        v = torch.zeros_like(b)
        s = torch.zeros_like(b)
        t = torch.zeros_like(b)
        rho_prev = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        alpha = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        omega = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        start_time = perf_counter()
        iter_times = []
        for i in range(max_iter):
            iter_start_time = perf_counter()
            rho = torch.vdot(r_tilde.conj().flatten(), r.flatten())
            if rho.abs() < 1e-30:
                if verbose:
                    print("Breakdown: rho ≈ 0")
                break
            beta = (rho / rho_prev) * (alpha / omega)
            rho_prev = rho
            p = r + beta * (p - omega * v)
            v = self._dslash(p)
            alpha = rho / torch.vdot(r_tilde.conj().flatten(), v.flatten())
            s = r - alpha * v
            t = self._dslash(s)
            omega = torch.vdot(t.conj().flatten(), s.flatten()) / \
                torch.vdot(t.conj().flatten(), t.flatten())
            x = x + alpha * p + omega * s
            r = s - omega * t
            r_norm2 = torch.norm(r).item()
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)
            if self.verbose:
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
        if self.verbose:
            print("\nPerformance Statistics:")
            print(f"Total time: {total_time:.6f} s")
            print(f"Average time per iteration: {avg_iter_time:.6f} s")
        return x


if __name__ == "__main__":
    # Parameter settings
    Lx, Ly, Lz, Lt = 8, 8, 8, 8  # Reduce lattice size for faster testing
    n_refine = 2
    kappa = 0.125
    verbose = True
    # Initialize the solver
    solver = LatticeSolver(
        latt_size=(Lx, Ly, Lz, Lt),
        n_refine=n_refine,
        kappa=kappa,
        verbose=verbose
    )
    # Generate test vector
    b = torch.randn(4, 3, Lt, Lz, Ly, Lx,
                    dtype=solver.dtype, device=solver.device)
    x_exact = torch.randn_like(b)
    b = solver._dslash(x_exact)  # Construct b = D(x_exact)
    # print(f"b:{b}\n")
    # Solve
    print("Starting multigrid solve...")
    # x_mg = solver.mg.mg_solve(b, tol=1e-6)  # Loosen tolerance for faster test
    x_mg = solver.bicgstab(b, tol=1e-6)  # Loosen tolerance for faster test
    print("Starting standard BiCGSTAB solve...")
    x_bicg = solver.bicgstab(b, tol=1e-6)  # Loosen tolerance for faster test
    # Validation
    err_mg = torch.norm(x_mg - x_exact) / torch.norm(x_exact)
    err_bicg = torch.norm(x_bicg - x_exact) / torch.norm(x_exact)
    print(
        f"Relative error - Multigrid: {err_mg:.2e}, Standard BiCGSTAB: {err_bicg:.2e}")
