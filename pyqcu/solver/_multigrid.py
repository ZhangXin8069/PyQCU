from itertools import count
from pdb import Restart
import torch
from typing import Tuple
from pyqcu import _torch, tools, dslash
import mpi4py.MPI as MPI
from time import perf_counter


class multigrid:
    def __init__(self, dtype_list: Tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype], device_list: Tuple[torch.device, torch.device, torch.device, torch.device],  U: torch.Tensor, clover_term: torch.Tensor, kappa: float = 0.1, u_0: float = 1.0, min_size: int = 2, max_levels: int = 4, mg_grid_size: Tuple[int, int, int, int] = [2, 2, 2, 2], num_convergence_sample: int = 50, dof_list: Tuple[int, int, int, int] = [12, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24], tol: float = 1e-6, max_iter: int = 1000, num_restart: int = 3, root: int = 0, verbose: bool = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.lat_size = list(U.shape[-4:])  # xyzt
        self.min_size = min_size
        self.max_levels = max_levels
        self.kappa = kappa
        self.mass = (1/kappa - 8)/2  # just for plot......
        self.u_0 = u_0
        self.tol = tol
        self.max_iter = max_iter
        self.num_restart = num_restart
        self.root = root
        self.verbose = verbose
        # Build grid list
        self.lat_size_list = []
        self.mg_grid_size = mg_grid_size
        _lat_size = self.lat_size
        while all(_ >= self.min_size for _ in _lat_size) and len(self.lat_size_list) < self.max_levels:
            self.lat_size_list.append(_lat_size)
            _lat_size = [_lat_size[d] // self.mg_grid_size[d]
                         for d in range(4)]
        self.num_levels = len(self.lat_size_list)
        self.dof_list = dof_list[:self.num_levels]
        self.dtype_list = dtype_list[:self.num_levels]
        self.device_list = device_list[:self.num_levels]
        if self.rank == self.root:
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n self.dof_list:{self.dof_list}")
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n self.dtype_list:{self.dtype_list}")
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n self.device_list:{self.device_list}")
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n self.lat_size_list:{self.lat_size_list}")
        for device in self.device_list:
            tools.set_device(device=device)
        self.op_list = [dslash.operator(U=U,
                                        clover_term=clover_term, verbose=self.verbose, kappa=kappa, u_0=u_0)]
        self.b = _torch.randn(size=[12]+self.lat_size,
                              dtype=self.dtype_list[0], device=self.device_list[0])
        self.x0 = _torch.randn(
            size=[12]+self.lat_size, dtype=self.dtype_list[0], device=self.device_list[0])
        self.b_list = [self.b.clone()]
        self.nv_list = []  # null_vecs_list
        self.lonv_list = []  # local_ortho_null_vecs_list
        self.num_convergence_sample = num_convergence_sample
        self.convergence_history = []
        self.convergence_tol = 0

    def init(self):
        # Build local-orthonormal near-null space vectors
        comm = MPI.COMM_WORLD
        comm.Barrier()
        for i in range(1, len(self.lat_size_list)):
            _null_vecs = _torch.randn(size=[self.dof_list[i], self.dof_list[i-1]] +
                                      self.lat_size_list[i-1], dtype=self.dtype_list[i-1], device=self.device_list[i-1])
            _null_vecs = tools.give_null_vecs(
                null_vecs=_null_vecs,
                matvec=self.op_list[i-1].matvec,
                verbose=self.verbose).to(dtype=self.dtype_list[i], device=self.device_list[i])
            self.nv_list.append(_null_vecs)
            _local_ortho_null_vecs = tools.local_orthogonalize(
                null_vecs=_null_vecs,
                coarse_lat_size=self.lat_size_list[i],
                verbose=self.verbose)
            self.lonv_list.append(_local_ortho_null_vecs)
            self.b_list.append(torch.zeros(
                size=[self.dof_list[i]]+self.lat_size_list[i], dtype=self.dtype_list[i], device=self.device_list[i]))
            self.op_list.append(dslash.operator(fine_hopping=self.op_list[i-1].hopping, fine_sitting=self.op_list[i -
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
        r_norm = tools.norm(r)
        _tol = r_norm*0.5 if level != self.num_levels - 1 else r_norm*0.1
        if self.verbose:
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n {level}:Norm of b:{tools.norm(b)}")
            print(f"PYQCU::SOLVER::MULTIGRID:\n {level}:Norm of r:{r_norm}")
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n {level}:Norm of x0:{tools.norm(x)}")
        if level == 0:
            self.convergence_history.append(r_norm)
            _tol = self.tol
        if r_norm < _tol:
            print("PYQCU::SOLVER::MULTIGRID:\n x0 is just right!")
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
        count_restart = 0
        for i in range(self.max_iter):
            iter_start_time = perf_counter()
            rho = tools.vdot(r_tilde, r)
            beta = (rho / rho_prev) * (alpha / omega)
            rho_prev = rho
            p = r + beta * (p - omega * v)
            v = matvec(p)
            alpha = rho / tools.vdot(r_tilde, v)
            s = r - alpha * v
            t = matvec(s)
            omega = tools.vdot(t, s) / tools.vdot(t, t)
            x = x + alpha * p + omega * s
            r = s - omega * t
            r_norm = tools.norm(r)
            if level == 0:
                self.convergence_history.append(r_norm)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"PYQCU::SOLVER::MULTIGRID:\n B-{level}-bistabcg-Iteration {i}: Residual = {r_norm:.6e}")
            count_restart += 1
            # cycle start
            if level < self.num_levels-1 and count_restart > self.num_restart:
                r_coarse = tools.restrict(
                    local_ortho_null_vecs=self.lonv_list[level], fine_vec=r)
                self.b_list[level+1] = r_coarse.clone().to(dtype=self.dtype_list[level+1],
                                                           device=self.device_list[level+1])
                e_coarse = self.cycle(level=level+1).to(dtype=self.dtype_list[level],
                                                        device=self.device_list[level])
                e_fine = tools.prolong(
                    local_ortho_null_vecs=self.lonv_list[level], coarse_vec=e_coarse)
                x = x + e_fine
                r = b - matvec(x)
                count_restart = 0
            r_norm = tools.norm(r)
            if level == 0:
                self.convergence_history.append(r_norm)
                self.adaptive(iter=i)
            # cycle end
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"PYQCU::SOLVER::MULTIGRID:\n F-{level}-bistabcg-Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
            if r_norm < _tol:
                if self.verbose:
                    print(
                        f"PYQCU::SOLVER::MULTIGRID:\n Converged at iteration {i} with residual {r_norm:.6e}")
                break
        else:
            print(
                "PYQCU::SOLVER::MULTIGRID:\n Warning: Maximum iterations reached, may not have converged")
        total_time = perf_counter() - start_time
        avg_iter_time = sum(iter_times) / len(iter_times)
        print(f"PYQCU::SOLVER::MULTIGRID:\n Performance Statistics:")
        print(
            f"PYQCU::SOLVER::MULTIGRID:\n Total iterations: {len(iter_times)}")
        print(
            f"PYQCU::SOLVER::MULTIGRID:\n Total time: {total_time:.6f} seconds")
        print(
            f"PYQCU::SOLVER::MULTIGRID:\n Average time per iteration: {avg_iter_time:.6f} s")
        print(f"PYQCU::SOLVER::MULTIGRID:\n Final residual: {r_norm:.2e}")
        return x.clone()

    def solve(self, b: torch.Tensor = None, x0: torch.Tensor = None) -> torch.Tensor:
        if b is not None:
            self.b = b.reshape([12]+list(b.shape)[2:]).clone()  # sc->e
            self.b_list[0] = self.b.clone()
        if x0 is not None:
            self.x0 = x0.reshape([12]+list(x0.shape)[2:]).clone()  # sc->e
        self.levels_back()
        start_time = perf_counter()
        x = self.cycle()
        total_time = perf_counter() - start_time
        print(f"PYQCU::SOLVER::MULTIGRID:\n Performance Statistics:")
        print(
            f"PYQCU::SOLVER::MULTIGRID:\n Total time: {total_time:.6f} seconds")
        print(
            f"PYQCU::SOLVER::MULTIGRID:\n Final residual: {self.convergence_history[-1]:.2e}")
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
                f"PYQCU::SOLVER::MULTIGRID:\n convergence_history(mass={self.mass})\n self.dof_list:{self.dof_list}\n self.lat_size_list:{self.lat_size_list}\n self.dtype_list:{self.dtype_list}\n self.device_list:{self.device_list}", fontsize=12)
            plt.semilogy(range(1, len(self.convergence_history) + 1),
                         self.convergence_history, 'b-o', markersize=4, linewidth=2)
            plt.xlabel(
                f"Iteration*2(record two r_norm between x changes in one iteration)", fontsize=12)
            plt.ylabel('Residual Norm', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path is None:
                save_path = "convergence_history.png"
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close()
