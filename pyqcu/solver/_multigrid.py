import torch
import numpy as np
from typing import Tuple
from pyqcu import _torch, tools, lattice, dslash
import mpi4py.MPI as MPI
from time import perf_counter


class hopping:
    def __init__(self,  U: torch.Tensor = None, if_multi: bool = lattice.give_if_multi(), kappa: float = 0.1, u_0: float = 1.0):
        self.M_plus_list = [torch.zeros([]), torch.zeros(
            []), torch.zeros([]), torch.zeros([])]  # xyzt
        self.M_minus_list = [torch.zeros([]), torch.zeros(
            []), torch.zeros([]), torch.zeros([])]  # xyzt
        self.U = U
        self.grid_size = tools.give_grid_size()
        self.grid_index = tools.give_grid_index()
        if self.U is not None:
            for ward in range(4):  # xyzt
                if if_multi and self.grid_size[ward] != 1:
                    comm = MPI.COMM_WORLD
                    rank = comm.Get_rank()
                    U_tail4send = self.U[tools.slice_dim(
                        dim=7, ward=ward, point=-1)].cpu().contiguous().numpy().copy()
                    U_head4recv = np.zeros_like(U_tail4send).copy()
                    rank_plus = tools.give_rank_plus(ward=ward)
                    rank_minus = tools.give_rank_minus(ward=ward)
                    comm.Sendrecv(sendbuf=U_tail4send, dest=rank_plus, sendtag=rank,
                                  recvbuf=U_head4recv, source=rank_minus, recvtag=rank_minus)
                    comm.Barrier()
                    U_head = torch.from_numpy(U_head4recv).to(
                        device=U.device).clone()
                else:
                    U_head = None
                self.M_plus_list[ward] = dslash.give_hopping_plus(
                    ward_key=lattice.ward_keys[ward], U=self.U, kappa=kappa, u_0=u_0)
                self.M_minus_list[ward] = dslash.give_hopping_minus(
                    ward_key=lattice.ward_keys[ward], U=self.U, U_head=U_head, kappa=kappa, u_0=u_0)

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
            src_head4send = src[tools.slice_dim(
                dim_num=5, ward=ward, point=0)].cpu().contiguous().numpy().copy()
            src_tail4recv = np.zeros_like(src_head4send).copy()
            rank_plus = tools.give_rank_plus(ward=ward)
            rank_minus = tools.give_rank_minus(ward=ward)
            comm.Sendrecv(sendbuf=src_head4send, dest=rank_minus, sendtag=rank_minus,
                          recvbuf=src_tail4recv, source=rank_plus, recvtag=rank)
            comm.Barrier()
            src_tail = torch.from_numpy(src_tail4recv).to(
                device=src.device).clone()
        else:
            src_tail = None
        return dslash.give_wilson_plus(ward_key=lattice.ward_keys[ward], src=src, hopping=self.M_plus_list[ward], src_tail=src_tail).to(dtype=dtype, device=device)

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
            src_tail4send = src[tools.slice_dim(
                dim_num=5, ward=ward, point=-1)].cpu().contiguous().numpy().copy()
            src_head4recv = np.zeros_like(src_tail4send).copy()
            rank_plus = tools.give_rank_plus(ward=ward)
            rank_minus = tools.give_rank_minus(ward=ward)
            comm.Sendrecv(sendbuf=src_tail4send, dest=rank_plus, sendtag=rank,
                          recvbuf=src_head4recv, source=rank_minus, recvtag=rank_minus)
            comm.Barrier()
            src_head = torch.from_numpy(src_head4recv).to(
                device=src.device).clone()
        else:
            src_head = None
        return dslash.give_wilson_minus(ward_key=lattice.ward_keys[ward], src=src, hopping=self.M_minus_list[ward], src_head=src_head).to(dtype=dtype, device=device)

    def matvec(self, src: torch.Tensor, if_multi: bool = lattice.give_if_multi()) -> torch.Tensor:
        dest = torch.zeros_like(src)
        for ward in range(4):
            dest += self.matvec_plus(ward=ward, src=src, if_multi=if_multi)
            dest += self.matvec_minus(ward=ward, src=src, if_multi=if_multi)
        return dest.clone()


class sitting:
    def __init__(self, clover_term: torch.Tensor = None):
        self.M = torch.zeros([])
        self.clover_term = clover_term
        if self.clover_term is not None:  # remmber to add I
            self.M = dslash.add_I(clover_term=self.clover_term).reshape(
                [12, 12]+list(self.clover_term.shape[-4:])).clone()  # A = I + T

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        dtype = src.dtype
        device = src.device
        _dtype = self.M.dtype
        _device = self.M.device
        if dtype != _dtype or device != _device:
            src = src.to(dtype=_dtype, device=_device)
        return _torch.einsum(
            "EeXYZT, eXYZT->EXYZT", self.M, src).clone().to(dtype=dtype, device=device)


class op:
    def __init__(self,  U: torch.Tensor = None, clover_term: torch.Tensor = None, fine_hopping: hopping = None, fine_sitting: sitting = None, local_ortho_null_vecs: torch.Tensor = None, kappa: float = 0.1, u_0: float = 1.0, if_multi: bool = lattice.give_if_multi(), verbose: bool = True):
        self.if_multi = if_multi
        self.hopping = hopping(U=U, kappa=kappa, u_0=u_0)
        self.sitting = sitting(clover_term=clover_term)
        self.verbose = verbose
        if fine_hopping is not None and fine_sitting is not None and local_ortho_null_vecs is not None:
            shape = local_ortho_null_vecs.shape  # EeXxYyZzTt
            coarse_shape = [shape[-8], shape[-6],
                            shape[-4], shape[-2]]  # XYZT
            coarse_dof = shape[0]  # E
            self.sitting.M = torch.zeros(
                size=[coarse_dof, coarse_dof]+coarse_shape, dtype=local_ortho_null_vecs.dtype, device=local_ortho_null_vecs.device)  # EEXYZT
            for ward in range(4):  # xyzt
                self.hopping.M_plus_list[ward] = torch.zeros_like(
                    self.sitting.M)
                self.hopping.M_minus_list[ward] = torch.zeros_like(
                    self.sitting.M)
            for e in range(coarse_dof):
                for ward in range(4):  # xyzt
                    # give partly sitting.ee and whole hopping.oe
                    _src_c = torch.zeros_like(self.sitting.M[0])
                    _src_c[e][tools.slice_dim(ward=ward, start=0)] = 1.0
                    _src_f = tools.prolong(
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                    _dest_f_plus = fine_hopping.matvec_plus(
                        ward=ward, src=_src_f, if_multi=self.if_multi)
                    _dest_f_minus = fine_hopping.matvec_minus(
                        ward=ward, src=_src_f, if_multi=self.if_multi)
                    _dest_c_plus = tools.restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_plus, verbose=self.verbose)
                    _dest_c_minus = tools.restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_minus, verbose=self.verbose)
                    self.sitting.M[:, e][tools.slice_dim(dims_num=5,
                                                         ward=ward, start=0)] += _dest_c_plus[tools.slice_dim(dims_num=5, ward=ward, start=0)].clone()
                    self.sitting.M[:, e][tools.slice_dim(dims_num=5,
                                                         ward=ward, start=0)] += _dest_c_minus[tools.slice_dim(dims_num=5, ward=ward, start=0)].clone()
                    self.hopping.M_plus_list[ward][:, e][tools.slice_dim(dims_num=5,
                                                                         ward=ward, start=1)] = _dest_c_plus[tools.slice_dim(dims_num=5, ward=ward, start=1)].clone()
                    self.hopping.M_minus_list[ward][:, e][tools.slice_dim(dims_num=5,
                                                                          ward=ward, start=1)] = _dest_c_minus[tools.slice_dim(dims_num=5, ward=ward, start=1)].clone()
                    # give partly sitting.oo and whole hopping.eo
                    _src_c = torch.zeros_like(self.sitting.M[0])
                    _src_c[e][tools.slice_dim(ward=ward, start=1)] = 1.0
                    _src_f = tools.prolong(
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                    _dest_f_plus = fine_hopping.matvec_plus(
                        ward=ward, src=_src_f, if_multi=self.if_multi)
                    _dest_f_minus = fine_hopping.matvec_minus(
                        ward=ward, src=_src_f, if_multi=self.if_multi)
                    _dest_c_plus = tools.restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_plus, verbose=self.verbose)
                    _dest_c_minus = tools.restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_minus, verbose=self.verbose)
                    self.sitting.M[:, e][tools.slice_dim(dims_num=5,
                                                         ward=ward, start=1)] += _dest_c_plus[tools.slice_dim(dims_num=5, ward=ward, start=1)].clone()
                    self.sitting.M[:, e][tools.slice_dim(dims_num=5,
                                                         ward=ward, start=1)] += _dest_c_minus[tools.slice_dim(dims_num=5, ward=ward, start=1)].clone()
                    self.hopping.M_plus_list[ward][:, e][tools.slice_dim(dims_num=5,
                                                                         ward=ward, start=0)] = _dest_c_plus[tools.slice_dim(dims_num=5, ward=ward, start=0)].clone()
                    self.hopping.M_minus_list[ward][:, e][tools.slice_dim(dims_num=5,
                                                                          ward=ward, start=0)] = _dest_c_minus[tools.slice_dim(dims_num=5, ward=ward, start=0)].clone()
                # give aother partly sitting.ee and sitting.oo
                _src_c = torch.zeros_like(self.sitting.M[0])
                _src_c[e] = 1.0
                _src_f = tools.prolong(
                    local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c, verbose=self.verbose)
                _dest_f = fine_sitting.matvec(src=_src_f)
                _dest_c = tools.restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f, verbose=self.verbose)
                self.sitting.M[:, e] += _dest_c.clone()

    def matvec(self, src: torch.Tensor, if_multi: bool = lattice.give_if_multi()) -> torch.Tensor:
        if src.shape[0] == 4 and src.shape[1] == 3:
            return (self.hopping.matvec(src=src.reshape([12]+list(src.shape)[2:]), if_multi=self.if_multi and if_multi)+self.sitting.matvec(src=src.reshape([12]+list(src.shape)[2:]))).reshape([4, 3]+list(src.shape)[2:])
        else:
            return self.hopping.matvec(src=src, if_multi=self.if_multi and if_multi)+self.sitting.matvec(src=src)


class multigrid:
    def __init__(self, dtype_list: Tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype], device_list: Tuple[torch.device, torch.device, torch.device, torch.device],  U: torch.Tensor, clover_term: torch.Tensor, kappa: float = 0.1, u_0: float = 1.0, min_size: int = 2, max_levels: int = 4, mg_grid_size: Tuple[int, int, int, int] = [2, 2, 2, 2], num_convergence_sample: int = 50, dof_list: Tuple[int, int, int, int] = [12, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24], mass: float = -0.05, tol: float = 1e-6, max_iter: int = 1000, root: int = 0, verbose: bool = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.lat_size = list(U.shape[-4:])  # xyzt
        self.min_size = min_size
        self.max_levels = max_levels
        self.mass = mass  # just for plot......
        self.kappa = kappa
        self.u_0 = u_0
        self.tol = tol
        self.max_iter = max_iter
        self.root = root
        self.verbose = verbose
        self.dtype_list = dtype_list[:max_levels]
        self.device_list = device_list[:max_levels]
        self.dof_list = dof_list
        if self.rank == self.root:
            print(f"PYQCU::SOLVER::MULTIGRID:\n self.dof_list:{self.dof_list}")
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n self.dtype_list:{self.dtype_list}")
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n self.device_list:{self.device_list}")
        for device in self.device_list:
            tools.set_device(device=device)
        self.op_list = [op(U=U,
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
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n self.lat_size_list:{self.lat_size_list}")
        self.dof_list = self.dof_list[:self.num_levels]

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
        r_norm = _torch.norm(r)
        _tol = r_norm*0.5 if level != self.num_levels - 1 else r_norm*0.1
        if self.verbose:
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n {level}:Norm of b:{_torch.norm(b)}")
            print(f"PYQCU::SOLVER::MULTIGRID:\n {level}:Norm of r:{r_norm}")
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n {level}:Norm of x0:{_torch.norm(x)}")
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
        for i in range(self.max_iter):
            iter_start_time = perf_counter()
            rho = _torch.dot(r_tilde, r)
            beta = (rho / rho_prev) * (alpha / omega)
            rho_prev = rho
            p = r + beta * (p - omega * v)
            v = matvec(p)
            alpha = rho / _torch.dot(r_tilde, v)
            s = r - alpha * v
            t = matvec(s)
            omega = _torch.dot(t, s) / _torch.dot(t, t)
            x = x + alpha * p + omega * s
            r = s - omega * t
            r_norm = _torch.norm(r)
            if level == 0:
                self.convergence_history.append(r_norm)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"PYQCU::SOLVER::MULTIGRID:\n B-{level}-BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}")
            # cycle start
            if level < self.num_levels-1:
                r_coarse = tools.restrict(
                    local_ortho_null_vecs=self.lonv_list[level], fine_vec=r, verbose=self.verbose)
                self.b_list[level+1] = r_coarse.clone().to(dtype=self.dtype_list[level+1],
                                                           device=self.device_list[level+1])
                e_coarse = self.cycle(level=level+1).to(dtype=self.dtype_list[level],
                                                        device=self.device_list[level])
                e_fine = tools.prolong(
                    local_ortho_null_vecs=self.lonv_list[level], coarse_vec=e_coarse, verbose=self.verbose)
                x = x + e_fine
                r = b - matvec(x)
            r_norm = _torch.norm(r)
            if level == 0:
                self.convergence_history.append(r_norm)
                self.adaptive(iter=i)
            # cycle end
            iter_time = perf_counter() - iter_start_time
            iter_times.append(iter_time)
            if self.verbose:
                # print(f"alpha,beta,omega:{alpha,beta,omega}\n")
                print(
                    f"PYQCU::SOLVER::MULTIGRID:\n F-{level}-BICGSTAB-Iteration {i}: Residual = {r_norm:.6e}, Time = {iter_time:.6f} s")
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
