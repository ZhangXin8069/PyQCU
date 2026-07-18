import torch
from typing import List, Optional
from pyqcu import tools, dslash
import pyqcu.cann as _torch
import mpi4py.MPI as MPI
from time import perf_counter
from pyqcu.cuda import qcu, define


class multigrid:
    def __init__(self, dtype_list: List[torch.dtype], device_list: List[torch.device],  U: torch.Tensor, clover_term: torch.Tensor, kappa: Optional[torch.Tensor] = torch.Tensor([0.1]), u_0: Optional[torch.Tensor] = torch.Tensor([1.0]), clover_ee_inv: Optional[torch.Tensor] = None, clover_oo_inv: Optional[torch.Tensor] = None, min_size: int = 4, max_level: int = 4, mg_grid_size: List[int] = [2, 2, 2, 2], num_convergence_sample: int = 50, dof_list: List[int] = [12, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24], tol: float = 1e-6, max_iter: int = 1000, num_restart: int = 5, root: int = 0, support_parity: bool = False, verbose: bool = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.lat_size = list(U.shape[-4:])  # xyzt
        self.min_size = min_size
        self.max_level = max_level
        self.kappa = kappa if kappa is not None else torch.Tensor([0.1])
        self.clover_ee_inv = clover_ee_inv
        self.clover_oo_inv = clover_oo_inv
        self.mass = (1/self.kappa - 8)/2  # just for plot......
        self.u_0 = u_0
        self.tol = tol
        self.max_iter = max_iter
        self.num_restart = num_restart
        self.root = root
        self.with_cuda_qcu = True if self.clover_ee_inv is not None and self.clover_oo_inv is not None else False
        self.support_parity = support_parity
        self.verbose = verbose
        # Build grid list
        self.lat_size_list = []
        self.mg_grid_size = mg_grid_size
        _lat_size = self.lat_size
        while all(_ >= self.min_size for _ in _lat_size) and len(self.lat_size_list) < self.max_level:
            self.lat_size_list.append(_lat_size)
            _lat_size = [_lat_size[d] // self.mg_grid_size[d]
                         for d in range(4)]
        self.num_level = len(self.lat_size_list)
        self.dof_list = dof_list[:self.num_level]
        self.dtype_list = dtype_list[:self.num_level]
        self.device_list = device_list[:self.num_level]
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
                                        clover_term=clover_term, verbose=self.verbose, kappa=self.kappa, u_0=self.u_0, support_parity=self.support_parity, clover_ee_inv=self.clover_ee_inv, clover_oo_inv=self.clover_oo_inv,)]
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
        # Pre-allocated CUDA tensors for restrict/prolong (level 0 -> 1)
        self._mg_restrict_out = None
        self._mg_prolong_out = None
        self._mg_coarse_dslash_out = None
        self._mg_coarse_hopping_packed = None
        self._mg_coarse_hopping_packed_level = None  # track which level is cached

    def _restrict_cuda(self, fine_vec: torch.Tensor, level: int) -> torch.Tensor:
        """CUDA-accelerated restriction using C++ backend."""
        lonv = self.lonv_list[level]
        E = lonv.shape[0]
        e = fine_vec.shape[0]
        Xf, Yf, Zf, Tf = fine_vec.shape[1:]
        Xc, Yc, Zc, Tc = self.lat_size_list[level+1]
        # Set multigrid params
        self.params[define._LAT_X_] = Xf
        self.params[define._LAT_Y_] = Yf
        self.params[define._LAT_Z_] = Zf
        self.params[define._LAT_T_] = Tf
        self.params[define._MG_LEVEL1_X_] = Xc
        self.params[define._MG_LEVEL1_Y_] = Yc
        self.params[define._MG_LEVEL1_Z_] = Zc
        self.params[define._MG_LEVEL1_T_] = Tc
        self.params[define._MG_LEVEL1_E_] = E
        self.params[define._MG_NUM_LEVEL_] = e  # fine DOF
        # Allocate or reuse output tensor
        if self._mg_restrict_out is None or self._mg_restrict_out.shape != torch.Size([E, Xc, Yc, Zc, Tc]):
            self._mg_restrict_out = torch.zeros(
                [E, Xc, Yc, Zc, Tc], dtype=fine_vec.dtype, device=fine_vec.device)
        qcu.applyMultigridRestrictQcu(
            self._mg_restrict_out, fine_vec, lonv, self.set_ptrs, self.params)
        return self._mg_restrict_out

    def _prolong_cuda(self, coarse_vec: torch.Tensor, level: int) -> torch.Tensor:
        """CUDA-accelerated prolongation using C++ backend."""
        lonv = self.lonv_list[level]
        E = coarse_vec.shape[0]
        e = lonv.shape[1]
        Xc, Yc, Zc, Tc = coarse_vec.shape[1:]
        Xf = Xc * self.mg_grid_size[0]
        Yf = Yc * self.mg_grid_size[1]
        Zf = Zc * self.mg_grid_size[2]
        Tf = Tc * self.mg_grid_size[3]
        # Set multigrid params
        self.params[define._LAT_X_] = Xf
        self.params[define._LAT_Y_] = Yf
        self.params[define._LAT_Z_] = Zf
        self.params[define._LAT_T_] = Tf
        self.params[define._MG_LEVEL1_X_] = Xc
        self.params[define._MG_LEVEL1_Y_] = Yc
        self.params[define._MG_LEVEL1_Z_] = Zc
        self.params[define._MG_LEVEL1_T_] = Tc
        self.params[define._MG_LEVEL1_E_] = E
        self.params[define._MG_NUM_LEVEL_] = e  # fine DOF
        # Allocate or reuse output tensor
        if self._mg_prolong_out is None or self._mg_prolong_out.shape != torch.Size([e, Xf, Yf, Zf, Tf]):
            self._mg_prolong_out = torch.zeros(
                [e, Xf, Yf, Zf, Tf], dtype=coarse_vec.dtype, device=coarse_vec.device)
        qcu.applyMultigridProLongQcu(
            self._mg_prolong_out, coarse_vec, lonv, self.set_ptrs, self.params)
        return self._mg_prolong_out

    def _coarse_dslash_cuda(self, src: torch.Tensor, level: int) -> torch.Tensor:
        """CUDA-accelerated coarse-grid dslash using C++ backend.

        Applies the coarse-grid Dirac operator D_coarse = sitting + hopping_plus + hopping_minus.
        Data layout:
          - fermion:  [E, Xc, Yc, Zc, Tc]
          - hopping:  [2, 4, E, E, Xc, Yc, Zc, Tc] (pm × dir × Eout × Ein × XYZT)
          - sitting:  [E, E, Xc, Yc, Zc, Tc]
        """
        op = self.op_list[level]
        E, Xc, Yc, Zc, Tc = src.shape
        # Ensure we use the base LatticeSet (index 0) for the CUDA stream
        self.params[define._SET_INDEX_] = 0
        # Pack hopping matrices: [2, 4, E, E, Xc, Yc, Zc, Tc]
        # dim0: 0=plus, 1=minus; dim1: 0=X, 1=Y, 2=Z, 3=T
        target_shape = torch.Size([2, 4, E, E, Xc, Yc, Zc, Tc])
        if (self._mg_coarse_hopping_packed is None or
                self._mg_coarse_hopping_packed.shape != target_shape or
                self._mg_coarse_hopping_packed.device != src.device or
                self._mg_coarse_hopping_packed_level != level):
            hopping_packed = torch.zeros(
                target_shape, dtype=src.dtype, device=src.device)
            for ward in range(4):
                hopping_packed[0, ward] = op.hopping.M_plus_list[ward].to(
                    dtype=src.dtype, device=src.device)
                hopping_packed[1, ward] = op.hopping.M_minus_list[ward].to(
                    dtype=src.dtype, device=src.device)
            self._mg_coarse_hopping_packed = hopping_packed
            self._mg_coarse_hopping_packed_level = level
        # Set multigrid params for the C++ backend
        self.params[define._MG_LEVEL1_X_] = Xc
        self.params[define._MG_LEVEL1_Y_] = Yc
        self.params[define._MG_LEVEL1_Z_] = Zc
        self.params[define._MG_LEVEL1_T_] = Tc
        self.params[define._MG_NUM_LEVEL_] = E  # coarse DOF (repurposed)
        # Ensure sitting tensor is on the correct device and dtype
        sitting = op.sitting.M.to(dtype=src.dtype, device=src.device)
        # Allocate or reuse output tensor
        if (self._mg_coarse_dslash_out is None or
                self._mg_coarse_dslash_out.shape != src.shape or
                self._mg_coarse_dslash_out.device != src.device):
            self._mg_coarse_dslash_out = torch.zeros_like(src)
        qcu.applyMultigridCoarseDslashQcu(
            self._mg_coarse_dslash_out, src,
            self._mg_coarse_hopping_packed, sitting,
            self.set_ptrs, self.params)
        return self._mg_coarse_dslash_out

    def _verify_coarse_dslash(self, level: int = 1, tol: float = 1e-3) -> bool:
        """Debug helper: verify CUDA coarse dslash against Python einsum version.

        Generates random input, compares CUDA output with Python reference.
        Returns True if max absolute difference < tol.
        """
        op = self.op_list[level]
        Xc, Yc, Zc, Tc = self.lat_size_list[level]
        E = self.dof_list[level]
        dtype = self.dtype_list[level]
        device = self.device_list[level]
        src = torch.randn([E, Xc, Yc, Zc, Tc], dtype=dtype, device=device)
        # Python reference (einsum-based)
        ref = op.matvec(src)
        # CUDA path
        cu = self._coarse_dslash_cuda(src, level=level)
        diff = (ref - cu).abs().max().item()
        if self.rank == self.root:
            print(f"PYQCU::SOLVER::MULTIGRID::VERIFY:\n "
                  f"level={level}, max|ref-cuda| = {diff:.6e}")
        return diff < tol

    def init(self):
        # Build local-orthonormal near-null space vectors
        comm = MPI.COMM_WORLD
        comm.Barrier()
        if self.with_cuda_qcu:
            from pyqcu.cuda.define import params, argv, set_ptrs
            params[define._LAT_X_] = self.lat_size[define._LAT_X_]
            params[define._LAT_Y_] = self.lat_size[define._LAT_Y_]
            params[define._LAT_Z_] = self.lat_size[define._LAT_Z_]
            params[define._LAT_T_] = self.lat_size[define._LAT_T_]
            params[define._LAT_XYZT_] = params[define._LAT_X_] * \
                params[define._LAT_Y_]*params[define._LAT_Z_] * \
                params[define._LAT_T_]
            params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
                define._GRID_T_] = tools.give_grid_size()
            params[define._PARITY_] = 0
            params[define._NODE_RANK_] = self.comm.Get_rank()
            params[define._NODE_SIZE_] = self.comm.Get_size()
            params[define._DAGGER_] = 0
            params[define._MAX_ITER_] = self.max_iter
            params[define._DATA_TYPE_] = define.epytd(
                torch_dtype=self.dtype_list[0])
            params[define._SET_INDEX_] = 0
            params[define._SET_PLAN_] = 1
            params[define._VERBOSE_] = 0
            params[define._SEED_] = 42
            argv = argv.to(dtype=define.dtype(
                params[define._DATA_TYPE_]).to_real())
            argv[define._MASS_] = self.mass
            argv[define._ATOL_] = self.tol
            argv[define._SIGMA_] = 0.1
            self.set_ptrs = set_ptrs.clone()
            self.params = params.clone()
            self.argv = argv.clone()
            qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
            self.gauge_eo = tools.oooxyzt2poooxyzt(
                input_array=self.op_list[0].hopping.U)  # type: ignore
            self.clover_ee = self.op_list[0].sitting.M_e
            self.clover_oo = self.op_list[0].sitting.M_o
            self.fermion_in_eo = torch.zeros(size=[2, 4, 3]+define.lat_shape(params)).to(
                dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
            self.fermion_out_eo = torch.zeros(size=[2, 4, 3]+define.lat_shape(params)).to(
                dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
            self.dest_o = self.fermion_out_eo[1]

        for i in range(1, len(self.lat_size_list)):
            _null_vecs = _torch.randn(size=[self.dof_list[i], self.dof_list[i-1]] +
                                      self.lat_size_list[i-1], dtype=self.dtype_list[i-1], device=self.device_list[i-1])
            if self.with_cuda_qcu and i == 1:
                def _bistabcg(b: torch.Tensor, tol: float = 1e-6, max_iter: int = 1000, x0:  Optional[torch.Tensor] = None, if_rtol: bool = False, verbose: bool = True) -> torch.Tensor:
                    if if_rtol:
                        _tol = tools.norm(b)*tol
                    else:
                        _tol = tol
                    self.fermion_in_eo = tools.oooxyzt2poooxyzt(input_array=b)
                    self.params[define._MAX_ITER_] = max_iter
                    self.params[define._SET_INDEX_] += 1
                    self.argv[define._ATOL_] = _tol
                    qcu.applyInitQcu(self.set_ptrs, self.params, self.argv)
                    qcu.applyCloverBistabCgQcu(self.fermion_out_eo, self.fermion_in_eo, self.gauge_eo, self.clover_ee,  # type: ignore
                                               self.clover_oo, self.clover_ee_inv, self.clover_oo_inv,  self.set_ptrs, self.params)  # type: ignore
                    return tools.poooxyzt2oooxyzt(input_array=self.fermion_out_eo).reshape(b.shape)
                bistabcg = _bistabcg
            else:
                bistabcg = None
            _null_vecs = tools.give_null_vecs(
                null_vecs=_null_vecs,
                matvec=self.op_list[i-1].matvec,
                bistabcg=bistabcg,
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
        self.num_level = len(self.op_list) if len(
            self.op_list) <= self.max_level else self.max_level

    def adaptive(self, iter: int = 0):
        if self.convergence_tol > 3:
            self.num_level = 1
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
        if self.with_cuda_qcu and level == 1:
            def _coarse_matvec(src: torch.Tensor) -> torch.Tensor:
                return self._coarse_dslash_cuda(src=src, level=level)
            matvec = _coarse_matvec
        b = self.b_list[level].clone()
        if level == 0 and self.support_parity:
            b_origin = b.clone()
            b_eo = tools.oooxyzt2poooxyzt(b)
            b_e = b_eo[0]
            b_o = b_eo[1]
            b = self.op_list[0].give_b_parity(b_e=b_e, b_o=b_o)
            if self.with_cuda_qcu is False:
                matvec = self.op_list[0].matvec_parity
            else:
                def _matvec(src_o: torch.Tensor) -> torch.Tensor:
                    self.params[define._SET_INDEX_] = 0
                    qcu.applyCloverBistabCgDslashQcu(self.dest_o, src_o, self.gauge_eo,
                                                     self.clover_ee, self.clover_oo, self.clover_ee_inv, self.clover_oo_inv,  self.set_ptrs, self.params)  # type: ignore
                    return self.dest_o.reshape(src_o.shape)
                matvec = _matvec
        x = torch.zeros_like(b)
        r = b - matvec(x)
        r_norm = tools.norm(r)
        _tol = r_norm*0.5 if level != self.num_level - 1 else r_norm*0.1
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
            return x
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
        print(f"PYQCU::SOLVER::MULTIGRID:\n {level}:Starting Iterations")
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
            if level < self.num_level-1 and count_restart > self.num_restart:
                if level == 0 and self.support_parity:
                    x_e = self.op_list[0].give_x_e(b_e=b_e, x_o=x)
                    x_origin = tools.poooxyzt2oooxyzt(
                        input_array=torch.stack([x_e, x], dim=0))
                    r = b_origin-self.op_list[0].matvec(src=x_origin)
                if self.with_cuda_qcu and level == 0:
                    r_coarse = self._restrict_cuda(fine_vec=r, level=level)
                else:
                    r_coarse = tools.restrict(
                        local_ortho_null_vecs=self.lonv_list[level], fine_vec=r)
                self.b_list[level+1] = r_coarse.to(dtype=self.dtype_list[level+1],
                                                   device=self.device_list[level+1])
                e_coarse = self.cycle(level=level+1).to(dtype=self.dtype_list[level],
                                                        device=self.device_list[level])
                if self.with_cuda_qcu and level == 0:
                    e_fine = self._prolong_cuda(coarse_vec=e_coarse, level=level)
                else:
                    e_fine = tools.prolong(
                        local_ortho_null_vecs=self.lonv_list[level], coarse_vec=e_coarse)
                if level == 0 and self.support_parity:
                    e_fine_eo = tools.oooxyzt2poooxyzt(e_fine)
                    e_fine = e_fine_eo[1]
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
        if level == 0 and self.support_parity:
            x_e = self.op_list[0].give_x_e(b_e=b_e, x_o=x)
            x_origin = tools.poooxyzt2oooxyzt(
                input_array=torch.stack([x_e, x], dim=0))
            r_origin = b_origin-self.op_list[0].matvec(src=x_origin)
            r_norm_origin = tools.norm(r_origin)
            print(
                f"PYQCU::SOLVER::MULTIGRID:\n Final residual(origin): {r_norm_origin:.2e}")
            x = x_origin.clone()
        return x

    def solve(self, b: Optional[torch.Tensor] = None, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        return x.reshape([4, 3]+list(x.shape[-4:]))

    def plot(self, save_path=None):
        if self.rank == self.root:
            import matplotlib.pyplot as plt
            import numpy as np
            try:
                np.Inf = np.inf  # type: ignore
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
