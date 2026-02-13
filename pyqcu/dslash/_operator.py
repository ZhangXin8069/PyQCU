import torch
import numpy as np
from pyqcu import _torch, tools, lattice, dslash
import mpi4py.MPI as MPI


class hopping:
    def __init__(self,  U: torch.Tensor = None, kappa: float = 0.1, u_0: float = 1.0):
        self.M_plus_list = [torch.zeros([]), torch.zeros(
            []), torch.zeros([]), torch.zeros([])]  # xyzt
        self.M_minus_list = [torch.zeros([]), torch.zeros(
            []), torch.zeros([]), torch.zeros([])]  # xyzt
        self.U = U
        self.grid_size = tools.give_grid_size()
        self.grid_index = tools.give_grid_index()
        if self.U is not None:
            for ward in range(4):  # xyzt
                if self.grid_size[ward] != 1:
                    comm = MPI.COMM_WORLD
                    rank = comm.Get_rank()
                    U_tail4send = self.U[tools.slice_dim(
                        dims_num=7, ward=ward, point=-1)].cpu().contiguous().numpy().copy()
                    U_head4recv = np.zeros_like(U_tail4send).copy()
                    rank_plus = tools.give_rank_plus(ward=ward)
                    rank_minus = tools.give_rank_minus(ward=ward)
                    # print(f"U.shape: {self.U.shape}")
                    # print(f"U_tail4send.shape: {U_tail4send.shape}")
                    # print(f"rank: {rank}, rank_plus: {rank_plus}, rank_minus: {rank_minus}, ward: {ward}")
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

    def matvec_plus(self, ward: int, src: torch.Tensor) -> torch.Tensor:
        dtype = src.dtype
        device = src.device
        _dtype = self.M_plus_list[ward].dtype
        _device = self.M_plus_list[ward].device
        if dtype != _dtype or device != _device:
            src = src.to(dtype=_dtype, device=_device)
        if self.grid_size[ward] != 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            src_head4send = src[tools.slice_dim(
                dims_num=5, ward=ward, point=0)].cpu().contiguous().numpy().copy()
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

    def matvec_minus(self, ward: int, src: torch.Tensor) -> torch.Tensor:
        dtype = src.dtype
        device = src.device
        _dtype = self.M_minus_list[ward].dtype
        _device = self.M_minus_list[ward].device
        if dtype != _dtype or device != _device:
            src = src.to(dtype=_dtype, device=_device)
        if self.grid_size[ward] != 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            src_tail4send = src[tools.slice_dim(
                dims_num=5, ward=ward, point=-1)].cpu().contiguous().numpy().copy()
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

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        dest = torch.zeros_like(src)
        for ward in range(4):
            dest += self.matvec_plus(ward=ward, src=src)
            dest += self.matvec_minus(ward=ward, src=src)
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


class operator:
    def __init__(self,  U: torch.Tensor = None, clover_term: torch.Tensor = None, fine_hopping: hopping = None, fine_sitting: sitting = None, local_ortho_null_vecs: torch.Tensor = None, kappa: float = 0.1, u_0: float = 1.0, verbose: bool = True):
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
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c)
                    _dest_f_plus = fine_hopping.matvec_plus(
                        ward=ward, src=_src_f)
                    _dest_f_minus = fine_hopping.matvec_minus(
                        ward=ward, src=_src_f)
                    _dest_c_plus = tools.restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_plus)
                    _dest_c_minus = tools.restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_minus)
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
                        local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c)
                    _dest_f_plus = fine_hopping.matvec_plus(
                        ward=ward, src=_src_f)
                    _dest_f_minus = fine_hopping.matvec_minus(
                        ward=ward, src=_src_f)
                    _dest_c_plus = tools.restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_plus)
                    _dest_c_minus = tools.restrict(
                        local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f_minus)
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
                    local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=_src_c)
                _dest_f = fine_sitting.matvec(src=_src_f)
                _dest_c = tools.restrict(
                    local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_dest_f)
                self.sitting.M[:, e] += _dest_c.clone()

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        if src.shape[0] == 4 and src.shape[1] == 3:
            return (self.hopping.matvec(src=src.reshape([12]+list(src.shape)[2:]))+self.sitting.matvec(src=src.reshape([12]+list(src.shape)[2:]))).reshape([4, 3]+list(src.shape)[2:])
        else:
            return self.hopping.matvec(src=src)+self.sitting.matvec(src=src)
