import mpi4py.MPI as MPI
import torch
from pyqcu.ascend.define import *
from pyqcu.ascend.dslash import *
from pyqcu.ascend.inverse import *
from pyqcu.ascend.io import *
from typing import Tuple
from time import perf_counter


class qcu:
    def __init__(self, lat_size: Tuple[int, int, int, int] = [8, 8, 8, 8], U: torch.Tensor = None, clover_term: torch.Tensor = None, min_size: int = 2, max_levels: int = 5, num_convergence_sample: int = 50, dof_list: Tuple[int, int, int, int] = [12, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24], max_iter: int = 1000, seed: int = 42, mg_grid_size: Tuple[int, int, int, int] = [2, 2, 2, 2], mass: float = 0.05, tol: float = 1e-6, sigma: float = 0.1, dtype: torch.dtype = None, device: torch.device = None, dtype_list: Tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype] = None, device_list: Tuple[torch.device, torch.device, torch.device, torch.device] = None, dslash: str = 'clover', solver: str = 'bistabcg', root: int = 0, verbose: bool = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.max_iter = max_iter
        self.seed = seed + self.rank  # make diffirent
        self.mass = mass
        self.kappa = 1 / (2 * self.mass + 8)
        self.tol = tol
        self.sigma = sigma
        self.root = root
        self.dslash = dslash
        self.solver = solver
        self.verbose = verbose
        self.num_convergence_sample = num_convergence_sample
        self.mg_grid_size = mg_grid_size
        if dtype != None:
            self.dtype_list = [dtype]*max_levels
        else:
            self.dtype_list = dtype_list[:max_levels]
        if device != None:
            self.device_list = [device]*max_levels
        else:
            self.device_list = device_list[:max_levels]
        self.dof_list = dof_list
        for device in self.device_list:
            set_device(device=device)
        self.lat_size = list(lat_size)
        self.grid_size = give_grid_size()
        self.grid_index = give_grid_index()
        self.local_lat_size = [self.lat_size[i]//self.grid_size[i]
                               for i in range(4)]
        if self.verbose:
            print(f"give_if_multi(): {give_if_multi()}")
            print(f"self.lat_size: {self.lat_size}")
            print(f"self.grid_size: {self.grid_size}")
            print(f"self.grid_index: {self.grid_index}")
            print(f"self.dtype_list:{self.dtype_list}")
            print(f"self.device_list:{self.device_list}")
            print(f"self.local_lat_size: {self.local_lat_size}")
        self.wilson = wilson_mg(
            latt_size=self.local_lat_size, kappa=self.kappa, dtype=self.dtype_list[0], device=self.device_list[0], verbose=False)
        self.clover = clover(latt_size=self.local_lat_size,
                             kappa=self.kappa, dtype=self.dtype_list[0], device=self.device_list[0], verbose=False)
        self.U = U
        self.clover_term = clover_term
        self.min_size = min_size
        self.max_levels = max_levels
        self.dof_list = dof_list
        self.x = None
        self.x0 = None
        self.refer_x = None
        self.b = None

    def init(self):
        try:
            if any(self.b.shape[-4:][::-1][d] != self.local_lat_size[d] for d in range(4)):
                print("Got wrong b with wrong lat_size from load or init!!!")
                self.b = None
        except Exception as e:
            print(f"Error: {e}")
        try:
            if any(self.refer_x.shape[-4:][::-1][d] != self.local_lat_size[d] for d in range(4)):
                print("Got wrong refer_x with wrong lat_size from load or init!!!")
                self.refer_x = None
        except Exception as e:
            print(f"Error: {e}")
        try:
            if any(self.x0.shape[-4:][::-1][d] != self.local_lat_size[d] for d in range(4)):
                print("Got wrong x0 with wrong lat_size from load or init!!!")
                self.x0 = None
        except Exception as e:
            print(f"Error: {e}")
        try:
            if any(self.U.shape[-4:][::-1][d] != self.local_lat_size[d] for d in range(4)):
                print("Got wrong U with wrong lat_size from load or init!!!")
                self.U = None
        except Exception as e:
            print(f"Error: {e}")
        try:
            if any(self.clover_term.shape[-4:][::-1][d] != self.local_lat_size[d] for d in range(4)):
                print("Got wrong clover_term with wrong lat_size from load or init!!!")
                self.clover_term = None
        except Exception as e:
            print(f"Error: {e}")
        if self.U == None:
            self.U = self.wilson.generate_gauge_field(
                sigma=self.sigma, seed=self.seed)
            if self.verbose:
                print(
                    f"self.wilson.check_su3(self.U): {self.wilson.check_su3(self.U)}")
        if self.b == None:
            self.b = torch_randn(
                size=[4, 3]+self.local_lat_size[::-1], dtype=self.dtype_list[0], device=self.device_list[0])
        if self.refer_x == None:
            self.refer_x = torch.zeros(
                size=[4, 3]+self.local_lat_size[::-1], dtype=self.dtype_list[0], device=self.device_list[0])
        if self.x0 == None:
            self.x0 = torch_randn(
                size=[4, 3]+self.local_lat_size[::-1], dtype=self.dtype_list[0], device=self.device_list[0])
        if self.dslash == 'clover':
            if self.clover_term == None:
                self.clover_term = self.clover.make_clover(U=self.U)
        else:
            self.clover_term = torch.zeros(
                size=[4, 3, 4, 3]+self.local_lat_size[::-1], dtype=self.dtype_list[0], device=self.device_list[0])
        self.mg = mg(lat_size=self.local_lat_size, dtype_list=self.dtype_list, device_list=self.device_list, num_convergence_sample=self.num_convergence_sample, mg_grid_size=self.mg_grid_size, wilson=self.wilson, U=self.U, clover=self.clover,
                     clover_term=self.clover_term, min_size=self.min_size, max_levels=self.max_levels, dof_list=self.dof_list, tol=self.tol, max_iter=self.max_iter, verbose=self.verbose)
        if self.solver == 'mg':
            self.mg.init()

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        return self.mg.op_list[0].matvec(src)

    def save(self, file_name: str = ''):
        try:
            grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.b, file_name=file_name +
                                      '-b.h5', lat_size=self.lat_size)
        except Exception as e:
            print(f"Error: {e}")
        try:
            grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.x, file_name=file_name +
                                      '-x.h5', lat_size=self.lat_size)
        except Exception as e:
            print(f"Error: {e}")
        try:
            grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.x0, file_name=file_name +
                                      '-x0.h5', lat_size=self.lat_size)
        except Exception as e:
            print(f"Error: {e}")
        try:
            grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.U, file_name=file_name +
                                      '-U.h5', lat_size=self.lat_size)
        except Exception as e:
            print(f"Error: {e}")
        try:
            grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.clover_term, file_name=file_name +
                                      '-clover_term.h5', lat_size=self.lat_size)
        except Exception as e:
            print(f"Error: {e}")

    def load(self, file_name: str = ''):
        try:
            self.b = hdf5_xxxtzyx2grid_xxxtzyx(
                file_name=file_name+'-b.h5', lat_size=self.lat_size, device=self.device_list[0])
        except Exception as e:
            print(f"Error: {e}")
        try:
            self.refer_x = hdf5_xxxtzyx2grid_xxxtzyx(
                file_name=file_name+'-x.h5', lat_size=self.lat_size, device=self.device_list[0])
        except Exception as e:
            print(f"Error: {e}")
        try:
            self.x0 = hdf5_xxxtzyx2grid_xxxtzyx(
                file_name=file_name+'-x0.h5', lat_size=self.lat_size, device=self.device_list[0])
        except Exception as e:
            print(f"Error: {e}")
        try:
            self.U = hdf5_xxxtzyx2grid_xxxtzyx(
                file_name=file_name+'-U.h5', lat_size=self.lat_size, device=self.device_list[0])
        except Exception as e:
            print(f"Error: {e}")
        try:
            self.clover_term = hdf5_xxxtzyx2grid_xxxtzyx(
                file_name=file_name+'-clover_term.h5', lat_size=self.lat_size, device=self.device_list[0])
        except Exception as e:
            print(f"Error: {e}")

    def solve(self, b: torch.Tensor = None, x0: torch.Tensor = None) -> torch.Tensor:
        start_time = perf_counter()
        if self.solver == 'bistabcg':
            x = bicgstab(b=self.b.clone() if b == None else b.clone(), matvec=self.matvec, tol=self.tol, max_iter=self.max_iter,
                         x0=self.x0.clone() if x0 == None else x0.clone(), verbose=self.verbose)
        elif self.solver == 'mg':
            x = self.mg.solve(b=self.b.clone() if b == None else b.clone(
            ), x0=self.x0.clone() if x0 == None else x0.clone())
        else:
            print('Not found Slover!')
        total_time = perf_counter() - start_time
        print("\nPerformance Statistics:")
        print(
            f"self.solver-{self.solver}-Total time: {total_time:.6f} seconds")
        self.x = x.reshape([4, 3]+list(x.shape[-4:])).clone()
        self.refer_x = self.x.clone() if self.refer_x == None else self.refer_x
        return self.x

    def test(self):
        print(f"torch_norm(self.U): {torch_norm(self.U)}")
        print(f"torch_norm(self.clover_term): {torch_norm(self.clover_term)}")
        print(f"torch_norm(self.b): {torch_norm(self.b)}")
        print(f"torch_norm(self.x): {torch_norm(self.x)}")
        print(f"multi_norm(self.U): {multi_norm(self.U)}")
        print(f"multi_norm(self.clover_term): {multi_norm(self.clover_term)}")
        print(f"multi_norm(self.b): {multi_norm(self.b)}")
        print(f"multi_norm(self.x): {multi_norm(self.x)}")
        full_wilson = wilson_mg(
            latt_size=self.lat_size, kappa=self.kappa, dtype=self.dtype_list[0], device=self.device_list[0], verbose=False)
        full_clover = clover(latt_size=self.lat_size,
                             kappa=self.kappa, dtype=self.dtype_list[0], device=self.device_list[0], verbose=False)

        def full_matvec(src: torch.Tensor, U: torch.Tensor, clover_term: torch.Tensor) -> torch.Tensor:
            if self.rank == self.root:
                return full_wilson.give_wilson(src=src, U=U)+full_clover.give_clover(src=src, clover_term=clover_term)
            else:
                return None
        full_U = local2full_tensor(local_tensor=self.U, root=self.root)
        full_clover_term = local2full_tensor(
            local_tensor=self.clover_term, root=self.root)
        Ax = self.matvec(src=self.x)
        full_Ax = local2full_tensor(local_tensor=Ax, root=self.root)
        _full_x = local2full_tensor(local_tensor=self.x, root=self.root)
        _full_b = local2full_tensor(local_tensor=self.b, root=self.root)
        if self.rank == self.root:
            _full_Ax = full_matvec(
                src=_full_x, U=full_U, clover_term=full_clover_term)
            print(f"torch_norm(_full_b): {torch_norm(_full_b)}")
            print(f"torch_norm(_full_x): {torch_norm(_full_x)}")
            print(f"torch_norm(full_Ax): {torch_norm(full_Ax)}")
            print(f"torch_norm(_full_Ax): {torch_norm(_full_Ax)}")
            print(
                f"torch_norm(full_Ax-_full_b).item()/torch_norm(full_Ax).item(): {torch_norm(full_Ax-_full_b).item()/torch_norm(full_Ax).item()}")
            print(
                f"torch_norm(_full_Ax-_full_b).item()/torch_norm(_full_Ax).item(): {torch_norm(_full_Ax-_full_b).item()/torch_norm(_full_Ax).item()}")
            print(
                f"torch_norm(full_Ax-_full_Ax).item()/torch_norm(full_Ax).item(): {torch_norm(full_Ax-_full_Ax).item()/torch_norm(full_Ax).item()}")
        try:
            print(
                f"multi_norm(self.refer_x-self.x)/multi_norm(self.x): {multi_norm(self.refer_x-self.x)/multi_norm(self.x)}")
        except Exception as e:
            print(f"Rank{self.rank}-Error: {e}")
