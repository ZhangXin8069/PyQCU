import mpi4py.MPI as MPI
import torch
from pyqcu.ascend.define import *
from pyqcu.ascend.dslash import *
from pyqcu.ascend.inverse import *
from pyqcu.ascend.io import *
from typing import Tuple
from time import perf_counter


class qcu:
    def __init__(self, lat_size: Tuple[int, int, int, int] = [8, 8, 8, 8], U: torch.Tensor = None, clover_term: torch.Tensor = None, min_size: int = 2, max_levels: int = 5, dof_list: Tuple[int, int, int, int] = [12, 24, 24, 24, 24], max_iter: int = 1000, seed: int = 42, mass: float = 0.05, tol: float = 1e-6, sigma: float = 0.1, dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu'), dslash: str = 'clover', solver: str = 'bistabcg', root: int = 0, verbose: bool = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.max_iter = max_iter
        self.seed = seed + self.rank  # make diffirent
        self.mass = mass
        self.kappa = 1 / (2 * self.mass + 8)
        self.tol = tol
        self.sigma = sigma
        self.dtype = dtype
        self.real_dtype = self.dtype.to_real()
        self.device = device
        self.root = root
        self.dslash = dslash
        self.solver = solver
        self.verbose = verbose
        self.local_rank = give_local_rank(device=self.device)
        try:
            torch.cuda.set_device(self.local_rank)
        except Exception as e:
            print(f"Rank{self.rank}-Error: {e}")
        self.lat_size = lat_size
        self.grid_size = give_grid_size()
        self.grid_index = give_grid_index()
        self.local_lat_size = [self.lat_size[i]//self.grid_size[i]
                               for i in range(4)]
        if self.verbose:
            print(f"self.lat_size: {self.lat_size}")
            print(f"self.grid_size: {self.grid_size}")
            print(f"self.grid_index: {self.grid_index}")
            print(f"self.local_lat_size: {self.local_lat_size}")
        print(f"Using device: {self.device}")
        print(
            f"@My Rank:{self.rank}/{self.size}, Local Rank:{self.local_rank}@\n")
        self.wilson = wilson_mg(
            latt_size=self.local_lat_size, kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
        self.clover = clover(latt_size=self.local_lat_size,
                             kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
        self.b = torch.randn(
            size=[4, 3]+self.local_lat_size[::-1], dtype=self.dtype, device=self.device)
        self.x0 = torch.zeros_like(self.b)
        self.x = torch.zeros_like(self.b)
        self.refer_x = torch.zeros_like(self.b)
        self.U = U
        self.clover_term = clover_term
        self.min_size = min_size
        self.max_levels = max_levels
        self.dof_list = dof_list
        self.op = op(if_multi=give_if_multi())
        if self.rank == self.root:
            self.full_wilson = wilson_mg(
                latt_size=self.lat_size, kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
            self.full_clover = clover(
                latt_size=self.lat_size, kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
        else:
            self.full_wilson = wilson_mg(verbose=False)
            self.full_clover = clover(verbose=False)

    def init(self):
        if self.U == None:
            self.U = self.wilson.generate_gauge_field(
                sigma=self.sigma, seed=self.seed)
            if self.verbose:
                print(
                    f"self.wilson.check_su3(self.U): {self.wilson.check_su3(self.U)}")
        if self.dslash == 'clover':
            if self.clover_term == None:
                self.clover_term = self.clover.make_clover(U=self.U)
        else:
            self.clover_term = torch.zeros(
                size=[4, 3, 4, 3]+self.local_lat_size[::-1], dtype=self.dtype, device=self.device)
        self.full_U = local2full_tensor(local_tensor=self.U, root=self.root)
        self.full_clover_term = local2full_tensor(
            local_tensor=self.clover_term, root=self.root)
        self.full_mg = mg(lat_size=self.lat_size, dtype=self.dtype, device=self.device, wilson=self.full_wilson, U=self.full_U, clover=self.full_clover,
                          clover_term=self.full_clover_term, min_size=self.min_size, max_levels=self.max_iter, dof_list=self.dof_list, tol=self.tol, max_iter=self.max_iter, verbose=self.verbose)
        for ward in range(4):  # xyzt
            self.op.hopping.M_plus_list[ward] = full2local_tensor(
                full_tensor=self.full_mg.op_list[0].hopping.M_plus_list[ward] if self.rank == self.root else torch.zeros(size=[12, 12]+self.lat_size[::-1],
                                                                                                                         dtype=self.dtype, device=self.device),  root=self.root)
            self.op.hopping.M_minus_list[ward] = full2local_tensor(
                full_tensor=self.full_mg.op_list[0].hopping.M_minus_list[ward] if self.rank == self.root else torch.zeros(size=[12, 12]+self.lat_size[::-1],
                                                                                                                          dtype=self.dtype, device=self.device), root=self.root)
        self.op.sitting.M = full2local_tensor(
            full_tensor=self.full_mg.op_list[0].sitting.M if self.rank == self.root else torch.zeros(size=[12, 12]+self.lat_size[::-1],
                                                                                                     dtype=self.dtype, device=self.device),  root=self.root).clone()
        if self.solver == 'mg':
            self.full_mg.sub_matvec = self.matvec
            self.full_mg.init()

    def full_matvec(self, src: torch.Tensor, U: torch.Tensor, clover_term: torch.Tensor) -> torch.Tensor:
        if self.rank == self.root:
            return self.full_wilson.give_wilson(src=src, U=U)+self.full_clover.give_clover(src=src, clover_term=clover_term)
        else:
            return None

    def matvec(self, src: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
        return self.op.matvec(src, if_multi=give_if_multi() and if_multi).clone()

    def save(self, file_name: str = ''):
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.b, file_name=file_name +
                                  '-b.h5', lat_size=self.lat_size)
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.x, file_name=file_name +
                                  '-x.h5', lat_size=self.lat_size)
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.x0, file_name=file_name +
                                  '-x0.h5', lat_size=self.lat_size)
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.U, file_name=file_name +
                                  '-U.h5', lat_size=self.lat_size)
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.clover_term, file_name=file_name +
                                  '-clover_term.h5', lat_size=self.lat_size)

    def load(self, file_name: str = ''):
        self.b = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'-b.h5', lat_size=self.lat_size, device=self.device)
        self.refer_x = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'-x.h5', lat_size=self.lat_size, device=self.device)
        self.x0 = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'-x0.h5', lat_size=self.lat_size, device=self.device)
        self.U = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'-U.h5', lat_size=self.lat_size, device=self.device)
        self.clover_term = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'-clover_term.h5', lat_size=self.lat_size, device=self.device)

    def solve(self, b: torch.Tensor = None, x0: torch.Tensor = None) -> torch.Tensor:
        """
        Main multigrid solver routine.
        Sets up the multigrid list, performs cycle iterations until
        convergence, and returns the solution.
        """
        start_time = perf_counter()
        if self.solver == 'bistabcg':
            x = bicgstab(b=self.b.clone() if b == None else b.clone(), matvec=self.matvec, tol=self.tol, max_iter=self.max_iter,
                         x0=self.x0.clone() if x0 == None else x0.clone(), verbose=self.verbose)
        elif self.solver == 'mg':
            x = self.full_mg.solve(b=self.b.clone() if b == None else b.clone(
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
        Ax = self.matvec(src=self.x)
        full_Ax = local2full_tensor(local_tensor=Ax, root=self.root)
        _full_x = local2full_tensor(local_tensor=self.x, root=self.root)
        _full_b = local2full_tensor(local_tensor=self.b, root=self.root)
        if self.rank == self.root:
            _full_Ax = self.full_matvec(
                src=_full_x, U=self.full_U, clover_term=self.full_clover_term)
            print(f"torch.norm(self.b): {torch.norm(self.b)}")
            print(f"torch.norm(self.x): {torch.norm(self.x)}")
            print(f"torch.norm(_full_b): {torch.norm(_full_b)}")
            print(f"torch.norm(_full_x): {torch.norm(_full_x)}")
            print(f"torch.norm(full_Ax): {torch.norm(full_Ax)}")
            print(f"torch.norm(_full_Ax): {torch.norm(_full_Ax)}")
            print(
                f"torch.norm(full_Ax-_full_b).item()/torch.norm(full_Ax).item(): {torch.norm(full_Ax-_full_b).item()/torch.norm(full_Ax).item()}")
            print(
                f"torch.norm(_full_Ax-_full_b).item()/torch.norm(_full_Ax).item(): {torch.norm(_full_Ax-_full_b).item()/torch.norm(_full_Ax).item()}")
            print(
                f"torch.norm(full_Ax-_full_Ax).item()/torch.norm(full_Ax).item(): {torch.norm(full_Ax-_full_Ax).item()/torch.norm(full_Ax).item()}")
        try:
            print(
                f"torch.norm(self.refer_x-self.x).item()/torch.norm(self.x).item(): {torch.norm(self.refer_x-self.x).item()/torch.norm(self.x).item()}")
        except Exception as e:
            print(f"Rank{self.rank}-Error: {e}")
