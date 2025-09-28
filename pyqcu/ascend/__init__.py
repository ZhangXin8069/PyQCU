import torch
from pyqcu.ascend.io import *
from pyqcu.ascend.define import *
from pyqcu.ascend.dslash import *
from pyqcu.ascend.inverse import *
import mpi4py.MPI as MPI
from typing import Tuple
from time import perf_counter


class qcu:
    def __init__(self, lat_size: Tuple[int, int, int, int] = [8, 8, 8, 8], b: torch.Tensor = None, U: torch.Tensor = None, clover_term: torch.Tensor = None,  min_size: int = 2, max_levels: int = 5, dof_list: Tuple[int, int, int, int] = [12, 24, 24, 24, 24], max_iter: int = 1000, seed: int = 42, mass: float = 0.05, tol: float = 1e-6, sigma: float = 0.1, dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu'), x0: torch.Tensor = None, dslash: str = 'clover',  solver: str = 'bistabcg', root: int = 0, verbose: bool = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.b = b
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
        self.x0 = x0
        self.dslash = dslash
        self.solver = solver
        self.verbose = verbose
        self.local_rank = give_local_rank(device=self.device)
        try:
            torch.cuda.set_device(self.local_rank)
        except Exception as e:
            print(f"Rank{self.rank}-Error: {e}")
        self.lat_size = lat_size
        self.grid_size = [i for i in split_into_four_factors(N=self.size)]
        self.grid_index = give_grid_index(grid_size=self.grid_size)
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
        if self.rank == self.root:
            self.full_wilson = wilson_mg(
                latt_size=self.lat_size, kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
        else:
            self.full_wilson = self.wilson
        self.clover = clover(latt_size=self.local_lat_size,
                             kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
        if self.rank == self.root:
            self.full_clover = clover(
                latt_size=self.lat_size, kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
        else:
            self.full_clover = self.clover
        self.U = U
        self.clover_term = clover_term
        self.min_size = min_size
        self.max_levels = max_levels
        self.dof_list = dof_list
        self.op = op(wilson=self.wilson, clover=self.clover,
                     grid_size=self.grid_size)

    def init(self):
        if self.U == None:
            self.U = self.wilson.generate_gauge_field(
                sigma=self.sigma, seed=self.seed)
            if self.verbose:
                print(
                    f"self.wilson.check_su3(self.U): {self.wilson.check_su3(self.U)}")
        else:
            self.U = self.U.clone()
        self.full_U = local2full_tensor(
            local_tensor=self.U, lat_size=self.lat_size, grid_size=self.grid_size, device=self.device, root=self.root)
        if self.dslash == 'clover':
            if self.clover_term == None:
                self.clover_term = self.clover.make_clover(U=self.U)
            else:
                self.clover_term = self.clover_term.clone()
        else:
            self.clover_term = torch.zeros(
                size=[4, 3, 4, 3]+self.local_lat_size[::-1], dtype=self.dtype, device=self.device)
        self.full_clover_term = local2full_tensor(
            local_tensor=self.clover_term, lat_size=self.lat_size, grid_size=self.grid_size, device=self.device, root=self.root)
        self.b = torch.ones(
            size=[4, 3]+self.local_lat_size[::-1], dtype=self.dtype, device=self.device) if self.b == None else self.b.clone()
        self.full_b = local2full_tensor(
            local_tensor=self.b, lat_size=self.lat_size, grid_size=self.grid_size, device=self.device, root=self.root)
        self.x0 = torch.zeros(
            size=[4, 3]+self.local_lat_size[::-1], dtype=self.dtype, device=self.device) if self.x0 == None else self.x0.clone()
        self.full_x0 = local2full_tensor(
            local_tensor=self.x0, lat_size=self.lat_size, grid_size=self.grid_size, device=self.device, root=self.root)
        self.mg = mg(b=self.b, wilson=self.wilson, U=self.U, clover=self.clover, clover_term=self.clover_term, min_size=self.min_size,
                     max_levels=self.max_iter, dof_list=self.dof_list, tol=self.tol, max_iter=self.max_iter, x0=self.x0,  verbose=self.verbose)
        if self.rank == self.root:
            self.full_mg = mg(b=self.full_b, wilson=self.full_wilson, U=self.full_U, clover=self.full_clover, clover_term=self.full_clover_term, min_size=self.min_size,
                              max_levels=self.max_iter, dof_list=self.dof_list, tol=self.tol, max_iter=self.max_iter, x0=self.x0,  verbose=self.verbose)
            if self.solver == 'mg':
                self.full_mg.init()
        else:
            self.full_mg = self.mg
        for ward in range(4):  # xyzt
            hopping_M_plus = full2local_tensor(
                full_tensor=self.full_mg.op_list[0].hopping.M_plus_list[ward], lat_size=self.lat_size, grid_size=self.grid_size, device=self.device, root=self.root)
            self.op.hopping.M_plus_list.append(hopping_M_plus.clone())
            hopping_M_minus = full2local_tensor(
                full_tensor=self.full_mg.op_list[0].hopping.M_minus_list[ward], lat_size=self.lat_size, grid_size=self.grid_size, device=self.device, root=self.root)
            self.op.hopping.M_minus_list.append(hopping_M_minus.clone())
        self.op.sitting.M = full2local_tensor(
            full_tensor=self.full_mg.op_list[0].sitting.M, lat_size=self.lat_size, grid_size=self.grid_size, device=self.device, root=self.root).clone()

    def matvec(self, src: torch.Tensor) -> torch.Tensor:
        return self.op.matvec(src).clone()

    def save(self, file_name: str = ''):
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.b, file_name=file_name +
                                  '-b.h5', lat_size=self.lat_size, grid_size=self.grid_size)
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.U, file_name=file_name +
                                  '-U.h5', lat_size=self.lat_size, grid_size=self.grid_size)
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.clover_term, file_name=file_name +
                                  '-clover_term.h5', lat_size=self.lat_size, grid_size=self.grid_size)

    def load(self, file_name: str = ''):
        self.b = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'-b.h5', lat_size=self.lat_size, grid_size=self.grid_size, device=self.device)
        self.U = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'-U.h5', lat_size=self.lat_size, grid_size=self.grid_size, device=self.device)
        self.clover_term = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'-clover_term.h5', lat_size=self.lat_size, grid_size=self.grid_size, device=self.device)

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
        else:
            print('Not found Slover!')
            x = self.x0 if x0 == None else x0.clone()
        total_time = perf_counter() - start_time
        print("\nPerformance Statistics:")
        print(f"Total time: {total_time:.6f} seconds")
        # print(f"Final residual: {self.convergence_history[-1]:.2e}")
        return x.reshape([4, 3]+list(x.shape[-4:])).clone()

    def test(self):
        self.U = torch.ones_like(self.U)*self.rank
        self.U.imag = tzyxccd2ccdtzyx(
            gauge=torch.arange(36).reshape(3, 3, 4).repeat([self.local_lat_size[-1], self.local_lat_size[-2], self.local_lat_size[-3], self.local_lat_size[-4], 1, 1, 1]))
        self.clover_term = torch.ones_like(self.clover_term)*self.rank
        self.clover_term.imag = tzyxscsc2scsctzyx(
            clover_term=torch.arange(144).reshape(4, 3, 4, 3).repeat([self.local_lat_size[-1], self.local_lat_size[-2], self.local_lat_size[-3], self.local_lat_size[-4], 1, 1, 1, 1]))
        print(
            f"Rank{self.rank}-torch.norm(self.U).item()**2: {torch.norm(self.U).item()**2}")
        print(
            f"Rank{self.rank}-torch.norm(self.clover_term).item()**2: {torch.norm(self.clover_term).item()**2}")
        try:
            print(
                f"Rank{self.rank}-torch.norm(self.full_U).item()**2: {torch.norm(self.full_U).item()**2}")
            print(
                f"Rank{self.rank}-torch.norm(self.full_clover_term).item()**2: {torch.norm(self.full_clover_term).item()**2}")
        except Exception as e:
            print(f"Rank{self.rank}-Error: {e}")
        print(
            f"Rank{self.rank}-multi_norm(self.U).item()**2: {multi_norm(self.U).item()**2}")
        print(
            f"Rank{self.rank}-multi_norm(self.clover_term).item()**2: {multi_norm(self.clover_term).item()**2}")
