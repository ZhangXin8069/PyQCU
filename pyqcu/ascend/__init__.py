import torch
import os
from pyqcu.ascend.io import *
from pyqcu.ascend.define import *
from pyqcu.ascend.dslash import wilson, clover
from pyqcu.ascend.inverse import cg, bicgstab, mg
import mpi4py.MPI as MPI
from typing import Tuple
from time import perf_counter


class qcu:
    def __init__(self, lat_size: Tuple[int, int, int, int] = [8, 8, 8, 8], U: torch.Tensor = None, clover_term: torch.Tensor = None, max_iter: int = 1000, seed: int = 42, mass: float = 0.05, tol: float = 1e-6, sigma: float = 0.1, dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu'), root: int = 0, verbose: bool = True):
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
        self.wilson = wilson(
            latt_size=self.local_lat_size, kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
        self.clover = clover(latt_size=self.local_lat_size,
                             kappa=self.kappa, dtype=self.dtype, device=self.device, verbose=False)
        self.U = U
        self.clover_term = clover_term

    def init(self):
        if self.U == None and self.clover_term == None:
            self.U = self.wilson.generate_gauge_field(
                sigma=self.sigma, seed=self.seed)
            if self.verbose:
                print(f"self.wilson.check_su3(self.U): {self.wilson.check_su3(self.U)}")
            self.clover_term = self.clover.make_clover(U=self.U)
        print(
            f"Rank{self.rank}-torch.norm(self.U).item()**2: {torch.norm(self.U).item()**2}")
        print(
            f"Rank{self.rank}-torch.norm(self.clover_term).item()**2: {torch.norm(self.clover_term).item()**2}")
        self.full_U = local2full_tensor(
            local_tensor=self.U, lat_size=self.lat_size, grid_size=self.grid_size, root=self.root)
        self.full_clover_term = local2full_tensor(
            local_tensor=self.clover_term, lat_size=self.lat_size, grid_size=self.grid_size, root=self.root)
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

    def test(self):
        print(f"self.U.shape: {self.U.shape}")
        self.U = torch.ones_like(self.U)*self.rank
        self.U.imag = tzyxccd2ccdtzyx(
            gauge=torch.arange(36).reshape(3, 3, 4).repeat([self.local_lat_size[-1], self.local_lat_size[-2], self.local_lat_size[-3], self.local_lat_size[-4], 1, 1, 1]))
        self.clover_term = torch.ones_like(self.clover_term)*self.rank
        self.clover_term.imag = tzyxscsc2scsctzyx(
            clover_term=torch.arange(144).reshape(4, 3, 4, 3).repeat([self.local_lat_size[-1], self.local_lat_size[-2], self.local_lat_size[-3], self.local_lat_size[-4], 1, 1, 1, 1]))

    def load(self, file_name: str = ''):
        self.U = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'U.h5', lat_size=self.lat_size, grid_size=self.grid_size)
        self.clover_term = hdf5_xxxtzyx2grid_xxxtzyx(
            file_name=file_name+'clover_term.h5', lat_size=self.lat_size, grid_size=self.grid_size)

    def save(self, file_name: str = ''):
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.U, file_name=file_name +
                                  'U.h5', lat_size=self.lat_size, grid_size=self.grid_size)
        grid_xxxtzyx2hdf5_xxxtzyx(input_tensor=self.clover_term, file_name=file_name +
                                  'clover_term.h5', lat_size=self.lat_size, grid_size=self.grid_size)

    def solve(self, b: torch.Tensor = None, x0: torch.Tensor = None, sovler: str = 'bistabcg') -> torch.Tensor:
        """
        Main multigrid solver routine.
        Sets up the multigrid list, performs cycle iterations until
        convergence, and returns the solution.
        """
        if b != None:
            self.b = b.reshape([12]+list(b.shape)[2:]).clone()  # sc->e
            self.b_list[0] = self.b.clone()
        if x0 != None:
            self.x0 = x0.reshape([12]+list(x0.shape)[2:]).clone()  # sc->e
            self.u_list[0] = self.x0.clone()
        start_time = perf_counter()
        if sovler == 'bistabcg':
            x = bicgstab(b=self.b)
        total_time = perf_counter() - start_time
        print("\nPerformance Statistics:")
        print(f"Total time: {total_time:.6f} seconds")
        print(f"Final residual: {self.convergence_history[-1]:.2e}")
        return x.reshape([4, 3]+list(x.shape[-4:])).clone()
