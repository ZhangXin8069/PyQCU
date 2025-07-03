import torch
import torch.nn as nn
from math import sqrt
from typing import Tuple, Optional


class LatticeGaugeTheory(nn.Module):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int],
                 kappa: float = 0.1,
                 r: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        Wilson-Dirac operator on a 4D lattice with SU(3) gauge fields
        Args:
            latt_size: Tuple (Lx, Ly, Lz, Lt) specifying lattice dimensions
            kappa: Hopping parameter (controls fermion mass)
            r: Wilson parameter (usually 1.0)
            Nc: Number of colors (3 for QCD)
            dtype: Data type for tensors
            device: Device to run on (default: CPU)
            verbose: Enable verbose output for debugging
        """
        super().__init__()
        self.latt_size = latt_size
        self.Lx, self.Ly, self.Lz, self.Lt = latt_size
        self.kappa = kappa
        self.r = r
        self.Nc = 3
        self.Nd = 4
        self.Ns = 4
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        self.R = r*torch.eye(self.Ns, dtype=self.dtype, device=self.device)
        self.verbose = verbose
        # Determine real dtype based on complex dtype
        self.real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        if self.verbose:
            print(f"Initializing lattice gauge theory:")
            print(f"  Lattice size: {latt_size} (x,y,z,t)")
            print(f"  Parameters: kappa={kappa}, r={r}")
            print(f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
            print(f"  Device: {self.device}")
        # Precompute gamma matrices
        self.gamma = self._define_gamma_matrices()
        # Precompute Gell-Mann matrices for SU(3) generation
        self.gell_mann = self._get_gell_mann_matrices()
        if self.verbose:
            print("Gamma matrices and Gell-Mann matrices initialized")

    def _get_gell_mann_matrices(self) -> torch.Tensor:
        """Generate Gell-Mann matrices for SU(3) algebra"""
        # Create all matrices using the real dtype
        matrices = [
            torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                         dtype=self.real_dtype, device=self.device),
            torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]],
                         dtype=self.real_dtype, device=self.device),  # Will multiply by 1j later
            torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]],
                         dtype=self.real_dtype, device=self.device),
            torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0]],
                         dtype=self.real_dtype, device=self.device),
            torch.tensor([[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                         dtype=self.real_dtype, device=self.device),  # Will multiply by 1j later
            torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                         dtype=self.real_dtype, device=self.device),
            torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                         dtype=self.real_dtype, device=self.device),  # Will multiply by 1j later
            torch.tensor([[1/sqrt(3), 0, 0], [0, 1/sqrt(3), 0], [0, 0, -2/sqrt(3)]],
                         dtype=self.real_dtype, device=self.device)
        ]
        # Apply imaginary factors where needed
        matrices[1] = matrices[1] * 1j
        matrices[4] = matrices[4] * 1j
        matrices[6] = matrices[6] * 1j
        # Convert to complex dtype if needed
        if self.dtype == torch.complex128 or self.dtype == torch.complex64:
            matrices = [m.to(self.dtype) for m in matrices]
        return torch.stack(matrices, dim=0)  # Shape: [8, 3, 3]

    def generate_gauge_field(self,
                             sigma: float = 0.1,
                             seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random SU(3) gauge field using Gaussian distribution
        Args:
            sigma: Width of Gaussian distribution (controls randomness)
            seed: Random seed for reproducibility
        Returns:
            U: Gauge field tensor [c, c, d, t, z, y, x]
        """
        if self.verbose:
            print(f"Generating gauge field with sigma={sigma}")
        # Set random seed if provided
        if seed is not None:
            if self.verbose:
                print(f"  Setting random seed: {seed}")
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
        # Initialize gauge field tensor
        U = torch.zeros((self.Nc, self.Nc, 4, self.Lt, self.Lz, self.Ly, self.Lx),
                        dtype=self.dtype, device=self.device)
        # Generate random coefficients with proper dtype and shape
        # Dimensions: [directions, sites_t, sites_z, sites_y, sites_x, gell_mann_index]
        a = torch.normal(0.0, 1.0, size=(4, self.Lt, self.Lz, self.Ly, self.Lx, 8),
                         dtype=self.real_dtype, device=self.device)
        if self.verbose:
            print(f"  Coefficient tensor shape: {a.shape}")
            print(f"  Coefficient dtype: {a.dtype}")
            print(f"  Gell-Mann dtype: {self.gell_mann.dtype}")
        # Generate SU(3) matrices for each lattice site and direction
        if self.verbose:
            print("  Computing SU(3) matrices...")
            total_sites = self.Lt * self.Lz * self.Ly * self.Lx * 4
            processed = 0
        # Iterate over all lattice sites
        for t in range(self.Lt):
            for z in range(self.Lz):
                for y in range(self.Ly):
                    for x in range(self.Lx):
                        for d in range(self.Nd):  # 4 directions
                            # Get coefficients for this site and direction
                            coeffs = a[d, t, z, y, x].to(
                                dtype=self.dtype)  # Shape: [8]
                            # Construct Hermitian matrix: H = Σ coeffs[i] * gell_mann[i]
                            H = torch.einsum(
                                'i,ijk->jk', coeffs, self.gell_mann)
                            # Compute SU(3) matrix via exponential map
                            U_mat = torch.matrix_exp(1j * sigma * H)
                            # Store in gauge field
                            U[:, :, d, t, z, y, x] = U_mat
                            if self.verbose and processed % 1000 == 0:
                                print(
                                    f"    Processed {processed}/{total_sites} sites")
                            processed += 1
        if self.verbose:
            print("  Gauge field generation complete")
            print(f"  Gauge field norm: {torch.norm(U).item()}")
        return U

    def _define_gamma_matrices(self) -> torch.Tensor:
        """Define Dirac gamma matrices in Euclidean space"""
        gamma = torch.zeros(4, 4, 4, dtype=self.dtype, device=self.device)
        # gamma_0 (temporal direction)
        gamma[0] = torch.tensor([
            [0, 0, 0, 1j],
            [0, 0, 1j, 0],
            [0, -1j, 0, 0],
            [-1j, 0, 0, 0]
        ], dtype=self.dtype)
        # gamma_1 (x-direction)
        gamma[1] = torch.tensor([
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0]
        ], dtype=self.dtype)
        # gamma_2 (y-direction)
        gamma[2] = torch.tensor([
            [0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [-1j, 0, 0, 0],
            [0, 1j, 0, 0]
        ], dtype=self.dtype)
        # gamma_3 (z-direction)
        gamma[3] = torch.tensor([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=self.dtype)
        return gamma

    def apply_dirac_operator(self,
                             src: torch.Tensor,
                             U: torch.Tensor) -> torch.Tensor:
        """
        Apply Wilson-Dirac operator to source field
        Args:
            src: Source field tensor [s, c, t, z, y, x]
            U: Gauge field tensor [c, c, d, t, z, y, x]
        Returns:
            Dest tensor [s, c, t, z, y, x]
        """
        if self.verbose:
            print("Applying Dirac operator...")
            print(f"  Source shape: {src.shape}")
            print(f"  Gauge field shape: {U.shape}")
            print(f"  Source norm: {torch.norm(src).item()}")
        # Compute adjoint gauge field (dagger conjugate)
        U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj()
        # Initialize dest tensor
        dest = src.clone()
        # Define directions with corresponding axes and gamma matrices
        directions = [
            {'mu': 0, 'axis': 5, 'name': 'x', 'gamma': self.gamma[0]},
            {'mu': 1, 'axis': 4, 'name': 'y', 'gamma': self.gamma[1]},
            {'mu': 2, 'axis': 3, 'name': 'z', 'gamma': self.gamma[2]},
            {'mu': 3, 'axis': 2, 'name': 't', 'gamma': self.gamma[3]},
        ]
        # Apply Wilson-Dirac operator for each direction
        for dir_info in directions:
            mu = dir_info['mu']
            axis = dir_info['axis']
            gamma_mu = dir_info['gamma']
            name = dir_info['name']
            if self.verbose:
                print(f"  Processing {name}-direction (axis={axis})...")
            # Extract gauge field for current direction
            U_mu = U[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x]
            U_dag_mu = U_dag[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x]
            # Term 1: (r - γ_μ) U_{x,μ} src_{x+μ}
            src_plus = torch.roll(src, shifts=-1, dims=axis)
            # Contract color indices: U_mu * src_plus
            U_src_plus = torch.einsum('Cctzyx,sctzyx->sCtzyx', U_mu, src_plus)
            # Apply (r - gamma_mu) in spin space
            term1 = torch.einsum(
                'Ss,sctzyx->Sctzyx', (self.R - gamma_mu), U_src_plus)
            # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
            src_minus = torch.roll(src, shifts=1, dims=axis)
            U_dag_minus = torch.roll(U_dag_mu, shifts=1, dims=axis)
            # Contract color indices: U_dag_minus * src_minus
            U_dag_src_minus = torch.einsum(
                'Cctzyx,sctzyx->sCtzyx', U_dag_minus, src_minus)
            # Apply (r + gamma_mu) in spin space
            term2 = torch.einsum(
                'Ss,sctzyx->Sctzyx', (self.R + gamma_mu), U_dag_src_minus)
            # Combine terms and subtract from dest
            hopping = term1 + term2
            dest -= self.kappa * hopping
            if self.verbose:
                print(f"    Hopping term norm: {torch.norm(hopping).item()}")
        if self.verbose:
            print("Dirac operator application complete")
            print(f"  Dest norm: {torch.norm(dest).item()}")
        return dest


# Example usage
if __name__ == "__main__":
    # Lattice parameters
    # latt_size = (16, 8, 8, 8)
    # latt_size = (1, 1, 1, 1)
    # latt_size = (2, 2, 2, 2)
    latt_size = (8, 4, 4, 4)
    # latt_size = (4, 4, 4, 4)
    # latt_size = (8, 4, 4, 8)
    kappa = 0.125
    dtype = torch.complex128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verbose = True
    print(f"Using device: {device}")
    # Initialize lattice gauge theory
    lgt = LatticeGaugeTheory(
        latt_size=latt_size,
        kappa=kappa,
        dtype=dtype,
        device=device,
        verbose=verbose
    )
    # Generate random gauge field
    U = lgt.generate_gauge_field(sigma=0.1, seed=42)
    # U = torch.ones_like(U)
    # U = torch.zeros_like(U)
    # U = torch.eye(3, 3, dtype=dtype, device=device).repeat(
    #     4, latt_size[-1], latt_size[-2], latt_size[-3], latt_size[-4], 1, 1).permute(5, 6, 0, 1, 2, 3, 4)
    # U = torch.tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype, device=device).reshape(3, 3).repeat(
    #     4, latt_size[-1], latt_size[-2], latt_size[-3], latt_size[-4], 1, 1).permute(5, 6, 0, 1, 2, 3, 4)
    # Generate random source field [s, c, t, z, y, x]
    src = torch.randn(4, 3, latt_size[3], latt_size[2], latt_size[1], latt_size[0],
                      dtype=dtype, device=device)
    # print(
    #     f" torch.tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=dtype, device=device).reshape(4, 3){ torch.tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=dtype, device=device).reshape(4, 3)}")
    # src = torch.tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=dtype, device=device).reshape(4, 3).repeat(
    #     latt_size[-1], latt_size[-2], latt_size[-3], latt_size[-4], 1, 1).permute(4, 5,  0, 1, 2, 3)
    # src = torch.ones_like(src)
    # Apply Wilson-Dirac operator
    dest = lgt.apply_dirac_operator(src, U)
    # Verify properties
    print("\nDests:")
    print(f"Dest shape: {dest.shape}")
    print(f"Max abs value: {torch.max(torch.abs(dest)).item()}")
    print(f"Dest norm: {torch.norm(dest).item()}")
    print(f"Dest dtype: {dest.dtype}")
    # print(f"U value:{U}")
    # print(f"Src value:{src}")
    # print(f"Dest value:{dest}")
    import warnings
    from pyqcu.set import *
    from pyqcu import io, gauge, cg, bistabcg, define, qcu, linalg, eigen
    import cupy as cp
    import numpy as np
    np.Inf = np.inf
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    params[define._LAT_X_] = latt_size[define._LAT_X_]
    params[define._LAT_Y_] = latt_size[define._LAT_Y_]
    params[define._LAT_Z_] = latt_size[define._LAT_Z_]
    params[define._LAT_T_] = latt_size[define._LAT_T_]
    params[define._LAT_XYZT_] = params[define._LAT_X_] * \
        params[define._LAT_Y_] * \
        params[define._LAT_Z_] * params[define._LAT_T_]
    params[define._NODE_RANK_] = define.rank
    params[define._NODE_SIZE_] = define.size
    params[define._DATA_TYPE_] = define._LAT_C128_
    argv = argv.astype(define.dtype_half(params[define._DATA_TYPE_]))
    argv[define._MASS_] = 0.0
    argv[define._TOL_] = 1e-8
    kappa = 1 / (2 * argv[define._MASS_] + 8)
    print('My rank is ', define.rank)
    print("Parameters:", params)
    print("Args:", argv)
    wilson_cg_params = params.copy()
    wilson_cg_params[define._SET_INDEX_] = 0
    wilson_cg_params[define._SET_PLAN_] = define._SET_PLAN1_
    qcu.applyInitQcu(set_ptrs, wilson_cg_params, argv)
    wilson_dslash_eo_params = params.copy()
    wilson_dslash_eo_params[define._SET_INDEX_] = 1
    wilson_dslash_eo_params[define._SET_PLAN_] = define._SET_PLAN0_
    wilson_dslash_eo_params[define._PARITY_] = define._EVEN_
    wilson_dslash_eo_params[define._DAGGER_] = define._NO_USE_
    qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_params, argv)
    wilson_dslash_eo_dag_params = params.copy()
    wilson_dslash_eo_dag_params[define._SET_INDEX_] = 2
    wilson_dslash_eo_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
    wilson_dslash_eo_dag_params[define._PARITY_] = define._EVEN_
    wilson_dslash_eo_dag_params[define._DAGGER_] = define._USE_
    qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_dag_params, argv)
    wilson_dslash_oe_params = params.copy()
    wilson_dslash_oe_params[define._SET_INDEX_] = 3
    wilson_dslash_oe_params[define._SET_PLAN_] = define._SET_PLAN0_
    wilson_dslash_oe_params[define._PARITY_] = define._ODD_
    wilson_dslash_oe_params[define._DAGGER_] = define._NO_USE_
    qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_params, argv)
    wilson_dslash_oe_dag_params = params.copy()
    wilson_dslash_oe_dag_params[define._SET_INDEX_] = 4
    wilson_dslash_oe_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
    wilson_dslash_oe_dag_params[define._PARITY_] = define._ODD_
    wilson_dslash_oe_dag_params[define._DAGGER_] = define._USE_
    qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_dag_params, argv)
    print("Set pointers:", set_ptrs)
    print("Set pointers data:", set_ptrs.data)

    def dslash_eo(src, U):
        dest = cp.zeros_like(src)
        qcu.applyWilsonDslashQcu(
            dest, src, U, set_ptrs, wilson_dslash_eo_params)
        return dest

    def dslash_oe(src, U):
        dest = cp.zeros_like(src)
        qcu.applyWilsonDslashQcu(
            dest, src, U, set_ptrs, wilson_dslash_oe_params)
        return dest

    def dslash(src, U):
        print(
            f"src.type:{type(src)},src.dtype:{src.dtype},src.shape:{src.shape}")
        print(f"U.type:{type(U)},U.dtype:{U.dtype},U.shape:{U.shape}")
        U_eo = io.xxxtzyx2pxxxtzyx(U)
        U_eo = io.pccdtzyx2ccdptzyx(U_eo)
        print(
            f"U_eo type:{type(U_eo)},U_eo.dtype:{U_eo.dtype},U_eo.shape:{U_eo.shape}")
        print(f"U_eo value:{U_eo}")
        src_eo = io.xxxtzyx2pxxxtzyx(src)
        src_eo = src_eo.copy()  # DEBUG!!!
        src_e = src_eo[define._EVEN_].copy()  # DEBUG!!!
        src_o = src_eo[define._ODD_].copy()  # DEBUG!!!
        dest = cp.zeros_like(src_eo)
        U_eo = U_eo.copy()  # DEBUG!!!
        dest[define._EVEN_] = src_e - kappa*dslash_eo(src_o, U_eo)
        dest[define._ODD_] = src_o-kappa * dslash_oe(src_e, U_eo)
        return io.pxxxtzyx2xxxtzyx(dest)
    # gauge.test_su3(U[:, :, -1, -1, -1, -1, -1])
    _dest = dslash(cp.array(src.cpu().numpy()), cp.array(U.cpu().numpy()))
    _dest = torch.tensor(
        data=_dest.get(), device=dest.device, dtype=dest.dtype)
    print(f"dest value:{dest}")
    print(f"dest norm value:{torch.linalg.norm(dest)}")
    print(f"_dest value:{_dest}")
    print(f"_dest norm value:{torch.linalg.norm(_dest)}")
    print(
        f"torch.linalg.norm(dest-_dest)/torch.linalg.norm(dest):{torch.linalg.norm(dest-_dest)/torch.linalg.norm(dest)*100}%")
    print(f"dest - _dest value:{dest-_dest}")
