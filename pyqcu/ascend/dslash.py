import torch
import torch.nn as nn
from math import sqrt
from typing import Tuple, Optional
from pyqcu.ascend.io import *
from pyqcu.ascend.define import *


class wilson(nn.Module):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int] = [8, 8, 8, 8],
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        Wilson-Dirac operator on a 4D lattice with SU(3) gauge fields.
        Args:
            latt_size: Tuple (Lx, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3.
            kappa: Hopping parameter (controls fermion mass).
            u_0: Wilson parameter (usually 1.0).
            dtype: Data type for tensors.
            device: Device to run on (default: CPU).
            verbose: Enable verbose output for debugging.
        reference:
            [1](1-60).
        """
        super().__init__()
        self.latt_size = latt_size
        self.Lx, self.Ly, self.Lz, self.Lt = latt_size
        self.kappa = kappa
        self.u_0 = u_0
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        # npu needed......
        self.I = torch_eye(4, dtype=self.dtype, device=self.device)
        self.verbose = verbose
        # Determine real dtype based on complex dtype
        self.real_dtype = dtype.to_real()
        if self.verbose:
            print(f"Initializing Wilson:")
            print(f"  Lattice size: {latt_size} (x,y,z,t)")
            print(f"  Parameters: kappa={kappa}, u_0={u_0}")
            print(f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
            print(f"  Device: {self.device}")
        # Precompute gamma matrices
        self.gamma = self._define_gamma_matrices()
        # Precompute Gell-Mann matrices for SU(3) generation
        self.gell_mann = self._get_gell_mann_matrices()
        if self.verbose:
            print("Gamma matrices and Gell-Mann matrices initialized")

    def _get_gell_mann_matrices(self) -> torch.Tensor:
        """Generate Gell-Mann matrices for SU(3) algebra."""
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

    def generate_gauge_field(self, sigma: float = 0.1, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random SU(3) gauge field using Gaussian distribution.
        Optimized: Fully vectorized, no Python loops over lattice sites.
        Args:
            sigma: Width of Gaussian distribution (controls randomness).
            seed: Random seed for reproducibility.
        Returns:
            U: Gauge field tensor [c, c, d, t, z, y, x].
        """
        if self.verbose:
            print(f"Generating gauge field with sigma={sigma}")
        # Random seed
        if seed is not None:
            if self.verbose:
                print(f"  Setting random seed: {seed}")
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
        # Random Gaussian coefficients: shape [4, Lt, Lz, Ly, Lx, 8]
        a = torch.normal(
            0.0, 1.0,
            size=(4, self.Lt, self.Lz, self.Ly, self.Lx, 8),
            dtype=self.real_dtype,
            device=self.device
        )
        if self.verbose:
            print(f"  Coefficient tensor shape: {a.shape}")
            print(f"  Coefficient dtype: {a.dtype}")
        # Expand gell_mann basis for broadcasting
        # gell_mann: (8, 3, 3) -> (1, 1, 1, 1, 1, 8, 3, 3)
        gell_mann_expanded = self.gell_mann.view(1, 1, 1, 1, 1, 8, 3, 3)
        # Compute all Hermitian matrices H in one go: shape [4, Lt, Lz, Ly, Lx, 3, 3]
        H = torch_einsum('...i,...ijk->...jk',
                         a.to(self.dtype), gell_mann_expanded)
        # Apply exponential map: shape stays [4, Lt, Lz, Ly, Lx, 3, 3]
        U_all = torch.matrix_exp(1j * sigma * H)
        # Rearrange to [3, 3, 4, Lt, Lz, Ly, Lx]
        U = U_all.permute(5, 6, 0, 1, 2, 3, 4).contiguous()
        if self.verbose:
            print("  Gauge field generation complete")
            print(f"  Gauge field norm: {torch_norm(U).item()}")
        return U

    def check_su3(self, U: torch.Tensor, tol: float = 1e-6) -> bool:
        """
        Check if the given tensor satisfies SU(3) conditions.
        Works for a single matrix or a batch of matrices (e.g., full lattice gauge field).
        Args:
            U: Tensor of shape [3, 3, 4, Lt, Lz, Ly, Lx], complex dtype
            tol: Numerical tolerance for checks
        Returns:
            bool: True if all matrices satisfy SU(3) conditions
        """
        U_mat = U.permute(*range(2, U.ndim), 0,
                          1).reshape(-1, 3, 3).clone()  # N x 3 x 3
        N = U_mat.shape[0]
        # Precompute the identity matrix for unitary check
        eye = torch_eye(3, dtype=U_mat.dtype,
                        device=U_mat.device).expand(N, -1, -1)
        # 1 Unitarity check: Uᴴ U ≈ I
        UH_U = torch_matmul(U_mat.conj().transpose(-1, -2), U_mat)
        unitary_ok = torch_allclose(UH_U, eye, atol=tol)
        # 2 Determinant check: det(U) ≈ 1
        det_U = torch.linalg.det(U_mat)
        det_ok = torch_allclose(det_U, torch.ones_like(det_U), atol=tol)
        # 3 Minor identities check
        # Flatten matrices to shape (N, 9) for easy indexing
        Uf = U_mat.reshape(N, 9)
        c6 = (Uf[:, 1] * Uf[:, 5] - Uf[:, 2] * Uf[:, 4]).conj()
        c7 = (Uf[:, 2] * Uf[:, 3] - Uf[:, 0] * Uf[:, 5]).conj()
        c8 = (Uf[:, 0] * Uf[:, 4] - Uf[:, 1] * Uf[:, 3]).conj()
        minors_ok = (torch_allclose(Uf[:, 6], c6, atol=tol) and
                     torch_allclose(Uf[:, 7], c7, atol=tol) and
                     torch_allclose(Uf[:, 8], c8, atol=tol))
        # --- Optional verbose output ---
        if self.verbose:
            print(f"[check_su3] Total matrices checked: {N}")
            print(f"  Unitary check   : {unitary_ok}")
            print(f"  Determinant=1   : {det_ok}")
            print(f"  Minor identities: {minors_ok}")
            if not unitary_ok:
                max_err = (UH_U - eye).abs().max().item()
                print(f"    Max unitary deviation: {max_err:e}")
            if not det_ok:
                max_det_err = (det_U - 1).abs().max().item()
                print(f"    Max det deviation: {max_det_err:e}")
        return unitary_ok and det_ok and minors_ok

    def _define_gamma_matrices(self) -> torch.Tensor:
        """Define Dirac gamma matrices in Euclidean space."""
        gamma = torch.zeros(4, 4, 4, dtype=self.dtype, device=self.device)
        # gamma_0 (x-direction)
        gamma[0] = torch.tensor([
            [0, 0, 0, 1j],
            [0, 0, 1j, 0],
            [0, -1j, 0, 0],
            [-1j, 0, 0, 0]
        ], dtype=self.dtype)
        # gamma_1 (y-direction)
        gamma[1] = torch.tensor([
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0]
        ], dtype=self.dtype)
        # gamma_2 (z-direction)
        gamma[2] = torch.tensor([
            [0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [-1j, 0, 0, 0],
            [0, 1j, 0, 0]
        ], dtype=self.dtype)
        # gamma_3 (t-direction)
        gamma[3] = torch.tensor([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=self.dtype)
        return gamma

    def give_wilson(self,
                    src: torch.Tensor,
                    U: torch.Tensor, with_I: bool = True) -> torch.Tensor:
        """
        Apply Wilson-Dirac operator to source field:
        $$
        2*\kappa*a*M_{x,y}=1-\kappa/u_0*\sum_{\mu}[(1-\gamma_{\mu})U_{x,\mu}\delta_{x,y-\mu}+(1+\gamma_{\mu})U_{x-\mu,\mu}^{\dag}\delta_{x,y+\mu}]
                          =1-\kappa/u_0*\sum_{\mu}[(1-\gamma_{\mu})U_{x,\mu}\delta_{x+\mu,y}+(1+\gamma_{\mu})U_{x-\mu,\mu}^{\dag}\delta_{x-\mu,y}]
        $$
        Args:
            src: Source field tensor [s, c, t, z, y, x].
            U: Gauge field tensor [c, c, d, t, z, y, x].
        Returns:
            Dest tensor [s, c, t, z, y, x].
        """
        if self.verbose:
            print("Applying Dirac operator...")
            print(f"  Source shape: {src.shape}")
            print(f"  Gauge field shape: {U.shape}")
            print(f"  Source norm: {torch_norm(src).item()}")
        # Compute adjoint gauge field (dagger conjugate)
        U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj().clone()
        # Initialize dest tensor
        dest = src.clone() if with_I else torch.zeros_like(src)
        # Define directions with corresponding axes and gamma matrices
        directions = [
            {'mu': 0, 'axis': -1-0, 'name': 'x',
             'gamma': self.gamma[0]},
            {'mu': 1, 'axis': -1-1, 'name': 'y',
             'gamma': self.gamma[1]},
            {'mu': 2, 'axis': -1-2, 'name': 'z',
             'gamma': self.gamma[2]},
            {'mu': 3, 'axis': -1-3, 'name': 't',
             'gamma': self.gamma[3]},
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
            src_plus = torch_roll(src, shifts=-1, dims=axis)
            # Contract color indices: U_mu * src_plus
            U_src_plus = torch_einsum('Cctzyx,sctzyx->sCtzyx', U_mu, src_plus)
            # Apply (r - gamma_mu) in spin space
            term1 = torch_einsum(
                'Ss,sCtzyx->SCtzyx', (self.I - gamma_mu), U_src_plus)
            # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
            src_minus = torch_roll(src, shifts=1, dims=axis)
            U_dag_minus = torch_roll(U_dag_mu, shifts=1, dims=axis)
            # Contract color indices: U_dag_minus * src_minus
            U_dag_src_minus = torch_einsum(
                'Cctzyx,sctzyx->sCtzyx', U_dag_minus, src_minus)
            # Apply (r + gamma_mu) in spin space
            term2 = torch_einsum(
                'Ss,sCtzyx->SCtzyx', (self.I + gamma_mu), U_dag_src_minus)
            # Combine terms and subtract from dest
            hopping = term1 + term2
            dest -= self.kappa/self.u_0 * hopping
            if self.verbose:
                print(f"    Hopping term norm: {torch_norm(hopping).item()}")
        if self.verbose:
            print("Dirac operator application complete")
            print(f"  Dest norm: {torch_norm(dest).item()}")
        return dest.clone()


class wilson_parity(wilson):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int] = [8, 8, 8, 8],
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        Wilson-Dirac operator on a 4D lattice with SU_eo(3) gauge fields with parity decomposition.
        Args:
            latt_size: Tuple (Lx_p, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3, p(parity)=2.
            kappa: Hopping parameter (controls fermion mass).
            u_0: Wilson parameter (usually 1.0).
            dtype: Data type for tensors.
            device: Device to run on (default: CPU).
            verbose: Enable verbose output for debugging.
        reference:
            [1](1-60);[1](1-160).
        addition:
            I once compared the results of 'dslash' and 'dslash_parity' on very small cells (i.e., cells with a side length of 1), but there was always a very large deviation. In fact, this is extremely foolish because one of the conditions for 'dslash_parity' to hold is that the parity of adjacent cells is different. However, if there are cells with a side length of 1, then there must be a situation where the parity of adjacent cells is the same. I hope future generations can take this as a reference......
        """
        super().__init__(latt_size=latt_size, kappa=kappa,
                         u_0=u_0, dtype=dtype, device=device, verbose=verbose)
        self.Lx_p = self.Lx // 2
        if self.verbose:
            print(f"Initializing Wilson with parity decomposition:")
            print(f"  Lattice size: {latt_size} (x,y,z,t)")
            print(f"  Lattice x with parity: {self.Lx_p}")
            print(f"  Parameters: kappa={kappa}, u_0={u_0}")
            print(f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
            print(f"  Device: {self.device}")

    def give_wilson_eo(self,
                       src_o: torch.Tensor,
                       U_eo: torch.Tensor) -> torch.Tensor:
        """
        Apply Wilson-Dirac operator to source field in eo:
        $$
        2*\kappa*a*M_{x,y}=1-\kappa/u_0*\sum_{\mu}[(1-\gamma_{\mu})U_{x,\mu}\delta_{x,y-\mu}+(1+\gamma_{\mu})U_{x-\mu,\mu}^{\dag}\delta_{x,y+\mu}]
                          =1-\kappa/u_0*\sum_{\mu}[(1-\gamma_{\mu})U_{x,\mu}\delta_{x+\mu,y}+(1+\gamma_{\mu})U_{x-\mu,\mu}^{\dag}\delta_{x-\mu,y}]
        $$
        Args:
            src_o: Source field tensor [s, c, t, z, y, x_p]
            U_eo: Gauge field tensor [p, c, c, d, t, z, y, x_p]
        Returns:
            dest_e tensor [s, c, t, z, y, x_p]
        """
        if self.verbose:
            print("Applying Dirac operator in eo...")
            print(f"  Source shape: {src_o.shape}")
            print(f"  Gauge field shape: {U_eo.shape}")
            print(f"  Source norm: {torch_norm(src_o).item()}")
        # Compute adjoint gauge field (dagger conjugate)
        U_e = U_eo[0].clone()
        U_o = U_eo[1].clone()
        U_o_dag = U_o.permute(1, 0, 2, 3, 4, 5, 6).conj()
        # Initialize dest_e tensor
        # dest_e = src_o.clone() # move this I term to sitting term from this hopping term(origin wilson term)
        dest_e = torch.zeros_like(src_o)
        # Define directions with corresponding axes and gamma matrices
        directions = [
            {'mu': 0, 'axis': -1-0, 'name': 'x_p',
             'gamma': self.gamma[0]},
            {'mu': 1, 'axis': -1-1, 'name': 'y',
             'gamma': self.gamma[1]},
            {'mu': 2, 'axis': -1-2, 'name': 'z',
             'gamma': self.gamma[2]},
            {'mu': 3, 'axis': -1-3, 'name': 't',
             'gamma': self.gamma[3]},
        ]
        # Apply Wilson-Dirac operator for each direction
        for dir_info in directions:
            mu = dir_info['mu']
            axis = dir_info['axis']
            gamma_mu = dir_info['gamma']
            name = dir_info['name']
            if self.verbose:
                print(f"  Processing {name}-direction (axis={axis})...")
            # Give eo mask for parity decomposition
            if name == 'x_p':
                even_mask = give_eo_mask(xxxtzy_x_p=src_o, eo=0)
                odd_mask = give_eo_mask(xxxtzy_x_p=src_o, eo=1)
            # Extract gauge field for current direction
            U_e_mu = U_e[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x_p]
            U_o_dag_mu = U_o_dag[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x_p]
            # Term 1: (r - γ_μ) U_{x,μ} src_{x+μ}
            src_o_plus = torch_roll(
                src_o, shifts=-1, dims=axis)
            if name == 'x_p':  # -1 to 0 caused by parity decomposition\
                src_o_plus[..., even_mask] = src_o[...,
                                                   even_mask]  # parity(src)=(t+y+z+x)%2=1,eo(src)==(t+y+z)%2=0,so:x_p(src)%2=1,move_plus=0
            # Contract color indices: U_eo_mu * src_o_plus
            U_e_src_o_plus = torch_einsum(
                'Cctzyx,sctzyx->sCtzyx', U_e_mu, src_o_plus)
            # Apply (r - gamma_mu) in spin space
            term1 = torch_einsum(
                'Ss,sCtzyx->SCtzyx', (self.I - gamma_mu), U_e_src_o_plus)
            # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
            src_o_minus = torch_roll(
                src_o, shifts=1, dims=axis)
            if name == 'x_p':  # 1 to 0 caused by parity decomposition\
                src_o_minus[..., odd_mask] = src_o[...,
                                                   odd_mask]  # parity(src)=(t+y+z+x)%2=1,eo(src)==(t+y+z)%2=1,so:x_p(src)%2=0,move_minus=0
            U_o_dag_minus = torch_roll(
                U_o_dag_mu, shifts=1, dims=axis)
            if name == 'x_p':  # 1 to 0 caused by parity decomposition\
                U_o_dag_minus[..., odd_mask] = U_o_dag_mu[...,
                                                          odd_mask]  # parity(U)=(t+y+z+x)%2=1,eo(U)==(t+y+z)%2=1,so:x_p(U)%2=0,move_minus=0
            # Contract color indices: U_eo_dag_minus * src_o_minus
            U_o_dag_src_o_minus = torch_einsum(
                'Cctzyx,sctzyx->sCtzyx', U_o_dag_minus, src_o_minus)
            # Apply (r + gamma_mu) in spin space
            term2 = torch_einsum(
                'Ss,sCtzyx->SCtzyx', (self.I + gamma_mu), U_o_dag_src_o_minus)
            # Combine terms and subtract from dest_e
            hopping = term1 + term2
            dest_e -= self.kappa/self.u_0 * hopping
            if self.verbose:
                print(f"    Hopping term norm: {torch_norm(hopping).item()}")
        if self.verbose:
            print("Dirac operator application complete in eo")
            print(f"  Dest_e norm: {torch_norm(dest_e).item()}")
        return dest_e.clone()

    def give_wilson_oe(self,
                       src_e: torch.Tensor,
                       U_eo: torch.Tensor) -> torch.Tensor:
        """
        Apply Wilson-Dirac operator to source field in oe:
        $$
        2*\kappa*a*M_{x,y}=1-\kappa/u_0*\sum_{\mu}[(1-\gamma_{\mu})U_{x,\mu}\delta_{x,y-\mu}+(1+\gamma_{\mu})U_{x-\mu,\mu}^{\dag}\delta_{x,y+\mu}]
                          =1-\kappa/u_0*\sum_{\mu}[(1-\gamma_{\mu})U_{x,\mu}\delta_{x+\mu,y}+(1+\gamma_{\mu})U_{x-\mu,\mu}^{\dag}\delta_{x-\mu,y}]
        $$
        Args:
            src_e: Source field tensor [s, c, t, z, y, x_p]
            U_eo: Gauge field tensor [p, c, c, d, t, z, y, x_p]
        Returns:
            dest_o tensor [s, c, t, z, y, x_p]
        """
        if self.verbose:
            print("Applying Dirac operator in oe...")
            print(f"  Source shape: {src_e.shape}")
            print(f"  Gauge field shape: {U_eo.shape}")
            print(f"  Source norm: {torch_norm(src_e).item()}")
        # Compute adjoint gauge field (dagger conjugate)
        U_e = U_eo[0].clone()
        U_o = U_eo[1].clone()
        U_e_dag = U_e.permute(1, 0, 2, 3, 4, 5, 6).conj()
        # Initialize dest_e tensor
        # dest_o = src_e.clone() # move this I term to sitting term from this hopping term(origin wilson term)
        dest_o = torch.zeros_like(src_e)
        # Define directions with corresponding axes and gamma matrices
        directions = [
            {'mu': 0, 'axis': -1-0, 'name': 'x_p',
             'gamma': self.gamma[0]},
            {'mu': 1, 'axis': -1-1, 'name': 'y',
             'gamma': self.gamma[1]},
            {'mu': 2, 'axis': -1-2, 'name': 'z',
             'gamma': self.gamma[2]},
            {'mu': 3, 'axis': -1-3, 'name': 't',
             'gamma': self.gamma[3]},
        ]
        # Apply Wilson-Dirac operator for each direction
        for dir_info in directions:
            mu = dir_info['mu']
            axis = dir_info['axis']
            gamma_mu = dir_info['gamma']
            name = dir_info['name']
            if self.verbose:
                print(f"  Processing {name}-direction (axis={axis})...")
            # Give eo mask for parity decomposition
            if name == 'x_p':
                even_mask = give_eo_mask(xxxtzy_x_p=src_e, eo=0)
                odd_mask = give_eo_mask(xxxtzy_x_p=src_e, eo=1)
            # Extract gauge field for current direction
            U_e_dag_mu = U_e_dag[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x_p]
            U_o_mu = U_o[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x_p]
            # Term 1: (r - γ_μ) U_{x,μ} src_{x+μ}
            src_e_plus = torch_roll(
                src_e, shifts=-1, dims=axis)
            if name == 'x_p':  # -1 to 0 caused by parity decomposition\
                src_e_plus[..., odd_mask] = src_e[...,
                                                  odd_mask]  # parity(src)=(t+y+z+x)%2=0,eo(src)==(t+y+z)%2=1,so:x_p(src)%2=1,move_plus=0
            # Contract color indices: U_eo_mu * src_o_plus
            U_o_src_e_plus = torch_einsum(
                'Cctzyx,sctzyx->sCtzyx', U_o_mu, src_e_plus)
            # Apply (r - gamma_mu) in spin space
            term1 = torch_einsum(
                'Ss,sCtzyx->SCtzyx', (self.I - gamma_mu), U_o_src_e_plus)
            # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
            src_e_minus = torch_roll(
                src_e, shifts=1, dims=axis)
            if name == 'x_p':  # 1 to 0 caused by parity decomposition\
                src_e_minus[..., even_mask] = src_e[...,
                                                    even_mask]  # parity(src)=(t+y+z+x)%2=0,eo(src)==(t+y+z)%2=0,so:x_p(src)%2=0,move_minus=0
            U_e_dag_minus = torch_roll(
                U_e_dag_mu, shifts=1, dims=axis)
            if name == 'x_p':  # 1 to 0 caused by parity decomposition\
                U_e_dag_minus[..., even_mask] = U_e_dag_mu[...,
                                                           even_mask]  # parity(U)=(t+y+z+x)%2=0,eo(U)==(t+y+z)%2=0,so:x_p(U)%2=0,move_minus=0
            # Contract color indices: U_eo_dag_minus * src_o_minus
            U_e_dag_src_e_minus = torch_einsum(
                'Cctzyx,sctzyx->sCtzyx', U_e_dag_minus, src_e_minus)
            # Apply (r + gamma_mu) in spin space
            term2 = torch_einsum(
                'Ss,sCtzyx->SCtzyx', (self.I + gamma_mu), U_e_dag_src_e_minus)
            # Combine terms and subtract from dest_e
            hopping = term1 + term2
            dest_o -= self.kappa/self.u_0 * hopping
            if self.verbose:
                print(f"    Hopping term norm: {torch_norm(hopping).item()}")
        if self.verbose:
            print("Dirac operator application complete in oe")
            print(f"  Dest_o norm: {torch_norm(dest_o).item()}")
        return dest_o.clone()

    def give_wilson_eoeo(self,
                         dest_eo: torch.Tensor,
                         src_eo: torch.Tensor) -> torch.Tensor:
        # give_wilson_eo + give_wilson_oe + give_wilson_eoeo(I term) = give_wilson(complete)
        return dest_eo+src_eo


class wilson_mg(wilson):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int] = [8, 8, 8, 8],
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        Wilson-Dirac operator on a 4D lattice with SU(3) gauge fields. [8 wards version]
        Args:
            latt_size: Tuple (Lx, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3.
            kappa: Hopping parameter (controls fermion mass).
            u_0: Wilson parameter (usually 1.0).
            dtype: Data type for tensors.
            device: Device to run on (default: CPU).
            verbose: Enable verbose output for debugging.
        reference:
            [1](1-60).
        """
        super().__init__(latt_size=latt_size, kappa=kappa,
                         u_0=u_0, dtype=dtype, device=device, verbose=verbose)
        if self.verbose:
            print(f"Initializing Wilson with 8 wards:")
            print(f"  Lattice size: {latt_size} (x,y,z,t)")
            print(f"  Parameters: kappa={kappa}, u_0={u_0}")
            print(f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
            print(f"  Device: {self.device}")
        self.directions = [
            {'mu': 0, 'axis': -1-0, 'name': 'x',
             'gamma': self.gamma[0]},
            {'mu': 1, 'axis': -1-1, 'name': 'y',
             'gamma': self.gamma[1]},
            {'mu': 2, 'axis': -1-2, 'name': 'z',
             'gamma': self.gamma[2]},
            {'mu': 3, 'axis': -1-3, 'name': 't',
             'gamma': self.gamma[3]},
        ]  # ward in [xyzt]

    def give_hopping_plus(self, ward: int, U: torch.Tensor) -> torch.Tensor:
        dir_info = self.directions[ward]
        mu = dir_info['mu']
        gamma_mu = dir_info['gamma']
        name = dir_info['name']
        if self.verbose:
            print(f"@give_hopping_{name}_plus......")
        U_mu = U[..., mu, :, :, :, :]
        return - self.kappa/self.u_0 * torch_einsum(
            'Ss,Cctzyx->SCsctzyx', (self.I - gamma_mu), U_mu).reshape([12, 12]+list(U.shape[-4:])).clone()  # sc->e

    def give_wilson_plus(self, ward: int, src: torch.Tensor, hopping: torch.Tensor, src_tail: torch.Tensor = None) -> torch.Tensor:
        dir_info = self.directions[ward]
        axis = dir_info['axis']
        name = dir_info['name']
        if self.verbose:
            print(f"@give_wilson_{name}_plus......")
        try:
            src_plus = torch_roll(src, shifts=-1, dims=axis)
        except Exception as e:
            print(f"src.shape: {src.shape}")
            print(f"Error: {e}")
            exit()
        if src_tail != None:
            src_plus[slice_dim(dim=5, ward=ward, point=-1)] = src_tail.clone()
        return torch_einsum(
            'Eetzyx,etzyx->Etzyx', hopping, src_plus).clone()

    def give_hopping_minus(self, ward: int, U: torch.Tensor, U_head: torch.Tensor = None) -> torch.Tensor:
        dir_info = self.directions[ward]
        mu = dir_info['mu']
        axis = dir_info['axis']
        gamma_mu = dir_info['gamma']
        name = dir_info['name']
        if self.verbose:
            print(f"@give_hopping_{name}_minus......")
        U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj().clone()
        U_dag_mu = U_dag[..., mu, :, :, :, :]
        U_dag_minus = torch_roll(U_dag_mu, shifts=1, dims=axis)
        if U_head != None:
            U_head_dag = U_head.permute(1, 0, 2, 3, 4, 5).conj().clone()
            U_head_dag_mu = U_head_dag[..., mu, :, :, :]
            U_dag_minus[slice_dim(dim=6, ward=ward, point=0)
                        ] = U_head_dag_mu.clone()
        return - self.kappa/self.u_0 * torch_einsum(
            'Ss,Cctzyx->SCsctzyx', (self.I + gamma_mu), U_dag_minus).reshape([12, 12]+list(U.shape[-4:])).clone()  # sc->e

    def give_wilson_minus(self, ward: int, src: torch.Tensor, hopping: torch.Tensor, src_head: torch.Tensor = None) -> torch.Tensor:
        dir_info = self.directions[ward]
        axis = dir_info['axis']
        name = dir_info['name']
        if self.verbose:
            print(f"@give_wilson_{name}_minus......")
        src_minus = torch_roll(src, shifts=1, dims=axis)
        if src_head != None:
            src_minus[slice_dim(dim=5, ward=ward, point=0)] = src_head.clone()
        return torch_einsum(
            'Eetzyx,etzyx->Etzyx', hopping, src_minus).clone()


class clover(wilson):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int] = [8, 8, 8, 8],
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        The Clover term corrected by adding the Wilson-Dirac operator.
        Args:
            latt_size: Tuple (Lx, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3.
            kappa: Hopping parameter (controls fermion mass).
            u_0: Wilson parameter (usually 1.0).
            dtype: Data type for tensors.
            device: Device to run on (default: CPU).
            verbose: Enable verbose output for debugging.
        reference:
            [1](1-60).
        """
        super().__init__(latt_size=latt_size, kappa=kappa,
                         u_0=u_0, dtype=dtype, device=device, verbose=verbose)
        if self.verbose:
            print(f"Initializing Clover:")
            print(f"  Lattice size: {latt_size} (x,y,z,t)")
            print(f"  Parameters: kappa={kappa}, u_0={u_0}")
            print(f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
            print(f"  Device: {self.device}")
        # Precompute gamma_gamma matrices
        self.gamma_gamma = self._define_gamma_gamma_matrices()
        if self.verbose:
            print("Gamma-Gamma matrices initialized")

    def _define_gamma_gamma_matrices(self) -> torch.Tensor:
        """Define Dirac gamma_gamma matrices in Euclidean space."""
        gamma_gamma = torch.zeros(
            6, 4, 4, dtype=self.dtype, device=self.device)
        # gamma_gamma0 xy-direction)
        gamma_gamma[0] = torch_einsum(
            'ab,bc->ac', self.gamma[0], self.gamma[1])
        # gamma_gamma1 xz-direction)
        gamma_gamma[1] = torch_einsum(
            'ab,bc->ac', self.gamma[0], self.gamma[2])
        # gamma_gamma2 xt-direction)
        gamma_gamma[2] = torch_einsum(
            'ab,bc->ac', self.gamma[0], self.gamma[3])
        # gamma_gamma3 yz-direction)
        gamma_gamma[3] = torch_einsum(
            'ab,bc->ac', self.gamma[1], self.gamma[2])
        # gamma_gamma4 yt-direction)
        gamma_gamma[4] = torch_einsum(
            'ab,bc->ac', self.gamma[1], self.gamma[3])
        # gamma_gamma5 zt-direction)
        gamma_gamma[5] = torch_einsum(
            'ab,bc->ac', self.gamma[2], self.gamma[3])
        return gamma_gamma

    def make_clover(self, U: torch.Tensor) -> torch.Tensor:
        """
        Give Clover term:
        $$
        \frac{a^2\kappa}{u_0^4}\sum_{\mu<\nu}\sigma_{\mu \nu}F_{\mu \nu}\delta_{x,y}
        $$
        Args:
            U: Gauge field tensor [c, c, d, t, z, y, x].
        Returns:
            Clover term tensor [s, c, s, c, t, z, y, x].
        PS:
            For convenience, combine constant parameters of sigma and F to the final step (-0.125).
        """
        if self.verbose:
            print("Applying Dirac operator...")
            print(f"  Gauge field shape: {U.shape}")
        # Compute adjoint gauge field (dagger conjugate)
        U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj()
        # Initialize clover term tensor
        clover = torch.zeros((4, 3, 4, 3, self.Lt, self.Lz, self.Ly, self.Lx),
                             dtype=self.dtype, device=self.device)
        # Define directions with corresponding axes and gamma_gamma matrices
        directions = [
            {'mu': 0, 'nu': 1, 'axis_mu': -1-0, 'axis_nu': -1-1, 'name': 'xy',
                'gamma_gamma': self.gamma_gamma[0]},
            {'mu': 0, 'nu': 2, 'axis_mu': -1-0, 'axis_nu': -1-2, 'name': 'xz',
                'gamma_gamma': self.gamma_gamma[1]},
            {'mu': 0, 'nu': 3, 'axis_mu': -1-0, 'axis_nu': -1-3, 'name': 'xt',
                'gamma_gamma': self.gamma_gamma[2]},
            {'mu': 1, 'nu': 2, 'axis_mu': -1-1, 'axis_nu': -1-2, 'name': 'yz',
                'gamma_gamma': self.gamma_gamma[3]},
            {'mu': 1, 'nu': 3, 'axis_mu': -1-1, 'axis_nu': -1-3, 'name': 'yt',
                'gamma_gamma': self.gamma_gamma[4]},
            {'mu': 2, 'nu': 3, 'axis_mu': -1-2, 'axis_nu': -1-3, 'name': 'zt',
                'gamma_gamma': self.gamma_gamma[5]},
        ]
        # Give clover term for each direction
        for dir_info in directions:
            '''
            \begin{align*}
            F_{\mu \nu} = \frac{1}{a^2*8*i}\gamma_{\mu}\gamma_{\nu}[\\
            & u(x,\mu)u(x+\mu,\nu)u^{\dag}(x+\nu,\mu)u^{\dag}(x,\nu)\\
            &+ u(x,\nu)u^{\dag}(x-\mu+\nu,\mu)u^{\dag}(x-\mu,\nu)u(x-\mu,\mu)\\
            &+ u^{\dag}(x-\mu,\mu)u^{\dag}(x-\mu-\nu,\nu)u(x-\mu-\nu,\mu)u(x-\nu,\nu)\\
            &+ u^{\dag}(x-\nu,\nu)u(x-\nu,\mu)u(x-\nu+\mu,\nu)u^{\dag}(x,\mu)-BEFORE^{\dag}]
            \end{align*}
            '''
            F = torch.zeros((3, 3, self.Lt, self.Lz, self.Ly, self.Lx),
                            dtype=self.dtype, device=self.device)
            mu = dir_info['mu']
            nu = dir_info['nu']
            axis_mu = dir_info['axis_mu']
            axis_nu = dir_info['axis_nu']
            # $$ \sigma_{\mu,\nu} &= -i/2*(\gamma_{\mu}\gamma_{\nu} - \gamma_{\nu}\gamma_{\mu}) &= -i/2*2\gamma_{\mu}\gamma_{\nu}\\ $$
            sigma = dir_info['gamma_gamma']
            name = dir_info['name']
            if self.verbose:
                print(
                    f"  Processing {name}-direction (axis_mu={axis_mu},axis_nu={axis_nu})...")
            # Extract gauge field for current direction
            U_mu = U[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x]
            U_nu = U[..., nu, :, :, :, :]  # [c1, c2, t, z, y, x]
            U_dag_mu = U_dag[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x]
            U_dag_nu = U_dag[..., nu, :, :, :, :]  # [c1, c2, t, z, y, x]
            # $$U_1 &= u(x,\mu)u(x+\mu,\nu)u^{\dag}(x+\nu,\mu)u^{\dag}(x,\nu)                \\$$
            temp1 = torch_einsum('abtzyx,bctzyx->actzyx', U_mu,
                                 torch_roll(U_nu, shifts=-1, dims=axis_mu))
            temp2 = torch_einsum('abtzyx,bctzyx->actzyx', temp1,
                                 torch_roll(U_dag_mu, shifts=-1, dims=axis_nu))
            F += torch_einsum('abtzyx,bctzyx->actzyx', temp2, U_dag_nu)
            # $$U_2 &= u(x,\nu)u^{\dag}(x-\mu+\nu,\mu)u^{\dag}(x-\mu,\nu)u(x-\mu,\mu)        \\$$
            temp1 = torch_einsum('abtzyx,bctzyx->actzyx', U_nu,
                                 torch_roll(torch_roll(U_dag_mu, shifts=1, dims=axis_mu), shifts=-1, dims=axis_nu))
            temp2 = torch_einsum('abtzyx,bctzyx->actzyx', temp1,
                                 torch_roll(U_dag_nu, shifts=1, dims=axis_mu))
            F += torch_einsum('abtzyx,bctzyx->actzyx', temp2,
                              torch_roll(U_mu, shifts=1, dims=axis_mu))
            # $$U_3 &= u^{\dag}(x-\mu,\mu)u^{\dag}(x-\mu-\nu,\nu)u(x-\mu-\nu,\mu)u(x-\nu,\nu)\\$$
            temp1 = torch_einsum('abtzyx,bctzyx->actzyx', torch_roll(U_dag_mu, shifts=1, dims=axis_mu),
                                 torch_roll(torch_roll(U_dag_nu, shifts=1, dims=axis_mu), shifts=1, dims=axis_nu))
            temp2 = torch_einsum('abtzyx,bctzyx->actzyx', temp1,
                                 torch_roll(torch_roll(U_mu, shifts=1, dims=axis_mu), shifts=1, dims=axis_nu))
            F += torch_einsum('abtzyx,bctzyx->actzyx', temp2,
                              torch_roll(U_nu, shifts=1, dims=axis_nu))
            # $$U_4 &= u^{\dag}(x-\nu,\nu)u(x-\nu,\mu)u(x-\nu+\mu,\nu)u^{\dag}(x,\mu)        \\$$
            temp1 = torch_einsum('abtzyx,bctzyx->actzyx', torch_roll(U_dag_nu, shifts=1, dims=axis_nu),
                                 torch_roll(U_mu, shifts=1, dims=axis_nu))
            temp2 = torch_einsum('abtzyx,bctzyx->actzyx', temp1,
                                 torch_roll(torch_roll(U_nu, shifts=-1, dims=axis_mu), shifts=1, dims=axis_nu))
            F += torch_einsum('abtzyx,bctzyx->actzyx', temp2, U_dag_mu)
            # Give whole F
            F -= F.permute(1, 0, 2, 3, 4, 5).conj()  # -BEFORE^{\dag}
            # Multiply F with sigma
            sigmaF = torch_einsum(
                'Ss,Cctzyx->SCsctzyx', sigma, F)
            # Make Clover term
            clover += -0.125/self.u_0*self.kappa*sigmaF
            if self.verbose:
                print(f"    sigmaF term norm: {multi_norm(sigmaF).item()}")
        if self.verbose:
            print("Clover term complete")
            print(f"  clover norm: {multi_norm(clover).item()}")
        return clover.clone()

    def add_I(self, clover_term: torch.Tensor) -> torch.Tensor:
        _clover_term = clover_term.reshape(12, 12, -1).clone()
        if self.verbose:
            print('Clover is adding I......')
            print(f"_clover_term.shape:{_clover_term.shape}")
        eye = torch_eye(12, dtype=_clover_term.dtype,
                        device=_clover_term.device)
        _clover_term += eye.unsqueeze(-1)
        dest = _clover_term.reshape(clover_term.shape)
        if self.verbose:
            print(f"dest.shape:{dest.shape}")
        return dest.clone()

    def inverse(self, clover_term: torch.Tensor) -> torch.Tensor:
        _clover_term = clover_term.reshape(12, 12, -1).clone()
        if self.verbose:
            print('Clover is inversing......')
            print(f"_clover_term.shape:{_clover_term.shape}")
        for i in range(_clover_term.shape[-1]):
            _clover_term[:, :, i] = torch.linalg.inv(_clover_term[:, :, i])
        dest = _clover_term.reshape(clover_term.shape)
        if self.verbose:
            print(f"dest.shape:{dest.shape}")
        return dest.clone()

    def give_clover(self, src: torch.Tensor, clover_term: torch.Tensor) -> torch.Tensor:
        if self.verbose:
            print('Clover is giving......')
            print(f"src.shape:{src.shape}")
        dest = torch_einsum('SCsctzyx,sctzyx->SCtzyx', clover_term, src)
        if self.verbose:
            print(f"dest.shape:{dest.shape}")
        return dest.clone()


class clover_parity(clover):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int] = [8, 8, 8, 8],
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        The Clover term corrected by adding the Wilson-Dirac operator.
        Args:
            latt_size: Tuple (Lx_p, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3, p(parity)=2.
            kappa: Hopping parameter (controls fermion mass).
            u_0: Wilson parameter (usually 1.0).
            dtype: Data type for tensors.
            device: Device to run on (default: CPU).
            verbose: Enable verbose output for debugging.
        reference:
            [1](1-60);[1](1-160).
        """
        super().__init__(latt_size=latt_size, kappa=kappa,
                         u_0=u_0, dtype=dtype, device=device, verbose=verbose)
        self.Lx_p = self.Lx // 2
        if self.verbose:
            print(f"Initializing Clover with parity decomposition:")
            print(f"  Lattice size: {latt_size} (x,y,z,t)")
            print(f"  Lattice x with parity: {self.Lx_p}")
            print(f"  Parameters: kappa={kappa}, u_0={u_0}")
            print(f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
            print(f"  Device: {self.device}")

    def make_clover_eoeo(self, U_eo: torch.Tensor) -> torch.Tensor:
        return xxxtzyx2pxxxtzyx(self.make_clover(U=pxxxtzyx2xxxtzyx(U_eo.clone())))

    def add_I_eoeo(self, clover_eo: torch.Tensor) -> torch.Tensor:
        return xxxtzyx2pxxxtzyx(self.add_I(clover_term=pxxxtzyx2xxxtzyx(clover_eo.clone())))

    def inverse_eoeo(self, clover_eo: torch.Tensor) -> torch.Tensor:
        return xxxtzyx2pxxxtzyx(self.inverse(clover_term=pxxxtzyx2xxxtzyx(clover_eo.clone())))

    def give_clover_ee(self, src_e: torch.Tensor, clover_eo: torch.Tensor) -> torch.Tensor:
        return self.give_clover(src=src_e.clone(), clover_term=clover_eo[0].clone())

    def give_clover_oo(self, src_o: torch.Tensor, clover_eo: torch.Tensor) -> torch.Tensor:
        return self.give_clover(src=src_o.clone(), clover_term=clover_eo[1].clone())
