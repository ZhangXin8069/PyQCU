import torch
import torch.nn as nn
from math import sqrt
from typing import Tuple, Optional


class wilson(nn.Module):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int],
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
        self.I = torch.eye(4, dtype=self.dtype, device=self.device)
        self.verbose = verbose
        # Determine real dtype based on complex dtype
        self.real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
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

    def generate_gauge_field(self,
                             sigma: float = 0.1,
                             seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random SU(3) gauge field using Gaussian distribution.
        Args:
            sigma: Width of Gaussian distribution (controls randomness).
            seed: Random seed for reproducibility.
        Returns:
            U: Gauge field tensor [c, c, d, t, z, y, x].
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
        U = torch.zeros((3, 3, 4, self.Lt, self.Lz, self.Ly, self.Lx),
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
                        for d in range(4):  # 4 directions
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
                    U: torch.Tensor) -> torch.Tensor:
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
            print(f"  Source norm: {torch.norm(src).item()}")
        # Compute adjoint gauge field (dagger conjugate)
        U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj()
        # Initialize dest tensor
        dest = src.clone()
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
            src_plus = torch.roll(src, shifts=-1, dims=axis)
            # Contract color indices: U_mu * src_plus
            U_src_plus = torch.einsum('Cctzyx,sctzyx->sCtzyx', U_mu, src_plus)
            # Apply (r - gamma_mu) in spin space
            term1 = torch.einsum(
                'Ss,sctzyx->Sctzyx', (self.I - gamma_mu), U_src_plus)
            # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
            src_minus = torch.roll(src, shifts=1, dims=axis)
            U_dag_minus = torch.roll(U_dag_mu, shifts=1, dims=axis)
            # Contract color indices: U_dag_minus * src_minus
            U_dag_src_minus = torch.einsum(
                'Cctzyx,sctzyx->sCtzyx', U_dag_minus, src_minus)
            # Apply (r + gamma_mu) in spin space
            term2 = torch.einsum(
                'Ss,sctzyx->Sctzyx', (self.I + gamma_mu), U_dag_src_minus)
            # Combine terms and subtract from dest
            hopping = term1 + term2
            dest -= self.kappa/self.u_0 * hopping
            if self.verbose:
                print(f"    Hopping term norm: {torch.norm(hopping).item()}")
        if self.verbose:
            print("Dirac operator application complete")
            print(f"  Dest norm: {torch.norm(dest).item()}")
        return dest


class clover(wilson):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int],
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
                         u_0=u_0, dtype=dtype, device=device, verbose=False)
        self.verbose = verbose
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
        gamma_gamma[0] = torch.einsum(
            'ab,bc->ac', self.gamma[0], self.gamma[1])
        # gamma_gamma1 xz-direction)
        gamma_gamma[1] = torch.einsum(
            'ab,bc->ac', self.gamma[0], self.gamma[2])
        # gamma_gamma2 xt-direction)
        gamma_gamma[2] = torch.einsum(
            'ab,bc->ac', self.gamma[0], self.gamma[3])
        # gamma_gamma3 yz-direction)
        gamma_gamma[3] = torch.einsum(
            'ab,bc->ac', self.gamma[1], self.gamma[2])
        # gamma_gamma4 yt-direction)
        gamma_gamma[4] = torch.einsum(
            'ab,bc->ac', self.gamma[1], self.gamma[3])
        # gamma_gamma5 zt-direction)
        gamma_gamma[5] = torch.einsum(
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
            temp1 = torch.einsum('abtzyx,bctzyx->actzyx', U_mu,
                                 torch.roll(U_nu, shifts=-1, dims=axis_mu))
            temp2 = torch.einsum('abtzyx,bctzyx->actzyx', temp1,
                                 torch.roll(U_dag_mu, shifts=-1, dims=axis_nu))
            F += torch.einsum('abtzyx,bctzyx->actzyx', temp2, U_dag_nu)
            # $$U_2 &= u(x,\nu)u^{\dag}(x-\mu+\nu,\mu)u^{\dag}(x-\mu,\nu)u(x-\mu,\mu)        \\$$
            temp1 = torch.einsum('abtzyx,bctzyx->actzyx', U_nu,
                                 torch.roll(torch.roll(U_dag_mu, shifts=1, dims=axis_mu), shifts=-1, dims=axis_nu))
            temp2 = torch.einsum('abtzyx,bctzyx->actzyx', temp1,
                                 torch.roll(U_dag_nu, shifts=1, dims=axis_mu))
            F += torch.einsum('abtzyx,bctzyx->actzyx', temp2,
                              torch.roll(U_mu, shifts=1, dims=axis_mu))
            # $$U_3 &= u^{\dag}(x-\mu,\mu)u^{\dag}(x-\mu-\nu,\nu)u(x-\mu-\nu,\mu)u(x-\nu,\nu)\\$$
            temp1 = torch.einsum('abtzyx,bctzyx->actzyx', torch.roll(U_dag_mu, shifts=1, dims=axis_mu),
                                 torch.roll(torch.roll(U_dag_nu, shifts=1, dims=axis_mu), shifts=1, dims=axis_nu))
            temp2 = torch.einsum('abtzyx,bctzyx->actzyx', temp1,
                                 torch.roll(torch.roll(U_mu, shifts=1, dims=axis_mu), shifts=1, dims=axis_nu))
            F += torch.einsum('abtzyx,bctzyx->actzyx', temp2,
                              torch.roll(U_nu, shifts=1, dims=axis_nu))
            # $$U_4 &= u^{\dag}(x-\nu,\nu)u(x-\nu,\mu)u(x-\nu+\mu,\nu)u^{\dag}(x,\mu)        \\$$
            temp1 = torch.einsum('abtzyx,bctzyx->actzyx', torch.roll(U_dag_nu, shifts=1, dims=axis_nu),
                                 torch.roll(U_mu, shifts=1, dims=axis_nu))
            temp2 = torch.einsum('abtzyx,bctzyx->actzyx', temp1,
                                 torch.roll(torch.roll(U_nu, shifts=-1, dims=axis_mu), shifts=1, dims=axis_nu))
            F += torch.einsum('abtzyx,bctzyx->actzyx', temp2, U_dag_mu)
            # Give whole F
            F -= F.permute(1, 0, 2, 3, 4, 5).conj()  # -BEFORE^{\dag}
            # Multiply F with sigma
            sigmaF = torch.einsum(
                'Ss,Cctzyx->SCsctzyx', sigma, F)
            # Make Clover term
            clover += -0.125/self.u_0*self.kappa*sigmaF
            if self.verbose:
                print(f"    sigmaF term norm: {torch.norm(sigmaF).item()}")
        if self.verbose:
            print("Clover term complete")
            print(f"  clover norm: {torch.norm(clover).item()}")
        return clover

    def add_I(self, clover: torch.Tensor) -> torch.Tensor:
        _clover = clover.reshape(12, 12, -1)
        if self.verbose:
            print('Clover is adding I......')
            print(f"_clover.shape:{_clover.shape}")
        for i in range(_clover.shape[-1]):
            _clover[:, :, i] += torch.eye(12, 12,
                                          dtype=_clover.dtype, device=_clover.device)
        dest = _clover.reshape(clover.shape)
        if self.verbose:
            print(f"dest.shape:{dest.shape}")
        return dest

    def inverse(self, clover: torch.Tensor) -> torch.Tensor:
        _clover = clover.reshape(12, 12, -1)
        if self.verbose:
            print('Clover is inversing......')
            print(f"_clover.shape:{_clover.shape}")
        for i in range(_clover.shape[-1]):
            _clover[:, :, i] = torch.linalg.inv(_clover[:, :, i])
        dest = _clover.reshape(clover.shape)
        if self.verbose:
            print(f"dest.shape:{dest.shape}")
        return dest

    def give_clover(self, src: torch.Tensor, clover: torch.Tensor) -> torch.Tensor:
        if self.verbose:
            print('Clover is giving......')
            print(f"src.shape:{src.shape}")
        dest = torch.einsum('SCsctzyx,sctzyx->SCtzyx', clover, src)
        if self.verbose:
            print(f"dest.shape:{dest.shape}")
        return dest
