from pyqcu.ascend.dslash import *


def xxxtzyx2pxxxtzyx(input_array: torch.Tensor) -> torch.Tensor:
    print("@xxxtzyx2pxxxtzyx......")
    shape = input_array.shape
    dtype = input_array.dtype
    device = input_array.device
    prefix_shape = shape[:-4]
    t, z, y, x = shape[-4:]
    # Create coordinate grids
    coords = torch.meshgrid(
        torch.arange(t, device=device),
        torch.arange(z, device=device),
        torch.arange(y, device=device),
        torch.arange(x, device=device),
        indexing='ij'
    )
    # Sum coordinates to determine checkerboard pattern
    sums = coords[0] + coords[1] + coords[2] + coords[3]
    even_mask = (sums % 2 == 0)
    odd_mask = ~even_mask
    # Initialize output tensor with two channels
    splited_array = torch.zeros(
        (2, *prefix_shape, t, z, y, x//2),
        dtype=dtype,
        device=device
    )
    # Reshape masked elements and assign to output
    splited_array[0] = input_array[..., even_mask].reshape(
        *prefix_shape, t, z, y, x//2)
    splited_array[1] = input_array[..., odd_mask].reshape(
        *prefix_shape, t, z, y, x//2)
    print(f"Splited Array Shape: {splited_array.shape}")
    return splited_array


def pxxxtzyx2xxxtzyx(input_array: torch.Tensor) -> torch.Tensor:
    print("@pxxxtzyx2xxxtzyx......")
    shape = input_array.shape
    dtype = input_array.dtype
    device = input_array.device
    prefix_shape = shape[1:-4]
    t, z, y, x_half = shape[-4:]
    x = x_half * 2  # Restore original x dimension
    # Create coordinate grids for original shape
    coords = torch.meshgrid(
        torch.arange(t, device=device),
        torch.arange(z, device=device),
        torch.arange(y, device=device),
        torch.arange(x, device=device),
        indexing='ij'
    )
    # Sum coordinates to determine checkerboard pattern
    sums = coords[0] + coords[1] + coords[2] + coords[3]
    even_mask = (sums % 2 == 0)
    odd_mask = ~even_mask
    # Initialize output tensor with original shape
    restored_array = torch.zeros(
        (*prefix_shape, t, z, y, x),
        dtype=dtype,
        device=device
    )
    # Assign values from input array using masks
    restored_array[..., even_mask] = input_array[0].reshape(*prefix_shape, -1)
    restored_array[..., odd_mask] = input_array[1].reshape(*prefix_shape, -1)
    print(f"Restored Array Shape: {restored_array.shape}")
    return restored_array


class wilson_parity(wilson):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int],
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        Wilson-Dirac operator on a 4D lattice with SU_eo(3) gauge fields with parity decomposition
        Args:
            latt_size: Tuple (Lx_p, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3, p(parity)=2
            kappa: Hopping parameter (controls fermion mass)
            u_0: Wilson parameter (usually 1.0)
            dtype: Data type for tensors
            device: Device to run on (default: CPU)
            verbose: Enable verbose output for debugging
        reference:
            [1](1-60);[1](1-160)
        """
        super().__init__(latt_size=latt_size, kappa=kappa,
                         u_0=u_0, dtype=dtype, device=device, verbose=False)
        self.Lx_p = self.Lx // 2
        self.verbose = verbose
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
            print(f"  Source norm: {torch.norm(src_o).item()}")
        # Compute adjoint gauge field (dagger conjugate)
        U_e = U_eo[0]
        U_o = U_eo[1]
        U_o_dag = U_o.permute(1, 0, 2, 3, 4, 5, 6).conj()
        # Initialize dest_e tensor
        dest_e = src_o.clone()
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
            # Extract gauge field for current direction
            U_e_mu = U_e[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x_p]
            U_o_dag_mu = U_o_dag[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x_p]
            # Term 1: (r - γ_μ) U_{x,μ} src_{x+μ}
            src_o_plus = torch.roll(
                src_o, shifts=-0, dims=axis)  # -1 to 0 caused by parity decomposition\
            # Contract color indices: U_eo_mu * src_o_plus
            U_e_src_o_plus = torch.einsum(
                'Cctzyx,sctzyx->sCtzyx', U_e_mu, src_o_plus)
            # Apply (r - gamma_mu) in spin space
            term1 = torch.einsum(
                'Ss,sctzyx->Sctzyx', (self.I - gamma_mu), U_e_src_o_plus)
            # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
            src_o_minus = torch.roll(
                src_o, shifts=1, dims=axis)  # 1 to 1 caused by parity decomposition\
            U_o_dag_minus = torch.roll(
                U_o_dag_mu, shifts=0, dims=axis)  # 1 to 0 caused by parity decomposition\
            # Contract color indices: U_eo_dag_minus * src_o_minus
            U_o_dag_src_o_minus = torch.einsum(
                'Cctzyx,sctzyx->sCtzyx', U_o_dag_minus, src_o_minus)
            # Apply (r + gamma_mu) in spin space
            term2 = torch.einsum(
                'Ss,sctzyx->Sctzyx', (self.I + gamma_mu), U_o_dag_src_o_minus)
            # Combine terms and subtract from dest_e
            hopping = term1 + term2
            dest_e -= self.kappa/self.u_0 * hopping
            if self.verbose:
                print(f"    Hopping term norm: {torch.norm(hopping).item()}")
        if self.verbose:
            print("Dirac operator application complete in eo")
            print(f"  Dest_e norm: {torch.norm(dest_e).item()}")
        return dest_e

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
            print("Applying Dirac operator in eo...")
            print(f"  Source shape: {src_e.shape}")
            print(f"  Gauge field shape: {U_eo.shape}")
            print(f"  Source norm: {torch.norm(src_e).item()}")
        # Compute adjoint gauge field (dagger conjugate)
        U_e = U_eo[0]
        U_o = U_eo[1]
        U_e_dag = U_e.permute(1, 0, 2, 3, 4, 5, 6).conj()
        # Initialize dest_e tensor
        dest_o = src_e.clone()
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
            # Extract gauge field for current direction
            U_e_dag_mu = U_e_dag[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x_p]
            U_o_mu = U_o[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x_p]
            # Term 1: (r - γ_μ) U_{x,μ} src_{x+μ}
            src_e_plus = torch.roll(
                src_e, shifts=-1, dims=axis)  # -1 to -1 caused by parity decomposition\
            # Contract color indices: U_eo_mu * src_o_plus
            U_o_src_e_plus = torch.einsum(
                'Cctzyx,sctzyx->sCtzyx', U_o_mu, src_e_plus)
            # Apply (r - gamma_mu) in spin space
            term1 = torch.einsum(
                'Ss,sctzyx->Sctzyx', (self.I - gamma_mu), U_o_src_e_plus)
            # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
            src_e_minus = torch.roll(
                src_e, shifts=0, dims=axis)  # 1 to 0 caused by parity decomposition\
            U_e_dag_minus = torch.roll(
                U_e_dag_mu, shifts=0, dims=axis)  # 1 to 0 caused by parity decomposition\
            # Contract color indices: U_eo_dag_minus * src_o_minus
            U_e_dag_src_e_minus = torch.einsum(
                'Cctzyx,sctzyx->sCtzyx', U_e_dag_minus, src_e_minus)
            # Apply (r + gamma_mu) in spin space
            term2 = torch.einsum(
                'Ss,sctzyx->Sctzyx', (self.I + gamma_mu), U_e_dag_src_e_minus)
            # Combine terms and subtract from dest_e
            hopping = term1 + term2
            dest_e -= self.kappa/self.u_0 * hopping
            if self.verbose:
                print(f"    Hopping term norm: {torch.norm(hopping).item()}")
        if self.verbose:
            print("Dirac operator application complete in oe")
            print(f"  Dest_o norm: {torch.norm(dest_o).item()}")
        return dest_o


class clover_parity(clover):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int],
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        The Clover term corrected by adding the Wilson-Dirac operator
        Args:
            latt_size: Tuple (Lx_p, Ly, Lz, Lt) specifying lattice dimensions, then s=4, d=4, c=3, p(parity)=2
            kappa: Hopping parameter (controls fermion mass)
            u_0: Wilson parameter (usually 1.0)
            dtype: Data type for tensors
            device: Device to run on (default: CPU)
            verbose: Enable verbose output for debugging
        reference:
            [1](1-60);[1](1-160)
        """
        super().__init__(latt_size=latt_size, kappa=kappa,
                         u_0=u_0, dtype=dtype, device=device, verbose=False)
        self.Lx_p = self.Lx // 2
        self.verbose = verbose
        if self.verbose:
            print(f"Initializing Clover with parity decomposition:")
            print(f"  Lattice size: {latt_size} (x,y,z,t)")
            print(f"  Lattice x with parity: {self.Lx_p}")
            print(f"  Parameters: kappa={kappa}, u_0={u_0}")
            print(f"  Complex dtype: {dtype}, Real dtype: {self.real_dtype}")
            print(f"  Device: {self.device}")

    def make_clover_eo(self, U_eo: torch.Tensor) -> torch.Tensor:
        return xxxtzyx2pxxxtzyx(self.make_clover(U=pxxxtzyx2xxxtzyx(U_eo)))

    def add_eye_eo(self, clover_eo: torch.Tensor) -> torch.Tensor:
        return xxxtzyx2pxxxtzyx(self.add_eye(clover=pxxxtzyx2xxxtzyx(clover_eo)))

    def inverse_eo(self, clover_eo: torch.Tensor) -> torch.Tensor:
        return xxxtzyx2pxxxtzyx(self.inverse(clover=pxxxtzyx2xxxtzyx(clover_eo)))

    def give_clover_ee(self, src_e: torch.Tensor, clover_eo: torch.Tensor) -> torch.Tensor:
        return self.give_clover(src=src_e, clover=clover_eo[0])

    def give_clover_oo(self, src_o: torch.Tensor, clover_eo: torch.Tensor) -> torch.Tensor:
        return self.give_clover(src=src_o, clover=clover_eo[1])
