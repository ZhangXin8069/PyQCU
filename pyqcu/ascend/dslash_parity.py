from pyqcu.ascend.dslash import *


class wilson_parity(wilson):
    def __init__(self,
                 latt_size: Tuple[int, int, int, int],
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype: torch.dtype = torch.complex128,
                 device: torch.device = None,
                 verbose: bool = False):
        """
        Wilson-Dirac operator on a 4D lattice with SU(3) gauge fields with parity decomposition
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
        self.latt_size = latt_size
        self.Lx_p, self.Ly, self.Lz, self.Lt = latt_size
        self.Lx_p //= 2
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
