from pyqcu.ascend.dslash_parity import *

class hopping(wilson_parity):
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
            [4]
        addition:
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


class sitting(clover_parity):
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