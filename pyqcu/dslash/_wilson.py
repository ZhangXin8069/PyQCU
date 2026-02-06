import torch
from pyqcu import _torch, lattice, tools


def give_wilson(src: torch.Tensor,
                U: torch.Tensor, kappa: float = 0.1,
                u_0: float = 1.0, with_I: bool = True, verbose: bool = False) -> torch.Tensor:
    """
    Apply Wilson-Dirac operator to source field:
    $$
    2*\kappa*a*M_{x,y}=1-\kappa/u_0*\sum_{\mu}[(1-\gamma_{\mu})U_{x,\mu}\delta_{x,y-\mu}+(1+\gamma_{\mu})U_{x-\mu,\mu}^{\dag}\delta_{x,y+\mu}]
                      =1-\kappa/u_0*\sum_{\mu}[(1-\gamma_{\mu})U_{x,\mu}\delta_{x+\mu,y}+(1+\gamma_{\mu})U_{x-\mu,\mu}^{\dag}\delta_{x-\mu,y}]
    $$
    """
    if verbose:
        print("PYQCU::DSLASH::WILSON:\n Applying Dirac operator...")
        print(f"PYQCU::DSLASH::WILSON:\n Source shape: {src.shape}")
        print(f"PYQCU::DSLASH::WILSON:\n Gauge field shape: {U.shape}")
        print(
            f"PYQCU::DSLASH::WILSON:\n Source norm: {_torch.norm(src).item()}")
    # Compute adjoint gauge field (dagger conjugate)
    U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj().clone()
    # Initialize dest tensor
    dest = src.clone() if with_I else torch.zeros_like(src)
    # Apply Wilson-Dirac operator for each direction
    I = lattice.I.to(src.device).type(src.dtype)
    for ward_key in lattice.ward_keys:
        ward = lattice.wards[ward_key]
        gamma_mu = lattice.gamma[ward].to(src.device).type(src.dtype)
        if verbose:
            print(
                f"PYQCU::DSLASH::WILSON:\n Processing {ward_key} (ward={ward})...")
        # Extract gauge field for current direction
        U_mu = U[..., ward, :, :, :, :]  # [c1, c2, x, y, z, t]
        U_dag_mu = U_dag[..., ward, :, :, :, :]  # [c1, c2, x, y, z, t]
        # Term 1: (r - γ_μ) U_{x,μ} src_{x+μ}
        src_plus = _torch.roll(src, shifts=-1, dims=ward)
        # Contract color indices: U_mu * src_plus
        U_src_plus = _torch.einsum('Ccxyzt,scxyzt->sCxyzt', U_mu, src_plus)
        # Apply (r - gamma_mu) in spin space
        term1 = _torch.einsum(
            'Ss,sCxyzt->SCxyzt', (I - gamma_mu), U_src_plus)
        # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
        src_minus = _torch.roll(src, shifts=1, dims=ward)
        U_dag_minus = _torch.roll(U_dag_mu, shifts=1, dims=ward)
        # Contract color indices: U_dag_minus * src_minus
        U_dag_src_minus = _torch.einsum(
            'Ccxyzt,scxyzt->sCxyzt', U_dag_minus, src_minus)
        # Apply (r + gamma_mu) in spin space
        term2 = _torch.einsum(
            'Ss,sCxyzt->SCxyzt', (I + gamma_mu), U_dag_src_minus)
        # Combine terms and subtract from dest
        hopping = term1 + term2
        dest -= kappa/u_0 * hopping
        if verbose:
            print(
                f"PYQCU::DSLASH::WILSON:\n Hopping term norm: {_torch.norm(hopping).item()}")
    if verbose:
        print("PYQCU::DSLASH::WILSON:\n Dirac operator application complete")
        print(
            f"PYQCU::DSLASH::WILSON:\n Dest norm: {_torch.norm(dest).item()}")
    return dest.clone()


def give_wilson_eo(
        src_o: torch.Tensor,
        U_eo: torch.Tensor, kappa: float = 0.1,
        u_0: float = 1.0, verbose: bool = False) -> torch.Tensor:
    if verbose:
        print("PYQCU::DSLASH::WILSON:\n Applying Dirac operator in eo...")
        print(f"PYQCU::DSLASH::WILSON:\n Source shape: {src_o.shape}")
        print(f"PYQCU::DSLASH::WILSON:\n Gauge field shape: {U_eo.shape}")
        print(
            f"PYQCU::DSLASH::WILSON:\n Source norm: {_torch.norm(src_o).item()}")
    # Compute adjoint gauge field (dagger conjugate)
    U_e = U_eo[0].clone()
    U_o = U_eo[1].clone()
    U_o_dag = U_o.permute(1, 0, 2, 3, 4, 5, 6).conj()
    # Initialize dest_e tensor
    # dest_e = src_o.clone() # move this I term to sitting term from this hopping term(origin wilson term)
    dest_e = torch.zeros_like(src_o)
    # Apply Wilson-Dirac operator for each direction
    I = lattice.I.to(src_o.device).type(src_o.dtype)
    for ward_key in lattice.ward_p_keys:
        ward = lattice.wards[ward_key]
        gamma_mu = lattice.gamma[ward].to(src_o.device).type(src_o.dtype)
        if verbose:
            print(
                f"PYQCU::DSLASH::WILSON:\n Processing {ward_key} (ward={ward})...")
        # Give eo mask for parity decomposition
        if ward_key == 't_p':
            even_mask = tools.give_eo_mask(___tzy_t_p=src_o, eo=0)
            odd_mask = tools.give_eo_mask(___tzy_t_p=src_o, eo=1)
        # Extract gauge field for current direction
        U_e_mu = U_e[..., ward, :, :, :, :]  # [c1, c2, x, y, z, t_p]
        U_o_dag_mu = U_o_dag[..., ward, :, :, :, :]  # [c1, c2, x, y, z, t_p]
        # Term 1: (r - γ_μ) U_{x,μ} src_{x+μ}
        src_o_plus = _torch.roll(
            src_o, shifts=-1, dims=ward)
        if ward_key == 't_p':  # -1 to 0 caused by parity decomposition\
            src_o_plus[..., even_mask] = src_o[...,
                                               even_mask]  # parity(src)=(x+y+z+t)%2=1,eo(src)==(x+y+z)%2=0,so:t_p(src)%2=1,move_plus=0
        # Contract color indices: U_eo_mu * src_o_plus
        U_e_src_o_plus = _torch.einsum(
            'Ccxyzt,scxyzt->sCxyzt', U_e_mu, src_o_plus)
        # Apply (r - gamma_mu) in spin space
        term1 = _torch.einsum(
            'Ss,sCxyzt->SCxyzt', (I - gamma_mu), U_e_src_o_plus)
        # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
        src_o_minus = _torch.roll(
            src_o, shifts=1, dims=ward)
        if ward_key == 't_p':  # 1 to 0 caused by parity decomposition\
            src_o_minus[..., odd_mask] = src_o[...,
                                               odd_mask]  # parity(src)=(x+y+z+t)%2=1,eo(src)==(x+y+z)%2=1,so:t_p(src)%2=0,move_minus=0
        U_o_dag_minus = _torch.roll(
            U_o_dag_mu, shifts=1, dims=ward)
        if ward_key == 't_p':  # 1 to 0 caused by parity decomposition\
            U_o_dag_minus[..., odd_mask] = U_o_dag_mu[...,
                                                      odd_mask]  # parity(U)=(x+y+z+t)%2=1,eo(U)==(x+y+z)%2=1,so:t_p(U)%2=0,move_minus=0
        # Contract color indices: U_eo_dag_minus * src_o_minus
        U_o_dag_src_o_minus = _torch.einsum(
            'Ccxyzt,scxyzt->sCxyzt', U_o_dag_minus, src_o_minus)
        # Apply (r + gamma_mu) in spin space
        term2 = _torch.einsum(
            'Ss,sCxyzt->SCxyzt', (I + gamma_mu), U_o_dag_src_o_minus)
        # Combine terms and subtract from dest_e
        hopping = term1 + term2
        dest_e -= kappa/u_0 * hopping
        if verbose:
            print(
                f"PYQCU::DSLASH::WILSON:\n Hopping term norm: {_torch.norm(hopping).item()}")
    if verbose:
        print("PYQCU::DSLASH::WILSON:\n Dirac operator application complete in eo")
        print(
            f"PYQCU::DSLASH::WILSON:\n Dest_e norm: {_torch.norm(dest_e).item()}")
    return dest_e.clone()


def give_wilson_oe(
    src_e: torch.Tensor,
    U_eo: torch.Tensor, kappa: float = 0.1,
        u_0: float = 1.0, verbose: bool = False) -> torch.Tensor:
    if verbose:
        print("Applying Dirac operator in oe...")
        print(f"PYQCU::DSLASH::WILSON:\n Source shape: {src_e.shape}")
        print(f"PYQCU::DSLASH::WILSON:\n Gauge field shape: {U_eo.shape}")
        print(
            f"PYQCU::DSLASH::WILSON:\n Source norm: {_torch.norm(src_e).item()}")
    # Compute adjoint gauge field (dagger conjugate)
    U_e = U_eo[0].clone()
    U_o = U_eo[1].clone()
    U_e_dag = U_e.permute(1, 0, 2, 3, 4, 5, 6).conj()
    # Initialize dest_e tensor
    # dest_o = src_e.clone() # move this I term to sitting term from this hopping term(origin wilson term)
    dest_o = torch.zeros_like(src_e)
    # Apply Wilson-Dirac operator for each direction
    I = lattice.I.to(src_e.device).type(src_e.dtype)
    for ward_key in lattice.ward_p_keys:
        ward = lattice.wards[ward_key]
        gamma_mu = lattice.gamma[ward].to(src_e.device).type(src_e.dtype)
        if verbose:
            print(
                f"PYQCU::DSLASH::WILSON:\n Processing {ward_key} (ward={ward})...")
        # Give eo mask for parity decomposition
        if ward_key == 't_p':
            even_mask = tools.give_eo_mask(___tzy_t_p=src_e, eo=0)
            odd_mask = tools.give_eo_mask(___tzy_t_p=src_e, eo=1)
        # Extract gauge field for current direction
        U_e_dag_mu = U_e_dag[..., ward, :, :, :, :]  # [c1, c2, x, y, z, t_p]
        U_o_mu = U_o[..., ward, :, :, :, :]  # [c1, c2, x, y, z, t_p]
        # Term 1: (r - γ_μ) U_{x,μ} src_{x+μ}
        src_e_plus = _torch.roll(
            src_e, shifts=-1, dims=ward)
        if ward_key == 't_p':  # -1 to 0 caused by parity decomposition\
            src_e_plus[..., odd_mask] = src_e[...,
                                              odd_mask]  # parity(src)=(x+y+z+t)%2=0,eo(src)==(x+y+z)%2=1,so:t_p(src)%2=1,move_plus=0
        # Contract color indices: U_eo_mu * src_o_plus
        U_o_src_e_plus = _torch.einsum(
            'Ccxyzt,scxyzt->sCxyzt', U_o_mu, src_e_plus)
        # Apply (r - gamma_mu) in spin space
        term1 = _torch.einsum(
            'Ss,sCxyzt->SCxyzt', (I - gamma_mu), U_o_src_e_plus)
        # Term 2: (r + γ_μ) U_{x-μ,μ}^† src_{x-μ}
        src_e_minus = _torch.roll(
            src_e, shifts=1, dims=ward)
        if ward_key == 't_p':  # 1 to 0 caused by parity decomposition\
            src_e_minus[..., even_mask] = src_e[...,
                                                even_mask]  # parity(src)=(x+y+z+t)%2=0,eo(src)==(x+y+z)%2=0,so:t_p(src)%2=0,move_minus=0
        U_e_dag_minus = _torch.roll(
            U_e_dag_mu, shifts=1, dims=ward)
        if ward_key == 't_p':  # 1 to 0 caused by parity decomposition\
            U_e_dag_minus[..., even_mask] = U_e_dag_mu[...,
                                                       even_mask]  # parity(U)=(x+y+z+t)%2=0,eo(U)==(x+y+z)%2=0,so:t_p(U)%2=0,move_minus=0
        # Contract color indices: U_eo_dag_minus * src_o_minus
        U_e_dag_src_e_minus = _torch.einsum(
            'Ccxyzt,scxyzt->sCxyzt', U_e_dag_minus, src_e_minus)
        # Apply (r + gamma_mu) in spin space
        term2 = _torch.einsum(
            'Ss,sCxyzt->SCxyzt', (I + gamma_mu), U_e_dag_src_e_minus)
        # Combine terms and subtract from dest_e
        hopping = term1 + term2
        dest_o -= kappa/u_0 * hopping
        if verbose:
            print(
                f"PYQCU::DSLASH::WILSON:\n Hopping term norm: {_torch.norm(hopping).item()}")
    if verbose:
        print("PYQCU::DSLASH::WILSON:\n Dirac operator application complete in oe")
        print(
            f"PYQCU::DSLASH::WILSON:\n Dest_o norm: {_torch.norm(dest_o).item()}")
    return dest_o.clone()


def give_wilson_eoeo(
        dest_eo: torch.Tensor,
        src_eo: torch.Tensor) -> torch.Tensor:
    # give_wilson_eo + give_wilson_oe + give_wilson_eoeo(I term) = give_wilson(complete)
    return dest_eo+src_eo


def give_hopping_plus(ward_key: str, U: torch.Tensor, kappa: float = 0.1,
                      u_0: float = 1.0, verbose: bool = False) -> torch.Tensor:
    ward = lattice.wards[ward_key]
    I = lattice.I.to(U.device).type(U.dtype)
    gamma_mu = lattice.gamma[ward].to(U.device).type(U.dtype)
    if verbose:
        print(f"PYQCU::DSLASH::WILSON:\n give_hopping_{ward_key}_plus......")
    U_mu = U[..., ward, :, :, :, :]
    return - kappa/u_0 * _torch.einsum(
        'Ss,Ccxyzt->SCscxyzt', (I - gamma_mu), U_mu).reshape([12, 12]+list(U.shape[-4:])).clone()  # sc->e


def give_wilson_plus(ward_key: str, src: torch.Tensor, hopping: torch.Tensor, src_tail: torch.Tensor = None, parity: int = None, verbose: bool = False) -> torch.Tensor:
    ward = lattice.wards[ward_key]
    if verbose:
        print(f"PYQCU::DSLASH::WILSON:\n give_wilson_{ward_key}_plus......")
    src_plus = _torch.roll(src, shifts=-1, dims=ward)
    if src_tail is not None:
        src_plus[tools.slice_dim(dims_num=5, ward=ward, point=-1)
                 ] = src_tail.clone()
    if parity == 0:
        odd_mask = tools.give_eo_mask(___tzy_t_p=src, eo=1)
        src_plus[..., odd_mask] = src[..., odd_mask]
    if parity == 1:
        even_mask = tools.give_eo_mask(___tzy_t_p=src, eo=0)
        src_plus[..., even_mask] = src[..., even_mask]
    return _torch.einsum(
        'Eexyzt,exyzt->Exyzt', hopping, src_plus).clone()


def give_hopping_minus(ward_key: str, U: torch.Tensor, U_head: torch.Tensor = None, kappa: float = 0.1,
                       u_0: float = 1.0, verbose: bool = False) -> torch.Tensor:
    ward = lattice.wards[ward_key]
    I = lattice.I.to(U.device).type(U.dtype)
    gamma_mu = lattice.gamma[ward].to(U.device).type(U.dtype)
    if verbose:
        print(f"PYQCU::DSLASH::WILSON:\n give_hopping_{ward_key}_minus......")
    U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj().clone()
    U_dag_mu = U_dag[..., ward, :, :, :, :]
    U_dag_minus = _torch.roll(U_dag_mu, shifts=1, dims=ward)
    if U_head is not None:
        U_head_dag = U_head.permute(1, 0, 2, 3, 4, 5).conj().clone()
        U_head_dag_mu = U_head_dag[..., ward, :, :, :]
        U_dag_minus[tools.slice_dim(dims_num=6, ward=ward, point=0)
                    ] = U_head_dag_mu.clone()
    return - kappa/u_0 * _torch.einsum(
        'Ss,Ccxyzt->SCscxyzt', (I + gamma_mu), U_dag_minus).reshape([12, 12]+list(U.shape[-4:])).clone()  # sc->e


def give_wilson_minus(ward_key: str, src: torch.Tensor, hopping: torch.Tensor, src_head: torch.Tensor = None, parity: int = None, verbose: bool = False) -> torch.Tensor:
    ward = lattice.wards[ward_key]
    if verbose:
        print(f"PYQCU::DSLASH::WILSON:\n give_wilson_{ward_key}_minus......")
    src_minus = _torch.roll(src, shifts=1, dims=ward)
    if src_head is not None:
        src_minus[tools.slice_dim(dims_num=5, ward=ward, point=0)
                  ] = src_head.clone()
    if parity == 0:
        even_mask = tools.give_eo_mask(___tzy_t_p=src, eo=0)
        src_minus[..., even_mask] = src[..., even_mask]
    if parity == 1:
        odd_mask = tools.give_eo_mask(___tzy_t_p=src, eo=1)
        src_minus[..., odd_mask] = src[..., odd_mask]
    return _torch.einsum(
        'Eexyzt,exyzt->Exyzt', hopping, src_minus).clone()
