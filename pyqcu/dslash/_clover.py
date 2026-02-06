import torch
from pyqcu import _torch, lattice, tools


def make_clover(U: torch.Tensor, kappa: float = 0.1,
                u_0: float = 1.0, verbose: bool = False) -> torch.Tensor:
    """
    Give Clover term:
    $$
    \frac{a^2\kappa}{u_0^4}\sum_{\mu<\nu}\sigma_{\mu \nu}F_{\mu \nu}\delta_{x,y}
    $$
    """
    if verbose:
        print("PYQCU::DSLASH::CLOVER:\n Applying Dirac operator...")
        print(f"PYQCU::DSLASH::CLOVER:\n Gauge field shape: {U.shape}")
    # Compute adjoint gauge field (dagger conjugate)
    U_dag = U.permute(1, 0, 2, 3, 4, 5, 6).conj()
    # Initialize clover term tensor
    clover = torch.zeros(
        (4, 3, 4, 3, U.shape[-4], U.shape[-3], U.shape[-2], U.shape[-1]), dtype=U.dtype, device=U.device)
    # Define directions with corresponding axes and gamma_gamma matrices
    ward_ward_keys = ['xy', 'xz', 'xt', 'yz', 'yt', 'zt']
    # Give clover term for each direction
    for ward_ward_key in ward_ward_keys:
        '''
        \begin{align*}
        F_{\mu \nu} = \frac{1}{a^2*8*i}\gamma_{\mu}\gamma_{\nu}[\\
        & u(x,\mu)u(x+\mu,\nu)u^{\dag}(x+\nu,\mu)u^{\dag}(x,\nu)\\
        &+ u(x,\nu)u^{\dag}(x-\mu+\nu,\mu)u^{\dag}(x-\mu,\nu)u(x-\mu,\mu)\\
        &+ u^{\dag}(x-\mu,\mu)u^{\dag}(x-\mu-\nu,\nu)u(x-\mu-\nu,\mu)u(x-\nu,\nu)\\
        &+ u^{\dag}(x-\nu,\nu)u(x-\nu,\mu)u(x-\nu+\mu,\nu)u^{\dag}(x,\mu)-BEFORE^{\dag}]
        \end{align*}
        '''
        ward_ward = lattice.ward_wards[ward_ward_key]
        F = torch.zeros((3, 3, U.shape[-4], U.shape[-3], U.shape[-2],
                        U.shape[-1]), dtype=U.dtype, device=U.device)
        mu = ward_ward['mu']
        nu = ward_ward['nu']
        # $$ \sigma_{\mu,\nu} &= -i/2*(\gamma_{\mu}\gamma_{\nu} - \gamma_{\nu}\gamma_{\mu}) &= -i/2*2\gamma_{\mu}\gamma_{\nu}\\ $$
        sigma = lattice.gamma_gamma[ward_ward['ward']].to(
            U.device).type(U.dtype)
        if verbose:
            print(
                f"PYQCU::DSLASH::CLOVER:\n Processing {ward_ward_key}-direction (mu={mu},nu={nu})...")
        # Extract gauge field for current direction
        U_mu = U[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x]
        U_nu = U[..., nu, :, :, :, :]  # [c1, c2, t, z, y, x]
        U_dag_mu = U_dag[..., mu, :, :, :, :]  # [c1, c2, t, z, y, x]
        U_dag_nu = U_dag[..., nu, :, :, :, :]  # [c1, c2, t, z, y, x]
        # $$U_1 &= u(x,\mu)u(x+\mu,\nu)u^{\dag}(x+\nu,\mu)u^{\dag}(x,\nu)                \\$$
        temp1 = _torch.einsum('abtzyx,bctzyx->actzyx', U_mu,
                              _torch.roll(U_nu, shifts=-1, dims=mu))
        temp2 = _torch.einsum('abtzyx,bctzyx->actzyx', temp1,
                              _torch.roll(U_dag_mu, shifts=-1, dims=nu))
        F += _torch.einsum('abtzyx,bctzyx->actzyx', temp2, U_dag_nu)
        # $$U_2 &= u(x,\nu)u^{\dag}(x-\mu+\nu,\mu)u^{\dag}(x-\mu,\nu)u(x-\mu,\mu)        \\$$
        temp1 = _torch.einsum('abtzyx,bctzyx->actzyx', U_nu,
                              _torch.roll(_torch.roll(U_dag_mu, shifts=1, dims=mu), shifts=-1, dims=nu))
        temp2 = _torch.einsum('abtzyx,bctzyx->actzyx', temp1,
                              _torch.roll(U_dag_nu, shifts=1, dims=mu))
        F += _torch.einsum('abtzyx,bctzyx->actzyx', temp2,
                           _torch.roll(U_mu, shifts=1, dims=mu))
        # $$U_3 &= u^{\dag}(x-\mu,\mu)u^{\dag}(x-\mu-\nu,\nu)u(x-\mu-\nu,\mu)u(x-\nu,\nu)\\$$
        temp1 = _torch.einsum('abtzyx,bctzyx->actzyx', _torch.roll(U_dag_mu, shifts=1, dims=mu),
                              _torch.roll(_torch.roll(U_dag_nu, shifts=1, dims=mu), shifts=1, dims=nu))
        temp2 = _torch.einsum('abtzyx,bctzyx->actzyx', temp1,
                              _torch.roll(_torch.roll(U_mu, shifts=1, dims=mu), shifts=1, dims=nu))
        F += _torch.einsum('abtzyx,bctzyx->actzyx', temp2,
                           _torch.roll(U_nu, shifts=1, dims=nu))
        # $$U_4 &= u^{\dag}(x-\nu,\nu)u(x-\nu,\mu)u(x-\nu+\mu,\nu)u^{\dag}(x,\mu)        \\$$
        temp1 = _torch.einsum('abtzyx,bctzyx->actzyx', _torch.roll(U_dag_nu, shifts=1, dims=nu),
                              _torch.roll(U_mu, shifts=1, dims=nu))
        temp2 = _torch.einsum('abtzyx,bctzyx->actzyx', temp1,
                              _torch.roll(_torch.roll(U_nu, shifts=-1, dims=mu), shifts=1, dims=nu))
        F += _torch.einsum('abtzyx,bctzyx->actzyx', temp2, U_dag_mu)
        # Give whole F
        F -= F.permute(1, 0, 2, 3, 4, 5).conj()  # -BEFORE^{\dag}
        # Multiply F with sigma
        sigmaF = _torch.einsum(
            'Ss,Cctzyx->SCsctzyx', sigma, F)
        # Make Clover term
        clover += -0.125/u_0*kappa*sigmaF
        if verbose:
            print(
                f"PYQCU::DSLASH::CLOVER:\n sigmaF term norm: {_torch.norm(sigmaF).item()}")
    if verbose:
        print("PYQCU::DSLASH::CLOVER:\n Clover term complete")
        print(
            f"PYQCU::DSLASH::CLOVER:\n clover norm: {_torch.norm(clover).item()}")
    return clover.clone()


def add_I(clover_term: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    _clover_term = clover_term.reshape(12, 12, -1).clone()
    if verbose:
        print('PYQCU::DSLASH::CLOVER:\n Clover is adding I......')
        print(
            f"PYQCU::DSLASH::CLOVER:\n _clover_term.shape:{_clover_term.shape}")
    eye = _torch.eye(12, dtype=_clover_term.dtype,
                     device=_clover_term.device)
    _clover_term += eye.unsqueeze(-1)
    dest = _clover_term.reshape(clover_term.shape)
    if verbose:
        print(f"PYQCU::DSLASH::CLOVER:\n dest.shape:{dest.shape}")
    return dest.clone()


def inverse(clover_term: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    _clover_term = clover_term.reshape(12, 12, -1).clone()
    if verbose:
        print('PYQCU::DSLASH::CLOVER:\n Clover is inversing......')
        print(
            f"PYQCU::DSLASH::CLOVER:\n _clover_term.shape:{_clover_term.shape}")
    for i in range(_clover_term.shape[-1]):
        _clover_term[:, :, i] = torch.linalg.inv(_clover_term[:, :, i])
    dest = _clover_term.reshape(clover_term.shape)
    if verbose:
        print(f"dest.shape:{dest.shape}")
    return dest.clone()


def give_clover(src: torch.Tensor, clover_term: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    if verbose:
        print('PYQCU::DSLASH::CLOVER:\n Clover is giving......')
        print(f"PYQCU::DSLASH::CLOVER:\n src.shape:{src.shape}")
    dest = _torch.einsum('SCsctzyx,sctzyx->SCtzyx', clover_term, src)
    if verbose:
        print(f"PYQCU::DSLASH::CLOVER:\n dest.shape:{dest.shape}")
    return dest.clone()


def make_clover_eoeo(U_eo: torch.Tensor) -> torch.Tensor:
    return tools.___xyzt2p___xyzt(make_clover(U=tools.p___xyzt2___xyzt(U_eo.clone())))


def add_I_eoeo(clover_eo: torch.Tensor) -> torch.Tensor:
    return tools.___xyzt2p___xyzt(add_I(clover_term=tools.p___xyzt2___xyzt(clover_eo.clone())))


def inverse_eoeo(clover_eo: torch.Tensor) -> torch.Tensor:
    return tools.___xyzt2p___xyzt(inverse(clover_term=tools.p___xyzt2___xyzt(clover_eo.clone())))


def give_clover_ee(src_e: torch.Tensor, clover_eo: torch.Tensor) -> torch.Tensor:
    return give_clover(src=src_e.clone(), clover_term=clover_eo[0].clone())


def give_clover_oo(src_o: torch.Tensor, clover_eo: torch.Tensor) -> torch.Tensor:
    return give_clover(src=src_o.clone(), clover_term=clover_eo[1].clone())
