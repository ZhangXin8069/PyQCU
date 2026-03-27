import torch
"""
    Copy from https://github.com/IHEP-LQCD/EasyDistillation/blob/master/lattice/generator/elemental.py
"""


def stout_smear(U: torch.Tensor, nstep: int = 20, rho: float = 0.12):
    for _ in range(nstep):
        Q = torch.zeros_like(U)
        for mu in range(4 - 1):
            Q_mu = torch.zeros_like(Q[:, :, mu, :, :, :, :])
            for nu in range(4 - 1):
                if mu != nu:
                    U_mu = U[:, :, mu, :, :, :, :]
                    U_nu = U[:, :, nu, :, :, :, :]
                    U_nu_conj = U[:, :, nu, :, :, :, :].permute(
                        1, 0, 2, 3, 4, 5).conj()
                    Q_mu += torch.einsum(
                        "abxyzt,bcxyzt,dcxyzt->adxyzt",
                        U_nu,
                        torch.roll(U_mu, -1, -4+nu),
                        torch.roll(U_nu_conj, -1, -4+mu),
                    )
                    Q_mu += torch.einsum(
                        "baxyzt,bcxyzt,cdxyzt->adxyzt",
                        torch.roll(U_nu_conj, +1, -4+nu),
                        torch.roll(U_mu, +1, -4+nu),
                        torch.roll(torch.roll(
                            U_nu, +1, -4+nu), -1, -4+mu),
                    )
            Q[:, :, mu, :, :, :, :] = Q_mu.clone()
        Q = torch.einsum("abDxyzt,cbDxyzt->acDxyzt", rho * Q, U.conj())
        Q = 0.5j * (torch.einsum("abDxyzt->baDxyzt", Q.conj()) - Q)
        Q -= 1 / 3 * torch.einsum("aaDxyzt,bc->bcDxyzt", Q, torch.eye(3))
        c0 = torch.einsum("abDxyzt,bcDxyzt,caDxyzt->Dxyzt", Q, Q, Q).real / 3
        c1 = torch.einsum("abDxyzt,baDxyzt->Dxyzt", Q, Q).real / 2

        c0_max = 2 * (c1 / 3) ** (3 / 2)
        parity = c0 < 0
        c0 = torch.abs(c0)
        theta = torch.arccos(c0 / c0_max)
        u = (c1 / 3) ** 0.5 * torch.cos(theta / 3)
        w = c1**0.5 * torch.sin(theta / 3)
        u_sq = u**2
        w_sq = w**2
        e_iu = torch.exp(-1j * u)
        e_2iu = torch.exp(2j * u)
        cos_w = torch.cos(w)
        sinc_w = 1 - w_sq / 6 * \
            (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
        large = torch.abs(w) > 0.05
        w_large = w[large]
        sinc_w[large] = torch.sin(w_large) / w_large
        f_denom = 1 / (9 * u_sq - w_sq)
        f0 = ((u_sq - w_sq) * e_2iu + e_iu * (8 * u_sq * cos_w +
              2j * u * (3 * u_sq + w_sq) * sinc_w)) * f_denom
        f1 = (2 * u * e_2iu - e_iu * (2 * u * cos_w -
              1j * (3 * u_sq - w_sq) * sinc_w)) * f_denom
        f2 = (e_2iu - e_iu * (cos_w + 3j * u * sinc_w)) * f_denom
        f0[parity] = f0[parity].conj()
        f1[parity] = -f1[parity].conj()
        f2[parity] = f2[parity].conj()

        f0 = torch.einsum("Dxyzt,ab->abDxyzt", f0, torch.eye(3))
        f1 = torch.einsum("Dxyzt,abDxyzt->abDxyzt", f1, Q)
        f2 = torch.einsum("Dxyzt,abDxyzt,bcDxyzt->acDxyzt", f2, Q, Q)
        dest = torch.einsum("abDxyzt,bcDxyzt->acDxyzt", f0 + f1 + f2, U)
    return dest


"""
    Copy from https://github.com/IHEP-LQCD/EasyDistillation/blob/master/lattice/generator/elemental.py
"""


def stout_smear_single(U: torch.Tensor, nstep: int = 20, rho: float = 0.12):
    return stout_smear(U=U, nstep=nstep, rho=rho)
