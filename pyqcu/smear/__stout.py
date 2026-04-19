def _stout_smear_ndarray_naive(self, nstep, rho):
    backend = get_backend()
    U = backend.ascontiguousarray(self._U)
    for _ in range(nstep):
        Q = backend.zeros_like(U)
        for mu in range(Nd - 1):
            for nu in range(Nd - 1):
                if mu != nu:
                    Q[mu] += contract(
                        "...ab,...bc,...dc->...ad",
                        U[nu],
                        backend.roll(U[mu], -1, 3 - nu),
                        backend.roll(U[nu], -1, 3 - mu).conj(),
                    )
                    Q[mu] += contract(
                        "...ba,...bc,...cd->...ad",
                        backend.roll(U[nu], +1, 3 - nu).conj(),
                        backend.roll(U[mu], +1, 3 - nu),
                        backend.roll(backend.roll(
                            U[nu], +1, 3 - nu), -1, 3 - mu),
                    )
        Q = contract("...ab,...cb->...ac", rho * Q, U.conj())
        Q = 0.5j * (contract("...ab->...ba", Q.conj()) - Q)
        Q -= 1 / Nc * contract("...aa,bc->...bc", Q, backend.identity(Nc))
        c0 = contract("...ab,...bc,...ca->...", Q, Q, Q).real / 3
        c1 = contract("...ab,...ba->...", Q, Q).real / 2
        c0_max = 2 * (c1 / 3) ** (3 / 2)
        parity = c0 < 0
        c0 = backend.abs(c0)
        theta = backend.arccos(c0 / c0_max)
        u = (c1 / 3) ** 0.5 * backend.cos(theta / 3)
        w = c1**0.5 * backend.sin(theta / 3)
        u_sq = u**2
        w_sq = w**2
        e_iu = backend.exp(-1j * u)
        e_2iu = backend.exp(2j * u)
        cos_w = backend.cos(w)
        sinc_w = 1 - w_sq / 6 * \
            (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
        large = backend.abs(w) > 0.05
        w_large = w[large]
        sinc_w[large] = backend.sin(w_large) / w_large
        f_denom = 1 / (9 * u_sq - w_sq)
        f0 = ((u_sq - w_sq) * e_2iu + e_iu * (8 * u_sq * cos_w +
              2j * u * (3 * u_sq + w_sq) * sinc_w)) * f_denom
        f1 = (2 * u * e_2iu - e_iu * (2 * u * cos_w -
              1j * (3 * u_sq - w_sq) * sinc_w)) * f_denom
        f2 = (e_2iu - e_iu * (cos_w + 3j * u * sinc_w)) * f_denom
        f0[parity] = f0[parity].conj()
        f1[parity] = -f1[parity].conj()
        f2[parity] = f2[parity].conj()
        f0 = contract("...,ab->...ab", f0, backend.identity(Nc))
        f1 = contract("...,...ab->...ab", f1, Q)
        f2 = contract("...,...ab,...bc->...ac", f2, Q, Q)
        U = contract("...ab,...bc->...ac", f0 + f1 + f2, U)
    self._U = U
