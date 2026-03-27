
    def _stout_smear_ndarray(self, nstep, rho):
        backend = get_backend()
        U = backend.ascontiguousarray(self._U)

        for _ in range(nstep):
            Q = backend.zeros_like(U)
            U_dag = U.transpose(0, 1, 2, 3, 4, 6, 5).conj()
            for mu in range(Nd - 1):
                for nu in range(Nd - 1):
                    if mu != nu:
                        Q[mu] += U[nu] @ backend.roll(U[mu], -1, 3 - nu) @ backend.roll(U_dag[nu], -1, 3 - mu)
                        Q[mu] += (
                            backend.roll(U_dag[nu], +1, 3 - nu)
                            @ backend.roll(U[mu], +1, 3 - nu)
                            @ backend.roll(backend.roll(U[nu], +1, 3 - nu), -1, 3 - mu)
                        )

            Q = rho * Q @ U_dag
            Q = 0.5j * (Q.transpose(0, 1, 2, 3, 4, 6, 5).conj() - Q)
            contract("...aa->...a", Q)[:] -= 1 / Nc * contract("...aa->...", Q)[..., None]
            Q_sq = Q @ Q
            c0 = contract("...aa->...", Q @ Q_sq).real / 3
            c1 = contract("...aa->...", Q_sq).real / 2
            c0_max = 2 * (c1 / 3) ** (3 / 2)
            parity = c0 < 0
            c0 = backend.abs(c0)
            theta = backend.arccos(c0 / c0_max)
            u = (c1 / 3) ** 0.5 * backend.cos(theta / 3)
            w = c1**0.5 * backend.sin(theta / 3)
            u_sq = u**2
            w_sq = w**2
            e_iu_real = backend.cos(u)
            e_iu_imag = backend.sin(u)
            e_2iu_real = backend.cos(2 * u)
            e_2iu_imag = backend.sin(2 * u)
            cos_w = backend.cos(w)
            sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
            large = backend.abs(w) > 0.05
            w_large = w[large]
            sinc_w[large] = backend.sin(w_large) / w_large
            f_denom = 1 / (9 * u_sq - w_sq)
            f0_real = (
                (u_sq - w_sq) * e_2iu_real
                + e_iu_real * 8 * u_sq * cos_w
                + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w
            ) * f_denom
            f0_imag = (
                (u_sq - w_sq) * e_2iu_imag
                - e_iu_imag * 8 * u_sq * cos_w
                + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w
            ) * f_denom
            f1_real = (
                2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w
            ) * f_denom
            f1_imag = (
                2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w
            ) * f_denom
            f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom
            f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom
            f0_imag[parity] *= -1
            f1_real[parity] *= -1
            f2_imag[parity] *= -1

            f = (f2_real + 1j * f2_imag)[..., None, None] * Q_sq
            f += (f1_real + 1j * f1_imag)[..., None, None] * Q
            contract("...aa->...a", f)[:] += (f0_real + 1j * f0_imag)[..., None]
            U = f @ U
        self._U = U

