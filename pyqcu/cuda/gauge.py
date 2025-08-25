import cupy as cp
from cupyx.scipy.linalg import expm
from pyqcu.cuda import define
from math import sqrt
from typing import Tuple, Optional


def get_gell_mann_matrices(dtype):
    lambda1 = cp.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]], dtype=dtype)
    lambda2 = cp.array([[0, -1j, 0],
                        [1j, 0, 0],
                        [0, 0, 0]], dtype=dtype)
    lambda3 = cp.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 0]], dtype=dtype)
    lambda4 = cp.array([[0, 0, 1],
                        [0, 0, 0],
                        [1, 0, 0]], dtype=dtype)
    lambda5 = cp.array([[0, 0, -1j],
                        [0, 0, 0],
                        [1j, 0, 0]], dtype=dtype)
    lambda6 = cp.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]], dtype=dtype)
    lambda7 = cp.array([[0, 0, 0],
                        [0, 0, -1j],
                        [0, 1j, 0]], dtype=dtype)
    lambda8 = cp.array([[1/3**0.5, 0, 0],
                        [0, 1/3**0.5, 0],
                        [0, 0, -2/3**0.5]], dtype=dtype)
    return [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]


def give_gauss_su3(sigma=0.1, dtype=cp.complex128, seed=None):
    if seed is not None:
        cp.random.seed(seed)
    gell_mann = get_gell_mann_matrices(dtype)
    a = cp.random.normal(0.0, 1.0, size=8)
    H = sum(ai * Ai for ai, Ai in zip(a, gell_mann))
    U = expm(1j * sigma * H)
    return U, a


def give_gauss_SU3(sigma=0.1, dtype=cp.complex128, seed=12138, size=100):
    U = cp.ones((size, define._LAT_C_, define._LAT_C_), dtype=dtype)
    print(f"U_size = {size}")
    for i in range(size):
        U[i], _ = give_gauss_su3(sigma, dtype, seed+i)
        # print(f"U_{i} is ready.")
    return U


def is_unitary(U, tol=1e-6):
    I = cp.eye(U.shape[0], dtype=U.dtype)
    print("U.conj().T @ U =\n", U.conj().T @ U)
    return cp.allclose(U.conj().T @ U, I, atol=tol)


def is_su3(U, tol=1e-6):
    return is_unitary(U, tol) and cp.allclose(cp.linalg.det(U), 1.0, atol=tol)


def validate_minor_identities(U, tol=1e-6):
    U_flat = U.flatten()
    c6 = (U_flat[1] * U_flat[5] - U_flat[2] * U_flat[4]).conj()
    c7 = (U_flat[2] * U_flat[3] - U_flat[0] * U_flat[5]).conj()
    c8 = (U_flat[0] * U_flat[4] - U_flat[1] * U_flat[3]).conj()
    print(f"U[6] = {U_flat[6]}, c6 = {c6}")
    print(f"U[7] = {U_flat[7]}, c7 = {c7}")
    print(f"U[8] = {U_flat[8]}, c8 = {c8}")
    return (cp.allclose(U_flat[6], c6, atol=tol) and
            cp.allclose(U_flat[7], c7, atol=tol) and
            cp.allclose(U_flat[8], c8, atol=tol))


def test_su3(U):
    print(" Whether unitary matrix :", is_unitary(U))
    print(" Is SU(3) a member :", is_su3(U))
    print(" Satisfy three-row complex conjugation properties :",
          validate_minor_identities(U))


class Gauge:
    def __init__(self,
                 latt_size: Tuple[int, int, int, int] = (8, 8, 8, 8),
                 kappa: float = 0.1,
                 u_0: float = 1.0,
                 dtype=cp.complex128,
                 verbose: bool = False):
        """
        Gauge-Dirac operator on a 4D lattice with SU(3) gauge fields.
        Args:
            latt_size: (Lx, Ly, Lz, Lt)
            kappa: Hopping parameter
            u_0: Gauge parameter
            dtype: cp.complex128 or cp.complex64
            verbose: debug flag
        """
        self.latt_size = latt_size
        self.Lx, self.Ly, self.Lz, self.Lt = latt_size
        self.kappa = kappa
        self.u_0 = u_0
        self.dtype = dtype
        self.verbose = verbose
        # real dtype
        self.real_dtype = cp.float64 if dtype == cp.complex128 else cp.float32
        if self.verbose:
            print(f"Initializing Gauge (CuPy):")
            print(f"  Lattice size: {latt_size}")
            print(f"  kappa={kappa}, u_0={u_0}")
            print(f"  dtype={dtype}, real_dtype={self.real_dtype}")
        # Gell-Mann matrices
        self.gell_mann = self._get_gell_mann_matrices()

    def _get_gell_mann_matrices(self):
        """Generate Gell-Mann matrices for SU(3)."""
        matrices = [
            cp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=self.real_dtype),
            cp.array([[0, -1, 0], [1,  0, 0], [0, 0, 0]],
                     dtype=self.real_dtype) * 1j,
            cp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]],
                     dtype=self.real_dtype),
            cp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=self.real_dtype),
            cp.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                     dtype=self.real_dtype) * 1j,
            cp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=self.real_dtype),
            cp.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                     dtype=self.real_dtype) * 1j,
            cp.array([[1/sqrt(3), 0, 0],
                      [0, 1/sqrt(3), 0],
                      [0, 0, -2/sqrt(3)]], dtype=self.real_dtype)
        ]
        # (8,3,3)
        return cp.stack([m.astype(self.dtype) for m in matrices], axis=0)

    def generate_gauge_field(self, sigma: float = 0.1, seed: Optional[int] = None):
        """
        Generate random SU(3) gauge field.
        Returns: U [3,3,4,Lt,Lz,Ly,Lx]
        """
        if self.verbose:
            print(f"Generating gauge field (sigma={sigma})")
        if seed is not None:
            cp.random.seed(seed)
            if self.verbose:
                print(f"  Seed set: {seed}")
        # Gaussian random coefficients: (4,Lt,Lz,Ly,Lx,8)
        a = cp.random.normal(
            0.0, 1.0,
            size=(4, self.Lt, self.Lz, self.Ly, self.Lx, 8)
        ).astype(self.real_dtype)
        if self.verbose:
            print(f"  Coeff shape={a.shape}")
        # Broadcast gell-mann: (1,1,1,1,1,8,3,3)
        gm_exp = self.gell_mann.reshape(1, 1, 1, 1, 1, 8, 3, 3)
        # Hermitian matrices H: (4,Lt,Lz,Ly,Lx,3,3)
        H = cp.einsum("...i,...ijk->...jk", a.astype(self.dtype), gm_exp)
        # Exponential map
        shape = H.shape
        H_flat = H.reshape(-1, 3, 3)
        U_flat = cp.stack([expm(1j * sigma * h) for h in H_flat])
        U_all = U_flat.reshape(shape)
        # Rearrange to [3,3,4,Lt,Lz,Ly,Lx]
        U = U_all.transpose(5, 6, 0, 1, 2, 3, 4).copy()
        if self.verbose:
            print("  Gauge field generated")
            print(f"  Norm={cp.linalg.norm(U)}")
        return U

    def check_su3(self, U, tol: float = 1e-6) -> bool:
        """
        Check SU(3) constraints.
        """
        U_mat = U.transpose(*range(2, U.ndim), 0, 1).reshape(-1, 3, 3)
        N = U_mat.shape[0]
        eye = cp.eye(3, dtype=U_mat.dtype).reshape(1, 3, 3).repeat(N, axis=0)
        # unitarity
        UH_U = U_mat.conj().transpose(0, 2, 1) @ U_mat
        unitary_ok = cp.allclose(UH_U, eye, atol=tol)
        # det=1
        det_U = cp.linalg.det(U_mat)
        det_ok = cp.allclose(det_U, cp.ones_like(det_U), atol=tol)
        # minors check
        Uf = U_mat.reshape(N, 9)
        c6 = (Uf[:, 1] * Uf[:, 5] - Uf[:, 2] * Uf[:, 4]).conj()
        c7 = (Uf[:, 2] * Uf[:, 3] - Uf[:, 0] * Uf[:, 5]).conj()
        c8 = (Uf[:, 0] * Uf[:, 4] - Uf[:, 1] * Uf[:, 3]).conj()
        minors_ok = (cp.allclose(Uf[:, 6], c6, atol=tol) and
                     cp.allclose(Uf[:, 7], c7, atol=tol) and
                     cp.allclose(Uf[:, 8], c8, atol=tol))
        if self.verbose:
            print(
                f"[check_su3] N={N}, unitary={unitary_ok}, det={det_ok}, minors={minors_ok}")
        return bool(unitary_ok and det_ok and minors_ok)
