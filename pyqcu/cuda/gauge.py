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


# --------- 工具：批量 3x3 矩阵指数（Pade-13，统一缩放）---------


def _inv3x3_batch(M: cp.ndarray) -> cp.ndarray:
    """批量 3x3 逆矩阵，M: (N,3,3)"""
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    g, h, k = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]

    A = (e * k - f * h)
    B = -(d * k - f * g)
    C = (d * h - e * g)
    D = -(b * k - c * h)
    E = (a * k - c * g)
    F = -(a * h - b * g)
    G = (b * f - c * e)
    H = -(a * f - c * d)
    K = (a * e - b * d)

    det = a * A + b * B + c * C
    # 为避免 0 除，添加极小正则（理论上这里不应接近奇异）
    det = det + 0.0j + (cp.abs(det) == 0) * (1e-30 + 0.0j)
    inv_det = 1.0 / det

    adjT = cp.stack([
        cp.stack([A, D, G], axis=1),
        cp.stack([B, E, H], axis=1),
        cp.stack([C, F, K], axis=1)
    ], axis=1)  # (N,3,3)

    return adjT * inv_det[:, None, None]


def expm_batch_3x3(A: cp.ndarray) -> cp.ndarray:
    """
    批量 3x3 矩阵指数，A: (N,3,3)
    使用 Higham 的 Pade-13 + 缩放与平方；批量统一缩放（更简单可靠）。
    """
    # Pade-13 系数（Higham/Scipy）
    b = cp.array([
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0,   129060195264000.0,   10559470521600.0,
        670442572800.0,      33522128640.0,       1323241920.0,
        40840800.0,            960960.0,            16380.0,
        182.0,                 1.0
    ], dtype=A.dtype)

    I = cp.eye(3, dtype=A.dtype)[None, :, :].repeat(A.shape[0], axis=0)

    # 统一缩放因子 s（基于整个批次的 1-范数上界）
    # theta_13 经验值（Higham）
    theta_13 = 4.25
    # 1-范数：列绝对值和的最大值
    colsum = cp.sum(cp.abs(A), axis=-2)  # (N,3)
    norm1_each = cp.max(colsum, axis=-1)  # (N,)
    norm1 = cp.max(norm1_each)            # 标量
    s = int(cp.maximum(0, cp.ceil(cp.log2(norm1 / theta_13))).item()
            ) if norm1 > 0 else 0

    As = A / (2 ** s)

    A2 = As @ As
    A4 = A2 @ A2
    A6 = A2 @ A4

    # 按 SciPy/Higham 写法构造 U, V
    # U = A * (A6*(b13 + b11*A2 + b9*A4) + b7*A6 + b5*A4 + b3*A2 + b1*I)
    # V = A6*(b12 + b10*A2 + b8*A4) + b6*A6 + b4*A4 + b2*A2 + b0*I
    # 注意 b 的索引：b[0]=b0, ..., b[13]=b13
    U_poly = (b[13] * A6) + (b[11] * A4) + (b[9] * A2)
    U_poly = A6 @ U_poly + (b[7] * A6) + (b[5] * A4) + (b[3] * A2) + (b[1] * I)
    U = As @ U_poly

    V = (b[12] * A6) + (b[10] * A4) + (b[8] * A2)
    V = A6 @ V + (b[6] * A6) + (b[4] * A4) + (b[2] * A2) + (b[0] * I)

    # [V - U]^{-1} @ [V + U]
    VMU = V - U
    VPU = V + U
    VMU_inv = _inv3x3_batch(VMU)
    X = VMU_inv @ VPU

    # 反缩放：重复平方 s 次（统一 s 已足够稳定）
    for _ in range(s):
        X = X @ X
    return X

# ---------------- Gauge (CuPy) ----------------


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

        # Broadcast gell-mann: (1,1,1,1,1,8,3,3)
        gm_exp = self.gell_mann.reshape(1, 1, 1, 1, 1, 8, 3, 3)

        # Hermitian H: (4,Lt,Lz,Ly,Lx,3,3)
        H = cp.einsum("...i,...ijk->...jk", a.astype(self.dtype), gm_exp)

        # A = i*sigma*H （反Hermitian），批量指数（完全向量化）
        A = 1j * sigma * H
        N = A.shape[0] * A.shape[1] * A.shape[2] * A.shape[3] * A.shape[4]
        A_flat = A.reshape(N, 3, 3)
        U_flat = expm_batch_3x3(A_flat)
        U_all = U_flat.reshape(A.shape)

        # 排列为 [3,3,4,Lt,Lz,Ly,Lx]
        U = U_all.transpose(5, 6, 0, 1, 2, 3, 4).copy()

        if self.verbose:
            print("  Gauge field generated (vectorized expm)")
            print(f"  ||U||_F={cp.linalg.norm(U).item():.6e}")

        return U

    def check_su3(self, U, tol: float = 1e-6) -> bool:
        """
        Check SU(3) constraints.
        """
        U_mat = U.transpose(*range(2, U.ndim), 0, 1).reshape(-1, 3, 3)
        N = U_mat.shape[0]

        eye = cp.eye(3, dtype=U_mat.dtype)[None, :, :].repeat(N, axis=0)

        # unitarity
        UH_U = cp.transpose(U_mat.conj(), (0, 2, 1)) @ U_mat
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
                f"[check_su3] N={N}, unitary={bool(unitary_ok)}, det={bool(det_ok)}, minors={bool(minors_ok)}")

        return bool(unitary_ok and det_ok and minors_ok)
