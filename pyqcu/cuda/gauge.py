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
    print(" Is SU(3) a member :", is_su3(U))
    print(" Satisfy three-row complex conjugation properties :",
          validate_minor_identities(U))