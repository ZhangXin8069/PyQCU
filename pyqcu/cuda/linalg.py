import cupy as cp
import numpy as np


def dot(x, y):
    return cp.sum(x.conj() * y)


def norm2(x):
    return dot(x, x).real


def norm(x):
    return cp.sqrt(norm2(x))


def rayleigh_quotient(x, matvec):
    return cp.dot(x.conj(), matvec(x)).real / cp.dot(x.conj(), x).real


def initialize_random_vector(v):
    _v = v.flatten()
    _v.real, _v.imag = cp.random.randn(_v.size).astype(
        _v.real.dtype), cp.random.randn(_v.size).astype(_v.imag.dtype)
    norm = cp.linalg.norm(_v)
    if norm > 0:
        cp.divide(_v, norm, out=_v)
    return _v.reshape(v.shape).copy()


def chebyshev_filter(src, alpha, beta, matvec, degree=20, tol=1e-12):
    t_prev, t_curr, t_next = cp.empty_like(
        src), cp.empty_like(src), cp.empty_like(src)
    c, e = (beta + alpha) / 2, (beta - alpha) / 2
    t_prev[:] = src
    t_curr[:] = (matvec(src) - c * src) / e
    for _ in range(1, degree):
        t_prev[:] = t_curr
        t_curr[:] = (matvec(t_curr) - c * t_curr) / e
        t_next[:] = 2 * t_curr - t_prev
        norm = cp.linalg.norm(t_next)
        if norm > tol:
            t_next /= norm
        else:
            t_next[:] = t_curr
        t_curr, t_prev = t_next.copy(), t_curr
    return t_curr


def orthogonalize_against_vectors(v, Q_ortho, tol=1e-12, print_max_proj=False):
    if Q_ortho.ndim != 2 or v.ndim != 1:
        print("Q_ortho.shape: ", Q_ortho.shape, "v.shape: ", v.shape)
        raise ValueError("Q_ortho must be 2D matrix, v must be 1D vector")
    if Q_ortho.shape[1] != v.shape[0]:
        print("Q_ortho.shape: ", Q_ortho.shape, "v.shape: ", v.shape)
        raise ValueError("Vector and matrix dimensions mismatch")
    if Q_ortho.shape[0] > 0:
        proj_coeffs = cp.dot(Q_ortho.conj(), v)
        v_ortho = v - cp.dot(Q_ortho.T, proj_coeffs)
    else:
        v_ortho = v.copy()
    norm = cp.linalg.norm(v_ortho)
    if norm < tol:
        raise ValueError(
            "Input vector is linearly dependent with existing orthogonal vectors")
    v_ortho /= norm
    if print_max_proj:
        projections = cp.dot(Q_ortho.conj(), v_ortho)
        max_proj = cp.max(cp.abs(projections)).get()
        print(f"Maximum projection onto existing basis: {max_proj:.2e}")
    cp.clear_memo()
    return v_ortho


def orthogonalize_matrix(Q, cond_tol=1e-2, tol=1e-12):
    if Q.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    print(f"Condition number of Q: {np.linalg.cond(Q.T.get())}")
    _Q = Q.copy()
    Q_ortho = cp.empty_like(_Q)
    while cp.abs(np.linalg.cond(Q_ortho.T.get())-1.0) > cond_tol:
        for i in range(_Q.shape[0]):
            v = _Q[i, :]
            if i != 0:
                try:
                    v_ortho = orthogonalize_against_vectors(
                        v, _Q[:i, :], tol=tol)
                except ValueError as e:
                    print(f"Skipping column {i}: {e}")
            else:
                v_ortho = v/cp.linalg.norm(v)
            Q_ortho[i, :] = v_ortho/cp.linalg.norm(v_ortho)
        print(
            f"Condition number of Q_ortho: {np.linalg.cond(Q_ortho.T.get())}")
        _Q = Q_ortho.copy()
    cp.clear_memo()
    return Q_ortho


def orthogonalize(eigenvectors):
    _eigenvectors = eigenvectors.copy()
    size_e, size_s, size_c, size_T, size_t, size_Z, size_z, size_Y, size_y, size_X, size_x = eigenvectors.shape
    print(size_e, size_s, size_c, size_T, size_t,
          size_Z, size_z, size_Y, size_y, size_X, size_x)
    for T in range(size_T):
        for Z in range(size_Z):
            for Y in range(size_Y):
                for X in range(size_X):
                    origin_matrix = eigenvectors[:,
                                                 :, :, T, :, Z, :, Y, :, X, :]
                    _shape = origin_matrix.shape
                    _origin_matrix = origin_matrix.reshape(size_e, -1)
                    condition_number = np.linalg.cond(_origin_matrix.get())
                    print(f"矩阵条件数: {condition_number}")
                    a = _origin_matrix[:, 0]
                    b = _origin_matrix[:, -1]
                    print(cp.dot(a.conj(), b))
                    Q = cp.linalg.qr(_origin_matrix.T)[0]
                    condition_number = np.linalg.cond(Q.get())
                    print(f"矩阵条件数: {condition_number}")
                    a = Q[:, 0]
                    b = Q[:, -1]
                    print(cp.dot(a.conj(), b))
                    _eigenvectors[:, :, :, T, :, Z, :, Y, :, X, :] = Q.T.reshape(
                        _shape)
    return _eigenvectors
