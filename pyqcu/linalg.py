import cupy as cp


def dot(x, y):
    return cp.sum(x.conj() * y)


def rayleigh_quotient(x, matvec):
    return cp.dot(x.conj(), matvec(x)).real / cp.dot(x.conj(), x).real

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

def initialize_random_vector(v):
    v.real, v.imag = cp.random.randn(v.size).astype(
        v.real.dtype), cp.random.randn(v.size).astype(v.imag.dtype)
    norm = cp.linalg.norm(v)
    if norm > 0:
        cp.divide(v, norm, out=v)
    return v


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
