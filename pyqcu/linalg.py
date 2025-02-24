import cupy as cp


def gram_schmidt(vectors):
    num_vectors = vectors.shape[1]
    orthogonal_vectors = cp.copy(vectors).astype(vectors.dtype)
    for i in range(num_vectors):
        orthogonal_vectors[:, i] /= cp.linalg.norm(orthogonal_vectors[:, i])
        for j in range(i + 1, num_vectors):
            projection = cp.dot(
                orthogonal_vectors[:, j].conj(), orthogonal_vectors[:, i])
            orthogonal_vectors[:, j] -= projection * orthogonal_vectors[:, i]
    return orthogonal_vectors


def householder(vectors):
    num_rows, num_cols = vectors.shape
    Q = cp.eye(num_rows, dtype=vectors.dtype)
    R = cp.copy(vectors).astype(vectors.dtype)
    for k in range(num_cols):
        x = R[k:, k]
        e1 = cp.zeros_like(x)
        e1[0] = 1
        alpha = cp.linalg.norm(x) * (-1 if x[0].real >= 0 else 1)
        v = x + alpha * e1
        v /= cp.linalg.norm(v)
        H_k = cp.eye(num_rows - k, dtype=vectors.dtype) - \
            2 * cp.outer(v, v.conj())
        H = cp.eye(num_rows, dtype=vectors.dtype)
        H[k:, k:] = H_k
        Q = cp.dot(Q, H)
        R = cp.dot(H, R)
    return Q[:, :num_cols]
