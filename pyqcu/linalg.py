import cupy as cp

def gram_schmidt(vectors):
    num_vectors = vectors.shape[1]
    orthogonal_vectors = cp.copy(vectors)
    for i in range(num_vectors):
        orthogonal_vectors[:, i] /= cp.linalg.norm(orthogonal_vectors[:, i])
        for j in range(i + 1, num_vectors):
            projection = cp.dot(orthogonal_vectors[:, j], orthogonal_vectors[:, i])
            orthogonal_vectors[:, j] -= projection * orthogonal_vectors[:, i]
    return orthogonal_vectors