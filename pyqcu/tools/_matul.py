import tilelang as tl
import tilelang.language as T


def matmul_gpu(M, N, K, block_M=128, block_N=128, block_K=32, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main


def matmul_cpu(M, N, K, block_M=32, block_N=32, block_K=32, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        for bx, by in T.grid(T.ceildiv(N, block_N), T.ceildiv(M, block_M)):
            acc = T.alloc_buffer((block_M, block_N),
                                 accum_dtype, scope="local")
            for i, j in T.grid(block_M, block_N):
                acc[i, j] = T.cast(0.0, accum_dtype)
            for ko in range(T.ceildiv(K, block_K)):
                k_start = ko * block_K
                k_end = T.min(k_start + block_K, K)
                for i in range(block_M):
                    row = by * block_M + i
                    for k in range(k_start, k_end):
                        val_a = T.cast(A[row, k], accum_dtype)
                        for j in range(block_N):
                            col = bx * block_N + j
                            acc[i, j] += val_a * T.cast(B[col, k], accum_dtype)
            for i, j in T.grid(block_M, block_N):
                C[by * block_M + i, bx * block_N +
                    j] = T.cast(acc[i, j], dtype)
    return main
