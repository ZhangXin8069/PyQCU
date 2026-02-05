import tilelang
import tilelang.language as T

def matmul_configs(M, N, K):
    # Example space — tailor to your target
    tiles = [64, 128]
    stages = [2, 3]
    threads = [128, 256]
    return [
        dict(block_M=BM, block_N=BN, block_K=BK, num_stages=S, threads=TH)
        for BM in tiles
        for BN in tiles
        for BK in [32, 64]
        for S in stages
        for TH in threads
    ]

@tilelang.autotune(configs=matmul_configs, warmup=25, rep=100, timeout=60)
@tilelang.jit(out_idx=[-1])
def matmul(M: int, N: int, K: int,
           block_M: int = 128, block_N: int = 128, block_K: int = 32,
           threads: int = 128, num_stages: int = 3,
           dtype: str = 'float16', accum_dtype: str = 'float32'):

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype),
               B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_f = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_f)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_f)

            T.copy(C_f, C[by * block_M, bx * block_N])

    return kernel

# Usage
# Provide inputs via context (recommended for reproducibility across configs)
import torch
M = N = K = 1024
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)
C = torch.empty(M, N, device='cuda', dtype=torch.float16)

from tilelang.autotuner import set_autotune_inputs
with set_autotune_inputs(A, B, C):
    tuned_kernel = matmul(M, N, K)   # compiles, tunes, returns best kernel
    tuned_kernel(A, B, C)            # run best kernel