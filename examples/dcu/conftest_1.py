import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="hip")
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def matmul_relu_kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_relu_kernel


M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32
matmul_relu_kernel = matmul(M, N, K, block_M, block_N, block_K)

# 使用 PyTorch 等与 HIP 兼容的数据在 DCU 上测试
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
matmul_relu_kernel(a, b, c)
ref_c = torch.relu(a @ b)
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches reference.")
