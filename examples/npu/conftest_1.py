import argparse

def main() -> int:
    try:
        import tilelang
        import tilelang.language as T
    except Exception as exc:
        print(f"PYQCU::NPU::CONFTEST_1:\n tilelang unavailable: {exc}")
        return 0

    try:
        import torch_npu  # noqa: F401
        import torch
        device = torch.device('npu')
    except Exception as exc:
        print(
            "PYQCU::NPU::CONFTEST_1:\n"
            f" torch_npu unavailable or npu device unsupported: {exc}\n"
            "PYQCU::NPU::CONFTEST_1:\n"
            " skip NPU kernel compilation example on this host"
        )
        return 0

    tilelang.cache.clear_cache()
    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
    args = parser.parse_args()
    M = args.m
    N = args.n
    K = args.k

    @tilelang.jit(out_idx=[-1])
    def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
        m_num = M // block_M
        n_num = N // block_N

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
                bx = cid // n_num
                by = cid % n_num

                A_L1 = T.alloc_L1((block_M, K_L1), dtype)
                B_L1 = T.alloc_L1((K_L1, block_N), dtype)

                C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

                with T.Scope("C"):
                    loop_k = T.ceildiv(K, K_L1)
                    for k in T.serial(loop_k):
                        T.copy(A[bx * block_M, k * K_L1], A_L1)
                        T.copy(B[k * K_L1, by * block_N], B_L1)

                        T.barrier_all()
                        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                        T.barrier_all()

                    T.copy(C_L0, C[bx * block_M, by * block_N])

        return main

    func = matmul(M, N, K, 128, 256, 64)
    torch.manual_seed(0)
    a = torch.randn(M, K).half().to(device)
    b = torch.randn(K, N).half().to(device)
    c = torch.empty(M, N).half().to(device)
    print("init successful!")
    c = func(a, b)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
    return 0


if __name__ == '__main__':
    main()
