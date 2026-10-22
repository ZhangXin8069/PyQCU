import tilelang.language as T


def _patch_tilelang_mma_legalize():
    """tilelang 0.1.7.post3 上游 bug 本地热修(2026-08-24)。

    mma_macro_generator.TensorCoreIntrinEmitter 的 copy/gemm lowering 调用
    self._legalize_to_buffer_region(共 4 处),但该类未定义此方法
    (仅 mfma_macro_generator.MatrixCoreIntrinEmitter 有)→ T.gemm 编译期
    AttributeError: 'TensorCoreIntrinEmitter' object has no attribute
    '_legalize_to_buffer_region'。照抄上游 mfma 实现补挂;已存在则不动。
    """
    try:
        from tilelang.intrinsics import mma_macro_generator as _mmg
        from tilelang.intrinsics import mfma_macro_generator as _mfma
        _src = getattr(_mfma.MatrixCoreIntrinEmitter,
                       "_legalize_to_buffer_region", None)
        if _src is None:
            return
        _fn = getattr(_src, "__func__", _src)
        # 三处同名 TensorCoreIntrinEmitter(mma/wgmma/tcgen05)+基类,幂等补挂
        import importlib
        for _mod_name in ("tilelang.intrinsics.mma_macro_generator",
                          "tilelang.intrinsics.wgemma_macro_generator",
                          "tilelang.intrinsics.wgmma_macro_generator",
                          "tilelang.intrinsics.tcgen05_macro_generator",
                          "tilelang.intrinsics.mma_sm70_macro_generator",
                          "tilelang.intrinsics.mfma_macro_generator"):
            try:
                _mod = importlib.import_module(_mod_name)
            except Exception:
                continue
            for _cls_name in ("TensorCoreIntrinEmitter", "MMAIntrinEmitter"):
                _cls = getattr(_mod, _cls_name, None)
                if _cls is not None and not hasattr(_cls,
                                                    "_legalize_to_buffer_region"):
                    setattr(_cls, "_legalize_to_buffer_region",
                            staticmethod(_fn))
    except Exception:
        pass


_patch_tilelang_mma_legalize()


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
