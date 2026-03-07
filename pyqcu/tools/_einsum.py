from concurrent.futures import thread
import torch
import tilelang
import tilelang.language as T
from functools import lru_cache, reduce
from pyqcu import _torch
from pyqcu.tools import torch_complex2real_dtype, torch2tl_dtype, to_contiguous_real


@lru_cache(maxsize=16)
def _build_kernel(E_size: int, e_size: int, xyzt_size: int, threads_per_block: int,
                  tl_dtype):
    @T.prim_func
    def main(
        A: T.Tensor((E_size, e_size, xyzt_size), tl_dtype),
        B: T.Tensor((e_size, xyzt_size), tl_dtype),
        C: T.Tensor((E_size, xyzt_size), tl_dtype),
    ):
        with T.Kernel(
            T.ceildiv(xyzt_size, threads_per_block),
            threads=threads_per_block
        ) as id:
            block_xyzt_size = xyzt_size // threads_per_block
            A_shared = T.alloc_shared((e_size, block_xyzt_size), tl_dtype)
            B_shared = T.alloc_shared((e_size, block_xyzt_size), tl_dtype)
            C_local = T.alloc_local(block_xyzt_size, tl_dtype)
            T.clear(C_local)
            for E, block_xyzt in T.Parallel(E_size, block_xyzt_size):
                T.copy(A[E, 0, block_xyzt_size*id], A_shared)
                T.copy(B[0, block_xyzt_size*id], B_shared)
                for e in T.Pipelined(e_size, num_stages=3):
                    C_local[block_xyzt*id] += A_shared[e,
                                                       block_xyzt*id] * B_shared[e, block_xyzt*id]
                T.copy(C_local, C[E, block_xyzt_size*id])
    kernel = tilelang.compile(main, out_idx=[2], target="cuda")
    return kernel


def Eexyzt_exyzt2Exyzt(Eexyzt: torch.Tensor, exyzt: torch.Tensor) -> torch.Tensor:
    if Eexyzt.dtype != exyzt.dtype:
        raise TypeError(
            f"Operand dtypes must match: got {Eexyzt.dtype} and {exyzt.dtype}."
        )
    torch_dtype = Eexyzt.dtype
    if Eexyzt.device.type not in ['cuda']:
        return _torch.einsum('Eexyzt,exyzt->Exyzt', Eexyzt, exyzt)
    is_complex = torch_dtype in torch_complex2real_dtype
    if is_complex:
        torch_real_dtype = torch_complex2real_dtype[torch_dtype]
        tl_dtype = torch2tl_dtype[torch_real_dtype]
    else:
        if torch_dtype not in torch2tl_dtype:
            raise NotImplementedError(
                f"Unsupported dtype {torch_dtype}. "
                f"Supported real dtypes    : {list(torch2tl_dtype.keys())}. "
                f"Supported complex dtypes : {list(torch_complex2real_dtype.keys())}."
            )
        tl_dtype = torch2tl_dtype[torch_dtype]
    E_size = Eexyzt.shape[0]
    e_size = exyzt.shape[0]
    xyzt_size = Eexyzt[0, 0].numel()
    output_shape = (E_size,) + Eexyzt.shape[2:]
    kernel = _build_kernel(
        E_size=E_size, e_size=e_size, xyzt_size=xyzt_size, threads_per_block=64,
        tl_dtype=tl_dtype,
    )
    if not is_complex:
        A_flat = Eexyzt.contiguous().reshape(E_size, e_size, xyzt_size)
        B_flat = exyzt.contiguous().reshape(e_size, xyzt_size)
        return kernel(A_flat, B_flat).reshape(output_shape)
    A_re = to_contiguous_real(Eexyzt, 0, E_size, e_size, xyzt_size)
    A_im = to_contiguous_real(Eexyzt, 1, E_size, e_size, xyzt_size)
    B_re = to_contiguous_real(exyzt,  0, e_size, xyzt_size)
    B_im = to_contiguous_real(exyzt,  1, e_size, xyzt_size)
    C_re = kernel(A_re, B_re) - kernel(A_im, B_im)
    C_im = kernel(A_re, B_im) + kernel(A_im, B_re)
    return torch.complex(C_re, C_im).reshape(output_shape)
