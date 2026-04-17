import torch
import tilelang.language as T
import pyqcu.cann as _torch
from tilelang import jit
from pyqcu.tools import torch2tl_dtype, warp_size, to_contiguous_real

pass_configs = {
    # tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    # tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@jit(out_idx=[-1], pass_configs=pass_configs, target="cuda")
def _Eetzyx_etzyx2Etzyx(E_size: int, e_size: int, tzyx_size: int, tl_dtype):
    @T.prim_func
    def main(
        Eetzyx: T.Tensor((E_size, e_size, tzyx_size), tl_dtype),
        etzyx: T.Tensor((e_size, tzyx_size), tl_dtype),
        Etzyx: T.Tensor((E_size, tzyx_size), tl_dtype),
    ):
        with T.Kernel(
            T.ceildiv(tzyx_size, warp_size),
            threads=warp_size, is_cpu=False
        ) as block_i:
            _Ee_warp = T.alloc_fragment(
                shape=(e_size, warp_size), dtype=tl_dtype)
            _e_warp = T.alloc_fragment(
                shape=(e_size, warp_size), dtype=tl_dtype)
            _E_warp = T.alloc_fragment(
                shape=warp_size, dtype=tl_dtype)
            start = warp_size * block_i
            end = warp_size*(block_i+1)
            T.copy(src=etzyx[:, start:end], dst=_e_warp)
            for E_i in T.Pipelined(E_size, num_stages=5):
                T.clear(_E_warp)
                T.copy(src=Eetzyx[E_i, :, start:end], dst=_Ee_warp)
                for thread_i in T.Parallel(warp_size):
                    for e_i in T.unroll(e_size):
                        # for e_i in T.Unroll(e_size):
                        _E_warp[thread_i] += _Ee_warp[e_i,
                                                      thread_i] * _e_warp[e_i, thread_i]
                T.copy(src=_E_warp,
                       dst=Etzyx[E_i, start:end])
    return main

# FOR TENSOR CORE
# @jit(out_idx=[-1], pass_configs=pass_configs, target="cuda")
# def _Eetzyx_etzyx2Etzyx(E_size: int, e_size: int, tzyx_size: int, tl_dtype):
#     E_pad_size = (E_size + 15) // 16 * 16
#     e_pad_size = (e_size + 15) // 16 * 16

#     @T.prim_func
#     def main(
#         Eetzyx: T.Tensor((E_size, e_size, tzyx_size), tl_dtype),
#         etzyx: T.Tensor((e_size, tzyx_size), tl_dtype),
#         Etzyx: T.Tensor((E_size, tzyx_size), tl_dtype),
#     ):
#         with T.Kernel(
#             T.ceildiv(tzyx_size, warp_size),
#             threads=warp_size, is_cpu=False
#         ) as block_i:
#             e_pad = T.alloc_fragment(
#                 shape=(e_pad_size, 1), dtype=tl_dtype)
#             E_pad = T.alloc_fragment(
#                 shape=(E_pad_size, 1), dtype=tl_dtype)
#             E_pad_e_pad = T.alloc_fragment(
#                 shape=(E_pad_size, e_pad_size), dtype=tl_dtype)
#             start = warp_size * block_i
#             for thread_i in T.Pipelined(warp_size, num_stages=5):
#                 T.clear(e_pad)
#                 T.clear(E_pad)
#                 T.clear(E_pad_e_pad)
#                 T.copy(
#                     src=Eetzyx[:, :, start+thread_i], dst=E_pad_e_pad[:E_size, :e_size])
#                 T.copy(
#                     src=etzyx[:, start+thread_i], dst=e_pad[:e_size,:])
#                 T.gemm(A=E_pad_e_pad, B=e_pad, C=E_pad)
#                 T.copy(src=E_pad[:E_size, :], dst=Etzyx[:, start+thread_i])
#     return main


def Eetzyx_etzyx2Etzyx(Eetzyx: torch.Tensor, etzyx: torch.Tensor) -> torch.Tensor:
    if Eetzyx.dtype != etzyx.dtype:
        raise TypeError(
            f"Operand dtypes must match: got {Eetzyx.dtype} and {etzyx.dtype}."
        )
    torch_dtype = Eetzyx.dtype
    if Eetzyx.device.type not in ['cuda']:
        return _torch.einsum('Eetzyx,etzyx->Etzyx', Eetzyx, etzyx)
    tl_dtype = torch2tl_dtype[torch_dtype.to_real()]
    E_size = Eetzyx.shape[0]
    e_size = etzyx.shape[0]
    tzyx_size = Eetzyx[0, 0].numel()
    output_shape = (E_size,) + Eetzyx.shape[2:]
    kernel = _Eetzyx_etzyx2Etzyx(
        E_size=E_size, e_size=e_size, tzyx_size=tzyx_size,
        tl_dtype=tl_dtype,
    )
    if not torch_dtype.is_complex:
        A_flat = Eetzyx.contiguous().reshape(E_size, e_size, tzyx_size)
        B_flat = etzyx.contiguous().reshape(e_size, tzyx_size)
        return kernel(A_flat, B_flat).reshape(output_shape)
    A_re = to_contiguous_real(Eetzyx, 0, E_size, e_size, tzyx_size)
    A_im = to_contiguous_real(Eetzyx, 1, E_size, e_size, tzyx_size)
    B_re = to_contiguous_real(etzyx,  0, e_size, tzyx_size)
    B_im = to_contiguous_real(etzyx,  1, e_size, tzyx_size)
    C_re = kernel(A_re, B_re) - kernel(A_im, B_im)
    C_im = kernel(A_re, B_im) + kernel(A_im, B_re)
    return torch.complex(C_re, C_im).reshape(output_shape)
