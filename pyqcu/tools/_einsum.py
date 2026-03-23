import torch
import tilelang.language as T
from pyqcu import _torch
from tilelang import jit
from pyqcu.tools import torch_complex2real_dtype, torch2tl_dtype, warp_size, to_contiguous_real

pass_configs = {
    # tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    # tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@jit(out_idx=[-1], pass_configs=pass_configs, target="cuda")
def _Eexyzt_exyzt2Exyzt(E_size: int, e_size: int, xyzt_size: int, tl_dtype):
    @T.prim_func
    def main(
        Eexyzt: T.Tensor((E_size, e_size, xyzt_size), tl_dtype),
        exyzt: T.Tensor((e_size, xyzt_size), tl_dtype),
        Exyzt: T.Tensor((E_size, xyzt_size), tl_dtype),
    ):
        with T.Kernel(
            T.ceildiv(xyzt_size, warp_size),
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
            T.copy(src=exyzt[:, start:end], dst=_e_warp)
            for E_i in T.Pipelined(E_size, num_stages=5):
                T.clear(_E_warp)
                T.copy(src=Eexyzt[E_i, :, start:end], dst=_Ee_warp)
                for thread_i in T.Parallel(warp_size):
                    for e_i in T.unroll(e_size):
                    # for e_i in T.Unroll(e_size):
                        _E_warp[thread_i] += _Ee_warp[e_i,
                                                      thread_i] * _e_warp[e_i, thread_i]
                T.copy(src=_E_warp,
                       dst=Exyzt[E_i, start:end])
    return main

# FOR TENSOR CORE
# @jit(out_idx=[-1], pass_configs=pass_configs, target="cuda")
# def _Eexyzt_exyzt2Exyzt(E_size: int, e_size: int, xyzt_size: int, tl_dtype):
#     E_pad_size = (E_size + 15) // 16 * 16
#     e_pad_size = (e_size + 15) // 16 * 16

#     @T.prim_func
#     def main(
#         Eexyzt: T.Tensor((E_size, e_size, xyzt_size), tl_dtype),
#         exyzt: T.Tensor((e_size, xyzt_size), tl_dtype),
#         Exyzt: T.Tensor((E_size, xyzt_size), tl_dtype),
#     ):
#         with T.Kernel(
#             T.ceildiv(xyzt_size, warp_size),
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
#                     src=Eexyzt[:, :, start+thread_i], dst=E_pad_e_pad[:E_size, :e_size])
#                 T.copy(
#                     src=exyzt[:, start+thread_i], dst=e_pad[:e_size,:])
#                 T.gemm(A=E_pad_e_pad, B=e_pad, C=E_pad)
#                 T.copy(src=E_pad[:E_size, :], dst=Exyzt[:, start+thread_i])
#     return main


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
    kernel = _Eexyzt_exyzt2Exyzt(
        E_size=E_size, e_size=e_size, xyzt_size=xyzt_size,
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
