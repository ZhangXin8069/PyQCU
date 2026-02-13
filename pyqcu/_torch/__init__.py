import re
import torch
from argparse import Namespace
Namespace.__module__ = "pyqcu._torch"
disable_patch_npu = False


def abs(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input):
        return torch.sqrt(input.real**2 + input.imag**2)
    else:
        return torch.abs(input)


def vdot(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input):
        return torch.sum(torch.conj(input.flatten()) * other.flatten())
    else:
        return torch.vdot(input.flatten(), other.flatten())


def norm(input: torch.Tensor, p='fro', dim=None, keepdim=False, out=None, dtype=None) -> torch.Tensor:
    if (input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input):
        abs_input = abs(input)
        if dim is None:
            return torch.norm(abs_input, p=p, keepdim=keepdim, out=out, dtype=dtype)
        else:
            return torch.norm(abs_input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
    else:
        if dim is None:
            return torch.norm(input, p=p, keepdim=keepdim, out=out, dtype=dtype)
        else:
            return torch.norm(input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def roll(input: torch.Tensor, shifts, dims=None) -> torch.Tensor:
    if (input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input):
        real_rolled = torch.roll(input.real, shifts, dims)
        imag_rolled = torch.roll(input.imag, shifts, dims)
        return real_rolled + imag_rolled * 1j
    else:
        return torch.roll(input, shifts, dims)


def allclose(input: torch.Tensor, other: torch.Tensor, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    if (input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input):
        real_close = torch.allclose(
            input.real, other.real, rtol, atol, equal_nan)
        imag_close = torch.allclose(
            input.imag, other.imag, rtol, atol, equal_nan)
        return real_close and imag_close
    else:
        return torch.allclose(input, other, rtol, atol, equal_nan)


def einsum(equation: str, *operands) -> torch.Tensor:
    if any((op.device.type == 'npu' or disable_patch_npu) and torch.is_complex(op) for op in operands):
        real_parts = [op.real if torch.is_complex(
            op) else op for op in operands]
        imag_parts = [op.imag if torch.is_complex(
            op) else torch.zeros_like(op) for op in operands]
        real_real = torch.einsum(equation, *real_parts)
        imag_imag = torch.einsum(equation, *imag_parts)
        if len(operands) == 2:
            if torch.is_complex(operands[0]) and torch.is_complex(operands[1]):
                real_imag = torch.einsum(
                    equation, real_parts[0], imag_parts[1])
                imag_real = torch.einsum(
                    equation, imag_parts[0], real_parts[1])
                real_result = real_real - imag_imag
                imag_result = real_imag + imag_real
            elif torch.is_complex(operands[0]):
                real_result = torch.einsum(
                    equation, real_parts[0], real_parts[1])
                imag_result = torch.einsum(
                    equation, imag_parts[0], real_parts[1])
            else:
                real_result = torch.einsum(
                    equation, real_parts[0], real_parts[1])
                imag_result = torch.einsum(
                    equation, real_parts[0], imag_parts[1])
        else:
            real_result = real_real
            imag_result = torch.zeros_like(real_real)
        return real_result + imag_result * 1j
    else:
        return torch.einsum(equation, *operands)


def linalg_qr(input: torch.Tensor, mode='reduced') -> tuple:
    if (input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input):
        input_cpu = input.cpu()
        Q_cpu, R_cpu = torch.linalg.qr(input_cpu, mode)
        return Q_cpu.to(input.device), R_cpu.to(input.device)
    else:
        return torch.linalg.qr(input, mode)


def eye(n: int, m=None, out=None, dtype: torch.dtype = None, layout=torch.strided, device: torch.device = None, requires_grad=False) -> torch.Tensor:
    if device is not None and (device.type == 'npu' or disable_patch_npu) and dtype is not None and dtype.is_complex:
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        if m is None:
            real_eye = torch.eye(n, out=out, dtype=real_dtype,
                                 layout=layout, device=device, requires_grad=requires_grad)
        else:
            real_eye = torch.eye(n, m, out=out, dtype=real_dtype,
                                 layout=layout, device=device, requires_grad=requires_grad)
        return real_eye.to(dtype)
    else:
        if m is None:
            return torch.eye(n, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
        else:
            return torch.eye(n, m, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)


def randn(*args, size=None, out=None, dtype: torch.dtype = None, layout=torch.strided, device: torch.device = None, requires_grad=False) -> torch.Tensor:
    if size is not None:
        args = size
    if device is not None and (device.type == 'npu' or disable_patch_npu) and dtype is not None and dtype.is_complex:
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real_part = torch.randn(*args, out=out, dtype=real_dtype,
                                layout=layout, device=device, requires_grad=requires_grad)
        imag_part = torch.randn(
            *args, dtype=real_dtype, layout=layout, device=device, requires_grad=requires_grad)
        return real_part + imag_part * 1j
    else:
        if size is not None:
            return torch.randn(size=size, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
        else:
            return torch.randn(*args, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)


def randn_like(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input):
        return torch.randn_like(input.real) + torch.randn_like(input.imag) * 1j
    else:
        return torch.randn_like(input)


def sqrt(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input):
        input_cpu = input.cpu()
        result_cpu = torch.sqrt(input_cpu)
        return result_cpu.to(input.device)
    else:
        return torch.sqrt(input)


def matmul(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if ((input.device.type == 'npu' or disable_patch_npu) and torch.is_complex(input)) or ((other.device.type == 'npu' or disable_patch_npu) and torch.is_complex(other)):
        input_real = input.real if torch.is_complex(input) else input
        input_imag = input.imag if torch.is_complex(
            input) else torch.zeros_like(input)
        other_real = other.real if torch.is_complex(other) else other
        other_imag = other.imag if torch.is_complex(
            other) else torch.zeros_like(other)
        real_real = torch.matmul(input_real, other_real)
        imag_imag = torch.matmul(input_imag, other_imag)
        real_imag = torch.matmul(input_real, other_imag)
        imag_real = torch.matmul(input_imag, other_real)
        return (real_real - imag_imag) + (real_imag + imag_real) * 1j
    else:
        return torch.matmul(input, other)
