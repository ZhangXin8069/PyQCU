import torch
from argparse import Namespace
Namespace.__module__ = "pyqcu.cann"
force_use_npu = False


def abs(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        return torch.sqrt(input.real**2 + input.imag**2)
    else:
        return torch.abs(input)


def vdot(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        return torch.sum(torch.conj(input.flatten()) * other.flatten())
    else:
        return torch.vdot(input.flatten(), other.flatten())


def norm(input: torch.Tensor, p='fro', dim=None, keepdim=False, out=None, dtype=None) -> torch.Tensor:
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
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


def roll(input: torch.Tensor, shifts, dims: int) -> torch.Tensor:
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        real_rolled = torch.roll(input.real, shifts, dims)
        imag_rolled = torch.roll(input.imag, shifts, dims)
        return real_rolled + imag_rolled * 1j
    else:
        return torch.roll(input, shifts, dims)


def allclose(input: torch.Tensor, other: torch.Tensor, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        real_close = torch.allclose(
            input.real, other.real, rtol, atol, equal_nan)
        imag_close = torch.allclose(
            input.imag, other.imag, rtol, atol, equal_nan)
        return real_close and imag_close
    else:
        return torch.allclose(input, other, rtol, atol, equal_nan)


def einsum(equation: str, *operands) -> torch.Tensor:
    if any((op.device.type == 'npu' or force_use_npu) and torch.is_complex(op) for op in operands):
        real_parts = [op.real if torch.is_complex(
            op) else op for op in operands]
        imag_parts = [op.imag if torch.is_complex(
            op) else torch.zeros_like(op) for op in operands]

        n_ops = len(operands)

        if n_ops == 2:
            # 2-operand case: "a+ib" * "c+id"
            # real: a*c - b*d, imag: a*d + b*c
            real_real = torch.einsum(equation, *real_parts)
            imag_imag = torch.einsum(equation, *imag_parts)
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
            # BUGFIX 2026-07-28: General N-operand complex einsum.
            # For Z = Prod(a_k + i*b_k):
            #   Re(Z) = sum_{mask with even #imag} (-1)^(|mask|/2) * prod(masked_parts)
            #   Im(Z) = sum_{mask with odd  #imag} (-1)^(|mask|//2) * prod(masked_parts)
            # where masked_parts[k] = imag if bit k set, else real.
            # For n_ops=3, explicit formula yields 4 real and 4 imag contributions
            # matching the general 2^n = 8 terms with correct i^n factors.
            #
            # Iterate over all 2^n sign combinations:
            real_result = torch.einsum(equation, *real_parts)
            # BUGFIX 2026-07-28: start imag_result at zero; the all-imaginary
            # combination is processed in the loop below with correct sign.
            imag_result = torch.zeros_like(real_result)

            for combo_bits in range(1, 1 << n_ops):
                # Select parts for this combination
                selected = []
                n_imag = 0
                for k in range(n_ops):
                    if (combo_bits >> k) & 1:
                        selected.append(imag_parts[k])
                        n_imag += 1
                    else:
                        selected.append(real_parts[k])

                term = torch.einsum(equation, *selected)
                # sign = i^(n_imag): i^1=i, i^2=-1, i^3=-i, i^4=1, ...
                sign = 1.0
                if n_imag % 4 == 0:
                    sign = 1.0
                elif n_imag % 4 == 1:
                    sign = 0.0  # contributes to imag with +1
                elif n_imag % 4 == 2:
                    sign = -1.0
                elif n_imag % 4 == 3:
                    sign = 0.0  # contributes to imag with -1

                if n_imag % 2 == 0:  # contributes to real
                    real_result = real_result + sign * term
                else:
                    imag_sign = 1.0 if n_imag % 4 == 1 else -1.0
                    imag_result = imag_result + imag_sign * term
        return real_result + imag_result * 1j
    else:
        return torch.einsum(equation, *operands)


def linalg_qr(input: torch.Tensor, mode='reduced') -> tuple:
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        input_cpu = input.cpu()
        Q_cpu, R_cpu = torch.linalg.qr(input_cpu, mode)
        return Q_cpu.to(input.device), R_cpu.to(input.device)
    else:
        return torch.linalg.qr(input, mode)


def eye(n: int, m=None, out=None, dtype: torch.dtype = torch.complex64, layout=torch.strided, device: torch.device = torch.device('cpu'), requires_grad=False) -> torch.Tensor:
    if device is not None and (device.type == 'npu' or force_use_npu) and dtype is not None and dtype.is_complex:
        real_dtype = dtype.to_real()
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


def zeros(*args, size=None, out=None, dtype: torch.dtype, layout=torch.strided, device: torch.device, requires_grad=False) -> torch.Tensor:
    if size is not None:
        args = size
    if device is not None and (device.type == 'npu' or force_use_npu) and dtype is not None and dtype.is_complex:
        real_dtype = dtype.to_real()
        real_part = torch.zeros(*args, out=out, dtype=real_dtype,
                                layout=layout, device=device, requires_grad=requires_grad)
        imag_part = torch.zeros(
            *args, dtype=real_dtype, layout=layout, device=device, requires_grad=requires_grad)
        return real_part + imag_part * 1j
    else:
        if size is not None:
            return torch.zeros(size=size, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
        else:
            return torch.zeros(*args, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)


def zeros_like(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        return torch.zeros_like(input.real) + torch.zeros_like(input.imag) * 1j
    else:
        return torch.zeros_like(input)


def randn(*args, size=None, out=None, dtype: torch.dtype, layout=torch.strided, device: torch.device, requires_grad=False) -> torch.Tensor:
    if size is not None:
        args = size
    if device is not None and (device.type == 'npu' or force_use_npu) and dtype is not None and dtype.is_complex:
        real_dtype = dtype.to_real()
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
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        return torch.randn_like(input.real) + torch.randn_like(input.imag) * 1j
    else:
        return torch.randn_like(input)


def sqrt(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        input_cpu = input.cpu()
        result_cpu = torch.sqrt(input_cpu)
        return result_cpu.to(input.device)
    else:
        return torch.sqrt(input)


def matmul(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if ((input.device.type == 'npu' or force_use_npu) and torch.is_complex(input)) or ((other.device.type == 'npu' or force_use_npu) and torch.is_complex(other)):
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


def linalg_inv(input: torch.Tensor) -> torch.Tensor:
    """Batch matrix inverse with the NPU complex compatibility path."""
    if (input.device.type == 'npu' or force_use_npu) and torch.is_complex(input):
        input_cpu = input.cpu()
        result_cpu = torch.linalg.inv(input_cpu)
        return result_cpu.to(input.device)
    return torch.linalg.inv(input)


def linalg_solve(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Linear solve with the NPU complex compatibility path."""
    if (input.device.type == 'npu' or force_use_npu) and (
            torch.is_complex(input) or torch.is_complex(other)):
        input_cpu = input.cpu()
        other_cpu = other.cpu()
        result_cpu = torch.linalg.solve(input_cpu, other_cpu)
        return result_cpu.to(input.device)
    return torch.linalg.solve(input, other)


def stack(tensors, dim=0) -> torch.Tensor:
    """Stack tensors while splitting complex inputs on NPU when necessary."""
    if tensors and (tensors[0].device.type == 'npu' or force_use_npu) and torch.is_complex(tensors[0]):
        real = torch.stack([tensor.real for tensor in tensors], dim=dim)
        imag = torch.stack([tensor.imag for tensor in tensors], dim=dim)
        return real + imag * 1j
    return torch.stack(tensors, dim=dim)


def manual_seed(seed: int):
    """Expose deterministic seeding without importing torch in pure-Python modules."""
    return torch.manual_seed(seed)
