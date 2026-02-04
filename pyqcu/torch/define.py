import mpi4py.MPI as MPI
import torch
import os
import numpy as np
from typing import Tuple, Optional
# if_test_npu = True
if_test_npu = False
# NumPy → Torch
np2torch_dtype = {
    np.bool_: torch.bool,
    np.int8: torch.int8,
    np.uint8: torch.uint8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
# Torch → NumPy
torch2np_dtype = {
    torch.bool: np.bool_,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def give_if_multi() -> bool:
    comm = MPI.COMM_WORLD
    return comm.Get_size() > 1


def torch_abs(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or if_test_npu) and torch.is_complex(input):
        return torch.sqrt(input.real**2 + input.imag**2)
    else:
        return torch.abs(input)


def torch_vdot(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or if_test_npu) and torch.is_complex(input):
        return torch.sum(torch.conj(input) * other)
    else:
        return torch.vdot(input, other)


def torch_norm(input: torch.Tensor, p='fro', dim=None, keepdim=False, out=None, dtype=None) -> torch.Tensor:
    if (input.device.type == 'npu' or if_test_npu) and torch.is_complex(input):
        abs_input = torch_abs(input)
        if dim is None:
            return torch.norm(abs_input, p=p, keepdim=keepdim, out=out, dtype=dtype)
        else:
            return torch.norm(abs_input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
    else:
        if dim is None:
            return torch.norm(input, p=p, keepdim=keepdim, out=out, dtype=dtype)
        else:
            return torch.norm(input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def torch_roll(input: torch.Tensor, shifts, dims=None) -> torch.Tensor:
    if (input.device.type == 'npu' or if_test_npu) and torch.is_complex(input):
        real_rolled = torch.roll(input.real, shifts, dims)
        imag_rolled = torch.roll(input.imag, shifts, dims)
        return real_rolled + imag_rolled * 1j
    else:
        return torch.roll(input, shifts, dims)


def torch_allclose(input: torch.Tensor, other: torch.Tensor, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    if (input.device.type == 'npu' or if_test_npu) and torch.is_complex(input):
        real_close = torch.allclose(
            input.real, other.real, rtol, atol, equal_nan)
        imag_close = torch.allclose(
            input.imag, other.imag, rtol, atol, equal_nan)
        return real_close and imag_close
    else:
        return torch.allclose(input, other, rtol, atol, equal_nan)


def torch_einsum(equation: str, *operands) -> torch.Tensor:
    if any((op.device.type == 'npu' or if_test_npu) and torch.is_complex(op) for op in operands):
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


def torch_linalg_qr(input: torch.Tensor, mode='reduced') -> tuple:
    if (input.device.type == 'npu' or if_test_npu) and torch.is_complex(input):
        input_cpu = input.cpu()
        Q_cpu, R_cpu = torch.linalg.qr(input_cpu, mode)
        return Q_cpu.to(input.device), R_cpu.to(input.device)
    else:
        return torch.linalg.qr(input, mode)


def torch_eye(n: int, m=None, out=None, dtype: torch.dtype = None, layout=torch.strided, device: torch.device = None, requires_grad=False) -> torch.Tensor:
    if device is not None and (device.type == 'npu' or if_test_npu) and dtype is not None and dtype.is_complex:
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


def torch_randn(*args, size=None, out=None, dtype: torch.dtype = None, layout=torch.strided, device: torch.device = None, requires_grad=False) -> torch.Tensor:
    if size is not None:
        args = size
    if device is not None and (device.type == 'npu' or if_test_npu) and dtype is not None and dtype.is_complex:
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


def torch_randn_like(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or if_test_npu) and torch.is_complex(input):
        return torch.randn_like(input.real) + torch.randn_like(input.imag) * 1j
    else:
        return torch.randn_like(input)


def torch_sqrt(input: torch.Tensor) -> torch.Tensor:
    if (input.device.type == 'npu' or if_test_npu) and torch.is_complex(input):
        input_cpu = input.cpu()
        result_cpu = torch.sqrt(input_cpu)
        return result_cpu.to(input.device)
    else:
        return torch.sqrt(input)


def torch_matmul(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if ((input.device.type == 'npu' or if_test_npu) and torch.is_complex(input)) or ((other.device.type == 'npu' or if_test_npu) and torch.is_complex(other)):
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


def multi_vdot(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-process dot product using mpi4py with buffer mode.
    Args:
        a, b: local tensors on GPU (per process)
    Returns:
        global dot product as a complex scalar (torch.Tensor)
    """
    device = a.device
    assert a.device == b.device, "a and b must be on the same device"
    comm = MPI.COMM_WORLD
    comm.Barrier()
    local_dot = torch_vdot(a.flatten(), b.flatten())
    sendbuf = local_dot.detach().cpu().contiguous().numpy()
    recvbuf = np.zeros_like(sendbuf).copy()
    comm.Allreduce(sendbuf=sendbuf, recvbuf=recvbuf, op=MPI.SUM)
    comm.Barrier()
    return torch.from_numpy(recvbuf).to(device=device).clone()


def multi_norm(
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-process norm with buffer mode.
    Args:
        a: local tensor
    Returns:
        global norm
    """
    return torch.sqrt(multi_vdot(a=a.flatten(), b=a.flatten()).real).item()


def torch_vdot_(a: torch.Tensor, b: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
    return multi_vdot(a, b) if give_if_multi() and if_multi else torch_vdot(a.flatten(), b.flatten())


def torch_norm_(a: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
    return multi_norm(a) if give_if_multi() and if_multi else torch_norm(a.flatten()).real.item()


def prime_factorization(n: int):
    """Return the prime factorization of n as a list (using numpy only)."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def give_grid_size() -> Tuple[int, int, int, int]:
    comm = MPI.COMM_WORLD
    factors = prime_factorization(comm.Get_size())
    groups = np.ones(4, dtype=int)
    for f in sorted(factors, reverse=True):
        idx = np.argmin(groups)   # index of smallest product
        groups[idx] *= f
    return sorted(groups.tolist())


def give_eo_mask(xxxtzy_x_p: torch.Tensor, eo: int, verbose=False) -> torch.Tensor:
    if verbose:
        print("@give_eo_mask......")
    shape = xxxtzy_x_p.shape
    t, z, y, x_p = shape[-4:]
    # Create coordinate grids for original shape
    coords = torch.meshgrid(
        torch.arange(t),
        torch.arange(z),
        torch.arange(y),
        torch.arange(x_p),
        indexing='ij'
    )
    # Sum coordinates to determine checkerboard pattern
    sums = coords[0] + coords[1] + coords[2]  # t+z+y
    return sums % 2 == eo


def slice_dim(dim: int = 4, ward: int = 0, start: int = None, stop: int = None, step: int = 2, point: int = None) -> tuple:
    """
    Slice tensor along a specific dimension.
    Args:
        input_array: input tensor.
        dim: number of dimensions.
        ward: dimension index to slice. [xyzt]
        start, stop, step: same as Python slicing [start:stop:step].
        point: just a point
    Returns:
        tuple(slices)
    """
    slices = [slice(None)] * dim
    if point == None:
        slices[-ward-1] = slice(start, stop, step)
    else:
        slices[-ward-1] = point
    return tuple(slices)


def give_grid_index(rank: int = None) -> Tuple[int, int, int, int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if rank == None else rank
    size = comm.Get_size()
    return torch.nonzero(
        torch.arange(size).reshape(
            give_grid_size()[::-1]) == rank).squeeze().tolist()[::-1]  # to output x,y,z,t


def give_rank_plus(ward: int, rank: int = None) -> Tuple[int, int, int, int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if rank == None else rank
    grid_size = give_grid_size()
    grid_index = give_grid_index(rank=rank)
    grid_index[ward] = 0 if grid_index[ward] == grid_size[ward] - \
        1 else grid_index[ward]+1
    return grid_index[0]+(grid_size[0]*(grid_index[1]+(grid_size[1]*(grid_index[2]+(grid_size[2]*(grid_index[3]))))))


def give_rank_minus(ward: int, rank: int = None) -> Tuple[int, int, int, int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if rank == None else rank
    grid_size = give_grid_size()
    grid_index = give_grid_index(rank=rank)
    grid_index[ward] = grid_size[ward] - \
        1 if grid_index[ward] == 0 else grid_index[ward]-1
    return grid_index[0]+(grid_size[0]*(grid_index[1]+(grid_size[1]*(grid_index[2]+(grid_size[2]*(grid_index[3]))))))


def local2full_tensor(
    local_tensor: torch.Tensor,
    root: int = 0
) -> Optional[torch.Tensor]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    prefix_shape = local_tensor.shape[:-4]
    grid_lat_x, grid_lat_y, grid_lat_z, grid_lat_t = local_tensor.shape[-4:][::-1]
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    lat_t = grid_lat_t * grid_t
    lat_z = grid_lat_z * grid_z
    lat_y = grid_lat_y * grid_y
    lat_x = grid_lat_x * grid_x
    global_shape = (*prefix_shape, lat_t, lat_z, lat_y, lat_x)
    local_shape = (*prefix_shape, grid_lat_t,
                   grid_lat_z, grid_lat_y, grid_lat_x)
    sendbuf = local_tensor.cpu().contiguous().numpy()
    if rank == root:
        recvbuf = np.zeros(shape=(size,) + local_shape,
                           dtype=torch2np_dtype[local_tensor.dtype])
    else:
        recvbuf = None
    comm.Gather(sendbuf=sendbuf, recvbuf=recvbuf, root=root)
    comm.Barrier()
    if rank == root:
        full = np.zeros(global_shape, dtype=torch2np_dtype[local_tensor.dtype])
        for r in range(size):
            grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index(
                rank=r)
            full[...,
                 grid_index_t*grid_lat_t:(grid_index_t+1)*grid_lat_t,
                 grid_index_z*grid_lat_z:(grid_index_z+1)*grid_lat_z,
                 grid_index_y*grid_lat_y:(grid_index_y+1)*grid_lat_y,
                 grid_index_x*grid_lat_x:(grid_index_x+1)*grid_lat_x] = recvbuf[r].copy()
        return torch.from_numpy(full).to(device=local_tensor.device).clone()
    else:
        return None


def full2local_tensor(
    full_tensor: Optional[torch.Tensor],
    root: int = 0,
) -> torch.Tensor:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    prefix_shape = full_tensor.shape[:-4]
    lat_x, lat_y, lat_z, lat_t = full_tensor.shape[-4:][::-1]
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    grid_lat_t = lat_t // grid_t
    grid_lat_z = lat_z // grid_z
    grid_lat_y = lat_y // grid_y
    grid_lat_x = lat_x // grid_x
    local_shape = (*prefix_shape, grid_lat_t,
                   grid_lat_z, grid_lat_y, grid_lat_x)
    if rank == root:
        full_np = full_tensor.cpu().contiguous().numpy()
        sendbuf = np.zeros(shape=(size,) + local_shape,
                           dtype=torch2np_dtype[full_tensor.dtype])
        for r in range(size):
            grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index(
                rank=r)
            block = full_np[...,
                            grid_index_t*grid_lat_t:(grid_index_t+1)*grid_lat_t,
                            grid_index_z*grid_lat_z:(grid_index_z+1)*grid_lat_z,
                            grid_index_y*grid_lat_y:(grid_index_y+1)*grid_lat_y,
                            grid_index_x*grid_lat_x:(grid_index_x+1)*grid_lat_x].copy()
            np.copyto(sendbuf[r], block)
    else:
        sendbuf = None
    recvbuf = np.zeros(shape=local_shape,
                       dtype=torch2np_dtype[full_tensor.dtype])
    comm.Scatter(sendbuf=sendbuf, recvbuf=recvbuf, root=root)
    comm.Barrier()
    return torch.from_numpy(recvbuf).to(device=full_tensor.device).clone()


def set_device(device: torch.device):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    dev_type = device.type
    local_rank = None
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        if dev_type == "cuda" and torch.cuda.is_available():
            local_rank = rank % torch.cuda.device_count()
        elif dev_type == "npu":
            try:
                import torch_npu
                local_rank = rank % torch.npu.device_count()
            except ImportError:
                raise RuntimeError(
                    "torch_npu not found; please install it for NPU support.")
        else:
            local_rank = 0
    if dev_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        torch.cuda.set_device(local_rank)
    elif dev_type == "npu":
        import torch_npu
        torch.npu.set_device(local_rank)
    elif dev_type == "cpu":
        pass
    else:
        raise ValueError(f"Unsupported device type: {dev_type}")
    print(f"[MPI Rank {rank}/{size}] Using {dev_type}:{local_rank}")
