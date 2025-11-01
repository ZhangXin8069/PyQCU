import mpi4py.MPI as MPI
import torch
import os
import numpy as np
from typing import Tuple, Optional

import torch_npu.npu
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


def torch_vdot(a: torch.Tensor, b: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
    return multi_vdot(a, b) if give_if_multi() and if_multi else torch.vdot(a.flatten(), b.flatten())


def torch_norm(a: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
    return multi_norm(a) if give_if_multi() and if_multi else torch.norm(a.flatten()).real.item()


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


def give_local_rank(device: torch.device):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if device == torch.device('cpu'):
        device_per_node = os.cpu_count()
        local_rank = rank % device_per_node
    elif device == torch.device('cuda'):
        device_per_node = torch.cuda.device_count()
        local_rank = rank % device_per_node
    else:
        device_per_node = 1
        local_rank = rank % device_per_node
    return local_rank


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
    local_dot = torch.vdot(a.flatten(), b.flatten())
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
    local_rank = give_local_rank(device=device)
    if device == torch.device('cuda'):
        torch.cuda.set_device(local_rank)
    elif device == torch.device('cpu'):
        pass
    elif device == torch.device('npu'):
        import torch_npu
        torch_npu.torch.cuda.set_device(local_rank)
    print(
        f"@Device:{device}, My Rank:{rank}/{size}, Local Rank:{local_rank}@\n")