import mpi4py.MPI as MPI
import os
import torch
import numpy as np
from typing import Tuple


def give_if_multi() -> bool:
    comm = MPI.COMM_WORLD
    return comm.Get_size() > 1


def torch_vdot(a: torch.Tensor, b: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
    return multi_vdot(a, b) if give_if_multi() and if_multi else torch.vdot(a, b)


def torch_norm(a: torch.Tensor, if_multi: bool = give_if_multi()) -> torch.Tensor:
    return multi_norm(a) if give_if_multi() and if_multi else torch.norm(a)


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


def multi_vdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multi-process dot product using mpi4py.
    a, b: local tensors on GPU (per process)
    returns: global dot product as a complex scalar (torch.Tensor)
    """
    device = a.device
    assert a.device == b.device, "a and b must be on the same device"
    comm = MPI.COMM_WORLD
    comm.Barrier()
    local_dot = torch.vdot(a, b)
    local_dot_cpu = local_dot.detach().cpu().contiguous().numpy()
    global_dot_cpu = comm.allreduce(local_dot_cpu, op=MPI.SUM)
    comm.Barrier()
    global_dot = torch.tensor(global_dot_cpu, device=device)
    return global_dot.clone()


def multi_norm(a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(multi_vdot(a=a.flatten(), b=a.flatten()).real).clone()


def local2full_tensor(
    local_tensor: torch.Tensor,
    lat_size: Tuple[int, int, int, int],
    device: torch.device,
    root: int = 0
) -> torch.Tensor:
    """
    Gather local PyTorch tensor blocks into a full tensor on root process.
    local_tensor: [..., local_t, local_z, local_y, local_x]
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    lat_x, lat_y, lat_z, lat_t = lat_size
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    grid_lat_t = lat_t // grid_t
    grid_lat_z = lat_z // grid_z
    grid_lat_y = lat_y // grid_y
    grid_lat_x = lat_x // grid_x
    prefix_shape = local_tensor.shape[:-4]
    global_shape = (*prefix_shape, lat_t, lat_z, lat_y, lat_x)
    dtype = local_tensor.cpu().numpy().dtype
    local_np = local_tensor.cpu().contiguous().numpy().copy()
    gathered = comm.gather(local_np, root=root)
    comm.Barrier()
    if rank == root:
        full = np.zeros(global_shape, dtype=dtype)
        for r, block in enumerate(gathered):
            grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index(
                rank=r)
            full[...,
                 grid_index_t*grid_lat_t:(grid_index_t+1)*grid_lat_t,
                 grid_index_z*grid_lat_z:(grid_index_z+1)*grid_lat_z,
                 grid_index_y*grid_lat_y:(grid_index_y+1)*grid_lat_y,
                 grid_index_x*grid_lat_x:(grid_index_x+1)*grid_lat_x] = block.copy()
        return torch.from_numpy(full).to(device=device).clone()
    else:
        return None


def full2local_tensor(
    full_tensor: torch.Tensor,
    lat_size: Tuple[int, int, int, int],
    device: torch.device,
    root: int = 0
) -> torch.Tensor:
    """
    Scatter full PyTorch tensor into local blocks on each process.
    full_tensor: only valid on root process
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    lat_x, lat_y, lat_z, lat_t = lat_size
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    grid_lat_t = lat_t // grid_t
    grid_lat_z = lat_z // grid_z
    grid_lat_y = lat_y // grid_y
    grid_lat_x = lat_x // grid_x
    if rank == root:
        data_list = []
        for r in range(size):
            grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index(
                rank=r)
            block = full_tensor[...,
                                grid_index_t*grid_lat_t:(grid_index_t+1)*grid_lat_t,
                                grid_index_z*grid_lat_z:(grid_index_z+1)*grid_lat_z,
                                grid_index_y*grid_lat_y:(grid_index_y+1)*grid_lat_y,
                                grid_index_x*grid_lat_x:(grid_index_x+1)*grid_lat_x].clone()
            data_list.append(block.cpu().contiguous().numpy())
    else:
        data_list = None
    local_np = comm.scatter(data_list, root=root)
    comm.Barrier()
    return torch.from_numpy(local_np).to(device=device).clone()
