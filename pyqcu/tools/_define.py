from pyqcu import lattice
import mpi4py.MPI as MPI
import torch
import h5py
import os
import numpy as np
from typing import Tuple, Optional
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

# Check if h5py was built with MPI support


def check_mpi_support():
    """Check if h5py supports MPI parallel I/O"""
    try:
        # Try to check h5py configuration
        return h5py.get_config().mpi
    except:
        # If can't determine, try to create a test file
        try:
            comm = MPI.COMM_WORLD
            test_file = f'_test_mpi_support_{comm.Get_rank()}.h5'
            with h5py.File(test_file, 'w', driver='mpio', comm=comm) as f:
                pass
            if comm.Get_rank() == 0 and os.path.exists(test_file):
                os.remove(test_file)
            return True
        except:
            return False


HAS_MPI_SUPPORT = check_mpi_support()
# HAS_MPI_SUPPORT = False


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


def give_eo_mask(___tzy_t_p: torch.Tensor, eo: int, verbose=False) -> torch.Tensor:
    if verbose:
        print("PYQCU::TOOLS::DEFINE:\n give_eo_mask......")
    shape = ___tzy_t_p.shape
    # Create coordinate grids for original shape
    coords = torch.meshgrid(
        torch.arange(shape[-4]),
        torch.arange(shape[-3]),
        torch.arange(shape[-2]),
        torch.arange(shape[-1]),
        indexing='ij'
    )
    # Sum coordinates to determine checkerboard pattern
    sums = coords[lattice.wards['x']] + coords[lattice.wards['y']] + \
        coords[lattice.wards['z']]  # x+y+z
    return sums % 2 == eo


def slice_dim(dims_num: int = 4, ward: int = 0, start: int = None, stop: int = None, step: int = 2, point: int = None) -> tuple:
    """
    Slice tensor along a specific dimension.
    """
    slices = [slice(None)] * dims_num
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
            give_grid_size()) == rank).squeeze().tolist()


def give_rank_plus(ward: int, rank: int = None) -> Tuple[int, int, int, int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if rank == None else rank
    grid_size = give_grid_size()
    grid_index = give_grid_index(rank=rank)
    grid_index[ward] = 0 if grid_index[ward] == grid_size[ward] - \
        1 else grid_index[ward]+1
    return grid_index[-1]+(grid_size[-1]*(grid_index[-2]+(grid_size[-2]*(grid_index[-2]+(grid_size[-3]*(grid_index[-4]))))))


def give_rank_minus(ward: int, rank: int = None) -> Tuple[int, int, int, int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if rank == None else rank
    grid_size = give_grid_size()
    grid_index = give_grid_index(rank=rank)
    grid_index[ward] = grid_size[ward] - \
        1 if grid_index[ward] == 0 else grid_index[ward]-1
    return grid_index[-1]+(grid_size[-1]*(grid_index[-2]+(grid_size[-2]*(grid_index[-2]+(grid_size[-3]*(grid_index[-4]))))))


def local2full_tensor(
    local_tensor: torch.Tensor,
    root: int = 0
) -> Optional[torch.Tensor]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    prefix_shape = local_tensor.shape[:-4]
    grid_lat_x, grid_lat_y, grid_lat_z, grid_lat_t = local_tensor.shape[-4:]
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    lat_x = grid_lat_x * grid_x
    lat_y = grid_lat_y * grid_y
    lat_z = grid_lat_z * grid_z
    lat_t = grid_lat_t * grid_t
    global_shape = (*prefix_shape, lat_x, lat_y, lat_z, lat_t)
    local_shape = (*prefix_shape, grid_lat_x,
                   grid_lat_y, grid_lat_z, grid_lat_t)
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
                 grid_index_x*grid_lat_x:(grid_index_x+1)*grid_lat_x,
                 grid_index_y*grid_lat_y:(grid_index_y+1)*grid_lat_y,
                 grid_index_z*grid_lat_z:(grid_index_z+1)*grid_lat_z,
                 grid_index_t*grid_lat_t:(grid_index_t+1)*grid_lat_t] = recvbuf[r].copy()
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
    lat_x, lat_y, lat_z, lat_t = full_tensor.shape[-4:]
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    grid_lat_x = lat_x // grid_x
    grid_lat_y = lat_y // grid_y
    grid_lat_z = lat_z // grid_z
    grid_lat_t = lat_t // grid_t
    local_shape = (*prefix_shape, grid_lat_x,
                   grid_lat_y, grid_lat_z, grid_lat_t)
    if rank == root:
        full_np = full_tensor.cpu().contiguous().numpy()
        sendbuf = np.zeros(shape=(size,) + local_shape,
                           dtype=torch2np_dtype[full_tensor.dtype])
        for r in range(size):
            grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index(
                rank=r)
            block = full_np[...,
                            grid_index_x*grid_lat_x:(grid_index_x+1)*grid_lat_x,
                            grid_index_y*grid_lat_y:(grid_index_y+1)*grid_lat_y,
                            grid_index_z*grid_lat_z:(grid_index_z+1)*grid_lat_z,
                            grid_index_t*grid_lat_t:(grid_index_t+1)*grid_lat_t].copy()
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
                    "PYQCU::TOOLS::DEFINE:\n torch_npu not found; please install it for NPU support.")
        else:
            local_rank = 0
    if dev_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PYQCU::TOOLS::DEFINE:\n CUDA device requested but not available.")
        torch.cuda.set_device(local_rank)
    elif dev_type == "npu":
        import torch_npu
        torch.npu.set_device(local_rank)
    elif dev_type == "cpu":
        pass
    else:
        raise ValueError(f"Unsupported device type: {dev_type}")
    print(
        f"PYQCU::TOOLS::DEFINE:\n [MPI Rank {rank}/{size}] Using {dev_type}:{local_rank}")


def ___xyzt2p___xyzt(input_array: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    if verbose:
        print("PYQCU::TOOLS::DEFINE:\n ___xyzt2p___xyzt......")
    shape = input_array.shape
    dtype = input_array.dtype
    device = input_array.device
    prefix_shape = shape[:-4]
    t, z, y, x = shape[-4:]
    # Create coordinate grids
    coords = torch.meshgrid(
        torch.arange(t),
        torch.arange(z),
        torch.arange(y),
        torch.arange(x),
        indexing='ij'
    )
    # Sum coordinates to determine checkerboard pattern
    sums = coords[0] + coords[1] + coords[2] + coords[3]
    even_mask = (sums % 2 == 0)
    odd_mask = ~even_mask
    # Initialize output tensor with two channels
    splited_array = torch.zeros(
        (2, *prefix_shape, t, z, y, x//2),
        dtype=dtype,
        device=device
    )
    # Reshape masked elements and assign to output
    splited_array[0] = input_array[..., even_mask].reshape(
        *prefix_shape, t, z, y, x//2)
    splited_array[1] = input_array[..., odd_mask].reshape(
        *prefix_shape, t, z, y, x//2)
    if verbose:
        print(
            f"PYQCU::TOOLS::DEFINE:\n Splited Array Shape: {splited_array.shape}")
    return splited_array


def p___xyzt2___xyzt(input_array: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    if verbose:
        print("PYQCU::TOOLS::DEFINE:\n p___xyzt2___xyzt......")
    shape = input_array.shape
    dtype = input_array.dtype
    device = input_array.device
    prefix_shape = shape[1:-4]
    t, z, y, x_half = shape[-4:]
    x = x_half * 2  # Restore original x dimension
    # Create coordinate grids for original shape
    coords = torch.meshgrid(
        torch.arange(t),
        torch.arange(z),
        torch.arange(y),
        torch.arange(x),
        indexing='ij'
    )
    # Sum coordinates to determine checkerboard pattern
    sums = coords[0] + coords[1] + coords[2] + coords[3]
    even_mask = (sums % 2 == 0)
    odd_mask = ~even_mask
    # Initialize output tensor with original shape
    restored_array = torch.zeros(
        (*prefix_shape, t, z, y, x),
        dtype=dtype,
        device=device
    )
    # Assign values from input array using masks
    restored_array[..., even_mask] = input_array[0].reshape(*prefix_shape, -1)
    restored_array[..., odd_mask] = input_array[1].reshape(*prefix_shape, -1)
    if verbose:
        print(
            f"PYQCU::TOOLS::DEFINE:\n Restored Array Shape: {restored_array.shape}")
    return restored_array
