import h5py
import torch
import mpi4py.MPI as MPI
from typing import Tuple, Callable
from pyqcu.ascend.define import *


def ccdptzyx2pccdtzyx(gauge: torch.Tensor) -> torch.Tensor:
    dest = gauge.permute(3, 0, 1, 2, 4, 5, 6, 7)
    return dest.clone()


def tzyxccd2ccdtzyx(gauge: torch.Tensor) -> torch.Tensor:
    dest = gauge.permute(4, 5, 6, 0, 1, 2, 3)
    return dest.clone()


def tzyxsc2sctzyx(fermion: torch.Tensor) -> torch.Tensor:
    dest = fermion.permute(4, 5, 0, 1, 2, 3)
    return dest.clone()


def tzyxscsc2scsctzyx(clover_term: torch.Tensor) -> torch.Tensor:
    dest = clover_term.permute(4, 5, 6, 7, 0, 1, 2, 3)
    return dest.clone()


def xxxtzyx2pxxxtzyx(input_array: torch.Tensor, verbose=False) -> torch.Tensor:
    if verbose:
        print("@xxxtzyx2pxxxtzyx......")
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
        print(f"Splited Array Shape: {splited_array.shape}")
    return splited_array


def pxxxtzyx2xxxtzyx(input_array: torch.Tensor, verbose=False) -> torch.Tensor:
    if verbose:
        print("@pxxxtzyx2xxxtzyx......")
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
        print(f"Restored Array Shape: {restored_array.shape}")
    return restored_array


def xxx2hdf5_xxx(input_array: torch.Tensor, file_name: str = 'xxx.h5'):
    comm = MPI.COMM_WORLD
    print(f"Input Array Shape: {input_array.shape}")
    dtype = input_array.dtype
    shape = input_array.shape
    with h5py.File(file_name, 'w', driver='mpio', comm=comm) as f:
        dest = f.create_dataset('data', shape=shape, dtype=dtype)
        dest[...] = input_array.cpu().numpy()
        print(f"Dest Shape: {dest.shape}")
        print(f"Data is saved to {file_name}")


def hdf5_xxx2xxx(device: torch.device, file_name: str = 'xxx.h5') -> torch.Tensor:
    comm = MPI.COMM_WORLD
    with h5py.File(file_name, 'r', driver='mpio', comm=comm) as f:
        all_dest = f['data']
        dest = all_dest[...]
        print(f"Dest Shape: {dest.shape}")
        return torch.from_numpy(dest).to(device=device).clone()


def grid_xxxtzyx2hdf5_xxxtzyx(
    input_tensor: torch.Tensor,
    file_name: str,
    lat_size: Tuple[int, int, int, int],
):
    """
    Write local PyTorch tensor blocks to a global HDF5 file using MPI parallel I/O.
    input_tensor: [..., local_t, local_z, local_y, local_x]
    comm: MPI communicator from mpi4py.MPI
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Input Tensor Shape: {input_tensor.shape}")
    lat_x, lat_y, lat_z, lat_t = lat_size
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    dtype = input_tensor.cpu().numpy().dtype
    prefix_shape = input_tensor.shape[:-4]
    # Compute rank indices in the 4D process grid
    grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index()
    print(
        f"Grid Index T: {grid_index_t}, Z: {grid_index_z}, Y: {grid_index_y}, X: {grid_index_x}")
    # Compute local lattice size per block
    grid_lat_t = lat_t // grid_t
    grid_lat_z = lat_z // grid_z
    grid_lat_y = lat_y // grid_y
    grid_lat_x = lat_x // grid_x
    print(
        f"Grid Lat T: {grid_lat_t}, Z: {grid_lat_z}, Y: {grid_lat_y}, X: {grid_lat_x}")
    # Open HDF5 file with MPI parallel I/O
    with h5py.File(file_name, 'w', driver='mpio', comm=comm) as f:
        dest = f.create_dataset('data', shape=(
            *prefix_shape, lat_t, lat_z, lat_y, lat_x), dtype=dtype)
        dest[...,
             grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t,
             grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
             grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
             grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x] = input_tensor.cpu().numpy()
        print(f"Dest Shape: {dest.shape}")
        print(f"rank {rank}: Data is saved to {file_name}")


def hdf5_xxxtzyx2grid_xxxtzyx(
    file_name: str,
    lat_size: Tuple[int, int, int, int],
    device: torch.device
) -> torch.Tensor:
    """
    Read the local block from a global HDF5 file using MPI parallel I/O.
    """
    comm = MPI.COMM_WORLD
    lat_x, lat_y, lat_z, lat_t = lat_size
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    # Compute rank indices in the 4D process grid
    grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index()
    print(
        f"Grid Index T: {grid_index_t}, Z: {grid_index_z}, Y: {grid_index_y}, X: {grid_index_x}")
    # Compute local lattice size per block
    grid_lat_t = lat_t // grid_t
    grid_lat_z = lat_z // grid_z
    grid_lat_y = lat_y // grid_y
    grid_lat_x = lat_x // grid_x
    print(
        f"Grid Lat T: {grid_lat_t}, Z: {grid_lat_z}, Y: {grid_lat_y}, X: {grid_lat_x}")
    with h5py.File(file_name, 'r', driver='mpio', comm=comm) as f:
        all_data = f['data']
        print(f"All Dest Shape: {all_data.shape}")
        dest = all_data[...,
                        grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t,
                        grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
                        grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
                        grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x]
        print(f"Dest Shape: {dest.shape}")
        return torch.from_numpy(dest).to(device=device).clone()
