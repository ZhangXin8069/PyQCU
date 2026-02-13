from pyqcu.tools import HAS_MPI_SUPPORT, give_grid_index, give_grid_size
import h5py
import torch
from mpi4py import MPI
from typing import Tuple
def gridoooxyzt2hdf5oooxyzt(
    input_tensor: torch.Tensor,
    file_name: str,
    lat_size: Tuple[int, int, int, int],
    verbose: bool = False
):
    """
    Write local PyTorch tensor blocks to a global HDF5 file using MPI parallel I/O.
    input_tensor: [..., local_t, local_z, local_y, local_x]
    comm: MPI communicator from mpi4py.MPI
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if verbose:
        print(
            f"PYQCU::TOOLS::IO:\n rank {rank}: Input Tensor Shape: {input_tensor.shape}")
    lat_x, lat_y, lat_z, lat_t = lat_size
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    dtype = input_tensor.cpu().numpy().dtype
    prefix_shape = input_tensor.shape[:-4]
    # Compute rank indices in the 4D process grid
    grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index()
    # Compute local lattice size per block
    grid_lat_x = lat_x // grid_x
    grid_lat_y = lat_y // grid_y
    grid_lat_z = lat_z // grid_z
    grid_lat_t = lat_t // grid_t
    if verbose:
        print(
            f"PYQCU::TOOLS::IO:\n rank {rank}: Grid Lat X: {grid_lat_x}, Y: {grid_lat_y}, Z: {grid_lat_z}, T: {grid_lat_t}")
        print(
            f"PYQCU::TOOLS::IO:\n rank {rank}: Grid Index X: {grid_index_x}, Y: {grid_index_y}, Z: {grid_index_z}, T: {grid_index_t}")
    if HAS_MPI_SUPPORT:
        # Use MPI parallel I/O
        with h5py.File(file_name, 'w', driver='mpio', comm=comm) as f:
            dest = f.create_dataset('data', shape=(
                *prefix_shape, lat_x, lat_y, lat_z, lat_t), dtype=dtype)
            dest[...,
                 grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x,
                 grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
                 grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
                 grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t] = input_tensor.cpu().contiguous().numpy()
            if verbose:
                print(
                    f"PYQCU::TOOLS::IO:\n rank {rank}: Dest Shape: {dest.shape}")
            print(
                f"PYQCU::TOOLS::IO:\n rank {rank}: Data is saved to {file_name} (MPI mode)")
    else:
        # Use serial I/O - gather all data to rank 0
        local_data = input_tensor.cpu().contiguous().numpy()
        # Gather all local data to rank 0
        all_data = comm.gather(local_data, root=0)
        all_indices = comm.gather(
            (grid_index_t, grid_index_z, grid_index_y, grid_index_x), root=0)
        if rank == 0:
            with h5py.File(file_name, 'w') as f:
                dest = f.create_dataset('data', shape=(
                    *prefix_shape, lat_t, lat_z, lat_y, lat_x), dtype=dtype)
                # Write each rank's data to the correct position
                for data, indices in zip(all_data, all_indices):
                    idx_x, idx_y, idx_z, idx_t = indices
                    dest[...,
                         idx_x*grid_lat_x:idx_x*grid_lat_x+grid_lat_x,
                         idx_y*grid_lat_y:idx_y*grid_lat_y+grid_lat_y,
                         idx_z*grid_lat_z:idx_z*grid_lat_z+grid_lat_z,
                         idx_t*grid_lat_t:idx_t*grid_lat_t+grid_lat_t] = data
                if verbose:
                    print(f"PYQCU::TOOLS::IO:\n Dest Shape: {dest.shape}")
                print(
                    f"PYQCU::TOOLS::IO:\n Data is saved to {file_name} (Serial mode)")
        comm.Barrier()
def hdf5oooxyzt2gridoooxyzt(
    file_name: str,
    lat_size: Tuple[int, int, int, int],
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:
    """
    Read the local block from a global HDF5 file using MPI parallel I/O.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    lat_x, lat_y, lat_z, lat_t = lat_size
    grid_x, grid_y, grid_z, grid_t = give_grid_size()
    # Compute rank indices in the 4D process grid
    grid_index_x, grid_index_y, grid_index_z, grid_index_t = give_grid_index()
    # Compute local lattice size per block
    grid_lat_x = lat_x // grid_x
    grid_lat_y = lat_y // grid_y
    grid_lat_z = lat_z // grid_z
    grid_lat_t = lat_t // grid_t
    if verbose:
        print(
            f"PYQCU::TOOLS::IO:\n rank {rank}: Grid Lat X: {grid_lat_x}, Y: {grid_lat_y}, Z: {grid_lat_z}, T: {grid_lat_t}")
        print(
            f"PYQCU::TOOLS::IO:\n rank {rank}: Grid Index X: {grid_index_x}, Y: {grid_index_y}, Z: {grid_index_z}, T: {grid_index_t}")
    if HAS_MPI_SUPPORT:
        # Use MPI parallel I/O
        with h5py.File(file_name, 'r', driver='mpio', comm=comm) as f:
            all_data = f['data']
            dest = all_data[...,
                            grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x,
                            grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
                            grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
                            grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t]
            if verbose:
                print(
                    f"PYQCU::TOOLS::IO:\n rank {rank}: Dest Shape: {dest.shape}")
                print(
                    f"PYQCU::TOOLS::IO:\n rank {rank}: All Dest Shape: {all_data.shape}")
            print(
                f"PYQCU::TOOLS::IO:\n rank {rank}: Data is loaded from {file_name} (MPI mode)")
            return torch.from_numpy(dest).to(device=device).clone()
    else:
        # Use serial I/O - rank 0 reads, then scatter to all ranks
        if rank == 0:
            with h5py.File(file_name, 'r') as f:
                all_data = f['data']
                # Read and scatter data to all ranks
                local_blocks = []
                for r in range(comm.Get_size()):
                    # Calculate indices for rank r
                    r_idx_x = r % grid_x
                    r_idx_y = (r // grid_x) % grid_y
                    r_idx_z = (r // (grid_x * grid_y)) % grid_z
                    r_idx_t = r // (grid_x * grid_y * grid_z)
                    block = all_data[...,
                                     r_idx_x*grid_lat_x:r_idx_x*grid_lat_x+grid_lat_x,
                                     r_idx_y*grid_lat_y:r_idx_y*grid_lat_y+grid_lat_y,
                                     r_idx_z*grid_lat_z:r_idx_z*grid_lat_z+grid_lat_z,
                                     r_idx_t*grid_lat_t:r_idx_t*grid_lat_t+grid_lat_t]
                    local_blocks.append(block)
                if verbose:
                    print(
                        f"PYQCU::TOOLS::IO:\n All Dest Shape: {all_data.shape}")
                print(
                    f"PYQCU::TOOLS::IO:\n Data is loaded from {file_name} (Serial mode)")
        else:
            local_blocks = None
        # Scatter data to all ranks
        dest = comm.scatter(local_blocks, root=0)
        if verbose:
            print(f"PYQCU::TOOLS::IO:\n rank {rank}: Dest Shape: {dest.shape}")
        return torch.from_numpy(dest).to(device=device).clone()
