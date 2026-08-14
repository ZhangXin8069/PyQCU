from pyqcu.tools import HAS_MPI_SUPPORT, give_grid_index, give_grid_size
import h5py
import torch
from mpi4py import MPI
from typing import List


def gridoooxyzt2hdf5oooxyzt(
    input_tensor: torch.Tensor,
    file_name: str,
    lat_size: List[int],
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
        comm.Barrier()
        # Use serial I/O - gather all data to rank 0
        local_data = input_tensor.cpu().contiguous().numpy()
        # Gather all local data to rank 0
        all_data = comm.gather(local_data, root=0)
        # BUGFIX 2026-07-28: unify shape with MPI path (x, y, z, t order).
        # Gather order is (t, z, y, x); unpack must match, then reorder for HDF5 slicing.
        all_indices = comm.gather(
            (grid_index_t, grid_index_z, grid_index_y, grid_index_x), root=0)
        if rank == 0:
            with h5py.File(file_name, 'w') as f:
                # Shape matches MPI path for cross-compatibility
                dest = f.create_dataset('data', shape=(
                    *prefix_shape, lat_x, lat_y, lat_z, lat_t), dtype=dtype)
                # Write each rank's data to the correct position
                for data, indices in zip(all_data, all_indices):
                    idx_t, idx_z, idx_y, idx_x = indices
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
    lat_size: List[int],
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
            return torch.from_numpy(dest).to(device=device)
    else:
        # Use serial I/O - rank 0 reads, then scatter to all ranks
        if rank == 0:
            with h5py.File(file_name, 'r') as f:
                all_data = f['data']
                # Read and scatter data to all ranks
                local_blocks = []
                for r in range(comm.Get_size()):
                    # BUGFIX 2026-07-28: use C-order (last-dim-fastest) decomposition
                    # matching give_grid_index() which uses arange.reshape(give_grid_size()).
                    r_idx_t = r % grid_t                         # last dim, fastest varying
                    r_idx_z = (r // grid_t) % grid_z
                    r_idx_y = (r // (grid_t * grid_z)) % grid_y
                    r_idx_x = r // (grid_t * grid_z * grid_y)    # first dim, slowest varying
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
        # NOTE: comm.scatter uses pickle serialization internally.
        # For very large lattices (> 64^4 with float32: > 4 GB per block),
        # this may hit pickle's 2GB size limit. In such cases, the MPI I/O
        # path (HAS_MPI_SUPPORT=True) should be preferred as it avoids
        # serialization entirely.
        dest = comm.scatter(local_blocks, root=0)
        if verbose:
            print(f"PYQCU::TOOLS::IO:\n rank {rank}: Dest Shape: {dest.shape}")
        return torch.from_numpy(dest).to(device=device)


def save_tensor_h5(tensor: torch.Tensor, file_name: str, dataset: str = 'data',
                   verbose: bool = False):
    """单张量 HDF5 保存（h5py，多线程安全）。

    每次调用新建独立 File 句柄（with 语句），不持有全局句柄；
    多线程各线程独立调用即可安全并发（HDF5 线程安全模式），
    适用于 null-vector / 粗网格算子缓存等本地持久化。
    张量内部布局 xyzt 原样保存；用户自行保证与加载端布局约定一致。
    """
    import numpy as np
    arr = tensor.detach().cpu().contiguous().numpy()
    with h5py.File(file_name, 'w') as f:
        f.create_dataset(dataset, data=arr)
    if verbose:
        print(f"PYQCU::TOOLS::IO:\n Tensor {tuple(tensor.shape)} saved to "
              f"{file_name} (dataset='{dataset}')")


def load_tensor_h5(file_name: str, dataset: str = 'data', device: torch.device = torch.device('cpu'),
                   verbose: bool = False) -> torch.Tensor:
    """单张量 HDF5 读取（h5py，多线程安全）。

    每次调用新建独立 File 句柄（with 语句）；多线程并发读取安全。
    """
    with h5py.File(file_name, 'r') as f:
        arr = f[dataset][...]
    if verbose:
        print(f"PYQCU::TOOLS::IO:\n Tensor {arr.shape} loaded from "
              f"{file_name} (dataset='{dataset}')")
    return torch.from_numpy(arr).to(device=device)
