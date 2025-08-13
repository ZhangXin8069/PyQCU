import torch
import h5py
import mpi4py.MPI as MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def any_xxx2hdf5_xxx(input_array: torch.Tensor, file_name: str = 'xxx.h5') -> torch.Tensor:
    _input_array = input_array.cpu().numpy()
    with h5py.File(file_name, 'w', driver='mpio', comm=mpi_comm) as f:
        hdf5_array = f.create_dataset(
            'data', shape=_input_array.shape, dtype=_input_array.dtype)
        hdf5_array[...] = _input_array
        print(f"Data Shape: {hdf5_array.shape}")
        print(f"Data is saved to {file_name}")


def hdf5_xxx2cpu_xxx(file_name: str = 'xxx.h5') -> torch.Tensor:
    with h5py.File(file_name, 'r', driver='mpio', comm=mpi_comm) as f:
        hdf5_array = f['data'][...]
        _hdf5_array = torch.from_numpy(hdf5_array)
        print(f"Data is loaded from {file_name}")
        print(f"Data Shape: {_hdf5_array.shape}")
        return _hdf5_array


def ccdptzyx2pccdtzyx(gauge: torch.Tensor) -> torch.Tensor:
    dest = gauge.permute(3, 0, 1, 2, 4, 5, 6, 7)
    return dest


def give_hdf5_U(device: str, file_name: str = 'xxx.h5') -> torch.Tensor:
    """
    U = pxxxtzyx2xxxtzyx(input_array=ccdptzyx2pccdtzyx(gauge=hdf5_xxx2cpu_xxx(file_name='quda_wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.h5')))
    U = hdf5_xxx2cpu_xxx().detach().clone().to(device)
    U = give_hdf5_U(device=device)
    """
    return hdf5_xxx2cpu_xxx(file_name=file_name).detach().clone().to(device)


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
        torch.arange(t, device=device),
        torch.arange(z, device=device),
        torch.arange(y, device=device),
        torch.arange(x, device=device),
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
        torch.arange(t, device=device),
        torch.arange(z, device=device),
        torch.arange(y, device=device),
        torch.arange(x, device=device),
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


def give_eo_mask(xxxtzy_x_p: torch.Tensor, eo: int, verbose=False) -> torch.Tensor:
    if verbose:
        print("@give_eo_mask......")
    shape = xxxtzy_x_p.shape
    device = xxxtzy_x_p.device
    t, z, y, x_p = shape[-4:]
    # Create coordinate grids for original shape
    coords = torch.meshgrid(
        torch.arange(t, device=device),
        torch.arange(z, device=device),
        torch.arange(y, device=device),
        torch.arange(x_p, device=device),
        indexing='ij'
    )
    # Sum coordinates to determine checkerboard pattern
    sums = coords[0] + coords[1] + coords[2]  # t+z+y
    return sums % 2 == eo
