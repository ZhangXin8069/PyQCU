import torch
import numpy as np
from mpi4py import MPI
from pyqcu import _torch


def vdot(
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
    local_dot = _torch.vdot(a.flatten(), b.flatten())
    sendbuf = local_dot.detach().cpu().contiguous().numpy()
    recvbuf = np.zeros_like(sendbuf).copy()
    comm.Allreduce(sendbuf=sendbuf, recvbuf=recvbuf, op=MPI.SUM)
    comm.Barrier()
    return torch.from_numpy(recvbuf).to(device=device).clone()


def norm(
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-process norm with buffer mode.
    Args:
        a: local tensor
    Returns:
        global norm
    """
    return torch.sqrt(vdot(a=a.flatten(), b=a.flatten()).real).item()
