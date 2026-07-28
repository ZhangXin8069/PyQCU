import torch
import numpy as np
from mpi4py import MPI
import pyqcu.cann as _torch


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
    # OPT 2026-07-28 R2: removed redundant Barrier() — Allreduce is already blocking.
    # Each BiCGStab iteration calls vdot ~5 times; removing these saves ~10
    # unnecessary global synchronizations per iteration.
    local_dot = _torch.vdot(a.flatten(), b.flatten())
    sendbuf = local_dot.detach().cpu().contiguous().numpy()
    recvbuf = np.zeros_like(sendbuf)
    comm.Allreduce(sendbuf=sendbuf, recvbuf=recvbuf, op=MPI.SUM)
    return torch.from_numpy(recvbuf).to(device=device)


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
    return torch.sqrt(vdot(a=a, b=a).real).item()
