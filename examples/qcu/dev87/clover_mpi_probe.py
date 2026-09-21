#!/usr/bin/env python3
"""Reproduce the legacy Clover gauge-halo path on a distributed lattice."""

import argparse
import os

import numpy as np
import torch
from mpi4py import MPI

import common
import pyqcu.cuda.define as define


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", type=int, nargs=4, default=[8, 8, 8, 16])
    parser.add_argument("--grid", type=int, nargs=4, default=None)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    grid = common.process_grid(args.grid)
    if int(comm.size) != int(torch.tensor(grid).prod()):
        raise RuntimeError(f"MPI size {comm.size} does not match grid {grid}")
    visible = int(torch.cuda.device_count())
    if visible <= 0:
        raise RuntimeError("no visible CUDA device")
    device_index = (
        int(comm.rank) if visible > 1 else 0)
    if device_index >= visible:
        raise RuntimeError(
            f"rank {comm.rank} maps outside {visible} visible CUDA devices")
    torch.cuda.set_device(device_index)
    if int(comm.size) > 1:
        os.environ["PYQCU_MPI_DEVICE_ID"] = str(device_index)
    print(
        f"rank={comm.rank} torch_device={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(device_index)}",
        flush=True,
    )
    dtype, data_type = common.parse_complex_dtype(args.dtype)
    gauge = common.load_local_gauge_h5(
        args.lattice, grid=grid, rank=comm.rank, device="cuda", dtype=dtype)
    tensors = common.make_clover_tensors(
        gauge, args.lattice, grid=grid, rank=comm.rank,
        dtype=dtype, data_type=data_type)
    ce, cei, coo, coi, _, _, _ = tensors
    # PyTorch 2.7's CUDA build has no sm_60 kernels, so perform the probe's
    # checksum on the host after a plain device-to-host copy.
    checksum = float(sum(
        float(np.abs(tensor.detach().cpu().numpy()).sum())
        for tensor in (ce, cei, coo, coi)))
    print(
        f"rank={comm.rank} grid={grid} device={torch.cuda.get_device_name(0)} "
        f"shape={tuple(ce.shape)} checksum={checksum:.9e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
