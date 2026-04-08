import datetime
import os
from pyqcu.testing import *
import torch
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
rank = comm.Get_rank()
prof = torch.profiler.profile(
    record_shapes=True,
    with_modules=True,
    with_flops=True,
    with_stack=True,
    acc_events=True
)
prof.start()
test_solver(method='bistabcg', dtype=torch.complex64, device=torch.device(
    'cpu'), lat_size=[8, 8, 16, 16], support_parity=True)
prof.stop()
prof.export_chrome_trace(
    f"{os.path.abspath(os.path.dirname(__file__))}/trace_{time}_{rank}.json")
