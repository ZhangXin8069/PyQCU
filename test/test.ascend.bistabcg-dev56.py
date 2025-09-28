import torch
import mpi4py.MPI as MPI
from pyqcu.ascend import qcu
_qcu = qcu(lat_size=[16, 16, 16, 16], dtype=torch.complex128,
           device=torch.device('cpu'), dslash='wilson', verbose=False)
_qcu.init()
# _qcu.load(file_name='test.ascend-dev56')
_qcu.solve()
_qcu.save(file_name='test.ascend-dev56')
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"rank,size: {rank,size}")
