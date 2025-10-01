import torch
import mpi4py.MPI as MPI
from pyqcu.ascend import qcu
lat_n = 8
_qcu = qcu(lat_size=[lat_n, lat_n, lat_n, lat_n], dtype=torch.complex64,
           device=torch.device('cpu'), dslash='clover', solver='mg', verbose=False)
# _qcu.load(file_name='test.ascend-dev56')
_qcu.init()
# _qcu.solve()
# _qcu.test()
# _qcu.save(file_name='test.ascend-dev56')
