from mpi4py import MPI
if not MPI.Is_initialized():
    MPI.Init()
from pyqcu.ascend import qcu
import torch
# lat_x, lat_y, lat_z, lat_t = 16, 16, 16, 16
lat_x, lat_y, lat_z, lat_t = 16, 16, 16, 8
# lat_x, lat_y, lat_z, lat_t = 16, 16, 8, 8
# lat_x, lat_y, lat_z, lat_t = 8, 8, 8, 8
# lat_x, lat_y, lat_z, lat_t = 4, 4, 4, 4
# lat_x, lat_y, lat_z, lat_t = 4, 4, 8, 8
_qcu = qcu(lat_size=[lat_x, lat_y, lat_z, lat_t], dtype_list=[torch.complex128, torch.complex64, torch.complex64, torch.complex32], device_list=[
           torch.device('cpu'), torch.device('cpu'), torch.device('cpu'), torch.device('cpu')], dslash='clover', solver='mg', verbose=False)
_qcu.load()
_qcu.init()
_qcu.solve()
_qcu.test()
_qcu.mg.plot()
_qcu.save()
