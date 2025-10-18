from pyqcu.ascend import qcu
import torch
lat_x, lat_y, lat_z, lat_t = 8, 8, 16, 16
_qcu = qcu(lat_size=[lat_x, lat_y, lat_z, lat_t], dtype=torch.complex128,
           device=torch.device('cpu'), dslash='clover', solver='mg', verbose=False)
_qcu.load()
_qcu.init()
_qcu.solve()
_qcu.test()
_qcu.full_mg.plot()
# _qcu.save()
