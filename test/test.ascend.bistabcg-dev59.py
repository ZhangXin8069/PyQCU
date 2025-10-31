from pyqcu.ascend import qcu
import torch
lat_x, lat_y, lat_z, lat_t = 32, 32, 32, 32
# lat_x, lat_y, lat_z, lat_t = 16, 16, 16, 16
# lat_x, lat_y, lat_z, lat_t = 16, 16, 16, 8
# lat_x, lat_y, lat_z, lat_t = 16, 16, 8, 8
# lat_x, lat_y, lat_z, lat_t = 8, 8, 8, 8
# lat_x, lat_y, lat_z, lat_t = 4, 4, 4, 4
# lat_x, lat_y, lat_z, lat_t = 4, 4, 4, 8
_qcu = qcu(lat_size=[lat_x, lat_y, lat_z, lat_t], dtype=torch.complex128, device=torch.device(
    'npu'), dslash='clover', solver='bistabcg', verbose=True)
_qcu.load()
_qcu.init()
_qcu.solve()
_qcu.test()
_qcu.mg.plot()
_qcu.save()
