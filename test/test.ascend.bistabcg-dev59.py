from pyqcu.ascend import qcu
import torch
lat_x, lat_y, lat_z, lat_t = 32, 32, 32, 32
_qcu = qcu(lat_size=[lat_x, lat_y, lat_z, lat_t], dtype=torch.complex64, device=torch.device(
    'cuda'), dslash='wilson', solver='bistabcg', verbose=True)
_qcu.load(file_name='quda_wilson_inverse_-32-32-32-32-f')
_qcu.init()
_qcu.solve()
_qcu.test()
_qcu.mg.plot()
_qcu.save(file_name='quda_wilson_inverse_-32-32-32-32-f')