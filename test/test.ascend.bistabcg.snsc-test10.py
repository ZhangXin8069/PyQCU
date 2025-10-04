from pyqcu.ascend import qcu
import torch
lat_n = 32
_qcu = qcu(lat_size=[lat_n, lat_n, lat_n, lat_n], dtype=torch.complex128,
           device=torch.device('cuda'), dslash='clover', solver='bistabcg', verbose=False)
# _qcu.load()
_qcu.init()
_qcu.solve()
_qcu.test()
_qcu.save()