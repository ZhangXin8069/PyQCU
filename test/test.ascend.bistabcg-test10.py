from pyqcu.ascend import qcu
import torch
lat_n = 32
_qcu = qcu(lat_size=[lat_n, lat_n, lat_n, lat_n], dtype=torch.complex128,
           device=torch.device('cpu'), dslash='clover', solver='bistabcg', verbose=False)
# _qcu.load(file_name='test.ascend-dev56')
_qcu.init()
_qcu.solve()
_qcu.test()
_qcu.save(file_name='test.ascend-dev56')