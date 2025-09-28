import torch
from pyqcu.ascend import qcu
_qcu = qcu(lat_size=[8, 8, 16, 16], if_multi=False,
           dtype=torch.complex128, device=torch.device('cpu'))
_qcu.init()
_qcu.save(file_name='test.ascend-dev56')
_qcu.solve()
