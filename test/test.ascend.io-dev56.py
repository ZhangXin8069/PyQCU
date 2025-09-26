from pyqcu.ascend import qcu
import torch
_qcu=qcu(lat_size=[2,2,2,2],dtype=torch.complex128)
_qcu.init()
_qcu.test()
_qcu.init()
_qcu.save()