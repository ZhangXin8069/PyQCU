from pyqcu.testing import *
import torch
import torch_npu
test_solver(method='bistabcg', dtype=torch.complex64, device=torch.device('npu'),
                   lat_size=[8, 16, 16, 16], support_parity=True)
