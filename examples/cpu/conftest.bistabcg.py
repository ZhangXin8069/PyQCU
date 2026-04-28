from pyqcu.testing import *
import torch
test_solver(kind='clover', method='bistabcg', dtype=torch.complex64,
            lat_size=[8, 8, 16, 16], device=torch.device('cpu'), support_parity=True)
