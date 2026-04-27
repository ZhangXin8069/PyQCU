from pyqcu.testing import *
import torch
test_solver(method='multigrid', dtype=torch.complex64, device=torch.device('cuda'),
                   lat_size=[16, 16, 16, 32], support_parity=True)
