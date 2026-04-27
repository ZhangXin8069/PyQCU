from pyqcu.testing import *
import torch
test_solver(method='multigrid', dtype=torch.complex128, device=torch.device('cuda'),
                   lat_size=[8, 8, 8, 16], support_parity=True)
