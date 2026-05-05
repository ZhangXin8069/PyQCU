from pyqcu.testing import *
import torch
mass = 0.05
kappa = 1 / (2 * mass + 8)
test_solver(method='multigrid', dtype=torch.complex64, device=torch.device('cuda'), kappa=torch.Tensor([kappa]),
                   lat_size=[16, 16, 16, 32], max_level=1, support_parity=True)
