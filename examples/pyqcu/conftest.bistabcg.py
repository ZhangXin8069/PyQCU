from pyqcu.testing import test_solver
import torch

mass = 0.05
kappa = 1 / (2 * mass + 8)
test_solver(method='bistabcg', dtype=torch.complex64, device=torch.device('cuda'),
            kappa=torch.Tensor([kappa]), lat_size=[8, 8, 8, 16], num_restart=3)
