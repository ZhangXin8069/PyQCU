from pyqcu.testing import *
import torch
import torch_npu
# test_solver(method='bistabcg', dtype=torch.complex64,
#             lat_size=[8, 8, 8, 8])
# test_solver(method='multigrid', dtype=torch.complex64,
#             lat_size=[8, 8, 8, 8], device=torch.device('npu'),max_levels=1)
# test_solver(method='multigrid', dtype=torch.complex64,
#             lat_size=[8, 8, 8, 8], device=torch.device('npu'),max_levels=2)
# test_solver(method='multigrid', dtype=torch.complex64,
#             lat_size=[8, 8, 16, 16], device=torch.device('npu'),max_levels=1)
# test_solver(method='multigrid', dtype=torch.complex64,
#             lat_size=[8, 8, 16, 16], device=torch.device('npu'),max_levels=2)
# test_solver(method='multigrid', dtype=torch.complex64,
#             lat_size=[8, 8, 8, 16], device=torch.device('npu'),max_levels=1)
# test_solver(method='multigrid', dtype=torch.complex64,
#             lat_size=[8, 8, 8, 16], device=torch.device('npu'),max_levels=2, num_restart=1)
# test_solver(method='multigrid', dtype=torch.complex64,
#             lat_size=[8, 8, 8, 16], device=torch.device('npu'),max_levels=2, num_restart=3)
# test_solver(method='multigrid', dtype=torch.complex64,
#             lat_size=[8, 8, 8, 16], device=torch.device('npu'),max_levels=2, num_restart=5)
# test_solver(method='bistabcg', dtype=torch.complex64,
#                    lat_size=[8, 8, 8, 8], device=torch.device('npu'),support_parity=False)
# test_solver(method='bistabcg', dtype=torch.complex64,
#                    lat_size=[8, 8, 8, 8], device=torch.device('npu'),support_parity=True)
# test_solver(method='multigrid', dtype=torch.complex64,
#                    lat_size=[8, 8, 8, 8], device=torch.device('npu'),support_parity=False)
# test_solver(method='multigrid', dtype=torch.complex64,
#                    lat_size=[8, 8, 8, 8], device=torch.device('npu'),support_parity=True)
# test_solver(method='bistabcg', dtype=torch.complex64,
#                    lat_size=[8, 16, 16, 16], device=torch.device('npu'),support_parity=False)
# test_solver(method='bistabcg', dtype=torch.complex64,
#                    lat_size=[8, 16, 16, 16], device=torch.device('npu'),support_parity=True)
# test_solver(method='multigrid', dtype=torch.complex64,
#                    lat_size=[8, 16, 16, 16], device=torch.device('npu'),support_parity=False)
# test_solver(method='multigrid', dtype=torch.complex64,
#                    lat_size=[8, 16, 16, 16], device=torch.device('npu'),support_parity=True)
# test_solver(method='bistabcg', dtype=torch.complex64,
#                    lat_size=[16, 16, 16, 32], device=torch.device('npu'),support_parity=False)
# test_solver(method='bistabcg', dtype=torch.complex64,
#                    lat_size=[16, 16, 16, 32], device=torch.device('npu'),support_parity=True)
# test_solver(method='multigrid', dtype=torch.complex64,
#                    lat_size=[16, 16, 16, 32], device=torch.device('npu'),support_parity=False)
# test_solver(method='multigrid', dtype=torch.complex64,
#                    lat_size=[16, 16, 16, 32], device=torch.device('npu'),support_parity=True)
# test_solver(method='bistabcg', dtype=torch.complex64,
#                    lat_size=[8, 8, 8, 16], device=torch.device('npu'),support_parity=True)
# test_smear_stout(device=torch.device('npu'), dtype=torch.complex64)
# test_dslash_clover(with_data=False, device=torch.device(
#     'npu'), dtype=torch.complex64)
test_solver(kind='wilson', method='multigrid', dtype=torch.complex64,
            lat_size=[32, 32, 32, 64], device=torch.device('npu'), support_parity=True)
