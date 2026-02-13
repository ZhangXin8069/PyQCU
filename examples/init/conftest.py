from pyqcu.testing import *
# test_import()
# test_lattice()
# test_dslash_wilson()
# test_dslash_clover()
test_solver(method='bistabcg', dtype=torch.complex128,
            lat_size=[8, 16, 16, 16])
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 16, 16, 16])
