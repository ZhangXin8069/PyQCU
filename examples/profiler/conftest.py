import datetime
import os
import comm
from pyqcu.testing import *
import cProfile
import torch
# test_import()
# test_lattice()
# test_dslash_wilson(with_data=True, suppoer_multi=True)
# test_dslash_wilson(with_data=True, suppoer_multi=False)
# test_dslash_wilson(with_data=False, suppoer_multi=False)
# test_dslash_wilson(with_data=False)
# test_dslash_clover()
# test_dslash_parity()
# test_solver(method='bistabcg', dtype=torch.complex128,
#             lat_size=[8, 8, 8, 8])
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 8, 8, 8], max_levels=1)
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 8, 8, 8], max_levels=2)
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 8, 16, 16], max_levels=1)
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 8, 16, 16], max_levels=2)
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 8, 8, 16], max_levels=1)
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 8, 8, 16], max_levels=2, num_restart=1)
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 8, 8, 16], max_levels=2, num_restart=3)
# test_solver(method='multigrid', dtype=torch.complex128,
#             lat_size=[8, 8, 8, 16], max_levels=2, num_restart=5)
# test_solver(method='bistabcg', dtype=torch.complex128,
#                    lat_size=[8, 8, 8, 8], support_parity=False)
# test_solver(method='bistabcg', dtype=torch.complex128,
#                    lat_size=[8, 8, 8, 8], support_parity=True)
# test_solver(method='multigrid', dtype=torch.complex128,
#                    lat_size=[8, 8, 8, 8], support_parity=False)
# test_solver(method='multigrid', dtype=torch.complex128,
#                    lat_size=[8, 8, 8, 8], support_parity=True)
# test_solver(method='bistabcg', dtype=torch.complex128,
#                    lat_size=[8, 16, 16, 16], support_parity=False)
# test_solver(method='bistabcg', dtype=torch.complex128,
#                    lat_size=[8, 16, 16, 16], support_parity=True)
# test_solver(method='multigrid', dtype=torch.complex128,
#                    lat_size=[8, 16, 16, 16], support_parity=False)
# test_solver(method='multigrid', dtype=torch.complex128,
#                    lat_size=[8, 16, 16, 16], support_parity=True)
# test_solver(method='bistabcg', dtype=torch.complex128,
#                    lat_size=[16, 16, 16, 32], support_parity=False)
# test_solver(method='bistabcg', dtype=torch.complex128,
#                    lat_size=[16, 16, 16, 32], support_parity=True)
# test_solver(method='multigrid', dtype=torch.complex128,
#                    lat_size=[16, 16, 16, 32], support_parity=False)
# test_solver(method='multigrid', dtype=torch.complex128,
#                    lat_size=[16, 16, 16, 32], support_parity=True)
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
rank = comm.Get_rank()
et_obs = torch.profiler.ExecutionTraceObserver()
et_obs.register_callback(
    f"{os.path.abspath(os.path.dirname(__file__))}/callback_{time}_{rank}.json")
prof = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
    acc_events=True,
    execution_trace_observer=et_obs,
    with_stack=True,
)
prof.start()
test_solver(method='bistabcg', dtype=torch.complex128,
                   lat_size=[8, 8, 8, 16], support_parity=True)
prof.stop()
prof.export_chrome_trace(
    f"{os.path.abspath(os.path.dirname(__file__))}/trace_{time}_{rank}.json")
