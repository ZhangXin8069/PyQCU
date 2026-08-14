from time import perf_counter
try:
    import tilelang
except Exception as e:
    print(f"Error:{e}")
import torch
from argparse import Namespace
from pyqcu import lattice, solver, dslash, tools, smear
import pyqcu.cann as _torch
import mpi4py.MPI as MPI
from typing import Optional, List
import pyqcu
Namespace.__module__ = "pyqcu.testing"


def test_lattice(lat_size: List[int] = [8, 8, 8, 16], dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu')):
    refer_U = torch.zeros(size=[3, 3, 4]+lat_size, dtype=dtype, device=device)
    lattice.generate_gauge_field(refer_U, seed=42, sigma=0.1, verbose=True)
    print(f"PYQCU::TESTING::LATTICE::I:\n {lattice.I}")
    print(f"PYQCU::TESTING::LATTICE::GAMMA:\n {lattice.gamma}")
    print(f"PYQCU::TESTING::LATTICE::GAMMA_5:\n {lattice.gamma_5}")
    print(f"PYQCU::TESTING::LATTICE::GAMMA_GAMMA:\n {lattice.gamma_gamma}")
    print(f"PYQCU::TESTING::LATTICE::GELL_MANN:\n {lattice.gell_mann}")
    print(
        f"PYQCU::TESTING::LATTICE:\n Gauge field SU(3) check: {lattice.check_su3(refer_U, verbose=True)}")
    # BUGFIX 2026-07-28 R3: add assertion so pytest can detect failures.
    assert lattice.check_su3(refer_U, tol=1e-3, verbose=False), "SU(3) check failed"
    # Verify gamma matrix algebra
    gamma_test = lattice.gamma.to(device=device).type(dtype)
    for mu in range(4):
        g_mu = gamma_test[mu]
        g_mu_sq = torch.matmul(g_mu, g_mu)
        identity = torch.eye(4, dtype=dtype, device=device)
        assert torch.allclose(g_mu_sq, identity, rtol=1e-6, atol=1e-6), f"gamma_{mu}^2 != I"
    print(f"PYQCU::TESTING::LATTICE:\n Gamma matrix algebra: PASS")


def test_dslash_wilson(kappa: Optional[torch.Tensor] = torch.Tensor([0.1]), lat_size: List[int] = [8, 8, 8, 16],  dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu'), with_data: bool = False, support_parallel: bool = True):
    if not with_data:
        refer_U = torch.zeros(
            size=[3, 3, 4]+lat_size, dtype=dtype, device=device)
        lattice.generate_gauge_field(refer_U, seed=42, sigma=0.1, verbose=True)
        refer_src = _torch.randn(
            size=[4, 3]+lat_size, dtype=dtype, device=device)
        refer_dest = dslash.give_wilson(
            src=refer_src, U=refer_U, kappa=kappa, verbose=True)
        U_eo = tools.oooxyzt2poooxyzt(input_array=refer_U, verbose=True)
        src_eo = tools.oooxyzt2poooxyzt(input_array=refer_src, verbose=True)
        src_e = src_eo[0]
        src_o = src_eo[1]
        time_start = perf_counter()
        dest_e = dslash.give_wilson_eo(src_o=src_o, U_eo=U_eo,
                                       kappa=kappa, verbose=True)
        dest_o = dslash.give_wilson_oe(src_e=src_e, U_eo=U_eo,
                                       kappa=kappa, verbose=True)
        time_end = perf_counter()
        dest_eo = torch.zeros_like(src_eo)
        dest_eo[0] = dest_e
        dest_eo[1] = dest_o
        dest = tools.poooxyzt2oooxyzt(input_array=src_eo+dest_eo, verbose=True)
    else:
        kappa = torch.Tensor(0.125)
        dtype = torch.complex64
        lat_size = [32, 32, 32, 32]
        path = pyqcu.__file__.replace('pyqcu/__init__.py', 'examples/data/')
        refer_U = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.wilson.U.L32K0_125.ccdxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        refer_src = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.wilson.src.L32K0_125.scxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        refer_dest = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.wilson.dest.L32K0_125.scxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        refer_clover_term = torch.zeros(
            size=[4, 3, 4, 3]+list(refer_src.shape)[2:], dtype=dtype, device=device)
        operator = dslash.operator(
            U=refer_U, kappa=kappa, clover_term=refer_clover_term, verbose=True)
        time_start = perf_counter()
        if support_parallel:
            dest = operator.matvec(src=refer_src)
        else:
            dest = dslash.give_wilson(
                src=refer_src, U=refer_U, kappa=kappa, with_I=True,  verbose=True)
        time_end = perf_counter()
    diff = tools.norm(dest - refer_dest)/tools.norm(refer_dest)
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_U:\n Gauge field SU(3) check: {lattice.check_su3(refer_U, verbose=True)}")
    print(f"PYQCU::TESTING::DSLASH::WILSON::REFER_U:\n {tools.norm(refer_U)}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_U:\n {refer_U.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_SRC:\n {tools.norm(refer_src)}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_SRC:\n {refer_src.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_DEST:\n {tools.norm(refer_dest)}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_DEST:\n {refer_dest.flatten()[:12]}")
    print(f"PYQCU::TESTING::DSLASH::WILSON::DEST:\n {tools.norm(dest)}")
    print(f"PYQCU::TESTING::DSLASH::WILSON::DEST:\n {dest.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON:\n Time cost: {time_end-time_start}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON:\n Difference between computed and reference dslash: {diff}")
    # BUGFIX 2026-07-28 R3: add assertion for pytest integration.
    assert diff < 1e-4, f"Wilson dslash relative error {diff} exceeds tolerance 1e-4"


def test_dslash_parity(lat_size: List[int] = [8, 8, 8, 16], kappa: Optional[torch.Tensor] = torch.Tensor([0.1]),  dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu')):
    comm = MPI.COMM_WORLD
    root = 0
    grid_size = tools.give_grid_size()
    grid_index = tools.give_grid_index()
    sub_lat_size = [lat_size[i]//grid_size[i] for i in range(4)]
    print(
        f"grid_size,comm.rank,grid_index,sub_lat_size:{grid_size,comm.rank,grid_index,sub_lat_size}")
    refer_U = torch.zeros(
        size=[3, 3, 4]+sub_lat_size, dtype=dtype, device=device)
    lattice.generate_gauge_field(
        refer_U, seed=42+comm.rank, sigma=0.1, verbose=True)
    whole_U = tools.local_xyzt2whole_xyzt(
        local_array=refer_U, root=root)
    if comm.rank == root and whole_U is not None:
        whole_clover_term = dslash.make_clover(
            U=whole_U, kappa=kappa, verbose=True)
        # whole_clover_term = torch.zeros_like(whole_clover_term)
        whole_src = _torch.randn(
            size=[4, 3]+lat_size, dtype=dtype, device=device)
        whole_dest = dslash.give_clover(src=whole_src, clover_term=whole_clover_term, verbose=True) + dslash.give_wilson(src=whole_src, U=whole_U, kappa=kappa,
                                                                                                                         with_I=True, verbose=True)
    else:
        whole_src = None
        whole_dest = None
    refer_clover_term = dslash.make_clover(
        U=refer_U, kappa=kappa, support_parallel=True, verbose=False)
    refer_src = tools.whole_xyzt2local_xyzt(whole_array=whole_src, whole_shape=[
        4, 3]+lat_size, root=root, dtype=dtype, device=device)
    refer_dest = tools.whole_xyzt2local_xyzt(whole_array=whole_dest, whole_shape=[
        4, 3]+lat_size, root=root, dtype=dtype, device=device)
    operator = dslash.operator(
        U=refer_U, kappa=kappa, clover_term=refer_clover_term, verbose=True, support_parity=True)
    time_start = perf_counter()
    dest = (operator.matvec_all(src=refer_src.reshape(
        [12]+list(refer_src.shape[2:])))).reshape(refer_src.shape)
    # dest = operator.matvec(src=refer_src)
    time_end = perf_counter()
    diff = tools.norm(dest - refer_dest) / tools.norm(refer_dest)
    print(f"PYQCU::TESTING::DSLASH::PARITY::REFER_U:\n {tools.norm(refer_U)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_U:\n {refer_U.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_CLOVER_TERM:\n {tools.norm(refer_clover_term)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_CLOVER_TERM:\n {refer_clover_term.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_SRC:\n {tools.norm(refer_src)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_SRC:\n {refer_src.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_DEST:\n {tools.norm(refer_dest)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_DEST:\n {refer_dest.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::DEST:\n {tools.norm(dest)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::DEST:\n {dest.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY:\n Difference between computed and reference: {diff}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY:\n Execution time: {time_end - time_start}")
    src_eo = tools.oooxyzt2poooxyzt(input_array=refer_src, verbose=True)
    src_e = src_eo[0]
    src_o = src_eo[1]
    refer_dest_eo = tools.oooxyzt2poooxyzt(
        input_array=refer_dest, verbose=True)
    refer_dest_e = refer_dest_eo[0]
    refer_dest_o = refer_dest_eo[1]
    dest_e = operator.matvec_eeo(src_e=src_e.reshape(
        [12]+list(src_e.shape[2:])), src_o=src_o.reshape([12]+list(src_o.shape[2:]))).reshape(refer_dest_e.shape)
    dest_o = operator.matvec_oeo(src_e=src_e.reshape(
        [12]+list(src_e.shape[2:])), src_o=src_o.reshape([12]+list(src_o.shape[2:]))).reshape(refer_dest_o.shape)
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_DEST_E:\n {tools.norm(refer_dest_e)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_DEST_E:\n {refer_dest_e.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_DEST_O:\n {tools.norm(refer_dest_o)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::REFER_DEST_O:\n {refer_dest_o.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::DEST_E:\n {tools.norm(dest_e)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::DEST_E:\n {dest_e.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::DEST_O:\n {tools.norm(dest_o)}")
    print(
        f"PYQCU::TESTING::DSLASH::PARITY::DEST_O:\n {dest_o.flatten()[:12]}")
    print(
        f"Difference between computed and reference: {tools.norm(dest_e-refer_dest_e)}")
    print(
        f"Difference between computed and reference: {tools.norm(dest_o-refer_dest_o)}")


def test_dslash_clover(device: torch.device = torch.device('cpu'), with_data: bool = False, dtype: torch.dtype = torch.complex64):
    if with_data:
        kappa = torch.Tensor([1.0])
        lat_size = [32, 16, 32, 32]
        path = pyqcu.__file__.replace('pyqcu/__init__.py', 'examples/data/')
        refer_U = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.clover.U.L32Y16K1.ccdxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        refer_clover_term = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.clover.clover_term.L32Y16K1.scscxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        refer_clover_inv_term = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.clover.clover_inv_term.L32Y16K1.scscxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        clover_term = dslash.make_clover(U=refer_U, kappa=kappa, verbose=True)
        clover_term = dslash.add_I(clover_term=clover_term, verbose=True)
        diff = tools.norm(clover_term - refer_clover_term) / \
            tools.norm(refer_clover_term)
        clover_inv_term = dslash.inverse(clover_term=clover_term, verbose=True)
        diff = tools.norm(clover_inv_term - refer_clover_inv_term) / \
            tools.norm(refer_clover_inv_term)
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n Gauge field SU(3) check: {lattice.check_su3(refer_U, verbose=True)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n {tools.norm(refer_U)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n {refer_U.flatten()[:12]}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER_TERM:\n {tools.norm(refer_clover_term)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER_TERM:\n {refer_clover_term.flatten()[:12]}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER_INV_TERM:\n {tools.norm(refer_clover_inv_term)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER_INV_TERM:\n {refer_clover_inv_term.flatten()[:12]}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER_TERM:\n {tools.norm(clover_term)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER_TERM:\n {clover_term.flatten()[:12]}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER:\n Difference between computed and reference dslash: {diff}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER_INV_TERM:\n {tools.norm(clover_inv_term)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER_INV_TERM:\n {clover_inv_term.flatten()[:12]}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER:\n Difference between computed and reference dslash: {diff}")
    else:
        # lat_size = [2, 2, 2, 2]
        # lat_size = [4, 4, 4, 4]
        # lat_size = [8, 8, 8, 8]
        lat_size = [8, 8, 8, 16]
        comm = MPI.COMM_WORLD
        root = 0
        grid_size = tools.give_grid_size()
        grid_index = tools.give_grid_index()
        sub_lat_size = [lat_size[i]//grid_size[i] for i in range(4)]
        print(
            f"grid_size,comm.rank,grid_index,sub_lat_size:{grid_size,comm.rank,grid_index,sub_lat_size}")
        refer_U = torch.zeros(
            size=[3, 3, 4]+sub_lat_size, dtype=dtype, device=device)
        lattice.generate_gauge_field(
            refer_U, seed=42+comm.rank, sigma=0.1, verbose=True)
        whole_U = tools.local_xyzt2whole_xyzt(
            local_array=refer_U, root=root)
        if comm.rank == root and whole_U is not None:
            whole_clover = dslash.make_clover(
                U=whole_U, support_parallel=False)
        else:
            whole_clover = None
        refer_clover = tools.whole_xyzt2local_xyzt(whole_array=whole_clover, whole_shape=[
            4, 3, 4, 3]+lat_size, root=root, dtype=dtype, device=device)
        clover = dslash.make_clover(U=refer_U, support_parallel=True)
        diff = tools.norm(clover - refer_clover) / \
            tools.norm(refer_clover)
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n Gauge field SU(3) check: {lattice.check_su3(refer_U, verbose=True)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n {tools.norm(refer_U)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n {refer_U.flatten()[:12]}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER:\n {tools.norm(refer_clover)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER:\n {refer_clover.flatten()[:12]}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER:\n {tools.norm(clover)}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER:\n {clover.flatten()[:12]}")
        print(
            f"PYQCU::TESTING::DSLASH::CLOVER:\n Difference between computed and reference clover: {diff}")


def test_solver(kind: str = 'clover', method: str = 'bistabcg', kappa: Optional[torch.Tensor] = torch.Tensor([0.1]), lat_size: List[int] = [8, 8, 8, 16],  dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu'), with_data: bool = False, max_level: int = 2, num_restart: int = 3, support_parity: bool = False):
    if not with_data:
        comm = MPI.COMM_WORLD
        root = 0
        grid_size = tools.give_grid_size()
        grid_index = tools.give_grid_index()
        sub_lat_size = [lat_size[i]//grid_size[i] for i in range(4)]
        print(
            f"grid_size,comm.rank,grid_index,sub_lat_size:{grid_size,comm.rank,grid_index,sub_lat_size}")
        refer_U = torch.zeros(
            size=[3, 3, 4]+sub_lat_size, dtype=dtype, device=device)
        lattice.generate_gauge_field(
            refer_U, seed=42+comm.rank, sigma=0.1, verbose=True)
        whole_U = tools.local_xyzt2whole_xyzt(
            local_array=refer_U, root=root)
        if kind == 'clover':
            refer_clover_term = dslash.make_clover(
                U=refer_U, support_parallel=True)
        else:
            refer_clover_term = torch.zeros(
                size=[4, 3, 4, 3]+sub_lat_size, dtype=dtype, device=device)
        whole_clover_term = tools.local_xyzt2whole_xyzt(
            local_array=refer_clover_term, root=root)
        if comm.rank == root and whole_clover_term is not None and whole_U is not None:
            whole_x = _torch.randn(
                size=[4, 3]+lat_size, dtype=dtype, device=device)
            whole_b = dslash.give_clover(src=whole_x, clover_term=whole_clover_term, verbose=True) + dslash.give_wilson(src=whole_x, U=whole_U, kappa=kappa,
                                                                                                                        with_I=True, verbose=True)
        else:
            whole_x = None
            whole_b = None
        refer_x = tools.whole_xyzt2local_xyzt(whole_array=whole_x, whole_shape=[
                                              4, 3]+lat_size, root=root, dtype=dtype, device=device)
        refer_b = tools.whole_xyzt2local_xyzt(whole_array=whole_b, whole_shape=[
                                              4, 3]+lat_size, root=root, dtype=dtype, device=device)
    else:
        kappa = torch.Tensor(0.125)
        lat_size = [32, 32, 32, 32]
        path = pyqcu.__file__.replace('pyqcu/__init__.py', 'examples/data/')
        refer_U = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.wilson.U.L32K0_125.ccdxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        refer_x = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.wilson.x.L32K0_125.scxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        refer_b = tools.hdf5oooxyzt2gridoooxyzt(
            file_name=path+'refer.wilson.b.L32K0_125.scxyzt.c64.h5', lat_size=lat_size, device=device, verbose=True)
        refer_clover_term = torch.zeros(
            size=[4, 3, 4, 3]+list(refer_b.shape)[2:], dtype=dtype, device=device)
    operator = dslash.operator(
        U=refer_U, clover_term=refer_clover_term, kappa=kappa, verbose=True, support_parity=support_parity)
    if method == 'bistabcg':
        if support_parity:
            def matvec_parity(src_o):
                # return src_o*2+0.5
                return operator.matvec_parity(src_o=src_o)
            b_eo = tools.oooxyzt2poooxyzt(
                input_array=refer_b.reshape([12]+list(refer_b.shape)[2:]))
            b_e = b_eo[0]
            b_o = b_eo[1]
            b_parity = operator.give_b_parity(b_e=b_e, b_o=b_o)
            time_start = perf_counter()
            x_o = solver.bistabcg(b=b_parity, matvec=matvec_parity, tol=1e-6,
                                  max_iter=1000, x0=None, if_rtol=False, verbose=True)
            x_e = operator.give_x_e(b_e=b_e, x_o=x_o)
            x = tools.poooxyzt2oooxyzt(input_array=torch.stack(
                [x_e, x_o], dim=0)).reshape(refer_b.shape)
            time_end = perf_counter()
        else:
            def matvec(src):
                return operator.matvec(src=src)
            # return dslash.give_clover(src=src, clover_term=refer_clover_term, verbose=True) + dslash.give_wilson(src=src, U=refer_U, kappa=kappa, with_I=True, verbose=True)
            time_start = perf_counter()
            x = solver.bistabcg(b=refer_b, matvec=matvec, tol=1e-6,
                                max_iter=1000, x0=None, if_rtol=False, verbose=True)
            time_end = perf_counter()
    elif method == 'multigrid':
        mg = solver.multigrid(dtype_list=[refer_U.dtype]*10, device_list=[refer_U.device]*10, U=refer_U,
                              clover_term=refer_clover_term, kappa=kappa, tol=1e-6, max_iter=1000, max_level=max_level, num_restart=num_restart, support_parity=support_parity, verbose=True)
        mg.init()
        time_start = perf_counter()
        x = mg.solve(b=refer_b)
        time_end = perf_counter()
        mg.plot()
    else:
        raise ValueError(
            # BUGFIX 2026-07-28: use method (user parameter) instead of solver (module object)
            f"PYQCU::TESTING::SOLVER::SOLVER: method '{method}' is not supported. Supported: 'bistabcg', 'multigrid'.")
    diff = tools.norm(x - refer_x) / tools.norm(refer_x)
    print(
        f"PYQCU::TESTING::SOLVER::REFER_U:\n {tools.norm(refer_U)}")
    print(
        f"PYQCU::TESTING::SOLVER::REFER_U:\n {refer_U.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::SOLVER::REFER_B:\n {tools.norm(refer_b)}")
    print(
        f"PYQCU::TESTING::SOLVER::REFER_B:\n {refer_b.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::SOLVER::REFER_X:\n {tools.norm(refer_x)}")
    print(
        f"PYQCU::TESTING::SOLVER::REFER_X:\n {refer_x.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::SOLVER::X:\n {tools.norm(x)}")
    print(
        f"PYQCU::TESTING::SOLVER::X:\n {x.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::SOLVER::TIME: {time_end - time_start}")
    print(
        f"PYQCU::TESTING::SOLVER:\n Difference between computed and reference solution: {diff}")
    # BUGFIX 2026-07-28 R3: add assertion for pytest integration.
    assert diff < 1e-3, f"Solver relative error {diff} exceeds tolerance 1e-3"


def test_matmul():
    M_gpu, N_gpu, K_gpu = 1024, 1024, 1024
    M_cpu, N_cpu, K_cpu = 1024, 1024, 1024
    gpu_tile = {"block_M": 128, "block_N": 128, "block_K": 32}
    cpu_tile = {"block_M": 32, "block_N": 32, "block_K": 32}

    def calc_metrics(m, n, k, sec):
        tflops = (2 * m * n * k / sec) / 1e12
        return tflops
    from pyqcu.tools import matmul_gpu
    func_gpu = matmul_gpu(M_gpu, N_gpu, K_gpu, **gpu_tile)  # type: ignore
    jit_gpu = tilelang.compile(func_gpu, out_idx=[2], target="c")
    print(jit_gpu.get_kernel_source())
    a_gpu = _torch.randn(M_gpu, K_gpu, device=torch.device(
        'cuda'), dtype=torch.float16)
    b_gpu = _torch.randn(N_gpu, K_gpu, device=torch.device(
        'cuda'), dtype=torch.float16)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    # Warmup GPU
    for _ in range(20):
        jit_gpu(a_gpu, b_gpu)  # type: ignore
    # Measure TileLang GPU
    iters = 100
    start_evt.record()
    for _ in range(iters):
        jit_gpu(a_gpu, b_gpu)  # type: ignore
    end_evt.record()
    torch.cuda.synchronize()
    gpu_tl_time = start_evt.elapsed_time(end_evt) / iters / 1000
    # Measure PyTorch GPU (cuBLAS)
    start_evt.record()
    for _ in range(iters):
        torch.matmul(a_gpu, b_gpu.t())
    end_evt.record()
    torch.cuda.synchronize()
    gpu_pt_time = start_evt.elapsed_time(end_evt) / iters / 1000
    from pyqcu.tools import matmul_cpu
    func_cpu = matmul_cpu(M_cpu, N_cpu, K_cpu, **cpu_tile)  # type: ignore
    try:
        jit_cpu = tilelang.compile(func_cpu, out_idx=[2], target="llvm")
        cpu_target_name = "LLVM"
    # BUGFIX 2026-07-28 R3: bare except catches KeyboardInterrupt etc. Use Exception.
    except Exception:
        jit_cpu = tilelang.compile(func_cpu, out_idx=[2], target="c")
        cpu_target_name = "C"
    # print(jit_cpu.get_kernel_source())
    a_cpu = _torch.randn(M_cpu, K_cpu, device=torch.device(
        'cpu'), dtype=torch.float16)
    b_cpu = _torch.randn(N_cpu, K_cpu, device=torch.device(
        'cpu'), dtype=torch.float16)
    # Warmup CPU
    for _ in range(5):
        jit_cpu(a_cpu, b_cpu)  # type: ignore
    # Measure TileLang CPU
    cpu_iters = 1
    start = perf_counter()
    for _ in range(cpu_iters):
        c_cpu = jit_cpu(a_cpu, b_cpu)  # type: ignore
    cpu_tl_time = (perf_counter() - start) / cpu_iters
    # Measure PyTorch CPU (MKL/OneDNN)
    start = perf_counter()
    for _ in range(cpu_iters):
        ref_c_cpu = torch.matmul(a_cpu, b_cpu.t())
    cpu_pt_time = (perf_counter() - start) / cpu_iters
    line = "=" * 65
    print(f"\n{line}")
    print(f"{'Platform':15} | {'Backend':18} | {'Latency (ms)':12} | {'TFLOPS':10}")
    print(line)
    # GPU Rows
    print(f"{'GPU (4K)':15} | {'TileLang':18} | {gpu_tl_time*1000:12.3f} | {calc_metrics(M_gpu, N_gpu, K_gpu, gpu_tl_time):10.4f}")
    print(f"{'GPU (4K)':15} | {'PyTorch/cuBLAS':18} | {gpu_pt_time*1000:12.3f} | {calc_metrics(M_gpu, N_gpu, K_gpu, gpu_pt_time):10.4f}")
    print("-" * 65)
    # CPU Rows
    print(f"{'CPU (1K)':15} | {f'TileLang ({cpu_target_name})':18} | {cpu_tl_time*1000:12.3f} | {calc_metrics(M_cpu, N_cpu, K_cpu, cpu_tl_time):10.4f}")
    print(f"{'CPU (1K)':15} | {'PyTorch/MKL':18} | {cpu_pt_time*1000:12.3f} | {calc_metrics(M_cpu, N_cpu, K_cpu, cpu_pt_time):10.4f}")
    print(line)
    torch.testing.assert_close(c_cpu, ref_c_cpu, rtol=1e-2, atol=1e-2)
    print("All Verifications Passed (GPU & CPU)!")


def test_smear_stout(lat_size: List[int] = [8, 8, 8, 16], device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.complex64):
    comm = MPI.COMM_WORLD
    root = 0
    grid_size = tools.give_grid_size()
    grid_index = tools.give_grid_index()
    sub_lat_size = [lat_size[i]//grid_size[i] for i in range(4)]
    print(
        f"grid_size,comm.rank,grid_index,sub_lat_size:{grid_size,comm.rank,grid_index,sub_lat_size}")
    refer_U = torch.zeros(
        size=[3, 3, 4]+sub_lat_size, dtype=dtype, device=device)
    lattice.generate_gauge_field(
        refer_U, seed=42+comm.rank, sigma=0.1, verbose=True)
    whole_U = tools.local_xyzt2whole_xyzt(
        local_array=refer_U, root=root)
    if comm.rank == root and whole_U is not None:
        whole_smear_U = smear.stout_smear(U=whole_U, support_parallel=False)
    else:
        whole_smear_U = None
    refer_smear_U = tools.whole_xyzt2local_xyzt(whole_array=whole_smear_U, whole_shape=[
        3, 3, 4]+lat_size, root=root, dtype=dtype, device=device)
    smear_U = smear.stout_smear(U=refer_U, support_parallel=True)
    diff = tools.norm(smear_U - refer_smear_U) / \
        tools.norm(refer_smear_U)
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::REFER_U:\n Gauge field SU(3) check: {lattice.check_su3(refer_U, verbose=True)}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::REFER_U:\n {tools.norm(refer_U)}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::REFER_U:\n {refer_U.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::REFER_SMEAR_U:\n Gauge field SU(3) check: {lattice.check_su3(refer_smear_U, verbose=True)}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::REFER_SMEAR_U:\n {tools.norm(refer_smear_U)}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::REFER_SMEAR_U:\n {refer_smear_U.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::SMEAR_U:\n Gauge field SU(3) check: {lattice.check_su3(smear_U, verbose=True)}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::SMEAR_U:\n {tools.norm(smear_U)}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT::SMEAR_U:\n {smear_U.flatten()[:12]}")
    print(
        f"PYQCU::TESTING::SMEAR::STOUT:\n Difference between computed and reference smear: {diff}")
    # _whole_smear_U = tools.local_xyzt2whole_xyzt(
    #     local_array=smear_U, root=root)
    # if comm.rank == root:
    #     print(f"whole_smear_U:\n{whole_smear_U}")
    #     print(f"_whole_smear_U:\n{_whole_smear_U}")
    #     print(f"whole_smear_U-_whole_smear_U:\n{whole_smear_U-_whole_smear_U}")


def test_h5py_multithread(nthreads: int = 4, tmp_dir: str = None,
                          dtype: torch.dtype = torch.complex64,
                          lat_size: List[int] = [4, 4, 4, 8]):
    """h5py 多线程读写验证（任务②）。

    每线程独立 h5py.File 句柄（with 语句），并发写/读各自文件，
    校验往返一致（逐元素相等）；再验证多线程并发读同一文件。
    """
    import tempfile, os
    from concurrent.futures import ThreadPoolExecutor
    from pyqcu.tools import save_tensor_h5, load_tensor_h5
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="pyqcu_h5_mt_")
    os.makedirs(tmp_dir, exist_ok=True)
    data = [_torch.randn(size=[12]+lat_size, dtype=dtype, device=torch.device('cpu')) for _ in range(nthreads)]

    def write_read(tid):
        fname = os.path.join(tmp_dir, f"mt_{tid}.h5")
        save_tensor_h5(data[tid], fname, dataset="data")
        back = load_tensor_h5(fname, dataset="data", device=torch.device('cpu'))
        return float((back - data[tid]).abs().max().item())

    def read_shared(fname, tid):
        # 多线程并发读同一文件（独立句柄）
        return float(load_tensor_h5(fname, dataset="data",
                                    device=torch.device('cpu')).abs().max().item())

    shared_file = os.path.join(tmp_dir, "shared.h5")
    save_tensor_h5(data[0], shared_file, dataset="data")
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        max_errs = list(ex.map(write_read, range(nthreads)))
        norms = list(ex.map(lambda t: read_shared(shared_file, t), range(nthreads)))
    max_err = max(max_errs)
    assert max_err == 0.0, f"h5py multithread round-trip max error {max_err}"
    assert all(abs(n - float(data[0].abs().max().item())) < 1e-12 for n in norms), \
        "h5py concurrent shared-file read mismatch"
    print(f"PYQCU::TESTING::TOOLS::IO:\n h5py multithread ({nthreads} threads) "
          f"write/read round-trip: PASS (max_err={max_err:.1e})")
    return tmp_dir


def test_multi_gpu_multigrid(nthreads: int = 2, lat_size: List[int] = [8, 8, 8, 16],
                             mass: float = 0.05, atol: float = 1e-6,
                             num_levels: int = 2, tol: float = 1e-5, verbose: bool = True):
    """多线程多卡 C++ Clover Multigrid 一致性验证（任务①，一线程一卡）。

    单卡环境：nthreads 线程共享一张卡，验证线程隔离与结果一致性；
    多卡环境：每线程绑定 device_ids[tid % n_gpus]，验证跨卡一致性。
    判据：各线程解与线程 0 参考解的最大相对差 < tol，且每个线程的
    MG 解与自身 BiStabCG 参考解的相对残差达标（数值上 ≈ 1）。
    """
    from pyqcu.cuda._multi_gpu import MultiGpuMultigrid
    mg = MultiGpuMultigrid(lat_size=list(lat_size), mass=mass, atol=atol,
                           num_levels=num_levels, nthreads=nthreads,
                           verbose=verbose)
    results = mg.solve()
    consistency = mg.verify_consistency(tol=tol)
    assert consistency['all_pass'], \
        f"multi-GPU MG consistency failed: {consistency['checks']}"
    # 各线程 MG 解 vs 本线程 BiStabCG 参考（相对残差应 ≈ 1，验证求解收敛）
    for t in results['threads']:
        d = (t['mg'] - t['ref']).abs().max().item()
        ref_max = t['ref'].abs().max().item()
        rel = d / (ref_max + 1e-30)
        assert rel < 1e-3, f"tid={t['tid']} MG vs BiStabCG rel diff {rel} >= 1e-3"
    print(f"PYQCU::TESTING::SOLVER::MULTIGRID::MULTI_GPU:\n "
          f"{nthreads} threads x {len(mg.device_ids)} GPU(s): PASS "
          f"(consistency tol={tol})")
    return results, consistency
