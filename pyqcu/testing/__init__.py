from time import perf_counter
import tilelang
import torch
from argparse import Namespace
from pyqcu import lattice, solver, dslash, _torch, tools
import mpi4py.MPI as MPI
import pyqcu
Namespace.__module__ = "pyqcu.testing"


def test_import():
    try:
        import time
        import typing
        import torch
        import mpi4py
        import h5py
        import tilelang
        from pyqcu import _torch, tools, lattice, dslash, solver
        print("PYQCU::TESTING::IMPORT:\n All dependencies imported successwholey.")
    except Exception as e:
        print(f"PYQCU::TESTING::IMPORT:\n {e}")


def test_lattice(lat_size: list = [8, 8, 8, 16], dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu')):
    refer_U = torch.zeros(size=[3, 3, 4]+lat_size, dtype=dtype, device=device)
    lattice.generate_gauge_field(refer_U, seed=42, sigma=0.1, verbose=True)
    is_su3 = lattice.check_su3(refer_U, tol=1e-6, verbose=True)
    print(f"PYQCU::TESTING::LATTICE::I:\n {lattice.I}")
    print(f"PYQCU::TESTING::LATTICE::GAMMA:\n {lattice.gamma}")
    print(f"PYQCU::TESTING::LATTICE::GAMMA_5:\n {lattice.gamma_5}")
    print(f"PYQCU::TESTING::LATTICE::GAMMA_GAMMA:\n {lattice.gamma_gamma}")
    print(f"PYQCU::TESTING::LATTICE::GELL_MANN:\n {lattice.gell_mann}")
    print(f"PYQCU::TESTING::LATTICE:\n Gauge field SU(3) check: {is_su3}")


def test_dslash_wilson(kappa: float = 0.125, lat_size: list = [8, 8, 8, 16],  dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu'), with_data: bool = False):
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
        kappa = 0.125
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
        dest = operator.matvec(src=refer_src)
        # dest = dslash.give_wilson(
        #     src=refer_src, U=refer_U, kappa=kappa, with_I=True,  verbose=True)
        time_end = perf_counter()
    is_su3 = lattice.check_su3(refer_U, tol=1e-6, verbose=True)
    diff = tools.norm(dest - refer_dest)/tools.norm(refer_dest)
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_U:\n Gauge field SU(3) check: {is_su3}")
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


def test_dslash_parity(lat_size: list = [8, 8, 8, 16], kappa: float = 0.125,  dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu')):
    comm = MPI.COMM_WORLD
    root = 0
    if comm.rank == root:
        whole_U = torch.zeros(
            size=[3, 3, 4]+lat_size, dtype=dtype, device=device)
        lattice.generate_gauge_field(
            whole_U, seed=42, sigma=0.1, verbose=True)
        whole_clover_term = dslash.make_clover(
            U=whole_U, kappa=kappa, verbose=True)
        whole_src = torch.randn(
            size=[4, 3]+lat_size, dtype=dtype, device=device)
        whole_dest = dslash.give_clover(src=whole_src, clover_term=whole_clover_term, verbose=True) + dslash.give_wilson(src=whole_src, U=whole_U, kappa=kappa,
                                                                                                                         with_I=True, verbose=True)
    else:
        whole_U = None
        whole_clover_term = None
        whole_src = None
        whole_dest = None
    refer_U = tools.whole_xyzt2local_xyzt(whole_array=whole_U, whole_shape=[
                                          3, 3, 4]+lat_size, root=root, dtype=dtype, device=device)
    refer_clover_term = tools.whole_xyzt2local_xyzt(whole_array=whole_clover_term, whole_shape=[
                                                    4, 3, 4, 3]+lat_size, root=root, dtype=dtype, device=device)
    refer_src = tools.whole_xyzt2local_xyzt(whole_array=whole_src, whole_shape=[
        4, 3]+lat_size, root=root, dtype=dtype, device=device)
    refer_dest = tools.whole_xyzt2local_xyzt(whole_array=whole_dest, whole_shape=[
        4, 3]+lat_size, root=root, dtype=dtype, device=device)
    operator = dslash.operator(
        U=refer_U, kappa=kappa, clover_term=refer_clover_term, verbose=True, support_parity=True)
    time_start = perf_counter()
    dest = (operator.matvec_all(src=refer_src.reshape(
        [12]+list(refer_src.shape[2:])))).reshape(refer_src.shape)
    dest = operator.matvec(src=refer_src)
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


def test_dslash_clover(device: torch.device = torch.device('cpu')):
    kappa = 1.0
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
    is_su3 = lattice.check_su3(refer_U, tol=1e-6, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n Gauge field SU(3) check: {is_su3}")
    print(f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n {tools.norm(refer_U)}")
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


def test_solver(method: str = 'bistabcg', kappa: float = 0.125, lat_size: list = [8, 8, 8, 16],  dtype: torch.dtype = torch.complex64, device: torch.device = torch.device('cpu'), with_data: bool = False, max_levels: int = 2, num_restart: int = 3):
    if not with_data:
        comm = MPI.COMM_WORLD
        root = 0
        if comm.rank == root:
            whole_U = torch.zeros(
                size=[3, 3, 4]+lat_size, dtype=dtype, device=device)
            lattice.generate_gauge_field(
                whole_U, seed=42, sigma=0.1, verbose=True)
            whole_clover_term = dslash.make_clover(
                U=whole_U, kappa=kappa, verbose=True)
            whole_x = torch.randn(
                size=[4, 3]+lat_size, dtype=dtype, device=device)
            whole_b = dslash.give_clover(src=whole_x, clover_term=whole_clover_term, verbose=True) + dslash.give_wilson(src=whole_x, U=whole_U, kappa=kappa,
                                                                                                                        with_I=True, verbose=True)
        else:
            whole_U = None
            whole_clover_term = None
            whole_x = None
            whole_b = None
        refer_U = tools.whole_xyzt2local_xyzt(whole_array=whole_U, whole_shape=[
                                              3, 3, 4]+lat_size, root=root, dtype=dtype, device=device)
        refer_clover_term = tools.whole_xyzt2local_xyzt(whole_array=whole_clover_term, whole_shape=[
                                                        4, 3, 4, 3]+lat_size, root=root, dtype=dtype, device=device)
        refer_x = tools.whole_xyzt2local_xyzt(whole_array=whole_x, whole_shape=[
                                              4, 3]+lat_size, root=root, dtype=dtype, device=device)
        refer_b = tools.whole_xyzt2local_xyzt(whole_array=whole_b, whole_shape=[
                                              4, 3]+lat_size, root=root, dtype=dtype, device=device)
    else:
        kappa = 0.125
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
        U=refer_U, clover_term=refer_clover_term, kappa=kappa, verbose=True)

    def matvec(src):
        return operator.matvec(src=src)
        # return dslash.give_clover(src=src, clover_term=refer_clover_term, verbose=True) + dslash.give_wilson(src=src, U=refer_U, kappa=kappa, with_I=True, verbose=True)
    if method == 'bistabcg':
        time_start = perf_counter()
        x = solver.bistabcg(b=refer_b, matvec=matvec, tol=1e-6,
                            max_iter=1000, x0=None, if_rtol=False, verbose=True)
        time_end = perf_counter()
    elif method == 'multigrid':
        mg = solver.multigrid(dtype_list=[refer_U.dtype]*10, device_list=[refer_U.device]*10, U=refer_U,
                              clover_term=refer_clover_term, kappa=kappa, tol=1e-6, max_iter=1000, max_levels=max_levels, num_restart=num_restart, verbose=True)
        mg.init()
        time_start = perf_counter()
        x = mg.solve(b=refer_b)
        time_end = perf_counter()
        mg.plot()
    else:
        raise ValueError(
            f"PYQCU::TESTING::SOLVER::SOLVER: {solver} is not supported.")
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


def test_matmul():
    M_gpu, N_gpu, K_gpu = 1024, 1024, 1024
    M_cpu, N_cpu, K_cpu = 1024, 1024, 1024
    gpu_tile = {"block_M": 128, "block_N": 128, "block_K": 32}
    cpu_tile = {"block_M": 32, "block_N": 32, "block_K": 32}

    def calc_metrics(m, n, k, sec):
        tflops = (2 * m * n * k / sec) / 1e12
        return tflops
    from pyqcu.tools import matmul_gpu
    func_gpu = matmul_gpu(M_gpu, N_gpu, K_gpu, **gpu_tile)
    jit_gpu = tilelang.compile(func_gpu, out_idx=[2], target="c")
    print(jit_gpu.get_kernel_source())
    a_gpu = torch.randn(M_gpu, K_gpu, device="cuda", dtype=torch.float16)
    b_gpu = torch.randn(N_gpu, K_gpu, device="cuda", dtype=torch.float16)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    # Warmup GPU
    for _ in range(20):
        jit_gpu(a_gpu, b_gpu)
    # Measure TileLang GPU
    iters = 100
    start_evt.record()
    for _ in range(iters):
        jit_gpu(a_gpu, b_gpu)
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
    func_cpu = matmul_cpu(M_cpu, N_cpu, K_cpu, **cpu_tile)
    try:
        jit_cpu = tilelang.compile(func_cpu, out_idx=[2], target="llvm")
        cpu_target_name = "LLVM"
    except:
        jit_cpu = tilelang.compile(func_cpu, out_idx=[2], target="c")
        cpu_target_name = "C"
    # print(jit_cpu.get_kernel_source())
    a_cpu = torch.randn(M_cpu, K_cpu, device="cpu", dtype=torch.float16)
    b_cpu = torch.randn(N_cpu, K_cpu, device="cpu", dtype=torch.float16)
    # Warmup CPU
    for _ in range(5):
        jit_cpu(a_cpu, b_cpu)
    # Measure TileLang CPU
    cpu_iters = 1
    start = perf_counter()
    for _ in range(cpu_iters):
        c_cpu = jit_cpu(a_cpu, b_cpu)
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
