import numpy as np
from time import perf_counter
import tilelang
import torch
from argparse import Namespace
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
        print("PYQCU::TESTING::IMPORT:\n All dependencies imported successfully.")
    except Exception as e:
        print(f"PYQCU::TESTING::IMPORT:\n {e}")


def test_lattice():
    import torch
    from pyqcu.lattice import I, gamma, gamma_5, gamma_gamma, gell_mann
    print("PYQCU::TESTING::LATTICE:\n imported successfully.")
    print("PYQCU::TESTING::LATTICE::I:\n ", I)
    print("PYQCU::TESTING::LATTICE::GAMMA:\n ", gamma)
    print("PYQCU::TESTING::LATTICE::GAMMA_5:\n ", gamma_5)
    print("PYQCU::TESTING::LATTICE::GAMMA_GAMMA:\n ", gamma_gamma)
    print("PYQCU::TESTING::LATTICE::GELL_MANN:\n ", gell_mann)
    U = torch.zeros(3, 3, 4, 4, 4, 4, 4, dtype=torch.complex64, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    from pyqcu.lattice import generate_gauge_field
    generate_gauge_field(U, seed=42, sigma=0.1, verbose=True)
    from pyqcu.lattice import check_su3
    is_su3 = check_su3(U, tol=1e-6, verbose=True)
    print(f"PYQCU::TESTING::LATTICE:\n Gauge field SU(3) check: {is_su3}")


def test_dslash_wilson():
    import torch
    import pyqcu
    from pyqcu.dslash import give_wilson, give_wilson_eo, give_wilson_oe, give_wilson_eoeo
    from pyqcu.tools import ___xyzt2p___xyzt, p___xyzt2___xyzt, hdf5___2___
    from pyqcu.lattice import check_su3
    from pyqcu import _torch
    print("PYQCU::TESTING::DSLASH::WILSON:\n imported successfully.")
    U = torch.zeros(3, 3, 4, 4, 4, 4, 4, dtype=torch.complex64, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    from pyqcu.lattice import generate_gauge_field
    generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
    print(f"PYQCU::TESTING::DSLASH::WILSON::U:\n {_torch.norm(U)}")
    src = _torch.randn(size=(
        4, 3, U.shape[-4], U.shape[-3], U.shape[-2], U.shape[-1]), dtype=U.dtype, device=U.device)
    print(f"PYQCU::TESTING::DSLASH::WILSON::SRC:\n {_torch.norm(src)}")
    dest = give_wilson(src=src, U=U, kappa=0.1, verbose=True)
    print(f"PYQCU::TESTING::DSLASH::WILSON::DEST:\n {_torch.norm(dest)}")
    U_eo = ___xyzt2p___xyzt(input_array=U, verbose=True)
    src_eo = ___xyzt2p___xyzt(input_array=src, verbose=True)
    src_e = src_eo[0]
    src_o = src_eo[1]
    dest_e = give_wilson_eo(src_o=src_o, U_eo=U_eo, kappa=0.1, verbose=True)
    dest_o = give_wilson_oe(src_e=src_e, U_eo=U_eo, kappa=0.1, verbose=True)
    dest_eo = torch.zeros_like(src_eo)
    dest_eo[0] = dest_e
    dest_eo[1] = dest_o
    dest_reconstructed = p___xyzt2___xyzt(input_array=give_wilson_eoeo(
        dest_eo=dest_eo, src_eo=src_eo), verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::DEST_RECONSTRUCTED:\n {_torch.norm(dest_reconstructed)}")
    diff = _torch.norm(dest_reconstructed-dest)/_torch.norm(dest)
    print(
        f"PYQCU::TESTING::DSLASH::WILSON:\n Difference between full and eo/oe dslash: {diff}")
    path = pyqcu.__file__.replace('pyqcu/__init__.py', 'examples/data/')
    refer_U = hdf5___2___(
        file_name=path+'refer.wilson.U.L32K0_125.ccdxyzt.c64.h5', device=src.device, verbose=True)
    is_su3 = check_su3(refer_U, tol=1e-6, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_U:\n Gauge field SU(3) check: {is_su3}")
    print(f"PYQCU::TESTING::DSLASH::WILSON::REFER_U:\n {_torch.norm(refer_U)}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_U:\n {refer_U.flatten()[:12]}")
    refer_src = hdf5___2___(
        file_name=path+'refer.wilson.src.L32K0_125.scxyzt.c64.h5', device=src.device, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_SRC:\n {_torch.norm(refer_src)}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_SRC:\n {refer_src.flatten()[:12]}")
    refer_dest = hdf5___2___(
        file_name=path+'refer.wilson.dest.L32K0_125.scxyzt.c64.h5', device=src.device, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_DEST:\n {_torch.norm(refer_dest)}")
    print(
        f"PYQCU::TESTING::DSLASH::WILSON::REFER_DEST:\n {refer_dest.flatten()[:12]}")
    dest = give_wilson(src=refer_src, U=refer_U, kappa=0.125, verbose=True)
    print(f"PYQCU::TESTING::DSLASH::WILSON::DEST:\n {_torch.norm(dest)}")
    print(f"PYQCU::TESTING::DSLASH::WILSON::DEST:\n {dest.flatten()[:12]}")
    diff = _torch.norm(dest - refer_dest)/_torch.norm(refer_dest)
    print(
        f"PYQCU::TESTING::DSLASH::WILSON:\n Difference between computed and reference dslash: {diff}")


def test_dslash_clover():
    import torch
    import pyqcu
    from pyqcu.dslash import make_clover, inverse, add_I
    from pyqcu.tools import hdf5___2___
    from pyqcu.lattice import check_su3
    from pyqcu import _torch
    path = pyqcu.__file__.replace('pyqcu/__init__.py', 'examples/data/')
    refer_U = hdf5___2___(
        file_name=path+'refer.clover.U.L32Y16K1.ccdxyzt.c64.h5', device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'), verbose=True)
    is_su3 = check_su3(refer_U, tol=1e-6, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n Gauge field SU(3) check: {is_su3}")
    print(f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n {_torch.norm(refer_U)}")
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::REFER_U:\n {refer_U.flatten()[:12]}")
    refer_clover_term = hdf5___2___(
        file_name=path+'refer.clover.clover_term.L32Y16K1.scscxyzt.c64.h5', device=refer_U.device, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER_TERM:\n {_torch.norm(refer_clover_term)}")
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER_TERM:\n {refer_clover_term.flatten()[:12]}")
    refer_clover_inv_term = hdf5___2___(
        file_name=path+'refer.clover.clover_inv_term.L32Y16K1.scscxyzt.c64.h5', device=refer_U.device, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER_INV_TERM:\n {_torch.norm(refer_clover_inv_term)}")
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::REFER_CLOVER_INV_TERM:\n {refer_clover_inv_term.flatten()[:12]}")
    clover_term = make_clover(U=refer_U, kappa=1, verbose=True)
    clover_term = add_I(clover_term=clover_term, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER_TERM:\n {_torch.norm(clover_term)}")
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER_TERM:\n {clover_term.flatten()[:12]}")
    diff = _torch.norm(clover_term - refer_clover_term) / \
        _torch.norm(refer_clover_term)
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER:\n Difference between computed and reference dslash: {diff}")
    clover_inv_term = inverse(clover_term=clover_term, verbose=True)
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER_INV_TERM:\n {_torch.norm(clover_inv_term)}")
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER::CLOVER_INV_TERM:\n {clover_inv_term.flatten()[:12]}")
    diff = _torch.norm(clover_inv_term - refer_clover_inv_term) / \
        _torch.norm(refer_clover_inv_term)
    print(
        f"PYQCU::TESTING::DSLASH::CLOVER:\n Difference between computed and reference dslash: {diff}")


def test_solver_bistabcg():
    import torch
    import pyqcu
    from pyqcu.dslash import give_wilson
    from pyqcu.solver import bicgstab
    from pyqcu.tools import hdf5___2___
    from pyqcu import _torch
    path = pyqcu.__file__.replace('pyqcu/__init__.py', 'examples/data/')
    refer_U = hdf5___2___(
        file_name=path+'refer.wilson.U.L32K0_125.ccdxyzt.c64.h5', device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'), verbose=True)
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG::REFER_U:\n {_torch.norm(refer_U)}")
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG::REFER_U:\n {refer_U.flatten()[:12]}")
    refer_b = hdf5___2___(
        file_name=path+'refer.wilson.b.L32K0_125.scxyzt.c64.h5', device=refer_U.device, verbose=True)
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG::REFER_B:\n {_torch.norm(refer_b)}")
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG::REFER_B:\n {refer_b.flatten()[:12]}")
    refer_x = hdf5___2___(
        file_name=path+'refer.wilson.x.L32K0_125.scxyzt.c64.h5', device=refer_U.device, verbose=True)
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG::REFER_X:\n {_torch.norm(refer_x)}")
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG::REFER_X:\n {refer_x.flatten()[:12]}")

    def matvec(v):
        return give_wilson(src=v, U=refer_U, kappa=0.125, verbose=False)
    x = bicgstab(b=refer_b, matvec=matvec, tol=1e-6, max_iter=1000,
                 x0=None, if_rtol=False, if_multi=False, verbose=True)
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG::X:\n {_torch.norm(x)}")
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG::X:\n {x.flatten()[:12]}")
    diff = _torch.norm(x - refer_x) / _torch.norm(refer_x)
    print(
        f"PYQCU::TESTING::SOLVER::BISTABCG:\n Difference between computed and reference solution: {diff}")


def test_solver_multigrid():
    import torch
    import pyqcu
    from pyqcu.dslash import give_clover, add_I
    from pyqcu.solver import multigrid
    from pyqcu.tools import hdf5___2___
    from pyqcu import _torch
    path = pyqcu.__file__.replace('pyqcu/__init__.py', 'examples/data/')
    refer_U = hdf5___2___(
        file_name=path+'refer.wilson.U.L32K0_125.ccdxyzt.c64.h5', device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'), verbose=True)
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID::REFER_U:\n {_torch.norm(refer_U)}")
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID::REFER_U:\n {refer_U.flatten()[:12]}")
    refer_b = hdf5___2___(
        file_name=path+'refer.wilson.b.L32K0_125.scxyzt.c64.h5', device=refer_U.device, verbose=True)
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID::REFER_B:\n {_torch.norm(refer_b)}")
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID::REFER_B:\n {refer_b.flatten()[:12]}")
    refer_x = hdf5___2___(
        file_name=path+'refer.wilson.x.L32K0_125.scxyzt.c64.h5', device=refer_U.device, verbose=True)
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID::REFER_X:\n {_torch.norm(refer_x)}")
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID::REFER_X:\n {refer_x.flatten()[:12]}")
    clover_term = torch.zeros(size=(4, 3, 4, 3, refer_U.shape[-4], refer_U.shape[-3],
                              refer_U.shape[-2], refer_U.shape[-1]), dtype=refer_U.dtype, device=refer_U.device)
    clover_term = add_I(clover_term=clover_term, verbose=True)
    mg = multigrid(dtype_list=[refer_U.dtype]*10, device_list=[refer_U.device]*10, U=refer_U,
                   clover_term=clover_term, kappa=0.125, tol=1e-6, max_iter=1000, max_levels=1, verbose=True)
    mg.init()
    x = mg.solve(b=refer_b)
    mg.plot()
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID::X:\n {_torch.norm(x)}")
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID::X:\n {x.flatten()[:12]}")
    diff = _torch.norm(x - refer_x) / _torch.norm(refer_x)
    print(
        f"PYQCU::TESTING::SOLVER::MULTIGRID:\n Difference between computed and reference solution: {diff}")


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
    jit_gpu = tilelang.compile(func_gpu, out_idx=[2], target="cuda")
    # print(jit_gpu.get_kernel_source())
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
