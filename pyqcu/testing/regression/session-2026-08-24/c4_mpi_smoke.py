import sys
import traceback
import torch
import pyqcu.testing as T

CUDA = torch.device('cuda')
results = []


def run(name, fn):
    try:
        fn()
        results.append((name, 'PASS', ''))
        print(f"\n[C4][PASS] {name}", flush=True)
    except Exception as e:
        results.append((name, 'FAIL', f"{type(e).__name__}: {e}"))
        print(f"\n[C4][FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


run("mpi_dslash_parity_8x8x8x16", lambda: T.test_dslash_parity(
    lat_size=[8, 8, 8, 16], device=CUDA))
run("mpi_smear_stout_8x8x8x16", lambda: T.test_smear_stout(
    lat_size=[8, 8, 8, 16], device=CUDA))
run("mpi_solver_bistabcg_c64", lambda: T.test_solver(
    method='bistabcg', lat_size=[8, 8, 8, 8], dtype=torch.complex64,
    device=CUDA))
run("mpi_lattice_8x8x8x16", lambda: T.test_lattice(
    lat_size=[8, 8, 8, 16], device=CUDA))

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
    print(f"\n=== C4 MPI SMOKE SUMMARY: {len(results)-n_fail}/{len(results)} PASS ===")
    for name, st, m in results:
        print(f"  [{st}] {name} {m}")
sys.exit(0)
