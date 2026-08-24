import sys
import traceback
import torch
import pyqcu.testing as T

CUDA = torch.device('cuda')
results = []


def run(name, fn):
    t0 = __import__('time').time()
    try:
        fn()
        dt = __import__('time').time() - t0
        results.append((name, 'PASS', '', dt))
        print(f"\n[R1-BASELINE][PASS] {name} ({dt:.1f}s)", flush=True)
    except Exception as e:
        dt = __import__('time').time() - t0
        results.append((name, 'FAIL', f"{type(e).__name__}: {e}", dt))
        print(f"\n[R1-BASELINE][FAIL] {name} ({dt:.1f}s): {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


run("lattice_cuda", lambda: T.test_lattice(lat_size=[8, 8, 8, 16], device=CUDA))
run("dslash_wilson_data_par", lambda: T.test_dslash_wilson(
    with_data=True, support_parallel=True, device=CUDA))
run("dslash_wilson_data_nopar", lambda: T.test_dslash_wilson(
    with_data=True, support_parallel=False, device=CUDA))
run("dslash_parity_cuda", lambda: T.test_dslash_parity(device=CUDA))
run("dslash_clover_cuda", lambda: T.test_dslash_clover(device=CUDA))
run("solver_bistabcg_8888_c128", lambda: T.test_solver(
    method='bistabcg', lat_size=[8, 8, 8, 8], dtype=torch.complex128, device=CUDA))
run("solver_mg_88816_L2_c128", lambda: T.test_solver(
    method='multigrid', lat_size=[8, 8, 8, 16], dtype=torch.complex128,
    max_level=2, device=CUDA))
run("smear_stout_cuda", lambda: T.test_smear_stout(lat_size=[8, 8, 8, 16], device=CUDA))
run("h5py_mt4", lambda: T.test_h5py_multithread(nthreads=4, dtype=torch.complex64,
                                                lat_size=[4, 4, 4, 8]))
run("mg_multithread_gate", lambda: T.test_multigrid_multithread(
    nthreads=2, lat_size=[8, 8, 8, 8], mass=0.05, tol=1e-5))

n_fail = sum(1 for r in results if r[1] == 'FAIL')
print(f"\n=== R1 BASELINE SUMMARY: {len(results)-n_fail}/{len(results)} PASS, {n_fail} FAIL ===")
for name, st, msg, dt in results:
    print(f"  [{st}] {name} ({dt:.1f}s) {msg}")
sys.exit(1 if n_fail else 0)
