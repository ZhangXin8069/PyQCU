#!/usr/bin/env python3
"""Quick single-point MG test"""
import os
import sys
from time import perf_counter

import torch

from pyqcu import tools, dslash
from pyqcu.cuda import qcu
from pyqcu.cuda._multi_gpu import build_schur_levels
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs


def main():
    log_dir = os.path.expanduser("~/PyQCU/logs/dev73")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "quick_test.log")
    log_file = open(log_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    Lx, Ly, Lz, Lt = 8, 8, 8, 16
    MASS = 0.05
    ATOL = 1e-6
    KAPPA = 1.0 / (2 * MASS + 8)

    print("=== Quick MG Test ===")

    for NUM_LEVELS in [1, 2]:
        print(f"\n--- NUM_LEVELS={NUM_LEVELS} ---")

        params[define._LAT_X_] = Lx
        params[define._LAT_Y_] = Ly
        params[define._LAT_Z_] = Lz
        params[define._LAT_T_] = Lt
        params[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
        params[define._GRID_X_], params[define._GRID_Y_], \
            params[define._GRID_Z_], params[define._GRID_T_] = tools.give_grid_size()
        params[define._PARITY_] = 0
        params[define._NODE_RANK_] = 0
        params[define._NODE_SIZE_] = 1
        params[define._DAGGER_] = 0
        params[define._MAX_ITER_] = 500
        params[define._DATA_TYPE_] = define._LAT_C64_
        params[define._SET_INDEX_] = 0
        params[define._SET_PLAN_] = 1
        params[define._VERBOSE_] = 1
        params[define._SEED_] = 42
        params[define._TEST_IN_CPU_] = 0
        params[define._MG_NUM_LEVEL_] = NUM_LEVELS
        if NUM_LEVELS >= 2:
            params[define._MG_LEVEL1_E_] = 48
            params[define._MG_LEVEL1_X_] = 4
            params[define._MG_LEVEL1_Y_] = 4
            params[define._MG_LEVEL1_Z_] = 4
            params[define._MG_LEVEL1_T_] = 8
            params[define._MG_LEVEL1_MAX_ITER_] = 200
            params[define._MG_LEVEL1_DATA_TYPE_] = define._LAT_C64_
            params[define._MG_LEVEL1_NUM_RESTART_] = 5

        av = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
        av[define._MASS_] = MASS
        av[define._ATOL_] = ATOL
        av[define._SIGMA_] = 0.1
        if NUM_LEVELS >= 2:
            av[define._MG_LEVEL1_ATOL_] = ATOL * 3000.0

        device = torch.device("cuda")
        dt = define.dtype(params[define._DATA_TYPE_])
        ls = define.lat_shape(params)
        g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
        fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
        fo_ref = torch.zeros_like(fi)
        fo_mg = torch.zeros_like(fi)
        ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
        cei = torch.zeros_like(ce)
        coo = torch.zeros_like(ce)
        coi = torch.zeros_like(ce)

        print("Setup gauge+clover...")
        params[define._SET_INDEX_] = 0
        params[define._SET_PLAN_] = -1
        qcu.applyInitQcu(set_ptrs, params, av)
        qcu.applyGaussGaugeQcu(g, set_ptrs, params)

        params[define._SET_INDEX_] += 1
        params[define._SET_PLAN_] = 2
        params[define._PARITY_] = 0
        qcu.applyInitQcu(set_ptrs, params, av)
        qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)

        params[define._SET_INDEX_] += 1
        params[define._SET_PLAN_] = 2
        params[define._PARITY_] = 1
        qcu.applyInitQcu(set_ptrs, params, av)
        qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

        qcu_U = tools.poooxyzt2oooxyzt(g)
        ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)

        # Build coarse ops for the modern SCHUR protocol.
        if NUM_LEVELS >= 2:
            print("Build coarse operators...")
            op_fine = dslash.operator(
                U=qcu_U,
                clover_term=ref_cl,
                kappa=torch.Tensor([KAPPA]),
                support_parity=True,
                verbose=False,
            )
            S_build = op_fine.matvec_parity if hasattr(op_fine, "matvec_parity") else op_fine.matvec
            lonv_list, hop_nn_l, hop_diag_l, sit_l = build_schur_levels(
                op_fine, S_build, NUM_LEVELS, [12, 48], [2, 2, 2, 1],
                [Lx, Ly, Lz, Lt], 48, dt, device,
                nv_iters=20, use_cache=True, cache_dir=None, verbose=True)
            for fl in range(len(lonv_list)):
                set_ptrs[30 + 4 * fl + 0] = lonv_list[fl].contiguous().data_ptr()
                set_ptrs[30 + 4 * fl + 1] = hop_nn_l[fl].contiguous().data_ptr()
                set_ptrs[30 + 4 * fl + 2] = hop_diag_l[fl].contiguous().data_ptr()
                set_ptrs[30 + 4 * fl + 3] = sit_l[fl].contiguous().data_ptr()
                print(
                    f"Coarse ops set[{fl}]: "
                    f"lonv={set_ptrs[30 + 4 * fl + 0]:#x} "
                    f"hop_nn={set_ptrs[30 + 4 * fl + 1]:#x} "
                    f"hop_diag={set_ptrs[30 + 4 * fl + 2]:#x} "
                    f"sit={set_ptrs[30 + 4 * fl + 3]:#x}"
                )

        # Ref BiStabCG
        print("Ref BiStabCG...")
        params[define._SET_INDEX_] += 1
        params[define._SET_PLAN_] = 1
        params[define._VERBOSE_] = 0
        qcu.applyInitQcu(set_ptrs, params, av)
        t0 = perf_counter()
        qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs, params)
        ref_time = perf_counter() - t0
        print(f"  Ref: {ref_time:.4f}s")

        # MG
        print("MG solver...")
        params[define._SET_INDEX_] += 1
        params[define._SET_PLAN_] = 1
        params[define._VERBOSE_] = 1
        qcu.applyInitQcu(set_ptrs, params, av)
        t0 = perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi, set_ptrs, params)
        mg_time = perf_counter() - t0

        qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
        qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
        qcu_src = tools.poooxyzt2oooxyzt(fi)
        mg_vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
        speedup = ref_time / mg_time if mg_time > 0 else 0
        print(
            f"RESULT_{NUM_LEVELS}L: ref={ref_time:.4f}s "
            f"mg={mg_time:.4f}s vs_ref={mg_vs_ref:.2e} "
            f"speedup={speedup:.2f}x"
        )

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
