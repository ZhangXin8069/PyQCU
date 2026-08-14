#!/usr/bin/env python3
"""PyQCU C++ CUDA Clover Multigrid — Multi-Configuration Test Suite

Tests the C++ MG solver across multiple lattice sizes, precisions, and MG parameters.
Compares against Python MG reference to establish expected convergence behavior.
All output is saved to logs/ directory.

Usage:
    mpirun --allow-run-as-root -np 1 python examples/qcu/conftest.clover.multigrid.py
"""
import torch, os, json, re, sys, copy, math
from datetime import datetime
from time import perf_counter
import numpy as np
from pyqcu import tools, dslash, solver
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

LOG_DIR = os.path.expanduser("~/PyQCU/logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ---- Configuration matrix ----
CONFIGS = [
    # (label, Lx,Ly,Lz,Lt, mass, atol, num_levels, dof_list, mg_grid, restart, coarse_max_iter, coarse_tol_factor)
    ("8x8x8x16_c64_m0.05_1L",  8, 8, 8, 16, 0.05, 1e-6, 1, [12],       [2,2,2,1], 5,  50,  10.0),
    ("8x8x8x16_c64_m0.05_2L",  8, 8, 8, 16, 0.05, 1e-6, 2, [12,48],    [2,2,2,1], 5,  50,  10.0),
    ("8x8x8x16_c64_m0.05_2L_r3",8, 8, 8, 16, 0.05, 1e-6, 2, [12,48],   [2,2,2,1], 3,  30,  10.0),
    ("12x12x12x16_c64_m0.05_2L",12,12,12,16,0.05, 1e-6, 2, [12,48],    [2,2,2,1], 5,  80,  10.0),
    ("16x16x16x16_c64_m0.05_2L",16,16,16,16,0.05,1e-6, 2, [12,48],     [2,2,2,1], 5, 100,  10.0),
    ("8x8x8x16_c64_m0.10_2L",  8, 8, 8, 16, 0.10, 1e-6, 2, [12,12],    [2,2,2,1], 5,  50,  10.0),
]

def log(msg, fn="clover_multigrid_test.log"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    print(msg, flush=True)
    with open(os.path.join(LOG_DIR, fn), "a") as f:
        f.write(line + "\n")

# Clear logs
for f in ["clover_multigrid_test.log", "clover_multigrid.log"]:
    open(os.path.join(LOG_DIR, f), "w").close()

log("=" * 80)
log("PyQCU C++ CUDA Clover Multigrid — Multi-Configuration Test Suite")
log(f"  Configs: {len(CONFIGS)}")
log(f"  TIMESTAMP: {datetime.now().isoformat()}")
log("=" * 80)

results = []

for ci, (label, Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID, NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR) in enumerate(CONFIGS):
    log(f"\n{'='*60}")
    log(f"[{ci+1}/{len(CONFIGS)}] Config: {label}")
    log(f"  Lattice: {Lx}×{Ly}×{Lz}×{Lt}, MASS={MASS}, ATOL={ATOL:.0e}, Levels={NUM_LEVELS}")
    log(f"  DOF: {DOF_LIST}, MG_GRID: {MG_GRID}, Restart: {NUM_RESTART}, CoarseIter: {COARSE_MAX_ITER}")
    log(f"  Total fine sites: {Lx*Ly*Lz*Lt}")

    KAPPA = 1.0/(2*MASS+8)

    # ---- Reset params ----
    params[define._LAT_X_] = Lx; params[define._LAT_Y_] = Ly
    params[define._LAT_Z_] = Lz; params[define._LAT_T_] = Lt
    params[define._LAT_XYZT_] = Lx*Ly*Lz*Lt
    params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[define._GRID_T_] = tools.give_grid_size()
    params[define._PARITY_] = 0; params[define._NODE_RANK_] = 0; params[define._NODE_SIZE_] = 1
    params[define._DAGGER_] = 0; params[define._MAX_ITER_] = 500
    params[define._DATA_TYPE_] = define._LAT_C64_
    params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 1; params[define._SEED_] = 42; params[define._TEST_IN_CPU_] = 0

    params[define._MG_NUM_LEVEL_] = NUM_LEVELS
    if NUM_LEVELS >= 2:
        level1_T = Lt // (2 * MG_GRID[3])  # SCHUR mode: odd-lattice T/2 再粗化 2 = T_full/4（对齐 C++ parse_params）
        params[define._MG_LEVEL1_E_] = DOF_LIST[1]
        params[define._MG_LEVEL1_X_] = Lx // MG_GRID[0]
        params[define._MG_LEVEL1_Y_] = Ly // MG_GRID[1]
        params[define._MG_LEVEL1_Z_] = Lz // MG_GRID[2]
        params[define._MG_LEVEL1_T_] = level1_T
        params[define._MG_LEVEL1_MAX_ITER_] = COARSE_MAX_ITER
        params[define._MG_LEVEL1_DATA_TYPE_] = define._LAT_C64_
        params[define._MG_LEVEL1_NUM_RESTART_] = 3  # coarse restart (not used for 2L currently)

    if NUM_LEVELS >= 3:
        params[define._MG_LEVEL2_E_] = DOF_LIST[2] if len(DOF_LIST) > 2 else 24
        params[define._MG_LEVEL2_X_] = Lx // (MG_GRID[0] * MG_GRID[0])
        params[define._MG_LEVEL2_Y_] = Ly // (MG_GRID[1] * MG_GRID[1])
        params[define._MG_LEVEL2_Z_] = Lz // (MG_GRID[2] * MG_GRID[2])
        params[define._MG_LEVEL2_T_] = Lt // (MG_GRID[3] * MG_GRID[3])
        params[define._MG_LEVEL2_MAX_ITER_] = 200
        params[define._MG_LEVEL2_DATA_TYPE_] = define._LAT_C64_
        params[define._MG_LEVEL2_NUM_RESTART_] = 3

    argv_new = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
    argv_new[define._MASS_] = MASS; argv_new[define._ATOL_] = ATOL; argv_new[define._SIGMA_] = 0.1
    if NUM_LEVELS >= 2:
        argv_new[define._MG_LEVEL1_ATOL_] = ATOL * COARSE_TOL_FACTOR
    if NUM_LEVELS >= 3:
        argv_new[define._MG_LEVEL2_ATOL_] = ATOL * 0.1

    device = torch.device('cuda')
    dtype_t = define.dtype(params[define._DATA_TYPE_])
    lat_shape = define.lat_shape(params)

    # ---- Phase 1: Setup ----
    log("  [1/5] Setup gauge + clover...")
    g = torch.zeros([2,3,3,4]+lat_shape, dtype=dtype_t, device=device)
    fi = torch.randn([2,4,3]+lat_shape, dtype=dtype_t, device=device)
    fo_ref = torch.zeros_like(fi); fo_mg = torch.zeros_like(fi)
    ce = torch.zeros([4,3,4,3]+lat_shape, dtype=dtype_t, device=device)
    cei = torch.zeros_like(ce); coo = torch.zeros_like(ce); coi = torch.zeros_like(ce)

    params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, argv_new)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)

    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 2; params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, argv_new)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)

    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 2; params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, argv_new)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    # ---- Phase 2: Reference BiStabCG ----
    log("  [2/5] Reference BiStabCG...")
    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 1; params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, argv_new)
    t0 = perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs, params)
    ref_time = perf_counter() - t0

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)
    log(f"    BiStabCG: {ref_time:.4f}s, residual={ref_res:.4e}")

    # ---- Phase 3: Build coarse operators ----
    if NUM_LEVELS >= 2:
        log("  [3/5] Building coarse-grid operators...")
        U_full = qcu_U
        op_fine = dslash.operator(U=U_full, clover_term=ref_cl, kappa=KAPPA,
                                   support_parity=True, verbose=False)
        # 粗算子构建需要 Schur 算子（奇数格输入）
        S_build = op_fine.matvec_parity if hasattr(op_fine, 'matvec_parity') else op_fine.matvec

        lat_sizes = [[Lx, Ly, Lz, Lt]]
        for i in range(1, NUM_LEVELS):
            lat_sizes.append([max(lat_sizes[i-1][d] // MG_GRID[d], 1) for d in range(4)])

        from pyqcu.cuda._multi_gpu import build_schur_levels as _bsl
        _lat_full = [Lx, Ly, Lz, Lt]
        lonv_list, hop_nn_l, hop_diag_l, sit_l = _bsl(
            op_fine, S_build,
            NUM_LEVELS, [12] + DOF_LIST[1:], MG_GRID, _lat_full, DOF_LIST[1],
            dtype_t, device, nv_iters=1, use_cache=False, cache_dir=None, verbose=False)

        log(f"    Coarse ops: {len(lonv_list)} level(s) built")

        # Wire into set_ptrs (新协议: base=30, 4 槽/层: lonv/hop_nn/hop_diag/sit)
        for fl in range(len(lonv_list)):
            set_ptrs[30 + 4*fl + 0] = lonv_list[fl].contiguous().data_ptr()
            set_ptrs[30 + 4*fl + 1] = hop_nn_l[fl].contiguous().data_ptr()
            set_ptrs[30 + 4*fl + 2] = hop_diag_l[fl].contiguous().data_ptr()
            set_ptrs[30 + 4*fl + 3] = sit_l[fl].contiguous().data_ptr()

    # ---- Phase 4: C++ MG solver ----
    log("  [4/5] C++ MG solver...")
    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 1; params[define._VERBOSE_] = 1
    qcu.applyInitQcu(set_ptrs, params, argv_new)
    t0 = perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi, set_ptrs, params)
    mg_time = perf_counter() - t0

    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, KAPPA, True) +
                        dslash.give_clover(qcu_mg, ref_cl) - qcu_src) / tools.norm(qcu_src)
    mg_vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
    speedup = ref_time/mg_time if mg_time > 0 else 0

    # Parse convergence
    conv = []
    log_path = os.path.join(LOG_DIR, "clover_multigrid.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
                if m:
                    conv = [float(x) for x in m.group(1).split(",") if x.strip()]
                    break

    status = "PASS" if mg_vs_ref < 1e-5 else ("OK" if mg_vs_ref < 1e-2 else "FAIL")
    log(f"    MG: {mg_time:.4f}s, residual={mg_res:.4e}, vs_ref={mg_vs_ref:.2e}, speedup={speedup:.2f}x")
    log(f"    Status: {status}, Conv pts: {len(conv)}")

    result = {
        "label": label, "lattice": [Lx,Ly,Lz,Lt], "mass": MASS,
        "num_levels": NUM_LEVELS, "dof_list": DOF_LIST,
        "ref_time": ref_time, "mg_time": mg_time,
        "ref_residual": float(ref_res), "mg_residual": float(mg_res),
        "mg_vs_ref": float(mg_vs_ref), "speedup": speedup,
        "convergence": [float(c) for c in conv[:500]],
        "status": status, "mg_iterations": len([c for c in conv if c > ATOL]),
    }
    results.append(result)

    # Save per-config convergence
    if conv:
        np.savetxt(os.path.join(LOG_DIR, f"conv_{label}.txt"), np.array(conv[:200]))

# ---- Phase 5: Summary ----
log(f"\n{'='*80}")
log("SUMMARY TABLE")
log(f"{'Config':<35} {'Ref(s)':>8} {'MG(s)':>8} {'Speedup':>8} {'MG_Res':>10} {'vs_Ref':>10} {'Status':>6}")
log("-" * 90)
for r in results:
    log(f"{r['label']:<35} {r['ref_time']:8.4f} {r['mg_time']:8.4f} {r['speedup']:8.2f} {r['mg_residual']:10.2e} {r['mg_vs_ref']:10.2e} {r['status']:>6}")

# Save JSON
with open(os.path.join(LOG_DIR, "multigrid_report.json"), "w") as f:
    json.dump({"results": results, "timestamp": datetime.now().isoformat()}, f, indent=2)

log(f"\nReport saved to {LOG_DIR}/multigrid_report.json")

# Charts
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if rows == 1 and cols == 1: axes = [[axes]]
    elif rows == 1: axes = [axes]
    elif cols == 1: axes = [[a] for a in axes]
    for idx, r in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        conv_data = [c for c in r['convergence'] if c > 0 and c < 1e6]
        if conv_data:
            ax.semilogy(conv_data, 'b-', linewidth=1)
            ax.axhline(y=ATOL, color='gray', linestyle='--', alpha=0.5, label=f'tol={ATOL:.0e}')
        ax.set_title(f"{r['label']}\nspeedup={r['speedup']:.2f}x, {r['status']}")
        ax.set_xlabel('Record'); ax.set_ylabel('Residual'); ax.grid(True, alpha=0.3)
    # Hide unused axes
    for idx in range(n, rows*cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "multigrid_result_all.png"), dpi=150)
    plt.close()
    log(f"Chart saved to {LOG_DIR}/multigrid_result_all.png")
except Exception as e:
    log(f"Chart error: {e}")

log(f"\n{'='*80}")
log("TEST SUITE COMPLETE")
log(f"{'='*80}")
