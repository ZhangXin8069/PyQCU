#!/usr/bin/env python3
"""
PyQCU Clover Multigrid CUDA Solver — Correctness & Performance Test

Compares the new C++ CUDA multigrid solver against the existing
BiStabCG solver on a Clover-preconditioned system.

Usage:
    python conftest.clover.multigrid.py
    mpirun -np 1 python conftest.clover.multigrid.py
"""
import torch
import os, sys, json
from datetime import datetime
from time import perf_counter
import numpy as np

from pyqcu import tools, dslash, lattice
from pyqcu.cuda import qcu, define
from pyqcu.cuda.define import params, argv, set_ptrs

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg, filename="test_multigrid.log"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {msg}"
    with open(os.path.join(LOG_DIR, filename), "a") as f:
        f.write(line + "\n")
    print(msg)

log("=" * 70)
log("PyQCU Clover Multigrid CUDA Solver — Test Start")
log("=" * 70)

# ============================================================
# Lattice & Grid Parameters
# ============================================================
Lx, Ly, Lz, Lt = 4*2, 4*2, 4*2, 8*2  # 8x8x8x16 for faster testing
params[define._LAT_X_] = Lx
params[define._LAT_Y_] = Ly
params[define._LAT_Z_] = Lz
params[define._LAT_T_] = Lt
params[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[define._GRID_T_] = tools.give_grid_size()
params[define._PARITY_] = 0
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DAGGER_] = 0
params[define._MAX_ITER_] = 200
params[define._DATA_TYPE_] = define._LAT_C64_
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = 1
params[define._VERBOSE_] = 0  # reduced verbosity
params[define._SEED_] = 42
params[define._TEST_IN_CPU_] = 0

# Multigrid params: single level (acts as standalone BiStabCG smoother)
params[define._MG_NUM_LEVEL_] = 1
params[define._MG_LEVEL1_E_] = 12
params[define._MG_LEVEL1_X_] = Lx // 2
params[define._MG_LEVEL1_Y_] = Ly // 2
params[define._MG_LEVEL1_Z_] = Lz // 2
params[define._MG_LEVEL1_T_] = Lt // 2
params[define._MG_LEVEL1_MAX_ITER_] = 50
params[define._MG_LEVEL1_DATA_TYPE_] = define._LAT_C64_
params[define._MG_LEVEL1_NUM_RESTART_] = 5

argv = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
MASS = 0.05
argv[define._MASS_] = MASS
argv[define._ATOL_] = 1e-6  # relaxed tolerance for faster convergence
argv[define._SIGMA_] = 0.1

log(f"Lattice: {Lx}x{Ly}x{Lz}x{Lt}")
log(f"Grid: {tools.give_grid_size()}")
log(f"Mass: {MASS}, kappa: {1/(2*MASS+8):.6f}")
log(f"Data type: complex64 (float)")
log(f"Num levels: {params[define._MG_NUM_LEVEL_].item()}")

# ============================================================
# Allocate tensors (parity-split layout, on GPU)
# ============================================================
device = torch.device('cuda')
dtype_t = define.dtype(params[define._DATA_TYPE_])
lat_shape = define.lat_shape(params)

gauge_eo = torch.zeros([2, 3, 3, 4] + lat_shape, dtype=dtype_t, device=device)
fermion_in_eo = torch.zeros([2, 4, 3] + lat_shape, dtype=dtype_t, device=device)
fermion_out_eo = torch.zeros([2, 4, 3] + lat_shape, dtype=dtype_t, device=device)

clover_ee = torch.zeros([4, 3, 4, 3] + lat_shape, dtype=dtype_t, device=device)
clover_ee_inv = torch.zeros([4, 3, 4, 3] + lat_shape, dtype=dtype_t, device=device)
clover_oo = torch.zeros([4, 3, 4, 3] + lat_shape, dtype=dtype_t, device=device)
clover_oo_inv = torch.zeros([4, 3, 4, 3] + lat_shape, dtype=dtype_t, device=device)

# ============================================================
# Step 1: Generate gauge field
# ============================================================
log("\n[Step 1] Generate Gaussian gauge field...")
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = -1  # Gauss gauge
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge_eo, set_ptrs, params)
log(f"  Gauge SU(3) check (even): {lattice.check_su3(U=gauge_eo[0])}")
log(f"  Gauge SU(3) check (odd):  {lattice.check_su3(U=gauge_eo[1])}")

# Generate random source
fermion_in_eo = torch.randn_like(fermion_in_eo)
log(f"  Source norm: {tools.norm(fermion_in_eo):.6e}")

# ============================================================
# Step 2: Build Clover term
# ============================================================
log("\n[Step 2] Build Clover term...")
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_ee, clover_ee_inv, gauge_eo, set_ptrs, params)

params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 1
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_oo, clover_oo_inv, gauge_eo, set_ptrs, params)
log(f"  Clover EE norm: {tools.norm(clover_ee):.6e}")
log(f"  Clover OO norm: {tools.norm(clover_oo):.6e}")

# ============================================================
# Step 3: Reference solution using BiStabCG
# ============================================================
log("\n[Step 3] Reference: BiStabCG solver...")
fermion_out_ref = torch.zeros_like(fermion_out_eo)
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
params[define._PARITY_] = 0
params[define._VERBOSE_] = 1

t0 = perf_counter()
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloverBistabCgQcu(fermion_out_ref, fermion_in_eo, gauge_eo,
                             clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
                             set_ptrs, params)
t1 = perf_counter()
ref_time = t1 - t0
log(f"  BiStabCG time: {ref_time:.6f} s")

# Verify reference
qcu_U = tools.poooxyzt2oooxyzt(input_array=gauge_eo)
qcu_src = tools.poooxyzt2oooxyzt(input_array=fermion_in_eo)
qcu_dest_ref = tools.poooxyzt2oooxyzt(input_array=fermion_out_ref)

refer_clover_term = dslash.make_clover(U=qcu_U, kappa=1/(2*MASS+8))
refer_src = dslash.give_wilson(src=qcu_dest_ref, U=qcu_U, kappa=1/(2*MASS+8), with_I=True) + \
            dslash.give_clover(src=qcu_dest_ref, clover_term=refer_clover_term)

ref_diff = tools.norm(refer_src - qcu_src) / tools.norm(qcu_src)
log(f"  BiStabCG relative residual: {ref_diff:.6e}")

# ============================================================
# Step 4: Multigrid solver
# ============================================================
log("\n[Step 4] Multigrid solver...")
fermion_out_mg = torch.zeros_like(fermion_out_eo)
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
params[define._VERBOSE_] = 0  # quiet mode for faster execution

t0 = perf_counter()
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloverMultigridQcu(fermion_out_mg, fermion_in_eo, gauge_eo,
                              clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
                              set_ptrs, params)
t1 = perf_counter()
mg_time = t1 - t0
log(f"  Multigrid time: {mg_time:.6f} s")

# Verify multigrid result
qcu_dest_mg = tools.poooxyzt2oooxyzt(input_array=fermion_out_mg)
mg_src = dslash.give_wilson(src=qcu_dest_mg, U=qcu_U, kappa=1/(2*MASS+8), with_I=True) + \
         dslash.give_clover(src=qcu_dest_mg, clover_term=refer_clover_term)
mg_diff = tools.norm(mg_src - qcu_src) / tools.norm(qcu_src)
log(f"  Multigrid relative residual: {mg_diff:.6e}")

# Compare multigrid vs reference
mg_vs_ref = tools.norm(qcu_dest_mg - qcu_dest_ref) / tools.norm(qcu_dest_ref)
log(f"  |x_mg - x_ref| / |x_ref|: {mg_vs_ref:.6e}")

# ============================================================
# Step 5: Performance comparison (multiple runs)
# ============================================================
log("\n[Step 5] Performance benchmark (10 iterations)...")
N_WARMUP = 1
N_BENCH = 3

# Warmup
for _ in range(N_WARMUP):
    params[define._SET_INDEX_] += 1
    qcu.applyInitQcu(set_ptrs, params, argv)
    qcu.applyCloverBistabCgQcu(fermion_out_ref, fermion_in_eo, gauge_eo,
                                 clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
                                 set_ptrs, params)

ref_times = []
for i in range(N_BENCH):
    params[define._SET_INDEX_] += 1
    qcu.applyInitQcu(set_ptrs, params, argv)
    t0 = perf_counter()
    qcu.applyCloverBistabCgQcu(fermion_out_ref, fermion_in_eo, gauge_eo,
                                 clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
                                 set_ptrs, params)
    t1 = perf_counter()
    ref_times.append(t1 - t0)

# Warmup MG
for _ in range(N_WARMUP):
    params[define._SET_INDEX_] += 1
    qcu.applyInitQcu(set_ptrs, params, argv)
    qcu.applyCloverMultigridQcu(fermion_out_mg, fermion_in_eo, gauge_eo,
                                  clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
                                  set_ptrs, params)

mg_times = []
for i in range(N_BENCH):
    params[define._SET_INDEX_] += 1
    qcu.applyInitQcu(set_ptrs, params, argv)
    t0 = perf_counter()
    qcu.applyCloverMultigridQcu(fermion_out_mg, fermion_in_eo, gauge_eo,
                                  clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,
                                  set_ptrs, params)
    t1 = perf_counter()
    mg_times.append(t1 - t0)

ref_mean, ref_std = np.mean(ref_times), np.std(ref_times)
mg_mean, mg_std = np.mean(mg_times), np.std(mg_times)

log(f"\n  BiStabCG: {ref_mean:.6f} ± {ref_std:.6f} s")
log(f"  Multigrid: {mg_mean:.6f} ± {mg_std:.6f} s")
log(f"  Speedup: {ref_mean/mg_mean:.2f}x" if mg_mean > 0 else "  Speedup: N/A")

# ============================================================
# Step 6: Save convergence history from C++ log
# ============================================================
log("\n[Step 6] Results summary...")

# ============================================================
# Step 7: Generate charts and reports
# ============================================================
log("\n[Step 7] Generating charts...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Performance comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    methods = ['BiStabCG', 'MG(1-level)']
    times = [ref_mean, mg_mean]
    errors = [ref_std, mg_std]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.bar(methods, times, yerr=errors, color=colors, capsize=10, alpha=0.85)
    ax1.set_ylabel('Time (s)')
    ax1.set_title(f'Performance Comparison\n{Lx}x{Ly}x{Lz}x{Lt}, complex64')
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{t:.4f}s', ha='center', va='bottom', fontweight='bold')

    # Residual comparison
    residuals = [ref_diff, mg_diff]
    ax2.bar(['BiStabCG', 'MG(1-level)'], residuals, color=['#e74c3c', '#f39c12'], alpha=0.85)
    ax2.set_ylabel('Relative Residual |Dx-b|/|b|')
    ax2.set_title('Solution Accuracy')
    ax2.set_yscale('log')
    ax2.axhline(y=1e-9, color='gray', linestyle='--', alpha=0.5, label='1e-9')
    ax2.legend()
    for i, (bar, r) in enumerate(zip(ax2.patches, residuals)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.5,
                f'{r:.2e}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(LOG_DIR, "multigrid_performance.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Chart saved to {chart_path}")

except Exception as e:
    log(f"  Chart generation error (non-fatal): {e}")

# ============================================================
# Step 8: Write JSON report
# ============================================================
report = {
    "timestamp": datetime.now().isoformat(),
    "lattice": [Lx, Ly, Lz, Lt],
    "grid": [int(x) for x in tools.give_grid_size()],
    "mass": float(MASS),
    "kappa": float(1.0 / (2 * MASS + 8)),
    "data_type": "complex64",
    "mg_levels": int(params[define._MG_NUM_LEVEL_].item()),
    "results": {
        "bistabcg": {
            "time_s": float(ref_mean),
            "time_std_s": float(ref_std),
            "relative_residual": float(ref_diff),
        },
        "multigrid": {
            "time_s": float(mg_mean),
            "time_std_s": float(mg_std),
            "relative_residual": float(mg_diff),
            "vs_reference": float(mg_vs_ref),
        },
        "speedup": float(ref_mean / mg_mean) if mg_mean > 0 else None,
    }
}

report_path = os.path.join(LOG_DIR, "multigrid_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
log(f"  JSON report saved to {report_path}")

# ============================================================
# Final verdict
# ============================================================
log("\n" + "=" * 70)
if mg_vs_ref < 1e-3:
    log("✓ PASS: Multigrid solution matches BiStabCG reference")
else:
    log("✗ WARNING: Multigrid solution differs from reference")
log(f"  BiStabCG residual: {ref_diff:.2e}")
log(f"  Multigrid residual: {mg_diff:.2e}")
log(f"  MG vs Ref agreement: {mg_vs_ref:.2e}")
log("=" * 70)
