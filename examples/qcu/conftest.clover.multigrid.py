#!/usr/bin/env python3
"""PyQCU C++ CUDA Clover Multigrid Solver — Correctness Test"""
import torch, os, json, re
from datetime import datetime
from time import perf_counter
from pyqcu import tools, dslash
from pyqcu.cuda import qcu, define
from pyqcu.cuda.define import params, argv, set_ptrs

LOG_DIR = "/root/PyQCU/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg, fn="test_multigrid.log"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    print(msg)
    with open(os.path.join(LOG_DIR, fn), "a") as f:
        f.write(line + "\n")

open(os.path.join(LOG_DIR, "test_multigrid.log"), "w").close()
log("=" * 70)
log("PyQCU C++ CUDA Clover Multigrid — Correctness Test")

Lx, Ly, Lz, Lt = 8, 8, 8, 16
params[define._LAT_X_] = Lx; params[define._LAT_Y_] = Ly
params[define._LAT_Z_] = Lz; params[define._LAT_T_] = Lt
params[define._LAT_XYZT_] = Lx*Ly*Lz*Lt
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[define._GRID_T_] = tools.give_grid_size()
params[define._PARITY_] = 0; params[define._NODE_RANK_] = define.rank; params[define._NODE_SIZE_] = define.size
params[define._DAGGER_] = 0; params[define._MAX_ITER_] = 500; params[define._DATA_TYPE_] = define._LAT_C64_
params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = 1; params[define._VERBOSE_] = 0
params[define._SEED_] = 42; params[define._TEST_IN_CPU_] = 0
params[define._MG_NUM_LEVEL_] = 1; params[define._MG_LEVEL1_E_] = 12
params[define._MG_LEVEL1_X_] = Lx//2; params[define._MG_LEVEL1_Y_] = Ly//2
params[define._MG_LEVEL1_Z_] = Lz//2; params[define._MG_LEVEL1_T_] = Lt//2
params[define._MG_LEVEL1_MAX_ITER_] = 50; params[define._MG_LEVEL1_DATA_TYPE_] = define._LAT_C64_
params[define._MG_LEVEL1_NUM_RESTART_] = 3

MASS = 0.05; kappa = 1.0/(2*MASS+8)
argv_new = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
argv_new[define._MASS_] = MASS; argv_new[define._ATOL_] = 1e-6; argv_new[define._SIGMA_] = 0.1

device = torch.device('cuda'); dtype_t = define.dtype(params[define._DATA_TYPE_])
lat_shape = define.lat_shape(params)

g = torch.zeros([2,3,3,4]+lat_shape, dtype=dtype_t, device=device)
fi = torch.randn([2,4,3]+lat_shape, dtype=dtype_t, device=device)
fo_ref = torch.zeros_like(fi); fo_mg = torch.zeros_like(fi)
ce = torch.zeros([4,3,4,3]+lat_shape, dtype=dtype_t, device=device)
cei = torch.zeros_like(ce); coo = torch.zeros_like(ce); coi = torch.zeros_like(ce)

# Setup
log("[1] Setup: gauge + clover...")
params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = -1
qcu.applyInitQcu(set_ptrs, params, argv_new); qcu.applyGaussGaugeQcu(g, set_ptrs, params)
params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 2; params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv_new); qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 2; params[define._PARITY_] = 1
qcu.applyInitQcu(set_ptrs, params, argv_new); qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

# Reference BiStabCG
log("[2] Reference BiStabCG...")
params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 1; params[define._VERBOSE_] = 1
qcu.applyInitQcu(set_ptrs, params, argv_new)
t0 = perf_counter()
qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs, params)
ref_time = perf_counter() - t0

qcu_U = tools.poooxyzt2oooxyzt(g); qcu_src = tools.poooxyzt2oooxyzt(fi); qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
ref_cl = dslash.make_clover(qcu_U, kappa=kappa)
ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, kappa, True) +
                     dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)
log(f"  BiStabCG: {ref_time:.4f}s, residual={ref_res:.4e}")

# Multigrid
log("[3] Multigrid...")
params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 1; params[define._VERBOSE_] = 1
qcu.applyInitQcu(set_ptrs, params, argv_new)
t0 = perf_counter()
qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi, set_ptrs, params)
mg_time = perf_counter() - t0

qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, kappa, True) +
                    dslash.give_clover(qcu_mg, ref_cl) - qcu_src) / tools.norm(qcu_src)
mg_vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
log(f"  MG: {mg_time:.4f}s, residual={mg_res:.4e}, vs_ref={mg_vs_ref:.4e}")

# Parse C++ convergence log
conv = []
log_path = os.path.join(LOG_DIR, "clover_multigrid.log")
if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            m = re.search(r'CONVERGENCE_HISTORY: \[(.*?)\]', line)
            if m:
                conv = [float(x) for x in m.group(1).split(",") if x.strip()]
                break
    log(f"  Parsed {len(conv)} convergence points")

# Charts
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(['BiStabCG', 'MG'], [ref_time, mg_time], color=['#3498db','#2ecc71'], alpha=0.85)
    ax1.set_ylabel('Time (s)'); ax1.set_title(f'Performance: {Lx}x{Ly}x{Lz}x{Lt}, c64')
    for b, t in zip(ax1.patches, [ref_time, mg_time]):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{t:.4f}s', ha='center', fontweight='bold')
    if conv:
        ax2.semilogy(conv, 'b-', linewidth=1)
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('Residual')
        ax2.set_title(f'Convergence (tol={1e-6})'); ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1e-6, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(LOG_DIR, "multigrid_result.png"), dpi=150); plt.close()
    log("  Chart saved")
except Exception as e:
    log(f"  Chart error: {e}")

# Report
report = {
    "timestamp": datetime.now().isoformat(), "lattice": [Lx, Ly, Lz, Lt],
    "mass": MASS, "kappa": kappa, "precision": "complex64",
    "bistabcg_time_s": ref_time, "bistabcg_residual": float(ref_res),
    "multigrid_time_s": mg_time, "multigrid_residual": float(mg_res),
    "mg_vs_ref": float(mg_vs_ref), "speedup": ref_time/mg_time if mg_time else 0,
    "convergence": [float(c) for c in conv[:200]]
}
with open(os.path.join(LOG_DIR, "multigrid_report.json"), "w") as f:
    json.dump(report, f, indent=2)

log(f"\n{'='*70}")
if mg_vs_ref < 1e-3:
    log(f"PASS: |x_mg-x_ref|/|x_ref| = {mg_vs_ref:.2e}")
else:
    log(f"WARNING: |x_mg-x_ref|/|x_ref| = {mg_vs_ref:.2e}")
log(f"  BiStabCG: {ref_time:.4f}s, residual={ref_res:.2e}")
log(f"  MG: {mg_time:.4f}s, residual={mg_res:.2e}, speedup={ref_time/mg_time:.2f}x")
log("="*70)
