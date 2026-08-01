#!/usr/bin/env python3
"""Final C++ MG vs BiStabCG benchmark across configurations."""
import torch, os, json, re, sys
from datetime import datetime
from time import perf_counter
import numpy as np
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

LOG_DIR = "/root/PyQCU/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configurations to test
CONFIGS = [
    ("8x8x8x16_c64_m0.05",  8,8,8,16, 0.05, 1e-6, define._LAT_C64_),
    ("8x8x8x16_c128_m0.05", 8,8,8,16, 0.05, 1e-9, define._LAT_C128_),
    ("12x12x12x16_c64_m0.05",12,12,12,16,0.05,1e-6, define._LAT_C64_),
    ("8x8x8x16_c64_m0.10",  8,8,8,16, 0.10, 1e-6, define._LAT_C64_),
]

results = []

for label, Lx, Ly, Lz, Lt, MASS, ATOL, DTYPE in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  {Lx}x{Ly}x{Lz}x{Lt}, mass={MASS}, tol={ATOL:.0e}, dtype={DTYPE}")
    KAPPA = 1.0/(2*MASS+8)

    # Reset params
    params[define._LAT_X_]=Lx;params[define._LAT_Y_]=Ly;params[define._LAT_Z_]=Lz;params[define._LAT_T_]=Lt
    params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
    params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
    params[define._PARITY_]=0;params[define._NODE_RANK_]=0;params[define._NODE_SIZE_]=1
    params[define._DAGGER_]=0;params[define._MAX_ITER_]=500;params[define._DATA_TYPE_]=DTYPE
    params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1;params[define._VERBOSE_]=0
    params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0
    params[define._MG_NUM_LEVEL_]=1;params[define._MG_LEVEL1_NUM_RESTART_]=3

    av = argv.to(dtype=define.dtype(DTYPE).to_real())
    av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1

    device=torch.device('cuda');dt_val=define.dtype(DTYPE);ls=define.lat_shape(params)

    g=torch.zeros((2,3,3,4)+tuple(ls),dtype=dt_val,device=device)
    fi=torch.randn((2,4,3)+tuple(ls),dtype=dt_val,device=device)
    fo_ref=torch.zeros_like(fi);fo_mg=torch.zeros_like(fi)
    ce=torch.zeros((4,3,4,3)+tuple(ls),dtype=dt_val,device=device);cei=torch.zeros_like(ce)
    coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)

    # Setup
    params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs,params,av);qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
    params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

    # === Test 1: C++ BiStabCG ===
    for trial in range(3):
        fi_trial=torch.randn_like(fi);fo_ref.zero_()
        params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1
        qcu.applyInitQcu(set_ptrs,params,av)
        t0=perf_counter()
        qcu.applyCloverBistabCgQcu(fo_ref,fi_trial,g,ce,coo,cei,coi,set_ptrs,params)
        t_bicg=perf_counter()-t0

        qcu_U=tools.poooxyzt2oooxyzt(g);qcu_src=tools.poooxyzt2oooxyzt(fi_trial)
        qcu_ref=tools.poooxyzt2oooxyzt(fo_ref)
        ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
        res=tools.norm(dslash.give_wilson(qcu_ref,qcu_U,KAPPA,True)+dslash.give_clover(qcu_ref,ref_cl)-qcu_src)/tools.norm(qcu_src)

        # === Test 2: C++ MG 1L ===
        fo_mg.zero_()
        params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1;params[define._VERBOSE_]=1
        qcu.applyInitQcu(set_ptrs,params,av)
        t0=perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg,fi_trial,g,ce,coo,cei,coi,set_ptrs,params)
        t_mg=perf_counter()-t0

        qcu_mg=tools.poooxyzt2oooxyzt(fo_mg)
        mg_res=tools.norm(dslash.give_wilson(qcu_mg,qcu_U,KAPPA,True)+dslash.give_clover(qcu_mg,ref_cl)-qcu_src)/tools.norm(qcu_src)
        mg_vs_ref=tools.norm(qcu_mg-qcu_ref)/tools.norm(qcu_ref)

        speedup = t_bicg/t_mg if t_mg>0 else 0
        status = "PASS" if float(mg_vs_ref)<1e-5 else ("OK" if float(mg_vs_ref)<1e-3 else "FAIL")
        print(f"  Trial {trial}: BiStabCG={t_bicg:.4f}s MG={t_mg:.4f}s speedup={speedup:.2f}x vs_ref={float(mg_vs_ref):.2e} {status}")

        results.append({
            "label": f"{label}_t{trial}", "lattice": [Lx,Ly,Lz,Lt],
            "mass": MASS, "tol": ATOL, "dtype": "c64" if DTYPE==2 else "c128",
            "bistabcg_time": t_bicg, "bistabcg_res": float(res),
            "mg_time": t_mg, "mg_res": float(mg_res), "mg_vs_ref": float(mg_vs_ref),
            "speedup": speedup, "status": status
        })

# Summary
print(f"\n{'='*80}")
print(f"{'Config':<40} {'BiStabCG':>9} {'MG':>9} {'Speedup':>7} {'Status':>6}")
print("-"*80)
for r in results:
    print(f"{r['label']:<40} {r['bistabcg_time']:9.4f} {r['mg_time']:9.4f} {r['speedup']:7.2f} {r['status']:>6}")

avg = np.mean([r['speedup'] for r in results])
print(f"\nAverage speedup: {avg:.2f}x across {len(results)} runs")
print(f"All PASS: {all(r['status']=='PASS' for r in results)}")

# Save
with open(os.path.join(LOG_DIR, "bench_report.json"), "w") as f:
    json.dump({"results": results, "timestamp": datetime.now().isoformat()}, f, indent=2)
print(f"Report: {LOG_DIR}/bench_report.json")
