#!/usr/bin/env python3
"""Standalone test harness: C++ Clover Multigrid vs BiStabCG.

Builds gauge + clover, runs reference applyCloverBistabCgQcu, builds coarse
operators (Galerkin via Python tools), runs applyCloverMultigridQcu, and
compares convergence + accuracy + timing.

Usage:
    source ./env.sh && python examples/qcu/mg_dev_fulltest.py
"""
import torch, os, sys, json, re
from time import perf_counter
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

def build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
                 NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR, DT=define._LAT_C64_):
    params[define._LAT_X_] = Lx; params[define._LAT_Y_] = Ly
    params[define._LAT_Z_] = Lz; params[define._LAT_T_] = Lt
    params[define._LAT_XYZT_] = Lx*Ly*Lz*Lt
    params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[define._GRID_T_] = tools.give_grid_size()
    params[define._PARITY_] = 0; params[define._NODE_RANK_] = 0; params[define._NODE_SIZE_] = 1
    params[define._DAGGER_] = 0; params[define._MAX_ITER_] = 1000
    params[define._DATA_TYPE_] = DT
    params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0; params[define._SEED_] = 42; params[define._TEST_IN_CPU_] = 0
    params[define._MG_NUM_LEVEL_] = NUM_LEVELS
    if NUM_LEVELS >= 2:
        params[define._MG_LEVEL1_E_] = DOF_LIST[1]
        params[define._MG_LEVEL1_X_] = Lx // MG_GRID[0]
        params[define._MG_LEVEL1_Y_] = Ly // MG_GRID[1]
        params[define._MG_LEVEL1_Z_] = Lz // MG_GRID[2]
        params[define._MG_LEVEL1_T_] = Lt // MG_GRID[3]
        params[define._MG_LEVEL1_MAX_ITER_] = COARSE_MAX_ITER
        params[define._MG_LEVEL1_DATA_TYPE_] = DT
        params[define._MG_LEVEL1_NUM_RESTART_] = NUM_RESTART
    if NUM_LEVELS >= 3:
        params[define._MG_LEVEL2_E_] = DOF_LIST[2] if len(DOF_LIST) > 2 else 24
        params[define._MG_LEVEL2_X_] = Lx // (MG_GRID[0] * MG_GRID[0])
        params[define._MG_LEVEL2_Y_] = Ly // (MG_GRID[1] * MG_GRID[1])
        params[define._MG_LEVEL2_Z_] = Lz // (MG_GRID[2] * MG_GRID[2])
        params[define._MG_LEVEL2_T_] = Lt // (MG_GRID[3] * MG_GRID[3])
        params[define._MG_LEVEL2_MAX_ITER_] = 200
        params[define._MG_LEVEL2_DATA_TYPE_] = DT
        params[define._MG_LEVEL2_NUM_RESTART_] = 3
    if NUM_LEVELS >= 4:
        params[define._MG_LEVEL3_E_] = DOF_LIST[3] if len(DOF_LIST) > 3 else 24
        params[define._MG_LEVEL3_X_] = Lx // (MG_GRID[0]**3)
        params[define._MG_LEVEL3_Y_] = Ly // (MG_GRID[1]**3)
        params[define._MG_LEVEL3_Z_] = Lz // (MG_GRID[2]**3)
        params[define._MG_LEVEL3_T_] = Lt // (MG_GRID[3]**3)
        params[define._MG_LEVEL3_MAX_ITER_] = 200
        params[define._MG_LEVEL3_DATA_TYPE_] = DT
        params[define._MG_LEVEL3_NUM_RESTART_] = 3

    av = argv.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_] = MASS; av[define._ATOL_] = ATOL; av[define._SIGMA_] = 0.1
    if NUM_LEVELS >= 2:
        av[define._MG_LEVEL1_ATOL_] = ATOL * COARSE_TOL_FACTOR
    if NUM_LEVELS >= 3:
        av[define._MG_LEVEL2_ATOL_] = ATOL * COARSE_TOL_FACTOR
    return av

def run(label, Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
        NUM_RESTART=5, COARSE_MAX_ITER=100, COARSE_TOL_FACTOR=10.0, DT=define._LAT_C64_,
        verbose=True):
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
                      NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR, DT)
    KAPPA = 1.0/(2*MASS+8)
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)
    real_dt = dt.to_real()

    # Allocate GPU tensors
    g = torch.zeros([2,3,3,4]+ls, dtype=dt, device=device)
    fi = torch.randn([2,4,3]+ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi); fo_mg = torch.zeros_like(fi)
    ce = torch.zeros([4,3,4,3]+ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce); coo = torch.zeros_like(ce); coi = torch.zeros_like(ce)

    # Generate gauge + clover
    params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 2; params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 2; params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    # Reference BiStabCG
    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 1; params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    torch.cuda.synchronize(); t0 = perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs, params)
    torch.cuda.synchronize(); ref_time = perf_counter() - t0

    # Python-side full reference for residual check
    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)

    # Build coarse operators if multi-level
    if NUM_LEVELS >= 2:
        U_full = qcu_U
        op_fine = dslash.operator(U=U_full, clover_term=ref_cl, kappa=KAPPA,
                                   support_parity=False, verbose=False)
        lat_sizes = [[Lx, Ly, Lz, Lt]]
        for i in range(1, NUM_LEVELS):
            lat_sizes.append([max(lat_sizes[i-1][d] // MG_GRID[d], 1) for d in range(4)])
        op_list = [op_fine]
        lonv_list = []; hop_packed_list = []; sit_packed_list = []
        for i in range(1, NUM_LEVELS):
            dof_fine = DOF_LIST[i-1]; dof_coarse = DOF_LIST[i]
            lat_fine = lat_sizes[i-1]; lat_coarse = lat_sizes[i]
            _null_vecs = torch.randn([dof_coarse, dof_fine] + lat_fine,
                                      dtype=dt, device=device)
            _null_vecs = tools.give_null_vecs(null_vecs=_null_vecs,
                matvec=op_list[i-1].matvec, bistabcg=None, verbose=False)
            _lonv = tools.local_orthogonalize(null_vecs=_null_vecs,
                coarse_lat_size=lat_coarse, verbose=False)
            E_lonv = _lonv.shape[0]; e_lonv = _lonv.shape[1]
            Xc=_lonv.shape[2]; mgx=_lonv.shape[3]; Yc=_lonv.shape[4]; mgy=_lonv.shape[5]
            Zc=_lonv.shape[6]; mgz=_lonv.shape[7]; Tc=_lonv.shape[8]; mgt=_lonv.shape[9]
            _lonv_flat = _lonv.reshape(E_lonv, e_lonv,
                Xc*mgx, Yc*mgy, Zc*mgz, Tc*mgt).contiguous()
            lonv_list.append(_lonv_flat)
            coarse_op = dslash.operator(fine_hopping=op_list[i-1].hopping,
                fine_sitting=op_list[i-1].sitting,
                local_ortho_null_vecs=_lonv, verbose=False)
            op_list.append(coarse_op)
            E = dof_coarse; Xc, Yc, Zc, Tc = lat_coarse
            hp = torch.zeros([2,4,E,E,Xc,Yc,Zc,Tc], dtype=dt, device=device)
            for ward in range(4):
                hp[0,ward] = coarse_op.hopping.M_plus_list[ward].to(dtype=dt, device=device)
                hp[1,ward] = coarse_op.hopping.M_minus_list[ward].to(dtype=dt, device=device)
            hop_packed_list.append(hp)
            # NOTE: To match pyqcu/solver/_multigrid.py EXACTLY, the coarse-grid
            # sitting operator is the IDENTITY, not the Galerkin-projected M.
            # In the Python reference, coarse_op.sitting.matvec() returns `src`
            # (clover_term is None for coarse operators), so the effective coarse
            # operator is hopping + I.  This damped coarse operator gives a more
            # reliable V-cycle correction than the (mathematically correct)
            # hopping + M, which overshoots when the null-space basis is imperfect.
            sit_identity = torch.zeros([E,E,Xc,Yc,Zc,Tc], dtype=dt, device=device)
            for e_i in range(E):
                sit_identity[e_i,e_i] = 1.0
            sit_packed_list.append(sit_identity)
        for fl in range(len(lonv_list)):
            set_ptrs[10 + 3*fl + 0] = lonv_list[fl].contiguous().data_ptr()
            set_ptrs[10 + 3*fl + 1] = hop_packed_list[fl].contiguous().data_ptr()
            set_ptrs[10 + 3*fl + 2] = sit_packed_list[fl].contiguous().data_ptr()

    # ---- Save fixture + coarse ops for debugging ----
    def save_t(t, name):
        torch.save(t.cpu(), f"/tmp/mgfx_{name}.pt")
    save_t(g, "g"); save_t(fi, "fi")
    if NUM_LEVELS >= 2:
        save_t(lonv_list[0], "lonv"); save_t(hop_packed_list[0], "hp")
        save_t(sit_packed_list[0], "sp")
        import numpy as np
        np.save("/tmp/mgfx_kappa.npy", np.array([KAPPA]))

    # Run C++ MG
    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 1; params[define._VERBOSE_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    torch.cuda.synchronize(); t0 = perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi, set_ptrs, params)
    torch.cuda.synchronize(); mg_time = perf_counter() - t0

    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, KAPPA, True) +
                        dslash.give_clover(qcu_mg, ref_cl) - qcu_src) / tools.norm(qcu_src)
    mg_vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
    speedup = ref_time/mg_time if mg_time > 0 else 0

    # Parse convergence from log
    conv = []
    log_path = os.path.join(os.path.expanduser("~/PyQCU/logs/dev73"), "clover_multigrid.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
                if m:
                    conv = [float(x) for x in m.group(1).split(",") if x.strip()]

    if verbose:
        print("="*70)
        print(f"[{label}] {Lx}x{Ly}x{Lz}x{Lt}, m={MASS}, levels={NUM_LEVELS}, dof={DOF_LIST}, restart={NUM_RESTART}")
        print(f"  BiStabCG : {ref_time*1000:.1f} ms, res={ref_res:.3e}")
        print(f"  MG       : {mg_time*1000:.1f} ms, res={mg_res:.3e}, vs_ref={mg_vs_ref:.3e}, speedup={speedup:.3f}x")
        print(f"  MG conv pts: {len(conv)}")
        if conv:
            print(f"  MG initial residual: {conv[0]:.3e}, final: {conv[-1]:.3e}")
    return {"label":label,"lattice":[Lx,Ly,Lz,Lt],"mass":MASS,"levels":NUM_LEVELS,
            "ref_time":ref_time,"mg_time":mg_time,"ref_res":float(ref_res),
            "mg_res":float(mg_res),"mg_vs_ref":float(mg_vs_ref),"speedup":float(speedup),
            "conv":conv,"dof_list":DOF_LIST,"restart":NUM_RESTART}

if __name__ == "__main__":
    import sys
    # ---- Multi-config test matrix ----
    # (label, Lx,Ly,Lz,Lt, mass, atol, num_levels, dof_list, mg_grid, restart, coarse_max_iter, coarse_tol_factor, DT)
    CONFIGS = [
        # Single-precision, small lattice, 1/2/3 levels
        ("8x8x8x16_c64_1L",     8,8,8,16,   0.05, 1e-6, 1, [12],    [2,2,2,2], 10, 100, 10.0, define._LAT_C64_),
        ("8x8x8x16_c64_2L",     8,8,8,16,   0.05, 1e-6, 2, [12,24], [2,2,2,2], 10, 100, 10.0, define._LAT_C64_),
        ("8x8x8x16_c64_3L",     8,8,8,16,   0.05, 1e-6, 3, [12,24,24], [2,2,2,2], 10, 100, 10.0, define._LAT_C64_),
        # Single-precision, medium lattice (user's default {8,16,16,16})
        ("8x16x16x16_c64_2L",   8,16,16,16, 0.05, 1e-6, 2, [12,24], [2,2,2,2], 10, 100, 10.0, define._LAT_C64_),
        ("8x16x16x16_c64_3L",   8,16,16,16, 0.05, 1e-6, 3, [12,24,24], [2,2,2,2], 10, 100, 10.0, define._LAT_C64_),
        # Double-precision
        ("8x8x8x16_c128_2L",    8,8,8,16,   0.05, 1e-10, 2, [12,24], [2,2,2,2], 10, 100, 10.0, define._LAT_C128_),
        # Different mass
        ("8x8x8x16_c64_m0.1_2L",8,8,8,16,   0.10, 1e-6, 2, [12,24], [2,2,2,2], 10, 100, 10.0, define._LAT_C64_),
    ]
    results = []
    for cfg in CONFIGS:
        label,Lx,Ly,Lz,Lt,MASS,ATOL,LVL,DOF,MGGRID,NR,CMI,CTF,DT = cfg
        try:
            results.append(run(label,Lx,Ly,Lz,Lt,MASS,ATOL,LVL,DOF,MGGRID,NR,CMI,CTF,DT))
        except Exception as e:
            print(f"[{label}] FAILED: {e}")
    print("\n" + "="*70)
    print("SUMMARY")
    for r in results:
        print(f"{r['label']}: BiStabCG={r['ref_time']*1000:.1f}ms MG={r['mg_time']*1000:.1f}ms "
              f"speedup={r['speedup']:.3f}x res={r['mg_res']:.3e} vs_ref={r['mg_vs_ref']:.3e} "
              f"iter={len(r['conv'])}")
    # Save JSON report
    import json
    with open(os.path.expanduser("~/PyQCU/logs/dev73/mg_dev_results.json"),"w") as f:
        json.dump({"results":[{k:(v if not isinstance(v,list) else v) for k,v in r.items() if k!="conv"} for r in results]}, f, indent=2)
    print(f"Report saved to {os.path.expanduser('~/PyQCU/logs/dev73/mg_dev_results.json')}")
