#!/usr/bin/env python3
"""SCHUR-consistent C++ Clover Multigrid — multi-level test harness.

Level-0 solves the parity-preconditioned (Schur) Clover system
  S·x_o = b__o ,  S = D_oo - k^2 H_oe D_ee^{-1} H_eo
(matching applyCloverBistabCgDslashQcu).  The coarse space is built from the
SCHUR operator's own null vectors (capturing S's low modes — the previous
full-D-based coarse space was ineffective).  The coarse operator is the
33-tensor Galerkin A_c = P^T S P (on-site + nearest + diagonal couplings).

Usage:
    source ./env.sh && python examples/qcu/conftest.schur.multigrid.py
"""
import torch, os, sys, re, json, time
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_stencil_build import build_stencil, PAIRS, SIGN

def build_config(Lx,Ly,Lz,Lt,MASS,ATOL,NUM_LEVELS,DOF_LIST,MG_GRID,NUM_RESTART,
                 COARSE_MAX_ITER,COARSE_TOL_FACTOR,DT=define._LAT_C64_):
    params[define._LAT_X_]=Lx; params[define._LAT_Y_]=Ly
    params[define._LAT_Z_]=Lz; params[define._LAT_T_]=Lt
    params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
    params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
    params[define._PARITY_]=0; params[define._NODE_RANK_]=0; params[define._NODE_SIZE_]=1
    params[define._DAGGER_]=0; params[define._MAX_ITER_]=1000
    params[define._DATA_TYPE_]=DT
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=1
    params[define._VERBOSE_]=0; params[define._SEED_]=42; params[define._TEST_IN_CPU_]=0
    params[define._MG_NUM_LEVEL_]=NUM_LEVELS
    # Coarse levels are the coarsened ODD lattice: [X/2, Y/2, Z/2, T/4] for
    # coarsening factor 2 in all directions (level-0 odd-lattice T is T/2).
    if NUM_LEVELS>=2:
        params[define._MG_LEVEL1_E_]=DOF_LIST[1]
        params[define._MG_LEVEL1_X_]=Lx//MG_GRID[0]
        params[define._MG_LEVEL1_Y_]=Ly//MG_GRID[1]
        params[define._MG_LEVEL1_Z_]=Lz//MG_GRID[2]
        params[define._MG_LEVEL1_T_]=Lt//(2*MG_GRID[3])   # T/4 (odd-lattice coarsening)
        params[define._MG_LEVEL1_MAX_ITER_]=COARSE_MAX_ITER
        params[define._MG_LEVEL1_DATA_TYPE_]=DT
        # _MG_LEVEL1_NUM_RESTART_ doubles as the FINE (level-0) V-cycle frequency
        # (see lattice_clover_multigrid.h parse_params).
        params[define._MG_LEVEL1_NUM_RESTART_]=NUM_RESTART
    if NUM_LEVELS>=3:
        params[define._MG_LEVEL2_E_]=DOF_LIST[2]
        params[define._MG_LEVEL2_X_]=Lx//(MG_GRID[0]*MG_GRID[0])
        params[define._MG_LEVEL2_Y_]=Ly//(MG_GRID[1]*MG_GRID[1])
        params[define._MG_LEVEL2_Z_]=Lz//(MG_GRID[2]*MG_GRID[2])
        params[define._MG_LEVEL2_T_]=Lt//(4*MG_GRID[3])   # T/8
        params[define._MG_LEVEL2_MAX_ITER_]=200
        params[define._MG_LEVEL2_DATA_TYPE_]=DT
        params[define._MG_LEVEL2_NUM_RESTART_]=3
    av=argv.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_]=MASS; av[define._ATOL_]=ATOL; av[define._SIGMA_]=0.1
    if NUM_LEVELS>=2: av[define._MG_LEVEL1_ATOL_]=ATOL*COARSE_TOL_FACTOR
    if NUM_LEVELS>=3: av[define._MG_LEVEL2_ATOL_]=ATOL*COARSE_TOL_FACTOR
    return av

CACHE_DIR = "/root/PyQCU/logs/nullvec_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def build_schur_levels(op, S, NUM_LEVELS, DOF_LIST, MG_GRID, lat_full, E, dt, device, nv_iters=2, use_cache=True):
    """Build S null vectors + 33-tensor A_c for each coarse level.
    Returns lists [lonv[0],lonv[1]], [hop_nn[0],...], [hop_diag[...]], [sit[...]].
    Level 0 fine = odd lattice [X,Y,Z,T/2]; each coarse level halves all dims.
    Results are cached to disk (keyed by lattice/dof/nv_iters) so repeated
    runs (e.g. solver-parameter sweeps) skip the expensive setup."""
    lonvs, hnn_l, hdg_l, sit_l = [], [], [], []
    lat_fine_odd = [lat_full[0],lat_full[1],lat_full[2],lat_full[3]//2]
    E_prev = 12
    for lvl in range(1, NUM_LEVELS):
        E_c = DOF_LIST[lvl]
        lat_coarse_odd = [lat_fine_odd[d]//MG_GRID[d] for d in range(4)]
        tag = f"L{lat_full[0]}x{lat_full[1]}x{lat_full[2]}x{lat_full[3]}_lv{lvl}_E{E_c}_nvi{nv_iters}"
        cache = os.path.join(CACHE_DIR, tag)
        cached = (use_cache and all(os.path.exists(cache+"_"+k+".pt") for k in ["lonv","hnn","hdg","sit"]))
        if cached:
            lonv = torch.load(cache+"_lonv.pt", map_location=device)
            hnn = torch.load(cache+"_hnn.pt", map_location=device)
            hdg = torch.load(cache+"_hdg.pt", map_location=device)
            sit = torch.load(cache+"_sit.pt", map_location=device)
            print(f"  [level {lvl}] E={E_c} CACHED coarse={lat_coarse_odd}")
        else:
            t0=time.perf_counter()
            _null = torch.randn([E_c, E_prev]+lat_fine_odd, dtype=dt, device=device)
            for _ in range(nv_iters):
                _null = tools.give_null_vecs(null_vecs=_null, matvec=S, bistabcg=None, verbose=False)
            lonv = tools.local_orthogonalize(null_vecs=_null, coarse_lat_size=lat_coarse_odd, verbose=False)
            hnn, hdg, sit = build_stencil(S, lonv, E_c, E_prev, lat_fine_odd, lat_coarse_odd, dt, device)
            print(f"  [level {lvl}] E={E_c} nv_time={time.perf_counter()-t0:.1f}s coarse={lat_coarse_odd}")
            torch.save(lonv.cpu(), cache+"_lonv.pt"); torch.save(hnn.cpu(), cache+"_hnn.pt")
            torch.save(hdg.cpu(), cache+"_hdg.pt"); torch.save(sit.cpu(), cache+"_sit.pt")
        lonvs.append(lonv); hnn_l.append(hnn); hdg_l.append(hdg); sit_l.append(sit)
        # --- For the NEXT level, the "fine operator" is the materialized A_c ---
        def make_A(S_in, hnn_i, hdg_i, sit_i):
            from mg_stencil_build import apply_stencil
            def A(v): return apply_stencil(hnn_i, hdg_i, sit_i, v)
            return A
        S = make_A(S, hnn, hdg, sit)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd
    return lonvs, hnn_l, hdg_l, sit_l

def run(label, Lx,Ly,Lz,Lt,MASS,ATOL,NUM_LEVELS,DOF_LIST,MG_GRID,NUM_RESTART=10,
        COARSE_MAX_ITER=200,COARSE_TOL_FACTOR=1e3,DT=define._LAT_C64_,NV_ITERS=2,
        log_fn="clover_multigrid.log"):
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,NUM_LEVELS,DOF_LIST,MG_GRID,NUM_RESTART,
                      COARSE_MAX_ITER,COARSE_TOL_FACTOR,DT)
    KAPPA=1.0/(2*MASS+8)
    device=torch.device('cuda'); dt=define.dtype(DT); ls=define.lat_shape(params)
    g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
    fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
    fo_ref=torch.zeros_like(fi); fo_mg=torch.zeros_like(fi)
    ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device)
    cei=torch.zeros_like(ce); coo=torch.zeros_like(ce); coi=torch.zeros_like(ce)

    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

    # Reference: parity-preconditioned Clover BiStabCG
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); ref_time=time.perf_counter()-t0

    qcu_U=tools.poooxyzt2oooxyzt(g)
    qcu_src=tools.poooxyzt2oooxyzt(fi)
    qcu_ref=tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    ref_res=tools.norm(dslash.give_wilson(qcu_ref,qcu_U,KAPPA,True)+
                       dslash.give_clover(qcu_ref,ref_cl)-qcu_src)/tools.norm(qcu_src)

    # Build Schur coarse operators
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity
    lonvs, hnn_l, hdg_l, sit_l = build_schur_levels(op, S, NUM_LEVELS, DOF_LIST, MG_GRID,
                                                    [Lx,Ly,Lz,Lt], DOF_LIST[1], dt, device, NV_ITERS)
    # Pass to C++ via set_ptrs (4 slots per fine level, base 10)
    for fl in range(len(lonvs)):
        set_ptrs[30+4*fl+0]=lonvs[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+3]=sit_l[fl].contiguous().data_ptr()

    # Run C++ MG
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=1
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); mg_time=time.perf_counter()-t0

    qcu_mg=tools.poooxyzt2oooxyzt(fo_mg)
    mg_res=tools.norm(dslash.give_wilson(qcu_mg,qcu_U,KAPPA,True)+
                      dslash.give_clover(qcu_mg,ref_cl)-qcu_src)/tools.norm(qcu_src)
    mg_vs_ref=tools.norm(qcu_mg-qcu_ref)/tools.norm(qcu_ref)
    speedup=ref_time/mg_time if mg_time>0 else 0

    conv=[]
    log_path=os.path.join("/root/PyQCU/logs", log_fn)
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                m=re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
                if m: conv=[float(x) for x in m.group(1).split(",") if x.strip()]

    print("="*70)
    print(f"[{label}] {Lx}x{Ly}x{Lz}x{Lt} m={MASS} levels={NUM_LEVELS} dof={DOF_LIST} "
          f"restart={NUM_RESTART} nvi={NV_ITERS}")
    print(f"  BiStabCG : {ref_time*1000:.1f} ms  res={ref_res:.3e}")
    print(f"  MG       : {mg_time*1000:.1f} ms  res={mg_res:.3e}  vs_ref={mg_vs_ref:.3e}  "
          f"speedup={speedup:.3f}x")
    print(f"  MG conv pts: {len(conv)}" + (f"  iters={len([c for c in conv if c>ATOL])}" if conv else ""))
    status="PASS" if mg_vs_ref<1e-5 else ("OK" if mg_vs_ref<1e-2 else "FAIL")
    print(f"  Status: {status}")
    return {"label":label,"lattice":[Lx,Ly,Lz,Lt],"mass":MASS,"levels":NUM_LEVELS,
            "ref_time":ref_time,"mg_time":mg_time,"mg_res":float(mg_res),
            "mg_vs_ref":float(mg_vs_ref),"speedup":float(speedup),"conv":conv,
            "dof_list":DOF_LIST,"restart":NUM_RESTART,"status":status}

if __name__=="__main__":
    # (label, Lx,Ly,Lz,Lt, mass, atol, levels, dof, mg_grid, restart, coarse_max_iter, coarse_tol_factor, DT, nvi)
    CONFIGS = [
        ("8x8x8x16_c64_2L",  8, 8, 8, 16, 0.05, 1e-6, 2, [12,48], [2,2,2,2], 10, 200, 1e4, define._LAT_C64_, 2),
        ("8x16x16x16_c64_2L",8,16,16,16,0.05,1e-6, 2, [12,48], [2,2,2,2], 10, 200, 1e4, define._LAT_C64_, 2),
        ("8x16x16x16_c64_3L",8,16,16,16,0.05,1e-6, 3, [12,48,48],[2,2,2,2],10, 200, 1e4, define._LAT_C64_, 2),
    ]
    results=[]
    for cfg in CONFIGS:
        label,Lx,Ly,Lz,Lt,MASS,ATOL,LVL,DOF,MGGRID,NR,CMI,CTF,DT,NVI=cfg
        try:
            results.append(run(label,Lx,Ly,Lz,Lt,MASS,ATOL,LVL,DOF,MGGRID,NR,CMI,CTF,DT,NVI))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{label}] FAILED: {e}")
    print("\n"+"="*70)
    print("SUMMARY")
    for r in results:
        print(f"{r['label']}: BiStabCG={r['ref_time']*1000:.1f}ms MG={r['mg_time']*1000:.1f}ms "
              f"speedup={r['speedup']:.3f}x res={r['mg_res']:.3e} vs_ref={r['mg_vs_ref']:.3e} "
              f"iters={len([c for c in r['conv'] if c>1e-6])} {r['status']}")
    with open("/root/PyQCU/logs/schur_mg_results.json","w") as f:
        json.dump({"results":[{k:(v if not isinstance(v,list) else v) for k,v in r.items() if k!="conv"} for r in results]},f,indent=2)
