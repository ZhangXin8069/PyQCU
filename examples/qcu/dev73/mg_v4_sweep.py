#!/usr/bin/env python3
"""MG solver parameter sweep on cached null-vecs. Sweeps num_restart and
coarse tolerance factor; reports fine iterations, MG time, speedup vs the
parity-preconditioned BiStabCG reference."""
import torch, os, sys, time, json
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
_csm = _load("csm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "conftest.schur.multigrid.py"))
build_config = _csm.build_config; build_schur_levels = _csm.build_schur_levels

def run_one(Lx,Ly,Lz,Lt,MASS,ATOL,NUM_LEVELS,DOF_LIST,MG_GRID,NUM_RESTART,COARSE_MAX_ITER,COARSE_TOL_FACTOR,DT,NV_ITERS):
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,NUM_LEVELS,DOF_LIST,MG_GRID,NUM_RESTART,COARSE_MAX_ITER,COARSE_TOL_FACTOR,DT)
    KAPPA=1.0/(2*MASS+8); device=torch.device('cuda'); dt=define.dtype(DT); ls=define.lat_shape(params)
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
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); ref_time=time.perf_counter()-t0
    qcu_U=tools.poooxyzt2oooxyzt(g); qcu_src=tools.poooxyzt2oooxyzt(fi)
    qcu_ref=tools.poooxyzt2oooxyzt(fo_ref); ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]), support_parity=True, verbose=False)
    S = op.matvec_parity
    lonvs, hnn_l, hdg_l, sit_l = build_schur_levels(op, S, NUM_LEVELS, DOF_LIST, MG_GRID, [Lx,Ly,Lz,Lt], DOF_LIST[1], dt, device, NV_ITERS)
    for fl in range(len(lonvs)):
        set_ptrs[30+4*fl+0]=lonvs[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+3]=sit_l[fl].contiguous().data_ptr()
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); mg_time=time.perf_counter()-t0
    qcu_mg=tools.poooxyzt2oooxyzt(fo_mg)
    mg_vs_ref=tools.norm(qcu_mg-qcu_ref)/tools.norm(qcu_ref)
    conv=[]
    lp=os.path.expanduser("~/PyQCU/logs/clover_multigrid.log")
    if os.path.exists(lp):
        import re
        for line in open(lp):
            m=re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
            if m: conv=[float(x) for x in m.group(1).split(",") if x.strip()]
    iters=len([c for c in conv if c>ATOL])
    return {"ref":ref_time,"mg":mg_time,"speedup":ref_time/mg_time,"vs_ref":float(mg_vs_ref),"iters":iters,"conv":conv[-3:]}

def main():
    # (label, Lx,Ly,Lz,Lt, restart, coarse_max_iter, coarse_tol_factor)
    SWEEPS = [
        ("8x8x8x16 r=10 ct=1e4", 8,8,8,16, 10, 200, 1e4),
        ("8x8x8x16 r=15 ct=1e4", 8,8,8,16, 15, 200, 1e4),
        ("8x8x8x16 r=20 ct=1e4", 8,8,8,16, 20, 200, 1e4),
        ("8x8x8x16 r=5 ct=1e4",  8,8,8,16, 5,  200, 1e4),
        ("8x8x8x16 r=10 ct=1e3", 8,8,8,16, 10, 200, 1e3),
        ("8x8x8x16 r=10 ct=3e3", 8,8,8,16, 10, 200, 3e3),
        ("8x8x8x16 r=15 ct=1e3", 8,8,8,16, 15, 200, 1e3),
        ("8x8x8x16 r=10 ct=1e2", 8,8,8,16, 10, 200, 1e2),
    ]
    results=[]
    for label,Lx,Ly,Lz,Lt,NR,CMI,CTF in SWEEPS:
        r=run_one(Lx,Ly,Lz,Lt,0.05,1e-6,2,[12,48],[2,2,2,2],NR,CMI,CTF,define._LAT_C64_,2)
        r["label"]=label
        results.append(r)
        print(f"[{label}] ref={r['ref']*1000:.0f}ms mg={r['mg']*1000:.0f}ms "
              f"speedup={r['speedup']:.3f}x iters={r['iters']} vs_ref={r['vs_ref']:.2e} {r['conv']}")
    results.sort(key=lambda r:-r['speedup'])
    print("\n=== BEST ===")
    for r in results[:3]:
        print(f"{r['label']}: {r['speedup']:.3f}x  mg={r['mg']*1000:.0f}ms iters={r['iters']}")
    with open(os.path.expanduser("~/PyQCU/logs/dev73/mg_v4_sweep.json"),"w") as f:
        json.dump(results,f,indent=2)

if __name__=="__main__":
    main()
