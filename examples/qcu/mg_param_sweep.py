#!/usr/bin/env python3
"""Sweep MG solver parameters on 8x8x8x16 (cached setup) to find the fastest
configuration. Measures fine iterations, coarse iterations, and MG vs BiStabCG
wall-clock ratio."""
import torch, os, sys, time, re
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location("csm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "conftest.schur.multigrid.py"))
_csm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_csm)
build_config = _csm.build_config; build_schur_levels = _csm.build_schur_levels

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6
KAPPA=1.0/(2*MASS+8); device=torch.device('cuda'); dt=define.dtype(define._LAT_C64_)

def run_solve(restart, coarse_tol_factor, coarse_max_iter=200):
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,2,[12,48],[2,2,2,2],restart,coarse_max_iter,coarse_tol_factor,define._LAT_C64_)
    ls=define.lat_shape(params)
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
    qcu_U=tools.poooxyzt2oooxyzt(g); qcu_src=tools.poooxyzt2oooxyzt(fi); qcu_ref=tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=torch.Tensor([KAPPA]),support_parity=True,verbose=False)
    lonvs,hnn_l,hdg_l,sit_l=build_schur_levels(op,op.matvec_parity,2,[12,48],[2,2,2,2],[Lx,Ly,Lz,Lt],48,dt,device,2)
    for fl in range(len(lonvs)):
        set_ptrs[10+4*fl+0]=lonvs[fl].contiguous().data_ptr()
        set_ptrs[10+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
        set_ptrs[10+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
        set_ptrs[10+4*fl+3]=sit_l[fl].contiguous().data_ptr()
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); mg_time=time.perf_counter()-t0
    qcu_mg=tools.poooxyzt2oooxyzt(fo_mg)
    mg_vs_ref=tools.norm(qcu_mg-qcu_ref)/tools.norm(qcu_ref)
    # parse fine iters from solve_time log? (verbose off -> no log). Use conv from log.
    conv=[]
    log_path="/root/PyQCU/logs/clover_multigrid.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                m=re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
                if m: conv=[float(x) for x in m.group(1).split(",") if x.strip()]
    fine_iters = len([c for c in conv if c > ATOL]) if conv else -1
    return ref_time, mg_time, mg_vs_ref, fine_iters

if __name__=="__main__":
    print("== Parameter sweep on 8x8x8x16 ==")
    configs = [
        ("r=10 ct=1e-3",  10, 1e3),
        ("r=10 ct=1e-2",  10, 1e4),
        ("r=10 ct=3e-2",  10, 3e4),
        ("r=20 ct=1e-3",  20, 1e3),
        ("r=5  ct=1e-3",   5, 1e3),
        ("r=10 ct=1e-4",  10, 1e4*0.1),
    ]
    for name, restart, ctf in configs:
        ref_time, mg_time, vs, fine = run_solve(restart, ctf)
        speedup = ref_time/mg_time
        print(f"[{name}]: BiStabCG={ref_time*1000:.0f}ms MG={mg_time*1000:.0f}ms "
              f"speedup={speedup:.3f}x vs_ref={vs:.2e} fine_iters={fine}")
