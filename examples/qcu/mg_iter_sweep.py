#!/usr/bin/env python3
"""Sweep MG configs on 8x8x8x16 to MINIMIZE fine iterations (cached setup).
Reports fine iters + MG solve time for each config."""
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

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
device=torch.device('cuda'); dt=define.dtype(define._LAT_C64_)

def run_cfg(E, nvi, restart, ctf, cmi=300):
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,2,[12,E],[2,2,2,2],restart,cmi,ctf,define._LAT_C64_)
    ls=define.lat_shape(params)
    g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
    fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
    fo=torch.zeros_like(fi)
    ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device)
    cei=torch.zeros_like(ce); coo=torch.zeros_like(ce); coi=torch.zeros_like(ce)
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)
    qcu_U=tools.poooxyzt2oooxyzt(g)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=torch.Tensor([KAPPA]),support_parity=True,verbose=False)
    lonvs,hnn_l,hdg_l,sit_l=build_schur_levels(op,op.matvec_parity,2,[12,E],[2,2,2,2],[Lx,Ly,Lz,Lt],E,dt,device,nvi)
    for fl in range(len(lonvs)):
        set_ptrs[30+4*fl+0]=lonvs[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+3]=sit_l[fl].contiguous().data_ptr()
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverMultigridQcu(fo,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); mg_time=time.perf_counter()-t0
    qcu_mg=tools.poooxyzt2oooxyzt(fo); qcu_src=tools.poooxyzt2oooxyzt(fi)
    res=tools.norm(dslash.give_wilson(qcu_mg,qcu_U,KAPPA,True)+dslash.give_clover(qcu_mg,ref_cl)-qcu_src)/tools.norm(qcu_src)
    conv=[]
    lp=os.path.expanduser("~/PyQCU/logs/clover_multigrid.log")
    if os.path.exists(lp):
        with open(lp) as f:
            for line in f:
                m=re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
                if m: conv=[float(x) for x in m.group(1).split(",") if x.strip()]
    fine = len([c for c in conv if c>ATOL]) if conv else -1
    print(f"[E={E} nvi={nvi} r={restart} ct={ctf:.0e}]: fine={fine} solve={mg_time*1000:.0f}ms res={res:.2e}")
    return fine, mg_time

if __name__=="__main__":
    time.sleep(10)  # cooldown
    configs = [
        (48,2,10,1e3), (48,2,10,3e2), (48,2,10,1e4), (48,2,15,1e3), (48,2,7,1e3),
        (48,3,10,1e3), (64,2,10,1e3), (48,2,12,1e3),
    ]
    for cfg in configs:
        try: run_cfg(*cfg)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"FAILED {cfg}: {e}")
