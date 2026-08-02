#!/usr/bin/env python3
"""Micro-benchmark: time individual coarse-level operations via the C++ MG
by running it with different num_restart and measuring the coarse-only cost."""
import torch, os, sys, time
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

def setup(restart=10, ctf=3e2):
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,2,[12,48],[2,2,2,2],restart,200,ctf,define._LAT_C64_)
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
    lonvs,hnn_l,hdg_l,sit_l=build_schur_levels(op,op.matvec_parity,2,[12,48],[2,2,2,2],[Lx,Ly,Lz,Lt],48,dt,device,2)
    for fl in range(len(lonvs)):
        set_ptrs[30+4*fl+0]=lonvs[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+3]=sit_l[fl].contiguous().data_ptr()
    return av, g, fi, fo, ce, coo, cei, coi

def run_mg(av, fi, g, fo, ce, coo, cei, coi):
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverMultigridQcu(fo,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); return time.perf_counter()-t0

if __name__=="__main__":
    time.sleep(3)
    import re
    for ctf in [3e2, 1e3, 3e3, 1e4, 3e4, 1e5]:
        av,g,fi,fo,ce,coo,cei,coi = setup(restart=10, ctf=ctf)
        t = run_mg(av,fi,g,fo,ce,coo,cei,coi)  # includes warmup effect (2nd call)
        # parse fine iters from log
        conv=[]
        lp="/root/PyQCU/logs/clover_multigrid.log"
        if os.path.exists(lp):
            with open(lp) as f2:
                for line in f2:
                    m=re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
                    if m: conv=[float(x) for x in m.group(1).split(",") if x.strip()]
        fine = len([c for c in conv if c>1e-6]) if conv else -1
        print(f"ct={ctf:.0e}: fine={fine} MG={t*1000:.0f}ms")
