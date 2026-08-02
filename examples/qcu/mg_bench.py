#!/usr/bin/env python3
"""Fair interleaved benchmark: BiStabCG vs Schur MG on 8x8x8x16 (cached setup).
Runs each 3 times interleaved, reports best-of-3 for a stable ratio despite
GPU clock throttling drift."""
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

def setup(restart=10, ctf=1e3):
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,2,[12,48],[2,2,2,2],restart,200,ctf,define._LAT_C64_)
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
    qcu_U=tools.poooxyzt2oooxyzt(g)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=torch.Tensor([KAPPA]),support_parity=True,verbose=False)
    lonvs,hnn_l,hdg_l,sit_l=build_schur_levels(op,op.matvec_parity,2,[12,48],[2,2,2,2],[Lx,Ly,Lz,Lt],48,dt,device,2)
    for fl in range(len(lonvs)):
        set_ptrs[30+4*fl+0]=lonvs[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+3]=sit_l[fl].contiguous().data_ptr()
    return av, g, fi, fo_ref, fo_mg, ce, coo, cei, coi, qcu_U, ref_cl

def run_ref(av, fi, g, fo_ref, ce, coo, cei, coi):
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); return time.perf_counter()-t0

def run_mg(av, fi, g, fo_mg, ce, coo, cei, coi):
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); return time.perf_counter()-t0

if __name__=="__main__":
    print("Cooling GPU 20s..."); time.sleep(20)
    av,g,fi,fo_ref,fo_mg,ce,coo,cei,coi,qcu_U,ref_cl = setup()
    # warmup
    run_ref(av,fi,g,fo_ref,ce,coo,cei,coi); run_mg(av,fi,g,fo_mg,ce,coo,cei,coi)
    ref_times=[]; mg_times=[]
    for i in range(3):
        ref_times.append(run_ref(av,fi,g,fo_ref,ce,coo,cei,coi))
        mg_times.append(run_mg(av,fi,g,fo_mg,ce,coo,cei,coi))
    # correctness
    qcu_mg=tools.poooxyzt2oooxyzt(fo_mg); qcu_ref=tools.poooxyzt2oooxyzt(fo_ref)
    vs=tools.norm(qcu_mg-qcu_ref)/tools.norm(qcu_ref)
    ref_best=min(ref_times); mg_best=min(mg_times)
    print(f"BiStabCG: {[f'{t*1000:.0f}' for t in ref_times]} ms  best={ref_best*1000:.0f}")
    print(f"MG      : {[f'{t*1000:.0f}' for t in mg_times]} ms  best={mg_best*1000:.0f}")
    print(f"speedup (best) = {ref_best/mg_best:.3f}x   vs_ref={vs:.2e}")
