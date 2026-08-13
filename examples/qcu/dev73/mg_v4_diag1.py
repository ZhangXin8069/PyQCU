#!/usr/bin/env python3
"""Diagnostic: run config 1 (8x8x8x16, 2L, E=48) through the Schur MG with
full visible output to catch the error swallowed by the conftest."""
import torch, os, sys, traceback
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
build_config = _csm.build_config
build_schur_levels = _csm.build_schur_levels

def main():
    Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; NUM_LEVELS=2
    DOF_LIST=[12,48]; MG_GRID=[2,2,2,2]; NUM_RESTART=10
    COARSE_MAX_ITER=200; COARSE_TOL_FACTOR=1e4; DT=define._LAT_C64_; NV_ITERS=2
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,NUM_LEVELS,DOF_LIST,MG_GRID,NUM_RESTART,
                      COARSE_MAX_ITER,COARSE_TOL_FACTOR,DT)
    KAPPA=1.0/(2*MASS+8)
    device=torch.device('cuda'); dt=define.dtype(DT); ls=define.lat_shape(params)
    print("lat_shape:", ls)
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
    print("gauge+clover done")

    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); ref_time=time.perf_counter()-t0
    print(f"BiStabCG ref: {ref_time*1000:.1f} ms")

    qcu_U=tools.poooxyzt2oooxyzt(g)
    qcu_src=tools.poooxyzt2oooxyzt(fi)
    qcu_ref=tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)

    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity
    print("building schur levels...")
    lonvs, hnn_l, hdg_l, sit_l = build_schur_levels(op, S, NUM_LEVELS, DOF_LIST, MG_GRID,
                                                    [Lx,Ly,Lz,Lt], DOF_LIST[1], dt, device, NV_ITERS)
    for fl in range(len(lonvs)):
        set_ptrs[30+4*fl+0]=lonvs[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+3]=sit_l[fl].contiguous().data_ptr()
    print("set_ptrs wired, running MG...")
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); mg_time=time.perf_counter()-t0
    print(f"MG: {mg_time*1000:.1f} ms")

    qcu_mg=tools.poooxyzt2oooxyzt(fo_mg)
    mg_res=tools.norm(dslash.give_wilson(qcu_mg,qcu_U,KAPPA,True)+
                      dslash.give_clover(qcu_mg,ref_cl)-qcu_src)/tools.norm(qcu_src)
    mg_vs_ref=tools.norm(qcu_mg-qcu_ref)/tools.norm(qcu_ref)
    print(f"MG res={mg_res:.3e} vs_ref={mg_vs_ref:.3e} speedup={ref_time/mg_time:.3f}x")

if __name__=="__main__":
    import time
    try:
        main()
    except Exception:
        traceback.print_exc()
