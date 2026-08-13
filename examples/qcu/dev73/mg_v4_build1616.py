#!/usr/bin/env python3
"""Build + cache the Schur null vectors / 33-tensor coarse operators for the
DEFAULT lattice {8,16,16,16} (and the 3-level variant).  Exits after caching;
the MG solve itself is run separately so cache-building cost is amortized."""
import torch, os, sys, time
from pyqcu import tools, dslash
import pyqcu.cuda.define as define
from pyqcu.cuda import qcu
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

def build_for(Lx,Ly,Lz,Lt,NUM_LEVELS,DOF_LIST,MG_GRID,DT=define._LAT_C64_,NV_ITERS=2):
    av = build_config(Lx,Ly,Lz,Lt,0.05,1e-6,NUM_LEVELS,DOF_LIST,MG_GRID,10,200,1e4,DT)
    KAPPA=1.0/(2*0.05+8); device=torch.device('cuda'); dt=define.dtype(DT); ls=define.lat_shape(params)
    g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    qcu_U=tools.poooxyzt2oooxyzt(g)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]), support_parity=True, verbose=False)
    S = op.matvec_parity
    t0=time.perf_counter()
    lonvs, hnn_l, hdg_l, sit_l = build_schur_levels(op, S, NUM_LEVELS, DOF_LIST, MG_GRID, [Lx,Ly,Lz,Lt], DOF_LIST[1], dt, device, NV_ITERS)
    print(f"built {NUM_LEVELS}-level cache for {Lx}x{Ly}x{Lz}x{Lt} in {time.perf_counter()-t0:.1f}s")
    for i,(lo,hn,hd,st) in enumerate(zip(lonvs,hnn_l,hdg_l,sit_l)):
        print(f"  level {i+1}: lonv={tuple(lo.shape)} hnn={tuple(hn.shape)} hdg={tuple(hd.shape)} sit={tuple(st.shape)}")

if __name__=="__main__":
    build_for(8,16,16,16,2,[12,48],[2,2,2,2])
    build_for(8,16,16,16,3,[12,48,48],[2,2,2,2])
    print("DONE")
