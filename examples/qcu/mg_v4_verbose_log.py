#!/usr/bin/env python3
"""Produce a format-compliant per-iteration log (reference format:
examples/pyqcu/conftest.clover.multigrid-v20260506.log) for the default
lattice {8,16,16,16} with VERBOSE=1. Output -> logs/clover_multigrid.log."""
import sys, os, torch, time
sys.path.insert(0, os.path.expanduser("~/PyQCU/examples/qcu"))
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
_csm = _load("csm", os.path.expanduser("~/PyQCU/examples/qcu/conftest.schur.multigrid.py"))
build_config = _csm.build_config
from mg_nullvec_cache import build_or_load_coarse_ops

open(os.path.expanduser("~/PyQCU/logs/clover_multigrid.log"), "w").close()
Lx,Ly,Lz,Lt=8,16,16,16; MASS=0.05; ATOL=1e-6; DT=define._LAT_C64_
av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,2,[12,48],[2,2,2,2],12,15,1e5,DT)
device=torch.device('cuda'); dt=define.dtype(DT); ls=define.lat_shape(params)
g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device); fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
fo=torch.zeros_like(fi); ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device); cei=torch.zeros_like(ce)
coo=torch.zeros_like(ce); coi=torch.zeros_like(ce)
params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
qcu.applyInitQcu(set_ptrs,params,av); qcu.applyGaussGaugeQcu(g,set_ptrs,params)
params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=0
qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=1
qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)
qcu_U=tools.poooxyzt2oooxyzt(g); ref_cl=dslash.make_clover(qcu_U,kappa=1.0/(2*MASS+8))
op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([1.0/(2*MASS+8)]), support_parity=True, verbose=False)
S = op.matvec_parity
lonv, hnn, hdg, sit = build_or_load_coarse_ops(42, [Lx,Ly,Lz,Lt], 1, 48, 12,
                                               [Lx,Ly,Lz,Lt//2], [4,8,8,4], S, dt, device, 2)
set_ptrs[30]=lonv.contiguous().data_ptr(); set_ptrs[31]=hnn.contiguous().data_ptr()
set_ptrs[32]=hdg.contiguous().data_ptr(); set_ptrs[33]=sit.contiguous().data_ptr()
params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=1
qcu.applyInitQcu(set_ptrs,params,av)
torch.cuda.synchronize(); t0=time.perf_counter()
qcu.applyCloverMultigridQcu(fo, fi, g, ce, coo, cei, coi, set_ptrs, params)
torch.cuda.synchronize()
print(f"VERBOSE RUN: {time.perf_counter()-t0:.3f}s -> logs/clover_multigrid.log")
