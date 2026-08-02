#!/usr/bin/env python3
"""Verify the level-2 coarse operator A_cc = P_2^T A_c P_2 (materialized 33-stencil)
against the operator-free Galerkin, on 8x16x16x16 (3-level setup)."""
import torch, os, sys, time
from pyqcu import tools, dslash
import pyqcu.cuda.define as define
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location("csm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "conftest.schur.multigrid.py"))
_csm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_csm)
build_schur_levels = _csm.build_schur_levels
from mg_stencil_build import apply_stencil

Lx,Ly,Lz,Lt=8,16,16,16; MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
device=torch.device('cuda'); dt=define.dtype(define._LAT_C64_)
# Need a U/clover to build op. Reuse conftest setup path (build_config + gauge).
av = _csm.build_config(Lx,Ly,Lz,Lt,MASS,ATOL,3,[12,48,48],[2,2,2,2],10,200,1e4,define._LAT_C64_)
from pyqcu.cuda.define import params, argv, set_ptrs
from pyqcu.cuda import qcu
ls=define.lat_shape(params)
g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
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
# Build 3 levels (level-2 uses cached)
lonvs,hnn_l,hdg_l,sit_l=build_schur_levels(op,op.matvec_parity,3,[12,48,48],[2,2,2,2],[Lx,Ly,Lz,Lt],48,dt,device,2)
print(f"levels built: {len(lonvs)}")
# Verify level-1 and level-2 A against operator-free
# Level-1: A_c = P1^T S P1
for lvl in [1, 2]:
    E = lonvs[lvl-1].shape[0]; e_prev = lonvs[lvl-1].shape[1]
    # lonv block structure [E,e,Xc,2,Yc,2,Zc,2,Tc,2]
    sh = lonvs[lvl-1].shape
    lat_c = [sh[2], sh[4], sh[6], sh[8]]       # coarse lattice
    lat_f = [sh[2]*sh[3], sh[4]*sh[5], sh[6]*sh[7], sh[8]*sh[9]]  # fine lattice
    print(f"level {lvl}: E={E} e_prev={e_prev} fine={lat_f} coarse={lat_c}")
    v = torch.randn([E]+lat_c, dtype=dt, device=device)
    A_mat = apply_stencil(hnn_l[lvl-1], hdg_l[lvl-1], sit_l[lvl-1], v)
    # operator-free: P^T A_{lvl-1} P v
    f = tools.prolong(local_ortho_null_vecs=lonvs[lvl-1], coarse_vec=v)
    if lvl == 1:
        Af = op.matvec_parity(f)
    else:
        Af = apply_stencil(hnn_l[0], hdg_l[0], sit_l[0], f)  # level-1 A_c
    A_op = tools.restrict(local_ortho_null_vecs=lonvs[lvl-1], fine_vec=Af)
    err = tools.norm(A_mat - A_op)/tools.norm(A_op)
    print(f"  level {lvl}: materialized vs operator-free rel err = {err:.4e}")
