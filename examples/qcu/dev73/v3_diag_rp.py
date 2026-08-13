#!/usr/bin/env python3
"""V3: Verify C++ restrict/prolong after fixing params (e and Tf)."""
import torch
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
params[define._LAT_X_]=Lx;params[define._LAT_Y_]=Ly;params[define._LAT_Z_]=Lz;params[define._LAT_T_]=Lt
params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
params[define._PARITY_]=0;params[define._NODE_RANK_]=0;params[define._NODE_SIZE_]=1
params[define._DAGGER_]=0;params[define._MAX_ITER_]=500;params[define._DATA_TYPE_]=define._LAT_C64_
params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1;params[define._VERBOSE_]=0
params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0
av=argv.to(dtype=torch.float32)
av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1

device=torch.device('cuda');dt_c64=torch.complex64;ls=define.lat_shape(params)
g=torch.zeros((2,3,3,4)+tuple(ls),dtype=dt_c64,device=device)
ce=torch.zeros((4,3,4,3)+tuple(ls),dtype=dt_c64,device=device);cei=torch.zeros_like(ce)
coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)

params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=-1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyGaussGaugeQcu(g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

qcu_U=tools.poooxyzt2oooxyzt(g);ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op_fine=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=False,verbose=False)
coarse_lat=[4,4,4,8]
_nv=torch.randn((12,12,8,8,8,16),dtype=dt_c64,device=device)
_nv=tools.give_null_vecs(null_vecs=_nv,matvec=op_fine.matvec,bistabcg=None,verbose=False)
_lonv_b=tools.local_orthogonalize(null_vecs=_nv,coarse_lat_size=coarse_lat,verbose=False)
E,e=_lonv_b.shape[0],_lonv_b.shape[1]
Xc,mgx=_lonv_b.shape[2],_lonv_b.shape[3]
Yc,mgy=_lonv_b.shape[4],_lonv_b.shape[5]
Zc,mgz=_lonv_b.shape[6],_lonv_b.shape[7]
Tc,mgt=_lonv_b.shape[8],_lonv_b.shape[9]
Xf,Yf,Zf,Tf=Xc*mgx,Yc*mgy,Zc*mgz,Tc*mgt
_lonv_f=_lonv_b.reshape(E,e,Xf,Yf,Zf,Tf).contiguous()

# === Setup C++ params (FIXED: e=12 for fine DOF, LAT_T=16 for full T) ===
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1
# CRITICAL FIXES for restrict/prolong:
params[define._LAT_T_] = Tf      # full T (16), not halved T (8)
params[define._LAT_XYZT_] = Xf*Yf*Zf*Tf  # recalculate volume
params[define._MG_NUM_LEVEL_] = e  # fine DOF (12), not num_levels (2)
params[define._MG_LEVEL1_E_]=E;params[define._MG_LEVEL1_X_]=Xc
params[define._MG_LEVEL1_Y_]=Yc;params[define._MG_LEVEL1_Z_]=Zc;params[define._MG_LEVEL1_T_]=Tc
params[define._MG_LEVEL1_MAX_ITER_]=30;params[define._MG_LEVEL1_DATA_TYPE_]=2;params[define._MG_LEVEL1_NUM_RESTART_]=3
qcu.applyInitQcu(set_ptrs,params,av)

# === Test 1: Restrict with unit vector ===
fine_unit = torch.zeros((12,8,8,8,16),dtype=dt_c64,device=device)
fine_unit[0,0,0,0,0] = torch.tensor(1+0j,dtype=dt_c64,device=device)
coarse_py = tools.restrict(local_ortho_null_vecs=_lonv_b,fine_vec=fine_unit)
coarse_cpp = torch.zeros((E,Xc,Yc,Zc,Tc),dtype=dt_c64,device=device)
qcu.applyMultigridRestrictQcu(coarse_cpp,fine_unit,_lonv_f,set_ptrs,params)
restrict_diff = tools.norm(coarse_py-coarse_cpp)/tools.norm(coarse_py)
print(f'RESTRICT diff vs Python: {restrict_diff:.6e}')

# === Test 2: Prolong with unit vector ===
coarse_unit=torch.zeros((E,Xc,Yc,Zc,Tc),dtype=dt_c64,device=device)
coarse_unit[0,0,0,0,0]=torch.tensor(1+0j,dtype=dt_c64,device=device)
fine_py=tools.prolong(local_ortho_null_vecs=_lonv_b,coarse_vec=coarse_unit)
fine_cpp=torch.zeros((12,8,8,8,16),dtype=dt_c64,device=device)
qcu.applyMultigridProLongQcu(fine_cpp,coarse_unit,_lonv_f,set_ptrs,params)
prolong_diff = tools.norm(fine_py-fine_cpp)/tools.norm(fine_py)
print(f'PROLONG diff vs Python: {prolong_diff:.6e}')

# === Test 3: Random vector ===
print(f'\n=== Random vector test ===')
fine_rand = torch.randn((12,8,8,8,16),dtype=dt_c64,device=device)
coarse_py2 = tools.restrict(local_ortho_null_vecs=_lonv_b,fine_vec=fine_rand)
coarse_cpp2 = torch.zeros_like(coarse_py2)
qcu.applyMultigridRestrictQcu(coarse_cpp2,fine_rand,_lonv_f,set_ptrs,params)
diff_r2 = tools.norm(coarse_py2-coarse_cpp2)/tools.norm(coarse_py2)
print(f'Random restrict diff: {diff_r2:.6e}')
if diff_r2 < 1e-6:
    print(f'  PASS: C++ restrict matches Python!')
else:
    print(f'  FAIL: difference={diff_r2:.2e}')

coarse_rand = torch.randn((E,Xc,Yc,Zc,Tc),dtype=dt_c64,device=device)
fine_py2 = tools.prolong(local_ortho_null_vecs=_lonv_b,coarse_vec=coarse_rand)
fine_cpp2 = torch.zeros_like(fine_py2)
qcu.applyMultigridProLongQcu(fine_cpp2,coarse_rand,_lonv_f,set_ptrs,params)
diff_p2 = tools.norm(fine_py2-fine_cpp2)/tools.norm(fine_py2)
print(f'Random prolong diff: {diff_p2:.6e}')
if diff_p2 < 1e-6:
    print(f'  PASS: C++ prolong matches Python!')
else:
    print(f'  FAIL: difference={diff_p2:.2e}')

print(f'\nDone.')
