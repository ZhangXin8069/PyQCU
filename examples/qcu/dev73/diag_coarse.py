#!/usr/bin/env python3
"""Direct element-by-element comparison of Python coarse matvec vs C++ coarse dslash."""
import torch, numpy as np
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
params[define._LAT_X_]=Lx;params[define._LAT_Y_]=Ly;params[define._LAT_Z_]=Lz;params[define._LAT_T_]=Lt
params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
params[define._PARITY_]=0;params[define._NODE_RANK_]=0;params[define._NODE_SIZE_]=1
params[define._DAGGER_]=0;params[define._MAX_ITER_]=500
params[define._DATA_TYPE_]=define._LAT_C64_;params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1
params[define._VERBOSE_]=0;params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0
params[define._MG_NUM_LEVEL_]=2
Xc,Yc,Zc,Tc = 4,4,4,8
params[define._MG_LEVEL1_E_]=12;params[define._MG_LEVEL1_X_]=Xc;params[define._MG_LEVEL1_Y_]=Yc
params[define._MG_LEVEL1_Z_]=Zc;params[define._MG_LEVEL1_T_]=Tc
params[define._MG_LEVEL1_MAX_ITER_]=30;params[define._MG_LEVEL1_DATA_TYPE_]=2;params[define._MG_LEVEL1_NUM_RESTART_]=3

av=argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1

device=torch.device('cuda');dt=define.dtype(params[define._DATA_TYPE_]);ls=define.lat_shape(params)
g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device);fi=torch.zeros([2,4,3]+ls,dtype=dt,device=device)
ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device);cei=torch.zeros_like(ce);coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)

params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=-1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyGaussGaugeQcu(g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

qcu_U=tools.poooxyzt2oooxyzt(g);ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op_fine=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=False,verbose=False)
coarse_lat=[Xc,Yc,Zc,Tc]
_nv=torch.randn([12,12,8,8,8,16],dtype=dt,device=device)
_nv=tools.give_null_vecs(null_vecs=_nv,matvec=op_fine.matvec,bistabcg=None,verbose=False)
_lonv=tools.local_orthogonalize(null_vecs=_nv,coarse_lat_size=coarse_lat,verbose=False)
_lonv_flat=_lonv.reshape(12,12,8,8,8,16).contiguous()
coarse_op=dslash.operator(fine_hopping=op_fine.hopping,fine_sitting=op_fine.sitting,local_ortho_null_vecs=_lonv,verbose=False)
hp=torch.zeros([2,4,12,12,Xc,Yc,Zc,Tc],dtype=dt,device=device)
for ward in range(4):
    hp[0,ward]=coarse_op.hopping.M_plus_list[ward].to(dtype=dt,device=device)
    hp[1,ward]=coarse_op.hopping.M_minus_list[ward].to(dtype=dt,device=device)
sp=coarse_op.sitting.M.to(dtype=dt,device=device)

# Simple test: unit vector input, check each element
x_c = torch.randn([12,4,4,4,8], dtype=dt, device=device)

# Python
y_py = coarse_op.matvec(x_c)

# C++
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1
fo_tmp=torch.zeros([2,4,3]+ls,dtype=dt,device=device)
qcu.applyInitQcu(set_ptrs,params,av)
y_cpp = torch.zeros_like(x_c)
qcu.applyMultigridCoarseDslashQcu(y_cpp, x_c, hp, sp, set_ptrs, params)

diff = (y_py - y_cpp).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

# Compare sitting-only term
# Python sitting: just the onsite term
y_sit_py = coarse_op.sitting.matvec(x_c)
print(f"Python sitting norm: {tools.norm(y_sit_py):.6e}")

# C++: zero hopping, only sitting
hp_zero = torch.zeros_like(hp)
y_sit_cpp = torch.zeros_like(x_c)
qcu.applyMultigridCoarseDslashQcu(y_sit_cpp, x_c, hp_zero, sp, set_ptrs, params)
sit_diff = tools.norm(y_sit_py - y_sit_cpp) / tools.norm(y_sit_py)
print(f"Sitting-only C++/Py diff: {sit_diff:.6e}")
print(f"  Py sit norm: {tools.norm(y_sit_py):.6e}")
print(f"  C++ sit norm: {tools.norm(y_sit_cpp):.6e}")

# Compare hopping-only term
sp_zero = torch.zeros_like(sp)
y_hop_py = coarse_op.hopping.matvec(x_c)
y_hop_cpp = torch.zeros_like(x_c)
qcu.applyMultigridCoarseDslashQcu(y_hop_cpp, x_c, hp, sp_zero, set_ptrs, params)
hop_diff = tools.norm(y_hop_py - y_hop_cpp) / tools.norm(y_hop_py)
print(f"Hopping-only C++/Py diff: {hop_diff:.6e}")
print(f"  Py hop norm: {tools.norm(y_hop_py):.6e}")
print(f"  C++ hop norm: {tools.norm(y_hop_cpp):.6e}")

# Check first 10 elements of each
print("\n--- First 10 elements comparison ---")
print(f"{'Idx':>6} {'Py_real':>12} {'Py_imag':>12} {'C++_real':>12} {'C++_imag':>12}")
y_py_f = y_py.flatten()[:20]; y_cpp_f = y_cpp.flatten()[:20]
for i in range(20):
    print(f"{i:6d} {y_py_f[i].real.item():12.6e} {y_py_f[i].imag.item():12.6e} {y_cpp_f[i].real.item():12.6e} {y_cpp_f[i].imag.item():12.6e}")
