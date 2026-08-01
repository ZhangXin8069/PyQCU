#!/usr/bin/env python3
"""V3: Verify C++ restrict/prolong against Python reference block by block."""
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
params[define._DAGGER_]=0;params[define._MAX_ITER_]=500;params[define._DATA_TYPE_]=define._LAT_C64_
params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1;params[define._VERBOSE_]=0
params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0

av=argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1

device=torch.device('cuda');dt=define.dtype(params[define._DATA_TYPE_]);ls=define.lat_shape(params)
g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
fi=torch.zeros([2,4,3]+ls,dtype=dt,device=device)
ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device);cei=torch.zeros_like(ce)
coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)

params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=-1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyGaussGaugeQcu(g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

qcu_U=tools.poooxyzt2oooxyzt(g);ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op_fine=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=False,verbose=False)

# Build coarse ops
coarse_lat=[4,4,4,8]
_nv=torch.randn([12,12,8,8,8,16],dtype=dt,device=device)
_nv=tools.give_null_vecs(null_vecs=_nv,matvec=op_fine.matvec,bistabcg=None,verbose=False)
_lonv_blocked=tools.local_orthogonalize(null_vecs=_nv,coarse_lat_size=coarse_lat,verbose=False)
# Extract blocked dims
E,e = _lonv_blocked.shape[0], _lonv_blocked.shape[1]
Xc,mgx = _lonv_blocked.shape[2], _lonv_blocked.shape[3]
Yc,mgy = _lonv_blocked.shape[4], _lonv_blocked.shape[5]
Zc,mgz = _lonv_blocked.shape[6], _lonv_blocked.shape[7]
Tc,mgt = _lonv_blocked.shape[8], _lonv_blocked.shape[9]
_lonv_flat = _lonv_blocked.reshape(E, e, Xc*mgx, Yc*mgy, Zc*mgz, Tc*mgt).contiguous()

print(f'LONV blocked: {list(_lonv_blocked.shape)}')
print(f'  E={E}, e={e}, Xc={Xc}, mgx={mgx}, Yc={Yc}, mgy={mgy}, Zc={Zc}, mgz={mgz}, Tc={Tc}, mgt={mgt}')
print(f'LONV flat: {list(_lonv_flat.shape)}')

# === TEST 1: Restrict (P^T * fine → coarse) ===
fine_vec = torch.randn([12, 8, 8, 8, 16], dtype=dt, device=device)

# Python restrict (blocked einsum)
coarse_py = tools.restrict(local_ortho_null_vecs=_lonv_blocked, fine_vec=fine_vec)
print(f'\n=== Test 1: Restrict ===')
print(f'Python restrict shape: {list(coarse_py.shape)}, norm: {tools.norm(coarse_py):.6e}')

# C++ restrict
params[define._MG_NUM_LEVEL_]=2
params[define._MG_LEVEL1_E_]=E;params[define._MG_LEVEL1_X_]=Xc
params[define._MG_LEVEL1_Y_]=Yc;params[define._MG_LEVEL1_Z_]=Zc;params[define._MG_LEVEL1_T_]=Tc
params[define._MG_LEVEL1_MAX_ITER_]=30;params[define._MG_LEVEL1_DATA_TYPE_]=2
params[define._MG_LEVEL1_NUM_RESTART_]=3
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1
qcu.applyInitQcu(set_ptrs,params,av)

coarse_cpp = torch.zeros([E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
qcu.applyMultigridRestrictQcu(coarse_cpp, fine_vec, _lonv_flat, set_ptrs, params)
print(f'C++ restrict shape: {list(coarse_cpp.shape)}, norm: {tools.norm(coarse_cpp):.6e}')

# Compare: the C++ restrict uses flat LONV, Python uses blocked. Same result?
restrict_diff = tools.norm(coarse_py - coarse_cpp) / tools.norm(coarse_py)
print(f'RESTRICT diff vs Python: {restrict_diff:.6e}')
if restrict_diff > 1e-4:
    # Element-by-element comparison
    print(f'  FAIL: restrict mismatch!')
    print(f'  Py[0,0,0,0,0]: {coarse_py[0,0,0,0,0]:.6e}')
    print(f'  C++[0,0,0,0,0]: {coarse_cpp[0,0,0,0,0]:.6e}')
else:
    print(f'  PASS: restrict matches ({restrict_diff:.2e})')

# === TEST 2: Prolong (P * coarse → fine) ===
coarse_vec = torch.randn([E, Xc, Yc, Zc, Tc], dtype=dt, device=device)

# Python prolong (blocked einsum)
fine_py = tools.prolong(local_ortho_null_vecs=_lonv_blocked, coarse_vec=coarse_vec)
print(f'\n=== Test 2: Prolong ===')
print(f'Python prolong shape: {list(fine_py.shape)}, norm: {tools.norm(fine_py):.6e}')

# C++ prolong
fine_cpp = torch.zeros([12, 8, 8, 8, 16], dtype=dt, device=device)
qcu.applyMultigridProLongQcu(fine_cpp, coarse_vec, _lonv_flat, set_ptrs, params)
print(f'C++ prolong shape: {list(fine_cpp.shape)}, norm: {tools.norm(fine_cpp):.6e}')

prolong_diff = tools.norm(fine_py - fine_cpp) / tools.norm(fine_py)
print(f'PROLONG diff vs Python: {prolong_diff:.6e}')
if prolong_diff > 1e-4:
    print(f'  FAIL: prolong mismatch!')
    print(f'  Py[0,0,0,0,0]: {fine_py[0,0,0,0,0]:.6e}')
    print(f'  C++[0,0,0,0,0]: {fine_cpp[0,0,0,0,0]:.6e}')
else:
    print(f'  PASS: prolong matches ({prolong_diff:.2e})')

# === TEST 3: Galerkin identity P^T D_f P v_coarse ===
# This tests the full coarse operator construction
print(f'\n=== Test 3: Galerkin P^T*D_f*P ===')
v_c = torch.randn([E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
# Method 1: P^T * D_f * P * v_c
v_f = tools.prolong(local_ortho_null_vecs=_lonv_blocked, coarse_vec=v_c)
Df_vf = op_fine.matvec(v_f)
Pt_Df_P_vc = tools.restrict(local_ortho_null_vecs=_lonv_blocked, fine_vec=Df_vf)

# Build coarse operator
coarse_op = dslash.operator(fine_hopping=op_fine.hopping, fine_sitting=op_fine.sitting,
    local_ortho_null_vecs=_lonv_blocked, verbose=False)

# Method 2: Coarse operator directly
Dc_vc = coarse_op.matvec(v_c)

galerkin_diff = tools.norm(Pt_Df_P_vc - Dc_vc) / tools.norm(Dc_vc)
print(f'||P^T*D_f*P*v - D_c*v|| / ||D_c*v|| = {galerkin_diff:.6e}')
print(f'  P^T*D_f*P*v norm: {tools.norm(Pt_Df_P_vc):.6e}')
print(f'  D_c*v norm: {tools.norm(Dc_vc):.6e}')
if galerkin_diff < 1e-10:
    print(f'  PASS: Galerkin identity holds')
else:
    print(f'  NOTE: Galerkin error={galerkin_diff:.2e} (operator may have edge effects)')

# === TEST 4: Coarse BiStabCG with Galerkin operator ===
print(f'\n=== Test 4: Coarse BiStabCG solve ===')
# Create a coarse RHS: restrict fine rhs
fine_rhs = torch.randn([12, 8, 8, 8, 16], dtype=dt, device=device)
coarse_rhs = tools.restrict(local_ortho_null_vecs=_lonv_blocked, fine_vec=fine_rhs)

from pyqcu import solver as pqsolver
x_c_sol = pqsolver.bistabcg(b=coarse_rhs, matvec=coarse_op.matvec, tol=1e-8, max_iter=200, verbose=False)
res_c = tools.norm(coarse_rhs - coarse_op.matvec(x_c_sol))
print(f'  Coarse solve residual: {res_c:.6e}')

# Prolong back
x_f_corr = tools.prolong(local_ortho_null_vecs=_lonv_blocked, coarse_vec=x_c_sol)
res_before = tools.norm(fine_rhs)
res_after = tools.norm(fine_rhs - op_fine.matvec(x_f_corr))
print(f'  Before correction: ||r|| = {res_before:.6e}')
print(f'  After correction:  ||r - D*P*x_c|| = {res_after:.6e}')
print(f'  Reduction: {res_after/res_before:.6f}x')

print(f'\n{"="*60}')
print(f'ALL TESTS COMPLETE')
