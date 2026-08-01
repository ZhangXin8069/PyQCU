#!/usr/bin/env python3
"""Verify C++ coarse dslash against Python reference, and check restrict/prolong consistency."""
import torch
from time import perf_counter
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
params[define._MG_LEVEL1_E_]=12;params[define._MG_LEVEL1_X_]=4;params[define._MG_LEVEL1_Y_]=4
params[define._MG_LEVEL1_Z_]=4;params[define._MG_LEVEL1_T_]=8
params[define._MG_LEVEL1_MAX_ITER_]=30;params[define._MG_LEVEL1_DATA_TYPE_]=2;params[define._MG_LEVEL1_NUM_RESTART_]=3

av=argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1;av[define._MG_LEVEL1_ATOL_]=0

device=torch.device('cuda');dt=define.dtype(params[define._DATA_TYPE_]);ls=define.lat_shape(params)
g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
fi=torch.zeros([2,4,3]+ls,dtype=dt,device=device)
ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device);cei=torch.zeros_like(ce);coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)

params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=-1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyGaussGaugeQcu(g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

qcu_U=tools.poooxyzt2oooxyzt(g);ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op_fine=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=False,verbose=False)

# Build coarse operators
coarse_lat=[4,4,4,8]
_nv=torch.randn([12,12,8,8,8,16],dtype=dt,device=device)
_nv=tools.give_null_vecs(null_vecs=_nv,matvec=op_fine.matvec,bistabcg=None,verbose=False)
_lonv=tools.local_orthogonalize(null_vecs=_nv,coarse_lat_size=coarse_lat,verbose=False)
_lonv_flat=_lonv.reshape(12,12,8,8,8,16).contiguous()
coarse_op=dslash.operator(fine_hopping=op_fine.hopping,fine_sitting=op_fine.sitting,local_ortho_null_vecs=_lonv,verbose=False)
Xc,Yc,Zc,Tc=coarse_lat
hp=torch.zeros([2,4,12,12,Xc,Yc,Zc,Tc],dtype=dt,device=device)
for ward in range(4):
    hp[0,ward]=coarse_op.hopping.M_plus_list[ward].to(dtype=dt,device=device)
    hp[1,ward]=coarse_op.hopping.M_minus_list[ward].to(dtype=dt,device=device)
sp=coarse_op.sitting.M.to(dtype=dt,device=device)

print(f"Coarse op shapes: hop={list(hp.shape)}, sit={list(sp.shape)}")

# Test 1: Compare coarse matvec (Python) vs coarse dslash (C++)
x_c = torch.randn([12,4,4,4,8], dtype=dt, device=device)
y_c_py = coarse_op.matvec(x_c)
print(f"Python matvec norm: {tools.norm(y_c_py):.6e}")

# C++ coarse dslash: use applyMultigridCoarseDslashQcu
# Requires params set correctly
params[define._MG_LEVEL1_X_]=Xc;params[define._MG_LEVEL1_Y_]=Yc;params[define._MG_LEVEL1_Z_]=Zc;params[define._MG_LEVEL1_T_]=Tc
params[define._MG_LEVEL1_E_]=12

# Initialize C++ context for MG
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1
fo_tmp = torch.zeros_like(fi)
qcu.applyInitQcu(set_ptrs,params,av)

# Call coarse dslash
y_c_cpp = torch.zeros_like(x_c)
try:
    qcu.applyMultigridCoarseDslashQcu(y_c_cpp, x_c, hp, sp, set_ptrs, params)
    print(f"C++ coarse dslash norm: {tools.norm(y_c_cpp):.6e}")
    diff = tools.norm(y_c_py - y_c_cpp) / tools.norm(y_c_py)
    print(f"Test 1 - Coarse dslash vs matvec: |C++-Py|/|Py| = {diff:.6e}")
except Exception as e:
    print(f"C++ coarse dslash FAILED: {e}")

# Test 2: Verify Galerkin condition P^T * D_fine * P ≈ D_coarse
# Take a random coarse vector e_c, apply coarse dslash, compare with P^T(D_fine(P*e_c))
e_c = torch.randn([12,4,4,4,8], dtype=dt, device=device)
# Prolong: fine = P * e_c
e_f_py = tools.prolong(local_ortho_null_vecs=_lonv, coarse_vec=e_c)  # [12,8,8,8,16]
# Apply fine matvec
Df_Pe = op_fine.matvec(e_f_py)  # [12,8,8,8,16]
# Restrict
Pt_Df_Pe = tools.restrict(local_ortho_null_vecs=_lonv, fine_vec=Df_Pe)  # [12,4,4,4,8]

# Compare with coarse matvec
Dc_e = coarse_op.matvec(e_c)  # [12,4,4,4,8]
galerkin_diff = tools.norm(Pt_Df_Pe - Dc_e) / tools.norm(Dc_e)
print(f"Test 2 - Galerkin check: |P^T*D_f*P*e - D_c*e|/|D_c*e| = {galerkin_diff:.6e}")
print(f"  P^T*D_f*P*e norm: {tools.norm(Pt_Df_Pe):.6e}")
print(f"  D_c*e norm: {tools.norm(Dc_e):.6e}")

# Test 3: Check restrict(RHS) for V-cycle
# Compute full residual for x=0: r_full = b_full (since x=0)
# b_full comes from the input source
b_full = qcu_src = tools.poooxyzt2oooxyzt(fi)  # but fi is zeros!
# Use random b_full instead
b_full = torch.randn([12,8,8,8,16], dtype=dt, device=device)
r_c_py = tools.restrict(local_ortho_null_vecs=_lonv, fine_vec=b_full)
print(f"Test 3 - Restrict RHS norm: {tools.norm(r_c_py):.6e}")

# Solve coarse system: D_c * x_c = r_c
from pyqcu import solver as pqsolver
x_c_sol = pqsolver.bistabcg(b=r_c_py.flatten().reshape(12,4,4,4,8), matvec=coarse_op.matvec, tol=1e-6, max_iter=200, verbose=False)
x_c_sol = x_c_sol.reshape(12,4,4,4,8)
r_c_check = r_c_py - coarse_op.matvec(x_c_sol)
print(f"  Coarse solve residual: {tools.norm(r_c_check):.6e}")

# Prolong
e_f_sol = tools.prolong(local_ortho_null_vecs=_lonv, coarse_vec=x_c_sol)
print(f"  Prolonged correction norm: {tools.norm(e_f_sol):.6e}")

# Check if correction helps: ||b - D*(P*x_c)|| vs ||b||
res_before = tools.norm(b_full)
D_Px = op_fine.matvec(e_f_sol)
res_after = tools.norm(b_full - D_Px)
print(f"Test 4 - Correction effectiveness:")
print(f"  ||b||       = {res_before:.6e}")
print(f"  ||b-D*P*x_c|| = {res_after:.6e}")
print(f"  Reduction: {res_after/res_before:.4f}x")

print("\n=== ALL TESTS DONE ===")
