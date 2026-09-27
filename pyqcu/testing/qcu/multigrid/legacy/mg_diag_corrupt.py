#!/usr/bin/env python3
"""Diagnostic: does the C++ 2L MG corrupt the gauge/clover/source GPU buffers?

Compares norm(g), norm(fi), norm(clover) before and after applyCloverMultigridQcu,
and computes the Python residual using BOTH a snapshot (taken before MG) and a
fresh conversion (taken after MG).
"""
import torch, os
from time import perf_counter
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

Lx,Ly,Lz,Lt = 8,8,8,16
MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
NUM_LEVELS=2; DOF_LIST=[12,24]; MG_GRID=[2,2,2,2]; NUM_RESTART=5
DT=define._LAT_C64_

params[define._LAT_X_]=Lx; params[define._LAT_Y_]=Ly
params[define._LAT_Z_]=Lz; params[define._LAT_T_]=Lt
params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
params[define._PARITY_]=0; params[define._NODE_RANK_]=0; params[define._NODE_SIZE_]=1
params[define._DAGGER_]=0; params[define._MAX_ITER_]=1000
params[define._DATA_TYPE_]=DT
params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=1
params[define._VERBOSE_]=0; params[define._SEED_]=42; params[define._TEST_IN_CPU_]=0
params[define._MG_NUM_LEVEL_]=NUM_LEVELS
params[define._MG_LEVEL1_E_]=DOF_LIST[1]
params[define._MG_LEVEL1_X_]=Lx//MG_GRID[0]
params[define._MG_LEVEL1_Y_]=Ly//MG_GRID[1]
params[define._MG_LEVEL1_Z_]=Lz//MG_GRID[2]
params[define._MG_LEVEL1_T_]=Lt//MG_GRID[3]
params[define._MG_LEVEL1_MAX_ITER_]=100
params[define._MG_LEVEL1_DATA_TYPE_]=DT
params[define._MG_LEVEL1_NUM_RESTART_]=NUM_RESTART
av = argv.to(dtype=define.dtype(DT).to_real())
av[define._MASS_]=MASS; av[define._ATOL_]=ATOL; av[define._SIGMA_]=0.1
av[define._MG_LEVEL1_ATOL_]=ATOL*10.0

device=torch.device('cuda'); dt=define.dtype(DT); ls=define.lat_shape(params)
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

# Reference BiStabCG
params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
qcu.applyInitQcu(set_ptrs,params,av)
qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)

# Snapshots BEFORE MG
g_snap = g.clone()
fi_snap = fi.clone()
ce_snap = ce.clone(); coo_snap = coo.clone(); cei_snap = cei.clone(); coi_snap = coi.clone()

qcu_U = tools.poooxyzt2oooxyzt(g)
qcu_src = tools.poooxyzt2oooxyzt(fi)
qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, KAPPA, True) +
                     dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)
print(f"[before MG] ref_res={ref_res:.4e}  norm(g)={tools.norm(g):.6f} norm(fi)={tools.norm(fi):.6f}")

# Build coarse ops (copies from qcu_U which may be a view of g — check identity)
print(f"qcu_U is view of g? {qcu_U.data_ptr()==g.data_ptr() or qcu_U.storage().data_ptr()==g.storage().data_ptr()}")
print(f"qcu_U shape {tuple(qcu_U.shape)}, g shape {tuple(g.shape)}")

op_fine = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=KAPPA, support_parity=False, verbose=False)
lat_sizes = [[Lx,Ly,Lz,Lt],[Lx//MG_GRID[0],Ly//MG_GRID[1],Lz//MG_GRID[2],Lt//MG_GRID[3]]]
lonv_list=[]; hop_packed_list=[]; sit_packed_list=[]
for i in range(1,NUM_LEVELS):
    dof_fine=DOF_LIST[i-1]; dof_coarse=DOF_LIST[i]
    lat_fine=lat_sizes[i-1]; lat_coarse=lat_sizes[i]
    _null_vecs=torch.randn([dof_coarse,dof_fine]+lat_fine,dtype=dt,device=device)
    _null_vecs=tools.give_null_vecs(null_vecs=_null_vecs, matvec=op_fine.matvec, bistabcg=None, verbose=False)
    _lonv=tools.local_orthogonalize(null_vecs=_null_vecs, coarse_lat_size=lat_coarse, verbose=False)
    E_lonv=_lonv.shape[0]; e_lonv=_lonv.shape[1]
    Xc=_lonv.shape[2]; mgx=_lonv.shape[3]; Yc=_lonv.shape[4]; mgy=_lonv.shape[5]
    Zc=_lonv.shape[6]; mgz=_lonv.shape[7]; Tc=_lonv.shape[8]; mgt=_lonv.shape[9]
    _lonv_flat=_lonv.reshape(E_lonv,e_lonv,Xc*mgx,Yc*mgy,Zc*mgz,Tc*mgt).contiguous()
    lonv_list.append(_lonv_flat)
    coarse_op=dslash.operator(fine_hopping=op_fine.hopping, fine_sitting=op_fine.sitting,
                              local_ortho_null_vecs=_lonv, verbose=False)
    E=dof_coarse; Xc,Yc,Zc,Tc=lat_coarse
    hp=torch.zeros([2,4,E,E,Xc,Yc,Zc,Tc],dtype=dt,device=device)
    for ward in range(4):
        hp[0,ward]=coarse_op.hopping.M_plus_list[ward].to(dtype=dt,device=device)
        hp[1,ward]=coarse_op.hopping.M_minus_list[ward].to(dtype=dt,device=device)
    hop_packed_list.append(hp)
    sit_id=torch.zeros([E,E,Xc,Yc,Zc,Tc],dtype=dt,device=device)
    for e_i in range(E): sit_id[e_i,e_i]=1.0
    sit_packed_list.append(sit_id)
for fl in range(len(lonv_list)):
    set_ptrs[10+3*fl+0]=lonv_list[fl].contiguous().data_ptr()
    set_ptrs[10+3*fl+1]=hop_packed_list[fl].contiguous().data_ptr()
    set_ptrs[10+3*fl+2]=sit_packed_list[fl].contiguous().data_ptr()

# Verify null vec correctness against Python reference BEFORE running MG
# (check restrict/prolong/nv self-consistency)
print(f"\n[null-vec check] lonv shape={tuple(lonv_list[0].shape)}")

# Run C++ MG
params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
qcu.applyInitQcu(set_ptrs,params,av)
torch.cuda.synchronize(); t0=perf_counter()
qcu.applyCloverMultigridQcu(fo_mg,fi,g,ce,coo,cei,coi,set_ptrs,params)
torch.cuda.synchronize(); mg_time=perf_counter()-t0
print(f"[after MG]  norm(g)={tools.norm(g):.6f} norm(fi)={tools.norm(fi):.6f}")
print(f"[after MG]  g changed? {not torch.allclose(g, g_snap)}  fi changed? {not torch.allclose(fi, fi_snap)}")
print(f"[after MG]  ce changed? {not torch.allclose(ce, ce_snap)} coo? {not torch.allclose(coo, coo_snap)}")
print(f"[after MG]  cei changed? {not torch.allclose(cei, cei_snap)} coi? {not torch.allclose(coi, coi_snap)}")

# Residual using SNAPSHOT of operator state (pre-MG)
qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
U_snap = tools.poooxyzt2oooxyzt(g_snap)
src_snap = tools.poooxyzt2oooxyzt(fi_snap)
cl_snap = dslash.make_clover(U_snap, kappa=KAPPA)
res_snap = tools.norm(dslash.give_wilson(qcu_mg, U_snap, KAPPA, True) +
                      dslash.give_clover(qcu_mg, cl_snap) - src_snap)/tools.norm(src_snap)

# Residual using CURRENT operator state (post-MG)
qcu_U2 = tools.poooxyzt2oooxyzt(g)
qcu_src2 = tools.poooxyzt2oooxyzt(fi)
cl2 = dslash.make_clover(qcu_U2, kappa=KAPPA)
res_now = tools.norm(dslash.give_wilson(qcu_mg, qcu_U2, KAPPA, True) +
                     dslash.give_clover(qcu_mg, cl2) - qcu_src2)/tools.norm(qcu_src2)

vs_ref = tools.norm(qcu_mg - qcu_ref)/tools.norm(qcu_ref)
print(f"\nMG time={mg_time*1000:.1f}ms")
print(f"residual (pre-MG snapshot op) = {res_snap:.4e}")
print(f"residual (post-MG op)         = {res_now:.4e}")
print(f"vs_ref = {vs_ref:.4e}")
