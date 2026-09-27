#!/usr/bin/env python3
"""Test the Python solver.multigrid reference with support_parity=True vs False."""
import torch
from time import perf_counter
from pyqcu import tools, dslash, solver
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
from pyqcu.cuda import qcu

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
params[define._LAT_X_]=Lx;params[define._LAT_Y_]=Ly;params[define._LAT_Z_]=Lz;params[define._LAT_T_]=Lt
params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
params[define._PARITY_]=0;params[define._NODE_RANK_]=0;params[define._NODE_SIZE_]=1
params[define._DAGGER_]=0;params[define._MAX_ITER_]=500
params[define._DATA_TYPE_]=define._LAT_C64_;params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1
params[define._VERBOSE_]=0;params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0
params[define._MG_NUM_LEVEL_]=2
params[define._MG_LEVEL1_E_]=12;params[define._MG_LEVEL1_X_]=4
params[define._MG_LEVEL1_Y_]=4;params[define._MG_LEVEL1_Z_]=4;params[define._MG_LEVEL1_T_]=8
params[define._MG_LEVEL1_MAX_ITER_]=100;params[define._MG_LEVEL1_DATA_TYPE_]=define._LAT_C64_
params[define._MG_LEVEL1_NUM_RESTART_]=5
av=argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1
av[define._MG_LEVEL1_ATOL_]=ATOL*10.0
device=torch.device('cuda');dt=define.dtype(params[define._DATA_TYPE_]);ls=define.lat_shape(params)
g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
fo_ref=torch.zeros_like(fi)
ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device);cei=torch.zeros_like(ce)
coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)
params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=-1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyGaussGaugeQcu(g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)
qcu_U=tools.poooxyzt2oooxyzt(g);ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
qcu_src=tools.poooxyzt2oooxyzt(fi)
qcu_ref=tools.poooxyzt2oooxyzt(fo_ref)
# Reference BiStabCG
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1
qcu.applyInitQcu(set_ptrs,params,av)
t0=perf_counter();qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)
ref_time=perf_counter()-t0
qcu_ref=tools.poooxyzt2oooxyzt(fo_ref)
ref_res=tools.norm(dslash.give_wilson(qcu_ref,qcu_U,KAPPA,True)+dslash.give_clover(qcu_ref,ref_cl)-qcu_src)/tools.norm(qcu_src)
print(f"BiStabCG ref: {ref_time*1000:.1f}ms res={ref_res:.3e}")

for support_parity in [True, False]:
    try:
        mg = solver.multigrid(
            dtype_list=[dt, dt],
            device_list=[device, device],
            U=qcu_U, clover_term=ref_cl, kappa=KAPPA,
            clover_ee_inv=cei, clover_oo_inv=coi,
            min_size=4, max_level=2, mg_grid_size=[2,2,2,2],
            dof_list=[12,12], tol=ATOL, max_iter=500, num_restart=5,
            support_parity=support_parity, verbose=False)
        mg.init()
        t0=perf_counter()
        x = mg.solve(b=qcu_src, x0=torch.zeros_like(qcu_src))
        mg_time=perf_counter()-t0
        mg_res=tools.norm(dslash.give_wilson(x,qcu_U,KAPPA,True)+dslash.give_clover(x,ref_cl)-qcu_src)/tools.norm(qcu_src)
        mg_vs_ref=tools.norm(x-qcu_ref)/tools.norm(qcu_ref)
        print(f"MG support_parity={support_parity}: {mg_time*1000:.1f}ms res={mg_res:.3e} vs_ref={mg_vs_ref:.3e} iters={len(mg.convergence_history)}")
    except Exception as e:
        print(f"MG support_parity={support_parity} FAILED: {type(e).__name__}: {e}")
