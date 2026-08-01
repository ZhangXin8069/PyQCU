#!/usr/bin/env python3
"""Quick single-point MG test"""
import torch, os, sys
from time import perf_counter
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

Lx,Ly,Lz,Lt = 8,8,8,16
MASS = 0.05; ATOL = 1e-6; KAPPA = 1.0/(2*MASS+8)

sys.stdout = open('/root/PyQCU/logs/quick_test.log', 'w'); sys.stderr = sys.stdout
print("=== Quick MG Test ===")

for NUM_LEVELS in [1, 2]:
    print(f"\n--- NUM_LEVELS={NUM_LEVELS} ---")

    params[define._LAT_X_]=Lx; params[define._LAT_Y_]=Ly; params[define._LAT_Z_]=Lz; params[define._LAT_T_]=Lt
    params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
    params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
    params[define._PARITY_]=0;params[define._NODE_RANK_]=0;params[define._NODE_SIZE_]=1
    params[define._DAGGER_]=0;params[define._MAX_ITER_]=500
    params[define._DATA_TYPE_]=define._LAT_C64_
    params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1
    params[define._VERBOSE_]=1;params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0
    params[define._MG_NUM_LEVEL_]=NUM_LEVELS
    if NUM_LEVELS>=2:
        params[define._MG_LEVEL1_E_]=12; params[define._MG_LEVEL1_X_]=4
        params[define._MG_LEVEL1_Y_]=4; params[define._MG_LEVEL1_Z_]=4; params[define._MG_LEVEL1_T_]=8  # half t-dim
        params[define._MG_LEVEL1_MAX_ITER_]=30; params[define._MG_LEVEL1_DATA_TYPE_]=2; params[define._MG_LEVEL1_NUM_RESTART_]=3

    av=argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
    av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1
    if NUM_LEVELS>=2: av[define._MG_LEVEL1_ATOL_]=ATOL*0.1  # initial coarse tol (overridden by relative)

    device=torch.device('cuda'); dt=define.dtype(params[define._DATA_TYPE_]); ls=define.lat_shape(params)
    g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
    fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
    fo_ref=torch.zeros_like(fi);fo_mg=torch.zeros_like(fi)
    ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device);cei=torch.zeros_like(ce);coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)

    print("Setup gauge+clover...")
    params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs,params,av);qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
    params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

    # Build coarse ops for 2L
    qcu_U=tools.poooxyzt2oooxyzt(g)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)

    if NUM_LEVELS >= 2:
        print("Build coarse operators...")
        op_fine = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=KAPPA, support_parity=False, verbose=False)
        _null_vecs = torch.randn([12,12,8,8,8,16], dtype=dt, device=device)
        coarse_lat = [4,4,4,8]  # coarse lattice (half t-dim for more aggressive coarsening)
        _null_vecs = tools.give_null_vecs(null_vecs=_null_vecs, matvec=op_fine.matvec, bistabcg=None, verbose=True)
        _lonv = tools.local_orthogonalize(null_vecs=_null_vecs, coarse_lat_size=coarse_lat, verbose=True)
        Xc,Yc,Zc,Tc = coarse_lat
        mgx,mgy,mgz,mgt = 2,2,2,2  # coarsening factors
        _lonv_flat = _lonv.reshape(12,12,Xc*mgx,Yc*mgy,Zc*mgz,Tc*mgt).contiguous()

        coarse_op = dslash.operator(fine_hopping=op_fine.hopping, fine_sitting=op_fine.sitting, local_ortho_null_vecs=_lonv, verbose=True)
        hp = torch.zeros([2,4,12,12,Xc,Yc,Zc,Tc], dtype=dt, device=device)
        for ward in range(4):
            hp[0,ward] = coarse_op.hopping.M_plus_list[ward].to(dtype=dt, device=device)
            hp[1,ward] = coarse_op.hopping.M_minus_list[ward].to(dtype=dt, device=device)
        sp = coarse_op.sitting.M.to(dtype=dt, device=device)

        set_ptrs[10] = _lonv_flat.contiguous().data_ptr()
        set_ptrs[11] = hp.contiguous().data_ptr()
        set_ptrs[12] = sp.contiguous().data_ptr()
        print(f"Coarse ops set: lonv={set_ptrs[10]:#x} hop={set_ptrs[11]:#x} sit={set_ptrs[12]:#x}")

    # Ref BiStabCG
    print("Ref BiStabCG...")
    params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1;params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    t0=perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)
    ref_time=perf_counter()-t0
    print(f"  Ref: {ref_time:.4f}s")

    # MG
    print("MG solver...")
    params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1;params[define._VERBOSE_]=1
    qcu.applyInitQcu(set_ptrs,params,av)
    t0=perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg,fi,g,ce,coo,cei,coi,set_ptrs,params)
    mg_time=perf_counter()-t0

    qcu_mg=tools.poooxyzt2oooxyzt(fo_mg); qcu_ref=tools.poooxyzt2oooxyzt(fo_ref); qcu_src=tools.poooxyzt2oooxyzt(fi)
    mg_vs_ref=tools.norm(qcu_mg-qcu_ref)/tools.norm(qcu_ref)
    speedup=ref_time/mg_time if mg_time>0 else 0
    print(f'RESULT_{NUM_LEVELS}L: ref={ref_time:.4f}s mg={mg_time:.4f}s vs_ref={mg_vs_ref:.2e} speedup={speedup:.2f}x')

print("\n=== DONE ===")
