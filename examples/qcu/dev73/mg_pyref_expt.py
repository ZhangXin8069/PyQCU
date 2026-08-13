#!/usr/bin/env python3
"""GROUND TRUTH experiment: Python _multigrid.py on GPU.

Measures iteration counts and wall-clock for:
  (a) plain parity-preconditioned BiStabCG (C++ applyCloverBistabCgQcu)
  (b) Python multigrid support_parity=False (full operator), with_cuda_qcu=True
  (c) Python multigrid support_parity=True  (Schur complement), with_cuda_qcu=True

This tells us the ACHIEVABLE iteration count and whether parity-preconditioned
level-0 MG can beat plain parity BiStabCG. This is the reference the C++ MG must
match ("算法参考 _multigrid.py, 最终效果完全一致").

Usage: source ./env.sh && python examples/qcu/mg_pyref_expt.py
"""
import torch, os, sys, re, io
from time import perf_counter
from contextlib import redirect_stdout
from pyqcu import tools, dslash, solver
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

def setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL,DT=define._LAT_C64_):
    """Generate gauge+clover via C++ backend, return Python full-layout tensors."""
    KAPPA = 1.0/(2*MASS+8)
    params[define._LAT_X_]=Lx; params[define._LAT_Y_]=Ly
    params[define._LAT_Z_]=Lz; params[define._LAT_T_]=Lt
    params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
    params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
    params[define._PARITY_]=0; params[define._NODE_RANK_]=0; params[define._NODE_SIZE_]=1
    params[define._DAGGER_]=0; params[define._MAX_ITER_]=1000
    params[define._DATA_TYPE_]=DT
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=1
    params[define._VERBOSE_]=0; params[define._SEED_]=42; params[define._TEST_IN_CPU_]=0
    params[define._MG_NUM_LEVEL_]=1
    av = argv.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_]=MASS; av[define._ATOL_]=ATOL; av[define._SIGMA_]=0.1
    device=torch.device('cuda'); dt=define.dtype(DT); ls=define.lat_shape(params)
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
    U_full = tools.poooxyzt2oooxyzt(g)          # [3,3,4,Lx,Ly,Lz,Lt]
    b_full = tools.poooxyzt2oooxyzt(fi)         # [4,3,Lx,Ly,Lz,Lt]
    clover = dslash.make_clover(U_full, kappa=KAPPA)
    return U_full, b_full, clover, KAPPA, av, (g,fi,ce,coo,cei,coi)

def run_bistabcg_ref(fi, g, ce, coo, cei, coi, av):
    """C++ parity-preconditioned BiStabCG: returns (fo, time)."""
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    fo=torch.zeros_like(fi)
    torch.cuda.synchronize(); t0=perf_counter()
    qcu.applyCloverBistabCgQcu(fo,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); dt=perf_counter()-t0
    return fo, dt

def run_py_mg(U_full, b_full, clover, KAPPA, support_parity, num_restart=5, max_level=2,
              tol=1e-6, dt=torch.complex64, device=torch.device('cuda')):
    """Python multigrid. Returns (x, stats)."""
    kwargs = dict(dtype_list=[dt]*10, device_list=[device]*10,
                  U=U_full, clover_term=clover, kappa=torch.Tensor([KAPPA]),
                  tol=tol, max_iter=1000, max_level=max_level, num_restart=num_restart,
                  support_parity=support_parity, verbose=False)
    # with_cuda_qcu requires clover_ee_inv/oo_inv; only support_parity=True
    # produces them (and only that path is compatible with the C++ Schur dslash).
    if support_parity:
        op0 = dslash.operator(U=U_full, clover_term=clover, kappa=torch.Tensor([KAPPA]),
                              support_parity=True, verbose=False)
        kwargs['clover_ee_inv'] = op0.sitting.M_e_inv
        kwargs['clover_oo_inv'] = op0.sitting.M_o_inv
    mg = solver.multigrid(**kwargs)
    buf = io.StringIO()
    with redirect_stdout(buf):
        mg.init()
        torch.cuda.synchronize(); t0=perf_counter()
        x = mg.solve(b=b_full)
        torch.cuda.synchronize(); solve_time=perf_counter()-t0
    out = buf.getvalue()
    # Parse iteration counts: every level's cycle() prints "Total iterations".
    # Level 0's cycle() finishes LAST (it recurses into level 1 first), so the
    # LAST occurrence is the level-0 count. Also parse per-level totals.
    iters = re.findall(r'Total iterations:\s*(\d+)', out)
    n0 = int(iters[-1]) if iters else -1
    times = re.findall(r'Total time:\s*([\d.]+)\s*seconds', out)
    return x, solve_time, n0, mg, out

def main():
    Lx,Ly,Lz,Lt = 8,8,8,16
    MASS=0.05; ATOL=1e-6
    U_full, b_full, clover, KAPPA, av, (g,fi,ce,coo,cei,coi) = setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL)

    # (a) plain parity BiStabCG
    fo_ref, ref_time = run_bistabcg_ref(fi,g,ce,coo,cei,coi,av)
    x_ref = tools.poooxyzt2oooxyzt(fo_ref)
    res_ref = tools.norm(dslash.give_wilson(x_ref,U_full,KAPPA,True)+dslash.give_clover(x_ref,clover)-b_full)/tools.norm(b_full)
    print(f"[a] parity BiStabCG: {ref_time*1000:.1f} ms  res={res_ref:.3e}")

    # (b) MG support_parity=False
    x_mg, t_mg, n0, mg_b, _ = run_py_mg(U_full,b_full,clover,KAPPA,False)
    res_mg = tools.norm(dslash.give_wilson(x_mg,U_full,KAPPA,True)+dslash.give_clover(x_mg,clover)-b_full)/tools.norm(b_full)
    vs = tools.norm(x_mg-x_ref)/tools.norm(x_ref)
    print(f"[b] MG support_parity=False: {t_mg*1000:.1f} ms  iters={n0}  res={res_mg:.3e}  vs_ref={vs:.3e}  speedup_vs_ref={ref_time/t_mg:.3f}x")

    # (c) MG support_parity=True
    x_mgp, t_mgp, n0p, mg_p, _ = run_py_mg(U_full,b_full,clover,KAPPA,True)
    res_mgp = tools.norm(dslash.give_wilson(x_mgp,U_full,KAPPA,True)+dslash.give_clover(x_mgp,clover)-b_full)/tools.norm(b_full)
    vs_p = tools.norm(x_mgp-x_ref)/tools.norm(x_ref)
    print(f"[c] MG support_parity=True : {t_mgp*1000:.1f} ms  iters={n0p}  res={res_mgp:.3e}  vs_ref={vs_p:.3e}  speedup_vs_ref={ref_time/t_mgp:.3f}x")

if __name__ == "__main__":
    main()
