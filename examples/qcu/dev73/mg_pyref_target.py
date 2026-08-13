#!/usr/bin/env python3
"""Ground truth on the TARGET lattice {8,16,16,16}.

Measures parity BiStabCG vs Python MG (support_parity=True, with_cuda_qcu=True)
on the user's default lattice. Captures verbose output to extract level-0
iteration counts and V-cycle behavior.
"""
import torch, os, sys, re, io, json
from time import perf_counter
from contextlib import redirect_stdout
from pyqcu import tools, dslash, solver
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_pyref_expt import setup_gpu

def run():
    Lx,Ly,Lz,Lt = 8,16,16,16
    MASS=0.05; ATOL=1e-6
    U_full, b_full, clover, KAPPA, av, (g,fi,ce,coo,cei,coi) = setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL)

    # (a) parity BiStabCG
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    fo_ref=torch.zeros_like(fi)
    torch.cuda.synchronize(); t0=perf_counter()
    qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); ref_time=perf_counter()-t0
    x_ref=tools.poooxyzt2oooxyzt(fo_ref)
    res_ref=tools.norm(dslash.give_wilson(x_ref,U_full,KAPPA,True)+dslash.give_clover(x_ref,clover)-b_full)/tools.norm(b_full)
    print(f"[a] parity BiStabCG: {ref_time*1000:.1f} ms  res={res_ref:.3e}")

    # (b) MG support_parity=True, with_cuda_qcu=True, 2 levels
    op0 = dslash.operator(U=U_full, clover_term=clover, kappa=torch.Tensor([KAPPA]),
                          support_parity=True, verbose=False)
    for nlev, nrestart in [(2,5),(2,3),(3,5)]:
        mg = solver.multigrid(
            dtype_list=[torch.complex64]*10, device_list=[torch.device('cuda')]*10,
            U=U_full, clover_term=clover, kappa=torch.Tensor([KAPPA]),
            tol=ATOL, max_iter=1000, max_level=nlev, num_restart=nrestart,
            support_parity=True, verbose=True,
            clover_ee_inv=op0.sitting.M_e_inv, clover_oo_inv=op0.sitting.M_o_inv)
        buf=io.StringIO()
        with redirect_stdout(buf):
            mg.init()
            torch.cuda.synchronize(); t0=perf_counter()
            x=mg.solve(b=b_full)
            torch.cuda.synchronize(); solve_time=perf_counter()-t0
        out=buf.getvalue()
        iters=re.findall(r'Total iterations:\s*(\d+)', out)
        n0=int(iters[-1]) if iters else -1
        res=tools.norm(dslash.give_wilson(x,U_full,KAPPA,True)+dslash.give_clover(x,clover)-b_full)/tools.norm(b_full)
        vs=tools.norm(x-x_ref)/tools.norm(x_ref)
        # count level-0 V-cycle corrections
        vc=len(re.findall(r'cycle start', out))
        print(f"[b] MG parity=True levels={nlev} restart={nrestart}: {solve_time*1000:.1f} ms  "
              f"iters={n0}  res={res:.3e}  vs_ref={vs:.3e}  speedup={ref_time/solve_time:.3f}x")
        # dump verbose to a file for the record
        with open(os.path.expanduser(f"~/PyQCU/logs/dev73/pyref_target_L{nlev}_r{nrestart}.log"),"w") as f:
            f.write(out)

if __name__=="__main__":
    run()
