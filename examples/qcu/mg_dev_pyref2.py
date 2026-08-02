#!/usr/bin/env python3
"""Run the Python solver.multigrid reference (pure-Python, support_parity=False)
on [8,8,8,16] and compare iteration count vs plain BiStabCG."""
import torch
from time import perf_counter
from pyqcu import tools, dslash, solver
from pyqcu.lattice import generate_gauge_field

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
torch.manual_seed(42)
device=torch.device('cuda'); dt=torch.complex64
U0=torch.randn([3,3,4,Lx,Ly,Lz,Lt],dtype=dt,device=device)
qcu_U = generate_gauge_field(U=U0, sigma=0.1, seed=42)
ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=False,verbose=False)
b=torch.randn([4,3,Lx,Ly,Lz,Lt],dtype=dt,device=device)

# Plain BiStabCG
t0=perf_counter()
x_b=pqsolver_bistabcg if False else None
from pyqcu import solver as pqsolver
x_b=pqsolver.bistabcg(b=b,matvec=op.matvec,tol=ATOL,max_iter=300,verbose=False)
tb=perf_counter()-t0
rb=tools.norm(b-op.matvec(x_b))/tools.norm(b)
print(f"Python BiStabCG: {tb:.1f}s res={rb:.3e}")

# MG support_parity=False
mg=pqsolver.multigrid(
    dtype_list=[dt,dt], device_list=[device,device],
    U=qcu_U, clover_term=ref_cl, kappa=KAPPA,
    min_size=4, max_level=2, mg_grid_size=[2,2,2,2],
    dof_list=[12,24], tol=ATOL, max_iter=300, num_restart=5,
    support_parity=False, verbose=False)
t0=perf_counter(); mg.init()
x_mg=mg.solve(b=b, x0=torch.zeros_like(b))
tm=perf_counter()-t0
rm=tools.norm(b-op.matvec(x_mg))/tools.norm(b)
print(f"Python MG(no parity): {tm:.1f}s res={rm:.3e} iters={len(mg.convergence_history)}")
