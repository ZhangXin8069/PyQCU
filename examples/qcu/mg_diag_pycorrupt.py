#!/usr/bin/env python3
"""Find WHERE U_full / clover get corrupted in the Python coarse-op build."""
import torch
from pyqcu import tools, dslash, solver
import pyqcu.cuda.define as define
import sys, os
sys.path.insert(0, os.path.expanduser("~/PyQCU/examples/qcu"))
from mg_pyref_expt import setup_gpu

Lx,Ly,Lz,Lt = 8,8,8,16
MASS=0.05; ATOL=1e-6
U_full, b_full, clover, KAPPA, av, (g,fi,ce,coo,cei,coi) = setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL)

def snap(name, t):
    return name, float(tools.norm(t)), t.clone()

U0 = snap("U", U_full); cl0 = snap("cl", clover); b0 = snap("b", b_full)
print(f"initial: norm(U)={U0[1]:.6f} norm(cl)={cl0[1]:.6f}")

# Step 1: build op_fine
op = dslash.operator(U=U_full, clover_term=clover, kappa=torch.Tensor([KAPPA]),
                     support_parity=False, verbose=False)
print(f"after op_fine: norm(U)={tools.norm(U_full):.6f} (chg={abs(tools.norm(U_full)-U0[1]):.2e})  "
      f"norm(cl)={tools.norm(clover):.6f} (chg={abs(tools.norm(clover)-cl0[1]):.2e})")

# Step 2: give_null_vecs + local_orthogonalize (the coarse setup)
lat_coarse=[Lx//2,Ly//2,Lz//2,Lt//2]
_null = torch.randn([24,12,Lx,Ly,Lz,Lt], dtype=torch.complex64, device=torch.device('cuda'))
_null = tools.give_null_vecs(null_vecs=_null, matvec=op.matvec, bistabcg=None, verbose=False)
print(f"after give_null_vecs: norm(U)={tools.norm(U_full):.6f} (chg={abs(tools.norm(U_full)-U0[1]):.2e})  "
      f"norm(cl)={tools.norm(clover):.6f} (chg={abs(tools.norm(clover)-cl0[1]):.2e})")
_lonv = tools.local_orthogonalize(null_vecs=_null, coarse_lat_size=lat_coarse, verbose=False)
print(f"after local_ortho: norm(U)={tools.norm(U_full):.6f} (chg={abs(tools.norm(U_full)-U0[1]):.2e})  "
      f"norm(cl)={tools.norm(clover):.6f} (chg={abs(tools.norm(clover)-cl0[1]):.2e})")

# Step 3: build coarse_op (Galerkin)
coarse_op = dslash.operator(fine_hopping=op.hopping, fine_sitting=op.sitting,
                            local_ortho_null_vecs=_lonv, verbose=False)
print(f"after coarse_op: norm(U)={tools.norm(U_full):.6f} (chg={abs(tools.norm(U_full)-U0[1]):.2e})  "
      f"norm(cl)={tools.norm(clover):.6f} (chg={abs(tools.norm(clover)-cl0[1]):.2e})")

# Step 4: verify operator matvec still correct on the ORIGINAL system
xr = tools.poooxyzt2oooxyzt(torch.randn_like(fi))
res = tools.norm(op.matvec(xr) - (dslash.give_wilson(xr,U_full,KAPPA,True)+dslash.give_clover(xr,clover)))/tools.norm(op.matvec(xr))
print(f"op.matvec vs explicit operator: rel diff = {res:.3e}")

# Step 5: check op.hopping.U identity
print(f"op.hopping.U is U_full? {op.hopping.U.data_ptr()==U_full.data_ptr()}")
print(f"U_full is view of g? {U_full.storage().data_ptr()==g.storage().data_ptr()}")
