#!/usr/bin/env python3
"""Compare C++ 2-level V-cycle first correction against a pure-Python reference.
No C++ kernel calls — only torch ops + loaded dumps."""
import torch, numpy as np
from pyqcu import tools, dslash

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; KAPPA=1.0/(2*MASS+8)
Xc,Yc,Zc,Tc=4,4,4,8
device=torch.device('cuda'); dt=torch.complex64
g=torch.load('/tmp/mgfx_g.pt',weights_only=True).to(device)
fi=torch.load('/tmp/mgfx_fi.pt',weights_only=True).to(device)
lonv_flat=torch.load('/tmp/mgfx_lonv.pt',weights_only=True).to(device)
# blocked lonv: [24,12,4,2,4,2,4,2,8,2]
lonv = lonv_flat.reshape(24,12,4,2,4,2,4,2,8,2)
qcu_U=tools.poooxyzt2oooxyzt(g); ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=False,verbose=False)
b_eo=fi.reshape([2,12,Lx,Ly,Lz,Lt//2])
b=tools.poooxyzt2oooxyzt(b_eo).reshape([12,Lx,Ly,Lz,Lt])

# Replicate C++ fine BiStabCG (5 iters)
x=torch.zeros_like(b); r=b.clone(); rt=r.clone()
p=torch.zeros_like(r);v=torch.zeros_like(r);s=torch.zeros_like(r);t=torch.zeros_like(r)
rho=torch.tensor(1.,dtype=dt,device=device);rp=torch.tensor(1.,dtype=dt,device=device)
al=torch.tensor(1.,dtype=dt,device=device);om=torch.tensor(1.,dtype=dt,device=device)
for it in range(5):
    rho=tools.vdot(rt,r); be=(rho/rp)*(al/om); rp=rho
    p=r+be*(p-om*v); v=op.matvec(p); rtv=tools.vdot(rt,v); al=rho/rtv
    s=r-al*v; t=op.matvec(s); tts=tools.vdot(t,t); om=tools.vdot(t,s)/tts
    x=x+al*p+om*s; r=s-om*t

# Python reference: restrict r, solve coarse (Python coarse op), prolong
r_coarse_py = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=r)
# coarse operator via Galerkin (Python)
coarse_op = dslash.operator(fine_hopping=op.hopping, fine_sitting=op.sitting, local_ortho_null_vecs=lonv, verbose=False)
Dc = lambda v: coarse_op.hopping.matvec(v) + torch.einsum('EeXYZT,eXYZT->EXYZT', coarse_op.sitting.M, v)
from pyqcu import solver as pqsolver
e_coarse_py = pqsolver.bistabcg(b=r_coarse_py, matvec=Dc, tol=1e-6, max_iter=100, verbose=False)
e_fine_py = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=e_coarse_py)

# Load C++ dumps
def load(n, shape):
    d=np.fromfile(f'/tmp/mgdbg_{n}.bin',dtype=np.complex64)
    return torch.from_numpy(d).reshape(shape).to(dt).to(device)
r_coarse_cpp = load('coarse_rhs_f',[24,Xc,Yc,Zc,Tc])
e_coarse_cpp = load('e_coarse_f',[24,Xc,Yc,Zc,Tc])
e_fine_cpp = load('e_fine_f',[12,Lx,Ly,Lz,Lt])

def rel(a,b): return float(tools.norm(a-b)/tools.norm(b))
print("=== C++ V-cycle vs Python replica ===")
print(f"coarse_rhs rel diff: {rel(r_coarse_cpp, r_coarse_py):.6e}")
print(f"e_coarse   rel diff: {rel(e_coarse_cpp, e_coarse_py):.6e}   cpp={float(tools.norm(e_coarse_cpp)):.4e} py={float(tools.norm(e_coarse_py)):.4e}")
print(f"e_fine     rel diff: {rel(e_fine_cpp, e_fine_py):.6e}")
rn_before = float(tools.norm(r))
rn_cpp = float(tools.norm(b - op.matvec(x + e_fine_cpp)))
rn_py  = float(tools.norm(b - op.matvec(x + e_fine_py)))
print(f"|r| before={rn_before:.4e}, after C++ corr={rn_cpp:.4e}, after py corr={rn_py:.4e}")
