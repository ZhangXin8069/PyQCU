#!/usr/bin/env python3
"""Compare C++ 2-level full-operator MG first-V-cycle data against Python replica."""
import torch, numpy as np
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
from pyqcu import solver as pqsolver

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
Xc,Yc,Zc,Tc=4,4,4,8
device=torch.device('cuda'); dt=define.dtype(define._LAT_C64_)
g=torch.load('/tmp/mgfx_g.pt',weights_only=True).to(device)
fi=torch.load('/tmp/mgfx_fi.pt',weights_only=True).to(device)
lonv=torch.load('/tmp/mgfx_lonv.pt',weights_only=True).to(device)
hp=torch.load('/tmp/mgfx_hp.pt',weights_only=True).to(device)
sp=torch.load('/tmp/mgfx_sp.pt',weights_only=True).to(device)
qcu_U=tools.poooxyzt2oooxyzt(g); ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=False,verbose=False)
b_eo=fi.reshape([2,12,Lx,Ly,Lz,Lt//2])
b=tools.poooxyzt2oooxyzt(b_eo).reshape([12,Lx,Ly,Lz,Lt])

# Setup params for C++ kernels
params[define._LAT_X_]=Lx;params[define._LAT_Y_]=Ly;params[define._LAT_Z_]=Lz;params[define._LAT_T_]=Lt
params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
params[define._PARITY_]=0;params[define._NODE_RANK_]=0;params[define._NODE_SIZE_]=1
params[define._DAGGER_]=0;params[define._MAX_ITER_]=1000
params[define._DATA_TYPE_]=define._LAT_C64_;params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1
params[define._VERBOSE_]=0;params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0
params[define._MG_NUM_LEVEL_]=2
params[define._MG_LEVEL1_E_]=24;params[define._MG_LEVEL1_X_]=Xc;params[define._MG_LEVEL1_Y_]=Yc
params[define._MG_LEVEL1_Z_]=Zc;params[define._MG_LEVEL1_T_]=Tc;params[define._MG_LEVEL1_MAX_ITER_]=100
params[define._MG_LEVEL1_DATA_TYPE_]=define._LAT_C64_;params[define._MG_LEVEL1_NUM_RESTART_]=5
av=argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1
set_ptrs[10+0]=lonv.data_ptr();set_ptrs[10+1]=hp.data_ptr();set_ptrs[10+2]=sp.data_ptr()
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1
qcu.applyInitQcu(set_ptrs,params,av)

def coarse_matvec(src):
    out=torch.zeros_like(src); params[define._SET_INDEX_]=0
    qcu.applyMultigridCoarseDslashQcu(out,src,hp,sp,set_ptrs,params)
    return out
def restrict_cpp(v):
    out=torch.zeros([24,Xc,Yc,Zc,Tc],dtype=dt,device=device)
    params[define._SET_INDEX_]=0
    qcu.applyMultigridRestrictQcu(out, v, lonv, set_ptrs, params)
    return out
def prolong_cpp(v):
    out=torch.zeros([12,Lx,Ly,Lz,Lt],dtype=dt,device=device)
    params[define._SET_INDEX_]=0
    qcu.applyMultigridProLongQcu(out, v, lonv, set_ptrs, params)
    return out

# Replicate the C++ fine BiStabCG (full operator) up to the first V-cycle (5 iterations)
x=torch.zeros_like(b); r=b.clone(); rt=r.clone()
p=torch.zeros_like(r);v=torch.zeros_like(r);s=torch.zeros_like(r);t=torch.zeros_like(r)
rho=torch.tensor(1.,dtype=dt,device=device);rp=torch.tensor(1.,dtype=dt,device=device)
al=torch.tensor(1.,dtype=dt,device=device);om=torch.tensor(1.,dtype=dt,device=device)
for it in range(5):
    rho=tools.vdot(rt,r); be=(rho/rp)*(al/om); rp=rho
    p=r+be*(p-om*v); v=op.matvec(p); rtv=tools.vdot(rt,v); al=rho/rtv
    s=r-al*v; t=op.matvec(s); tts=tools.vdot(t,t); om=tools.vdot(t,s)/tts
    x=x+al*p+om*s; r=s-om*t

# Python reference coarse correction
r_coarse_py = restrict_cpp(r)
e_coarse_py = pqsolver.bistabcg(b=r_coarse_py, matvec=coarse_matvec, tol=1e-4, max_iter=100, verbose=False)
e_fine_py = prolong_cpp(e_coarse_py)

# C++ dumps
def load(n, shape):
    d=np.fromfile(f'/tmp/mgdbg_{n}.bin',dtype=np.complex64)
    return torch.from_numpy(d).reshape(shape).to(dt).to(device)
r_coarse_cpp = load('coarse_rhs_f',[24,Xc,Yc,Zc,Tc])
e_coarse_cpp = load('e_coarse_f',[24,Xc,Yc,Zc,Tc])
e_fine_cpp = load('e_fine_f',[12,Lx,Ly,Lz,Lt])

def rel(a,b): return float(tools.norm(a-b)/tools.norm(b))
print("=== C++ V-cycle vs Python replica ===")
print(f"coarse_rhs rel diff: {rel(r_coarse_cpp, r_coarse_py):.6e}")
print(f"e_coarse   rel diff: {rel(e_coarse_cpp, e_coarse_py):.6e}")
print(f"e_fine     rel diff: {rel(e_fine_cpp, e_fine_py):.6e}")
# correction effectiveness
rn_before = float(tools.norm(r))
rn_cpp = float(tools.norm(b - op.matvec(x + e_fine_cpp)))
rn_py  = float(tools.norm(b - op.matvec(x + e_fine_py)))
print(f"|r| before={rn_before:.4e}, after C++ corr={rn_cpp:.4e}, after py corr={rn_py:.4e}")
