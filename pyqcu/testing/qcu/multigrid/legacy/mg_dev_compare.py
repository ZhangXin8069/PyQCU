#!/usr/bin/env python3
"""Compare C++ MG debug dumps against the Python-orchestrated replica at the first V-cycle.

Loads the fixture (g, fi, coarse ops) saved by mg_dev_fulltest.py, runs the replica
up to the first V-cycle, and compares r_full / coarse_rhs / e_coarse / e_odd.
"""
import torch, numpy as np
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
from pyqcu import solver as pqsolver

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; NUM_RESTART=5
DT=define._LAT_C64_; KAPPA=float(np.load("/tmp/mgfx_kappa.npy"))
Xc,Yc,Zc,Tc = 4,4,4,8
device=torch.device('cuda'); dt=define.dtype(DT)

# Load fixture
g = torch.load("/tmp/mgfx_g.pt", weights_only=True).to(device)
fi = torch.load("/tmp/mgfx_fi.pt", weights_only=True).to(device)
lonv = torch.load("/tmp/mgfx_lonv.pt", weights_only=True).to(device)
hp = torch.load("/tmp/mgfx_hp.pt", weights_only=True).to(device)
sp = torch.load("/tmp/mgfx_sp.pt", weights_only=True).to(device)

# Setup params + init (must match fulltest to be able to call kernels)
params[define._LAT_X_]=Lx;params[define._LAT_Y_]=Ly;params[define._LAT_Z_]=Lz;params[define._LAT_T_]=Lt
params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
params[define._PARITY_]=0;params[define._NODE_RANK_]=0;params[define._NODE_SIZE_]=1
params[define._DAGGER_]=0;params[define._MAX_ITER_]=1000
params[define._DATA_TYPE_]=DT;params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1
params[define._VERBOSE_]=0;params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0
params[define._MG_NUM_LEVEL_]=2
params[define._MG_LEVEL1_E_]=12;params[define._MG_LEVEL1_X_]=Xc
params[define._MG_LEVEL1_Y_]=Yc;params[define._MG_LEVEL1_Z_]=Zc;params[define._MG_LEVEL1_T_]=Tc
params[define._MG_LEVEL1_MAX_ITER_]=100;params[define._MG_LEVEL1_DATA_TYPE_]=DT
params[define._MG_LEVEL1_NUM_RESTART_]=NUM_RESTART
av=argv.to(dtype=define.dtype(DT).to_real())
av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1
av[define._MG_LEVEL1_ATOL_]=ATOL*10.0

# Rebuild clover on GPU (g was saved as parity-split [2,3,3,4,X,Y,Z,T/2])
ls=define.lat_shape(params)
ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device);cei=torch.zeros_like(ce)
coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)
params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

qcu_U=tools.poooxyzt2oooxyzt(g);ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=True,verbose=False)
set_ptrs[10+0]=lonv.data_ptr();set_ptrs[10+1]=hp.data_ptr();set_ptrs[10+2]=sp.data_ptr()
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=1
qcu.applyInitQcu(set_ptrs,params,av)

# Load C++ dumps
def load_bin(name, nelem):
    data = np.fromfile(f"/tmp/mgdbg_{name}.bin", dtype=np.complex64)
    return torch.from_numpy(data).reshape(nelem).to(dt).to(device)
r_full_cpp = load_bin("r_full", [12,Lx,Ly,Lz,Lt])
coarse_rhs_cpp = load_bin("coarse_rhs", [12,Xc,Yc,Zc,Tc])
e_coarse_cpp = load_bin("e_coarse", [12,Xc,Yc,Zc,Tc])
e_odd_cpp = load_bin("e_odd", [12,Lx,Ly,Lz,Lt//2])

# Replica BiStabCG up to first V-cycle
b_eo=fi; b_e=b_eo[0].reshape([12,Lx,Ly,Lz,Lt//2]); b_o=b_eo[1].reshape([12,Lx,Ly,Lz,Lt//2])
b_full=tools.poooxyzt2oooxyzt(b_eo).reshape([12,Lx,Ly,Lz,Lt])
b__o=op.give_b_parity(b_e=b_e,b_o=b_o).reshape([12,Lx,Ly,Lz,Lt//2])
def matvec_precond(x_o): return op.matvec_parity(src_o=x_o)
x_o=torch.zeros([12,Lx,Ly,Lz,Lt//2],dtype=dt,device=device)
r=b__o.clone(); r_tilde=r.clone()
p=torch.zeros_like(r);v=torch.zeros_like(r);s=torch.zeros_like(r);t=torch.zeros_like(r)
rho=torch.tensor(1.0,dtype=dt,device=device);rho_prev=torch.tensor(1.0,dtype=dt,device=device)
alpha=torch.tensor(1.0,dtype=dt,device=device);omega=torch.tensor(1.0,dtype=dt,device=device)
for it in range(5):
    rho=tools.vdot(r_tilde,r); beta=(rho/rho_prev)*(alpha/omega); rho_prev=rho
    p=r+beta*(p-omega*v); v=matvec_precond(p)
    rtv=tools.vdot(r_tilde,v); alpha=rho/rtv
    s=r-alpha*v; t=matvec_precond(s)
    tts=tools.vdot(t,t); omega=tools.vdot(t,s)/tts
    x_o=x_o+alpha*p+omega*s; r=s-omega*t

x_e=op.give_x_e(b_e=b_e,x_o=x_o)
x_full=tools.poooxyzt2oooxyzt(torch.stack([x_e,x_o],dim=0))
r_full_py=b_full-op.matvec(x_full)

params[define._SET_INDEX_]=0
coarse_rhs_py=torch.zeros([12,Xc,Yc,Zc,Tc],dtype=dt,device=device)
qcu.applyMultigridRestrictQcu(coarse_rhs_py, r_full_py, lonv, set_ptrs, params)

def coarse_matvec(src):
    out=torch.zeros_like(src); params[define._SET_INDEX_]=0
    qcu.applyMultigridCoarseDslashQcu(out,src,hp,sp,set_ptrs,params)
    return out
e_coarse_py=pqsolver.bistabcg(b=coarse_rhs_py,matvec=coarse_matvec,tol=1e-5,max_iter=100,verbose=False)

params[define._SET_INDEX_]=0
e_fine_py=torch.zeros([12,Lx,Ly,Lz,Lt],dtype=dt,device=device)
qcu.applyMultigridProLongQcu(e_fine_py, e_coarse_py, lonv, set_ptrs, params)
e_fine_eo=tools.oooxyzt2poooxyzt(e_fine_py)
e_odd_py=e_fine_eo[1].reshape([12,Lx,Ly,Lz,Lt//2])

def rel(a,b): return float(tools.norm(a-b)/tools.norm(b))
print("=== C++ MG dump vs Python replica (first V-cycle) ===")
print(f"r_full     rel diff: {rel(r_full_cpp, r_full_py):.6e}   cpp_norm={tools.norm(r_full_cpp):.4e} py_norm={tools.norm(r_full_py):.4e}")
print(f"coarse_rhs rel diff: {rel(coarse_rhs_cpp, coarse_rhs_py):.6e}")
print(f"e_coarse   rel diff: {rel(e_coarse_cpp, e_coarse_py):.6e}   cpp_norm={tools.norm(e_coarse_cpp):.4e} py_norm={tools.norm(e_coarse_py):.4e}")
print(f"e_odd      rel diff: {rel(e_odd_cpp, e_odd_py):.6e}")

# Does the C++ coarse solution satisfy D_c·e_c ≈ rhs?
D_ec_cpp = coarse_matvec(e_coarse_cpp)
print(f"coarse solve check: ||rhs - D_c·e_c||/||rhs|| = {float(tools.norm(coarse_rhs_cpp - D_ec_cpp)/tools.norm(coarse_rhs_cpp)):.6e}")
D_ec_py = coarse_matvec(e_coarse_py)
print(f"replica  solve check: ||rhs - D_c·e_c||/||rhs|| = {float(tools.norm(coarse_rhs_py - D_ec_py)/tools.norm(coarse_rhs_py)):.6e}")

# Direction: cosine similarity between C++ and replica coarse corrections
cos = float((tools.vdot(e_coarse_cpp, e_coarse_py) / (tools.norm(e_coarse_cpp)*tools.norm(e_coarse_py))).real)
print(f"cosine(C++ e_coarse, replica e_coarse) = {cos:.6f}  (1.0 = same direction, -1.0 = opposite)")

# Direct test: apply C++ e_odd vs replica e_odd to x_o, measure preconditioned residual
rn_before = float(tools.norm(b__o - matvec_precond(x_o)))
x_o_cpp = x_o + e_odd_cpp.reshape(x_o.shape)
x_o_py  = x_o + e_odd_py.reshape(x_o.shape)
rn_cpp = float(tools.norm(b__o - matvec_precond(x_o_cpp)))
rn_py  = float(tools.norm(b__o - matvec_precond(x_o_py)))
print(f"precond residual: before={rn_before:.6e}, after C++ corr={rn_cpp:.6e}, after replica corr={rn_py:.6e}")

# Verify C++ e_odd kernel: prolong C++ e_coarse with C++ kernel, extract odd via Python, compare to dump
params[define._SET_INDEX_]=0
e_fine_from_cpp_ec = torch.zeros([12,Lx,Ly,Lz,Lt], dtype=dt, device=device)
qcu.applyMultigridProLongQcu(e_fine_from_cpp_ec, e_coarse_cpp, lonv, set_ptrs, params)
e_odd_from_py_tools = tools.oooxyzt2poooxyzt(e_fine_from_cpp_ec)[1].reshape([12,Lx,Ly,Lz,Lt//2])
print(f"e_odd: C++ dump vs py-extract-of-prolong(C++ e_coarse): rel diff = {float(tools.norm(e_odd_cpp - e_odd_from_py_tools)/tools.norm(e_odd_cpp)):.6e}")
print(f"e_odd: C++ dump norm={float(tools.norm(e_odd_cpp)):.4e}, py-extract norm={float(tools.norm(e_odd_from_py_tools)):.4e}")
print(f"prolong(C++ e_coarse) vs r_full dump: rel diff = {float(tools.norm(r_full_cpp - e_fine_from_cpp_ec)/tools.norm(r_full_cpp)):.6e}")

# Correction quality: does D * P * e_coarse approximate r_full?
D_ef_cpp = op.matvec(e_fine_from_cpp_ec)
rf = r_full_py
cosq = float((tools.vdot(rf, D_ef_cpp)/(tools.norm(rf)*tools.norm(D_ef_cpp))).real)
print(f"corr quality: ||r_full||={float(tools.norm(rf)):.4e}, ||D·P·e_c||/||r_full||={float(tools.norm(D_ef_cpp)/tools.norm(rf)):.4e}, "
      f"cosine(r_full, D·P·e_c)={cosq:.6f}, ||r_full-D·P·e_c||/||r_full||={float(tools.norm(rf-D_ef_cpp)/tools.norm(rf)):.4e}")
