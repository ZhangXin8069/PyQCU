#!/usr/bin/env python3
"""Python replica of the C++ full-site level-0 BiStabCG (num_levels=1).
Compares the residual sequence to isolate whether the C++ BiStabCG is buggy.
"""
import torch
from pyqcu import tools, dslash
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
from pyqcu.cuda import qcu

Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
params[define._LAT_X_]=Lx;params[define._LAT_Y_]=Ly;params[define._LAT_Z_]=Lz;params[define._LAT_T_]=Lt
params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
params[define._PARITY_]=0;params[define._NODE_RANK_]=0;params[define._NODE_SIZE_]=1
params[define._DAGGER_]=0;params[define._MAX_ITER_]=1000
params[define._DATA_TYPE_]=define._LAT_C64_;params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=1
params[define._VERBOSE_]=0;params[define._SEED_]=42;params[define._TEST_IN_CPU_]=0
params[define._MG_NUM_LEVEL_]=1
av=argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
av[define._MASS_]=MASS;av[define._ATOL_]=ATOL;av[define._SIGMA_]=0.1
device=torch.device('cuda');dt=define.dtype(params[define._DATA_TYPE_]);ls=define.lat_shape(params)
g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device);cei=torch.zeros_like(ce)
coo=torch.zeros_like(ce);coi=torch.zeros_like(ce)
params[define._SET_INDEX_]=0;params[define._SET_PLAN_]=-1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyGaussGaugeQcu(g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=0
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
params[define._SET_INDEX_]+=1;params[define._SET_PLAN_]=2;params[define._PARITY_]=1
qcu.applyInitQcu(set_ptrs,params,av);qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)
qcu_U=tools.poooxyzt2oooxyzt(g);ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=False,verbose=False)
b_eo=fi.reshape([2,12,Lx,Ly,Lz,Lt//2])
b=tools.poooxyzt2oooxyzt(b_eo).reshape([12,Lx,Ly,Lz,Lt])

# Plain BiStabCG (Python)
x=torch.zeros_like(b); r=b.clone(); rt=r.clone()
p=torch.zeros_like(r);v=torch.zeros_like(r);s=torch.zeros_like(r);t=torch.zeros_like(r)
rho=torch.tensor(1.,dtype=dt,device=device);rp=torch.tensor(1.,dtype=dt,device=device)
al=torch.tensor(1.,dtype=dt,device=device);om=torch.tensor(1.,dtype=dt,device=device)
resid=[]
for it in range(200):
    rho=tools.vdot(rt,r); be=(rho/rp)*(al/om); rp=rho
    p=r+be*(p-om*v); v=op.matvec(p)
    rtv=tools.vdot(rt,v); al=rho/rtv
    s=r-al*v; t=op.matvec(s)
    tts=tools.vdot(t,t); om=tools.vdot(t,s)/tts
    x=x+al*p+om*s; r=s-om*t
    resid.append(float(tools.norm(r)))
    if it<6 or it%50==0:
        print(f"PY it={it}: r={resid[-1]:.4e}")
    if resid[-1] < ATOL:
        print(f"PY CONVERGED at it={it}")
        break
