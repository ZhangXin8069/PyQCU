#!/usr/bin/env python3
"""Python-orchestrated replica of the C++ Clover MG V-cycle algorithm.

Replicates the EXACT algorithm in lattice_clover_multigrid.h using Python
operators for the fine level (even-odd preconditioned Schur complement) and
the C++ restrict/prolong/coarse-dslash kernels for the coarse level.

If this replica converges while the C++ MG diverges, the bug is in the C++
orchestration (compute_full_residual / extract_odd / state reset). If this
replica also diverges, the bug is in the algorithm design or the kernels.
"""
import torch
from time import perf_counter
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
import sys

def setup(Lx=8, Ly=8, Lz=8, Lt=16, MASS=0.05, ATOL=1e-6, DT=define._LAT_C64_,
          NUM_LEVELS=2, DOF_LIST=[12,12], MG_GRID=[2,2,2,2], NUM_RESTART=5,
          COARSE_MAX_ITER=100, COARSE_TOL_FACTOR=10.0):
    params[define._LAT_X_]=Lx; params[define._LAT_Y_]=Ly
    params[define._LAT_Z_]=Lz; params[define._LAT_T_]=Lt
    params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
    params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=tools.give_grid_size()
    params[define._PARITY_]=0; params[define._NODE_RANK_]=0; params[define._NODE_SIZE_]=1
    params[define._DAGGER_]=0; params[define._MAX_ITER_]=1000
    params[define._DATA_TYPE_]=DT; params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=1
    params[define._VERBOSE_]=0; params[define._SEED_]=42; params[define._TEST_IN_CPU_]=0
    params[define._MG_NUM_LEVEL_]=NUM_LEVELS
    Xc,Yc,Zc,Tc = Lx//MG_GRID[0], Ly//MG_GRID[1], Lz//MG_GRID[2], Lt//MG_GRID[3]
    params[define._MG_LEVEL1_E_]=DOF_LIST[1]; params[define._MG_LEVEL1_X_]=Xc
    params[define._MG_LEVEL1_Y_]=Yc; params[define._MG_LEVEL1_Z_]=Zc; params[define._MG_LEVEL1_T_]=Tc
    params[define._MG_LEVEL1_MAX_ITER_]=COARSE_MAX_ITER; params[define._MG_LEVEL1_DATA_TYPE_]=DT
    params[define._MG_LEVEL1_NUM_RESTART_]=NUM_RESTART
    av=argv.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_]=MASS; av[define._ATOL_]=ATOL; av[define._SIGMA_]=0.1
    av[define._MG_LEVEL1_ATOL_]=ATOL*COARSE_TOL_FACTOR
    KAPPA=1.0/(2*MASS+8)
    device=torch.device('cuda'); dt=define.dtype(DT); ls=define.lat_shape(params)
    g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
    fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
    ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device); cei=torch.zeros_like(ce)
    coo=torch.zeros_like(ce); coi=torch.zeros_like(ce)
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)
    qcu_U=tools.poooxyzt2oooxyzt(g); ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    op_fine=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=KAPPA,support_parity=True,verbose=False)
    # Build coarse operators
    coarse_lat=[Xc,Yc,Zc,Tc]
    op_list=[op_fine]
    lonv_list=[]; hop_list=[]; sit_list=[]; nv_flat_list=[]
    for i in range(1,NUM_LEVELS):
        lat_fine=[Lx,Ly,Lz,Lt]
        dof_c=DOF_LIST[i]
        _nv=torch.randn([dof_c,12]+lat_fine,dtype=dt,device=device)
        _nv=tools.give_null_vecs(null_vecs=_nv,matvec=op_list[i-1].matvec,bistabcg=None,verbose=False)
        _lonv=tools.local_orthogonalize(null_vecs=_nv,coarse_lat_size=coarse_lat,verbose=False)
        E,e=_lonv.shape[0],_lonv.shape[1]
        Xc0,mgx=_lonv.shape[2],_lonv.shape[3]; Yc0,mgy=_lonv.shape[4],_lonv.shape[5]
        Zc0,mgz=_lonv.shape[6],_lonv.shape[7]; Tc0,mgt=_lonv.shape[8],_lonv.shape[9]
        lonv_flat=_lonv.reshape(E,e,Xc0*mgx,Yc0*mgy,Zc0*mgz,Tc0*mgt).contiguous()
        nv_flat_list.append(lonv_flat); lonv_list.append(_lonv)
        coarse_op=dslash.operator(fine_hopping=op_list[i-1].hopping,fine_sitting=op_list[i-1].sitting,
                                  local_ortho_null_vecs=_lonv,verbose=False)
        op_list.append(coarse_op)
        hp=torch.zeros([2,4,E,E,Xc0,Yc0,Zc0,Tc0],dtype=dt,device=device)
        for ward in range(4):
            hp[0,ward]=coarse_op.hopping.M_plus_list[ward].to(dtype=dt,device=device)
            hp[1,ward]=coarse_op.hopping.M_minus_list[ward].to(dtype=dt,device=device)
        hop_list.append(hp); sit_list.append(coarse_op.sitting.M.to(dtype=dt,device=device))
    for fl in range(len(nv_flat_list)):
        set_ptrs[10+3*fl+0]=nv_flat_list[fl].contiguous().data_ptr()
        set_ptrs[10+3*fl+1]=hop_list[fl].contiguous().data_ptr()
        set_ptrs[10+3*fl+2]=sit_list[fl].contiguous().data_ptr()
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1
    qcu.applyInitQcu(set_ptrs,params,av)
    return dict(op_fine=op_fine, KAPPA=KAPPA, ATOL=ATOL, NUM_RESTART=NUM_RESTART,
                dt=dt, device=device, Lx=Lx, Ly=Ly, Lz=Lz, Lt=Lt, MASS=MASS,
                fi=fi, qcu_U=qcu_U, ref_cl=ref_cl, nv_flat_list=nv_flat_list,
                hop_list=hop_list, sit_list=sit_list, DOF_LIST=DOF_LIST)

S=setup()
op=S['op_fine']; KAPPA=S['KAPPA']; ATOL=S['ATOL']; NUM_RESTART=S['NUM_RESTART']
dt=S['dt']; device=S['device']; Lx,Ly,Lz,Lt=S['Lx'],S['Ly'],S['Lz'],S['Lt']
MASS=S['MASS']
Xc,Yc,Zc,Tc=int(params[define._MG_LEVEL1_X_]),int(params[define._MG_LEVEL1_Y_]),int(params[define._MG_LEVEL1_Z_]),int(params[define._MG_LEVEL1_T_])
E_coarse=int(params[define._MG_LEVEL1_E_])
def lat_shape_full(): return [Lx,Ly,Lz,Lt]

# b in parity-split [2,12,X,Y,Z,T/2]
b_eo = S['fi']
b_e = b_eo[0].reshape([12,Lx,Ly,Lz,Lt//2])
b_o = b_eo[1].reshape([12,Lx,Ly,Lz,Lt//2])
b_full = tools.poooxyzt2oooxyzt(b_eo).reshape([12,Lx,Ly,Lz,Lt])  # full-site [12,X,Y,Z,T]

# b__o = b_o - kappa^2... use Python give_b_parity (Schur RHS)
b__o = op.give_b_parity(b_e=b_e, b_o=b_o).reshape([12,Lx,Ly,Lz,Lt//2])

def matvec_precond(x_o):
    return op.matvec_parity(src_o=x_o)

def compute_full_residual(x_o):
    x_e = op.give_x_e(b_e=b_e, x_o=x_o)
    x_full = tools.poooxyzt2oooxyzt(torch.stack([x_e, x_o],dim=0))
    r_full = b_full - op.matvec(x_full)
    return r_full

def coarse_matvec_cpp(src):
    """C++ coarse dslash (full-site coarse)"""
    out = torch.zeros_like(src)
    params[define._SET_INDEX_] = 0
    qcu.applyMultigridCoarseDslashQcu(out, src, S['hop_list'][0], S['sit_list'][0], set_ptrs, params)
    return out

def coarse_solve(r_coarse):
    """Solve coarse system with BiStabCG using C++ coarse dslash. Return e_coarse."""
    params[define._SET_INDEX_] = 0
    # Use Python bistabcg with C++ matvec
    from pyqcu import solver as pqsolver
    e_c = pqsolver.bistabcg(b=r_coarse, matvec=coarse_matvec_cpp, tol=1e-5,
                            max_iter=100, verbose=False)
    return e_c

def restrict_cpp(r_full):
    out = torch.zeros([E_coarse, Xc,Yc,Zc,Tc], dtype=dt, device=device)
    params[define._SET_INDEX_] = 0
    params[define._MG_NUM_LEVEL_]=2; params[define._MG_LEVEL1_E_]=E_coarse
    params[define._MG_LEVEL1_X_]=Xc; params[define._MG_LEVEL1_Y_]=Yc
    params[define._MG_LEVEL1_Z_]=Zc; params[define._MG_LEVEL1_T_]=Tc
    qcu.applyMultigridRestrictQcu(out, r_full, S['nv_flat_list'][0], set_ptrs, params)
    return out

def prolong_cpp(e_coarse):
    out = torch.zeros([12, Lx,Ly,Lz,Lt], dtype=dt, device=device)
    params[define._SET_INDEX_] = 0
    params[define._MG_NUM_LEVEL_]=2; params[define._MG_LEVEL1_E_]=E_coarse
    params[define._MG_LEVEL1_X_]=Xc; params[define._MG_LEVEL1_Y_]=Yc
    params[define._MG_LEVEL1_Z_]=Zc; params[define._MG_LEVEL1_T_]=Tc
    qcu.applyMultigridProLongQcu(out, e_coarse, S['nv_flat_list'][0], set_ptrs, params)
    return out

# ---- Main BiStabCG loop replicating the C++ MG ----
x_o = torch.zeros([12,Lx,Ly,Lz,Lt//2], dtype=dt, device=device)
r = b__o.clone(); r_tilde = r.clone()
p = torch.zeros_like(r); v = torch.zeros_like(r); s = torch.zeros_like(r); t = torch.zeros_like(r)
rho = torch.tensor(1.0,dtype=dt,device=device); rho_prev=torch.tensor(1.0,dtype=dt,device=device)
alpha=torch.tensor(1.0,dtype=dt,device=device); omega=torch.tensor(1.0,dtype=dt,device=device)
count_restart = 0
conv = []
MAXITER = 200
for it in range(MAXITER):
    rho = tools.vdot(r_tilde, r)
    beta = (rho/rho_prev)*(alpha/omega)
    rho_prev = rho
    p = r + beta*(p - omega*v)
    v = matvec_precond(p)
    rtv = tools.vdot(r_tilde, v)
    alpha = rho/rtv
    s = r - alpha*v
    t = matvec_precond(s)
    tts = tools.vdot(t,t)
    omega = tools.vdot(t,s)/tts
    x_o = x_o + alpha*p + omega*s
    r = s - omega*t
    rn = tools.norm(r)
    conv.append(float(rn))
    count_restart += 1
    if count_restart >= NUM_RESTART:
        # V-cycle
        r_full = compute_full_residual(x_o)
        r_coarse = restrict_cpp(r_full)
        e_coarse = coarse_solve(r_coarse)
        e_fine = prolong_cpp(e_coarse)
        e_fine_eo = tools.oooxyzt2poooxyzt(e_fine)
        e_odd = e_fine_eo[1].reshape(x_o.shape)
        x_o = x_o + e_odd
        r = b__o - matvec_precond(x_o)
        rn = tools.norm(r)
        conv.append(float(rn))
        r_tilde = r.clone()
        p = torch.zeros_like(r); v = torch.zeros_like(r); s = torch.zeros_like(r); t = torch.zeros_like(r)
        rho = torch.tensor(1.0,dtype=dt,device=device); rho_prev=torch.tensor(1.0,dtype=dt,device=device)
        alpha=torch.tensor(1.0,dtype=dt,device=device); omega=torch.tensor(1.0,dtype=dt,device=device)
        count_restart = 0
        if it > 0 and (it+1) % 100 == 0:
            print(f"iter {it}: rn={rn:.3e}")
    if rn < ATOL:
        print(f"CONVERGED at iter {it}, rn={rn:.3e}")
        break
    if it > 0 and (it+1) % 50 == 0:
        print(f"iter {it}: rn={rn:.3e}")

print(f"Final: iterations={len(conv)}, last rn={conv[-1]:.3e}")
# Check solution quality
x_e = op.give_x_e(b_e=b_e, x_o=x_o)
x_full = tools.poooxyzt2oooxyzt(torch.stack([x_e, x_o], dim=0))
res_full = tools.norm(b_full - op.matvec_all(x_full))/tools.norm(b_full)
print(f"Full residual |b - D x|/|b| = {res_full:.3e}")
