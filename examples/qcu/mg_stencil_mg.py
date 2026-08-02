#!/usr/bin/env python3
"""Full production-path validation: Schur MG with the MATERIALIZED 33-tensor
coarse operator A_c. Measures the achievable fine iteration count and wall-clock
(using the cheap materialized coarse solve)."""
import torch, os, sys, io, re, time
from contextlib import redirect_stdout
from pyqcu import tools, dslash, solver
import pyqcu.cuda.define as define
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_pyref_expt import setup_gpu
from mg_stencil_build import build_stencil, apply_stencil, PAIRS, SIGN

def schur_mg_mat(op, b_full, lat_fine_odd, lat_coarse_odd, lonv, hop_nn, hop_diag, sit,
                 num_restart=10, tol=1e-6, max_iter=1000, coarse_tol=1e-4, dof_fine=12):
    dt=b_full.dtype; device=b_full.device
    b_eo=tools.oooxyzt2poooxyzt(b_full.reshape([dof_fine]+list(b_full.shape)[2:]))
    b_e,b_o=b_eo[0],b_eo[1]
    b__o=op.give_b_parity(b_e=b_e,b_o=b_o)
    S=op.matvec_parity
    def A_c(v): return apply_stencil(hop_nn,hop_diag,sit,v)
    x=torch.zeros_like(b__o)
    r=b__o.clone(); rt=r.clone()
    p=torch.zeros_like(r); v=torch.zeros_like(r); s=torch.zeros_like(r); t=torch.zeros_like(r)
    rp=torch.tensor(1.0,dtype=dt,device=device); al=torch.tensor(1.0,dtype=dt,device=device); om=torch.tensor(1.0,dtype=dt,device=device)
    n=0; cr=0; vc=0; bd=False
    t_solve=time.perf_counter()
    for it in range(max_iter):
        rho=tools.vdot(rt,r)
        if abs(rho)<1e-30: bd=True; break
        beta=(rho/rp)*(al/om); rp=rho
        p=r+beta*(p-om*v)
        v=S(p); rtv=tools.vdot(rt,v)
        if abs(rtv)<1e-30: bd=True; break
        al=rho/rtv
        s=r-al*v
        t=S(s); tts=tools.vdot(t,t)
        if abs(tts)<1e-30: bd=True; break
        om=tools.vdot(t,s)/tts
        x=x+al*p+om*s
        r=s-om*t
        n+=1; cr+=1
        rn=tools.norm(r)
        if rn<tol: break
        if cr>=num_restart:
            r_c=tools.restrict(local_ortho_null_vecs=lonv,fine_vec=r)
            e_c=solver.bistabcg(b=r_c,matvec=A_c,tol=coarse_tol,max_iter=200,x0=None,if_rtol=True,verbose=False)
            e_f=tools.prolong(local_ortho_null_vecs=lonv,coarse_vec=e_c)
            x=x+e_f
            r=b__o-S(x); rn=tools.norm(r)
            cr=0; vc+=1
            rt=r.clone(); p=torch.zeros_like(r); v=torch.zeros_like(r); s=torch.zeros_like(r); t=torch.zeros_like(r)
            rp=torch.tensor(1.0,dtype=dt,device=device); al=torch.tensor(1.0,dtype=dt,device=device); om=torch.tensor(1.0,dtype=dt,device=device)
            if rn<tol: break
    t_solve=time.perf_counter()-t_solve
    x_e=op.give_x_e(b_e=b_e,x_o=x)
    x_out=tools.poooxyzt2oooxyzt(torch.stack([x_e,x],dim=0)).reshape(b_full.shape)
    return x_out,n,vc,float(rn),bd,t_solve

def main():
    Lx,Ly,Lz,Lt=8,16,16,16
    MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
    U_full,b_full,clover,KAPPA,av,(g,fi,ce,coo,cei,coi)=setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL)
    dt=torch.complex64; device=torch.device('cuda')
    op=dslash.operator(U=U_full,clover_term=clover,kappa=torch.Tensor([KAPPA]),support_parity=True,verbose=False)
    b_eo=tools.oooxyzt2poooxyzt(b_full.reshape([12]+list(b_full.shape)[2:]))
    b__o=op.give_b_parity(b_e=b_eo[0],b_o=b_eo[1])
    buf=io.StringIO()
    with redirect_stdout(buf):
        t0=time.perf_counter()
        x_o_ref=solver.bistabcg(b=b__o,matvec=op.matvec_parity,tol=ATOL,max_iter=1000,verbose=False)
        t_ref=time.perf_counter()-t0
    m=re.findall(r'Converged at iteration (\d+)',buf.getvalue())
    n_ref=int(m[0])+1 if m else '?'
    x_e_ref=op.give_x_e(b_e=b_eo[0],x_o=x_o_ref)
    x_ref=tools.poooxyzt2oooxyzt(torch.stack([x_e_ref,x_o_ref],dim=0)).reshape(b_full.shape)
    print(f"[ref] parity BiStabCG on {Lx}x{Ly}x{Lz}x{Lt}: iters={n_ref} time={t_ref*1000:.0f}ms")

    lat_fine_odd=[Lx,Ly,Lz,Lt//2]
    for E in [48]:
        lat_coarse_odd=[Lx//2,Ly//2,Lz//2,Lt//4]
        _null=torch.randn([E,12]+lat_fine_odd,dtype=dt,device=device)
        t0=time.perf_counter()
        _null=tools.give_null_vecs(null_vecs=_null,matvec=op.matvec_parity,bistabcg=None,verbose=False)
        t_nv=time.perf_counter()-t0
        lonv=tools.local_orthogonalize(null_vecs=_null,coarse_lat_size=lat_coarse_odd,verbose=False)
        hop_nn,hop_diag,sit=build_stencil(op.matvec_parity,lonv,E,12,lat_fine_odd,lat_coarse_odd,dt,device)
        print(f"  setup (nv={t_nv:.0f}s + stencil): done")
        for restart in [10]:
            for ct in [1e-3, 1e-4]:
                x,n,vc,rn,bd,t_solve=schur_mg_mat(op,b_full,lat_fine_odd,lat_coarse_odd,lonv,hop_nn,hop_diag,sit,
                                                  num_restart=restart,tol=ATOL,coarse_tol=ct)
                vs=tools.norm(x-x_ref)/tools.norm(x_ref)
                res=tools.norm(dslash.give_wilson(x,U_full,KAPPA,True)+dslash.give_clover(x,clover)-b_full)/tools.norm(b_full)
                print(f"[mg E={E} r={restart} ct={ct:.0e}]: iters={n} vc={vc} solve={t_solve*1000:.0f}ms "
                      f"speedup={t_ref/t_solve:.2f}x res={res:.2e} vs_ref={vs:.2e} bd={bd}")

if __name__=="__main__":
    main()
