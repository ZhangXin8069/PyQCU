#!/usr/bin/env python3
"""Extended concept sweep: can the Schur MG reach enough iteration reduction?
Tests E={48,64}, null-vector inverse iterations {1,2}, coarse solve tolerance.
Uses operator-free exact A_c (the coarse solve cost is NOT the focus here —
we want the ACHIEVABLE fine iteration count to validate the C++ design).
"""
import torch, os, sys, re, io
from time import perf_counter
from contextlib import redirect_stdout
from pyqcu import tools, dslash, solver
import pyqcu.cuda.define as define
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_pyref_expt import setup_gpu

def schur_mg(op, b_full, lat_fine, lat_coarse_odd, dof_coarse, num_restart=10,
             tol=1e-6, max_iter=1000, coarse_tol=1e-6, dof_fine=12, nv_iters=1):
    dt=b_full.dtype; device=b_full.device
    b_eo=tools.oooxyzt2poooxyzt(b_full.reshape([dof_fine]+list(b_full.shape)[2:]))
    b_e,b_o=b_eo[0],b_eo[1]
    b__o=op.give_b_parity(b_e=b_e,b_o=b_o)
    S=op.matvec_parity
    X,Y,Z,T=lat_fine; Th=T//2
    _null=torch.randn([dof_coarse,dof_fine,X,Y,Z,Th],dtype=dt,device=device)
    for _ in range(nv_iters):
        _null=tools.give_null_vecs(null_vecs=_null,matvec=S,bistabcg=None,verbose=False)
    Xc,Yc,Zc,Tc=lat_coarse_odd
    lonv=tools.local_orthogonalize(null_vecs=_null,coarse_lat_size=[Xc,Yc,Zc,Tc],verbose=False)
    def A_c(v):
        f=tools.prolong(local_ortho_null_vecs=lonv,coarse_vec=v)
        return tools.restrict(local_ortho_null_vecs=lonv,fine_vec=S(f))
    x=torch.zeros_like(b__o)
    r=b__o.clone(); rt=r.clone()
    p=torch.zeros_like(r); v=torch.zeros_like(r); s=torch.zeros_like(r); t=torch.zeros_like(r)
    rp=torch.tensor(1.0,dtype=dt,device=device); al=torch.tensor(1.0,dtype=dt,device=device); om=torch.tensor(1.0,dtype=dt,device=device)
    n=0; cr=0; vc=0; bd=False
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
            e_c=solver.bistabcg(b=r_c,matvec=A_c,tol=coarse_tol,max_iter=300,x0=None,if_rtol=True,verbose=False)
            e_f=tools.prolong(local_ortho_null_vecs=lonv,coarse_vec=e_c)
            x=x+e_f
            r=b__o-S(x); rn=tools.norm(r)
            cr=0; vc+=1
            rt=r.clone(); p=torch.zeros_like(r); v=torch.zeros_like(r); s=torch.zeros_like(r); t=torch.zeros_like(r)
            rp=torch.tensor(1.0,dtype=dt,device=device); al=torch.tensor(1.0,dtype=dt,device=device); om=torch.tensor(1.0,dtype=dt,device=device)
            if rn<tol: break
    x_e=op.give_x_e(b_e=b_e,x_o=x)
    x_out=tools.poooxyzt2oooxyzt(torch.stack([x_e,x],dim=0)).reshape(b_full.shape)
    return x_out,n,vc,float(rn),bd

def main():
    Lx,Ly,Lz,Lt=8,8,8,16
    MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
    U_full,b_full,clover,KAPPA,av,(g,fi,ce,coo,cei,coi)=setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL)
    dt=torch.complex64; device=torch.device('cuda')
    op=dslash.operator(U=U_full,clover_term=clover,kappa=torch.Tensor([KAPPA]),support_parity=True,verbose=False)
    b_eo=tools.oooxyzt2poooxyzt(b_full.reshape([12]+list(b_full.shape)[2:]))
    b__o=op.give_b_parity(b_e=b_eo[0],b_o=b_eo[1])
    buf=io.StringIO()
    with redirect_stdout(buf):
        x_o_ref=solver.bistabcg(b=b__o,matvec=op.matvec_parity,tol=ATOL,max_iter=1000,verbose=False)
    m=re.findall(r'Converged at iteration (\d+)',buf.getvalue())
    n_ref=int(m[0])+1 if m else '?'
    x_e_ref=op.give_x_e(b_e=b_eo[0],x_o=x_o_ref)
    x_ref=tools.poooxyzt2oooxyzt(torch.stack([x_e_ref,x_o_ref],dim=0)).reshape(b_full.shape)
    print(f"[ref] parity BiStabCG: iters={n_ref}")
    lat_fine=[Lx,Ly,Lz,Lt]
    lat_coarse_odd=[Lx//2,Ly//2,Lz//2,Lt//4]
    for E in [48, 64]:
        for nvi in [1, 2]:
            for ct in [1e-3, 1e-6]:
                x,n,vc,rn,bd=schur_mg(op,b_full,lat_fine,lat_coarse_odd,E,num_restart=10,tol=ATOL,coarse_tol=ct,nv_iters=nvi)
                vs=tools.norm(x-x_ref)/tools.norm(x_ref)
                res=tools.norm(dslash.give_wilson(x,U_full,KAPPA,True)+dslash.give_clover(x,clover)-b_full)/tools.norm(b_full)
                print(f"[mg E={E} nvi={nvi} ct={ct:.0e}]: iters={n} vc={vc} res={res:.2e} vs_ref={vs:.2e} bd={bd}")

if __name__=="__main__":
    main()
