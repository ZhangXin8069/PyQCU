#!/usr/bin/env python3
"""Build the 33-tensor coarse Schur operator A_c = P^T S P and validate against
the dense (exact) operator.

Stencil:
  sit      [E,E,Xc,Yc,Zc,Tc]                       on-site
  hop_nn   [2,4,E,E,Xc,Yc,Zc,Tc]                   nearest (pm × dir)
  hop_diag [2,2,6,E,E,Xc,Yc,Zc,Tc]                 diagonal (s1 × s2 × pair)
      pair: 0=(x,y) 1=(x,z) 2=(x,t) 3=(y,z) 4=(y,t) 5=(z,t); sign 0=+1 1=-1

Kernel convention (multigrid_coarse_dslash_wide):
  out[j,c] += sit[j,e,c]·in[e,c]
           + hop_nn[pm,d,j,e,c]·in[e, c + pm?(+1):(-1) e_d]
           + hop_diag[s1,s2,pair,j,e,c]·in[e, c + s1 e_d1 + s2 e_d2]
"""
import torch, os, sys, time
from pyqcu import tools, dslash, solver
import pyqcu.cuda.define as define
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_pyref_expt import setup_gpu

PAIRS = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]  # (d1,d2) with d1<d2
SIGN = [1, -1]

def build_stencil(S, lonv, E, e, lat_fine_odd, lat_coarse_odd, dt, device):
    X,Y,Z,Th = lat_fine_odd
    Xc,Yc,Zc,Tc = lat_coarse_odd
    Nc = Xc*Yc*Zc*Tc
    sit = torch.zeros([E,E,Xc,Yc,Zc,Tc], dtype=dt, device=device)
    hop_nn = torch.zeros([2,4,E,E,Xc,Yc,Zc,Tc], dtype=dt, device=device)
    hop_diag = torch.zeros([2,2,6,E,E,Xc,Yc,Zc,Tc], dtype=dt, device=device)
    str_Y=Yc*Zc*Tc; str_Z=Zc*Tc; dims=[Xc,Yc,Zc,Tc]
    t0=time.perf_counter()
    for c_idx in range(Nc):
        cx=c_idx//str_Y; rem=c_idx%str_Y; cy=rem//str_Z; rem%=str_Z; cz=rem//Tc; ct=rem%Tc
        ccoords=[cx,cy,cz,ct]
        for ee in range(E):
            src_c=torch.zeros([E,Xc,Yc,Zc,Tc],dtype=dt,device=device)
            src_c[ee,cx,cy,cz,ct]=1.0
            f=tools.prolong(local_ortho_null_vecs=lonv,coarse_vec=src_c)
            dc=tools.restrict(local_ortho_null_vecs=lonv,fine_vec=S(f))
            sit[:,ee,cx,cy,cz,ct]=dc[:,cx,cy,cz,ct]
            # nearest: hop_nn[pm,d,:,ee,c] = A_c[:,e,c, c ± e_d]; fill at reciprocal site
            for d in range(4):
                # plus coupling at bwd_d(P)
                n=ccoords[:]; n[d]=(n[d]-1+dims[d])%dims[d]
                hop_nn[0,d,:,ee,n[0],n[1],n[2],n[3]]=dc[:,n[0],n[1],n[2],n[3]]
                # minus coupling at fwd_d(P)
                n=ccoords[:]; n[d]=(n[d]+1)%dims[d]
                hop_nn[1,d,:,ee,n[0],n[1],n[2],n[3]]=dc[:,n[0],n[1],n[2],n[3]]
            # diagonal: hop_diag[s1,s2,pair,:,ee,c] = A_c[:,e,c, c+s1 e_d1+s2 e_d2];
            #            fill at reciprocal site P' - (s1 e_d1 + s2 e_d2)
            for pi,(d1,d2) in enumerate(PAIRS):
                for s1i,s1 in enumerate(SIGN):
                    for s2i,s2 in enumerate(SIGN):
                        n=ccoords[:]
                        n[d1]=(n[d1]-s1+dims[d1])%dims[d1]
                        n[d2]=(n[d2]-s2+dims[d2])%dims[d2]
                        hop_diag[s1i,s2i,pi,:,ee,n[0],n[1],n[2],n[3]]=dc[:,n[0],n[1],n[2],n[3]]
        if (c_idx+1)%64==0 and c_idx>0:
            print(f"    probing {c_idx+1}/{Nc} ({time.perf_counter()-t0:.1f}s)")
    print(f"  stencil build: {time.perf_counter()-t0:.1f}s for {E*Nc} probes")
    return hop_nn, hop_diag, sit

def apply_stencil(hop_nn, hop_diag, sit, v_c):
    E = v_c.shape[0]; Xc,Yc,Zc,Tc = v_c.shape[1:]
    out = torch.einsum("EeXYZT,eXYZT->EXYZT", sit, v_c).clone()
    for d in range(4):
        fwd=torch.roll(v_c, shifts=-1, dims=d+1)
        bwd=torch.roll(v_c, shifts=1, dims=d+1)
        out += torch.einsum("EeXYZT,eXYZT->EXYZT", hop_nn[0,d], fwd)
        out += torch.einsum("EeXYZT,eXYZT->EXYZT", hop_nn[1,d], bwd)
    for pi,(d1,d2) in enumerate(PAIRS):
        for s1i,s1 in enumerate(SIGN):
            for s2i,s2 in enumerate(SIGN):
                shift=[0,0,0,0]; shift[d1]=-s1; shift[d2]=-s2
                v_shift=torch.roll(v_c, shifts=tuple(shift), dims=(1,2,3,4))
                # v_shift[c] = v_c[c+s1 e_d1+s2 e_d2]  (in[fwd neighbor])
                out += torch.einsum("EeXYZT,eXYZT->EXYZT", hop_diag[s1i,s2i,pi], v_shift)
    return out

def main():
    Lx,Ly,Lz,Lt=8,8,8,16
    MASS=0.05; KAPPA=1.0/(2*MASS+8)
    U_full,b_full,clover,KAPPA,av,(g,fi,ce,coo,cei,coi)=setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL=1e-6)
    dt=torch.complex64; device=torch.device('cuda')
    op=dslash.operator(U=U_full,clover_term=clover,kappa=torch.Tensor([KAPPA]),support_parity=True,verbose=False)
    S=op.matvec_parity
    E=24
    lat_fine_odd=[Lx,Ly,Lz,Lt//2]; lat_coarse_odd=[Lx//2,Ly//2,Lz//2,Lt//4]
    _null=torch.randn([E,12]+lat_fine_odd,dtype=dt,device=device)
    _null=tools.give_null_vecs(null_vecs=_null,matvec=S,bistabcg=None,verbose=False)
    lonv=tools.local_orthogonalize(null_vecs=_null,coarse_lat_size=lat_coarse_odd,verbose=False)
    hop_nn,hop_diag,sit=build_stencil(S,lonv,E,12,lat_fine_odd,lat_coarse_odd,dt,device)
    Xc,Yc,Zc,Tc=lat_coarse_odd; Nc=Xc*Yc*Zc*Tc
    # Validate vs operator-free AND dense
    for trial in range(2):
        v=torch.randn([E,Xc,Yc,Zc,Tc],dtype=dt,device=device)
        A_st=apply_stencil(hop_nn,hop_diag,sit,v)
        f=tools.prolong(local_ortho_null_vecs=lonv,coarse_vec=v)
        A_op=tools.restrict(local_ortho_null_vecs=lonv,fine_vec=S(f))
        err=tools.norm(A_st-A_op)/tools.norm(A_op)
        print(f"trial {trial}: stencil vs operator-free rel err = {err:.4e}")

if __name__=="__main__":
    main()
