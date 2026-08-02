#!/usr/bin/env python3
"""Test: does A_c = P^T S P have diagonal (corner) couplings beyond nearest-neighbor?

Builds the DENSE A_c from single-site probes and checks whether it matches the
operator-free application, and whether the nearest-neighbor-only materialization
is missing diagonal couplings.
"""
import torch, os, sys
from pyqcu import tools, dslash
import pyqcu.cuda.define as define
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_pyref_expt import setup_gpu
from mg_schur_build import build_schur_coarse_op, apply_coarse_matvec, local_ortho_odd

def main():
    Lx,Ly,Lz,Lt = 8,8,8,16
    MASS=0.05; KAPPA=1.0/(2*MASS+8)
    U_full, b_full, clover, KAPPA, av, (g,fi,ce,coo,cei,coi) = setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL=1e-6)
    dt=torch.complex64; device=torch.device('cuda')
    op = dslash.operator(U=U_full, clover_term=clover, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity
    E=24
    lat_fine_odd=[Lx,Ly,Lz,Lt//2]; lat_coarse_odd=[Lx//2,Ly//2,Lz//2,Lt//4]
    _null = torch.randn([E,12]+lat_fine_odd, dtype=dt, device=device)
    _null = tools.give_null_vecs(null_vecs=_null, matvec=S, bistabcg=None, verbose=False)
    lonv = local_ortho_odd(_null, lat_coarse_odd)
    hop, sit = build_schur_coarse_op(S, lonv, E, 12, lat_fine_odd, lat_coarse_odd, dt, device)
    Xc,Yc,Zc,Tc = lat_coarse_odd
    Nc = Xc*Yc*Zc*Tc

    # ---- Build DENSE A_c: [E, Nc, E, Nc] from probes ----
    dense = torch.zeros([E, Nc, E, Nc], dtype=dt, device=device)  # [row_j, site_j, col_e, site_c]
    str_Y=Yc*Zc*Tc; str_Z=Zc*Tc
    t0=torch.cuda.synchronize(); import time; t0=time.perf_counter()
    for c_idx in range(Nc):
        cx=c_idx//str_Y; rem=c_idx%str_Y; cy=rem//str_Z; rem%=str_Z; cz=rem//Tc; ct=rem%Tc
        for ee in range(E):
            src_c=torch.zeros([E,Xc,Yc,Zc,Tc],dtype=dt,device=device)
            src_c[ee,cx,cy,cz,ct]=1.0
            f=tools.prolong(local_ortho_null_vecs=lonv,coarse_vec=src_c)
            d=tools.restrict(local_ortho_null_vecs=lonv,fine_vec=S(f))
            dense[:, :, ee, c_idx] = d.reshape(E, Nc)
    print(f"dense build: {time.perf_counter()-t0:.1f}s")

    # ---- Compare: (dense A_c)·v vs operator-free ----
    v = torch.randn([E,Nc],dtype=dt,device=device)
    A_dense = torch.einsum('ijkl,kl->ij', dense, v)  # [E,Nc]
    v_c = v.reshape(E,Xc,Yc,Zc,Tc)
    f=tools.prolong(local_ortho_null_vecs=lonv,coarse_vec=v_c)
    A_op = tools.restrict(local_ortho_null_vecs=lonv,fine_vec=S(f)).reshape(E,Nc)
    err = tools.norm(A_dense-A_op)/tools.norm(A_op)
    print(f"dense A_c vs operator-free rel err = {err:.4e}")

    # ---- Check diagonal coupling strength: A_c[:, e, c, c+ex+ey] ----
    # for a random column, how much weight is on on-site vs nearest vs diagonal?
    col = dense[:, :, 0, 5].reshape(E, Xc,Yc,Zc,Tc)  # column e=0, site c=5
    c5x=5//str_Y; rem=5%str_Y; c5y=rem//str_Z; rem%=str_Z; c5z=rem//Tc; c5t=rem%Tc
    on = tools.norm(col[:, c5x,c5y,c5z,c5t])
    nn = 0.0; diag = 0.0; far = 0.0
    for dx in range(Xc):
        for dy in range(Yc):
            for dz in range(Zc):
                for dt2 in range(Tc):
                    dist = abs((dx-c5x+Xc)%Xc - 0)  # not a real dist; just classify
                    n = sum([(dx-c5x)%Xc!=0, (dy-c5y)%Yc!=0, (dz-c5z)%Zc!=0, (dt2-c5t)%Tc!=0])
                    w = tools.norm(col[:,dx,dy,dz,dt2])
                    if n==0: pass
                    elif n==1: nn += w
                    elif n==2: diag += w
                    else: far += w
    print(f"column e=0 site=({c5x},{c5y},{c5z},{c5t}): |on-site|={on:.4e} |nearest|={nn:.4e} |diagonal|={diag:.4e} |farther|={far:.4e}")

if __name__=="__main__":
    main()
