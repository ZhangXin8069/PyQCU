#!/usr/bin/env python3
"""Element-wise debug of A_c = P^T S P materialization vs operator-free."""
import torch, os, sys
from pyqcu import tools, dslash, solver
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

    # Compare on multiple random vectors
    Xc,Yc,Zc,Tc = lat_coarse_odd
    for trial in range(3):
        v_c = torch.randn([E]+lat_coarse_odd, dtype=dt, device=device)
        A_mat = apply_coarse_matvec(hop, sit, v_c)
        f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=v_c)
        A_op = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S(f))
        diff = (A_mat - A_op)
        err = tools.norm(diff)/tools.norm(A_op)
        # find the largest discrepancy site
        flat = diff.reshape(E, -1).abs()          # [E, Nsites]
        site_idx = flat.max(dim=0).values.argmax().item()
        e_idx = flat[:, site_idx].argmax().item()
        # decode site
        cx = site_idx//(Yc*Zc*Tc); rem=site_idx%(Yc*Zc*Tc)
        cy = rem//(Zc*Tc); rem%=Zc*Tc
        cz = rem//Tc; ct = rem%Tc
        print(f"trial {trial}: rel err={err:.4e}  largest at e={e_idx} site=({cx},{cy},{cz},{ct})")
        print(f"   A_mat[e_idx, cx,cy,cz,ct]={A_mat[e_idx,cx,cy,cz,ct]}")
        print(f"   A_op [e_idx, cx,cy,cz,ct]={A_op[e_idx,cx,cy,cz,ct]}")
        # Check if diff is localized to boundaries or spread
        diff_norm_per_site = diff.reshape(E, Xc,Yc,Zc,Tc).abs().sum(dim=0)
        print(f"   per-site |diff| sum (x-slices): {[float(diff_norm_per_site[xi,:,:,:].sum()) for xi in range(Xc)]}")
        print(f"   per-site |diff| sum (t-slices): {[float(diff_norm_per_site[:,:,:,ti].sum()) for ti in range(Tc)]}")

if __name__=="__main__":
    main()
