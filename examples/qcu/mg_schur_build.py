#!/usr/bin/env python3
"""Build the Schur-consistent coarse operator A_c = P^T S P and validate the
production MG path (materialized coarse operator).

A_c is built by SINGLE-SITE probing: for each null vector e and each coarse
site c, prolong a delta, apply S, restrict → full column A_c[:, e, c]. The
on-site part goes to `sit`, forward-neighbor parts to `hop_plus`, backward
to `hop_minus` (matching multigrid_coarse_dslash kernel conventions).

Usage: source ./env.sh && python examples/qcu/mg_schur_build.py
"""
import torch, os, sys, re, io
from time import perf_counter
from contextlib import redirect_stdout
from pyqcu import tools, dslash, solver
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_pyref_expt import setup_gpu

def local_ortho_odd(null_vecs, lat_coarse_odd, verbose=False):
    """local_orthogonalize for ODD-site null vectors [E, e, X, Y, Z, T/2]."""
    return tools.local_orthogonalize(null_vecs=null_vecs,
                                     coarse_lat_size=lat_coarse_odd, verbose=verbose)

def build_schur_coarse_op(S, lonv, E, e, lat_fine_odd, lat_coarse_odd, dt, device):
    """A_c = P^T S P via single-site probing.
    lonv: [E, e, X, Y, Z, T/2] local-ortho'd (block structure [E,e,Xc,2,Yc,2,Zc,2,Tc,2]).
    Returns (hop_packed [2,4,E,E,Xc,Yc,Zc,Tc], sit [E,E,Xc,Yc,Zc,Tc]).
    """
    X, Y, Z, Th = lat_fine_odd
    Xc, Yc, Zc, Tc = lat_coarse_odd
    Nc = Xc*Yc*Zc*Tc
    sit = torch.zeros([E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop = torch.zeros([2, 4, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    # strides for coarse lattice (C-order, t fastest)
    str_Y = Yc*Zc*Tc; str_Z = Zc*Tc
    t0 = perf_counter()
    for c_idx in range(Nc):
        cx = c_idx // str_Y; rem = c_idx % str_Y
        cy = rem // str_Z; rem %= str_Z
        cz = rem // Tc; ct = rem % Tc
        for ee in range(E):
            src_c = torch.zeros([E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
            src_c[ee, cx, cy, cz, ct] = 1.0
            src_f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=src_c)
            dest_f = S(src_f)
            dest_c = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=dest_f)
            sit[:, ee, cx, cy, cz, ct] = dest_c[:, cx, cy, cz, ct]
            # Probe at site P gives the COLUMN dest_c[:, s] = A_c[:, e, s, P].
            #   hop_plus[d,:,e,P]  = A_c[:, e, P, fwd_d(P)]  = dest_c from probe at fwd(P)
            #   hop_minus[d,:,e,P] = A_c[:, e, P, bwd_d(P)]  = dest_c from probe at bwd(P)
            # So while probing P, fill the transpose-slot: hop_plus at bwd(P) and
            # hop_minus at fwd(P) (A_c[:, e, bwd(P), P] and A_c[:, e, fwd(P), P]).
            for d in range(4):
                dims=[Xc,Yc,Zc,Tc]
                # bwd neighbor of P: fill hop_plus there
                n = [cx, cy, cz, ct]; n[d] = (n[d]-1+dims[d]) % dims[d]
                hop[0, d, :, ee, n[0], n[1], n[2], n[3]] = dest_c[:, n[0], n[1], n[2], n[3]]
                # fwd neighbor of P: fill hop_minus there
                n = [cx, cy, cz, ct]; n[d] = (n[d]+1) % dims[d]
                hop[1, d, :, ee, n[0], n[1], n[2], n[3]] = dest_c[:, n[0], n[1], n[2], n[3]]
        if c_idx % 128 == 0 and c_idx > 0:
            print(f"    probing {c_idx}/{Nc} sites ({perf_counter()-t0:.1f}s)")
    print(f"  A_c build: {perf_counter()-t0:.1f}s for {E*Nc} probes")
    return hop, sit

def apply_coarse_matvec(hop, sit, v_c):
    """Apply the materialized coarse operator A_c (matches multigrid_coarse_dslash)."""
    E = v_c.shape[0]; Xc,Yc,Zc,Tc = v_c.shape[1:]
    out = torch.einsum("EeXYZT,eXYZT->EXYZT", sit, v_c).clone()
    str_Y=Yc*Zc*Tc; str_Z=Zc*Tc
    for d in range(4):
        dims=[Xc,Yc,Zc,Tc]; off=[str_Y,str_Z,Tc,1]
        fwd = torch.roll(v_c, shifts=-1, dims=d+1)  # in[fwd_site] = v_c at c+1
        bwd = torch.roll(v_c, shifts=1, dims=d+1)   # in[bwd_site] = v_c at c-1
        out += torch.einsum("EeXYZT,eXYZT->EXYZT", hop[0,d], fwd)
        out += torch.einsum("EeXYZT,eXYZT->EXYZT", hop[1,d], bwd)
    return out

def main():
    Lx,Ly,Lz,Lt = 8,8,8,16
    MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
    U_full, b_full, clover, KAPPA, av, (g,fi,ce,coo,cei,coi) = setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL)
    dt=torch.complex64; device=torch.device('cuda')
    op = dslash.operator(U=U_full, clover_term=clover, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity
    b_eo = tools.oooxyzt2poooxyzt(b_full.reshape([12]+list(b_full.shape)[2:]))
    b__o = op.give_b_parity(b_e=b_eo[0], b_o=b_eo[1])
    buf=io.StringIO()
    with redirect_stdout(buf):
        x_o_ref = solver.bistabcg(b=b__o, matvec=S, tol=ATOL, max_iter=1000, verbose=False)
    m=re.findall(r'Converged at iteration (\d+)', buf.getvalue())
    n_ref = int(m[0])+1 if m else '?'
    x_e_ref = op.give_x_e(b_e=b_eo[0], x_o=x_o_ref)
    x_ref = tools.poooxyzt2oooxyzt(torch.stack([x_e_ref, x_o_ref], dim=0)).reshape(b_full.shape)
    print(f"[ref] parity BiStabCG: iters={n_ref}")

    lat_fine_odd=[Lx,Ly,Lz,Lt//2]
    for E in [24, 48]:
        lat_coarse_odd=[Lx//2,Ly//2,Lz//2,Lt//4]
        _null = torch.randn([E,12]+lat_fine_odd, dtype=dt, device=device)
        t0=perf_counter()
        _null = tools.give_null_vecs(null_vecs=_null, matvec=S, bistabcg=None, verbose=False)
        print(f"  null vecs E={E}: {perf_counter()-t0:.1f}s")
        lonv = local_ortho_odd(_null, lat_coarse_odd)
        hop, sit = build_schur_coarse_op(S, lonv, E, 12, lat_fine_odd, lat_coarse_odd, dt, device)

        # ---- Validate A_c: materialized vs operator-free ----
        v_c = torch.randn([E]+lat_coarse_odd, dtype=dt, device=device)
        A_mat = apply_coarse_matvec(hop, sit, v_c)
        f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=v_c)
        A_opfree = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S(f))
        err = tools.norm(A_mat - A_opfree)/tools.norm(A_opfree)
        print(f"  A_c materialized vs operator-free rel err = {err:.3e}")

        # ---- Run Schur MG with the materialized A_c ----
        def coarse_matvec(v): return apply_coarse_matvec(hop, sit, v)
        for restart in [10]:
            x, n, vc, rn, bd = run_schur_mg(op, b_full, lonv, coarse_matvec, restart, ATOL)
            vs = tools.norm(x-x_ref)/tools.norm(x_ref)
            res = tools.norm(dslash.give_wilson(x,U_full,KAPPA,True)+dslash.give_clover(x,clover)-b_full)/tools.norm(b_full)
            print(f"[schurMG materialized E={E} r={restart}]: iters={n} vc={vc} res={res:.2e} vs_ref={vs:.2e} bd={bd}")

def run_schur_mg(op, b_full, lonv, coarse_matvec, num_restart, tol, max_iter=1000, coarse_tol=1e-8):
    dt=b_full.dtype; device=b_full.device
    b_eo = tools.oooxyzt2poooxyzt(b_full.reshape([12]+list(b_full.shape)[2:]))
    b_e, b_o = b_eo[0], b_eo[1]
    b__o = op.give_b_parity(b_e=b_e, b_o=b_o)
    S = op.matvec_parity
    x = torch.zeros_like(b__o)
    r = b__o.clone(); r_tilde = r.clone()
    p = torch.zeros_like(r); v = torch.zeros_like(r); s = torch.zeros_like(r); t = torch.zeros_like(r)
    rho_prev = torch.tensor(1.0, dtype=dt, device=device)
    alpha = torch.tensor(1.0, dtype=dt, device=device)
    omega = torch.tensor(1.0, dtype=dt, device=device)
    n=0; count_restart=0; vc=0; bd=False
    for it in range(max_iter):
        rho = tools.vdot(r_tilde, r)
        if abs(rho)<1e-30: bd=True; break
        beta=(rho/rho_prev)*(alpha/omega); rho_prev=rho
        p = r + beta*(p-omega*v)
        v = S(p)
        rtv = tools.vdot(r_tilde, v)
        if abs(rtv)<1e-30: bd=True; break
        alpha = rho/rtv
        s = r - alpha*v
        t = S(s)
        tts = tools.vdot(t,t)
        if abs(tts)<1e-30: bd=True; break
        omega = tools.vdot(t,s)/tts
        x = x + alpha*p + omega*s
        r = s - omega*t
        n+=1; count_restart+=1
        rn = tools.norm(r)
        if rn < tol: break
        if count_restart >= num_restart:
            r_c = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=r)
            e_c = solver.bistabcg(b=r_c, matvec=coarse_matvec, tol=coarse_tol, max_iter=300,
                                  x0=None, if_rtol=True, verbose=False)
            e_f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=e_c)
            x = x + e_f
            r = b__o - S(x)
            rn = tools.norm(r)
            count_restart=0; vc+=1
            r_tilde = r.clone(); p=torch.zeros_like(r); v=torch.zeros_like(r)
            s=torch.zeros_like(r); t=torch.zeros_like(r)
            rho_prev=torch.tensor(1.0,dtype=dt,device=device)
            alpha=torch.tensor(1.0,dtype=dt,device=device)
            omega=torch.tensor(1.0,dtype=dt,device=device)
            if rn < tol: break
    x_e = op.give_x_e(b_e=b_e, x_o=x)
    x_out = tools.poooxyzt2oooxyzt(torch.stack([x_e, x], dim=0)).reshape(b_full.shape)
    return x_out, n, vc, float(rn), bd

if __name__=="__main__":
    main()
