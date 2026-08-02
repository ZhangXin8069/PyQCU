#!/usr/bin/env python3
"""Validate the Schur-consistent MG concept.

Level 0: S·x_o = b__o (parity Schur), BiStabCG smoothing.
Coarse space: null vectors of S (NOT D). Coarse operator A_c = P^T S P,
applied operator-free (prolong -> S -> restrict) to validate the concept.
V-cycle: r -> P^T r -> solve coarse -> P e -> x += e -> r = b__o - S x.

If this converges in far fewer iterations than plain parity BiStabCG (87 on
8x16x16x16), the concept is validated and worth a C++ implementation.
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

def schur_mg(op, b_full, lat_fine, lat_coarse_odd, dof_coarse, num_restart=5,
             tol=1e-6, max_iter=1000, coarse_tol=1e-6, dof_fine=12):
    """Custom Schur-consistent MG (operator-free coarse solve)."""
    dt = b_full.dtype; device = b_full.device
    b_eo = tools.oooxyzt2poooxyzt(b_full.reshape([dof_fine]+list(b_full.shape)[2:]))
    b_e, b_o = b_eo[0], b_eo[1]
    b__o = op.give_b_parity(b_e=b_e, b_o=b_o)
    S = op.matvec_parity

    # ---- Build S null vectors (odd-site) ----
    X, Y, Z, T = lat_fine
    T_half = T // 2
    # odd-site template: [E, 12, X, Y, Z, T/2]
    _null = torch.randn([dof_coarse, dof_fine, X, Y, Z, T_half], dtype=dt, device=device)
    _null = tools.give_null_vecs(null_vecs=_null, matvec=S, bistabcg=None, verbose=False)
    # local ortho over coarse ODD lattice [X/2,Y/2,Z/2,T/4]
    Xc,Yc,Zc,Tc = lat_coarse_odd
    lonv = tools.local_orthogonalize(null_vecs=_null, coarse_lat_size=[Xc,Yc,Zc,Tc], verbose=False)
    # lonv shape [E, 12, Xc, 2, Yc, 2, Zc, 2, Tc, 2] -> flatten to odd-site layout
    E = lonv.shape[0]
    P = lonv  # prolongation: coarse [E,Xc,Yc,Zc,Tc] -> fine odd [12,X,Y,Z,T/2]

    def apply_coarse_A(v_c):
        """A_c · v_c = P^T S (P v_c)  (operator-free Galerkin)."""
        f = tools.prolong(local_ortho_null_vecs=P, coarse_vec=v_c)
        Sf = S(f)
        return tools.restrict(local_ortho_null_vecs=P, fine_vec=Sf)

    # ---- BiStabCG + V-cycle ----
    x = torch.zeros_like(b__o)
    r = b__o.clone(); r_tilde = r.clone()
    p = torch.zeros_like(r); v = torch.zeros_like(r); s = torch.zeros_like(r); t = torch.zeros_like(r)
    rho_prev = torch.tensor(1.0, dtype=dt, device=device)
    alpha = torch.tensor(1.0, dtype=dt, device=device)
    omega = torch.tensor(1.0, dtype=dt, device=device)
    n = 0; count_restart = 0; vc = 0; breakdown = False
    for it in range(max_iter):
        rho = tools.vdot(r_tilde, r)
        if abs(rho) < 1e-30: breakdown=True; break
        beta = (rho/rho_prev)*(alpha/omega); rho_prev = rho
        p = r + beta*(p - omega*v)
        v = S(p)
        rtv = tools.vdot(r_tilde, v)
        if abs(rtv) < 1e-30: breakdown=True; break
        alpha = rho/rtv
        s = r - alpha*v
        t = S(s)
        tts = tools.vdot(t,t)
        if abs(tts) < 1e-30: breakdown=True; break
        omega = tools.vdot(t,s)/tts
        x = x + alpha*p + omega*s
        r = s - omega*t
        n += 1; count_restart += 1
        rn = tools.norm(r)
        if rn < tol: break
        if count_restart >= num_restart:
            r_coarse = tools.restrict(local_ortho_null_vecs=P, fine_vec=r)
            e_coarse = solver.bistabcg(b=r_coarse, matvec=apply_coarse_A, tol=coarse_tol,
                                       max_iter=200, x0=None, if_rtol=True, verbose=False)
            e_fine = tools.prolong(local_ortho_null_vecs=P, coarse_vec=e_coarse)
            x = x + e_fine
            r = b__o - S(x)
            rn = tools.norm(r)
            count_restart = 0; vc += 1
            r_tilde = r.clone(); p = torch.zeros_like(r); v = torch.zeros_like(r)
            s = torch.zeros_like(r); t = torch.zeros_like(r)
            rho_prev = torch.tensor(1.0, dtype=dt, device=device)
            alpha = torch.tensor(1.0, dtype=dt, device=device)
            omega = torch.tensor(1.0, dtype=dt, device=device)
            if rn < tol: break
    x_e = op.give_x_e(b_e=b_e, x_o=x)
    x_out = tools.poooxyzt2oooxyzt(torch.stack([x_e, x], dim=0)).reshape(b_full.shape)
    return x_out, n, vc, float(rn), breakdown, E

def main():
    Lx,Ly,Lz,Lt = 8,8,8,16   # concept test lattice (cheaper)
    MASS=0.05; ATOL=1e-6; KAPPA=1.0/(2*MASS+8)
    U_full, b_full, clover, KAPPA, av, (g,fi,ce,coo,cei,coi) = setup_gpu(Lx,Ly,Lz,Lt,MASS,ATOL)
    dt=torch.complex64; device=torch.device('cuda')
    op = dslash.operator(U=U_full, clover_term=clover, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    b_eo = tools.oooxyzt2poooxyzt(b_full.reshape([12]+list(b_full.shape)[2:]))
    b__o = op.give_b_parity(b_e=b_eo[0], b_o=b_eo[1])
    # reference parity BiStabCG
    buf=io.StringIO()
    with redirect_stdout(buf):
        x_o_ref = solver.bistabcg(b=b__o, matvec=op.matvec_parity, tol=ATOL, max_iter=1000, verbose=False)
    m=re.findall(r'Converged at iteration (\d+)', buf.getvalue())
    n_ref = int(m[0])+1 if m else '?'
    x_e_ref = op.give_x_e(b_e=b_eo[0], x_o=x_o_ref)
    x_ref = tools.poooxyzt2oooxyzt(torch.stack([x_e_ref, x_o_ref], dim=0)).reshape(b_full.shape)
    print(f"[ref] parity BiStabCG on {Lx}x{Ly}x{Lz}x{Lt}: iters={n_ref}")

    lat_fine=[Lx,Ly,Lz,Lt]
    lat_coarse_odd=[Lx//2,Ly//2,Lz//2,Lt//4]   # coarse odd lattice (T_half//2)
    for dof in [24, 48]:
        for restart in [5, 10]:
            x, n, vc, rn, bd, E = schur_mg(op, b_full, lat_fine, lat_coarse_odd, dof,
                                            num_restart=restart, tol=ATOL, coarse_tol=1e-6)
            vs = tools.norm(x-x_ref)/tools.norm(x_ref)
            res = tools.norm(dslash.give_wilson(x,U_full,KAPPA,True)+dslash.give_clover(x,clover)-b_full)/tools.norm(b_full)
            print(f"[schurMG dof={dof} r={restart}]: iters={n} vc={vc} res={res:.2e} vs_ref={vs:.2e} bd={bd}")

if __name__=="__main__":
    main()
