#!/usr/bin/env python3
"""Strategy experiment: can a Schur-level-0 MG beat parity BiStabCG?

Level-0 solves S·x_o = b__o (parity Schur). V-cycle every num_restart.
Null vectors from the FULL operator D (standard, matching _multigrid.py).
Tests Krylov reset policy (full reset vs keep directions) and coarse
solve accuracy. Measures level-0 iterations to reach tol.

This determines the C++ MG design.
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

def custom_mg(op, b_full, lonv, coarse_op, num_restart=5, tol=1e-6, max_iter=2000,
              keep_krylov=False, coarse_tol=1e-8):
    dt = b_full.dtype; device = b_full.device
    b_eo = tools.oooxyzt2poooxyzt(b_full.reshape([12]+list(b_full.shape)[2:]))
    b_e, b_o = b_eo[0], b_eo[1]
    b_origin = b_full.reshape([12]+list(b_full.shape)[2:]).clone()
    b__o = op.give_b_parity(b_e=b_e, b_o=b_o)
    S = op.matvec_parity
    x = torch.zeros_like(b__o)
    r = b__o.clone(); r_tilde = r.clone()
    p = torch.zeros_like(r); v = torch.zeros_like(r); s = torch.zeros_like(r); t = torch.zeros_like(r)
    rho_prev = torch.tensor(1.0, dtype=dt, device=device)
    alpha = torch.tensor(1.0, dtype=dt, device=device)
    omega = torch.tensor(1.0, dtype=dt, device=device)
    n = 0; count_restart = 0; vc_applied = 0
    breakdown = False
    for it in range(max_iter):
        rho = tools.vdot(r_tilde, r)
        if abs(rho) < 1e-30: breakdown=True; break
        beta = (rho/rho_prev)*(alpha/omega); rho_prev = rho
        p = r + beta*(p - omega*v)
        v = S(p)
        rtv = tools.vdot(r_tilde, v)
        if abs(rtv) < 1e-30: breakdown=True; break
        alpha = rho / rtv
        s = r - alpha*v
        t = S(s)
        tts = tools.vdot(t,t)
        if abs(tts) < 1e-30: breakdown=True; break
        omega = tools.vdot(t,s)/tts
        x = x + alpha*p + omega*s
        r = s - omega*t
        n += 1; count_restart += 1
        rn = tools.norm(r)
        if rn < tol:
            break
        if count_restart >= num_restart:
            # full residual: reconstruct x_e, compute r_full
            x_e = op.give_x_e(b_e=b_e, x_o=x)
            x_origin = tools.poooxyzt2oooxyzt(torch.stack([x_e, x], dim=0))
            r_full = b_origin - op.matvec(x_origin)
            r_coarse = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=r_full)
            e_coarse = solver.bistabcg(b=r_coarse, matvec=coarse_op.matvec, tol=coarse_tol,
                                       max_iter=600, x0=None, if_rtol=False, verbose=False)
            e_fine = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=e_coarse)
            e_fine_odd = tools.oooxyzt2poooxyzt(e_fine)[1]
            x = x + e_fine_odd
            r = b__o - S(x)
            rn = tools.norm(r)
            count_restart = 0; vc_applied += 1
            r_tilde = r.clone()
            if not keep_krylov:
                p = torch.zeros_like(r); v = torch.zeros_like(r)
                s = torch.zeros_like(r); t = torch.zeros_like(r)
                rho_prev = torch.tensor(1.0, dtype=dt, device=device)
                alpha = torch.tensor(1.0, dtype=dt, device=device)
                omega = torch.tensor(1.0, dtype=dt, device=device)
            if rn < tol:
                break
    x_e = op.give_x_e(b_e=b_e, x_o=x)
    x_out = tools.poooxyzt2oooxyzt(torch.stack([x_e, x], dim=0)).reshape(b_full.shape)
    return x_out, n, vc_applied, float(rn), breakdown

def main():
    Lx,Ly,Lz,Lt = 8,16,16,16
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
        t0=perf_counter()
        x_o_ref = solver.bistabcg(b=b__o, matvec=op.matvec_parity, tol=ATOL, max_iter=1000, verbose=False)
        t_ref=perf_counter()-t0
    m=re.findall(r'Converged at iteration (\d+)', buf.getvalue())
    n_ref = int(m[0])+1 if m else '?'
    x_e_ref = op.give_x_e(b_e=b_eo[0], x_o=x_o_ref)
    x_ref = tools.poooxyzt2oooxyzt(torch.stack([x_e_ref, x_o_ref], dim=0)).reshape(b_full.shape)
    print(f"[ref] parity BiStabCG: {t_ref*1000:.1f} ms  iters={n_ref}")

    lat_fine=[Lx,Ly,Lz,Lt]; lat_coarse=[Lx//2,Ly//2,Lz//2,Lt//2]
    dof_coarse=24
    _null = torch.randn([dof_coarse,12]+lat_fine, dtype=dt, device=device)
    _null = tools.give_null_vecs(null_vecs=_null, matvec=op.matvec, bistabcg=None, verbose=False)
    lonv = tools.local_orthogonalize(null_vecs=_null, coarse_lat_size=lat_coarse, verbose=False)
    coarse_op = dslash.operator(fine_hopping=op.hopping, fine_sitting=op.sitting,
                                local_ortho_null_vecs=lonv, verbose=False)

    configs = [
        ("plain BiStabCG parity (no vcycle)", dict(num_restart=10**9, keep_krylov=False, coarse_tol=1e-8)),
        ("MG reset r=5",    dict(num_restart=5,  keep_krylov=False, coarse_tol=1e-8)),
        ("MG reset r=10",   dict(num_restart=10, keep_krylov=False, coarse_tol=1e-8)),
        ("MG keep r=5",     dict(num_restart=5,  keep_krylov=True,  coarse_tol=1e-8)),
        ("MG keep r=10",    dict(num_restart=10, keep_krylov=True,  coarse_tol=1e-8)),
        ("MG keep r=20",    dict(num_restart=20, keep_krylov=True,  coarse_tol=1e-8)),
    ]
    for name, kw in configs:
        x, n, vc, rn, bd = custom_mg(op, b_full, lonv, coarse_op, **kw)
        vs = tools.norm(x-x_ref)/tools.norm(x_ref)
        res = tools.norm(dslash.give_wilson(x,U_full,KAPPA,True)+dslash.give_clover(x,clover)-b_full)/tools.norm(b_full)
        print(f"[{name:28s}]: iters={n:4d} vc={vc:3d} res={res:.2e} vs_ref={vs:.2e} breakdown={bd}")

if __name__=="__main__":
    main()
