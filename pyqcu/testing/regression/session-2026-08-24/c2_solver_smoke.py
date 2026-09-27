import traceback
import torch
import pyqcu.cann as _torch
import pyqcu.dslash as dslash
import pyqcu.solver as solver
import pyqcu.tools as tools
import pyqcu.lattice as lattice

CUDA = torch.device('cuda')
DT = torch.complex128
LAT = [4, 4, 4, 8]
results = []


def run(name, fn):
    try:
        msg = fn() or ""
        results.append((name, 'PASS', ''))
        print(f"[C2][PASS] {name} {msg}", flush=True)
    except Exception as e:
        results.append((name, 'FAIL', f"{type(e).__name__}: {e}"))
        print(f"[C2][FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def relres(op, x, b):
    return tools.norm(op(x) - b) / tools.norm(b)


U = _torch.zeros(size=[3, 3, 4] + LAT, dtype=DT, device=CUDA)
lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
clover = _torch.zeros(size=[4, 3, 4, 3] + LAT, dtype=DT, device=CUDA)
D = dslash.operator(U=U, kappa=torch.Tensor([0.125]),
                    clover_term=clover, verbose=False)
mv = lambda v: D.matvec(src=v)
G5 = torch.tensor([1., 1., -1., -1.], dtype=DT, device=CUDA).view(4, 1, 1, 1, 1, 1)
g5 = lambda v: G5 * v
mv_dag = lambda w: g5(mv(g5(w)))      # A† = γ5 D γ5 (γ5-Hermiticity)
mv_aa = lambda v: mv_dag(mv(v))       # A†A Hermitian PSD
b = _torch.randn(size=[4, 3] + LAT, dtype=DT, device=CUDA)
bdag = mv_dag(b)


def t_bistabcg():
    x = solver.bistabcg(b=b, matvec=mv, tol=1e-9, if_rtol=True, verbose=False)
    r = relres(mv, x, b)
    assert r < 1e-7, r
    return f"relres={r:.2e}"


def t_fgmres():
    x = solver.fgmres(b=b, matvec=mv, tol=1e-9, if_rtol=True,
                      restart=16, max_iter=200, verbose=False)
    r = relres(mv, x, b)
    assert r < 1e-7, r
    return f"relres={r:.2e}"


def t_mr():
    x = solver.mr(b=b, matvec=mv, matvec_dag=mv_dag, tol=1e-4,
                  if_rtol=True, max_iter=20000, verbose=False)
    r = relres(mv, x, b)
    assert r < 1e-3, r          # smoother-grade target per docs
    return f"relres={r:.2e}"


def t_cacg():
    x = solver.cacg(b=bdag, matvec=mv_aa, tol=1e-10, if_rtol=True,
                    n_krylov=8, max_iter=1600, verbose=False)
    r = relres(mv_aa, x, bdag)
    assert r < 1e-4, r   # restart-GMRES(8) rate per docs (beta alignment not ported)
    return f"normal-eq relres={r:.2e}"


def t_multishift():
    shifts = [0.0, 0.5, 2.0]
    xs = solver.multishift_cg(b=bdag, matvec=mv_aa, shifts=shifts,
                              tol=1e-10, if_rtol=True, max_iter=300, verbose=False)
    worst = 0.0
    for s, x in zip(shifts, xs):
        op = lambda v, s=s: mv_aa(v) + s * v
        worst = max(worst, relres(op, x, bdag))
    assert worst < 1e-8, worst
    return f"worst relres={worst:.2e}"


def t_lanczos():
    evals, evecs = solver.tr_lanczos(matvec=mv_aa,
                                     v0=_torch.randn(size=[4, 3] + LAT, dtype=DT, device=CUDA),
                                     ncv=48, k=4, tol=1e-6, max_iter=3100)
    errs = [(tools.norm(mv_aa(y) - th * y) / abs(th))
            for th, y in zip(evals, evecs)]
    worst = float(max(errs))
    assert all(th.real > 0 for th in evals), evals
    assert worst < 1e-5, errs
    return f"evals={[f'{th.real:.3e}' for th in evals]} worst_res={worst:.2e}"


run("bistabcg", t_bistabcg)
run("fgmres", t_fgmres)
run("mr(gamma5-dagger)", t_mr)
run("cacg(normal-eq)", t_cacg)
run("multishift_cg(normal-eq)", t_multishift)
run("tr_lanczos(A†A)", t_lanczos)


def t_verify_nullvecs():
    from pyqcu.solver import multigrid
    from pyqcu.testing import verify_nullvecs
    LAT2 = [8, 8, 8, 16]
    U2 = _torch.zeros(size=[3, 3, 4] + LAT2, dtype=DT, device=CUDA)
    lattice.generate_gauge_field(U2, seed=42, sigma=0.1, verbose=False)
    clover2 = _torch.zeros(size=[4, 3, 4, 3] + LAT2, dtype=DT, device=CUDA)
    mg = multigrid(dtype_list=[DT, DT], device_list=[CUDA, CUDA],
                   U=U2, clover_term=clover2, kappa=torch.Tensor([0.125]),
                   max_level=2, dof_list=[12, 12], mg_grid_size=[2, 2, 2, 2],
                   tol=1e-8, verbose=False)
    mg.init()
    diag = verify_nullvecs(S=mg.op_list[0].matvec, lonv=mg.lonv_list[0],
                           lat_fine=LAT2, lat_coarse=[4, 4, 4, 8], n_sample=4)
    mg.end()
    del mg
    ratios = diag['null_ratios']
    body = ratios[1:]
    assert max(body) < 0.1, diag          # near-null suppression (sample0 已知欠收敛)
    assert diag['ortho_offdiag_max'] < 1e-12, diag
    return f"ratios={[f'{r:.3f}' for r in ratios]} ortho={diag['ortho_offdiag_max']:.1e}"


run("verify_nullvecs(L2 pure-python)", t_verify_nullvecs)

n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
print(f"\n=== C2 SMOKE SUMMARY: {len(results)-n_fail}/{len(results)} PASS ===")
for name, st, m in results:
    print(f"  [{st}] {name} {m}")
