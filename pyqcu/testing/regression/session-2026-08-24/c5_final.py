import traceback
import torch
import pyqcu.cann as _torch
import pyqcu.tools as tools
import pyqcu.lattice as lattice

CUDA = torch.device('cuda')
DT = torch.complex64
LAT = [8, 8, 8, 16]
results = []


def run(name, fn):
    try:
        msg = fn() or ""
        results.append((name, 'PASS', ''))
        print(f"[C5F][PASS] {name} {msg}", flush=True)
    except Exception as e:
        results.append((name, 'FAIL', f"{type(e).__name__}: {e}"))
        print(f"[C5F][FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


U = _torch.zeros(size=[3, 3, 4] + LAT, dtype=DT, device=CUDA)
lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)

from pyqcu.smear import wuppertal_smear


def t_wtal_const_freefield():
    # U=I（自由场）：核退化为普通高斯平均 ⇒ 常数场是精确不动点
    I4 = torch.eye(3, dtype=DT, device=CUDA)
    U_eye = I4[:, :, None, None, None, None, None].expand(3, 3, 4, *LAT).contiguous()
    c = _torch.randn(size=[4, 3] + LAT, dtype=DT, device=CUDA)
    const = c[0, 0, 0, 0, 0, 0].expand_as(c).contiguous()
    out = wuppertal_smear(const, U_eye, rho=4.0, nstep=10)
    dev = float((out - const).abs().max().item())
    import math
    predict = abs((1 - 6*0.1) + 8*0.1)**10 - 1   # 若 4 维全 smear 而系数按 6 邻居: 每步 x(1+2sigma)
    print(f"    [diag] dev={dev:.3e}  4dim-mismatch_predict_ratio={(abs(1.2)**10):.1f}", flush=True)
    assert dev < 1e-4, f"free-field constant not fixed: dev={dev:.2e} (predict x{predict:.1f} if 6-vs-8 neighbor mismatch)"
    return f"free_field_max_dev={dev:.2e}"


def t_wtal_default_shrink():
    src = _torch.randn(size=[4, 3] + LAT, dtype=DT, device=CUDA)
    out = wuppertal_smear(src, U, rho=4.0, nstep=40)   # 默认 σ=0.1<1/6 收缩域
    r_in = float(tools.norm(src)); r_out = float(tools.norm(out))
    shrink = r_out / r_in
    assert shrink < 1.0, f"white noise norm must shrink (docs), got ratio={shrink:.3f}"
    return f"norm_ratio={shrink:.3f} (<1, 高频压制)"


def t_wtal_nstep0():
    try:
        wuppertal_smear(_torch.randn(size=[4, 3] + LAT, dtype=DT, device=CUDA),
                        U, rho=4.0, nstep=0)
        return "no crash (unexpected)"
    except AssertionError as e:
        assert "nstep" in str(e), e
        return f"guarded: {str(e).splitlines()[-1].strip()[:60]}"
    except ZeroDivisionError:
        raise AssertionError("still raw ZeroDivisionError")


run("wtal_const_fixed_point_freefield", t_wtal_const_freefield)
run("wtal_default_norm_shrinks", t_wtal_default_shrink)
run("wtal_nstep0_guard", t_wtal_nstep0)

# Gauss gauge：conftest 生产协议（缓冲形状 = [2,3,3,4]+lat_shape(params)）


def t_gauss_conftest_protocol():
    from pyqcu.cuda import define
    import pyqcu.cuda.qcu as qcu_mod
    params = torch.zeros(54, dtype=torch.int32)
    set_ptrs = torch.zeros(100, dtype=torch.int64)
    argv = torch.zeros(7, dtype=torch.float32)
    params[define._LAT_X_] = LAT[0]; params[define._LAT_Y_] = LAT[1]
    params[define._LAT_Z_] = LAT[2]; params[define._LAT_T_] = LAT[3]
    params[define._GRID_X_] = params[define._GRID_Y_] = 1
    params[define._GRID_Z_] = params[define._GRID_T_] = 1
    params[define._PARITY_] = 0; params[define._NODE_RANK_] = 0; params[define._NODE_SIZE_] = 1
    params[define._DAGGER_] = 0; params[define._MAX_ITER_] = 1000
    params[define._DATA_TYPE_] = define._LAT_C64_
    params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = -1
    params[define._VERBOSE_] = 0; params[define._SEED_] = 42
    argv = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
    argv[define._MASS_] = 0.05; argv[define._ATOL_] = 1e-9; argv[define._SIGMA_] = 0.1
    g = torch.empty([2, 3, 3, 4] + define.lat_shape(params),
                    dtype=torch.complex64, device=CUDA)
    qcu_mod.applyInitQcu(set_ptrs, params, argv)
    qcu_mod.applyGaussGaugeQcu(g, set_ptrs, params)
    torch.cuda.synchronize()
    s0 = bool(lattice.check_su3(U=g[0])); s1 = bool(lattice.check_su3(U=g[1]))
    assert s0 and s1, (s0, s1)
    return "su3(e)=True su3(o)=True"


run("cpp_gauss_gauge_su3(conftest-protocol)", t_gauss_conftest_protocol)

n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
print(f"\n=== C5 FINAL SUMMARY: {len(results)-n_fail}/{len(results)} PASS ===")
for name, st, m in results:
    print(f"  [{st}] {name} {m}")
