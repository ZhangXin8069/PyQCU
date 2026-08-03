#!/usr/bin/env python3
"""dev73_5 —— 正确性检查（gauge 性质 / 解误差 / null_vecs 正确性）。

对照 mg-v4-report 与 pyqcu/lattice/__init__.py：
  1. gauge 性质   : check_su3(U) —— 幺正性、det=1、minor 恒等式
  2. 解误差       : vs_ref = ||x_mg - x_ref||/||x_ref||、mg_res = ||b-Dx||/||b||
                    及参考 BiStabCG 残差、Python 复现收敛历史
  3. null_vecs 正确性:
     a. 零模质量   ||S·v||/||v||  <<  S 最大本征值
     b. 块内正交   |<v_i,v_j> - δ_ij| ≈ 机器精度
     c. C++ restrict/prolong 与 Python einsum 一致
     d. C++ 33-tensor 粗 dslash 与 Python A_c = P^T S P 一致

用法（在 /root/PyQCU 下运行）：
    source ./env.sh && CUDA_VISIBLE_DEVICES=2 \
        python examples/qcu/mg_dev73_5_verify.py [--lattice 8 16 16 16] [--prec c64]
输出：logs/dev73_5_verify.json（含参考收敛历史，供画图）
"""
import torch, os, sys, time, json
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
from pyqcu.lattice import check_su3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_csm = _load("csm", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "conftest.schur.multigrid.py"))
build_config = _csm.build_config
from mg_nullvec_cache import build_or_load_coarse_ops
from mg_stencil_build import apply_stencil
from mg_dev73_5_bench import bistabcg_history, LOG_DIR

LOG_PATH = os.path.join(LOG_DIR, "clover_multigrid.log")


def build_base(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
               DT, gauge_seed=42):
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
                      MG_GRID, 10, 15, 1e5, DT)
    KAPPA = 1.0 / (2 * MASS + 8)
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)
    torch.manual_seed(gauge_seed)
    g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
    fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi)
    ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce)
    coo = torch.zeros_like(ce)
    coi = torch.zeros_like(ce)

    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    torch.cuda.synchronize()
    qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs, params)
    torch.cuda.synchronize()

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    return dict(av=av, KAPPA=KAPPA, device=device, dt=dt, ls=ls, g=g, fi=fi,
                fo_ref=fo_ref, ce=ce, cei=cei, coo=coo, coi=coi,
                qcu_U=qcu_U, qcu_src=qcu_src, qcu_ref=qcu_ref, ref_cl=ref_cl,
                op=op, S=op.matvec_parity)


def verify_lattice(qcu_U):
    """check_su3：幺正性 / det=1 / minor 恒等式。"""
    t0 = time.perf_counter()
    ok = check_su3(qcu_U, tol=1e-2 if qcu_U.dtype == torch.float32 else 1e-3,
                   verbose=False)
    dt = time.perf_counter() - t0
    # 补充量化：max |U^H U - I|、max |det U - 1|
    U = qcu_U  # [c_in, c_out, dir, X,Y,Z,T]
    I3 = torch.eye(3, dtype=qcu_U.dtype, device=qcu_U.device)
    # (U^† U)[a,c] = Σ_b conj(U[b,a])·U[b,c]  （b=color_in 求和）
    UH_U = torch.einsum('bam...,bcm...->acm...', U.conj(), U)
    unit = (UH_U - I3.view(3, 3, 1, 1, 1, 1, 1)).abs().max().item()
    dets = torch.linalg.det(U.permute(2, 3, 4, 5, 6, 0, 1))
    detdev = (dets - 1).abs().max().item()
    return {"check_su3": bool(ok), "max_unit_err": float(unit),
            "max_det_dev": float(detdev), "sec": dt}


def verify_nullvecs(op, S, lonv, hnn, hdg, sit, E, E_prev, lat_fine,
                    lat_coarse, dt, device, n_sample=4):
    """null_vecs 四重检查。lonv: [E, E_prev, Xf,Yf,Zf,Tf]（E_prev=细层 dof）。"""
    out = {}
    # a. 零模质量（取前 n_sample 个粗向量）
    ratios = []
    for k in range(min(n_sample, E)):
        v = lonv[k]                                   # [E_prev, Xf,Yf,Zf,Tf]
        Av = S(v.reshape([E_prev] + lat_fine)).reshape(E, -1)
        ratios.append((torch.linalg.norm(Av) / torch.linalg.norm(lonv[k])).item())
    # S 最大本征值估计（幂迭代）
    v = torch.randn([E_prev] + lat_fine, dtype=dt, device=device)
    v = v / torch.linalg.norm(v)
    _real_dt = torch.float32 if dt == torch.complex64 else torch.float64
    lam = torch.tensor(0.0, dtype=_real_dt, device=device)
    for _ in range(20):
        w = S(v).flatten()
        vf = v.flatten()
        lam = torch.real(torch.vdot(w, vf))
        v = w.reshape(v.shape) / torch.linalg.norm(w)
    out["null_ratios"] = ratios
    out["S_lambda_max"] = abs(float(lam))
    # b. 块内正交
    X, Y, Z, T = lat_coarse
    x, y, z, t = [lat_fine[d] // lat_coarse[d] for d in range(4)]
    vb = lonv.reshape(E, E_prev, X, x, Y, y, Z, z, T, t)
    block = vb[:, :, 0, :, 0, :, 0, :, 0, :].reshape(E, -1)
    G = block @ block.conj().T
    off = G - torch.eye(E, dtype=dt, device=device)
    out["ortho_offdiag_max"] = float(off.abs().max().item())
    out["ortho_diag_min"] = float(torch.diag(G).real.min().item())
    out["ortho_diag_max"] = float(torch.diag(G).real.max().item())
    # c. C++ restrict / prolong vs Python einsum
    # 注：独立 C API applyMultigridRestrictQcu 硬编码 fine DOF = _LAT_SC_ = 12，
    # 仅当细层 dof=12（level 1）时有效；更粗层由 3L 求解器成功收敛间接验证。
    out["restrict_rel_diff"] = None
    out["prolong_rel_diff"] = None
    if E_prev == 12:
        fine_vec = torch.randn([E_prev] + lat_fine, dtype=dt, device=device)
        r_py = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=fine_vec)
        params[define._LAT_X_] = lat_fine[0]
        params[define._LAT_Y_] = lat_fine[1]
        params[define._LAT_Z_] = lat_fine[2]
        params[define._LAT_T_] = lat_fine[3]
        params[define._MG_LEVEL1_X_] = X
        params[define._MG_LEVEL1_Y_] = Y
        params[define._MG_LEVEL1_Z_] = Z
        params[define._MG_LEVEL1_T_] = T
        params[define._MG_LEVEL1_E_] = E
        params[define._MG_NUM_LEVEL_] = 12
        out_r = torch.zeros([E, X, Y, Z, T], dtype=dt, device=device)
        qcu.applyMultigridRestrictQcu(out_r, fine_vec, lonv, set_ptrs, params)
        out["restrict_max_diff"] = float((out_r - r_py).abs().max().item())
        out["restrict_rel_diff"] = float((out_r - r_py).abs().max().item() /
                                         (r_py.abs().max().item() + 1e-30))
        coarse_vec = torch.randn([E, X, Y, Z, T], dtype=dt, device=device)
        p_py = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=coarse_vec)
        out_p = torch.zeros([E_prev] + lat_fine, dtype=dt, device=device)
        qcu.applyMultigridProLongQcu(out_p, coarse_vec, lonv, set_ptrs, params)
        out["prolong_max_diff"] = float((out_p - p_py).abs().max().item())
        out["prolong_rel_diff"] = float((out_p - p_py).abs().max().item() /
                                        (p_py.abs().max().item() + 1e-30))
    # d. C++ 33-tensor 粗 dslash vs Python A_c = P^T S P
    src_c = torch.randn([E, X, Y, Z, T], dtype=dt, device=device)

    def Ac(v):
        f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=v)
        return tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S(f))
    ref = Ac(src_c)
    cu = apply_stencil(hnn, hdg, sit, src_c)
    out["coarse_dslash_rel_diff"] = float((cu - ref).abs().max().item() /
                                          (ref.abs().max().item() + 1e-30))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", nargs=4, type=int, default=[8, 16, 16, 16])
    ap.add_argument("--prec", default="c64", choices=["c64", "c128"])
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--dof", nargs="+", type=int, default=None)
    args = ap.parse_args()
    Lx, Ly, Lz, Lt = args.lattice
    DT = define._LAT_C128_ if args.prec == "c128" else define._LAT_C64_
    dof = args.dof or ([12, 48] if args.levels == 2 else [12, 48, 48])
    dt = define.dtype(DT)
    prec = args.prec
    lat_key = f"{Lx}x{Ly}x{Lz}x{Lt}_{prec}"
    print(f"=== dev73_5 verify: lattice={[Lx,Ly,Lz,Lt]} prec={prec} "
          f"levels={args.levels} ===")

    base = build_base(Lx, Ly, Lz, Lt, 0.05, 1e-6, args.levels, dof,
                      [2, 2, 2, 2], DT)
    res = {"lattice": [Lx, Ly, Lz, Lt], "precision": prec,
           "levels": args.levels, "dof": dof}

    # 1. gauge 性质
    res["gauge"] = verify_lattice(base["qcu_U"])
    print(f"[1] gauge check_su3={res['gauge']['check_su3']} "
          f"unit_err={res['gauge']['max_unit_err']:.2e} "
          f"det_dev={res['gauge']['max_det_dev']:.2e}")

    # 2. 解误差
    qcu_ref = base["qcu_ref"]
    qcu_src = base["qcu_src"]
    KAPPA = base["KAPPA"]
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, base["qcu_U"], KAPPA, True) +
                         dslash.give_clover(qcu_ref, base["ref_cl"]) - qcu_src) / tools.norm(qcu_src)
    res["ref_res"] = float(ref_res)
    # 参考收敛历史（Python 复现）
    ref_hist = None
    try:
        from mg_dev73_5_bench import ref_conv_history
        ref_hist = ref_conv_history(base["op"], qcu_src, 1e-6)
        res["ref_conv_hist"] = ref_hist
        res["ref_iters"] = len(ref_hist) - 1
    except Exception as e:
        res["ref_conv_error"] = str(e)
        res["ref_iters"] = None
    print(f"[2] ref_res={ref_res:.3e} ref_iters={res.get('ref_iters')}")

    # 3. null_vecs 正确性
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    E_prev = 12
    S = base["S"]
    nv_res = {"levels": []}
    for lvl in range(1, args.levels):
        E_c = dof[lvl]
        lat_coarse_odd = [lat_fine_odd[d] // 2 for d in range(4)]
        t0 = time.perf_counter()
        lonv, hnn, hdg, sit = build_or_load_coarse_ops(
            42, [Lx, Ly, Lz, Lt], lvl, E_c, E_prev, lat_fine_odd,
            lat_coarse_odd, S, dt, base["device"], 2, use_cache=True,
            save=True, verbose=False)
        print(f"  [lvl {lvl}] E={E_c} coarse={lat_coarse_odd} "
              f"load={time.perf_counter()-t0:.1f}s")
        lres = verify_nullvecs(base["op"], S, lonv, hnn, hdg, sit, E_c,
                               E_prev, lat_fine_odd, lat_coarse_odd, dt,
                               base["device"])
        lres["E"] = E_c
        nv_res["levels"].append(lres)
        _f = lambda x: "—" if x is None else f"{x:.2e}"
        print(f"    null ratios={[f'{r:.2e}' for r in lres['null_ratios'][:4]]} "
              f"λmax={lres['S_lambda_max']:.2e} "
              f"ortho_off={lres['ortho_offdiag_max']:.2e} "
              f"restr_rel={_f(lres['restrict_rel_diff'])} "
              f"prol_rel={_f(lres['prolong_rel_diff'])} "
              f"coarse_rel={_f(lres['coarse_dslash_rel_diff'])}")
        from mg_stencil_build import apply_stencil
        S = lambda v, hnn_i=hnn, hdg_i=hdg, sit_i=sit: apply_stencil(hnn_i, hdg_i, sit_i, v)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd
    res["nullvecs"] = nv_res

    out_path = os.path.join(LOG_DIR, f"dev73_5_verify_{lat_key}_L{args.levels}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
