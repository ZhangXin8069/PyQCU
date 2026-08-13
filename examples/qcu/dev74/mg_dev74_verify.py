#!/usr/bin/env python3
"""dev74 —— 正确性验证（dev73_5 全套 + CudaSchurOp 对照）。

复用 mg_dev73_5_verify 的 gauge/解/null_vecs 检查，新增 dev74 专项：
  5. CudaSchurOp（applyCloverBistabCgDslashQcu）与 Python matvec_parity
     的数值一致性（rel err）与单次调用耗时对比 —— 多线程版本的算子正确性依据

用法：
    source ./env.sh && python examples/qcu/mg_dev74_verify.py [--lattice 8 8 8 16]
输出：logs/dev74/dev74_verify.json
"""
import torch, os, sys, time, json
from pyqcu import tools, dslash
import pyqcu.cuda.define as define

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_dev73_5_verify import build_base, verify_lattice, verify_nullvecs, LOG_DIR
from mg_dev74_dslash import make_cuda_schur_ops


def verify_dslash_op(base, n_trial=8):
    """CudaSchurOp vs Python matvec_parity：一致性 + 耗时。"""
    dt, device = base["dt"], base["device"]
    ls = base["ls"]
    x_o = torch.randn([12] + ls, dtype=dt, device=device)
    y_py = base["S"](x_o)
    ops = make_cuda_schur_ops(base["av"], base["g"], base["ce"], base["coo"],
                              base["cei"], base["coi"], n=1)
    y_cpp = ops[0].matvec(x_o)
    err = float(tools.norm(y_cpp - y_py) / tools.norm(y_py))

    def med(f, n):
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            f()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2] * 1000

    t_py = med(lambda: base["S"](x_o), n_trial)
    t_cpp = med(lambda: ops[0].matvec(x_o), n_trial)
    for o in ops:
        o.release()
    return {"rel_err": float(err), "t_py_ms": round(t_py, 3),
            "t_cpp_ms": round(t_cpp, 3), "speedup": round(t_py / t_cpp, 2)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", nargs=4, type=int, default=[8, 8, 8, 16])
    ap.add_argument("--prec", default="c64")
    args = ap.parse_args()
    Lx, Ly, Lz, Lt = args.lattice
    DT = define._LAT_C128_ if args.prec == "c128" else define._LAT_C64_
    base = build_base(Lx, Ly, Lz, Lt, 0.05, 1e-6, 2, [12, 48], [2, 2, 2, 2], DT)
    dt, device = base["dt"], base["device"]
    E = 48
    lat_fine = [Lx, Ly, Lz, Lt]
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    lat_coarse_odd = [Lx // 2, Ly // 2, Lz // 2, Lt // 4]

    out = {
        "lattice": [Lx, Ly, Lz, Lt], "precision": args.prec,
        "gauge": verify_lattice(base["qcu_U"]),
        "dslash_cpp_vs_py": verify_dslash_op(base),
    }

    # 解误差（C++ BiStabCG 参考）
    qcu_ref, qcu_src = base["qcu_ref"], base["qcu_src"]
    ref_cl = base["ref_cl"]
    KAPPA = base["KAPPA"]
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, base["qcu_U"], KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)
    out["ref_solution"] = {"ref_res": float(ref_res)}

    # null_vecs 检查（复用 dev73_5）
    from mg_nullvec_cache import build_or_load_coarse_ops
    lonv, hnn, hdg, sit = build_or_load_coarse_ops(
        42, lat_fine, 1, E, 12, lat_fine_odd, lat_coarse_odd,
        base["S"], dt, device, 2, use_cache=True, save=True, verbose=False)
    out["nullvecs"] = verify_nullvecs(base["op"], base["S"], lonv, hnn, hdg,
                                      sit, E, 12, lat_fine_odd,
                                      lat_coarse_odd, dt, device)

    path = os.path.join(LOG_DIR,
                        f"dev74_verify_{'x'.join(map(str, lat_fine))}_{args.prec}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
