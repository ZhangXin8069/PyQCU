#!/usr/bin/env python3
"""dev73_5 —— 汇总所有测量到 logs/dev73_5_results.json。

合并：
  * dev73_5_clean_*.json       —— 干净（独立进程）计时：ref/mg min+median 加速比
  * dev73_5_bench.json (warm)  —— 连续负载扫描的收敛历史 CONVERGENCE_HISTORY 与
                                  热点 PROF_SECTIONS（作为收敛/热点图表数据）
  * dev73_5_verify_*.json      —— 正确性数据（gauge / 解 / null_vecs）
每配置一条，统一 schema，供 mktable / plots / 报告使用。
"""
import json, os, glob

LOG_DIR = os.path.expanduser("~/PyQCU/logs")


def load(p):
    with open(p) as f:
        return json.load(f)


def cfg_key(r):
    return (tuple(r["lattice"]), r["precision"], r["levels"], r["restart"],
            float(r["ct"] if "ct" in r else r.get("coarse_tol_factor")),
            int(r["cmi"] if "cmi" in r else r.get("coarse_max_iter")))


def main():
    warm = load(os.path.join(LOG_DIR, "dev73_5_bench.json"))["results"]
    clean_files = sorted(glob.glob(os.path.join(LOG_DIR, "dev73_5_clean_L*.json")))
    clean = [load(f) for f in clean_files]

    # 以 clean 为主；warm 提供收敛/热点
    warm_by_key = {cfg_key(r): r for r in warm}
    out = []
    for r in clean:
        k = cfg_key(r)
        w = warm_by_key.get(k, {})
        entry = {
            "label": r["label"],
            "lattice": r["lattice"],
            "precision": r["precision"],
            "levels": r["levels"], "dof": r["dof"],
            "restart": r["restart"], "ct": r["ct"], "cmi": r["cmi"],
            # 干净计时
            "ref_min_ms": r["ref_min_ms"], "mg_min_ms": r["mg_min_ms"],
            "speedup_min": r["speedup_min"],
            "ref_med_ms": r["ref_med_ms"], "mg_med_ms": r["mg_med_ms"],
            "speedup_med": r["speedup_med"],
            "vs_ref": r["vs_ref"], "mg_res": r["mg_res"],
            "ref_res": r["ref_res"],
            # 收敛 / 热点（优先 clean 自带，否则 warm）
            "mg_iters": r.get("mg_iters", w.get("mg_iters", 0)),
            "ref_iters": r.get("ref_iters", w.get("ref_iters", 0)),
            "conv_mg": r.get("conv_mg") or w.get("conv_mg", []),
            "ref_hist": r.get("ref_hist") or w.get("ref_hist", []),
            "prof": r.get("prof") or w.get("prof", {}),
        }
        out.append(entry)

    # 附上 warm 中但 clean 缺失的默认格子配置（确保 12 个全）
    clean_keys = {cfg_key(r) for r in out}
    for r in warm:
        k = cfg_key(r)
        if k in clean_keys:
            continue
        out.append({
            "label": r["label"], "lattice": r["lattice"],
            "precision": r["precision"], "levels": r["levels"],
            "dof": r["dof"], "restart": r["restart"],
            "ct": r["coarse_tol_factor"], "cmi": r["coarse_max_iter"],
            "ref_min_ms": None, "mg_min_ms": None, "speedup_min": None,
            "ref_med_ms": None, "mg_med_ms": None, "speedup_med": None,
            "vs_ref": r["vs_ref"], "mg_res": r["mg_res"], "ref_res": None,
            "mg_iters": r["mg_iters"], "ref_iters": r["ref_iters"],
            "conv_mg": r.get("conv_mg", []), "ref_hist": r.get("ref_hist", []),
            "prof": r.get("prof", {}),
            "_note": "无 clean 计时（仅 warm）",
        })

    # 正确性数据
    verify = {}
    for vf in sorted(glob.glob(os.path.join(LOG_DIR, "dev73_5_verify_*.json"))):
        v = load(vf)
        key = (tuple(v["lattice"]), v["precision"], v["levels"])
        verify[str(key)] = v

    with open(os.path.join(LOG_DIR, "dev73_5_results.json"), "w") as f:
        json.dump({"results": out, "verify": verify}, f, indent=2)
    print(f"wrote {LOG_DIR}/dev73_5_results.json: {len(out)} configs, "
          f"{len(verify)} verify sets")
    for e in out:
        sp = e["speedup_min"]
        sm = e["speedup_med"]
        s1 = "—" if sp is None else f"{sp:.2f}x"
        s2 = "—" if sm is None else f"{sm:.2f}x"
        print(f"  {e['label']:52s} clean_min={s1}  med={s2}"
              f"  iters={e['mg_iters']}/{e['ref_iters']}  vs_ref={e['vs_ref']:.1e}")


if __name__ == "__main__":
    main()
