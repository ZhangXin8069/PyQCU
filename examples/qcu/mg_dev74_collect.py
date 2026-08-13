#!/usr/bin/env python3
"""dev74 —— 汇总所有测量到 logs/dev74_results.json。

合并：
  * dev74_bench.json            —— 本地/集群 bench（计时 + 资源统计 + 收敛/热点）
  * dev74_clean_*.json          —— 干净（独立进程）计时 + 资源统计
  * dev74_verify_*.json         —— 正确性数据（gauge / 解 / CudaSchurOp 对照 / null_vecs）
每配置一条，统一 schema，供 mktable / plots / 报告使用。
"""
import json, os, glob

LOG_DIR = os.path.expanduser("~/PyQCU/logs")


def load(p):
    with open(p) as f:
        return json.load(f)


def cfg_key(r):
    return (tuple(r["lattice"]), r.get("precision", "c64"), r.get("levels", 2),
            r.get("restart", 10),
            float(r.get("ct", r.get("coarse_tol_factor", 1e5))),
            int(r.get("cmi", r.get("coarse_max_iter", 15))))


def main():
    bench = load(os.path.join(LOG_DIR, "dev74_bench.json"))
    clean_files = sorted(glob.glob(os.path.join(LOG_DIR, "dev74_clean_L*.json")))
    clean = [load(f) for f in clean_files]
    warm = bench["results"]

    warm_by_key = {}
    for r in warm:
        if "lattice" not in r:
            continue
        k = cfg_key(r)
        warm_by_key.setdefault(k, []).append(r)

    out = []
    for r in clean:
        k = cfg_key(r)
        w = warm_by_key.get(k, [{}])[0]
        entry = {
            "label": r["label"],
            "lattice": r["lattice"],
            "precision": r["precision"],
            "levels": r["levels"], "dof": r["dof"],
            "restart": r["restart"], "ct": r["ct"], "cmi": r["cmi"],
            "ref_min_ms": r["ref_min_ms"], "mg_min_ms": r["mg_min_ms"],
            "speedup_min": r["speedup_min"],
            "ref_med_ms": r["ref_med_ms"], "mg_med_ms": r["mg_med_ms"],
            "speedup_med": r["speedup_med"],
            "vs_ref": r["vs_ref"], "mg_res": r["mg_res"],
            "ref_res": r["ref_res"],
            "mg_iters": r.get("mg_iters", w.get("mg_iters", 0)),
            "ref_iters": r.get("ref_iters", w.get("ref_iters", 0)),
            "conv_mg": r.get("conv_mg") or w.get("conv_mg", []),
            "ref_hist": r.get("ref_hist") or w.get("ref_hist", []),
            "prof": r.get("prof") or w.get("prof", {}),
            # dev74 资源统计
            "build_mode": r.get("build_mode", "py"),
            "build_s": r.get("build_s", w.get("build_s", 0.0)),
            "peak_vram_cold_mb": r.get("peak_vram_cold_mb", w.get("peak_vram_build_mb")),
            "peak_vram_warm_mb": r.get("peak_vram_warm_mb", w.get("peak_vram_mg_mb")),
            "rss_kb": r.get("rss_kb", w.get("rss_kb")),
            "disk_mb": r.get("disk_mb", w.get("disk_mb")),
        }
        out.append(entry)

    # 附上 bench 中 clean 缺失的配置（本地 3 格）
    clean_keys = {cfg_key(r) for r in out}
    for r in warm:
        if "lattice" not in r or "failed" in r or "skipped" in r:
            continue
        k = cfg_key(r)
        if k in clean_keys:
            continue
        out.append({
            "label": r["label"], "lattice": r["lattice"],
            "precision": r["precision"], "levels": r["levels"],
            "dof": r["dof"], "restart": r["restart"],
            "ct": r["coarse_tol_factor"], "cmi": r["coarse_max_iter"],
            "ref_min_ms": r["ref_ms"], "mg_min_ms": r["mg_ms"],
            "speedup_min": r["speedup"],
            "ref_med_ms": r["ref_ms"], "mg_med_ms": r["mg_ms"],
            "speedup_med": r["speedup"],
            "vs_ref": r["vs_ref"], "mg_res": r["mg_res"], "ref_res": r["ref_res"],
            "mg_iters": r["mg_iters"], "ref_iters": r["ref_iters"],
            "conv_mg": r.get("conv_mg", []), "ref_hist": r.get("ref_hist", []),
            "prof": r.get("prof", {}),
            "build_mode": r.get("build_mode", "py"),
            "build_s": r.get("build_s", 0.0),
            "peak_vram_cold_mb": r.get("peak_vram_build_mb"),
            "peak_vram_warm_mb": r.get("peak_vram_mg_mb"),
            "rss_kb": r.get("rss_kb"), "disk_mb": r.get("disk_mb"),
            "_note": "bench（非独立进程）计时与资源",
        })

    # 正确性数据
    verify = {}
    for vf in sorted(glob.glob(os.path.join(LOG_DIR, "dev74_verify_*.json"))):
        v = load(vf)
        key = (tuple(v["lattice"]), v["precision"])
        verify[str(key)] = v

    with open(os.path.join(LOG_DIR, "dev74_results.json"), "w") as f:
        json.dump({"results": out, "verify": verify,
                   "bench_mode": bench.get("mode")}, f, indent=2)
    print(f"wrote {LOG_DIR}/dev74_results.json: {len(out)} configs, "
          f"{len(verify)} verify sets")
    for e in out:
        sp = e["speedup_min"]
        s1 = "—" if sp is None else f"{sp:.2f}x"
        vram = e.get("peak_vram_warm_mb")
        vr = "—" if vram is None else f"{vram:.0f}MB"
        print(f"  {e['label']:44s} min={s1}  iters={e['mg_iters']}/{e['ref_iters']}"
              f"  vs_ref={e['vs_ref']:.1e}  vram={vr}")


if __name__ == "__main__":
    main()
