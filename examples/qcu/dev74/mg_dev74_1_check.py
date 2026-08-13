#!/usr/bin/env python3
"""dev74_1 —— 服务器加速比自动断言（MG vs BiStabCG >= 阈值）。

读取 dev74_1_sweep.json / dev74_clean_*.json / dev74_bench.json，
逐配置输出 speedup_min/med，并按 --gate 阈值断言（默认 1.5）：
  * 全部达标 → exit 0
  * 任一不达标 → exit 1（列出不达标配置与建议参数）
  * 缺数据 → exit 2

用法：
    source ./env.sh && python examples/qcu/mg_dev74_1_check.py [--gate 1.5] [--file ...]
"""
import json, os, glob, sys

LOG_DIR = os.path.expanduser("~/PyQCU/logs/dev74")


def load_all(files):
    """读取显式指定的 json 文件；缺省读 dev74_1_sweep.json。"""
    entries = []
    seen = set()
    def add(e):
        if e.get("speedup_min") is None and e.get("speedup") is None:
            return
        key = (tuple(e.get("lattice", [])), e.get("levels"),
               e.get("restart", e.get("NUM_RESTART")),
               e.get("ct", e.get("coarse_tol_factor")),
               e.get("cmi", e.get("coarse_max_iter")))
        if key in seen:
            return
        seen.add(key)
        entries.append(e)
    if files:
        for p in files:
            if not os.path.exists(p):
                print(f"[warn] 文件不存在: {p}")
                continue
            with open(p) as f:
                d = json.load(f)
            if "results" in d and isinstance(d["results"], list):
                for r in d["results"]:
                    add(r)
            elif isinstance(d, dict) and "lattice" in d:
                add(d)
    else:
        p = os.path.join(LOG_DIR, "dev74_1_sweep.json")
        if os.path.exists(p):
            with open(p) as f:
                for r in json.load(f)["results"]:
                    add(r)
    return entries


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=1.5)
    ap.add_argument("--label", default="dev74_1 server check")
    ap.add_argument("--file", action="append", default=None,
                    help="显式指定 json 文件（可多次）；缺省读 dev74_1_sweep.json")
    args = ap.parse_args()
    entries = load_all(args.file)
    if not entries:
        print(f"[{args.label}] NO DATA — 请先在服务器上运行 "
              f"mg_dev74_1_sweep.py / mg_dev74_clean.py")
        sys.exit(2)

    print(f"[{args.label}] gate = speedup_min >= {args.gate}")
    print(f"{'config':52s} {'speedup_min':>10s} {'speedup_med':>10s} {'iters':>10s}")
    n_ok = 0
    fails = []
    for e in entries:
        L = e.get("lattice", [])
        lv = e.get("levels", 2)
        r = e.get("restart", e.get("NUM_RESTART", 10))
        sp = e.get("speedup_min", e.get("speedup"))
        spm = e.get("speedup_med")
        it = f"{e.get('mg_iters', '?')}/{e.get('ref_iters', '?')}"
        name = e.get("label", f"{'x'.join(map(str, L))}_L{lv}_r{r}")
        if sp is None:
            print(f"{name:52s} {'n/a':>10s}")
            fails.append(name)
            continue
        ok = sp >= args.gate
        n_ok += ok
        mark = "OK " if ok else "FAIL"
        print(f"{name:52s} {sp:10.3f} {spm if spm else 0:10.3f} {it:>10s}  {mark}")
        if not ok:
            fails.append(name)

    print(f"\n达标 {n_ok}/{len(entries)}")
    if fails:
        print("不达标配置（建议：levels=3 / restart=20 / --build cpp）：")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
