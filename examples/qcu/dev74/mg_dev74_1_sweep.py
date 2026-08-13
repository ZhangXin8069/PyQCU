#!/usr/bin/env python3
"""dev74_1 —— MG 参数扫描（优化 speedup，服务服务器 >1.5 目标）。

对每个 (lattice, r, ct, cmi, levels) 配置以独立进程调用 mg_dev74_clean.py
（干净测量协议：ref/mg 交叉计时 min of pairs），汇总到
logs/dev74/dev74_1_sweep.json。

优化方向（物理机制）：MG 慢于 BiStabCG 的主因是粗层求解开销占比；
  * r（V-cycle 频率）大 → 少进入粗层
  * cmi（粗层最大迭代）小 → 粗层求解迭代少
  * ct（粗层收敛条件）大 → 粗层迭代少
  * 3L vs 2L：dev73_5 在 V100 上 3L 1.32x > 2L 1.16x

用法：
    source ./env.sh && python examples/qcu/mg_dev74_1_sweep.py [--lattice 8 8 8 16] [--pairs 3]
输出：logs/dev74/dev74_1_sweep.json（含每配置 speedup_min/med 与达标标记）
"""
import os, sys, json, subprocess

LOG_DIR = os.path.expanduser("~/PyQCU/logs/dev74")
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "mg_dev74_clean.py")


def cfgs_for(lattice, pairs):
    """(r, ct, cmi, levels) 组合 —— 侧重加速机制扫描。"""
    Lx, Ly, Lz, Lt = lattice
    base = dict(ct=1e5, cmi=15)
    out = []
    for r in (5, 10, 20):
        out.append(dict(restart=r, ct=1e5, cmi=15, levels=2))
    for cmi in (50, 200):
        out.append(dict(restart=10, ct=1e5, cmi=cmi, levels=2))
    for ct in (1e2, 1e3):
        out.append(dict(restart=10, ct=ct, cmi=15, levels=2))
    out.append(dict(restart=10, ct=1e5, cmi=15, levels=3))
    out.append(dict(restart=20, ct=1e5, cmi=15, levels=2))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", nargs=4, type=int, default=[8, 8, 8, 16])
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()
    L = args.lattice
    tag = "x".join(map(str, L))
    cfgs = cfgs_for(L, args.pairs)

    results = []
    for i, c in enumerate(cfgs):
        cmd = [sys.executable, CLEAN, "--lattice"] + list(map(str, L)) + \
              ["--prec", "c64", "--levels", str(c["levels"]),
               "--restart", str(c["restart"]), "--ct", str(c["ct"]),
               "--cmi", str(c["cmi"]), "--pairs", str(args.pairs)]
        print(f"[{i+1}/{len(cfgs)}] {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED rc={r.returncode}: {r.stderr[-500:]}")
            results.append({"lattice": L, "restart": c["restart"],
                            "ct": c["ct"], "cmi": c["cmi"],
                            "levels": c["levels"], "failed": True})
            continue
        # stdout 被 C++ 日志污染：从结果 json 读取
        out_json = None
        for line in r.stdout.splitlines():
            if line.strip().startswith("{"):
                out_json = line.strip()
                break
        if out_json is None:
            # 直接读文件
            label = (f"L{tag}_c64_L{c['levels']}_r{c['restart']}"
                     f"_ct{c['ct']:.0e}_cmi{c['cmi']}_py")
            p = os.path.join(LOG_DIR, f"dev74_clean_{label}.json")
            if os.path.exists(p):
                with open(p) as f:
                    out_json = f.read()
        d = json.loads(out_json)
        results.append(d)
        sp = d.get("speedup_min")
        print(f"  -> speedup_min={sp:.3f}  iters={d['mg_iters']}/{d['ref_iters']}"
              f"  vs_ref={d['vs_ref']:.1e}", flush=True)

    with open(os.path.join(LOG_DIR, "dev74_1_sweep.json"), "w") as f:
        json.dump({"lattice": L, "results": results}, f, indent=2)
    print(f"\nwrote {LOG_DIR}/dev74_1_sweep.json ({len(results)} configs)")
    ok = [d for d in results if not d.get("failed") and
          d.get("speedup_min", 0) >= 1.5]
    print(f"speedup_min >= 1.5: {len(ok)}/{len(results)} configs")


if __name__ == "__main__":
    main()
