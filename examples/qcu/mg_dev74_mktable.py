#!/usr/bin/env python3
"""dev74 —— 生成 LaTeX 表（性能 + 资源占用 + 预算预测）。

表：
  * dev74_tbl_main.tex    —— 本地验证结果（加速比、迭代、正确性）
  * dev74_tbl_res.tex     —— 资源占用（cold/warm 显存、RSS、磁盘、构建耗时）
  * dev74_tbl_budget.tex  —— 集群大格子预算预测（cold/warm × 32G 极限）
"""
import json, os

LOG_DIR = "/root/PyQCU/logs"


def esc(s):
    return s.replace("_", r"\_")


def main():
    with open(os.path.join(LOG_DIR, "dev74_results.json")) as f:
        data = json.load(f)
    results = data["results"]

    # ---- main 表 ----
    rows = []
    for r in results:
        sp = r.get("speedup_min")
        vram = r.get("peak_vram_warm_mb")
        rows.append((esc(r["label"]),
                     f"{r['ref_min_ms']:.0f}" if r.get("ref_min_ms") else "—",
                     f"{r['mg_min_ms']:.0f}" if r.get("mg_min_ms") else "—",
                     "—" if sp is None else f"{sp:.2f}",
                     f"{r['mg_iters']}/{r['ref_iters']}",
                     f"{r['vs_ref']:.1e}",
                     "—" if vram is None else f"{vram:.0f}"))
    with open(os.path.join(LOG_DIR, "dev74_tbl_main.tex"), "w") as f:
        f.write("% dev74 —— 本地验证结果（RTX 4060 8GB, c64, 2L, r10 ct1e5 cmi15）\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("配置 & ref(ms) & MG(ms) & 加速比 & iters(MG/ref) & vs\\_ref & 显存(MB) \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            f.write(" & ".join(["\\texttt{%s}" % r[0]] + list(r[1:])) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # ---- 资源表 ----
    with open(os.path.join(LOG_DIR, "dev74_tbl_res.tex"), "w") as f:
        f.write("% dev74 —— 资源占用统计（实测）\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("配置 & V & cold显存(MB) & warm显存(MB) & RSS(MB) & 磁盘(MB) & 构建(s) \\\\\n")
        f.write("\\hline\n")
        for r in results:
            V = r["lattice"][0] * r["lattice"][1] * r["lattice"][2] * r["lattice"][3]
            f.write(f"\\texttt{{{esc(r['label'])}}} & {V} & "
                    f"{r.get('peak_vram_cold_mb') or '—'} & "
                    f"{r.get('peak_vram_warm_mb') or '—'} & "
                    f"{(r.get('rss_kb') or 0)/1e3:.0f} & "
                    f"{r.get('disk_mb') or '—'} & "
                    f"{r.get('build_s') or 0:.0f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # ---- 预算预测表 ----
    bp = os.path.join(LOG_DIR, "dev74_budget_cluster.json")
    if os.path.exists(bp):
        with open(bp) as f:
            budget = json.load(f)
        with open(os.path.join(LOG_DIR, "dev74_tbl_budget.tex"), "w") as f:
            f.write("% dev74 —— 集群大格子预算预测（实测校准：cold 53KB/V, warm 27KB/V）\n")
            f.write("\\begin{tabular}{lcccc}\n\\hline\n")
            f.write("格子 & V & cold(GB) & warm(GB) & cold/32G \\\\\n\\hline\n")
            for row in budget["rows"]:
                L = "x".join(map(str, row["lattice"]))
                f.write(f"\\texttt{{{L}}} & {row['V']} & "
                        f"{row['pred_vram_mb']/1024:.1f} & "
                        f"{row['pred_vram_warm_mb']/1024:.1f} & "
                        f"{row['vram_frac_32g']:.2f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
