#!/usr/bin/env python3
"""dev73_5 —— 从 JSON 生成 LaTeX 表格片段，保证报告数字与数据一致。

输入：logs/dev73_5_bench.json, logs/dev73_5_verify_*.json
输出：logs/dev73_5_tbl_{main,prec,lattice,sweep,verify}.tex（供 dev73_5.tex \input）
"""
import json, os, glob

LOG_DIR = "/root/PyQCU/logs"


def fmt_time(ms):
    return f"{ms:.0f}"


def write(name, lines):
    with open(os.path.join(LOG_DIR, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {LOG_DIR}/{name}")


def main():
    with open(os.path.join(LOG_DIR, "dev73_5_results.json")) as f:
        data = json.load(f)
    results = data["results"]
    verify_all = data.get("verify", {})

    # ---------------- 主对照表 ----------------
    out = []
    out.append(r"\begin{longtable}{llrrrrrrl}")
    out.append(r"\toprule")
    out.append("格子 & 配置 & 精度 & BiStabCG(ms) & MG(ms) & 加速比 & "
               r"MG细迭代 vs BiStabCG & vs\_ref & 备注 \\")
    out.append(r"\midrule")
    out.append(r"\endhead")
    for r in sorted(results, key=lambda r: (r["lattice"], r["precision"],
                                            r["restart"], r["ct"])):
        if r.get("speedup_min") is None:
            continue
        lat = "×".join(str(x) for x in r["lattice"])
        cfg = f"L{r['levels']}, E={r['dof']}, r={r['restart']}"
        cfg += f", ct={r['ct']:.0e}, cmi={r['cmi']}"
        prec = "c128" if r["precision"] == "c128" else "c64"
        note = ""
        if tuple(r["lattice"]) == (8, 16, 16, 16) and r["levels"] == 2 \
                and r["restart"] == 10 and r["ct"] == 1e5 and r["cmi"] == 15:
            note = "基准"
        out.append(f"{lat} & {cfg} & {prec} & {fmt_time(r['ref_min_ms'])} & "
                   f"{fmt_time(r['mg_min_ms'])} & "
                   f"\\textbf{{{r['speedup_min']:.2f}$\\times$}} & "
                   f"{r['mg_iters']} vs {r['ref_iters']} & "
                   f"{r['vs_ref']:.1e} & {note} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")
    write("dev73_5_tbl_main.tex", out)

    # ---------------- 精度对照（默认格子） ----------------
    out = []
    out.append(r"\begin{table}[h]\centering")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\toprule")
    out.append("精度 & BiStabCG(ms) & MG(ms) & 加速比 & 迭代(MG/ref) \\\\")
    out.append(r"\midrule")
    for r in results:
        if tuple(r["lattice"]) != (8, 16, 16, 16) or r["levels"] != 2:
            continue
        if r["restart"] != 10 or r["ct"] != 1e5 or r["cmi"] != 15:
            continue
        if r.get("speedup_min") is None:
            continue
        prec = "c128（双精度）" if r["precision"] == "c128" else "c64（单精度）"
        out.append(f"{prec} & {fmt_time(r['ref_min_ms'])} & "
                   f"{fmt_time(r['mg_min_ms'])} & "
                   f"{r['speedup_min']:.2f}$\\times$ & "
                   f"{r['mg_iters']}/{r['ref_iters']} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\caption{精度对照：默认格子 $\{8,16,16,16\}$，2L, r=10, "
               r"ct=$10^5$, cmi=15（干净 min of 5 对）}")
    out.append(r"\end{table}")
    write("dev73_5_tbl_prec.tex", out)

    # ---------------- 格子对照 ----------------
    out = []
    out.append(r"\begin{table}[h]\centering")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\toprule")
    out.append("格子 & BiStabCG(ms) & MG(ms) & 加速比 & 迭代(MG/ref) \\\\")
    out.append(r"\midrule")
    for r in results:
        if r["levels"] != 2 or r["restart"] != 10 \
                or r["ct"] != 1e5 or r["cmi"] != 15:
            continue
        if r["precision"] != "c64" or r.get("speedup_min") is None:
            continue
        out.append(f"$\\{{{r['lattice'][0]},{r['lattice'][1]},{r['lattice'][2]},"
                   f"{r['lattice'][3]}\\}}$ & {fmt_time(r['ref_min_ms'])} & "
                   f"{fmt_time(r['mg_min_ms'])} & "
                   f"{r['speedup_min']:.2f}$\\times$ & "
                   f"{r['mg_iters']}/{r['ref_iters']} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\caption{格子大小对照：c64, 2L, r=10, ct=$10^5$, cmi=15"
               r"（干净 min of 5 对）}")
    out.append(r"\end{table}")
    write("dev73_5_tbl_lattice.tex", out)

    # ---------------- 参数扫描（默认格子 c64 2L） ----------------
    out = []
    out.append(r"\begin{longtable}{llcccc}")
    out.append(r"\toprule")
    out.append("变量 & 值 & BiStabCG(ms) & MG(ms) & 加速比 & 迭代(MG/ref) \\\\")
    out.append(r"\midrule")
    out.append(r"\endhead")
    for r in results:
        if tuple(r["lattice"]) != (8, 16, 16, 16) or r["levels"] != 2 \
                or r["precision"] != "c64" or r.get("speedup_min") is None:
            continue
        if r["restart"] == 10 and r["cmi"] == 15 and r["ct"] == 1e5:
            var, val = "基准", "r=10, ct=$10^5$, cmi=15"
        elif r["ct"] == 1e5 and r["cmi"] == 15:
            var, val = "V-cycle 频率 $r$", f"r={r['restart']}"
        elif r["restart"] == 10 and r["cmi"] == 15:
            var, val = "最粗层容差 $ct$", f"ct={r['ct']:.0e}"
        elif r["restart"] == 10 and r["ct"] == 1e5:
            var, val = "最粗层迭代 $cmi$", f"cmi={r['cmi']}"
        else:
            continue
        out.append(f"{var} & {val} & {fmt_time(r['ref_min_ms'])} & "
                   f"{fmt_time(r['mg_min_ms'])} & "
                   f"{r['speedup_min']:.2f}$\\times$ & "
                   f"{r['mg_iters']}/{r['ref_iters']} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")
    write("dev73_5_tbl_sweep.tex", out)

    # ---------------- 正确性汇总（verify） ----------------
    def _f(x, fmt=".2e"):
        return "—" if x is None else f"{x:{fmt}}"
    out = []
    for vk in sorted(verify_all):
        v = verify_all[vk]
        lat = "×".join(str(x) for x in v["lattice"])
        out.append(r"\begin{table}[h]\centering")
        out.append(r"\begin{tabular}{llc}")
        out.append(r"\toprule")
        out.append("项 & 指标 & 值 \\\\")
        out.append(r"\midrule")
        g = v.get("gauge", {})
        out.append(r"\multirow{3}{*}{Gauge} & $\max|U^\dagger U-I|$ & "
                   f"{g.get('max_unit_err', '—'):.2e} \\\\")
        out.append(f"& $\\max|\\det U - 1|$ & {g.get('max_det_dev','—'):.2e} \\\\")
        out.append(f"& check\\_su3 & {g.get('check_su3','—')} \\\\")
        out.append(f"参考残差 & $\\Vert b-D x_{{ref}}\\Vert/\\Vert b\\Vert$ & "
                   f"{v.get('ref_res','—'):.2e} \\\\")
        for lv in v.get("nullvecs", {}).get("levels", []):
            e = lv.get("E", "?")
            out.append(r"\multirow{4}{*}{NV E=" + f"{e}" + "}")
            out.append(f"& $\\Vert S v\\Vert/\\Vert v\\Vert$ (前4个) & "
                       f"{','.join(f'{x:.1e}' for x in lv.get('null_ratios',[])[:4])} \\\\")
            out.append(f"& 块内正交 $\\max|\\langle v_i,v_j\\rangle-\\delta|$ & "
                       f"{_f(lv.get('ortho_offdiag_max'))} \\\\")
            out.append(f"& restrict/prolong 相对差 & "
                       f"{_f(lv.get('restrict_rel_diff'),'.1e')} / "
                       f"{_f(lv.get('prolong_rel_diff'),'.1e')} \\\\")
            out.append(f"& 粗 dslash 相对差 & "
                       f"{_f(lv.get('coarse_dslash_rel_diff'),'.1e')} \\\\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\caption{正确性检查：lattice=" + lat + ", prec="
                   + v.get("precision", "?") + "}")
        out.append(r"\end{table}")
    write("dev73_5_tbl_verify.tex", out)


if __name__ == "__main__":
    main()
