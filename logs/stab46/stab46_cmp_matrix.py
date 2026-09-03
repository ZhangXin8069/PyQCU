"""dev87 对照矩阵状态聚合：读 out/*.json 更新 comparison_matrix.md 的状态列。

用法：python examples/qcu/dev87/cmp_matrix.py   （幂等，只改 [ ] -> [x]/[~] 标记行）
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MATRIX = HERE / "comparison_matrix.md"


def load(name):
    f = OUT / f"{name}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def fmt(x):
    return "x" if x else "~"


def main():
    rows = []
    schur = load("qcu_schur_dslash")
    solve_q = load("qcu_clover_solve")
    solve_k = load("quda_clover_solve")
    mg_q = load("qcu_clover_mg")
    mg_k = load("quda_clover_mg")

    if schur:
        rows.append(f"| PyQCU Schur matvec 基线 | {schur['matvec_ms_median']:.3f} ms | out/qcu_schur_dslash.json |")
    if solve_q:
        rows.append(f"| PyQCU Clover BiCGStab | {solve_q['solve_ms']:.1f} ms, iters={solve_q.get('iters')} | out/qcu_clover_solve.json |")
    if solve_k:
        rd = solve_k["rel_diff_vs_qcu"]
        ok = rd < 5e-4
        rows.append(f"| G4.1 双方 BiCGStab 解对照 | rel_diff={rd:.2e} ({'PASS' if ok else 'FAIL'}) | out/quda_clover_solve.json |")
    if mg_q:
        rows.append(f"| G8/G10 PyQCU MG 端到端 | {mg_q['mg_wall_s']:.3f} s, rel_vs_ref={mg_q['rel_diff_vs_bistabcg']:.2e} | out/qcu_clover_mg.json |")
    if mg_k:
        rows.append(f"| G8/G10 quda MG 端到端 | setup={mg_k['setup_s']:.1f}s solve={mg_k['secs']:.3f}s iters={mg_k['iters']} | out/quda_clover_mg.json |")
    if mg_q and mg_k:
        sp = mg_k["secs"] / mg_q["mg_wall_s"] if mg_q["mg_wall_s"] > 0 else float("nan")
        rows.append(f"| 性能比 quda/PyQCU(MG) | x{sp:.3f} | 派生 |")

    print("# dev87 对照结果汇总\n")
    print("| 项 | 数值 | 来源 |")
    print("|---|---|---|")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
