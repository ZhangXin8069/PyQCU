#!/usr/bin/env python3
"""dev78_1 补充图表：收敛图（conv_*）+ 显存图（vram）+ 日志分析图。

参考 logs/dev74 输出形式（dev74_conv_*.png / dev74_vram.png / dev74_budget.png）。
输入：版本目录内 bench.log（含 LOOP 残差与 PROF_SECTIONS）+ budget h5。
"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = sys.argv[1] if len(sys.argv) > 1 else 'v202608160214'
os.makedirs(WORK, exist_ok=True)

def _style(ax):
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    ax.set_yscale('log')

# ---- 1) 收敛图：从 bench.log 提取各配置残差曲线 ----
def parse_conv(path):
    """bench.log -> {label: [residuals]}。LOOP:N##Residual(norm2):(re,im) 按配置分段。"""
    if not os.path.exists(path):
        return {}
    conv = {}
    cur = None
    # 配置切换由每次 BiStabCG 求解的开始标记切分：MG_INIT_COMPLETE 或 LOOP 编号
    # 跳变。用 PROF_SECTIONS 前的求解序号分组：每个求解 = 一个 cfg。
    solve_idx = 0
    last_loop = -1
    with open(path) as f:
        for line in f:
            m = re.search(r"LOOP:(\d+)##Residual\(norm2\):\(([^,]+),", line)
            if m:
                loop = int(m.group(1))
                if loop < last_loop:
                    solve_idx += 1
                last_loop = loop
                cur = f"cfg{solve_idx}"
                conv.setdefault(cur, []).append(float(m.group(2)))
    return {k: v for k, v in conv.items() if len(v) >= 3}

conv = parse_conv(os.path.join(WORK, 'bench.log'))
if conv:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (k, res) in enumerate(sorted(conv.items(), key=lambda x: -len(x[1]))[:6]):
        ax.plot(range(1, len(res) + 1), res, 'o-', ms=3, label=f'cfg{i+1} ({len(res)} iters)')
    ax.set_xlabel('BiStabCG iteration')
    ax.set_ylabel('residual norm2')
    ax.set_title('dev78_1: convergence (BiStabCG, log scale)')
    ax.legend(fontsize=8)
    _style(ax)
    fig.savefig(os.path.join(WORK, 'dev78_1_conv_bench.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {WORK}/dev78_1_conv_bench.png ({len(conv)} configs)")

# ---- 2) vram 图：budget 16g/32g 数据 ----
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import load_dict_h5, _entries_list
def load_budget(fn):
    if not os.path.exists(fn):
        return None
    d = load_dict_h5(fn)
    return d, _entries_list({'entries': d['rows']})

for tag, fn in [('16g', os.path.join(WORK, 'test78_1_budget_16g.h5')),
                ('32g', os.path.join(WORK, 'test78_1_budget_32g.h5'))]:
    b = load_budget(fn)
    if not b:
        print(f"[skip] {fn} not found"); continue
    _, rows = b
    lats = ['x'.join(map(str, r.get('lattice', r.get('d_lattice', [])))) for r in rows]
    cold = [r['cold_gb'] for r in rows]
    warm = [r['warm_gb'] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(lats))
    ax.bar(x - 0.2, cold, 0.4, label='cold (full flow)', color='coral')
    ax.bar(x + 0.2, warm, 0.4, label='warm (solve only)', color='steelblue')
    for xi, c, w in zip(x, cold, warm):
        ax.text(xi - 0.2, c + 0.1, f'{c:.1f}', ha='center', fontsize=8)
        ax.text(xi + 0.2, w + 0.1, f'{w:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(lats, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('GPU memory (GB)')
    ax.set_title(f'dev78_1: VRAM budget ({tag} card)')
    ax.legend()
    _style(ax)
    fig.savefig(os.path.join(WORK, f'dev78_1_vram_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {WORK}/dev78_1_vram_{tag}.png")

# ---- 3) 日志分析图：PROF_SECTIONS 各段占比 ----
prof = []
with open(os.path.join(WORK, 'bench.log')) as f:
    for line in f:
        m = re.search(r"PROF_SECTIONS: fine_iter=([\d.]+)ms vcycle=([\d.]+)ms n_vcycles=(\d+) coarse_solve=([\d.]+)ms", line)
        if m:
            prof.append((float(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))))
if prof:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(prof))
    ax.bar(x, [p[0] for p in prof], 0.6, label='fine_iter', color='steelblue')
    ax.bar(x, [p[1] - p[0] for p in prof], 0.6, bottom=[p[0] for p in prof],
           label='vcycle_other', color='coral')
    ax.set_xlabel('run (ref/mg per config)')
    ax.set_ylabel('time (ms)')
    ax.set_title('dev78_1: PROF_SECTIONS breakdown')
    ax.legend(fontsize=8)
    _style(ax)
    fig.savefig(os.path.join(WORK, 'dev78_1_prof.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {WORK}/dev78_1_prof.png ({len(prof)} runs)")

print("done")
