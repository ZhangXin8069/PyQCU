#!/usr/bin/env python3
"""dev73_5 图表生成：读取 JSON 结果，生成收敛曲线、热点柱状图、正确性图。

输出到 logs/dev73_5/figs/*.png
"""
import json, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'AR PL UMing CN',
                                   'AR PL KaitiM GB', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

LOG_DIR = "/root/PyQCU/logs/dev73_5"
FIG_DIR = os.path.join(LOG_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

C = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
     '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def load(lat, dt):
    tag = f"{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_{dt}"
    with open(os.path.join(LOG_DIR, f"dev73_5_{tag}_results.json")) as f:
        data = json.load(f)
    with open(os.path.join(LOG_DIR, f"dev73_5_{tag}_conv.json")) as f:
        convs = json.load(f)
    bcg = None
    bp = os.path.join(LOG_DIR, f"bistabcg_{tag}_conv.json")
    if os.path.exists(bp):
        with open(bp) as f:
            bcg = json.load(f)
    return data, convs, bcg


def plot_convergence(results, convs, bcg, tag, fname, max_iter=None, log_scale=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    if bcg is not None:
        r = bcg['residual_norm']
        ax.plot(range(len(r)), r, color=C[0], lw=2.0, label=f"BiStabCG 参考 (iters={bcg['iters_to_atol']})")
    for i, res in enumerate(results):
        c = convs.get(res['label'], [])
        if not c:
            continue
        ax.plot(range(len(c)), c, color=C[(i+1) % len(C)], lw=1.5,
                label=f"{res['label']} (iters={res['iters']})")
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('残差范数 ||r||')
    if log_scale:
        ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(f"收敛曲线 — {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_hotspots(results, tag, fname):
    """PROF_SECTIONS 热点柱状图（按配置）。"""
    labels = [r['label'] for r in results]
    keys = ['fine_iter', 'vcycle', 'coarse_solve', 'coarse_vec', 'coarse_dslash']
    cols = {'fine_iter': C[0], 'vcycle': C[1], 'coarse_solve': C[2],
            'coarse_vec': C[4], 'coarse_dslash': C[3]}
    names = {'fine_iter': '细层迭代', 'vcycle': 'V-cycle', 'coarse_solve': '粗层求解',
             'coarse_vec': '粗层向量', 'coarse_dslash': '粗层dslash'}
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for k in keys:
        vals = [r['prof'].get(k, 0) for r in results]
        ax.bar(x, vals, bottom=bottom, color=cols[k], label=names[k], width=0.6)
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('耗时 (ms)')
    ax.set_title(f"计算热点分布 — {tag}")
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_speedup(results, tag, fname):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r['label'] for r in results]
    speed = [r['speedup'] for r in results]
    bars = ax.bar(np.arange(len(labels)), speed, color=C[:len(labels)], width=0.6)
    for b, s in zip(bars, speed):
        ax.text(b.get_x()+b.get_width()/2, s+0.02, f"{s:.2f}x", ha='center', fontsize=8)
    ax.axhline(1.0, color='red', ls='--', lw=1)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('加速比 (vs BiStabCG)')
    ax.set_title(f"MultiGrid 相对 BiStabCG 加速比 — {tag}")
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_solution_error(results, tag, fname):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r['label'] for r in results]
    vs_ref = [max(r['vs_ref'], 1e-16) for r in results]
    full_res = [max(r['mg_full_res'], 1e-16) for r in results]
    x = np.arange(len(labels))
    ax.bar(x-0.2, vs_ref, width=0.4, color=C[0], label='MG 解 vs BiStabCG 参考')
    ax.bar(x+0.2, full_res, width=0.4, color=C[1], label='全算子残差 |Dx-b|/|b|')
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('相对误差')
    ax.set_title(f"求解精度 — {tag}")
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_gauge(gauges, tag, fname):
    keys = ['max_unitary_err', 'max_det_err', 'max_minor_err']
    names = ['|U^H U - I| max', '|det U - 1| max', 'minor identity 残差 max']
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (k, n) in enumerate(zip(keys, names)):
        vals = [g[k] for g in gauges]
        ax.bar(np.arange(len(vals))+i*0.28, vals, width=0.28, color=C[i], label=n)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(gauges)))
    ax.set_xticklabels([g['tag'] for g in gauges], fontsize=9)
    ax.set_ylabel('最大偏差')
    ax.set_title(f"gauge SU(3) 性质检查 — {tag}")
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_nullvec(nulls, tag, fname):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [('null_res_ratio', '||S·P||/||P|| (零模性质)'),
               ('gram_vs_I_max', '块内 Gram-I max (正交性)'),
               ('stencil_rel_err', 'stencil vs 算子 相对误差')]
    for ax, (k, n) in zip(axes, metrics):
        labels = [f"L{n['level']}" for n in nulls]
        vals = [n[k] for n in nulls]
        ax.bar(range(len(vals)), vals, color=C[:len(vals)])
        ax.set_yscale('log')
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels)
        ax.set_title(n, fontsize=10)
        ax.grid(True, axis='y', which='both', alpha=0.3)
    fig.suptitle(f"null_vecs 正确性 — {tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


def main():
    # 收集所有已存在的配置
    metas = []
    for fn in sorted(os.listdir(LOG_DIR)):
        m = re.match(r'dev73_5_(\d+x\d+x\d+x\d+)_(c64|c128)_results\.json', fn)
        if m:
            metas.append((tuple(int(v) for v in m.group(1).split('x')), m.group(2)))
    print("configs found:", metas)
    all_results = []
    all_gauges = []
    all_null = {}
    for lat, dt in metas:
        tag = f"{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_{dt}"
        data, convs, bcg = load(lat, dt)
        results = data['results']
        all_results.extend(results)
        g = dict(data['gauge']); g['tag'] = tag; all_gauges.append(g)
        if data.get('nullvecs'):
            all_null[tag] = data['nullvecs']

        # 收敛曲线：base 配置（含 BiStabCG 对照）
        base_results = [r for r in results if r['label'].startswith('base')]
        base_labels = {r['label'] for r in base_results}
        base_conv = {k: v for k, v in convs.items() if k in base_labels}
        plot_convergence(base_results, base_conv, bcg, tag,
                         f"conv_base_{tag}.png")

        # 参数扫描收敛曲线
        for sweep, fname in [('restart', 'conv_restart'), ('ct_', 'conv_coarsetol'),
                             ('cmi_', 'conv_maxiter'), ('levels_', 'conv_levels')]:
            sel = [r for r in results if r['label'].startswith(sweep)]
            if sel:
                sel_conv = {k: v for k, v in convs.items() if k in {r['label'] for r in sel}}
                plot_convergence(sel, sel_conv, bcg, tag, f"{fname}_{tag}.png")

        # 热点 / 加速比 / 精度
        plot_hotspots(results, tag, f"hotspot_{tag}.png")
        plot_speedup(results, tag, f"speedup_{tag}.png")
        plot_solution_error(results, tag, f"solution_err_{tag}.png")

    if all_gauges:
        plot_gauge(all_gauges, "all", "gauge_su3.png")
    for tag, nulls in all_null.items():
        if nulls:
            plot_nullvec(nulls, tag, f"nullvec_{tag}.png")
    print("ALL PLOTS DONE")


if __name__ == "__main__":
    main()
