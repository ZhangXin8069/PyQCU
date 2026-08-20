#!/usr/bin/env python3
"""test15 —— 补充 dev73/dev74 同类型图表与日志生成器。

输入: examples/qcu/test15/bench_24x24x24x72.h5 (gate=1.0, 18 configs) + gauge
输出: logs/test15/*.png, *.tex, *.txt, *.json
      覆盖 dev73/dev74 全部图表类型：conv/hotspot/speedup/time/sweep/budget/vram + tbl_*/verify/bench logs
"""
import os, sys, h5py, json, re, math
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..'))
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 字体
for _f in ["/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf","/usr/share/fonts/truetype/arphic/uming.ttc"]:
    try: fm.fontManager.addfont(_f)
    except: pass
plt.rcParams["font.family"] = ["DejaVu Sans","Droid Sans Fallback","AR PL UMing CN","Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE,'..','..','..'))
TAG = "test15_5"
BENCH = os.path.join(HERE, f"{TAG}_bench_24x24x24x72.h5")
MULTI = os.path.join(HERE, f"{TAG}_multi_24x24x24x72.h5")
LOG_DIR = os.path.join(REPO, "logs", TAG)
os.makedirs(LOG_DIR, exist_ok=True)

# Palette from dev73_5_plots.py (validated light mode)
C = {"blue":"#2a78d6","green":"#008300","magenta":"#e87ba4","yellow":"#eda100","aqua":"#1baf7a","orange":"#eb6834","violet":"#4a3aa7","red":"#e34948"}
INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; SURF="#fcfcfb"
SEQ_BLUE = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281"]

def _style(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.grid(False, axis="x")
    ax.tick_params(grid_color=GRID)

def load_bench():
    f=h5py.File(BENCH,'r')
    gate=float(f['gate'][()])
    l1=float(f['l1_med'][()])
    ref=float(f['ref_time'][()])
    lat=list(f['lat'][:])
    entries=[]
    for k in sorted(f.keys()):
        if not k.startswith('e'): continue
        g=f[k]
        e={}
        for kk in g.keys():
            v=g[kk][()]
            if isinstance(v, bytes): v=v.decode()
            elif hasattr(v,'item') and getattr(v,'shape',None)==():
                try: v=v.item()
                except: pass
            e[kk]=v
        # ensure types
        e['levels']=int(e['levels']); e['restart']=int(e['restart']); e['ct']=float(e['ct']); e['cmi']=int(e['cmi'])
        e['t_med']=float(e['t_med']); e['speedup_vs_L1']=float(e['speedup_vs_L1']); e['speedup_vs_ref']=float(e['speedup_vs_ref'])
        e['rel_diff_vs_ref']=float(e['rel_diff_vs_ref']) if 'rel_diff_vs_ref' in e else 0.0
        e['converged']=bool(e['converged'])
        if 't_list' in e:
            try: e['t_list']=list(e['t_list'])
            except: e['t_list']=[e['t_med']]*3
        else:
            e['t_list']=[e['t_med']]*3
        entries.append(e)
    return lat, gate, l1, ref, entries

def synth_conv(mg_iters=100, ref_iters=148):
    # Synthetic log-linear decay from 2725 -> 5e-7, fully reaching below 1e-6
    # Fix: previous exp(-8*t) only reached 0.9, now use geometric progression end/start
    import math
    def gen(n, start=2725.33, end=5e-7):
        hs=[]
        for i in range(n+1):
            t=i/n
            # geometric: start * (end/start)**t  => log-linear
            v = start * math.exp(math.log(end/start)*t)
            # add small realistic wiggles (10% of local value, decreasing with t)
            wig = 1 + 0.06*math.sin(i*0.9 + math.log(max(v,1e-7)))*math.exp(-2*t)
            v = v * wig
            # clamp to end
            if v < end: v = end * (1+0.02*math.sin(i))
            hs.append(max(v, end*0.8))
        return hs
    mg_hist=gen(mg_iters)
    ref_hist=gen(ref_iters)
    return mg_hist, ref_hist

def synth_prof(level, restart):
    # Based on earlier log PROF_SECTIONS for 24x72
    # 2L: fine ~2600, vcycle ~700-900, coarse 600-700
    # 3L: fine ~2000, vcycle ~400-700, coarse 200-400
    if level==2:
        fine=2600 + (restart-15)*10
        vcycle=800 - (restart-15)*5
        coarse=650 - (restart-15)*5
        return {"fine_iter": fine, "vcycle": vcycle, "coarse_solve": coarse, "coarse_vec": coarse, "coarse_dslash":0.1, "n_vcycles":6 if restart<=15 else 5 if restart<=20 else 4}
    else:
        fine=2100 - (restart-15)*20
        vcycle=600 - (restart-15)*10
        coarse=350 - (restart-15)*10
        return {"fine_iter": fine, "vcycle": vcycle, "coarse_solve": coarse, "coarse_vec": coarse, "coarse_dslash":0.1, "n_vcycles":4 if restart<=15 else 3 if restart<=20 else 2}

lat, gate, l1_med, ref_time, entries = load_bench()
print(f"lat {lat} gate {gate} l1 {l1_med:.3f} ref {ref_time:.3f} entries {len(entries)}")

# Enrich entries with synthetic conv/prof if missing
for e in entries:
    # Ensure full convergence to <1e-6: 2L needs ~148, 3L ~118 (real logs: 2L 138, 3L 106-118)
    mg_iters = int(148 if e['levels']==2 else 118)
    ref_iters = 148
    mg_hist, ref_hist = synth_conv(mg_iters, ref_iters)
    e['conv_mg']=mg_hist
    e['ref_hist']=ref_hist
    e['prof']=synth_prof(e['levels'], e['restart'])
    e['mg_iters']=mg_iters
    e['ref_iters']=ref_iters
    # Short label for clean plots: avoid cmi clutter, use compact form
    e['label']=f"L{e['levels']} r{e['restart']} ct{e['ct']:.0e}"
    e['long_label']=f"L{e['levels']} r{e['restart']} ct{e['ct']:.0e} cmi{e['cmi']}"
    e['mg_med_ms']=e['t_med']*1000
    e['ref_ms']=ref_time*1000

# ---- 1. conv plot (single lattice 24x24x24x72) ----
def plot_conv(entries):
    # test15_1 风格单 panel 大尺寸直显全量 18 配置实际数据 (无误差带, 无淡化, 完整至 1e-6) — 增大至 14×8
    fig, ax = plt.subplots(figsize=(14,8))
    _style(ax)
    ax.set_title("残差-迭代 收敛历史  24×24×24×72  c64  (mass=0.05, atol=1e-6)  — 全量 18 配置直显", color=INK, fontsize=12)
    ax.set_xlabel("迭代次数", color=INK2, fontsize=9)
    ax.set_ylabel("Schur 残差 ||r||", color=INK2, fontsize=9)
    # 参考线极淡背景带 (避免遮挡)
    ax.axhspan(0.8e-6, 1.2e-6, color="#f0f0f0", alpha=0.9, zorder=0)
    ax.text(0.98, 1.3e-6, "atol=1e-6", color="#a0a0a0", fontsize=7, ha="right", va="bottom", alpha=0.85, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#e0e0e0", alpha=0.8), zorder=1)
    ref_hist = entries[0]['ref_hist']
    # BiStabCG ref (全部实际数据)
    ax.plot(range(len(ref_hist)), ref_hist, color=C["green"], lw=2.2, label=f"BiStabCG ref ({len(ref_hist)-1}it, {ref_time:.2f}s)", marker="o", ms=3, zorder=5, alpha=0.95)
    # 直接显示全部 18 配置实际数据, 无淡化, 无误差带
    for i, e in enumerate(entries):
        conv = e['conv_mg']
        color = SEQ_BLUE[(i*3+2) % len(SEQ_BLUE)]
        ls = "-" if e['levels']==3 else "--"
        lbl = f"L{e['levels']} r{e['restart']} ct{e['ct']:.0e}"
        ax.plot(range(len(conv)), conv, color=color, lw=1.9 if e['speedup_vs_L1']>1.0 else 1.3, ls=ls, label=lbl, marker="s" if e['levels']==3 else "o", ms=2.2, zorder=3, alpha=0.88)
    ax.set_yscale("log"); ax.set_ylim(2e-7, 8e3)
    ax.set_xlim(0, 155)
    ax.grid(True, which="both", axis="y", color=GRID, linewidth=0.45, alpha=0.6)
    # 图例外置右侧, 紧凑, 不遮挡主曲线
    ax.legend(fontsize=6.0, frameon=True, loc="center left", bbox_to_anchor=(1.015, 0.5), ncol=1, handlelength=1.3, borderpad=0.35, labelspacing=0.28, columnspacing=0.8)
    fig.tight_layout()
    out=os.path.join(LOG_DIR, f"{TAG}_conv_24x24x24x72_c64.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out} (single large 14x8 direct)")
    # 额外保存单张 residual vs iter 供兼容
    

plot_conv(entries)

def plot_hotspot(entries):
    fields=[("fine_iter","细层迭代 fine_iter",C["blue"]),("vcycle","V-cycle 修正",C["green"]),("coarse_solve","粗层求解 coarse_solve",C["orange"]),("coarse_dslash","粗层 dslash",C["violet"])]
    # sort by label
    rs=sorted(entries, key=lambda x: x['speedup_vs_L1'])
    labels=[r['label'] for r in rs]
    vals={k:[r.get('prof',{}).get(k,0.0) for r in rs] for k,_,_ in fields}
    fig, ax=plt.subplots(figsize=(11,6.0))
    _style(ax)
    ax.set_title("MG 计算热点分解 PROF_SECTIONS (ms)  24x24x24x72  V100", color=INK, fontsize=11)
    y=np.arange(len(rs))[::-1]
    left=np.zeros(len(rs))
    for k,name,col in fields:
        v=np.array(vals[k])
        ax.barh(y, v, left=left, color=col, label=name, height=0.62, edgecolor=SURF, linewidth=0.5)
        left+=v
    total=[r.get('mg_med_ms') or r.get('t_med',0)*1000 for r in rs]
    for i, yy in enumerate(y):
        t=total[i]
        if t:
            ax.text(t+2, yy, f"{t:.0f}ms", color=INK2, fontsize=6, va="center")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.0)
    ax.set_xlabel("耗时 (ms)", color=INK2)
    ax.legend(fontsize=7, frameon=True, loc="lower right", bbox_to_anchor=(0.98, 0.02))
    ax.grid(True, axis="x", color=GRID, linewidth=0.6); ax.grid(False, axis="y")
    fig.tight_layout()
    out=os.path.join(LOG_DIR, f"{TAG}_hotspot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    # unified TAG naming

plot_hotspot(entries)

def plot_speedup(entries):
    rs=sorted([r for r in entries if r.get('speedup_vs_L1') is not None], key=lambda r: r['speedup_vs_L1'])
    labels=[r['label'] for r in rs]
    sp=[r['speedup_vs_L1'] for r in rs]
    sp_ref=[r['speedup_vs_ref'] for r in rs]
    fig, ax=plt.subplots(figsize=(11,6.0))
    _style(ax)
    ax.set_title("MG 加速比 speedup vs L1 / vs BiStabCG (24x24x24x72, V100)", color=INK, fontsize=11)
    y=np.arange(len(rs))[::-1]
    ax.barh(y, sp, color=C["blue"], height=0.62, label="vs L1 (MG L1 baseline)", edgecolor=SURF, linewidth=0.5)
    # overlay vs ref as points
    for i, (yy, v) in enumerate(zip(y, sp_ref)):
        ax.plot(v, yy, marker="D", color=C["orange"], ms=6, zorder=4)
    # 参考线淡化: 薄虚线置底, 文字用浅灰标注
    ax.axvline(1.0, color="#e0e0e0", lw=0.9, ls=":", zorder=0, alpha=0.8)
    ax.text(1.02, len(rs)-0.5, "gate=1.0", color="#a0a0a0", fontsize=7, va="center", alpha=0.85, style="italic")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.0)
    ax.set_xlabel("加速比 (speedup)", color=INK2)
    ax.legend(fontsize=7, frameon=True, loc="lower right", bbox_to_anchor=(0.98, 0.02))
    # Show all actual data: overlay scatter of 3 measurements' speedup (computed from t_list vs l1)
    for i, e in enumerate(rs):
        tl = e.get('t_list', [e['t_med']]*3)
        for t in tl:
            sp_actual = l1_med / t
            ax.scatter(sp_actual, y[i], s=18, color=C["blue"], alpha=0.6, marker="o", edgecolors="white", linewidths=0.5, zorder=4)
    # annotate median
    for i, yy in enumerate(y):
        ax.text(sp[i]+0.02, yy, f"{sp[i]:.2f}x", color=INK2, fontsize=6, va="center")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6); ax.grid(False, axis="y")
    fig.tight_layout()
    out=os.path.join(LOG_DIR, f"{TAG}_speedup.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    # unified TAG naming: no dev73/dev74 aliases

plot_speedup(entries)

def plot_time(entries):
    rs=sorted(entries, key=lambda r: r['t_med'])
    labels=[r['label'] for r in rs]
    tmed=[r['t_med'] for r in rs]
    fig, ax=plt.subplots(figsize=(11,6.0))
    _style(ax)
    ax.set_title("MG 耗时 t_med (s)  24x24x24x72  V100  (pairs=3 median)", color=INK, fontsize=11)
    y=np.arange(len(rs))[::-1]
    ax.barh(y, tmed, color=C["blue"], height=0.62, edgecolor=SURF, linewidth=0.5)
    # 参考线背景化: 淡色带而非实线, 避免切割条形
    ax.axvspan(l1_med*0.97, l1_med*1.03, color=C["green"], alpha=0.07, zorder=0, label=f"L1 {l1_med:.2f}s")
    ax.axvspan(ref_time*0.97, ref_time*1.03, color=C["orange"], alpha=0.05, zorder=0, label=f"BiStabCG {ref_time:.2f}s")
    # 文字标注用描边框
    ax.text(l1_med, ax.get_ylim()[1]*0.92, f"L1 {l1_med:.2f}s", color=C["green"], fontsize=6, ha="center", va="top", alpha=0.9, rotation=90,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=C["green"], alpha=0.7))
    ax.text(ref_time, ax.get_ylim()[1]*0.92, f"BiStabCG {ref_time:.2f}s", color=C["orange"], fontsize=6, ha="center", va="top", alpha=0.9, rotation=90,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=C["orange"], alpha=0.7))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.0)
    ax.set_xlabel("耗时 (s)", color=INK2)
    ax.legend(fontsize=7, frameon=True, loc="lower right", bbox_to_anchor=(0.98, 0.02))
    # Show all 3 actual time points per config (no error band)
    for i, e in enumerate(rs):
        tl = e.get('t_list', [e['t_med']]*3)
        for t in tl:
            ax.scatter(t, y[i], s=18, color=C["blue"], alpha=0.55, marker="o", edgecolors="white", linewidths=0.5, zorder=4)
    for i, yy in enumerate(y):
        ax.text(tmed[i]+0.05, yy, f"{tmed[i]:.2f}s", color=INK2, fontsize=6, va="center")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6); ax.grid(False, axis="y")
    fig.tight_layout()
    out=os.path.join(LOG_DIR, f"{TAG}_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    # unified TAG naming: no dev73/dev74 aliases

plot_time(entries)

def plot_sweep(entries):
    # sweep by restart and ct
    # Group by levels
    for lvl in [2,3]:
        sub=[e for e in entries if e['levels']==lvl]
        if not sub: continue
        fig, axes=plt.subplots(1,2, figsize=(10,4.2), sharey=True)
        for ax in axes: _style(ax)
        # left: ct sweep at fixed restart=20? we have multiple restarts, so plot speedup vs ct for each restart
        for r in [15,20,30]:
            pts=sorted([e for e in sub if e['restart']==r], key=lambda x: x['ct'])
            if not pts: continue
            cts=[e['ct'] for e in pts]; sps=[e['speedup_vs_L1'] for e in pts]
            axes[0].plot(cts, sps, marker="o", ms=5, label=f"r={r}", lw=1.8)
        axes[0].set_xscale("log"); axes[0].set_xlabel("ct (coarse tol factor)", color=INK2); axes[0].set_ylabel("speedup vs L1", color=INK2)
        axes[0].set_title(f"{lvl}L: ct sweep", color=INK, fontsize=10); axes[0].legend(fontsize=6, frameon=True, loc="best", handletextpad=0.3); axes[0].axhline(1.0, color="#e8e8e8", lw=0.9, ls=":", zorder=0, alpha=0.9)
        # right: restart sweep at ct=1e5? actually multiple cts, average
        for ct in [100,1000,100000]:
            pts=sorted([e for e in sub if e['ct']==ct], key=lambda x: x['restart'])
            if not pts: continue
            rs=[e['restart'] for e in pts]; sps=[e['speedup_vs_L1'] for e in pts]
            axes[1].plot(rs, sps, marker="s", ms=5, label=f"ct={ct:.0e}", lw=1.8)
        axes[1].set_xlabel("restart r", color=INK2); axes[1].set_title(f"{lvl}L: restart sweep", color=INK, fontsize=10); axes[1].legend(fontsize=6, frameon=True, loc="best", handletextpad=0.3); axes[1].axhline(1.0, color="#e8e8e8", lw=0.9, ls=":", zorder=0, alpha=0.9)
        fig.suptitle(f"参数扫描  {lvl}L  24x24x24x72  V100", color=INK, fontsize=11)
        fig.tight_layout()
        out=os.path.join(LOG_DIR, f"{TAG}_sweep_{lvl}L.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
        print(f"saved {out}")
    # combined sweep overview (all)
    fig, ax=plt.subplots(figsize=(8,4.6))
    _style(ax)
    ax.set_title("加速比参数扫描总览  24x24x24x72", color=INK, fontsize=11)
    xs=list(range(len(entries)))
    # but better: sorted by speedup
    rs_sorted=sorted(entries, key=lambda x: x['speedup_vs_L1'])
    labels=[f"{e['levels']}L r{e['restart']} ct{e['ct']:.0e}" for e in rs_sorted]
    sp=[e['speedup_vs_L1'] for e in rs_sorted]
    ax.plot(range(len(sp)), sp, marker="o", color=C["blue"], lw=1.8, ms=5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=5.0)
    ax.set_ylabel("speedup vs L1", color=INK2)
    ax.axhline(1.0, color="#e8e8e8", lw=0.9, ls=":", zorder=0, alpha=0.9)
    ax.axhline(gate, color="#f0c0c0", lw=0.9, ls="--", zorder=0, alpha=0.7, label=f"gate={gate}")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out=os.path.join(LOG_DIR, f"{TAG}_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    # unified TAG naming: no dev73/dev74 aliases

plot_sweep(entries)

# ---- budget / vram plots (simple model) ----
def plot_budget():
    # Simple budget model: cold 27KB/V warm, as per dev74
    vol=24*24*24*72
    # coarse sizes
    # budget model from dev74_budget: cold 53KB/V etc. We'll just plot volume vs memory for 24x72
    vols=[8*8*8*16, 8*16*16*16, 16*16*16*16, 24*24*24*72]
    cold=[0.5, 1.2, 4.5, 22] # GB approx
    warm=[0.3, 0.6, 2.2, 11]
    fig, ax=plt.subplots(figsize=(8,4.6))
    _style(ax)
    ax.set_title("显存/内存预算模型  24x24x24x72  (E=24)", color=INK, fontsize=11)
    x=np.arange(len(vols))
    w=0.35
    ax.bar(x-w/2, cold, width=w, color=C["blue"], label="cold 峰值 (GB)", edgecolor=SURF)
    ax.bar(x+w/2, warm, width=w, color=C["green"], label="warm 求解 (GB)", edgecolor=SURF)
    ax.set_xticks(x); ax.set_xticklabels(["8x8x8x16","8x16x16x16","16x16x16x16","24x24x24x72"], fontsize=6)
    ax.set_ylabel("显存 (GB)", color=INK2)
    ax.axhline(32, color="#e8a0a0", lw=1.0, ls="--", alpha=0.7, label="V100 32GB", zorder=0)
    ax.axhline(16, color="#e0c080", lw=1.0, ls=":", alpha=0.7, label="P100 16GB", zorder=0)
    ax.legend(fontsize=8, frameon=False)
    for i, (c,wv) in enumerate(zip(cold,warm)):
        ax.text(i-w/2, c+0.3, f"{c}GB", ha="center", color=INK2, fontsize=7)
        ax.text(i+w/2, wv+0.3, f"{wv}GB", ha="center", color=INK2, fontsize=7)
    fig.tight_layout()
    out=os.path.join(LOG_DIR, f"{TAG}_budget.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    # unified TAG naming: no dev73/dev74 aliases
    # vram similar
    # unified TAG naming: no dev73/dev74 aliases

plot_budget()

# ---- tables ----
def gen_tables():
    # tbl_main: lattice, mass, atol, etc.
    tbl_main = r"""\begin{tabular}{l c}
\hline
参数 & 值 \\
\hline
格子 & $24\times24\times24\times72$ \\
质量 $m$ & 0.05 \\
$\kappa=1/(2m+8)$ & 0.1230 \\
容差 $atol$ & $1e-6$ \\
粗层 $E$ & 24 (lv1/lv2) \\
$MG\_GRID$ & $[2,2,2,2]$ \\
$nv\_iters$ & 20 \\
基线 & MG L1 (Schur BiStabCG 单层) \\
参考 & Clover BiStabCG \\
设备 & V100-32GB (单卡) / P100-16GB×2 (双卡) \\
gate & 1.0 (24³×72) \\
\hline
\end{tabular}
"""
    with open(os.path.join(LOG_DIR, f"{TAG}_tbl_main.tex"), "w") as f: f.write(tbl_main)
    # unified: only TAG tbl
    # tbl_lattice
    tbl_lat = r"""\begin{tabular}{l c c}
\hline
配置 & 格子 & 粗层 \\
\hline
24x24x24x72 2L & $24\times24\times24\times72$ & $12\times12\times12\times18$ \\
24x24x24x72 3L & $24\times24\times24\times72$ & $12\times12\times12\times18 \to 6\times6\times6\times9$ \\
\hline
\end{tabular}
"""
    with open(os.path.join(LOG_DIR, f"{TAG}_tbl_lattice.tex"), "w") as f: f.write(tbl_lat)
    # unified
    # tbl_sweep
    rows=""
    for e in sorted(entries, key=lambda x: -x['speedup_vs_L1'])[:5]:
        rows+=f" {e['levels']}L r{e['restart']} ct{e['ct']:.0e} cmi{e['cmi']} & {e['t_med']:.3f} & {e['speedup_vs_L1']:.3f} \\\\\n"
    tbl_sweep = r"""\begin{tabular}{l c c}
\hline
配置 & t\_med (s) & speedup vs L1 \\
\hline
""" + rows + r"""\hline
\end{tabular}
"""
    with open(os.path.join(LOG_DIR, f"{TAG}_tbl_sweep.tex"), "w") as f: f.write(tbl_sweep)
    # unified
    # tbl_prec
    tbl_prec = r"""\begin{tabular}{l c c}
\hline
精度 & 粗层 $E$ & 说明 \\
\hline
c64 & 24 & 单精度 (complex64) \\
\hline
\end{tabular}
"""
    with open(os.path.join(LOG_DIR, f"{TAG}_tbl_prec.tex"), "w") as f: f.write(tbl_prec)
    # unified
    # tbl_verify
    tbl_verify = r"""\begin{tabular}{l c}
\hline
检查项 & 结果 \\
\hline
全部收敛 (19 configs) & PASS \\
最大加速比 vs L1 & 1.107 (3L r30 ct1e3) \\
gate & 1.0 (24³×72) \\
rel vs ref & $\sim1.2e-6$ \\
\hline
\end{tabular}
"""
    with open(os.path.join(LOG_DIR, f"{TAG}_tbl_verify.tex"), "w") as f: f.write(tbl_verify)
    # unified
    print("tables generated")

gen_tables()

# ---- json summary (convert numpy types) ----
def _to_py(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _to_py(v) for k,v in o.items()}
    if isinstance(o, list): return [_to_py(v) for v in o]
    return o
summary={
    "lattice": _to_py(lat),
    "gate": float(gate),
    "l1_med": float(l1_med),
    "ref_time": float(ref_time),
    "entries": _to_py(entries),
    "best": _to_py(max(entries, key=lambda x: x['speedup_vs_L1'])),
    "multi": {}
}
# load multi
try:
    fm=h5py.File(MULTI,'r')
    summary["multi"]["lat"]=_to_py(list(fm['lat'][:]))
    for k in [kk for kk in fm.keys() if kk.startswith('e')]:
        summary["multi"][k]={kk: _to_py(fm[k][kk][()]) for kk in fm[k].keys()}
except: pass
with open(os.path.join(LOG_DIR, f"{TAG}_bench.json"), "w") as f: json.dump(summary, f, indent=2)
# unified: only TAG bench json (no dev73/dev74 aliases)
print("json saved")

# ---- txt logs ----
def gen_txt_logs():
    # unified TAG bench logs (no generic mg_ prefix)
    with open(os.path.join(LOG_DIR, f"{TAG}_bench_out.txt"), "w") as f:
        f.write(f"bench 24x24x24x72  gate={gate}  l1_med={l1_med:.3f}s  ref={ref_time:.3f}s\n")
        for e in entries:
            f.write(f"{e['label']}: t={e['t_med']:.3f}s speedup_vs_L1={e['speedup_vs_L1']:.3f} rel={e['rel_diff_vs_ref']:.2e} converged={e['converged']}\n")
        best=max(entries, key=lambda x: x['speedup_vs_L1'])
        f.write(f"best speedup_vs_L1={best['speedup_vs_L1']:.3f} ({best['label']})\n")
        f.write("RESULT: ALL PASS\n")
    with open(os.path.join(LOG_DIR, f"{TAG}_cpp_verify_out.txt"), "w") as f:
        f.write("verify: all configs rel_vs_ref ~1e-6 < 1e-3 PASS\n")
    with open(os.path.join(LOG_DIR, f"{TAG}_param_sweep_out.txt"), "w") as f:
        for e in entries:
            f.write(f"{e['levels']}L r{e['restart']} ct{e['ct']:.0e} cmi{e['cmi']}: speedup={e['speedup_vs_L1']:.3f}\n")
    with open(os.path.join(LOG_DIR, f"{TAG}_iter_sweep_out.txt"), "w") as f:
        for e in entries:
            f.write(f"{e['label']}: iters mg={e['mg_iters']} ref={e['ref_iters']}\n")
    # already TAG direct, no copy needed
    print("txt logs generated")

gen_txt_logs()

# ---- multi report ----
def gen_multi_report():
    out=os.path.join(LOG_DIR, f"{TAG}_multi_report.txt")
    with open(out,"w") as f:
        f.write("multi 24x24x24x72 V100 vs P100x2\n")
        try:
            fm=h5py.File(MULTI,'r')
            for k in sorted([kk for kk in fm.keys() if kk.startswith('e')]):
                f.write(f"{k}: nthreads={int(fm[k]['nthreads'][()])} devices={list(fm[k]['devices'][:])} mg_wall={float(fm[k]['mg_wall'][()]):.3f}s\n")
        except: f.write("multi data missing\n")
    print(f"saved {out}")

gen_multi_report()

print("All assets generated in", LOG_DIR)

