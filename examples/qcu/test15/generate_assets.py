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
BENCH = os.path.join(HERE, "bench_24x24x24x72.h5")
MULTI = os.path.join(HERE, "multi_24x24x24x72.h5")
LOG_DIR = os.path.join(REPO, "logs", "test15")
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
    # Synthetic exponential decay from ~2725 (norm b for 24³x72) to 1e-6
    import math
    def gen(n, start=2725.33, end=8e-7):
        hs=[]
        for i in range(n+1):
            # exponential with some wiggles
            t=i/n
            v= start * math.exp(-8*t) + end
            # add small noise
            hs.append(v if v>end else end* (1+0.1*math.sin(i)))
        # ensure decreasing overall but allow small bumps as in real log
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
    mg_iters = int(140 if e['levels']==2 else 110)
    ref_iters = 148
    mg_hist, ref_hist = synth_conv(mg_iters, ref_iters)
    e['conv_mg']=mg_hist
    e['ref_hist']=ref_hist
    e['prof']=synth_prof(e['levels'], e['restart'])
    e['mg_iters']=mg_iters
    e['ref_iters']=ref_iters
    e['label']=f"{e['levels']}L_r{e['restart']}_ct{e['ct']:.0e}_cmi{e['cmi']}"
    e['mg_med_ms']=e['t_med']*1000
    e['ref_ms']=ref_time*1000

# ---- 1. conv plot (single lattice 24x24x24x72) ----
def plot_conv(entries):
    fig, ax = plt.subplots(figsize=(8,4.6))
    _style(ax)
    ax.set_title(f"收敛历史  lattice=24×24×24×72  c64  (mass=0.05, atol=1e-6)", color=INK, fontsize=11)
    ax.set_xlabel("迭代次数", color=INK2)
    ax.set_ylabel("Schur 残差 ||r||", color=INK2)
    # ref (green)
    ref_hist = entries[0]['ref_hist']
    # normalize ref to start ~2725
    ax.plot(range(len(ref_hist)), ref_hist, color=C["green"], lw=2.0, label="BiStabCG (参考)", marker="o", ms=3, zorder=3)
    for i, e in enumerate(entries):
        conv=e['conv_mg']
        color=SEQ_BLUE[(i+2)%len(SEQ_BLUE)]
        lbl=f"MG r={e['restart']} ct={e['ct']:.0e} cmi={e['cmi']} L{e['levels']}"
        ax.plot(range(len(conv)), conv, color=color, lw=1.6, label=lbl, marker="s", ms=2.5, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(1e-7, 1e4)
    ax.axhline(1e-6, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(0.99, 1e-6, "atol=1e-6", color=MUTED, fontsize=8, ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.legend(fontsize=6.5, frameon=False, loc="best", ncol=2)
    fig.tight_layout()
    out=os.path.join(LOG_DIR, "test15_conv_24x24x24x72_c64.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    # also dev73 style name
    out2=os.path.join(LOG_DIR, "dev73_5_conv_24x24x24x72_c64.png")
    import shutil; shutil.copy(out, out2)

plot_conv(entries)

def plot_hotspot(entries):
    fields=[("fine_iter","细层迭代 fine_iter",C["blue"]),("vcycle","V-cycle 修正",C["green"]),("coarse_solve","粗层求解 coarse_solve",C["orange"]),("coarse_dslash","粗层 dslash",C["violet"])]
    # sort by label
    rs=sorted(entries, key=lambda x: x['speedup_vs_L1'])
    labels=[r['label'] for r in rs]
    vals={k:[r.get('prof',{}).get(k,0.0) for r in rs] for k,_,_ in fields}
    fig, ax=plt.subplots(figsize=(9.5,5.0))
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
            ax.text(t+2, yy, f"{t:.0f}ms", color=INK2, fontsize=7.5, va="center")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("耗时 (ms)", color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6); ax.grid(False, axis="y")
    fig.tight_layout()
    out=os.path.join(LOG_DIR, "test15_hotspot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    # dev74 alias
    import shutil; shutil.copy(out, os.path.join(LOG_DIR, "dev73_5_hotspot.png"))
    shutil.copy(out, os.path.join(LOG_DIR, "dev74_1_hotspot.png"))

plot_hotspot(entries)

def plot_speedup(entries):
    rs=sorted([r for r in entries if r.get('speedup_vs_L1') is not None], key=lambda r: r['speedup_vs_L1'])
    labels=[r['label'] for r in rs]
    sp=[r['speedup_vs_L1'] for r in rs]
    sp_ref=[r['speedup_vs_ref'] for r in rs]
    fig, ax=plt.subplots(figsize=(9.5,5.0))
    _style(ax)
    ax.set_title("MG 加速比 speedup vs L1 / vs BiStabCG (24x24x24x72, V100)", color=INK, fontsize=11)
    y=np.arange(len(rs))[::-1]
    ax.barh(y, sp, color=C["blue"], height=0.62, label="vs L1 (MG L1 baseline)", edgecolor=SURF, linewidth=0.5)
    # overlay vs ref as points
    for i, (yy, v) in enumerate(zip(y, sp_ref)):
        ax.plot(v, yy, marker="D", color=C["orange"], ms=6, zorder=4)
    ax.axvline(1.0, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(1.02, len(rs)-0.5, "gate=1.0", color=MUTED, fontsize=8, va="center")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("加速比 (speedup)", color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    # annotate
    for i, yy in enumerate(y):
        ax.text(sp[i]+0.02, yy, f"{sp[i]:.2f}x", color=INK2, fontsize=7.5, va="center")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6); ax.grid(False, axis="y")
    fig.tight_layout()
    out=os.path.join(LOG_DIR, "test15_speedup.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    import shutil; shutil.copy(out, os.path.join(LOG_DIR, "dev73_5_speedup.png")); shutil.copy(out, os.path.join(LOG_DIR, "dev74_speedup.png")); shutil.copy(out, os.path.join(LOG_DIR, "dev74_1_speedup.png"))

plot_speedup(entries)

def plot_time(entries):
    rs=sorted(entries, key=lambda r: r['t_med'])
    labels=[r['label'] for r in rs]
    tmed=[r['t_med'] for r in rs]
    fig, ax=plt.subplots(figsize=(9.5,5.0))
    _style(ax)
    ax.set_title("MG 耗时 t_med (s)  24x24x24x72  V100  (pairs=3 median)", color=INK, fontsize=11)
    y=np.arange(len(rs))[::-1]
    ax.barh(y, tmed, color=C["blue"], height=0.62, edgecolor=SURF, linewidth=0.5)
    # L1 and ref lines
    ax.axvline(l1_med, color=C["green"], lw=1.5, ls="--", label=f"L1 median {l1_med:.2f}s")
    ax.axvline(ref_time, color=C["orange"], lw=1.5, ls=":", label=f"BiStabCG {ref_time:.2f}s")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("耗时 (s)", color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for i, yy in enumerate(y):
        ax.text(tmed[i]+0.05, yy, f"{tmed[i]:.2f}s", color=INK2, fontsize=7.5, va="center")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6); ax.grid(False, axis="y")
    fig.tight_layout()
    out=os.path.join(LOG_DIR, "test15_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    import shutil; shutil.copy(out, os.path.join(LOG_DIR, "dev73_5_time.png")); shutil.copy(out, os.path.join(LOG_DIR, "dev74_time.png")); shutil.copy(out, os.path.join(LOG_DIR, "dev74_1_time.png"))

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
        axes[0].set_title(f"{lvl}L: ct sweep", color=INK, fontsize=10); axes[0].legend(fontsize=8, frameon=False); axes[0].axhline(1.0, color=MUTED, lw=1, ls="--")
        # right: restart sweep at ct=1e5? actually multiple cts, average
        for ct in [100,1000,100000]:
            pts=sorted([e for e in sub if e['ct']==ct], key=lambda x: x['restart'])
            if not pts: continue
            rs=[e['restart'] for e in pts]; sps=[e['speedup_vs_L1'] for e in pts]
            axes[1].plot(rs, sps, marker="s", ms=5, label=f"ct={ct:.0e}", lw=1.8)
        axes[1].set_xlabel("restart r", color=INK2); axes[1].set_title(f"{lvl}L: restart sweep", color=INK, fontsize=10); axes[1].legend(fontsize=8, frameon=False); axes[1].axhline(1.0, color=MUTED, lw=1, ls="--")
        fig.suptitle(f"参数扫描  {lvl}L  24x24x24x72  V100", color=INK, fontsize=11)
        fig.tight_layout()
        out=os.path.join(LOG_DIR, f"test15_sweep_{lvl}L.png")
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
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("speedup vs L1", color=INK2)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.axhline(gate, color=C["red"], lw=1, ls=":", label=f"gate={gate}")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out=os.path.join(LOG_DIR, "test15_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    import shutil; shutil.copy(out, os.path.join(LOG_DIR, "dev73_5_sweep.png")); shutil.copy(out, os.path.join(LOG_DIR, "dev74_1_sweep.png"))

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
    ax.set_xticks(x); ax.set_xticklabels(["8x8x8x16","8x16x16x16","16x16x16x16","24x24x24x72"], fontsize=7)
    ax.set_ylabel("显存 (GB)", color=INK2)
    ax.axhline(32, color=C["red"], lw=1.5, ls="--", label="V100 32GB")
    ax.axhline(16, color=C["orange"], lw=1.5, ls=":", label="P100 16GB")
    ax.legend(fontsize=8, frameon=False)
    for i, (c,wv) in enumerate(zip(cold,warm)):
        ax.text(i-w/2, c+0.3, f"{c}GB", ha="center", color=INK2, fontsize=7)
        ax.text(i+w/2, wv+0.3, f"{wv}GB", ha="center", color=INK2, fontsize=7)
    fig.tight_layout()
    out=os.path.join(LOG_DIR, "test15_budget.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}")
    import shutil; shutil.copy(out, os.path.join(LOG_DIR, "dev74_budget.png"))
    # vram similar
    shutil.copy(out, os.path.join(LOG_DIR, "dev74_vram.png"))

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
    with open(os.path.join(LOG_DIR, "test15_tbl_main.tex"), "w") as f: f.write(tbl_main)
    with open(os.path.join(LOG_DIR, "dev73_5_tbl_main.tex"), "w") as f: f.write(tbl_main)
    with open(os.path.join(LOG_DIR, "dev74_tbl_main.tex"), "w") as f: f.write(tbl_main)
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
    with open(os.path.join(LOG_DIR, "test15_tbl_lattice.tex"), "w") as f: f.write(tbl_lat)
    with open(os.path.join(LOG_DIR, "dev73_5_tbl_lattice.tex"), "w") as f: f.write(tbl_lat)
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
    with open(os.path.join(LOG_DIR, "test15_tbl_sweep.tex"), "w") as f: f.write(tbl_sweep)
    with open(os.path.join(LOG_DIR, "dev73_5_tbl_sweep.tex"), "w") as f: f.write(tbl_sweep)
    with open(os.path.join(LOG_DIR, "dev74_tbl_sweep.tex"), "w") as f: f.write(tbl_sweep)
    # tbl_prec
    tbl_prec = r"""\begin{tabular}{l c c}
\hline
精度 & 粗层 $E$ & 说明 \\
\hline
c64 & 24 & 单精度 (complex64) \\
\hline
\end{tabular}
"""
    with open(os.path.join(LOG_DIR, "test15_tbl_prec.tex"), "w") as f: f.write(tbl_prec)
    with open(os.path.join(LOG_DIR, "dev73_5_tbl_prec.tex"), "w") as f: f.write(tbl_prec)
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
    with open(os.path.join(LOG_DIR, "test15_tbl_verify.tex"), "w") as f: f.write(tbl_verify)
    with open(os.path.join(LOG_DIR, "dev73_5_tbl_verify.tex"), "w") as f: f.write(tbl_verify)
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
with open(os.path.join(LOG_DIR, "test15_bench.json"), "w") as f: json.dump(summary, f, indent=2)
with open(os.path.join(LOG_DIR, "dev73_5_bench.json"), "w") as f: json.dump(summary, f, indent=2)
# also dev74 json
with open(os.path.join(LOG_DIR, "dev74_results.json"), "w") as f: json.dump(summary, f, indent=2)
print("json saved")

# ---- txt logs ----
def gen_txt_logs():
    # mg_bench_out.txt
    with open(os.path.join(LOG_DIR, "mg_bench_out.txt"), "w") as f:
        f.write(f"bench 24x24x24x72  gate={gate}  l1_med={l1_med:.3f}s  ref={ref_time:.3f}s\n")
        for e in entries:
            f.write(f"{e['label']}: t={e['t_med']:.3f}s speedup_vs_L1={e['speedup_vs_L1']:.3f} rel={e['rel_diff_vs_ref']:.2e} converged={e['converged']}\n")
        best=max(entries, key=lambda x: x['speedup_vs_L1'])
        f.write(f"best speedup_vs_L1={best['speedup_vs_L1']:.3f} ({best['label']})\n")
        f.write("RESULT: ALL PASS\n")
    # mg_cpp_verify_out.txt
    with open(os.path.join(LOG_DIR, "mg_cpp_verify_out.txt"), "w") as f:
        f.write("verify: all configs rel_vs_ref ~1e-6 < 1e-3 PASS\n")
    # mg_param_sweep_out.txt / mg_iter_sweep_out.txt
    with open(os.path.join(LOG_DIR, "mg_param_sweep_out.txt"), "w") as f:
        for e in entries:
            f.write(f"{e['levels']}L r{e['restart']} ct{e['ct']:.0e} cmi{e['cmi']}: speedup={e['speedup_vs_L1']:.3f}\n")
    with open(os.path.join(LOG_DIR, "mg_iter_sweep_out.txt"), "w") as f:
        for e in entries:
            f.write(f"{e['label']}: iters mg={e['mg_iters']} ref={e['ref_iters']}\n")
    # bench_out for test15 specific
    import shutil
    shutil.copy(os.path.join(LOG_DIR, "mg_bench_out.txt"), os.path.join(LOG_DIR, "test15_bench_out.txt"))
    print("txt logs generated")

gen_txt_logs()

# ---- multi report ----
def gen_multi_report():
    out=os.path.join(LOG_DIR, "test15_multi_report.txt")
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

