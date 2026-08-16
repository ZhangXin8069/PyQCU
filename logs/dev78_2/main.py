#!/usr/bin/env python3
"""dev78_2 —— 多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件。

背景：test12 测试单线程 C++ MultiGrid；dev78_2 测试 `MultiGpuMultigrid`
（pyqcu/cuda/_multi_gpu.py）—— N 线程并行、每线程绑定一张 GPU（一线程一卡），
每线程在绑定的设备上运行完整 C++ 后端流程（参考 BiStabCG + Clover Multigrid）。

硬件分配（本机 3 卡，CUDA 运行时视角 device 0=V100-32GB, 1/2=P100-16GB×2）：
  * 多线程测试：2 线程 × P100×2（device_ids=[1,2]）—— 本套件主测试对象
  * 单线程大格子：V100（device_ids=[0]）
  * 三卡并行不测（任务约束）

基准：多线程版 CUDA C++ BiStabCG —— MultiGpuMultigrid.solve() 内部每线程并行
运行 applyCloverBistabCgQcu（参考解），多线程墙钟 multi_ref_wall = max(各线程
ref_time)；MG 多线程墙钟 multi_mg_wall = max(各线程 mg_time)；
speedup = multi_ref_wall / multi_mg_wall（加速比基准）。

数据持久化约定（本套件核心约束）：**所有数据/结果文件读写只用 h5py**
（参考 pyqcu/tools/_io.py 的 save_tensor_h5/load_tensor_h5：每次调用独立
File 句柄、with 语句，多线程安全；多 dataset 单句柄一次写完）。
结果字典以 attrs+datasets 存入 .h5（save_dict_h5/load_dict_h5）。
PNG/TeX 为图表展示产物（matplotlib / 文本渲染，非数据持久化）。

真多线程约束：求解热点（BiStabCG/MG 内核）全部在 worker 线程各自的卡上并行
执行（qcu.pyx with nogil 真并行），测试脚本只做编排与收集（测试用途允许）；
粗算子构建（setup 阶段，h5 缓存命中后秒级）在主线程 V100 完成（P100 sm_60
无 torch kernel image，见 pyqcu/cuda/AGENTS.md）。

子命令：
  verify        正确性（一致性模式 P100×2、独立问题模式、V100 单线程）+ h5py IO
  clean         干净测量（独立进程交叉计时 + RSS）→ dev78_2_clean_*.h5
  bench         批量基准（P100×2 多线程组 + V100 单线程大格子组）→ dev78_2_bench.h5
  sweep         参数扫描（r/ct/cmi/levels × speedup，P100×2）→ dev78_2_sweep.h5
  check         加速比断言（--gate 1.5 默认，exit 0/1/2）
  budget        显存/内存预算表（--vram 16 P100 档 / 32 V100 档）→ dev78_2_budget_*.h5
  conv          全求解器迭代残差历史收集（实测 MG CONVERGENCE_HISTORY +
                参考 BiStabCG Python 复现）→ dev78_2_conv.h5
  conv_plots    迭代残差图：逐配置 / 逐格子多参数对比 / 全配置汇总 → dev78_2_conv_*.png
  collect       汇总 → dev78_2_results.h5
  mktable       LaTeX 表 → dev78_2_tbl_*.tex
  plots         PNG 图 → dev78_2_*.png

用法（PyQCU 根下）：
  source ./env.sh && python logs/dev78_2/main.py <subcommand> [options]

约定（与 test12 一致）：mass=0.05, atol=1e-6, gauge_seed=42, kappa=1/(2m+8),
E=48, NV_ITERS=2, MG_GRID=[2,2,2,2]；nullvec 缓存共享 logs/nullvec_cache。
"""
import os, sys, time, glob, subprocess, resource, argparse

# ----------------------------------------------------------------------
# 路径与公共工具
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))            # logs/dev78_2
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

# 输出目录重定向：优先级 --outdir > 环境变量 TEST78_2_OUTDIR > 默认 logs/dev78_2
WORKDIR = os.environ.get("TEST78_2_OUTDIR", _HERE)
LOG_PATH = os.path.join(REPO, "logs", "clover_multigrid.log")

# 设备分配（任务约束：多线程=P100×2，单线程大格子=V100，不测三卡）
P100_IDS = [1, 2]
V100_ID = [0]
MASS, ATOL = 0.05, 1e-6
DOF_LIST = [12, 48, 48, 48, 48]
MG_GRID = [2, 2, 2, 2]
NV_ITERS = 2
GAUGE_SEED = 42


def _git_snapshot():
    try:
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            cwd=REPO, capture_output=True, text=True, timeout=10)
        hd = subprocess.run(["git", "log", "-1", "--oneline"],
                            cwd=REPO, capture_output=True, text=True, timeout=10)
        return br.stdout.strip() or "?", (hd.stdout.strip() or "?")
    except Exception:
        return "?", "?"


def _gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return "?"


def _gpu_used_mb():
    """当前进程在各卡的显存占用（nvidia-smi 采样，MB）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().replace("\n", "; ")
    except Exception:
        return "?"


def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def cache_dir():
    return os.environ.get("PYQCU_NULLVEC_CACHE",
                          os.path.join(REPO, "logs", "nullvec_cache"))


def cache_disk_mb():
    try:
        out = subprocess.run(["du", "-s", cache_dir()], capture_output=True,
                             text=True, timeout=10)
        return int(out.stdout.split()[0]) / 1024.0
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# h5py 结果 I/O（本套件唯一数据持久化方式）
# ----------------------------------------------------------------------
def save_dict_h5(path, d):
    """dict（标量/字符串/列表/嵌套标量 dict）→ .h5（attrs + datasets）。

    单句柄一次写入全部内容（h5py 多线程安全约定：每次调用独立 File 句柄）。
    标量/字符串 → attrs；list/tuple/ndarray → dataset 'd_<key>'。
    """
    import h5py
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, 'w') as f:
        for k, v in d.items():
            if isinstance(v, dict):
                g = f.create_group(k)
                for kk, vv in v.items():
                    _h5_write(g, kk, vv)
            else:
                _h5_write(f, k, v)
    os.replace(tmp, path)


def _h5_write(g, k, v):
    import numpy as np
    if isinstance(v, dict):
        sub = g.create_group(k)
        for kk, vv in v.items():
            _h5_write(sub, kk, vv)
    elif isinstance(v, (bool, int, float, complex)):
        g.attrs[k] = v
    elif isinstance(v, (str, np.str_)):
        # h5py 对 numpy.str_（定长 unicode）无转换路径，统一转 python str
        g.attrs[k] = str(v)
    elif isinstance(v, (list, tuple)):
        if v and all(isinstance(x, dict) for x in v):
            # dict 列表 → 子组 '0','1',...（h5py 无 Object dtype 等价物）
            sub = g.create_group(k)
            for i, x in enumerate(v):
                _h5_write(sub, str(i), x)
        elif v and all(isinstance(x, (list, tuple)) for x in v):
            # 变长数值列表（如 MG 残差历史每线程一条）→ h5py vlen dataset
            import h5py
            import numpy as _np
            dt = h5py.vlen_dtype(_np.dtype('f8'))
            dset = g.create_dataset("d_" + k, (len(v),), dtype=dt)
            for i, x in enumerate(v):
                dset[i] = _np.asarray(x, dtype='f8')
        else:
            g.create_dataset("d_" + k, data=np.asarray(v))
    elif isinstance(v, np.ndarray):
        g.create_dataset("d_" + k, data=v)
    elif v is None:
        g.attrs[k] = "None"
    else:
        g.attrs[k] = str(v)


def load_dict_h5(path):
    """save_dict_h5 的逆：h5 → dict（attrs 标量 + datasets 数组 + 子组）。"""
    import h5py
    d = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.attrs.items():
            d[k] = _h5_attr_val(v)
        for k in f.keys():
            item = f[k]
            if isinstance(item, h5py.Dataset):
                d[k] = item[...]
            else:
                d[k] = load_dict_h5_group(item)
    return d


def load_dict_h5_group(g):
    import h5py
    d = {}
    for k, v in g.attrs.items():
        d[k] = _h5_attr_val(v)
    for k in g.keys():
        item = g[k]
        if isinstance(item, h5py.Dataset):
            d[k] = item[...]
        else:
            d[k] = load_dict_h5_group(item)
    if d and all(k.isdigit() for k in d):
        # 数字 key 组 → dict 列表还原（_h5_write 的对称操作）
        return [d[str(i)] for i in range(len(d))]
    return d


def _h5_attr_val(v):
    import numpy as np
    if isinstance(v, (bytes,)):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in "US":
        return v.item().decode() if v.size == 1 else [x.decode() for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v


def dump_env_h5(cmdline=None):
    """输出目录写入 env.h5（环境快照，跨环境比对基准）。"""
    import socket, torch
    branch, head = _git_snapshot()
    env = {"branch": branch, "head": head,
           "gpu": _gpu_snapshot(),
           "torch": torch.__version__ if torch.cuda.is_available() else "cpu",
           "cuda_devices": torch.cuda.device_count(),
           "host": socket.gethostname(),
           "cmdline": " ".join(sys.argv) if cmdline is None else cmdline}
    save_dict_h5(os.path.join(WORKDIR, "env.h5"), {"env": env})
    return env


def parse_mg_log(path=LOG_PATH):
    """C++ 收敛日志 → (残差列表, PROF_SECTIONS 字典, 总迭代数)。"""
    if not os.path.exists(path):
        return [], {}, 0
    conv, prof = [], {}
    with open(path) as f:
        for line in f:
            if "Residual(norm2)" in line:
                import re
                m = re.search(r"Residual\(norm2\):\(([^,]+),", line)
                if m:
                    conv.append(float(m.group(1)))
            elif "PROF_SECTIONS" in line:
                for tok in line.split("PROF_SECTIONS:")[1].split():
                    if "=" in tok:
                        k, v = tok.split("=")
                        prof[k] = float(v)
    n_iter = max(0, len(conv) - 1)
    return conv, prof, n_iter


# ----------------------------------------------------------------------
# 测量：多线程（一线程一卡）CUDA C++ MultiGrid vs 多线程 BiStabCG
# ----------------------------------------------------------------------
def measure_multi(lat, num_levels=2, num_restart=5, coarse_max_iter=15,
                  coarse_tol_factor=1e5, nthreads=2, device_ids=None,
                  nv_iters=NV_ITERS, use_cache=True, verbose=False,
                  independent=False, label=None):
    """一次多线程求解测量（MultiGpuMultigrid.solve 内部 ref/mg 交叉计时）。

    返回结果字典（h5 可存）：每线程 tid/device/ref_time/mg_time/rel_diff，
    多线程墙钟 multi_ref_wall/multi_mg_wall，speedup，一致性摘要，资源统计。
    """
    from pyqcu.cuda._multi_gpu import MultiGpuMultigrid
    if device_ids is None:
        device_ids = list(P100_IDS if nthreads >= 2 else V100_ID)
    mg = MultiGpuMultigrid(lat_size=list(lat), mass=MASS, atol=ATOL,
                           num_levels=num_levels, dof_list=DOF_LIST,
                           mg_grid=MG_GRID, num_restart=num_restart,
                           coarse_max_iter=coarse_max_iter,
                           coarse_tol_factor=coarse_tol_factor,
                           nv_iters=nv_iters, nthreads=nthreads,
                           device_ids=list(device_ids), use_cache=use_cache,
                           cache_dir=cache_dir(), verbose=verbose,
                           independent_problems=independent)
    t0 = time.perf_counter()
    r = mg.solve()
    total_solve = time.perf_counter() - t0
    threads = r['threads']
    ref_max = max(t['ref_time'] for t in threads)
    mg_max = max(t['mg_time'] for t in threads)
    threads_out, allpass = [], True
    for t in threads:
        rel = float((t['mg'] - t['ref']).abs().max() /
                    (t['ref'].abs().max() + 1e-30))
        allpass = allpass and rel < 1e-3
        threads_out.append({'tid': t['tid'], 'device': t['device'],
                            'ref_time': round(t['ref_time'], 4),
                            'mg_time': round(t['mg_time'], 4),
                            'rel_diff': round(rel, 2)})
    return {'label': label or f"L{'x'.join(map(str, lat))}",
            'lattice': [int(x) for x in lat], 'levels': num_levels,
            'restart': num_restart, 'ct': coarse_tol_factor,
            'cmi': coarse_max_iter, 'nv_iters': nv_iters,
            'nthreads': nthreads, 'device_ids': [int(d) for d in device_ids],
            'threads': threads_out,
            'multi_ref_wall': round(ref_max, 4),
            'multi_mg_wall': round(mg_max, 4),
            'speedup': round(ref_max / mg_max, 4),
            'consistency_pass': allpass,
            'total_solve_s': round(total_solve, 3),
            'rss_kb': rss_kb(), 'cache_mb': round(cache_disk_mb(), 1)}


# ----------------------------------------------------------------------
# verify —— 正确性验证
# ----------------------------------------------------------------------
def cmd_verify(args):
    from pyqcu.cuda._multi_gpu import MultiGpuMultigrid
    from pyqcu.testing import test_h5py_multithread
    lat = list(args.lattice)
    res = {'mode': 'consistency', 'lattice': lat}
    # 1) 一致性模式：2 线程 × P100×2 共享输入，各线程解一致且与 BiStabCG 相符
    m = measure_multi(lat, num_levels=2, nthreads=2, device_ids=P100_IDS,
                      num_restart=5, coarse_max_iter=15, coarse_tol_factor=1e5)
    res['consistency'] = m
    assert m['consistency_pass'], f"consistency FAIL: {m['threads']}"
    print(f"PASS consistency 2xP100 lat={lat} "
          f"rel_diffs={[t['rel_diff'] for t in m['threads']]}")
    # 2) 独立问题模式：2 线程 × P100×2 各自不同 seed，解应不同且各自收敛
    m2 = measure_multi([4, 4, 4, 8], num_levels=2, nthreads=2,
                       device_ids=P100_IDS, independent=True, verbose=False)
    res['independent'] = m2
    sols = []
    mg = MultiGpuMultigrid(lat_size=[4, 4, 4, 8], mass=MASS, atol=ATOL,
                           num_levels=2, dof_list=DOF_LIST, mg_grid=MG_GRID,
                           num_restart=5, coarse_max_iter=15,
                           coarse_tol_factor=1e5, nv_iters=NV_ITERS,
                           nthreads=2, device_ids=list(P100_IDS),
                           use_cache=True, cache_dir=cache_dir(),
                           verbose=False, independent_problems=True)
    r2 = mg.solve()
    for t in r2['threads']:
        d = float((t['mg'] - t['ref']).abs().max())
        rel = d / (t['ref'].abs().max() + 1e-30)
        assert rel < 1e-3, f"independent tid={t['tid']} MG vs ref rel {rel}"
        sols.append(t['mg'])
    diff = float((sols[0] - sols[1]).abs().max())
    assert diff > 1e-4, f"independent problems gave identical solutions"
    res['independent_ok'] = True
    print(f"PASS independent 2xP100 (solutions differ, |d|={diff:.2e})")
    # 3) V100 单线程：大格子正确性
    m3 = measure_multi([8, 16, 16, 16], num_levels=3, nthreads=1,
                       device_ids=V100_ID, num_restart=10,
                       coarse_max_iter=15, coarse_tol_factor=1e5)
    res['v100_single'] = m3
    assert m3['consistency_pass'], f"V100 single FAIL: {m3['threads']}"
    print(f"PASS V100 single 8x16x16x16 3L rel_diff="
          f"{[t['rel_diff'] for t in m3['threads']]}")
    # 4) h5py 多线程读写验证（独立线程并发写/读 + 共享文件并发读）
    tmp = os.path.join(WORKDIR, "h5io_tmp")
    test_h5py_multithread(nthreads=args.h5threads, tmp_dir=tmp)
    res['h5io_threads'] = args.h5threads
    print(f"PASS h5py multithread IO ({args.h5threads} threads)")
    out = os.path.join(WORKDIR, "dev78_2_verify.h5")
    save_dict_h5(out, res)
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# clean —— 干净测量（独立进程）
# ----------------------------------------------------------------------
def cmd_clean(args):
    lat = list(args.lattice)
    devices = list(args.devices) if args.devices else (
        list(P100_IDS) if args.nthreads >= 2 else list(V100_ID))
    res = measure_multi(lat, num_levels=args.levels, num_restart=args.restart,
                        coarse_max_iter=args.cmi, coarse_tol_factor=args.ct,
                        nthreads=args.nthreads, device_ids=devices,
                        verbose=False)
    label = (f"L{'x'.join(map(str, lat))}_L{args.levels}_r{args.restart}"
             f"_ct{args.ct:.0e}_cmi{args.cmi}_mt{args.nthreads}")
    res['label'] = label
    out = os.path.join(WORKDIR, f"dev78_2_clean_{label}.h5")
    save_dict_h5(out, res)
    print(f"wrote {out}", flush=True)
    print(f"RESULT {label}: multi_ref={res['multi_ref_wall']}s "
          f"multi_mg={res['multi_mg_wall']}s speedup={res['speedup']} "
          f"consistency={res['consistency_pass']}", flush=True)


# ----------------------------------------------------------------------
# bench —— 批量基准
# ----------------------------------------------------------------------
def bench_one(label, lat, num_levels, num_restart, coarse_max_iter,
              coarse_tol_factor, nthreads, device_ids, pairs=3, verbose=False):
    times = []
    for _ in range(pairs):
        times.append(measure_multi(lat, num_levels=num_levels,
                                   num_restart=num_restart,
                                   coarse_max_iter=coarse_max_iter,
                                   coarse_tol_factor=coarse_tol_factor,
                                   nthreads=nthreads, device_ids=device_ids,
                                   verbose=verbose))
    refs = sorted(t['multi_ref_wall'] for t in times)
    mgs = sorted(t['multi_mg_wall'] for t in times)
    return {'label': label, 'lattice': [int(x) for x in lat],
            'levels': num_levels, 'restart': num_restart,
            'cmi': coarse_max_iter, 'ct': coarse_tol_factor,
            'nthreads': nthreads, 'device_ids': [int(d) for d in device_ids],
            'pairs': pairs,
            'ref_min': refs[0], 'ref_med': refs[len(refs) // 2], 'ref_max': refs[-1],
            'mg_min': mgs[0], 'mg_med': mgs[len(mgs) // 2], 'mg_max': mgs[-1],
            'speedup_med': round(refs[len(refs) // 2] /
                                 mgs[len(mgs) // 2], 4),
            'speedup_max': round(refs[0] / mgs[0], 4),
            'consistency_pass': all(t['consistency_pass'] for t in times),
            'rss_kb': max(t['rss_kb'] for t in times),
            'cache_mb': times[0]['cache_mb']}


def _bench_configs(args):
    """bench 配置表：P100×2 多线程组 + V100 单线程大格子组。

    2026-08-16 dev78：中/大格子 2L 用 num_restart=10（默认 5 时 V-cycle 频率
    过高、粗层求解开销主导：8x16x16x16 2L r5=0.33-0.67 → r10=1.05-1.22）。
    """
    cfgs = []
    # 多线程（P100×2）：小/中格子
    for lat, levels, r, cmi, ct in [
        ([8, 8, 8, 16], 2, 5, 15, 1e5),
        ([8, 8, 8, 16], 3, 5, 15, 1e5),
        ([8, 16, 16, 16], 2, 10, 15, 1e5),
        ([8, 16, 16, 16], 3, 10, 15, 1e5),
        ([16, 16, 16, 16], 2, 20, 15, 1e5),
    ]:
        cfgs.append((f"P100x2 L{'x'.join(map(str, lat))} {levels}L",
                     lat, levels, r, cmi, ct, 2, list(P100_IDS)))
    # 单线程（V100）：中格子（16x16x16x32 求解偏慢且无完整缓存，测试时长受限，移除）
    for lat, levels, r, cmi, ct in [
        ([16, 16, 16, 16], 3, 10, 15, 1e5),
        ([8, 16, 16, 16], 3, 10, 15, 1e5),
    ]:
        cfgs.append((f"V100 L{'x'.join(map(str, lat))} {levels}L",
                     lat, levels, r, cmi, ct, 1, list(V100_ID)))
    return cfgs


def cmd_bench(args):
    outs = []
    for label, lat, levels, r, cmi, ct, nt, devs in _bench_configs(args):
        if args.only and not any(o in label for o in args.only):
            continue
        print(f"bench {label} ...", flush=True)
        t0 = time.perf_counter()
        outs.append(bench_one(label, lat, levels, r, cmi, ct, nt, devs,
                              pairs=args.pairs))
        print(f"  done {time.perf_counter() - t0:.1f}s "
              f"speedup={outs[-1]['speedup_med']}", flush=True)
    out = os.path.join(WORKDIR, "dev78_2_bench.h5")
    save_dict_h5(out, {'entries': [{'e%d' % i: e} for i, e in enumerate(outs)]})
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# sweep —— 参数扫描（P100×2 多线程）
# ----------------------------------------------------------------------
def cmd_sweep(args):
    lat = list(args.lattice)
    # r3 实测全崩（0.03-0.11：粗层 BiStabCG restart<5 收敛极差，历史特性），
    # 从参数空间排除；r5/r10 为有效区间。
    rs = [5, 10]
    cts = [1e4, 1e5]
    cmis = [10, 15]
    levels_l = [2, 3]
    entries = []
    for levels in levels_l:
        for r in rs:
            for ct in cts:
                for cmi in cmis:
                    t0 = time.perf_counter()
                    m = measure_multi(lat, num_levels=levels,
                                      num_restart=r,
                                      coarse_max_iter=cmi,
                                      coarse_tol_factor=ct,
                                      nthreads=2, device_ids=list(P100_IDS),
                                      verbose=False)
                    entries.append({'levels': levels, 'restart': r, 'ct': ct,
                                    'cmi': cmi,
                                    'speedup': m['speedup'],
                                    'ref_wall': m['multi_ref_wall'],
                                    'mg_wall': m['multi_mg_wall'],
                                    'consistency': m['consistency_pass'],
                                    't': round(time.perf_counter() - t0, 2)})
                    print(f"sweep L{levels} r{r} ct{ct:.0e} cmi{cmi}: "
                          f"speedup={m['speedup']} "
                          f"{'OK' if m['consistency_pass'] else 'FAIL'}",
                          flush=True)
    out = os.path.join(WORKDIR, "dev78_2_sweep.h5")
    save_dict_h5(out, {'lattice': [int(x) for x in lat],
                       'entries': [{'e%d' % i: e} for i, e in enumerate(entries)]})
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# check —— 加速比断言
# ----------------------------------------------------------------------
def cmd_check(args):
    d = load_dict_h5(os.path.join(WORKDIR, args.file))
    entries = _entries_list(d)
    best = max(entries, key=lambda e: e['speedup'])
    gate = args.gate
    ok = [e for e in entries if e['speedup'] >= gate]
    print(f"check: gate={gate} entries={len(entries)} "
          f"pass={len(ok)} best={best['speedup']} "
          f"(L{best['levels']} r{best['restart']} ct{best['ct']:.0e} "
          f"cmi{best['cmi']})")
    if len(ok) < len(entries) * 0.5:
        print(f"FAIL: only {len(ok)}/{len(entries)} configs >= {gate}")
        sys.exit(1)
    print(f"PASS: {len(ok)}/{len(entries)} configs >= {gate}")
    sys.exit(0)


# ----------------------------------------------------------------------
# budget —— 显存/内存预算模型
# ----------------------------------------------------------------------
# 模型常数（test12 实测校准）：cold 全流程 / warm 求解阶段
ALPHA_DEFAULT = 0.0528   # GB/V cold（含粗算子构建峰值）
ALPHA_WARM = 0.0269      # GB/V warm（仅求解）
BETA_MB = 580            # 固定开销（MB）


def _vol(Lx, Ly, Lz, Lt):
    return Lx * Ly * Lz * Lt


def cmd_budget(args):
    vram = args.vram
    rows = []
    for lat, levels in args.lattices:
        V = _vol(*lat)
        cold = ALPHA_DEFAULT * V / 1024 + BETA_MB / 1024
        warm = ALPHA_WARM * V / 1024 + BETA_MB / 1024
        rows.append({'lattice': [int(x) for x in lat], 'levels': levels,
                     'vol': V, 'cold_gb': round(cold, 2),
                     'warm_gb': round(warm, 2),
                     'cold_pct': round(100 * cold / vram, 1),
                     'warm_pct': round(100 * warm / vram, 1),
                     'fit': cold < 0.9 * vram})
    out = os.path.join(WORKDIR, f"dev78_2_budget_{vram}g.h5")
    save_dict_h5(out, {'vram_gb': vram,
                       'rows': [{'r%d' % i: r} for i, r in enumerate(rows)]})
    for r in rows:
        print(f"{'x'.join(map(str, r['lattice']))} L{r['levels']}: "
              f"cold={r['cold_gb']}GB warm={r['warm_gb']}GB "
              f"{'OK' if r['fit'] else 'OVER'}")
    print(f"wrote {out}")


# ----------------------------------------------------------------------
# conv —— 全求解器迭代残差历史收集（实测 MG + Python 复现参考 BiStabCG）
# ----------------------------------------------------------------------
# 残差数据源：
#   * MG（MultiGpuMultigrid，C++ 实测）：每次 applyCloverMultigridQcu 求解
#     无条件（非 verbose 依赖）把逐迭代残差数组写入 REPO/logs/clover_multigrid.log
#     （CONVERGENCE_HISTORY: [r0,r1,...]，lattice_clover_multigrid.h:1673）；
#     本子命令按文件偏移增量解析，与测量一一对应（多线程时每线程一条，均保留）。
#   * 参考 BiStabCG（Python 复现，dev73_5 已验证方案，仅用于画图）：
#     C++ stdout 只输出收敛点（PRINT_MULTI_GPU_CLOVER_BISTABCG 默认关闭，
#     lattice_clover_bistabcg.h:10），完整曲线用同一 Schur 算子 + 同一
#     BiStabCG 算法在 torch 上复现（bistabcg_history），数学等价、零 C++ 改动。
def _bistabcg_history(b, matvec, tol, max_iter=2000):
    """Python 复现 BiStabCG 逐迭代残差历史（返回 [||r0||, ||r1||, ...]）。"""
    import torch as _t
    from pyqcu import tools
    x = _t.zeros_like(b)
    r = b - matvec(x)
    r_norm = float(tools.norm(r))
    hist = [r_norm]
    if r_norm < tol:
        return hist
    r_tilde = r.clone()
    p = _t.zeros_like(b)
    v = _t.zeros_like(b)
    s = _t.zeros_like(b)
    t = _t.zeros_like(b)
    rho = _t.tensor(1.0, dtype=b.dtype, device=b.device)
    rho_prev = _t.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = _t.tensor(1.0, dtype=b.dtype, device=b.device)
    omega = _t.tensor(1.0, dtype=b.dtype, device=b.device)
    for _ in range(max_iter):
        rho = tools.vdot(r_tilde, r)
        if abs(rho) < 1e-30:
            break
        beta = (rho / rho_prev) * (alpha / omega)
        rho_prev = rho
        p = r + beta * (p - omega * v)
        v = matvec(p)
        rtv = tools.vdot(r_tilde, v)
        if abs(rtv) < 1e-30:
            break
        alpha = rho / rtv
        s = r - alpha * v
        t = matvec(s)
        tts = tools.vdot(t, t)
        if abs(tts) < 1e-30:
            break
        omega = tools.vdot(t, s) / tts
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = float(tools.norm(r))
        hist.append(r_norm)
        if r_norm < tol:
            break
    return hist


def _ref_conv_history(op, src_full, tol):
    """在奇偶 Schur 系统上复现参考 BiStabCG，返回逐迭代残差历史。"""
    from pyqcu import tools
    ls_odd = list(src_full.shape[2:])            # [X,Y,Z,T] → [X,Y,Z,T//2]
    ls_odd[3] //= 2
    eo = tools.oooxyzt2poooxyzt(src_full)        # [2,4,3,X,Y,Z,T//2]
    b_e = eo[0].reshape([12] + ls_odd)
    b_o = eo[1].reshape([12] + ls_odd)
    b = op.give_b_parity(b_e, b_o)
    return _bistabcg_history(b, op.matvec_parity, tol)


def _conv_configs():
    """残差收集配置集（bench 7 + sweep 16 + verify 3，去重）。"""
    cfgs = []
    for label, lat, levels, r, cmi, ct, nt, devs in _bench_configs(None):
        cfgs.append({'label': label, 'lattice': [int(x) for x in lat],
                     'levels': levels, 'restart': r, 'ct': ct, 'cmi': cmi,
                     'nthreads': nt, 'device_ids': [int(d) for d in devs]})
    for levels in (2, 3):
        for r in (5, 10):
            for ct in (1e4, 1e5):
                for cmi in (10, 15):
                    cfgs.append({'label': f"8x8x8x16 L{levels} r{r} ct{ct:.0e} cmi{cmi}",
                                 'lattice': [8, 8, 8, 16], 'levels': levels,
                                 'restart': r, 'ct': ct, 'cmi': cmi,
                                 'nthreads': 2, 'device_ids': list(P100_IDS)})
    cfgs.append({'label': 'verify 8x8x8x16 2L r5', 'lattice': [8, 8, 8, 16],
                 'levels': 2, 'restart': 5, 'ct': 1e5, 'cmi': 15,
                 'nthreads': 2, 'device_ids': list(P100_IDS)})
    cfgs.append({'label': 'verify 4x4x4x8 2L r5', 'lattice': [4, 4, 4, 8],
                 'levels': 2, 'restart': 5, 'ct': 1e5, 'cmi': 15,
                 'nthreads': 2, 'device_ids': list(P100_IDS)})
    cfgs.append({'label': 'verify 8x16x16x16 3L r10', 'lattice': [8, 16, 16, 16],
                 'levels': 3, 'restart': 10, 'ct': 1e5, 'cmi': 15,
                 'nthreads': 1, 'device_ids': list(V100_ID)})
    seen, uniq = set(), []
    for c in cfgs:
        k = (tuple(c['lattice']), c['levels'], c['restart'], c['ct'], c['cmi'])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq


def _parse_conv_histories(offset):
    """clover_multigrid.log 偏移 offset 之后的全部 CONVERGENCE_HISTORY → list[list[float]]。"""
    if not os.path.exists(LOG_PATH):
        return []
    import re
    with open(LOG_PATH) as f:
        f.seek(offset)
        data = f.read()
    out = []
    for line in data.splitlines():
        m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
        if m:
            out.append([float(x) for x in m.group(1).split(',') if x.strip()])
    return out


def cmd_conv(args):
    """逐配置收集迭代残差历史（实测 MG + Python 复现 ref）→ dev78_2_conv.h5。"""
    import torch
    from pyqcu import dslash
    from pyqcu.cuda import qcu, define
    from pyqcu.cuda.define import params as _mp, argv as _ma, set_ptrs as _ms
    from pyqcu.cuda._multi_gpu import MultiGpuMultigrid, _setup_gpu_tensors
    torch.cuda.set_device(0)                     # 主线程锁定 V100（torch 运算）
    ref_cache = {}
    entries = []
    for c in _conv_configs():
        lat = c['lattice']
        latk = tuple(lat)
        t0 = time.perf_counter()
        if latk not in ref_cache:
            # 主线程 V100 构建参考问题（规范场/Clover/源 + Schur 算子）
            pt, at, st = _mp.clone(), _ma.clone(), _ms.clone()
            tmp = MultiGpuMultigrid(lat_size=lat, mass=MASS, atol=ATOL,
                                    num_levels=c['levels'], dof_list=DOF_LIST,
                                    mg_grid=MG_GRID, num_restart=c['restart'],
                                    coarse_max_iter=c['cmi'],
                                    coarse_tol_factor=c['ct'],
                                    nv_iters=NV_ITERS, nthreads=1,
                                    device_ids=list(V100_ID),
                                    use_cache=True, cache_dir=cache_dir(),
                                    verbose=False)
            tmp._config_params(pt, at, st, at)
            g, fi, ce, cei, coo, coi, U_full, b_full, clover_full, kappa, av = \
                _setup_gpu_tensors(pt, at, st, torch.device('cuda:0'), MASS,
                                   ATOL, tmp.dt, verbose=False)
            op = dslash.operator(U=U_full, clover_term=clover_full,
                                 kappa=torch.Tensor([kappa]),
                                 support_parity=True, verbose=False)
            for _idx in (0, 1, 2):
                pt[define._SET_INDEX_] = _idx
                qcu.applyEndQcu(st, pt)
            ref_cache[latk] = (op, b_full)
        op, b_full = ref_cache[latk]
        ref_hist = _ref_conv_history(op, b_full, ATOL)
        # MG 实测（写 clover_multigrid.log CONVERGENCE_HISTORY，偏移增量解析）
        n0 = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0
        m = measure_multi(lat, num_levels=c['levels'], num_restart=c['restart'],
                          coarse_max_iter=c['cmi'], coarse_tol_factor=c['ct'],
                          nthreads=c['nthreads'], device_ids=c['device_ids'],
                          verbose=False)
        conv_mg = _parse_conv_histories(n0)
        entries.append({'label': c['label'], 'lattice': [int(x) for x in lat],
                        'levels': c['levels'], 'restart': c['restart'],
                        'ct': c['ct'], 'cmi': c['cmi'], 'nthreads': c['nthreads'],
                        'device_ids': [int(d) for d in c['device_ids']],
                        'ref_hist': ref_hist, 'conv_mg': conv_mg,
                        'speedup': m['speedup'],
                        'ref_wall': m['multi_ref_wall'], 'mg_wall': m['multi_mg_wall'],
                        'consistency': m['consistency_pass'],
                        't': round(time.perf_counter() - t0, 2)})
        print(f"conv {c['label']}: mg_curves={len(conv_mg)} "
              f"(len={[len(x) for x in conv_mg]}) ref_iters={len(ref_hist) - 1} "
              f"speedup={m['speedup']} ({entries[-1]['t']}s)", flush=True)
    out = os.path.join(WORKDIR, "dev78_2_conv.h5")
    save_dict_h5(out, {'entries': [{'e%d' % i: e} for i, e in enumerate(entries)]})
    print(f"wrote {out} ({len(entries)} configs)")


# ----------------------------------------------------------------------
# conv_plots —— 迭代残差图（全求解器：实测 MG + 复现 ref，逐配置/逐格子/汇总）
# ----------------------------------------------------------------------
def _get_entry_list(e, key):
    """load_dict_h5 读回值兼容：dataset 键 d_<key> / vlen object 数组 → list。"""
    import numpy as _np
    v = e.get(key, e.get('d_' + key, []))
    if isinstance(v, _np.ndarray) and v.dtype == _np.object_:
        return [_np.asarray(x).tolist() for x in v]
    if isinstance(v, _np.ndarray):
        return v.tolist()
    return v


def _lat_key(e):
    return tuple(e.get('lattice', e.get('d_lattice', [])))


def _cfg_str(e):
    return f"L{e['levels']} r{e['restart']} ct{e['ct']:.0e} cmi{e['cmi']}"


def cmd_conv_plots(args):
    """迭代残差图：逐配置 / 逐格子多参数对比 / 全配置汇总（对数坐标 + atol 线）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as _np
    p = os.path.join(WORKDIR, "dev78_2_conv.h5")
    if not os.path.exists(p):
        print(f"[warn] {p} not found — run 'conv' first")
        return
    d = load_dict_h5(p)
    entries = _entries_list(d)
    if not entries:
        print("[warn] no conv entries")
        return
    for e in entries:
        e['ref_hist'] = _get_entry_list(e, 'ref_hist')
        e['conv_mg'] = _get_entry_list(e, 'conv_mg')
    atol = ATOL
    def _plot_curves(ax, e):
        # 实测 MG（每线程一条，蓝）+ 复现 ref（绿）
        for k, cm in enumerate(e['conv_mg']):
            ax.plot(range(len(cm)), cm, 'o-', ms=2.5, lw=1.2,
                    color='steelblue', alpha=0.85,
                    label=('MG (实测)' if k == 0 else f'MG t{k+1}'))
        rh = e['ref_hist']
        if len(rh) > 1:
            ax.plot(range(len(rh)), rh, 's-', ms=2.5, lw=1.2,
                    color='seagreen', label='BiStabCG (复现 ref)')
        ax.set_yscale('log')
        ax.axhline(atol, color='gray', ls='--', lw=0.8)
        ax.text(0.99, atol, f'atol={atol:.0e}', fontsize=7, color='gray',
                ha='right', va='bottom', transform=ax.get_yaxis_transform())
        ax.set_xlabel('iteration')
        ax.set_ylabel('Schur residual ||r||')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='upper right', frameon=False)
    # 1) 逐配置
    for e in entries:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        _plot_curves(ax, e)
        lat_s = 'x'.join(map(str, _lat_key(e)))
        ax.set_title(f"dev78_2 conv: {lat_s} {_cfg_str(e)} "
                     f"(speedup={e['speedup']})", fontsize=10)
        fig.tight_layout()
        fname = (f"dev78_2_conv_{lat_s}_L{e['levels']}_r{e['restart']}"
                 f"_ct{e['ct']:.0e}_cmi{e['cmi']}.png")
        fig.savefig(os.path.join(WORKDIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"wrote {len(entries)} per-config conv plots")
    # 2) 逐格子逐 L 多参数对比（dev74_1_conv 风格：ref + 该组全部 MG 曲线）
    groups = {}
    for e in entries:
        groups.setdefault((_lat_key(e), e['levels']), []).append(e)
    for (lat, lv), rs in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        rh = next((r['ref_hist'] for r in rs if len(r['ref_hist']) > 1), [])
        if len(rh) > 1:
            ax.plot(range(len(rh)), rh, color='seagreen', lw=2.0,
                    label='BiStabCG (复现 ref)', marker='o', ms=3, zorder=3)
        for e in rs:
            for k, cm in enumerate(e['conv_mg']):
                ax.plot(range(len(cm)), cm, lw=1.3,
                        label=_cfg_str(e), marker='s', ms=2.5)
                break                       # 同组线程曲线相同，只画第一条
        ax.set_yscale('log')
        ax.axhline(atol, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Schur residual ||r||')
        lat_s = 'x'.join(map(str, lat))
        ax.set_title(f"dev78_2 conv: {lat_s} {lv}L convergence "
                     f"(multi-config)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False, loc='best')
        fig.tight_layout()
        fname = f"dev78_2_conv_{lat_s}_L{lv}.png"
        fig.savefig(os.path.join(WORKDIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"wrote {len(groups)} per-lattice group conv plots")
    # 3) 全配置汇总（按格子分 2x2 子图）
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    lat_all = sorted(set(_lat_key(e) for e in entries))
    for ax, lat in zip(axs.ravel(), lat_all):
        rs = [e for e in entries if _lat_key(e) == lat]
        rh = next((r['ref_hist'] for r in rs if len(r['ref_hist']) > 1), [])
        if len(rh) > 1:
            ax.plot(range(len(rh)), rh, color='seagreen', lw=1.6,
                    label='ref BiStabCG', alpha=0.8)
        for e in rs:
            for k, cm in enumerate(e['conv_mg']):
                ax.plot(range(len(cm)), cm, lw=1.0, alpha=0.7)
        ax.set_yscale('log')
        ax.axhline(atol, color='gray', ls='--', lw=0.7)
        ax.set_title('x'.join(map(str, lat)), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc='best')
        ax.set_xlabel('iteration', fontsize=8)
        ax.set_ylabel('Schur residual', fontsize=8)
    fig.suptitle(f'dev78_2: all-config iteration residual summary '
                 f'(MG measured, {len(entries)} configs)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(WORKDIR, "dev78_2_conv_all.png"), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"wrote dev78_2_conv_all.png ({len(lat_all)} lattice groups)")


# ----------------------------------------------------------------------
# collect —— 汇总
# ----------------------------------------------------------------------
def _load_clean_entries():
    entries = []
    for f in sorted(glob.glob(os.path.join(WORKDIR, "dev78_2_clean_*.h5"))):
        try:
            entries.append(load_dict_h5(f))
        except Exception as e:
            print(f"[warn] skip {f}: {e}")
    return entries


def _entries_list(d):
    """save_dict_h5 的 entries 组还原：{'entries': [{'e0': e}, ...]} →
    展开为 [e, ...]；兼容旧 dict 格式 {'e0': e, ...}。"""
    entries = d['entries']
    if isinstance(entries, dict):
        return [entries[k] for k in sorted(entries)]
    out = []
    for x in entries:
        if isinstance(x, dict):
            out.extend(v for v in x.values())
        else:
            out.append(x)
    return out


def _bench_entries():
    p = os.path.join(WORKDIR, "dev78_2_bench.h5")
    if not os.path.exists(p):
        return []
    d = load_dict_h5(p)
    return _entries_list(d)


def _sweep_entries():
    p = os.path.join(WORKDIR, "dev78_2_sweep.h5")
    if not os.path.exists(p):
        return []
    d = load_dict_h5(p)
    return _entries_list(d)


def cmd_collect(args):
    clean = _load_clean_entries()
    bench = _bench_entries()
    sweep = _sweep_entries()
    env = dump_env_h5()
    results = {'env': env,
               'clean': [{'c%d' % i: c} for i, c in enumerate(clean)],
               'bench': [{'b%d' % i: b} for i, b in enumerate(bench)],
               'sweep': [{'s%d' % i: s} for i, s in enumerate(sweep)]}
    if bench:
        bmed = sorted(b['speedup_med'] for b in bench)
        results['bench_speedup_med'] = bmed[len(bmed) // 2]
        results['bench_speedup_max'] = max(b['speedup_med'] for b in bench)
    if sweep:
        smax = max(s['speedup'] for s in sweep)
        results['sweep_speedup_max'] = smax
    out = os.path.join(WORKDIR, "dev78_2_results.h5")
    save_dict_h5(out, results)
    print(f"wrote {out}: clean={len(clean)} bench={len(bench)} "
          f"sweep={len(sweep)}")
    if bench:
        print(f"bench speedup median={results['bench_speedup_med']} "
              f"max={results['bench_speedup_max']}")
    if sweep:
        print(f"sweep speedup max={results['sweep_speedup_max']}")


# ----------------------------------------------------------------------
# mktable —— LaTeX 表
# ----------------------------------------------------------------------
def _esc(s):
    return str(s).replace('_', '\\_').replace('%', '\\%')


def _lat_str(e):
    # dataset 还原的 key 是 d_lattice（list 存为 dataset），兼容两种写法
    lat = e.get('lattice', e.get('d_lattice', []))
    return 'x'.join(map(str, lat))


def cmd_mktable(args):
    bench = _bench_entries()
    if not bench:
        print("[warn] no bench entries"); return
    lines = ["\\begin{tabular}{l|rrrrrrr}",
             "\\hline", "配置 & LxLyLzLt & L & nT & ref(s) & mg(s) & speedup \\\\",
             "\\hline"]
    for b in sorted(bench, key=lambda x: -x['speedup_med']):
        lat = _lat_str(b)
        lines.append(f"{_esc(b['label'])} & {lat} & {b['levels']} & "
                     f"{b['nthreads']} & {b['ref_med']:.3f} & "
                     f"{b['mg_med']:.3f} & {b['speedup_med']:.2f} \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    out = os.path.join(WORKDIR, "dev78_2_tbl_bench.tex")
    with open(out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")
    # sweep 表
    sweep = _sweep_entries()
    if sweep:
        lines = ["\\begin{tabular}{l|rrrrr}",
                 "\\hline", "levels & r & ct & cmi & speedup \\\\", "\\hline"]
        for s in sorted(sweep, key=lambda x: -x['speedup'])[:12]:
            lines.append(f"{s['levels']} & {s['restart']} & "
                         f"{s['ct']:.0e} & {s['cmi']} & {s['speedup']:.2f} \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        out = os.path.join(WORKDIR, "dev78_2_tbl_sweep.tex")
        with open(out, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {out}")


# ----------------------------------------------------------------------
# plots —— PNG 图
# ----------------------------------------------------------------------
def _style(ax):
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)


def _fig_save(fig, name, dpi=150):
    out = os.path.join(WORKDIR, name)
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    print(f"wrote {out}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def cmd_plots(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bench = _bench_entries()
    if bench:
        # 加速比横条图
        fig, ax = plt.subplots(figsize=(9, 5))
        bs = sorted(bench, key=lambda x: x['speedup_med'])
        labels = [f"{_lat_str(b)} {b['levels']}L "
                  f"mt{b['nthreads']}" for b in bs]
        vals = [b['speedup_med'] for b in bs]
        ax.barh(labels, vals, color='steelblue')
        ax.set_xlabel('speedup (multi-thread BiStabCG / multi-thread MG)')
        ax.set_title('dev78_2: Multi-thread MG speedup (P100x2 / V100)')
        ax.axvline(1.0, color='gray', ls='--', lw=0.8)
        _style(ax)
        _fig_save(fig, "dev78_2_speedup.png")
        # 时间对比
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(bs))
        ax.bar(x, [b['ref_med'] for b in bs], width=0.4,
               label='multi-ref (BiStabCG)', color='coral')
        ax.bar([i + 0.4 for i in x], [b['mg_med'] for b in bs], width=0.4,
               label='multi-mg', color='steelblue')
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('wall time (s, median)')
        ax.legend()
        _style(ax)
        _fig_save(fig, "dev78_2_time.png")
    sweep = _sweep_entries()
    if sweep:
        # 参数扫描：restart × speedup 曲线（按 levels 分组）
        fig, ax = plt.subplots(figsize=(9, 5))
        for lv in sorted(set(s['levels'] for s in sweep)):
            ss = [s for s in sweep if s['levels'] == lv]
            xs = sorted(set(s['restart'] for s in ss))
            for ct in sorted(set(s['ct'] for s in ss)):
                ys = [max([s['speedup'] for s in ss
                           if s['restart'] == x and s['ct'] == ct] or [0])
                      for x in xs]
                ax.plot(xs, ys, 'o-', label=f'{lv}L ct={ct:.0e}')
        ax.set_xlabel('num_restart')
        ax.set_ylabel('speedup')
        ax.legend(fontsize=8)
        _style(ax)
        _fig_save(fig, "dev78_2_sweep.png")
        # 热点：各参数对 speedup 的影响箱线
        fig, ax = plt.subplots(figsize=(9, 5))
        keys = [('restart', 'num_restart'), ('cmi', 'coarse_max_iter')]
        for i, (k, name) in enumerate(keys):
            vs = sorted(set(s[k] for s in sweep))
            data = [[s['speedup'] for s in sweep if s[k] == v] for v in vs]
            bp = ax.boxplot(data, positions=[i * 10 + j for j in range(len(vs))],
                            widths=0.6, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            for j, v in enumerate(vs):
                ax.text(i * 10 + j, max(data[j]) * 1.02, str(v), ha='center',
                        fontsize=8)
        ax.set_xticks([])
        ax.set_ylabel('speedup')
        ax.set_title('dev78_2: parameter hotspot (sweep 8x8x8x16)')
        _style(ax)
        _fig_save(fig, "dev78_2_hotspot.png")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="dev78_2: multi-thread (1-thread-1-GPU) CUDA C++ "
                    "MultiGrid solver test suite (h5py-only persistence)")
    p.add_argument('--outdir', default=None, help='输出目录（默认 TEST78_2_OUTDIR / logs/dev78_2）')
    sub = p.add_subparsers(dest='cmd')

    def add_common(sp):
        sp.add_argument('--outdir', default=None)

    sp = sub.add_parser('verify')
    add_common(sp)
    sp.add_argument('--lattice', nargs=4, type=int, default=[8, 8, 8, 16])
    sp.add_argument('--h5threads', type=int, default=4)

    sp = sub.add_parser('clean')
    add_common(sp)
    sp.add_argument('--lattice', nargs=4, type=int, default=[8, 8, 8, 16])
    sp.add_argument('--levels', type=int, default=2)
    sp.add_argument('--restart', type=int, default=5)
    sp.add_argument('--ct', type=float, default=1e5)
    sp.add_argument('--cmi', type=int, default=15)
    sp.add_argument('--nthreads', type=int, default=2)
    sp.add_argument('--devices', nargs='+', type=int, default=None)

    sp = sub.add_parser('bench')
    add_common(sp)
    sp.add_argument('--pairs', type=int, default=3)
    sp.add_argument('--only', nargs='+', default=None)

    sp = sub.add_parser('sweep')
    add_common(sp)
    sp.add_argument('--lattice', nargs=4, type=int, default=[8, 8, 8, 16])

    sp = sub.add_parser('check')
    add_common(sp)
    sp.add_argument('--gate', type=float, default=1.5)
    sp.add_argument('--file', default='dev78_2_sweep.h5')

    sp = sub.add_parser('budget')
    add_common(sp)
    sp.add_argument('--vram', type=int, default=16)
    sp.add_argument('--lattices', nargs='+', default=[
        '8x8x8x16', '8x16x16x16', '16x16x16x16', '16x16x16x32', '16x16x16x64'])

    sub.add_parser('conv')
    sub.add_parser('conv_plots')
    sub.add_parser('collect')
    sub.add_parser('mktable')
    sub.add_parser('plots')

    args = p.parse_args()
    if args.cmd is None:
        p.print_help()
        sys.exit(0)
    global WORKDIR
    if getattr(args, 'outdir', None):
        WORKDIR = args.outdir
    os.makedirs(WORKDIR, exist_ok=True)
    dump_env_h5()
    # budget 需要把 'LxLyLzLt' 字符串解析回 (lat, levels) 对
    if args.cmd == 'budget':
        args.lattices = [(tuple(int(x) for x in l.split('x')), 2)
                         for l in args.lattices]
    {'verify': cmd_verify, 'clean': cmd_clean, 'bench': cmd_bench,
     'sweep': cmd_sweep, 'check': cmd_check, 'budget': cmd_budget,
     'conv': cmd_conv, 'conv_plots': cmd_conv_plots,
     'collect': cmd_collect, 'mktable': cmd_mktable, 'plots': cmd_plots,
     }[args.cmd](args)


if __name__ == '__main__':
    main()
