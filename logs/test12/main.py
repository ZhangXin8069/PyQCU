#!/usr/bin/env python3
"""test12 —— dev74* 整合测试套件（单文件 main.py），test11_1 的优化版。

背景：dev74（大格子 + 资源统计 + 多线程构建）与 dev74_1（服务器加速比 >1.5
验证）两套脚本整合为单一入口；工作目录默认为 logs/test12/，全部输出
（json/png/tex/收敛日志归档）落在该目录。

test12 相对 test11 的优化：输出目录可重定向 —— 每次运行由运行脚本创建
版本目录 logs/test12/v<YYYYMMDDHHMM>/ 并传入 --outdir（或环境变量
TEST12_OUTDIR），该运行的全部产物与运行日志落在版本目录内，互不覆盖；
跨环境（本地 4060 / 服务器 16G / 32G）的测试结果各自归档，便于横向比对。
每次调用会在输出目录写入 env.json（GPU/软件/git/cmdline 快照）。

子命令：
  clean        干净测量（独立进程交叉计时 + 资源统计）→ test12_clean_*.json
  bench        批量基准（--mode local|server，预算自动跳过超限配置）
  verify       正确性验证（gauge/解/null_vecs/CudaSchurOp 对照）
  sweep        参数扫描（r/ct/cmi/levels × speedup）→ test12_sweep.json
  check        加速比断言（--gate 1.5 默认，exit 0/1/2）
  budget       显存/内存/磁盘预算表（--vram 16|32，默认 16GB）
  collect      汇总 → test12_results.json
  mktable      LaTeX 表 → test12_tbl_*.tex
  plots        dev74 风格图 → test12_*.png
  plots1       dev74_1 风格图（作图范围与 dev73_5 一致）→ test12_1_*.png
  layout_test  C++ Schur 算子输入布局对照实验
  stencil_mt   多线程 stencil build 对照验证

用法（PyQCU 根下）：
  source ./env.sh && python logs/test12/main.py <subcommand> [options]

约定（与 dev73_5/dev74 一致）：mass=0.05, atol=1e-6, gauge_seed=42,
kappa=1/(2m+8), E=48, NV_ITERS=2, MG_GRID=[2,2,2,2]；参考求解器 =
applyCloverBistabCgQcu（VERBOSE=0）；nullvec 缓存默认共享
logs/nullvec_cache（PYQCU_NULLVEC_CACHE 可覆盖），避免跨版本重复构建粗算子。
"""
import os, sys, time, json, glob, subprocess, resource, argparse

# ----------------------------------------------------------------------
# 路径与公共工具
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))            # logs/test12
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

WORKDIR = _HERE
# C++ 端写死 "logs/clover_multigrid.log"（相对 REPO），Python 端必须读同一路径
LOG_PATH = os.path.join(REPO, "logs", "clover_multigrid.log")

import torch
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
from pyqcu.lattice import check_su3

# ----------------------------------------------------------------------
# 内联辅助实现（照抄自 examples/qcu/dev73* 与 conftest.schur.multigrid；
# main.py 自包含 —— 不 import 任何 dev73*/dev74* 模块）
# ----------------------------------------------------------------------
# 1) 33-tensor 粗算子 stencil（原 mg_stencil_build.py）
#    sit[E,E,Xc,Yc,Zc,Tc] 近邻 hop_nn[2,4,...] 对角 hop_diag[2,2,6,...]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # (d1,d2), d1<d2
SIGN = [1, -1]


def build_stencil(S, lonv, E, e, lat_fine_odd, lat_coarse_odd, dt, device):
    X, Y, Z, Th = lat_fine_odd
    Xc, Yc, Zc, Tc = lat_coarse_odd
    Nc = Xc * Yc * Zc * Tc
    sit = torch.zeros([E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_nn = torch.zeros([2, 4, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_diag = torch.zeros([2, 2, 6, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    str_Y = Yc * Zc * Tc
    str_Z = Zc * Tc
    dims = [Xc, Yc, Zc, Tc]
    t0 = time.perf_counter()
    for c_idx in range(Nc):
        cx = c_idx // str_Y; rem = c_idx % str_Y
        cy = rem // str_Z; rem %= str_Z
        cz = rem // Tc; ct = rem % Tc
        ccoords = [cx, cy, cz, ct]
        for ee in range(E):
            src_c = torch.zeros([E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
            src_c[ee, cx, cy, cz, ct] = 1.0
            f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=src_c)
            dc = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S(f))
            sit[:, ee, cx, cy, cz, ct] = dc[:, cx, cy, cz, ct]
            for d in range(4):
                b = ccoords[:]; b[d] = (b[d] - 1 + dims[d]) % dims[d]
                fwd = ccoords[:]; fwd[d] = (fwd[d] + 1) % dims[d]
                if b[d] == fwd[d]:
                    hop_nn[0, d, :, ee, b[0], b[1], b[2], b[3]] = 0.5 * dc[:, b[0], b[1], b[2], b[3]]
                    hop_nn[1, d, :, ee, fwd[0], fwd[1], fwd[2], fwd[3]] = 0.5 * dc[:, fwd[0], fwd[1], fwd[2], fwd[3]]
                else:
                    hop_nn[0, d, :, ee, b[0], b[1], b[2], b[3]] = dc[:, b[0], b[1], b[2], b[3]]
                    hop_nn[1, d, :, ee, fwd[0], fwd[1], fwd[2], fwd[3]] = dc[:, fwd[0], fwd[1], fwd[2], fwd[3]]
            for pi, (d1, d2) in enumerate(PAIRS):
                targets = {}
                for s1i, s1 in enumerate(SIGN):
                    for s2i, s2 in enumerate(SIGN):
                        n = ccoords[:]
                        n[d1] = (n[d1] - s1 + dims[d1]) % dims[d1]
                        n[d2] = (n[d2] - s2 + dims[d2]) % dims[d2]
                        key = (n[0], n[1], n[2], n[3])
                        targets.setdefault(key, []).append((s1i, s2i))
                for key, combos in targets.items():
                    w = 1.0 / len(combos)
                    for (s1i, s2i) in combos:
                        hop_diag[s1i, s2i, pi, :, ee, key[0], key[1], key[2], key[3]] = w * dc[:, key[0], key[1], key[2], key[3]]
        if (c_idx + 1) % 64 == 0 and c_idx > 0:
            print(f"    probing {c_idx + 1}/{Nc} ({time.perf_counter() - t0:.1f}s)")
    print(f"  stencil build: {time.perf_counter() - t0:.1f}s for {E * Nc} probes")
    return hop_nn, hop_diag, sit


def apply_stencil(hop_nn, hop_diag, sit, v_c):
    E = v_c.shape[0]
    Xc, Yc, Zc, Tc = v_c.shape[1:]
    out = torch.einsum("EeXYZT,eXYZT->EXYZT", sit, v_c).clone()
    for d in range(4):
        fwd = torch.roll(v_c, shifts=-1, dims=d + 1)
        bwd = torch.roll(v_c, shifts=1, dims=d + 1)
        out += torch.einsum("EeXYZT,eXYZT->EXYZT", hop_nn[0, d], fwd)
        out += torch.einsum("EeXYZT,eXYZT->EXYZT", hop_nn[1, d], bwd)
    for pi, (d1, d2) in enumerate(PAIRS):
        for s1i, s1 in enumerate(SIGN):
            for s2i, s2 in enumerate(SIGN):
                shift = [0, 0, 0, 0]
                shift[d1] = -s1
                shift[d2] = -s2
                v_shift = torch.roll(v_c, shifts=tuple(shift), dims=(1, 2, 3, 4))
                out += torch.einsum("EeXYZT,eXYZT->EXYZT", hop_diag[s1i, s2i, pi], v_shift)
    return out


# 2) nullvec 粗算子缓存（原 mg_nullvec_cache.py；PYQCU_NULLVEC_CACHE 可覆盖）
DEFAULT_CACHE_DIR = os.environ.get("PYQCU_NULLVEC_CACHE",
                                   os.path.join(REPO, "logs", "nullvec_cache"))
_KEYS = ["lonv", "hnn", "hdg", "sit"]


def cache_dir():
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    return DEFAULT_CACHE_DIR


def cache_tag(gauge_seed, lat_full, level, E, nv_iters, dt_name="c64"):
    L = "x".join(str(x) for x in lat_full)
    return f"L{L}_lv{level}_E{E}_nvi{nv_iters}_{dt_name}"


def load_coarse_ops(gauge_seed, lat_full, level, E, nv_iters, dt_name="c64",
                    device=torch.device("cuda")):
    d = cache_dir()
    tags = [cache_tag(gauge_seed, lat_full, level, E, nv_iters, dt_name)]
    if dt_name == "c64":
        tags.append(cache_tag(gauge_seed, lat_full, level, E, nv_iters, "").rstrip("_"))
    for tag in tags:
        if all(os.path.exists(os.path.join(d, tag + "_" + k + ".pt")) for k in _KEYS):
            return [torch.load(os.path.join(d, tag + "_" + k + ".pt"),
                               map_location=device) for k in _KEYS]
    return None


def save_coarse_ops(gauge_seed, lat_full, level, E, nv_iters, dt_name, lonv,
                    hnn, hdg, sit):
    tag = cache_tag(gauge_seed, lat_full, level, E, nv_iters, dt_name)
    d = cache_dir()
    for k, t in zip(_KEYS, [lonv, hnn, hdg, sit]):
        torch.save(t.detach().cpu(), os.path.join(d, tag + "_" + k + ".pt"))


def build_or_load_coarse_ops(gauge_seed, lat_full, level, E, E_prev,
                             lat_fine, lat_coarse, S, dt, device,
                             nv_iters=2, use_cache=True, save=True,
                             verbose=True):
    _real = dt.to_real() if hasattr(dt, "to_real") else dt
    dt_name = {torch.float32: "c64", torch.float64: "c128"}[_real]
    if use_cache:
        cached = load_coarse_ops(gauge_seed, lat_full, level, E, nv_iters,
                                 dt_name, device)
        if cached is not None:
            if verbose:
                print(f"  [level {level}] E={E} CACHED coarse={lat_coarse}")
            return cached
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    _null = torch.randn([E, E_prev] + lat_fine, dtype=dt, device=device)
    for _ in range(nv_iters):
        _null = tools.give_null_vecs(null_vecs=_null, matvec=S,
                                     bistabcg=None, verbose=False)
    lonv = tools.local_orthogonalize(null_vecs=_null,
                                     coarse_lat_size=lat_coarse, verbose=False)
    t0.record()
    hnn, hdg, sit = build_stencil(S, lonv, E, E_prev, lat_fine, lat_coarse,
                                  dt, device)
    t1.record()
    torch.cuda.synchronize()
    if verbose:
        print(f"  [level {level}] E={E} built nv+stencil in "
              f"{t0.elapsed_time(t1) / 1000:.1f}s coarse={lat_coarse}")
    if save:
        save_coarse_ops(gauge_seed, lat_full, level, E, nv_iters, dt_name,
                        lonv, hnn, hdg, sit)
    return lonv, hnn, hdg, sit


# 3) C++ 参数协议 build_config（原 conftest.schur.multigrid.py）
def build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
                 NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR,
                 DT=define._LAT_C64_):
    params[define._LAT_X_] = Lx; params[define._LAT_Y_] = Ly
    params[define._LAT_Z_] = Lz; params[define._LAT_T_] = Lt
    params[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[define._GRID_T_] = tools.give_grid_size()
    params[define._PARITY_] = 0; params[define._NODE_RANK_] = 0; params[define._NODE_SIZE_] = 1
    params[define._DAGGER_] = 0; params[define._MAX_ITER_] = 1000
    params[define._DATA_TYPE_] = DT
    params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0; params[define._SEED_] = 42; params[define._TEST_IN_CPU_] = 0
    params[define._MG_NUM_LEVEL_] = NUM_LEVELS
    if NUM_LEVELS >= 2:
        params[define._MG_LEVEL1_E_] = DOF_LIST[1]
        params[define._MG_LEVEL1_X_] = Lx // MG_GRID[0]
        params[define._MG_LEVEL1_Y_] = Ly // MG_GRID[1]
        params[define._MG_LEVEL1_Z_] = Lz // MG_GRID[2]
        params[define._MG_LEVEL1_T_] = Lt // (2 * MG_GRID[3])
        params[define._MG_LEVEL1_MAX_ITER_] = COARSE_MAX_ITER
        params[define._MG_LEVEL1_DATA_TYPE_] = DT
        params[define._MG_LEVEL1_NUM_RESTART_] = NUM_RESTART
    if NUM_LEVELS >= 3:
        params[define._MG_LEVEL2_E_] = DOF_LIST[2]
        params[define._MG_LEVEL2_X_] = Lx // (MG_GRID[0] * MG_GRID[0])
        params[define._MG_LEVEL2_Y_] = Ly // (MG_GRID[1] * MG_GRID[1])
        params[define._MG_LEVEL2_Z_] = Lz // (MG_GRID[2] * MG_GRID[2])
        params[define._MG_LEVEL2_T_] = Lt // (4 * MG_GRID[3])
        params[define._MG_LEVEL2_MAX_ITER_] = 200
        params[define._MG_LEVEL2_DATA_TYPE_] = DT
        params[define._MG_LEVEL2_NUM_RESTART_] = 3
    av = argv.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_] = MASS; av[define._ATOL_] = ATOL; av[define._SIGMA_] = 0.1
    if NUM_LEVELS >= 2:
        av[define._MG_LEVEL1_ATOL_] = ATOL * COARSE_TOL_FACTOR
    if NUM_LEVELS >= 3:
        av[define._MG_LEVEL2_ATOL_] = ATOL * COARSE_TOL_FACTOR
    return av


# 4) 参考 BiStabCG 收敛历史 + C++ 日志解析（原 mg_dev73_5_bench.py）
def parse_mg_log(path=LOG_PATH):
    """返回 (conv_history, prof_sections, total_iterations)。"""
    import re
    conv, prof, n_iter = [], None, None
    if os.path.exists(path):
        for line in open(path):
            m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
            if m:
                conv = [float(x) for x in m.group(1).split(",") if x.strip()]
            m = re.search(r'PROF_SECTIONS:\s*(.*)', line)
            if m:
                prof = m.group(1).strip()
            m = re.search(r'Total iterations:\s*(\d+)', line)
            if m:
                n_iter = int(m.group(1))
    prof_d = {}
    if prof:
        for tok in prof.split():
            if "=" in tok:
                k, v = tok.split("=")
                v = v.rstrip("ms")
                prof_d[k] = float(v)
    return conv, prof_d, n_iter


def bistabcg_history(b, matvec, tol, max_iter=2000):
    x = torch.zeros_like(b)
    r = b - matvec(x)
    r_norm = float(tools.norm(r))
    hist = [r_norm]
    if r_norm < tol:
        return x, hist
    r_tilde = r.clone()
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    s = torch.zeros_like(b)
    t = torch.zeros_like(b)
    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    rho_prev = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    for i in range(max_iter):
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
    return x, hist


def ref_conv_history(op, qcu_src, tol):
    """在奇偶 Schur 系统上复现参考 BiStabCG，返回逐迭代残差历史。"""
    ls_full = list(qcu_src.shape[2:])        # [X,Y,Z,T]
    ls_odd = ls_full.copy()
    ls_odd[3] //= 2
    eo = tools.oooxyzt2poooxyzt(qcu_src)     # [2,4,3,X,Y,Z,T//2]
    b_e = eo[0].reshape([12] + ls_odd)
    b_o = eo[1].reshape([12] + ls_odd)
    b = op.give_b_parity(b_e, b_o)
    _, hist = bistabcg_history(b, op.matvec_parity, tol)
    return hist


# 5) setup_gpu（原 mg_pyref_expt.py，layout_test / stencil_mt 用）
def setup_gpu(Lx, Ly, Lz, Lt, MASS, ATOL, DT=define._LAT_C64_):
    KAPPA = 1.0 / (2 * MASS + 8)
    params[define._LAT_X_] = Lx; params[define._LAT_Y_] = Ly
    params[define._LAT_Z_] = Lz; params[define._LAT_T_] = Lt
    params[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[define._GRID_T_] = tools.give_grid_size()
    params[define._PARITY_] = 0; params[define._NODE_RANK_] = 0; params[define._NODE_SIZE_] = 1
    params[define._DAGGER_] = 0; params[define._MAX_ITER_] = 1000
    params[define._DATA_TYPE_] = DT
    params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0; params[define._SEED_] = 42; params[define._TEST_IN_CPU_] = 0
    params[define._MG_NUM_LEVEL_] = 1
    av = argv.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_] = MASS; av[define._ATOL_] = ATOL; av[define._SIGMA_] = 0.1
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)
    g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
    fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
    ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce); coo = torch.zeros_like(ce); coi = torch.zeros_like(ce)
    params[define._SET_INDEX_] = 0; params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 2; params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1; params[define._SET_PLAN_] = 2; params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)
    U_full = tools.poooxyzt2oooxyzt(g)
    b_full = tools.poooxyzt2oooxyzt(fi)
    clover = dslash.make_clover(U_full, kappa=KAPPA)
    return U_full, b_full, clover, KAPPA, av, (g, fi, ce, coo, cei, coi)


# 6) verify 辅助（原 mg_dev73_5_verify.py）
def build_base(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
               DT, gauge_seed=42):
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
                      MG_GRID, 10, 15, 1e5, DT)
    KAPPA = 1.0 / (2 * MASS + 8)
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)
    torch.manual_seed(gauge_seed)
    g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
    fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi)
    ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce)
    coo = torch.zeros_like(ce)
    coi = torch.zeros_like(ce)

    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    torch.cuda.synchronize()
    qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs, params)
    torch.cuda.synchronize()

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    return dict(av=av, KAPPA=KAPPA, device=device, dt=dt, ls=ls, g=g, fi=fi,
                fo_ref=fo_ref, ce=ce, cei=cei, coo=coo, coi=coi,
                qcu_U=qcu_U, qcu_src=qcu_src, qcu_ref=qcu_ref, ref_cl=ref_cl,
                op=op, S=op.matvec_parity)


def verify_lattice(qcu_U):
    """check_su3：幺正性 / det=1 / minor 恒等式 + 量化指标。"""
    t0 = time.perf_counter()
    ok = check_su3(qcu_U, tol=1e-2 if qcu_U.dtype == torch.float32 else 1e-3,
                   verbose=False)
    dt = time.perf_counter() - t0
    U = qcu_U  # [c_in, c_out, dir, X,Y,Z,T]
    I3 = torch.eye(3, dtype=qcu_U.dtype, device=qcu_U.device)
    UH_U = torch.einsum('bam...,bcm...->acm...', U.conj(), U)
    unit = (UH_U - I3.view(3, 3, 1, 1, 1, 1, 1)).abs().max().item()
    dets = torch.linalg.det(U.permute(2, 3, 4, 5, 6, 0, 1))
    detdev = (dets - 1).abs().max().item()
    return {"check_su3": bool(ok), "max_unit_err": float(unit),
            "max_det_dev": float(detdev), "sec": dt}


def verify_nullvecs(op, S, lonv, hnn, hdg, sit, E, E_prev, lat_fine,
                    lat_coarse, dt, device, n_sample=4):
    """null_vecs 四重检查。lonv: [E, E_prev, Xf,Yf,Zf,Tf]。"""
    out = {}
    ratios = []
    for k in range(min(n_sample, E)):
        v = lonv[k]
        Av = S(v.reshape([E_prev] + lat_fine)).reshape(E, -1)
        ratios.append((torch.linalg.norm(Av) / torch.linalg.norm(lonv[k])).item())
    v = torch.randn([E_prev] + lat_fine, dtype=dt, device=device)
    v = v / torch.linalg.norm(v)
    _real_dt = torch.float32 if dt == torch.complex64 else torch.float64
    lam = torch.tensor(0.0, dtype=_real_dt, device=device)
    for _ in range(20):
        w = S(v).flatten()
        vf = v.flatten()
        lam = torch.real(torch.vdot(w, vf))
        v = w.reshape(v.shape) / torch.linalg.norm(w)
    out["null_ratios"] = ratios
    out["S_lambda_max"] = abs(float(lam))
    X, Y, Z, T = lat_coarse
    x, y, z, t = [lat_fine[d] // lat_coarse[d] for d in range(4)]
    vb = lonv.reshape(E, E_prev, X, x, Y, y, Z, z, T, t)
    block = vb[:, :, 0, :, 0, :, 0, :, 0, :].reshape(E, -1)
    G = block @ block.conj().T
    off = G - torch.eye(E, dtype=dt, device=device)
    out["ortho_offdiag_max"] = float(off.abs().max().item())
    out["ortho_diag_min"] = float(torch.diag(G).real.min().item())
    out["ortho_diag_max"] = float(torch.diag(G).real.max().item())
    out["restrict_rel_diff"] = None
    out["prolong_rel_diff"] = None
    if E_prev == 12:
        fine_vec = torch.randn([E_prev] + lat_fine, dtype=dt, device=device)
        r_py = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=fine_vec)
        params[define._LAT_X_] = lat_fine[0]
        params[define._LAT_Y_] = lat_fine[1]
        params[define._LAT_Z_] = lat_fine[2]
        params[define._LAT_T_] = lat_fine[3]
        params[define._MG_LEVEL1_X_] = X
        params[define._MG_LEVEL1_Y_] = Y
        params[define._MG_LEVEL1_Z_] = Z
        params[define._MG_LEVEL1_T_] = T
        params[define._MG_LEVEL1_E_] = E
        params[define._MG_NUM_LEVEL_] = 12
        out_r = torch.zeros([E, X, Y, Z, T], dtype=dt, device=device)
        qcu.applyMultigridRestrictQcu(out_r, fine_vec, lonv, set_ptrs, params)
        out["restrict_max_diff"] = float((out_r - r_py).abs().max().item())
        out["restrict_rel_diff"] = float((out_r - r_py).abs().max().item() /
                                         (r_py.abs().max().item() + 1e-30))
        coarse_vec = torch.randn([E, X, Y, Z, T], dtype=dt, device=device)
        p_py = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=coarse_vec)
        out_p = torch.zeros([E_prev] + lat_fine, dtype=dt, device=device)
        qcu.applyMultigridProLongQcu(out_p, coarse_vec, lonv, set_ptrs, params)
        out["prolong_max_diff"] = float((out_p - p_py).abs().max().item())
        out["prolong_rel_diff"] = float((out_p - p_py).abs().max().item() /
                                        (p_py.abs().max().item() + 1e-30))
    src_c = torch.randn([E, X, Y, Z, T], dtype=dt, device=device)

    def Ac(v):
        f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=v)
        return tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S(f))
    ref = Ac(src_c)
    cu = apply_stencil(hnn, hdg, sit, src_c)
    out["coarse_dslash_rel_diff"] = float((cu - ref).abs().max().item() /
                                          (ref.abs().max().item() + 1e-30))
    return out


def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def cache_disk_mb():
    d = os.path.expanduser("~/PyQCU/logs/nullvec_cache")
    if not os.path.isdir(d):
        d = os.path.join(REPO, "logs", "nullvec_cache")
    total = 0
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


# ----------------------------------------------------------------------
# CudaSchurOp（原 mg_dev74_dslash.py，内联照抄）
# ----------------------------------------------------------------------
_GLOBAL_SET_COUNTER = [0]


def _next_set_index():
    _GLOBAL_SET_COUNTER[0] += 1
    return _GLOBAL_SET_COUNTER[0]


class CudaSchurOp(object):
    """C++ CUDA 实现的 Schur 奇偶算子（matvec_parity 等价物）。

    matvec(x_o) -> y_o：x_o/y_o 为 [12,X,Y,Z,T/2] 奇子格场。
    构造即分配独立 LatticeSet（scratch 缓冲）；release() 释放。
    """

    def __init__(self, av, g, ce, coo, cei, coi):
        self.params = params.clone()
        self.params[define._SET_INDEX_] = _next_set_index()
        self.set_index = int(self.params[define._SET_INDEX_])
        self.params[define._SET_PLAN_] = 1
        self.params[define._VERBOSE_] = 0
        self._g, self._ce, self._coo, self._cei, self._coi = g, ce, coo, cei, coi
        qcu.applyInitQcu(set_ptrs, self.params, av)

    def matvec(self, x_o):
        y_o = torch.empty_like(x_o)
        qcu.applyCloverBistabCgDslashQcu(
            y_o, x_o, self._g, self._ce, self._coo, self._cei, self._coi,
            set_ptrs, self.params)
        # BUGFIX test12: C++ dslash 在私有 stream 异步执行（dev74 MT 移除全局同步），
        # 返回后 y_o 可能未写完 —— 必须先同步再返回，否则读取产生竞态（非确定结果）
        torch.cuda.synchronize()
        return y_o

    def release(self):
        qcu.applyEndQcu(set_ptrs, self.params)
        self.set_index = None


def make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=1, verbose=False):
    """创建 n 个互不冲突的 CudaSchurOp 实例（多线程各持一个）。单线程调用。"""
    ops = [CudaSchurOp(av, g, ce, coo, cei, coi) for _ in range(n)]
    if verbose:
        print(f"[test12] created {n} CudaSchurOp set_index="
              f"{[o.set_index for o in ops]}")
    return ops


# ----------------------------------------------------------------------
# 预算模型（原 mg_dev74_budget.py，内联；默认用实测校准系数）
# ----------------------------------------------------------------------
CONST_PER_V = 24192.0   # bytes per lattice point (Python-side tensors, c64, 2L, E=48)
ALPHA_DEFAULT = 30.83   # KB/V cold 校准系数（dev74 --fit 实测：cold 53KB/V）
BETA_DEFAULT = -27.0    # MB 固定开销（校准拟合 β）
ALPHA_WARM = 2.8        # KB/V warm 校准系数（dev74 实测：warm 27KB/V）
ALPHA_RAM = 5.0         # KB/V（RSS 模型）
BETA_RAM = 1200.0       # MB（RSS 模型）


def disk_cache_bytes(V, levels=2, E=48):
    """nullvec 缓存（lonv/hnn/hdg/sit, CPU 保存）磁盘占用。"""
    Vc = V / 32.0
    lonv = E * 12 * V / 2 * 8
    hnn = 2 * 4 * E * E * Vc * 8
    hdg = 2 * 2 * 6 * E * E * Vc * 8
    sit = E * E * Vc * 8
    return (lonv + hnn + hdg + sit) * levels


def vram_model(v, alpha_kb_per_v=ALPHA_DEFAULT, beta_mb=BETA_DEFAULT):
    """预测峰值显存（MB）——cold（含粗算子构建）峰值。"""
    return CONST_PER_V * v / 1e6 + alpha_kb_per_v * v / 1024.0 + beta_mb


def vram_model_warm(v, alpha_kb_per_v=ALPHA_WARM, beta_mb=BETA_DEFAULT):
    """预测峰值显存（MB）——warm（nullvec 缓存命中，仅求解）。"""
    return CONST_PER_V * v / 1e6 + alpha_kb_per_v * v / 1024.0 + beta_mb


def rss_model(v, alpha_ram_kb_per_v=ALPHA_RAM, beta_ram_mb=BETA_RAM):
    """预测进程峰值内存 RSS（MB）。"""
    return alpha_ram_kb_per_v * v / 1024.0 + beta_ram_mb


def vram_gb_option(vram):
    """显存档位：16（默认）/ 32（预留）。"""
    return 16 if vram is None or int(vram) <= 16 else 32


LATTICES = {
    "local":  [(8, 8, 8, 16), (8, 16, 16, 16), (16, 16, 16, 16)],
    "server": [(8, 32, 32, 32), (16, 32, 32, 32), (16, 32, 32, 64)],
}
# 32GB 档服务器格子（预留，--vram 32 启用）
LATTICES32 = {
    "server": [(16, 32, 32, 32), (16, 32, 32, 64), (24, 32, 32, 64)],
}


def fit_from_bench():
    """从 test12_bench.json 实测（peak_vram_mb, V）线性拟合 α/β。"""
    path = os.path.join(WORKDIR, "test12_bench.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    pts = [(r["lattice"], r.get("peak_vram_mb")) for r in data.get("results", [])
           if r.get("peak_vram_mb")]
    if len(pts) < 2:
        return None
    import numpy as np
    Vs = np.array([L[0] * L[1] * L[2] * L[3] for L, _ in pts], dtype=float)
    y = np.array([m for _, m in pts], dtype=float)
    a, b = np.polyfit(Vs, y, 1)
    alpha = (a * 1e6 - CONST_PER_V) / (1e6 / 1024.0)
    return (float(alpha), float(b))


def budget_table(mode="server", vram=16, alpha=None, beta=None, alpha_warm=None):
    if alpha is None:
        alpha = ALPHA_DEFAULT
    if beta is None:
        beta = BETA_DEFAULT
    if alpha_warm is None:
        alpha_warm = ALPHA_WARM
    lats = LATTICES32[mode] if vram >= 32 else LATTICES[mode]
    vram_mb = vram * 1024
    rows = []
    for L in lats:
        V = L[0] * L[1] * L[2] * L[3]
        vram_c = vram_model(V, alpha, beta)
        vram_w = vram_model_warm(V, alpha_warm, beta)
        rss = rss_model(V)
        disk = disk_cache_bytes(V, levels=2)
        rows.append({"lattice": list(L), "V": V,
                     "pred_vram_mb": round(vram_c), "pred_vram_warm_mb": round(vram_w),
                     "pred_rss_mb": round(rss),
                     "pred_disk_mb": round(disk / 1e6, 1),
                     "vram_frac": round(vram_c / vram_mb, 3),
                     "vram_warm_frac": round(vram_w / vram_mb, 3),
                     "rss_frac_512g": round(rss / (512 * 1024), 3),
                     "vram_gb": vram})
    return rows


# ----------------------------------------------------------------------
# clean —— 干净（独立进程）性能测量 + 资源统计（原 mg_dev74_clean.py）
# ----------------------------------------------------------------------
def measure_one(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST, MG_GRID,
                NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR, DT,
                NV_ITERS=2, gauge_seed=42, pairs=5, build_mode="py"):
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
                      MG_GRID, NUM_RESTART, COARSE_MAX_ITER,
                      COARSE_TOL_FACTOR, DT)
    KAPPA = 1.0 / (2 * MASS + 8)
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)
    torch.manual_seed(gauge_seed)
    g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
    fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi)
    fo_mg = torch.zeros_like(fi)
    ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce)
    coo = torch.zeros_like(ce)
    coi = torch.zeros_like(ce)
    torch.cuda.reset_peak_memory_stats()

    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity

    if build_mode == "cpp":
        cpp_ops = make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=1)
        S_build = cpp_ops[0].matvec
    else:
        cpp_ops = None
        S_build = None
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    E_prev = 12
    t_build = 0.0
    for lvl in range(1, NUM_LEVELS):
        E_c = DOF_LIST[lvl]
        lat_coarse_odd = [lat_fine_odd[d] // MG_GRID[d] for d in range(4)]
        t0 = time.perf_counter()
        lonv, hnn, hdg, sit = build_or_load_coarse_ops(
            gauge_seed, [Lx, Ly, Lz, Lt], lvl, E_c, E_prev, lat_fine_odd,
            lat_coarse_odd, S_build or S, dt, device, NV_ITERS, use_cache=True,
            save=True, verbose=False)
        t_build += time.perf_counter() - t0
        set_ptrs[30 + 4 * (lvl - 1) + 0] = lonv.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 1] = hnn.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 2] = hdg.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 3] = sit.contiguous().data_ptr()
        S = lambda v, hnn_i=hnn, hdg_i=hdg, sit_i=sit: apply_stencil(hnn_i, hdg_i, sit_i, v)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd
    peak_cold = torch.cuda.max_memory_allocated() / 1e6

    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    ref_times, mg_times = [], []
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    torch.cuda.synchronize()
    for i in range(pairs):
        fo_ref.zero_(); fo_mg.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi,
                                   set_ptrs, params)
        torch.cuda.synchronize()
        ref_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi,
                                    set_ptrs, params)
        torch.cuda.synchronize()
        mg_times.append(time.perf_counter() - t0)
    peak_warm = torch.cuda.max_memory_allocated() / 1e6
    conv_mg, prof, _n_iter = parse_mg_log(LOG_PATH)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
    mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, KAPPA, True) +
                        dslash.give_clover(qcu_mg, ref_cl) - qcu_src) / tools.norm(qcu_src)
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)
    rmin = min(ref_times); mmin = min(mg_times)
    rmed = sorted(ref_times)[len(ref_times) // 2]
    mmed = sorted(mg_times)[len(mg_times) // 2]
    ref_hist = ref_conv_history(op, qcu_src, ATOL)
    if cpp_ops:
        for o in cpp_ops:
            o.release()
    res = {"lattice": [Lx, Ly, Lz, Lt],
           "precision": "c128" if dt == torch.complex128 else "c64",
           "levels": NUM_LEVELS, "dof": DOF_LIST,
           "restart": NUM_RESTART, "ct": COARSE_TOL_FACTOR,
           "cmi": COARSE_MAX_ITER,
           "ref_times_ms": [t * 1000 for t in ref_times],
           "mg_times_ms": [t * 1000 for t in mg_times],
           "ref_min_ms": rmin * 1000, "mg_min_ms": mmin * 1000,
           "speedup_min": rmin / mmin,
           "ref_med_ms": rmed * 1000, "mg_med_ms": mmed * 1000,
           "speedup_med": rmed / mmed,
           "vs_ref": float(vs_ref), "mg_res": float(mg_res),
           "ref_res": float(ref_res),
           "mg_iters": len(conv_mg) - 1 if conv_mg else 0,
           "ref_iters": len(ref_hist) - 1,
           "conv_mg": conv_mg, "ref_hist": ref_hist, "prof": prof,
           "build_mode": build_mode,
           "build_s": t_build,
           "peak_vram_cold_mb": round(peak_cold, 1),
           "peak_vram_warm_mb": round(peak_warm, 1),
           "rss_kb": rss_kb(),
           "disk_mb": round(cache_disk_mb(), 1)}
    return res


def cmd_clean(args):
    Lx, Ly, Lz, Lt = args.lattice
    DT = define._LAT_C128_ if args.prec == "c128" else define._LAT_C64_
    dof = args.dof or ([12, 48] if args.levels == 2 else [12, 48, 48])
    res = measure_one(Lx, Ly, Lz, Lt, 0.05, 1e-6, args.levels, dof,
                      [2, 2, 2, 2], args.restart, args.cmi, args.ct, DT,
                      NV_ITERS=2, gauge_seed=42, pairs=args.pairs,
                      build_mode=args.build)
    label = (f"L{'x'.join(map(str,[Lx,Ly,Lz,Lt]))}_{args.prec}_"
             f"L{args.levels}_r{args.restart}_ct{args.ct:.0e}_cmi{args.cmi}"
             f"_{args.build}")
    res["label"] = label
    out_path = os.path.join(WORKDIR, f"test12_clean_{label}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    try:
        print(json.dumps(res), flush=True)
    except BrokenPipeError:
        pass


# ----------------------------------------------------------------------
# bench —— 批量基准 + 资源统计（原 mg_dev74_bench.py）
# ----------------------------------------------------------------------
_REF_HIST_CACHE = {}


def bench_one(label, Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
              MG_GRID, NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR,
              DT=define._LAT_C64_, NV_ITERS=2, gauge_seed=42,
              ntrials_mg=3, ntrials_ref=3, build_mode="py"):
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, NUM_LEVELS, DOF_LIST,
                      MG_GRID, NUM_RESTART, COARSE_MAX_ITER, COARSE_TOL_FACTOR, DT)
    KAPPA = 1.0 / (2 * MASS + 8)
    device = torch.device('cuda')
    dt = define.dtype(DT)
    ls = define.lat_shape(params)
    torch.manual_seed(gauge_seed)
    g = torch.zeros([2, 3, 3, 4] + ls, dtype=dt, device=device)
    fi = torch.randn([2, 4, 3] + ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi)
    fo_mg = torch.zeros_like(fi)
    ce = torch.zeros([4, 3, 4, 3] + ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce)
    coo = torch.zeros_like(ce)
    coi = torch.zeros_like(ce)
    torch.cuda.reset_peak_memory_stats()

    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    ref_times = []
    for _ in range(ntrials_ref):
        fo_ref.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi,
                                   set_ptrs, params)
        torch.cuda.synchronize()
        ref_times.append(time.perf_counter() - t0)
    ref_times_sorted = sorted(ref_times)
    ref_time = ref_times_sorted[len(ref_times_sorted) // 2]
    peak_ref = torch.cuda.max_memory_allocated() / 1e6

    qcu_U = tools.poooxyzt2oooxyzt(g)
    qcu_src = tools.poooxyzt2oooxyzt(fi)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, qcu_U, KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)

    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity

    ref_key = (tuple([Lx, Ly, Lz, Lt]), "c128" if dt == torch.complex128 else "c64")
    if ref_key not in _REF_HIST_CACHE:
        _REF_HIST_CACHE[ref_key] = ref_conv_history(op, qcu_src, ATOL)
    ref_hist = _REF_HIST_CACHE[ref_key]
    ref_iters = len(ref_hist) - 1

    if build_mode == "cpp":
        cpp_ops = make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=1)
        S_cpp = cpp_ops[0].matvec
    else:
        cpp_ops = None
        S_cpp = None
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    E_prev = 12
    t_build = 0.0
    for lvl in range(1, NUM_LEVELS):
        E_c = DOF_LIST[lvl]
        lat_coarse_odd = [lat_fine_odd[d] // MG_GRID[d] for d in range(4)]
        t0 = time.perf_counter()
        lonv, hnn, hdg, sit = build_or_load_coarse_ops(
            gauge_seed, [Lx, Ly, Lz, Lt], lvl, E_c, E_prev, lat_fine_odd,
            lat_coarse_odd, S_cpp or S, dt, device, NV_ITERS, use_cache=True,
            save=True, verbose=False)
        t_build += time.perf_counter() - t0
        set_ptrs[30 + 4 * (lvl - 1) + 0] = lonv.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 1] = hnn.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 2] = hdg.contiguous().data_ptr()
        set_ptrs[30 + 4 * (lvl - 1) + 3] = sit.contiguous().data_ptr()
        S = lambda v, hnn_i=hnn, hdg_i=hdg, sit_i=sit: apply_stencil(hnn_i, hdg_i, sit_i, v)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd
    peak_build = torch.cuda.max_memory_allocated() / 1e6

    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    params[define._VERBOSE_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    mg_times = []
    torch.cuda.synchronize()
    for _ in range(ntrials_mg):
        fo_mg.zero_()
        t0 = time.perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg, fi, g, ce, coo, cei, coi,
                                    set_ptrs, params)
        torch.cuda.synchronize()
        mg_times.append(time.perf_counter() - t0)
    mg_times_sorted = sorted(mg_times)
    mg_time = mg_times_sorted[len(mg_times_sorted) // 2]
    peak_mg = torch.cuda.max_memory_allocated() / 1e6

    conv, prof, n_iter = parse_mg_log(LOG_PATH)

    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    mg_res = tools.norm(dslash.give_wilson(qcu_mg, qcu_U, KAPPA, True) +
                        dslash.give_clover(qcu_mg, ref_cl) - qcu_src) / tools.norm(qcu_src)
    mg_vs_ref = tools.norm(qcu_mg - qcu_ref) / tools.norm(qcu_ref)
    mg_iters = len(conv) - 1 if conv else (n_iter or 0)

    if cpp_ops:
        for o in cpp_ops:
            o.release()

    res = {
        "label": label,
        "lattice": [Lx, Ly, Lz, Lt],
        "precision": "c128" if dt == torch.complex128 else "c64",
        "mass": MASS, "atol": ATOL,
        "levels": NUM_LEVELS, "dof": DOF_LIST, "restart": NUM_RESTART,
        "coarse_tol_factor": COARSE_TOL_FACTOR, "coarse_max_iter": COARSE_MAX_ITER,
        "ref_ms": ref_time * 1000, "ref_iters": ref_iters,
        "ref_res": float(ref_res),
        "mg_ms": mg_time * 1000, "mg_iters": mg_iters,
        "speedup": ref_time / mg_time if mg_time > 0 else 0.0,
        "mg_res": float(mg_res), "vs_ref": float(mg_vs_ref),
        "conv_mg": conv,
        "ref_hist": ref_hist,
        "prof": prof,
        "build_mode": build_mode,
        "build_s": t_build,
        "peak_vram_ref_mb": round(peak_ref, 1),
        "peak_vram_build_mb": round(peak_build, 1),
        "peak_vram_mg_mb": round(peak_mg, 1),
        "peak_vram_mb": round(peak_mg, 1),
        "rss_kb": rss_kb(),
        "disk_mb": round(cache_disk_mb(), 1),
    }
    print(f"[{label}] ref={res['ref_ms']:.0f}ms({ref_iters}it) "
          f"mg={res['mg_ms']:.0f}ms({mg_iters}it) "
          f"speedup={res['speedup']:.3f}x mg_res={mg_res:.2e} "
          f"vs_ref={mg_vs_ref:.2e} "
          f"vram={res['peak_vram_mb']:.0f}MB rss={res['rss_kb']/1e3:.0f}MB "
          f"disk={res['disk_mb']:.0f}MB build={t_build:.0f}s", flush=True)
    return res


_BASE = dict(MASS=0.05, ATOL=1e-6, NUM_LEVELS=2, DOF_LIST=[12, 48],
             MG_GRID=[2, 2, 2, 2], NUM_RESTART=10, COARSE_MAX_ITER=15,
             COARSE_TOL_FACTOR=1e5, DT=define._LAT_C64_, NV_ITERS=2)

LOCAL_CONFIGS = [
    dict(label="8x8x8x16_c64_2L_r10_ct1e5_cmi15", Lx=8, Ly=8, Lz=8, Lt=16, **_BASE),
    dict(label="8x16x16x16_c64_2L_r10_ct1e5_cmi15", Lx=8, Ly=16, Lz=16, Lt=16, **_BASE),
    dict(label="16x16x16x16_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=16, Lz=16, Lt=16, **_BASE),
]

SERVER16_CONFIGS = [
    dict(label="8x32x32x32_c64_2L_r10_ct1e5_cmi15", Lx=8, Ly=32, Lz=32, Lt=32, **_BASE),
    dict(label="16x32x32x32_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=32, Lz=32, Lt=32, **_BASE),
    dict(label="16x32x32x64_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=32, Lz=32, Lt=64, **_BASE),
]

SERVER32_CONFIGS = [
    dict(label="16x32x32x32_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=32, Lz=32, Lt=32, **_BASE),
    dict(label="16x32x32x64_c64_2L_r10_ct1e5_cmi15", Lx=16, Ly=32, Lz=32, Lt=64, **_BASE),
    dict(label="24x32x32x64_c64_2L_r10_ct1e5_cmi15", Lx=24, Ly=32, Lz=32, Lt=64, **_BASE),
]


def cmd_bench(args):
    cfgs = SERVER32_CONFIGS if args.vram >= 32 else (
        SERVER16_CONFIGS if args.mode == "server" else LOCAL_CONFIGS)
    gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
    vram_mb = args.vram * 1024
    results = []
    for cfg in cfgs:
        label = cfg["label"]
        if args.only and not any(label.startswith(p) for p in args.only):
            print(f"[skip] {label}")
            continue
        V = cfg["Lx"] * cfg["Ly"] * cfg["Lz"] * cfg["Lt"]
        pred = vram_model(V)
        if pred > min(gpu_mem_mb, vram_mb) * 0.9:
            print(f"[skip] {label}: 预算预测 {pred:.0f}MB > 显存档 "
                  f"{min(gpu_mem_mb, vram_mb):.0f}MB×0.9")
            results.append({"label": label, "lattice": [cfg["Lx"], cfg["Ly"],
                            cfg["Lz"], cfg["Lt"]], "skipped": "vram-budget",
                            "pred_vram_mb": round(pred)})
            continue
        try:
            r = bench_one(**cfg, build_mode=args.build)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{label}] FAILED: {e}")
            results.append({"label": label, "lattice": [cfg["Lx"], cfg["Ly"],
                           cfg["Lz"], cfg["Lt"]], "failed": str(e)})

    for key, hist in _REF_HIST_CACHE.items():
        with open(os.path.join(WORKDIR, "test12_ref_conv.json"), "w") as f:
            json.dump({str(key): {"hist": hist, "iters": len(hist) - 1}}, f, indent=2)

    out = {"results": results, "mode": args.mode, "build_mode": args.build,
           "vram_gb": args.vram}
    out_path = os.path.join(WORKDIR, "test12_bench.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== SUMMARY ({'SERVER' if args.mode == 'server' else 'LOCAL'}, "
          f"build={args.build}, vram={args.vram}G) ===")
    for r in results:
        if "skipped" in r or "failed" in r:
            print(f"  {r['label']}: {r.get('skipped') or r.get('failed')}")
            continue
        print(f"{r['label']}: {r['speedup']:.3f}x  mg={r['mg_ms']:.0f}ms "
              f"ref={r['ref_ms']:.0f}ms iters={r['mg_iters']}/{r['ref_iters']} "
              f"vs_ref={r['vs_ref']:.2e} vram={r['peak_vram_mb']:.0f}MB "
              f"rss={r['rss_kb']/1e3:.0f}MB")
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------
# verify —— 正确性验证（原 mg_dev74_verify.py + CudaSchurOp 对照）
# ----------------------------------------------------------------------
def verify_dslash_op(base, n_trial=8):
    """CudaSchurOp vs Python matvec_parity：一致性 + 耗时。"""
    dt, device = base["dt"], base["device"]
    ls = base["ls"]
    x_o = torch.randn([12] + ls, dtype=dt, device=device)
    y_py = base["S"](x_o)
    ops = make_cuda_schur_ops(base["av"], base["g"], base["ce"], base["coo"],
                              base["cei"], base["coi"], n=1)
    y_cpp = ops[0].matvec(x_o)
    err = float(tools.norm(y_cpp - y_py) / tools.norm(y_py))

    def med(f, n):
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            f()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2] * 1000

    t_py = med(lambda: base["S"](x_o), n_trial)
    t_cpp = med(lambda: ops[0].matvec(x_o), n_trial)
    for o in ops:
        o.release()
    return {"rel_err": float(err), "t_py_ms": round(t_py, 3),
            "t_cpp_ms": round(t_cpp, 3), "speedup": round(t_py / t_cpp, 2)}


def cmd_verify(args):
    Lx, Ly, Lz, Lt = args.lattice
    DT = define._LAT_C128_ if args.prec == "c128" else define._LAT_C64_
    base = build_base(Lx, Ly, Lz, Lt, 0.05, 1e-6, 2, [12, 48], [2, 2, 2, 2], DT)
    dt, device = base["dt"], base["device"]
    E = 48
    lat_fine = [Lx, Ly, Lz, Lt]
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    lat_coarse_odd = [Lx // 2, Ly // 2, Lz // 2, Lt // 4]

    out = {
        "lattice": lat_fine, "precision": args.prec,
        "gauge": verify_lattice(base["qcu_U"]),
        "dslash_cpp_vs_py": verify_dslash_op(base),
    }

    qcu_ref, qcu_src = base["qcu_ref"], base["qcu_src"]
    ref_cl = base["ref_cl"]
    KAPPA = base["KAPPA"]
    ref_res = tools.norm(dslash.give_wilson(qcu_ref, base["qcu_U"], KAPPA, True) +
                         dslash.give_clover(qcu_ref, ref_cl) - qcu_src) / tools.norm(qcu_src)
    out["ref_solution"] = {"ref_res": float(ref_res)}

    lonv, hnn, hdg, sit = build_or_load_coarse_ops(
        42, lat_fine, 1, E, 12, lat_fine_odd, lat_coarse_odd,
        base["S"], dt, device, 2, use_cache=True, save=True, verbose=False)
    out["nullvecs"] = verify_nullvecs(base["op"], base["S"], lonv, hnn, hdg,
                                      sit, E, 12, lat_fine_odd,
                                      lat_coarse_odd, dt, device)

    path = os.path.join(WORKDIR,
                        f"test12_verify_{'x'.join(map(str, lat_fine))}_{args.prec}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


# ----------------------------------------------------------------------
# sweep —— 参数扫描（原 mg_dev74_1_sweep.py；子进程 = main.py clean）
# ----------------------------------------------------------------------
def cfgs_for(lattice):
    """(r, ct, cmi, levels) 组合 —— 侧重加速机制扫描。"""
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


def cmd_sweep(args):
    L = args.lattice
    tag = "x".join(map(str, L))
    cfgs = cfgs_for(L)
    main_py = os.path.abspath(__file__)
    results = []
    for i, c in enumerate(cfgs):
        cmd = [sys.executable, main_py, "clean", "--lattice"] + list(map(str, L)) + \
              ["--prec", "c64", "--levels", str(c["levels"]),
               "--restart", str(c["restart"]), "--ct", str(c["ct"]),
               "--cmi", str(c["cmi"]), "--pairs", str(args.pairs)]
        print(f"\n[{i+1}/{len(cfgs)}] {' '.join(cmd)}", flush=True)
        try:
            r = subprocess.run(cmd, timeout=args.timeout)  # 透传完整输出
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>{args.timeout}s)，跳过", flush=True)
            results.append({"lattice": L, "restart": c["restart"],
                            "ct": c["ct"], "cmi": c["cmi"],
                            "levels": c["levels"], "failed": "timeout"})
            continue
        if r.returncode != 0:
            print(f"  FAILED rc={r.returncode}", flush=True)
            results.append({"lattice": L, "restart": c["restart"],
                            "ct": c["ct"], "cmi": c["cmi"],
                            "levels": c["levels"], "failed": True})
            continue
        label = (f"L{tag}_c64_L{c['levels']}_r{c['restart']}"
                 f"_ct{c['ct']:.0e}_cmi{c['cmi']}_py")
        p = os.path.join(WORKDIR, f"test12_clean_{label}.json")
        if not os.path.exists(p):
            print(f"  MISSING json: {p}", flush=True)
            results.append({"lattice": L, "restart": c["restart"],
                            "ct": c["ct"], "cmi": c["cmi"],
                            "levels": c["levels"], "failed": "no-json"})
            continue
        with open(p) as f:
            d = json.load(f)
        results.append(d)
        sp = d.get("speedup_min")
        print(f"  -> speedup_min={sp:.3f}  iters={d['mg_iters']}/{d['ref_iters']}"
              f"  vs_ref={d['vs_ref']:.1e}", flush=True)

    out_path = os.path.join(WORKDIR, "test12_sweep.json")
    with open(out_path, "w") as f:
        json.dump({"lattice": L, "results": results}, f, indent=2)
    print(f"\nwrote {out_path} ({len(results)} configs)")
    ok = [d for d in results if not d.get("failed") and
          d.get("speedup_min", 0) >= args.gate]
    print(f"speedup_min >= {args.gate}: {len(ok)}/{len(results)} configs")


# ----------------------------------------------------------------------
# check —— 加速比断言（原 mg_dev74_1_check.py）
# ----------------------------------------------------------------------
def _add_entry(entries, seen, e):
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


def load_all(files):
    """读取显式指定的 json 文件；缺省读 test12_sweep.json。"""
    entries = []
    seen = set()
    if files:
        for p in files:
            if not os.path.exists(p):
                print(f"[warn] 文件不存在: {p}")
                continue
            with open(p) as f:
                d = json.load(f)
            if "results" in d and isinstance(d["results"], list):
                for r in d["results"]:
                    _add_entry(entries, seen, r)
            elif isinstance(d, dict) and "lattice" in d:
                _add_entry(entries, seen, d)
    else:
        p = os.path.join(WORKDIR, "test12_sweep.json")
        if os.path.exists(p):
            with open(p) as f:
                for r in json.load(f)["results"]:
                    _add_entry(entries, seen, r)
    return entries


def cmd_check(args):
    entries = load_all(args.file)
    if not entries:
        print(f"[{args.label}] NO DATA — 请先运行 "
              f"python logs/test12/main.py sweep / clean")
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


# ----------------------------------------------------------------------
# budget —— 预算表（原 mg_dev74_budget.py，--vram 16|32）
# ----------------------------------------------------------------------
def cmd_budget(args):
    alpha, beta = ALPHA_DEFAULT, BETA_DEFAULT
    if args.fit:
        f = fit_from_bench()
        if f:
            alpha, beta = f
            print(f"[fit] alpha={alpha:.2f} KB/V  beta={beta:.0f} MB "
                  f"(从 test12_bench.json 实测)")
        else:
            print("[fit] test12_bench.json 无实测数据，使用默认系数")
    vram = vram_gb_option(args.vram)
    rows = budget_table(args.mode, vram, alpha, beta)
    print(f"{'lattice':20s} {'V':>9s} {'VRAM_cold':>10s} {'VRAM_warm':>10s} "
          f"{'RSS(MB)':>9s} {'disk(MB)':>9s} {'cold/fr':>8s} {'warm/fr':>8s}")
    for r in rows:
        print(f"{'x'.join(map(str, r['lattice'])):20s} {r['V']:9d} "
              f"{r['pred_vram_mb']:10d} {r['pred_vram_warm_mb']:10d} "
              f"{r['pred_rss_mb']:9d} "
              f"{r['pred_disk_mb']:9.1f} {r['vram_frac']:8.3f} "
              f"{r['vram_warm_frac']:8.3f}")
    out = os.path.join(WORKDIR, f"test12_budget_{args.mode}_{vram}g.json")
    with open(out, "w") as f:
        json.dump({"alpha_kb_per_v": alpha, "beta_mb": beta, "vram_gb": vram,
                   "rows": rows}, f, indent=2)
    print(f"wrote {out}")
    print(f"说明：16x32x32x64 warm {rows[-1]['pred_vram_warm_mb']}MB —— "
          f"{'可行' if rows[-1]['vram_warm_frac'] < 0.95 else f'超 {vram}G 档，需分阶段/多卡'}")


# ----------------------------------------------------------------------
# collect —— 汇总（原 mg_dev74_collect.py）
# ----------------------------------------------------------------------
def cfg_key(r):
    return (tuple(r["lattice"]), r.get("precision", "c64"), r.get("levels", 2),
            r.get("restart", 10),
            float(r.get("ct", r.get("coarse_tol_factor", 1e5))),
            int(r.get("cmi", r.get("coarse_max_iter", 15))))


def cmd_collect(args):
    bench_path = os.path.join(WORKDIR, "test12_bench.json")
    bench = json.load(open(bench_path)) if os.path.exists(bench_path) else {"results": []}
    clean_files = sorted(glob.glob(os.path.join(WORKDIR, "test12_clean_L*.json")))
    clean = [json.load(open(f)) for f in clean_files]
    warm = bench["results"]

    warm_by_key = {}
    for r in warm:
        if "lattice" not in r:
            continue
        warm_by_key.setdefault(cfg_key(r), []).append(r)

    out = []
    for r in clean:
        k = cfg_key(r)
        w = warm_by_key.get(k, [{}])[0]
        entry = {
            "label": r["label"], "lattice": r["lattice"],
            "precision": r["precision"], "levels": r["levels"], "dof": r["dof"],
            "restart": r["restart"], "ct": r["ct"], "cmi": r["cmi"],
            "ref_min_ms": r["ref_min_ms"], "mg_min_ms": r["mg_min_ms"],
            "speedup_min": r["speedup_min"],
            "ref_med_ms": r["ref_med_ms"], "mg_med_ms": r["mg_med_ms"],
            "speedup_med": r["speedup_med"],
            "vs_ref": r["vs_ref"], "mg_res": r["mg_res"], "ref_res": r["ref_res"],
            "mg_iters": r.get("mg_iters", w.get("mg_iters", 0)),
            "ref_iters": r.get("ref_iters", w.get("ref_iters", 0)),
            "conv_mg": r.get("conv_mg") or w.get("conv_mg", []),
            "ref_hist": r.get("ref_hist") or w.get("ref_hist", []),
            "prof": r.get("prof") or w.get("prof", {}),
            "build_mode": r.get("build_mode", "py"),
            "build_s": r.get("build_s", w.get("build_s", 0.0)),
            "peak_vram_cold_mb": r.get("peak_vram_cold_mb", w.get("peak_vram_build_mb")),
            "peak_vram_warm_mb": r.get("peak_vram_warm_mb", w.get("peak_vram_mg_mb")),
            "rss_kb": r.get("rss_kb", w.get("rss_kb")),
            "disk_mb": r.get("disk_mb", w.get("disk_mb")),
        }
        out.append(entry)

    clean_keys = {cfg_key(r) for r in out}
    for r in warm:
        if "lattice" not in r or "failed" in r or "skipped" in r:
            continue
        k = cfg_key(r)
        if k in clean_keys:
            continue
        out.append({
            "label": r["label"], "lattice": r["lattice"],
            "precision": r["precision"], "levels": r["levels"], "dof": r["dof"],
            "restart": r["restart"], "ct": r["coarse_tol_factor"],
            "cmi": r["coarse_max_iter"],
            "ref_min_ms": r["ref_ms"], "mg_min_ms": r["mg_ms"],
            "speedup_min": r["speedup"],
            "ref_med_ms": r["ref_ms"], "mg_med_ms": r["mg_ms"],
            "speedup_med": r["speedup"],
            "vs_ref": r["vs_ref"], "mg_res": r["mg_res"], "ref_res": r["ref_res"],
            "mg_iters": r["mg_iters"], "ref_iters": r["ref_iters"],
            "conv_mg": r.get("conv_mg", []), "ref_hist": r.get("ref_hist", []),
            "prof": r.get("prof", {}),
            "build_mode": r.get("build_mode", "py"),
            "build_s": r.get("build_s", 0.0),
            "peak_vram_cold_mb": r.get("peak_vram_build_mb"),
            "peak_vram_warm_mb": r.get("peak_vram_mg_mb"),
            "rss_kb": r.get("rss_kb"), "disk_mb": r.get("disk_mb"),
            "_note": "bench（非独立进程）计时与资源",
        })

    verify = {}
    for vf in sorted(glob.glob(os.path.join(WORKDIR, "test12_verify_*.json"))):
        v = json.load(open(vf))
        key = (tuple(v["lattice"]), v["precision"])
        verify[str(key)] = v

    out_path = os.path.join(WORKDIR, "test12_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": out, "verify": verify,
                   "bench_mode": bench.get("mode")}, f, indent=2)
    print(f"wrote {out_path}: {len(out)} configs, {len(verify)} verify sets")
    for e in out:
        sp = e["speedup_min"]
        s1 = "—" if sp is None else f"{sp:.2f}x"
        vram = e.get("peak_vram_warm_mb")
        vr = "—" if vram is None else f"{vram:.0f}MB"
        print(f"  {e['label']:44s} min={s1}  iters={e['mg_iters']}/{e['ref_iters']}"
              f"  vs_ref={e['vs_ref']:.1e}  vram={vr}")


# ----------------------------------------------------------------------
# mktable —— LaTeX 表（原 mg_dev74_mktable.py）
# ----------------------------------------------------------------------
def _esc(s):
    return s.replace("_", r"\_")


def cmd_mktable(args):
    res_path = os.path.join(WORKDIR, "test12_results.json")
    if not os.path.exists(res_path):
        print(f"[error] {res_path} 不存在，先运行 collect")
        sys.exit(2)
    data = json.load(open(res_path))
    results = data["results"]

    rows = []
    for r in results:
        sp = r.get("speedup_min")
        vram = r.get("peak_vram_warm_mb")
        rows.append((_esc(r["label"]),
                     f"{r['ref_min_ms']:.0f}" if r.get("ref_min_ms") else "—",
                     f"{r['mg_min_ms']:.0f}" if r.get("mg_min_ms") else "—",
                     "—" if sp is None else f"{sp:.2f}",
                     f"{r['mg_iters']}/{r['ref_iters']}",
                     f"{r['vs_ref']:.1e}",
                     "—" if vram is None else f"{vram:.0f}"))
    with open(os.path.join(WORKDIR, "test12_tbl_main.tex"), "w") as f:
        f.write("% test12 —— 性能结果表\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("配置 & ref(ms) & MG(ms) & 加速比 & iters(MG/ref) & vs\\_ref & 显存(MB) \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            f.write(" & ".join(["\\texttt{%s}" % r[0]] + list(r[1:])) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    with open(os.path.join(WORKDIR, "test12_tbl_res.tex"), "w") as f:
        f.write("% test12 —— 资源占用统计（实测）\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("配置 & V & cold显存(MB) & warm显存(MB) & RSS(MB) & 磁盘(MB) & 构建(s) \\\\\n")
        f.write("\\hline\n")
        for r in results:
            V = r["lattice"][0] * r["lattice"][1] * r["lattice"][2] * r["lattice"][3]
            f.write(f"\\texttt{{{_esc(r['label'])}}} & {V} & "
                    f"{r.get('peak_vram_cold_mb') or '—'} & "
                    f"{r.get('peak_vram_warm_mb') or '—'} & "
                    f"{(r.get('rss_kb') or 0)/1e3:.0f} & "
                    f"{r.get('disk_mb') or '—'} & "
                    f"{r.get('build_s') or 0:.0f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    bp = os.path.join(WORKDIR, f"test12_budget_{args.mode}_{args.vram}g.json")
    if os.path.exists(bp):
        budget = json.load(open(bp))
        with open(os.path.join(WORKDIR, "test12_tbl_budget.tex"), "w") as f:
            f.write(f"% test12 —— 服务器大格子预算预测（{args.vram}G 显存档，校准系数）\n")
            f.write("\\begin{tabular}{lcccc}\n\\hline\n")
            f.write("格子 & V & cold(GB) & warm(GB) & cold/lim \\\\\n\\hline\n")
            for row in budget["rows"]:
                L = "x".join(map(str, row["lattice"]))
                f.write(f"\\texttt{{{L}}} & {row['V']} & "
                        f"{row['pred_vram_mb']/1024:.1f} & "
                        f"{row['pred_vram_warm_mb']/1024:.1f} & "
                        f"{row['vram_frac']:.2f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n")
    print(f"wrote test12_tbl_*.tex 到 {WORKDIR}")


# ----------------------------------------------------------------------
# plots —— dev74 风格图（原 mg_dev74_plots.py）
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cmd_plots(args):
    res_path = os.path.join(WORKDIR, "test12_results.json")
    if not os.path.exists(res_path):
        print(f"[error] {res_path} 不存在，先运行 collect")
        sys.exit(2)
    data = json.load(open(res_path))
    results = data["results"]
    V = [r["lattice"][0] * r["lattice"][1] * r["lattice"][2] * r["lattice"][3]
         for r in results]
    sp = [r.get("speedup_min") for r in results]
    ref_ms = [r.get("ref_min_ms") for r in results]
    mg_ms = [r.get("mg_min_ms") for r in results]
    cold = [r.get("peak_vram_cold_mb") for r in results]
    warm = [r.get("peak_vram_warm_mb") for r in results]
    labels = [r["label"] for r in results]
    vram_lim = args.vram * 1024

    plt.figure(figsize=(7, 4.5))
    plt.plot(V, sp, "o-", label="test12 实测")
    ref_v = [8 * 8 * 8 * 16, 8 * 16 * 16 * 16, 16 * 16 * 16 * 16, 8 * 16 * 16 * 32]
    ref_sp = [2.43, 1.16, 0.81, 1.11]
    plt.plot(ref_v, ref_sp, "s--", label="dev73_5 V100-32G 参考")
    plt.axhline(1.0, color="gray", ls=":", lw=0.8)
    plt.xscale("log")
    plt.xlabel("lattice volume V")
    plt.ylabel("speedup (ref/MG)")
    plt.title(f"test12: speedup vs V (vram={args.vram}G)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(WORKDIR, "test12_speedup.png"), dpi=130)

    plt.figure(figsize=(7, 4.5))
    plt.plot(V, cold, "o-", label="cold 实测（含粗算子构建）")
    plt.plot(V, warm, "o-", label="warm 实测（缓存命中求解）")
    Vs = np.logspace(np.log10(8e3), np.log10(2e6), 40)
    model_cold = (24192 + 30.83 * 1024) * Vs / 1e6 - 27
    model_warm = (24192 + 2.8 * 1024) * Vs / 1e6 - 27
    plt.plot(Vs, model_cold, "--", color="C0", alpha=0.6, label="cold 模型 53KB/V")
    plt.plot(Vs, model_warm, "--", color="C1", alpha=0.6, label="warm 模型 27KB/V")
    plt.axhline(vram_lim, color="red", ls="--", lw=1.2, label=f"{args.vram}GB 显存极限")
    plt.xscale("log")
    plt.xlabel("lattice volume V")
    plt.ylabel("peak VRAM (MB)")
    plt.title(f"test12: peak VRAM vs V (vram={args.vram}G)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(WORKDIR, "test12_vram.png"), dpi=130)

    plt.figure(figsize=(7, 4.5))
    plt.plot(V, ref_ms, "o-", label="BiStabCG (min)")
    plt.plot(V, mg_ms, "s-", label="MG (min)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("lattice volume V")
    plt.ylabel("time (ms, log)")
    plt.title("test12: solve time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(WORKDIR, "test12_time.png"), dpi=130)

    for r in results:
        conv = r.get("conv_mg")
        if not conv:
            continue
        plt.figure(figsize=(6.5, 4))
        plt.semilogy(conv, "o-", ms=3, label="MG")
        rh = r.get("ref_hist")
        if rh:
            plt.semilogy(rh, "s-", ms=3, label="BiStabCG (Python 复现)")
        plt.xlabel("iteration")
        plt.ylabel("residual norm")
        plt.title(f"test12: {r['label']}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(WORKDIR,
                                 f"test12_conv_{r['label'].split('_')[0]}.png"),
                    dpi=130)

    bp = os.path.join(WORKDIR, f"test12_budget_server_{args.vram}g.json")
    if os.path.exists(bp):
        rows = json.load(open(bp))["rows"]
        Vc = [r["V"] for r in rows]
        cold_gb = [r["pred_vram_mb"] / 1024 for r in rows]
        warm_gb = [r["pred_vram_warm_mb"] / 1024 for r in rows]
        x = np.arange(len(rows))
        plt.figure(figsize=(7, 4.5))
        plt.bar(x - 0.2, cold_gb, 0.4, label="cold（首次构建）")
        plt.bar(x + 0.2, warm_gb, 0.4, label="warm（缓存命中）")
        plt.axhline(args.vram, color="red", ls="--", lw=1.2,
                    label=f"{args.vram}GB 极限")
        plt.xticks(x, [f"{r['V']//1024}k" for r in rows])
        plt.ylabel("VRAM (GB)")
        plt.title(f"test12: server lattice budget ({args.vram}G 档)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(WORKDIR, "test12_budget.png"), dpi=130)

    nfig = len([f for f in os.listdir(WORKDIR)
                if f.startswith("test12") and f.endswith(".png")])
    print(f"wrote test12_*.png ({nfig} figures) → {WORKDIR}")


# ----------------------------------------------------------------------
# plots1 —— dev74_1 风格图（原 mg_dev74_1_plots.py，作图范围同 dev73_5）
# ----------------------------------------------------------------------
import matplotlib.font_manager as fm

for _f in ["/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
           "/usr/share/fonts/truetype/arphic/uming.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback",
                               "AR PL UMing CN", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

C = {
    "blue": "#2a78d6", "green": "#008300", "magenta": "#e87ba4",
    "yellow": "#eda100", "aqua": "#1baf7a", "orange": "#eb6834",
    "violet": "#4a3aa7", "red": "#e34948",
}
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"
SURF = "#fcfcfb"
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281"]


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.grid(False, axis="x")
    ax.tick_params(grid_color=GRID)


def _fig_save(fig, name, dpi=150):
    path = os.path.join(WORKDIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"saved {path}")


def _short_label(r):
    return (f"L{r['levels']} r{r['restart']} ct{r['ct']:.0e} "
            f"cmi{r['cmi']}")


def _plot1_conv(results):
    groups = {}
    for r in results:
        key = (tuple(r["lattice"]), r["precision"])
        groups.setdefault(key, []).append(r)
    for (lat, prec), rs in groups.items():
        ref_hist = next((r["ref_hist"] for r in rs if r.get("ref_hist")), [])
        fig, ax = plt.subplots(figsize=(8, 4.6))
        _style(ax)
        lat_s = "×".join(str(x) for x in lat)
        lat_f = "x".join(str(x) for x in lat)
        ax.set_title(f"收敛历史  lattice={lat_s}  {prec}  (mass=0.05, atol=1e-6)",
                     color=INK, fontsize=11)
        ax.set_xlabel("迭代次数", color=INK2)
        ax.set_ylabel("Schur 残差 ||r||", color=INK2)
        if ref_hist:
            ax.plot(range(len(ref_hist)), ref_hist, color=C["green"],
                    lw=2.0, label="BiStabCG (参考)", marker="o", ms=3, zorder=3)
        for i, r in enumerate(rs):
            conv = r.get("conv_mg")
            if not conv:
                continue
            color = SEQUENTIAL_BLUE[(i + 2) % len(SEQUENTIAL_BLUE)]
            ax.plot(range(len(conv)), conv, color=color, lw=1.6,
                    label=_short_label(r), marker="s", ms=2.5, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e3)
        ax.axhline(1e-6, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(0.99, 1e-6, "atol=1e-6", color=MUTED, fontsize=8,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        _fig_save(fig, f"test12_1_conv_{lat_f}_{prec}.png")


def _plot1_hotspot(results):
    fields = [("fine_iter", "细层迭代 fine_iter", C["blue"]),
              ("vcycle", "V-cycle 修正", C["green"]),
              ("coarse_solve", "粗层求解 coarse_solve", C["orange"]),
              ("coarse_dslash", "粗层 dslash", C["violet"])]
    rs = [r for r in results if r.get("prof")]
    labels = [_short_label(r) for r in rs]
    vals = {k: [r.get("prof", {}).get(k, 0.0) for r in rs] for k, _, _ in fields}
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    _style(ax)
    ax.set_title("MG 计算热点分解 PROF_SECTIONS (ms)", color=INK, fontsize=11)
    y = np.arange(len(rs))[::-1]
    left = np.zeros(len(rs))
    for k, name, col in fields:
        v = np.array(vals[k])
        ax.barh(y, v, left=left, color=col, label=name, height=0.62,
                edgecolor=SURF, linewidth=0.5)
        left += v
    total = [r.get("mg_med_ms") or r.get("mg_min_ms") or 0 for r in rs]
    for i, yy in enumerate(y):
        t = total[i]
        if t and t > 0:
            ax.text(t + 2, yy, f"{t:.0f}ms", color=INK2, fontsize=7.5,
                    va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("耗时 (ms)", color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    fig.tight_layout()
    _fig_save(fig, "test12_1_hotspot.png")


def _plot1_speedup(results):
    rs = sorted([r for r in results if r.get("speedup_min") is not None],
                key=lambda r: r["speedup_min"])
    labels = [_short_label(r) for r in rs]
    sp = [r["speedup_min"] for r in rs]
    n = len(rs)
    colors = [SEQUENTIAL_BLUE[6 - (n - 1 - i) * 5 // max(n - 1, 1)]
              if n > 1 else C["blue"]
              for i in range(n)]
    pairs = max((len(r.get("ref_times_ms", [])) for r in rs), default=3)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _style(ax)
    ax.set_title(f"MultiGrid vs BiStabCG 加速比（干净测量，min of {pairs} 对）",
                 color=INK, fontsize=11)
    y = np.arange(n)[::-1]
    ax.barh(y, sp, color=colors, height=0.62, edgecolor=SURF, linewidth=0.5)
    for i, yy in enumerate(y):
        r = rs[i]
        lo = r.get("speedup_med")
        ax.text(sp[i] + 0.02, yy,
                f"{sp[i]:.2f}x  (MG {r['mg_min_ms']:.0f} / ref {r['ref_min_ms']:.0f} ms, "
                f"{r['mg_iters']}/{r['ref_iters']} it)", color=INK2, fontsize=7.5,
                va="center")
        if lo is not None and abs(lo - sp[i]) > 0.03:
            ax.plot([sp[i], lo], [yy, yy], color=MUTED, lw=0.8, marker="_",
                    markersize=6)
    ax.axvline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("加速比 (ref/min / mg/min)", color=INK2)
    ax.set_xlim(0, max(sp) * 1.35 + 0.1)
    fig.tight_layout()
    _fig_save(fig, "test12_1_speedup.png")


def _plot1_time(results):
    rs = sorted([r for r in results if r.get("mg_min_ms") is not None],
                key=lambda r: r["ref_min_ms"] - r["mg_min_ms"])
    labels = [_short_label(r) for r in rs]
    ref = [r["ref_min_ms"] for r in rs]
    mg = [r["mg_min_ms"] for r in rs]
    y = np.arange(len(rs))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _style(ax)
    ax.set_title("求解耗时对照（干净 min）：BiStabCG（参考） vs MultiGrid",
                 color=INK, fontsize=11)
    ax.barh(y + h / 2, ref, height=h, color=C["green"], label="BiStabCG 参考")
    ax.barh(y - h / 2, mg, height=h, color=C["blue"], label="MultiGrid")
    for i, yy in enumerate(y):
        ax.text(ref[i] + 2, yy + h / 2, f"{ref[i]:.0f}", color=INK2, fontsize=7,
                va="center")
        ax.text(mg[i] + 2, yy - h / 2, f"{mg[i]:.0f}", color=INK2, fontsize=7,
                va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("耗时 (ms)", color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    _fig_save(fig, "test12_1_time.png")


def _plot1_sweep_curves(results, lattice):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, xlabel, title in [
        (axes[0], "restart", "V-cycle 频率 r", "V-cycle 频率扫描 (ct=1e5, cmi=15)"),
        (axes[1], "ct", "最粗层容差因子 ct", "最粗层收敛条件扫描 (r=10, cmi=15)"),
        (axes[2], "cmi", "最粗层最大迭代 cmi", "最粗层迭代上限扫描 (r=10, ct=1e5)"),
    ]:
        _style(ax)
        pts = [r for r in results if tuple(r["lattice"]) == lattice
               and r["precision"] == "c64" and r["levels"] == 2
               and r.get("speedup_min") is not None]
        seen = {}
        for r in pts:
            if key == "restart" and (r["ct"] != 1e5 or r["cmi"] != 15):
                continue
            if key == "ct" and (r["restart"] != 10 or r["cmi"] != 15):
                continue
            if key == "cmi" and (r["restart"] != 10 or r["ct"] != 1e5):
                continue
            seen[r[key]] = r["speedup_min"]
        xs = sorted(seen)
        ax.plot(xs, [seen[x] for x in xs], color=C["blue"], marker="o",
                ms=5, lw=2)
        for x in xs:
            ax.annotate(f"{seen[x]:.2f}", (x, seen[x]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=INK2)
        ax.set_xlabel(xlabel, color=INK2)
        ax.set_ylabel("加速比", color=INK2)
        ax.set_title(title, color=INK, fontsize=10)
        if key == "ct":
            ax.set_xscale("log")
        ax.axhline(1.0, color=MUTED, lw=1, ls="--")
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.grid(False, axis="x")
    fig.tight_layout()
    _fig_save(fig, "test12_1_sweep.png")


def cmd_plots1(args):
    path = args.file
    if not os.path.exists(path):
        print(f"[error] {path} 不存在，先运行 sweep")
        sys.exit(2)
    data = json.load(open(path))
    results = data["results"]
    if not results:
        print("no results yet")
        return
    lattice = tuple(args.lattice) if args.lattice else tuple(data.get("lattice", (8, 16, 16, 16)))
    _plot1_conv(results)
    _plot1_hotspot(results)
    _plot1_speedup(results)
    _plot1_time(results)
    _plot1_sweep_curves(results, lattice)


# ----------------------------------------------------------------------
# layout_test —— 布局对照实验（原 mg_dev74_layout_test.py）
# ----------------------------------------------------------------------
def cmd_layout_test(args):
    Lx, Ly, Lz, Lt = 8, 8, 8, 16
    MASS = 0.05
    KAPPA = 1.0 / (2 * MASS + 8)
    ATOL = 1e-6

    U_full, b_full, clover, KAPPA, av, (g, fi, ce, coo, cei, coi) = setup_gpu(
        Lx, Ly, Lz, Lt, MASS, ATOL=ATOL)
    op = dslash.operator(U=U_full, clover_term=clover,
                         kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S_py = op.matvec_parity
    ls_odd = [Lx, Ly, Lz, Lt // 2]
    torch.manual_seed(7)
    x_o = torch.randn([12] + ls_odd, dtype=torch.complex64, device="cuda")
    y_py = S_py(x_o)
    print(f"Python S(x_o): shape={tuple(y_py.shape)} norm={float(tools.norm(y_py)):.6e}")

    dt = define.dtype(define._LAT_C64_)
    params[define._SET_INDEX_] = 0
    params[define._SET_PLAN_] = -1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 0
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 2
    params[define._PARITY_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)
    params[define._SET_INDEX_] += 1
    params[define._SET_PLAN_] = 1
    qcu.applyInitQcu(set_ptrs, params, av)

    y_cpp = torch.zeros_like(x_o)
    qcu.applyCloverBistabCgDslashQcu(y_cpp, x_o, g, ce, coo, cei, coi,
                                     set_ptrs, params)
    torch.cuda.synchronize()
    err1 = float(tools.norm(y_cpp - y_py) / tools.norm(y_py))
    print(f"try1 [12,XYZT/2]: rel_err = {err1:.6e}")

    x_po = torch.stack([torch.zeros_like(x_o), x_o])
    y_po = torch.zeros_like(x_po)
    qcu.applyCloverBistabCgDslashQcu(y_po, x_po, g, ce, coo, cei, coi,
                                     set_ptrs, params)
    torch.cuda.synchronize()
    err2 = float(tools.norm(y_po[1] - y_py) / tools.norm(y_py))
    print(f"try2 [2,12,XYZT/2] p=1: rel_err = {err2:.6e}")

    x_po0 = torch.stack([x_o, torch.zeros_like(x_o)])
    y_po0 = torch.zeros_like(x_po0)
    qcu.applyCloverBistabCgDslashQcu(y_po0, x_po0, g, ce, coo, cei, coi,
                                     set_ptrs, params)
    torch.cuda.synchronize()
    err3 = float(tools.norm(y_po0[0] - y_py) / tools.norm(y_py))
    print(f"try3 [2,12,XYZT/2] p=0: rel_err = {err3:.6e}")

    qcu.applyEndQcu(set_ptrs, params)


# ----------------------------------------------------------------------
# stencil_mt —— 多线程 stencil build 对照（原 mg_dev74_stencil_mt.py）
# ----------------------------------------------------------------------
def probe_point(S, lonv, E, ee, c_idx, sit, hop_nn, hop_diag, dims, Nc):
    """单点探测：(c_idx, ee) 处的 33-tensor 耦合。写集互不相交，可并行。"""
    Xc, Yc, Zc, Tc = dims
    str_Y, str_Z = Yc * Zc * Tc, Zc * Tc
    cx = c_idx // str_Y; rem = c_idx % str_Y
    cy = rem // str_Z; rem %= str_Z
    cz = rem // Tc; ct = rem % Tc
    ccoords = [cx, cy, cz, ct]
    src_c = torch.zeros([E, Xc, Yc, Zc, Tc], dtype=sit.dtype, device=sit.device)
    src_c[ee, cx, cy, cz, ct] = 1.0
    f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=src_c)
    dc = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S(f))
    sit[:, ee, cx, cy, cz, ct] = dc[:, cx, cy, cz, ct]
    for d in range(4):
        b = ccoords[:]; b[d] = (b[d] - 1 + dims[d]) % dims[d]
        fwd = ccoords[:]; fwd[d] = (fwd[d] + 1) % dims[d]
        if b[d] == fwd[d]:
            hop_nn[0, d, :, ee, b[0], b[1], b[2], b[3]] = 0.5 * dc[:, b[0], b[1], b[2], b[3]]
            hop_nn[1, d, :, ee, fwd[0], fwd[1], fwd[2], fwd[3]] = 0.5 * dc[:, fwd[0], fwd[1], fwd[2], fwd[3]]
        else:
            hop_nn[0, d, :, ee, b[0], b[1], b[2], b[3]] = dc[:, b[0], b[1], b[2], b[3]]
            hop_nn[1, d, :, ee, fwd[0], fwd[1], fwd[2], fwd[3]] = dc[:, fwd[0], fwd[1], fwd[2], fwd[3]]
    for pi, (d1, d2) in enumerate(PAIRS):
        targets = {}
        for s1i, s1 in enumerate(SIGN):
            for s2i, s2 in enumerate(SIGN):
                n = ccoords[:]
                n[d1] = (n[d1] - s1 + dims[d1]) % dims[d1]
                n[d2] = (n[d2] - s2 + dims[d2]) % dims[d2]
                key = (n[0], n[1], n[2], n[3])
                targets.setdefault(key, []).append((s1i, s2i))
        for key, combos in targets.items():
            w = 1.0 / len(combos)
            for (s1i, s2i) in combos:
                hop_diag[s1i, s2i, pi, :, ee, key[0], key[1], key[2], key[3]] = w * dc[:, key[0], key[1], key[2], key[3]]


def build_stencil_mt(S_ops, lonv, E, E_prev, lat_fine_odd, lat_coarse_odd,
                     dt, device, nthreads=4, verbose=True):
    """多线程 33-tensor stencil build。S_ops: CudaSchurOp 列表（每线程一个）。"""
    from concurrent.futures import ThreadPoolExecutor
    Xc, Yc, Zc, Tc = lat_coarse_odd
    Nc = Xc * Yc * Zc * Tc
    dims = [Xc, Yc, Zc, Tc]
    sit = torch.zeros([E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_nn = torch.zeros([2, 4, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_diag = torch.zeros([2, 2, 6, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    t0 = time.perf_counter()
    chunk = (Nc + nthreads - 1) // nthreads

    def worker(tid):
        op = S_ops[tid % len(S_ops)]
        c0 = tid * chunk
        c1 = min(Nc, c0 + chunk)
        for c_idx in range(c0, c1):
            for ee in range(E):
                probe_point(op.matvec, lonv, E, ee, c_idx, sit,
                            hop_nn, hop_diag, dims, Nc)

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(worker, range(nthreads)))
    dt_build = time.perf_counter() - t0
    if verbose:
        print(f"  [stencil_mt] {nthreads} threads: {dt_build:.1f}s for "
              f"{E * Nc} probes ({E * Nc / max(dt_build, 1e-9):.0f} probes/s)")
    return hop_nn, hop_diag, sit


def cmd_stencil_mt(args):
    Lx, Ly, Lz, Lt = 8, 8, 8, 16
    E, nthreads = 48, args.threads
    MASS, ATOL = 0.05, 1e-6
    KAPPA = 1.0 / (2 * MASS + 8)
    U_full, b_full, clover, KAPPA, av, (g, fi, ce, coo, cei, coi) = setup_gpu(
        Lx, Ly, Lz, Lt, MASS, ATOL=ATOL)
    op = dslash.operator(U=U_full, clover_term=clover,
                         kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S_py = op.matvec_parity
    dt = torch.complex64
    device = torch.device('cuda')
    lat_fine_odd = [Lx, Ly, Lz, Lt // 2]
    lat_coarse_odd = [Lx // 2, Ly // 2, Lz // 2, Lt // 4]
    torch.manual_seed(42)
    _null = torch.randn([E, 12] + lat_fine_odd, dtype=dt, device=device)
    _null = tools.give_null_vecs(null_vecs=_null, matvec=S_py,
                                 bistabcg=None, verbose=False)
    lonv = tools.local_orthogonalize(null_vecs=_null,
                                     coarse_lat_size=lat_coarse_odd,
                                     verbose=False)

    t0 = time.perf_counter()
    hop_nn_py, hop_diag_py, sit_py = build_stencil(
        S_py, lonv, E, 12, lat_fine_odd, lat_coarse_odd, dt, device)
    t_py = time.perf_counter() - t0

    ops = make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=nthreads)
    t0 = time.perf_counter()
    hop_nn, hop_diag, sit = build_stencil_mt(
        ops, lonv, E, 12, lat_fine_odd, lat_coarse_odd, dt, device,
        nthreads=nthreads)
    t_mt = time.perf_counter() - t0
    for o in ops:
        o.release()

    err = {
        "sit": float(tools.norm(sit - sit_py) / tools.norm(sit_py)),
        "hop_nn": float(tools.norm(hop_nn - hop_nn_py) / tools.norm(hop_nn_py)),
        "hop_diag": float(tools.norm(hop_diag - hop_diag_py) / tools.norm(hop_diag_py)),
    }
    torch.manual_seed(3)
    v = torch.randn([E] + lat_coarse_odd, dtype=dt, device=device)
    a_py = apply_stencil(hop_nn_py, hop_diag_py, sit_py, v)
    a_mt = apply_stencil(hop_nn, hop_diag, sit, v)
    err_st = float(tools.norm(a_mt - a_py) / tools.norm(a_py))
    f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=v)
    a_op = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S_py(f))
    err_op = float(tools.norm(a_mt - a_op) / tools.norm(a_op))

    res = {"lattice": [Lx, Ly, Lz, Lt], "E": E, "nthreads": nthreads,
           "t_py_s": t_py, "t_mt_s": t_mt,
           "speedup": t_py / t_mt,
           "tensor_err": err, "stencil_err": err_st, "vs_operator_free": err_op}
    print(json.dumps(res, indent=2))
    with open(os.path.join(WORKDIR, "test12_stencil_mt.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {WORKDIR}/test12_stencil_mt.json")


# ----------------------------------------------------------------------
# 子命令分派
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="test12 —— dev74* 整合测试套件")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("clean", help="干净测量 + 资源统计（独立进程）")
    p.add_argument("--lattice", nargs=4, type=int, default=[8, 16, 16, 16])
    p.add_argument("--prec", default="c64", choices=["c64", "c128"])
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--dof", nargs="+", type=int, default=None)
    p.add_argument("--restart", type=int, default=10)
    p.add_argument("--ct", type=float, default=1e5)
    p.add_argument("--cmi", type=int, default=15)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--build", default="py", choices=["py", "cpp"])
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("bench", help="批量基准（预算自动跳过超限配置）")
    p.add_argument("--mode", default="local", choices=["local", "server"])
    p.add_argument("--vram", type=int, default=16, help="显存档（GB），默认 16")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--build", default="py", choices=["py", "cpp"])
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("verify", help="正确性验证")
    p.add_argument("--lattice", nargs=4, type=int, default=[8, 8, 8, 16])
    p.add_argument("--prec", default="c64", choices=["c64", "c128"])
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("sweep", help="参数扫描（r/ct/cmi/levels × speedup）")
    p.add_argument("--lattice", nargs=4, type=int, default=[8, 8, 8, 16])
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--gate", type=float, default=1.5)
    p.add_argument("--timeout", type=int, default=1800,
                   help="每配置子进程超时（秒），防卡壳")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("check", help="加速比断言（exit 0/1/2）")
    p.add_argument("--gate", type=float, default=1.5)
    p.add_argument("--label", default="test12 check")
    p.add_argument("--file", action="append", default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("budget", help="预算表（默认 16G 档，--vram 32 预留）")
    p.add_argument("--mode", default="server", choices=["local", "server"])
    p.add_argument("--vram", type=int, default=16, help="显存档（GB）：16 默认 / 32 预留")
    p.add_argument("--fit", action="store_true")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("collect", help="汇总 → test12_results.json")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("mktable", help="LaTeX 表 → test12_tbl_*.tex")
    p.add_argument("--mode", default="server", choices=["local", "server"])
    p.add_argument("--vram", type=int, default=16)
    p.set_defaults(func=cmd_mktable)

    p = sub.add_parser("plots", help="dev74 风格图 → test12_*.png")
    p.add_argument("--vram", type=int, default=16)
    p.set_defaults(func=cmd_plots)

    p = sub.add_parser("plots1", help="dev74_1 风格图（dev73_5 范围）→ test12_1_*.png")
    p.add_argument("--file", default=os.path.join(WORKDIR, "test12_sweep.json"))
    p.add_argument("--lattice", nargs=4, type=int, default=None)
    p.set_defaults(func=cmd_plots1)

    p = sub.add_parser("layout_test", help="C++ Schur 算子布局对照实验")
    p.set_defaults(func=cmd_layout_test)

    p = sub.add_parser("stencil_mt", help="多线程 stencil build 对照")
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=cmd_stencil_mt)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
