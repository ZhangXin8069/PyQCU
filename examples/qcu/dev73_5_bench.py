#!/usr/bin/env python3
"""dev73_5 综合性能基准：CUDA-C++ Clover MultiGrid 求解器（git tag dev73_4）。

对给定 lattice / precision / 求解器参数：
  * 参考求解器 : applyCloverBistabCgQcu（奇偶预条件 Clover BiStabCG）
  * MultiGrid   : applyCloverMultigridQcu（Schur-一致 MG，多层粗层）

同时收集：
  - 收敛残差轨迹（MG 的 CONVERGENCE_HISTORY，BiStabCG 走 instrumented lib）
  - 计算热点（PROF_SECTIONS：fine_iter / vcycle / coarse_solve / coarse_vec / coarse_dslash）
  - 正确性：gauge SU(3) 性质、解的误差（vs_ref 与全算子残差）、null_vecs 正交性/零模性质/粗算子一致性

用法:
    source ./env.sh
    python examples/qcu/dev73_5_bench.py --lattice 8 16 16 16 --dtype c64 --sweeps base
    python examples/qcu/dev73_5_bench.py --lattice 8 16 16 16 --dtype c64 --sweeps restart,coarse,maxiter,levels,base
    python examples/qcu/dev73_5_bench.py --lattice 16 16 16 16 --dtype c64 --sweeps base
    python examples/qcu/dev73_5_bench.py --lattice 8 16 16 16 --dtype c128 --sweeps base
"""
import torch, os, sys, time, json, re, argparse, math
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
_csm = _load("csm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "conftest.schur.multigrid.py"))
build_config = _csm.build_config
from mg_nullvec_cache import build_or_load_coarse_ops
from mg_stencil_build import apply_stencil

REPO = "/root/PyQCU"
LOG_DIR = os.path.join(REPO, "logs", "dev73_5")
os.makedirs(LOG_DIR, exist_ok=True)
# C++ 后端把日志写到 cwd 下的 logs/clover_multigrid.log；我们从 /root/PyQCU 运行
MGR_LOG = os.path.join(REPO, "logs", "clover_multigrid.log")


# =====================================================================
# 系统搭建：gauge + clover（C++ 后端），并返回 Python 全布局参考张量
# =====================================================================
def setup_system(Lx, Ly, Lz, Lt, MASS, ATOL, DT, gauge_seed=42):
    KAPPA = 1.0/(2*MASS+8)
    av = build_config(Lx, Ly, Lz, Lt, MASS, ATOL, 1, [12], [2,2,2,2],
                      10, 200, 1e4, DT)
    device = torch.device('cuda'); dt = define.dtype(DT); ls = define.lat_shape(params)
    torch.manual_seed(gauge_seed)
    g = torch.zeros([2,3,3,4]+ls, dtype=dt, device=device)
    fi = torch.randn([2,4,3]+ls, dtype=dt, device=device)
    fo_ref = torch.zeros_like(fi)
    ce = torch.zeros([4,3,4,3]+ls, dtype=dt, device=device)
    cei = torch.zeros_like(ce); coo = torch.zeros_like(ce); coi = torch.zeros_like(ce)

    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyGaussGaugeQcu(g, set_ptrs, params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyCloversQcu(ce, cei, g, set_ptrs, params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyCloversQcu(coo, coi, g, set_ptrs, params)

    qcu_U = tools.poooxyzt2oooxyzt(g)           # [3,3,4,Lx,Ly,Lz,Lt]
    qcu_src = tools.poooxyzt2oooxyzt(fi)        # [4,3,Lx,Ly,Lz,Lt]
    ref_cl = dslash.make_clover(qcu_U, kappa=KAPPA)
    return dict(Lx=Lx, Ly=Ly, Lz=Lz, Lt=Lt, MASS=MASS, ATOL=ATOL, DT=DT,
                KAPPA=KAPPA, av=av, dt=dt, device=device,
                g=g, fi=fi, ce=ce, coo=coo, cei=cei, coi=coi,
                qcu_U=qcu_U, qcu_src=qcu_src, ref_cl=ref_cl, gauge_seed=gauge_seed)


# =====================================================================
# 参考 BiStabCG（C++ 后端）：计时 + 解
# =====================================================================
def run_bistabcg_ref(S, ntrials=3, verbose=True):
    g, fi, ce, coo, cei, coi, av = S['g'], S['fi'], S['ce'], S['coo'], S['cei'], S['coi'], S['av']
    fo_ref = torch.zeros_like(fi)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs, params, av)
    times = []
    for _ in range(ntrials):
        fo_ref.zero_()
        torch.cuda.synchronize(); t0=time.perf_counter()
        qcu.applyCloverBistabCgQcu(fo_ref, fi, g, ce, coo, cei, coi, set_ptrs, params)
        torch.cuda.synchronize(); times.append(time.perf_counter()-t0)
    ref_time = min(times)
    qcu_ref = tools.poooxyzt2oooxyzt(fo_ref)
    # 全算子残差 |D x - b|/|b|
    full_res = tools.norm(dslash.give_wilson(qcu_ref, S['qcu_U'], S['KAPPA'], True) +
                          dslash.give_clover(qcu_ref, S['ref_cl']) - S['qcu_src']) / tools.norm(S['qcu_src'])
    if verbose:
        print(f"[BiStabCG] ref_time={ref_time*1000:.1f}ms full_res={full_res:.3e}")
    return dict(fo_ref=fo_ref, qcu_ref=qcu_ref, ref_time=ref_time, full_res=float(full_res))


# =====================================================================
# gauge SU(3) 性质检查
# =====================================================================
def gauge_checks(S):
    from pyqcu import lattice
    U = S['qcu_U']  # [3,3,4,Lx,Ly,Lz,Lt]
    U_mat = U.permute(*range(2, U.ndim), 0, 1).reshape(-1, 3, 3)
    N = U_mat.shape[0]
    UH_U = torch.matmul(U_mat.conj().transpose(-1,-2), U_mat)
    eye = torch.eye(3, dtype=U_mat.dtype, device=U_mat.device).expand(N,-1,-1)
    max_unitary = (UH_U - eye).abs().max().item()
    det_U = torch.linalg.det(U_mat)
    max_det = (det_U - 1).abs().max().item()
    Uf = U_mat.reshape(N, 9)
    c6 = (Uf[:,1]*Uf[:,5]-Uf[:,2]*Uf[:,4]).conj()
    c7 = (Uf[:,2]*Uf[:,3]-Uf[:,0]*Uf[:,5]).conj()
    c8 = (Uf[:,0]*Uf[:,4]-Uf[:,1]*Uf[:,3]).conj()
    max_minor = max((Uf[:,6]-c6).abs().max().item(), (Uf[:,7]-c7).abs().max().item(),
                    (Uf[:,8]-c8).abs().max().item())
    ok = lattice.check_su3(U, tol=1e-3, verbose=False)
    return dict(n_links=N, su3_ok=bool(ok), max_unitary_err=max_unitary,
                max_det_err=max_det, max_minor_err=max_minor)


# =====================================================================
# null_vecs 正确性检查（每层）
# =====================================================================
def nullvec_checks(S, lvl, lonv, hnn, hdg, sit, E, E_prev, lat_fine_odd, lat_coarse_odd):
    Xc,Yc,Zc,Tc = lat_coarse_odd; Nc = Xc*Yc*Zc*Tc
    Xx,Yy,Zz,Tt = lat_fine_odd
    X,Y,Z,T = Xc,Yc,Zc,Tc
    x,y,z,t = Xx//X, Yy//Y, Zz//Z, Tt//T
    local_dim = E_prev * x * y * z * t
    res = {}
    # lonv 为 local_orthogonalize 输出（10 维块布局 [E,e,X,x,Y,y,Z,z,T,t]），
    # 其内存顺序与 6 维 fine 布局 [E,e,Xx,Yy,Zz,Tt] 等价（C++ restrict/prolong 与
    # Schur 算子 S 均按 6 维布局索引）。
    lonv6 = lonv.view(E, E_prev, Xx, Yy, Zz, Tt)
    # (1) 零模性质：||S·P|| / ||P||  (Schur 算子作用于每个零模向量的残差)
    ratios = []
    sp_parts = []
    for i in range(E):
        svi = S(lonv6[i].contiguous())
        ratios.append(float(tools.norm(svi))/float(tools.norm(lonv6[i])))
        sp_parts.append(svi)
    Sp = torch.stack(sp_parts, dim=0)                # [E, E_prev, Xx,Yy,Zz,Tt]
    nr = tools.norm(Sp)/tools.norm(lonv)
    res['null_res_ratio'] = float(nr)
    res['null_res_min'] = min(ratios)
    res['null_res_max'] = max(ratios)
    # 参考：S 的最大本征值（幂迭代）—— 用于判断零模向量是否显著捕获低模
    vp = torch.randn([E_prev, Xx, Yy, Zz, Tt], dtype=lonv.dtype, device=lonv.device)
    vp = vp/torch.linalg.norm(vp)
    lam = 0.0
    for _ in range(20):
        wp = S(vp)
        lam = torch.real(torch.vdot(wp.flatten(), vp.flatten()))
        vp = wp/torch.linalg.norm(wp)
    res['schur_lambda_max'] = float(abs(lam))
    # (2) 正交性：每粗块内 P^H P ≈ I（与 local_orthogonalize 相同的块重排）
    v = lonv.permute(2,4,6,8,0,1,3,5,7,9).contiguous().view(Nc, E, local_dim)
    gram = torch.einsum('bce,bde->bcd', v.conj(), v)  # [Nc, E, E] 每块 Gram
    I = torch.eye(E, dtype=gram.dtype, device=gram.device)
    res['gram_vs_I_max'] = float((gram - I).abs().max().item())
    res['gram_diag_min'] = float(gram.diagonal(dim1=-2,dim2=-1).abs().min().item())
    res['gram_offdiag_max'] = float((gram - gram.diagonal(dim1=-2,dim2=-1).diag_embed()).abs().max().item())
    # (3) 粗算子一致性：apply_stencil 与 restrict(S(prolong(v))) 对比
    errs = []
    for _ in range(2):
        v = torch.randn([E, Xc, Yc, Zc, Tc], dtype=lonv.dtype, device=lonv.device)
        A_st = apply_stencil(hnn, hdg, sit, v)
        f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=v)
        A_op = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S(f))
        errs.append(float(tools.norm(A_st-A_op))/float(tools.norm(A_op)))
    res['stencil_rel_err'] = max(errs)
    return res


# =====================================================================
# 构造/加载粗算子，设置 set_ptrs
# =====================================================================
def build_coarse_levels(S, NUM_LEVELS, DOF_LIST, MG_GRID, NV_ITERS, verbose=True):
    Lx,Ly,Lz,Lt = S['Lx'],S['Ly'],S['Lz'],S['Lt']
    dt, device, gauge_seed = S['dt'], S['device'], S['gauge_seed']
    op = dslash.operator(U=S['qcu_U'], clover_term=S['ref_cl'],
                         kappa=torch.Tensor([S['KAPPA']]), support_parity=True, verbose=False)
    Sc = op.matvec_parity
    lat_fine_odd = [Lx, Ly, Lz, Lt//2]
    E_prev = 12
    lonvs, hnn_l, hdg_l, sit_l = [], [], [], []
    checks = []
    for lvl in range(1, NUM_LEVELS):
        E_c = DOF_LIST[lvl]
        lat_coarse_odd = [lat_fine_odd[d]//MG_GRID[d] for d in range(4)]
        lonv, hnn, hdg, sit = build_or_load_coarse_ops(
            gauge_seed, [Lx,Ly,Lz,Lt], lvl, E_c, E_prev, lat_fine_odd,
            lat_coarse_odd, Sc, dt, device, NV_ITERS, use_cache=True,
            save=True, verbose=verbose)
        set_ptrs[30+4*(lvl-1)+0] = lonv.contiguous().data_ptr()
        set_ptrs[30+4*(lvl-1)+1] = hnn.contiguous().data_ptr()
        set_ptrs[30+4*(lvl-1)+2] = hdg.contiguous().data_ptr()
        set_ptrs[30+4*(lvl-1)+3] = sit.contiguous().data_ptr()
        # 记录每层 null_vecs 检查（第一层用 Schur 算子，后续用 materialized A_c）
        if verbose:
            nc = nullvec_checks(Sc, lvl, lonv, hnn, hdg, sit, E_c, E_prev,
                                lat_fine_odd, lat_coarse_odd)
            nc['level'] = lvl; nc['E'] = E_c
            nc['lat_fine_odd'] = list(lat_fine_odd); nc['lat_coarse_odd'] = list(lat_coarse_odd)
            checks.append(nc)
            print(f"  [nullvec lvl{lvl}] null_res={nc['null_res_ratio']:.3e} "
                  f"(min={nc['null_res_min']:.3e} max={nc['null_res_max']:.3e}) "
                  f"lambda_max={nc['schur_lambda_max']:.3e} "
                  f"gram_I={nc['gram_vs_I_max']:.3e} stencil_err={nc['stencil_rel_err']:.3e}")
        lonvs.append(lonv); hnn_l.append(hnn); hdg_l.append(hdg); sit_l.append(sit)
        # 下一层的 fine 算子 = materialized A_c
        def make_A(hnn_i, hdg_i, sit_i):
            def A(v): return apply_stencil(hnn_i, hdg_i, sit_i, v)
            return A
        Sc = make_A(hnn, hdg, sit)
        E_prev = E_c
        lat_fine_odd = lat_coarse_odd
    return dict(lonvs=lonvs, hnn_l=hnn_l, hdg_l=hdg_l, sit_l=sit_l, checks=checks)


# =====================================================================
# 运行一个 MG 配置：计时 + 收敛 + 热点 + 误差
# =====================================================================
def run_mg_config(S, label, NUM_LEVELS, DOF_LIST, MG_GRID, NUM_RESTART,
                  COARSE_MAX_ITER, COARSE_TOL_FACTOR, ntrials=3, verbose=True):
    av = build_config(S['Lx'],S['Ly'],S['Lz'],S['Lt'],S['MASS'],S['ATOL'],
                      NUM_LEVELS, DOF_LIST, MG_GRID, NUM_RESTART,
                      COARSE_MAX_ITER, COARSE_TOL_FACTOR, S['DT'])
    g, fi = S['g'], S['fi']
    fo_mg = torch.zeros_like(fi)
    # 清空 MG 日志
    open(MGR_LOG, 'w').close()
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs, params, av)
    times = []
    for _ in range(ntrials):
        fo_mg.zero_()
        torch.cuda.synchronize(); t0=time.perf_counter()
        qcu.applyCloverMultigridQcu(fo_mg, fi, g, S['ce'], S['coo'], S['cei'], S['coi'],
                                    set_ptrs, params)
        torch.cuda.synchronize(); times.append(time.perf_counter()-t0)
    mg_time = min(times)
    qcu_mg = tools.poooxyzt2oooxyzt(fo_mg)
    mg_full_res = tools.norm(dslash.give_wilson(qcu_mg, S['qcu_U'], S['KAPPA'], True) +
                             dslash.give_clover(qcu_mg, S['ref_cl']) - S['qcu_src']) / tools.norm(S['qcu_src'])
    mg_vs_ref = tools.norm(qcu_mg - S['qcu_ref'])/tools.norm(S['qcu_ref'])
    # 解析日志
    conv = []
    prof = {}
    raw = open(MGR_LOG).read() if os.path.exists(MGR_LOG) else ""
    m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', raw)
    if m: conv = [float(x) for x in m.group(1).split(",") if x.strip()]
    def _num(s):
        return float(re.sub(r'[^0-9.eE+-]', '', s)) if re.sub(r'[^0-9.eE+-]', '', s) else 0.0
    m = re.search(r'PROF_SECTIONS:\s*(.*)', raw)
    if m:
        for kv in m.group(1).strip().split():
            if '=' in kv:
                k,v = kv.split('='); prof[k] = _num(v)
    m = re.search(r'PROF_COARSE:\s*(.*)', raw)
    if m:
        for kv in m.group(1).strip().split():
            if '=' in kv:
                k,v = kv.split('='); prof[k] = _num(v)
    iters = len([c for c in conv if c > S['ATOL']])
    speedup = S['ref_time']/mg_time if mg_time > 0 else 0
    # 保存本配置的原始 MG 日志副本
    safe = re.sub(r'[^0-9A-Za-z_]', '_', label)
    try:
        import shutil
        shutil.copy(MGR_LOG, os.path.join(LOG_DIR, f"mg_{safe}_{S['Lx']}x{S['Ly']}x{S['Lz']}x{S['Lt']}_{'c64' if S['DT']==define._LAT_C64_ else 'c128'}.log"))
    except Exception:
        pass
    res = dict(label=label, lattice=[S['Lx'],S['Ly'],S['Lz'],S['Lt']],
               mass=S['MASS'], atol=S['ATOL'], dt_name='c64' if S['DT']==define._LAT_C64_ else 'c128',
               levels=NUM_LEVELS, dof=list(DOF_LIST), mg_grid=list(MG_GRID),
               restart=NUM_RESTART, coarse_max_iter=COARSE_MAX_ITER,
               coarse_tol_factor=COARSE_TOL_FACTOR,
               ref_ms=S['ref_time']*1000, mg_ms=mg_time*1000, speedup=speedup,
               iters=iters, final_res=conv[-1] if conv else None,
               mg_full_res=float(mg_full_res), vs_ref=float(mg_vs_ref),
               conv=conv, prof=prof)
    if verbose:
        print(f"[{label}] ref={res['ref_ms']:.0f}ms mg={res['mg_ms']:.0f}ms "
              f"speedup={res['speedup']:.3f}x iters={iters} final_res={res['final_res']:.2e} "
              f"full_res={mg_full_res:.2e} vs_ref={mg_vs_ref:.2e} "
              f"prof={ {k:(round(v,1) if k!='n_vcycles' else int(v)) for k,v in prof.items()} }")
    return res


# =====================================================================
# 主流程
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', nargs=4, type=int, default=[8,16,16,16])
    ap.add_argument('--dtype', default='c64', choices=['c64','c128'])
    ap.add_argument('--mass', type=float, default=0.05)
    ap.add_argument('--atol', type=float, default=1e-6)
    ap.add_argument('--sweeps', default='base',
                    help='comma-separated: base,restart,coarse,maxiter,levels')
    ap.add_argument('--ntrials', type=int, default=3)
    ap.add_argument('--nv_iters', type=int, default=2)
    args = ap.parse_args()
    Lx,Ly,Lz,Lt = args.lattice
    DT = define._LAT_C128_ if args.dtype == 'c128' else define._LAT_C64_
    sweeps = set(s.strip() for s in args.sweeps.split(',') if s.strip())

    out = {}
    out['meta'] = dict(tag='dev73_4', lattice=[Lx,Ly,Lz,Lt], dtype=args.dtype,
                       mass=args.mass, atol=args.atol, ntrials=args.ntrials,
                       nv_iters=args.nv_iters,
                       torch=torch.__version__,
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')

    print("="*72)
    print(f"=== dev73_5 bench: lattice={Lx}x{Ly}x{Lz}x{Lt} dtype={args.dtype} mass={args.mass} atol={args.atol}")
    print("="*72)

    S = setup_system(Lx,Ly,Lz,Lt,args.mass,args.atol,DT)
    out['gauge'] = gauge_checks(S)
    print(f"[gauge] su3_ok={out['gauge']['su3_ok']} max_unitary={out['gauge']['max_unitary_err']:.3e} "
          f"max_det={out['gauge']['max_det_err']:.3e} max_minor={out['gauge']['max_minor_err']:.3e}")

    ref = run_bistabcg_ref(S, ntrials=args.ntrials)
    S['ref_time'] = ref['ref_time']; S['qcu_ref'] = ref['qcu_ref']
    out['bistabcg'] = dict(ref_ms=ref['ref_time']*1000, full_res=ref['full_res'])

    results = []
    dof_base = [12, 48]
    dof_3l = [12, 48, 48]
    mg_grid = [2,2,2,2]
    # 与 mg_v4_bench.py 一致：restart=10, coarse_max_iter=15, coarse_tol_factor=1e5
    base = dict(NUM_LEVELS=2, DOF_LIST=dof_base, MG_GRID=mg_grid,
                NUM_RESTART=10, COARSE_MAX_ITER=15, COARSE_TOL_FACTOR=1e5)

    # 先构造粗算子（缓存），并做 null_vecs 检查
    NV = build_coarse_levels(S, 3, dof_3l, mg_grid, args.nv_iters)
    out['nullvecs'] = NV['checks']

    def run(label, cfg):
        r = run_mg_config(S, label, cfg['NUM_LEVELS'], cfg['DOF_LIST'], cfg['MG_GRID'],
                          cfg['NUM_RESTART'], cfg['COARSE_MAX_ITER'],
                          cfg['COARSE_TOL_FACTOR'], ntrials=args.ntrials)
        results.append(r); return r

    # ---- base ----
    if 'base' in sweeps:
        run("base_2L", base)
        if args.dtype == 'c64':
            # 3 层 base（仅单精度，双精度可选）
            run("base_3L", dict(base, NUM_LEVELS=3, DOF_LIST=dof_3l))

    # ---- restart 扫描（进入下一层判断条件） ----
    if 'restart' in sweeps:
        for r in [5, 10, 20]:
            run(f"restart_{r}", dict(base, NUM_RESTART=r))

    # ---- coarse tol factor 扫描（最粗层收敛条件） ----
    if 'coarse' in sweeps:
        for ct in [1e2, 1e3, 1e4, 1e5]:
            run(f"ct_1e{int(round(math.log10(ct)))}", dict(base, COARSE_TOL_FACTOR=ct))

    # ---- coarse max iter 扫描（平滑器迭代上限） ----
    if 'maxiter' in sweeps:
        for cmi in [15, 50, 200]:
            run(f"cmi_{cmi}", dict(base, COARSE_MAX_ITER=cmi))

    # ---- 层数扫描 ----
    if 'levels' in sweeps:
        run("levels_2", dict(base, NUM_LEVELS=2, DOF_LIST=dof_base))
        run("levels_3", dict(base, NUM_LEVELS=3, DOF_LIST=dof_3l))

    out['results'] = results
    # 分离收敛轨迹与大 JSON（避免超大文件）
    out2 = dict(out); out2['results'] = [{k:(v if k!='conv' else None) for k,v in r.items()} for r in results]
    with open(os.path.join(LOG_DIR, f"dev73_5_{Lx}x{Ly}x{Lz}x{Lt}_{args.dtype}_results.json"), "w") as f:
        json.dump(out2, f, indent=2)
    conv_path = os.path.join(LOG_DIR, f"dev73_5_{Lx}x{Ly}x{Lz}x{Lt}_{args.dtype}_conv.json")
    with open(conv_path, "w") as f:
        json.dump({r['label']: r['conv'] for r in results}, f)
    print("\n=== SUMMARY ===")
    for r in sorted(results, key=lambda x: -x['speedup']):
        print(f"{r['label']}: speedup={r['speedup']:.3f}x  mg={r['mg_ms']:.0f}ms  ref={r['ref_ms']:.0f}ms  "
              f"iters={r['iters']}  full_res={r['mg_full_res']:.2e}  vs_ref={r['vs_ref']:.2e}")
    print(f"\nSaved -> {LOG_DIR}")


if __name__ == "__main__":
    main()
