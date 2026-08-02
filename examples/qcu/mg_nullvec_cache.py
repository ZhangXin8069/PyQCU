#!/usr/bin/env python3
"""null_vecs 缓存：保存/读取 multigrid 的零模（null vectors）与粗算子。

背景
----
null_vecs 只与 gauge 场一一对应（对同一个 gauge + 相同随机种子，逆迭代生成的
零模是确定性的）。生成一组零模 + 33-tensor 粗算子（stencil build）在
{8,16,16,16} 上要 ~5-10 分钟（主要花在 Python 端的逆迭代 BiStabCG 与
单点探测 stencil 上）。为免在参数扫描 / 性能对比时反复重复计算，这里把
  - lonv : local-orthogonalized null vectors  [E, e, Xf, Yf, Zf, Tf]
  - hnn  : 最近邻 hopping 张量               [2,4,E,E,Xc,Yc,Zc,Tc]
  - hdg  : 对角 hopping 张量                 [2,2,6,E,E,Xc,Yc,Zc,Tc]
  - sit  : on-site 块                        [E,E,Xc,Yc,Zc,Tc]
按 (gauge_seed, lattice, level, E, nv_iters, precision) 存成 .pt 文件。

默认开启（save=True/use_cache=True）。缓存目录通过 PYQCU_NULLVEC_CACHE 环境
变量覆盖，缺省为 <repo>/logs/nullvec_cache。
"""
import os, torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CACHE_DIR = os.environ.get("PYQCU_NULLVEC_CACHE",
                                   os.path.join(_REPO, "logs", "nullvec_cache"))

_KEYS = ["lonv", "hnn", "hdg", "sit"]


def cache_dir():
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    return DEFAULT_CACHE_DIR


def cache_tag(gauge_seed, lat_full, level, E, nv_iters, dt_name="c64"):
    """Construct a deterministic cache key from the gauge identity + MG config.

    NOTE: the naming matches the legacy build_schur_levels() cache
    (L{lat}x..._lv{level}_E{E}_nvi{nv_iters}) so already-cached files are
    reused.  null_vecs are generated from the gauge produced by the fixed
    C++ seed (gauge_seed=42) in all benchmarks; if the gauge changes the
    cache must be cleared (the key would not capture it).
    """
    L = "x".join(str(x) for x in lat_full)
    return f"L{L}_lv{level}_E{E}_nvi{nv_iters}_{dt_name}"


def load_coarse_ops(gauge_seed, lat_full, level, E, nv_iters, dt_name="c64",
                    device=torch.device("cuda")):
    """Load cached (lonv, hnn, hdg, sit) for a level.  Returns None if absent."""
    # Prefer the dtype-qualified key; fall back to the legacy dtype-less key
    # (created by the earlier build_schur_levels / v4 code) for c64 only —
    # a c128 run must NEVER reuse a c64 cache (wrong memory layout).
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
    """Persist (lonv, hnn, hdg, sit) for a level (CPU tensors for portability)."""
    tag = cache_tag(gauge_seed, lat_full, level, E, nv_iters, dt_name)
    d = cache_dir()
    for k, t in zip(_KEYS, [lonv, hnn, hdg, sit]):
        torch.save(t.detach().cpu(), os.path.join(d, tag + "_" + k + ".pt"))


def build_or_load_coarse_ops(gauge_seed, lat_full, level, E, E_prev,
                             lat_fine, lat_coarse, S, dt, device,
                             nv_iters=2, use_cache=True, save=True,
                             verbose=True):
    """Build null vectors + 33-tensor coarse operator for one level, or load
    from cache.  Returns (lonv, hnn, hdg, sit) on `device`.

    S       : the operator to build null vectors for (matvec callable).
    lat_fine: fine-lattice [X,Y,Z,T] the null vectors live on.
    """
    from mg_stencil_build import build_stencil
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
        from pyqcu import tools
        _null = tools.give_null_vecs(null_vecs=_null, matvec=S,
                                     bistabcg=None, verbose=False)
    lonv = tools.local_orthogonalize(null_vecs=_null,
                                     coarse_lat_size=lat_coarse, verbose=False)
    t0.record()
    hnn, hdg, sit = build_stencil(S, lonv, E, E_prev, lat_fine, lat_coarse,
                                  dt, device)
    t1.record(); torch.cuda.synchronize()
    if verbose:
        print(f"  [level {level}] E={E} built nv+stencil in "
              f"{t0.elapsed_time(t1)/1000:.1f}s coarse={lat_coarse}")
    if save:
        save_coarse_ops(gauge_seed, lat_full, level, E, nv_iters, dt_name,
                        lonv, hnn, hdg, sit)
    return lonv, hnn, hdg, sit
