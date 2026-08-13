#!/usr/bin/env python3
"""dev74 —— 多线程 stencil build（粗算子构建并行化）。

参考 cpp/cuda/qcu/python/pyqcu.h:applyCloverBistabCgDslashQcu：
单点探测 S(f)（Schur 奇偶算子）由 C++ CUDA 内核实现（实测比 Python
matvec_parity 快 ~10x），且各 (c_idx, ee) 探测互不依赖、写集不相交
（sit/hop_nn/hop_diag 的写入位置由 (c_idx, ee) 唯一确定），因此可以
多线程并行：每个 worker 持独立 CudaSchurOp（独立 set_index / LatticeSet
scratch），线程内串行探测自己分到的 c_idx 区间。

用法（正确性对照 + 性能对比）：
    source ./env.sh && python examples/qcu/mg_dev74_stencil_mt.py
输出：logs/dev74_stencil_mt.json（串行/多线程耗时、加速比、stencil 等价误差）
"""
import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor
import torch
from pyqcu import tools
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, set_ptrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_pyref_expt import setup_gpu
from mg_stencil_build import build_stencil, PAIRS, SIGN
from mg_dev74_dslash import make_cuda_schur_ops

LOG_DIR = os.path.expanduser("~/PyQCU/logs")


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
    """多线程 33-tensor stencil build。

    S_ops: CudaSchurOp 实例列表（每线程一个，长度 >= nthreads）。
    返回 (hop_nn, hop_diag, sit)。与 build_stencil（Python S 串行）数值一致。
    """
    Xc, Yc, Zc, Tc = lat_coarse_odd
    Nc = Xc * Yc * Zc * Tc
    dims = [Xc, Yc, Zc, Tc]
    sit = torch.zeros([E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_nn = torch.zeros([2, 4, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_diag = torch.zeros([2, 2, 6, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    t0 = time.perf_counter()
    # 按 c_idx 分块：每线程连续区间（缓存友好 + 写集天然分区）
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


def validate(Lx, Ly, Lz, Lt, E=48, nthreads=4):
    """对照：MT+C++ vs 串行 Python S 的 stencil 数值一致性 + 性能对比。"""
    MASS, ATOL = 0.05, 1e-6
    KAPPA = 1.0 / (2 * MASS + 8)
    U_full, b_full, clover, KAPPA, av, (g, fi, ce, coo, cei, coi) = setup_gpu(
        Lx, Ly, Lz, Lt, MASS, ATOL=ATOL)
    from pyqcu import dslash
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

    # ---- 串行 Python S 基准 ----
    t0 = time.perf_counter()
    hop_nn_py, hop_diag_py, sit_py = build_stencil(
        S_py, lonv, E, 12, lat_fine_odd, lat_coarse_odd, dt, device)
    t_py = time.perf_counter() - t0

    # ---- MT + C++ dslash ----
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
    # ---- stencil 本身等价性（同一 stencil 两套张量 → 同一算子）----
    from mg_stencil_build import apply_stencil
    torch.manual_seed(3)
    v = torch.randn([E] + lat_coarse_odd, dtype=dt, device=device)
    a_py = apply_stencil(hop_nn_py, hop_diag_py, sit_py, v)
    a_mt = apply_stencil(hop_nn, hop_diag, sit, v)
    err_st = float(tools.norm(a_mt - a_py) / tools.norm(a_py))
    # 与 operator-free 对照
    f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=v)
    a_op = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S_py(f))
    err_op = float(tools.norm(a_mt - a_op) / tools.norm(a_op))

    res = {"lattice": [Lx, Ly, Lz, Lt], "E": E, "nthreads": nthreads,
           "t_py_s": t_py, "t_mt_s": t_mt,
           "speedup": t_py / t_mt,
           "tensor_err": err, "stencil_err": err_st, "vs_operator_free": err_op}
    print(json.dumps(res, indent=2))
    return res


def main():
    res = validate(8, 8, 8, 16, E=48, nthreads=4)
    with open(os.path.join(LOG_DIR, "dev74_stencil_mt.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {LOG_DIR}/dev74_stencil_mt.json")


if __name__ == "__main__":
    main()
