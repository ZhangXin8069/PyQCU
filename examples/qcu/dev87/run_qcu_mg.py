"""dev87 PyQCU C++ MultiGrid 端到端运行器。

复用 data/ 统一 gauge、源 b 与 33-tensor stencil 缓存；支持 1/2/3-level、
GCR(FGMRES)、deflate 以及同一 RHS 的 warm start。3-level 只读取已有缓存，
缺缓存时立即失败，不自动触发长时间 setup。

示例（source ./env.sh 后）：
  python examples/qcu/dev87/run_qcu_mg.py --levels 1
  python examples/qcu/dev87/run_qcu_mg.py --levels 2 --E 12
  python examples/qcu/dev87/run_qcu_mg.py --lat 8 8 8 16 --levels 3 --E 24 --coarse-E 24
  python examples/qcu/dev87/run_qcu_mg.py --levels 2 --deflate --warm
  python examples/qcu/dev87/run_qcu_mg.py --levels 2 --gcr --mu-pre 4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QCU_LOG_DIR", str(Path(__file__).resolve().parents[2] / "logs" / "dev87"))
Path(os.environ["QCU_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ATOL_DEFAULT, LAT_DEFAULT, MASS_DEFAULT, load_gauge_h5,
                    load_stencil, pick_v100, save_result)
from pyqcu.cuda import define, qcu
from pyqcu.cuda._multi_gpu import _SET_PTRS_COARSE_BASE_

OUT = Path(__file__).resolve().parent / "out"


def _parse_last_history():
    """读取本次 C++ 调用写入的最后一条收敛历史（缺失时返回空）。"""
    import re

    log_file = Path(os.environ["QCU_LOG_DIR"]) / "clover_multigrid.log"
    if not log_file.exists():
        return []
    matches = re.findall(r"CONVERGENCE_HISTORY:\s*\[([^\]]*)\]",
                        log_file.read_text(errors="ignore"))
    if not matches:
        return []
    return [float(v) for v in matches[-1].split(",") if v.strip()]


def _full_residual_rel(x, b, g, mass, U=None, clover=None):
    """Python 参考实现计算 full Clover-Wilson 相对残差。"""
    from pyqcu import dslash, tools

    kappa = 1.0 / (2.0 * mass + 8.0)
    if U is None:
        U = tools.poooxyzt2oooxyzt(g)
    if clover is None:
        clover = dslash.make_clover(U, kappa=torch.tensor(
            [kappa], dtype=torch.float32))
    xf = tools.poooxyzt2oooxyzt(x)
    bf = tools.poooxyzt2oooxyzt(b)
    r = dslash.give_wilson(xf, U, kappa, True)
    r = r + dslash.give_clover(xf, clover) - bf
    return float(tools.norm(r) / max(tools.norm(bf), 1e-30))


def _slug(text):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=int, nargs=4, default=LAT_DEFAULT)
    ap.add_argument("--mass", type=float, default=MASS_DEFAULT)
    ap.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--E", type=int, default=12)
    ap.add_argument("--coarse-E", type=int, nargs="+", default=None,
                    help="level-2 及以后每层的 E；例如 3-level 用 --E 24 --coarse-E 24")
    ap.add_argument("--nvi", type=int, default=1)
    ap.add_argument("--mg-grid", type=int, nargs=4, default=[2, 2, 2, 2])
    ap.add_argument("--restart", type=int, default=5)
    ap.add_argument("--cmi", type=int, default=200)
    ap.add_argument("--ctf", type=float, default=3000.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--mu-pre", type=int, default=4)
    ap.add_argument("--gcr", action="store_true",
                    help="启用 C++ _MG_USE_GCR_ 路径（实现为 FGMRES(10)+MG 预条件）")
    ap.add_argument("--deflate", action="store_true",
                    help="启用一次 V-cycle 初始 deflation（仅非 GCR 主循环有意义）")
    ap.add_argument("--warm", action="store_true",
                    help="cold 求解后以 cold 解作为 x0 再运行一次 warm 求解")
    ap.add_argument("--label", default=None,
                    help="结果文件标签；默认按配置生成")
    ap.add_argument("--backend-verbose", action="store_true",
                    help="启用 C++ run_test；会把真残差检查计入调用时间")
    args = ap.parse_args()

    if args.levels < 1 or args.levels > 4:
        raise ValueError("--levels 必须在 1..4（params 协议最多支持 4 个粗层）")
    if any(v <= 0 for v in args.lat) or any(v <= 0 for v in args.mg_grid):
        raise ValueError("--lat 与 --mg-grid 必须为正整数")
    if args.levels > 1 and args.E <= 0:
        raise ValueError("--E 必须为正整数")
    coarse_es = list(args.coarse_E or [args.E] * max(0, args.levels - 2))
    if len(coarse_es) != max(0, args.levels - 2):
        raise ValueError("--coarse-E 的数量必须等于 levels-2")
    level_es = [args.E] + coarse_es if args.levels > 1 else []

    dev = pick_v100()
    print(f"[dev87-mg] device={torch.cuda.get_device_name(dev)}")
    lat = args.lat
    Lx, Ly, Lz, Lt = lat
    g = load_gauge_h5(lat, args.mass, device="cuda")
    coarse_assets = []
    for level, E in enumerate(level_es, start=1):
        # 只读既有缓存；load_stencil 不会生成文件。
        coarse_assets.append(load_stencil(lat, E, args.nvi, device="cuda",
                                          level=level))
        print(f"[dev87-mg] cache level={level} E={E} loaded", flush=True)

    # Clover 张量构建会返回新的参数/指针副本（SET_INDEX 已推进到 2），
    # 所以先完成它，再把 MG 配置写回最终会传给 solver 的 p/av。否则
    # 返回值会覆盖这里提前写入的 _MG_NUM_LEVEL_ 等字段。
    from common import make_clover_tensors
    ce, cei, coo, coi, s, p, av = make_clover_tensors(g, lat, args.mass)

    dt = define._LAT_C64_
    p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
    p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    p[define._GRID_X_] = p[define._GRID_Y_] = p[define._GRID_Z_] = p[define._GRID_T_] = 1
    p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
    p[define._DATA_TYPE_] = dt
    av[define._MASS_] = args.mass; av[define._ATOL_] = args.atol; av[define._SIGMA_] = 0.1
    p[define._MG_NUM_LEVEL_] = args.levels
    p[define._MG_USE_GCR_] = int(args.gcr)
    p[define._MG_USE_DEFLATE_] = int(args.deflate)
    p[define._MG_MU_PRE_] = args.mu_pre
    p[define._MAX_ITER_] = args.max_iter
    for level, E in enumerate(level_es, start=1):
        # 每条记录 8 个 int：E,X,Y,Z,T,max_iter,data_type,num_restart。
        off = (level - 1) * define._MG_PARAMS_SIZE_
        p[define._MG_LEVEL1_E_ + off] = E
        p[define._MG_LEVEL1_X_ + off] = Lx // (args.mg_grid[0] ** level)
        p[define._MG_LEVEL1_Y_ + off] = Ly // (args.mg_grid[1] ** level)
        p[define._MG_LEVEL1_Z_ + off] = Lz // (args.mg_grid[2] ** level)
        p[define._MG_LEVEL1_T_ + off] = Lt // (2 * (args.mg_grid[3] ** level))
        p[define._MG_LEVEL1_MAX_ITER_ + off] = args.cmi
        p[define._MG_LEVEL1_DATA_TYPE_ + off] = dt
        p[define._MG_LEVEL1_NUM_RESTART_ + off] = args.restart
        av[define._MG_LEVEL1_ATOL_ + level - 1] = args.atol * args.ctf

    # 清理并填充每一条 fine→coarse transition 的四个指针。
    s[_SET_PTRS_COARSE_BASE_:_SET_PTRS_COARSE_BASE_ + 16] = 0
    coarse_keep = []
    for fl, assets in enumerate(coarse_assets):
        base = _SET_PTRS_COARSE_BASE_ + 4 * fl
        for j, tensor in enumerate(assets):
            keep = tensor.contiguous()
            coarse_keep.append(keep)
            s[base + j] = keep.data_ptr()

    # dev87 形状守卫：基线 npz 与目标格子不一致时自动经 run_qcu_ops 重建
    import subprocess
    expect = [Lx, Ly, Lz, Lt // 2]
    npz_path = OUT / "qcu_clover_solve.npz"
    need = True
    if npz_path.exists():
        zz = np.load(npz_path)
        need = list(zz["b_eo"].shape[-4:]) != expect
    if need:
        print("[dev87-mg] 基线形状不匹配 -> 重跑 run_qcu_ops clover_solve", flush=True)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "run_qcu_ops.py"),
                            "--case", "clover_solve", "--lat", *[str(v) for v in lat],
                            "--mass", str(args.mass)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("baseline rebuild failed: " + r.stderr[-800:])
    npz = np.load(npz_path)
    assert list(npz["b_eo"].shape[-4:]) == expect, "baseline shape still mismatched"
    b_eo = torch.from_numpy(npz["b_eo"]).to("cuda")
    x_ref = torch.from_numpy(npz["x_eo"]).to("cuda")

    s[:_SET_PTRS_COARSE_BASE_] = 0   # 清除已结束集合的陈旧句柄

    def run_once(slot, use_init_guess, initial=None):
        """在独立 LatticeSet 槽位运行一次，返回 (x, wall, history)。"""
        p_run = p.clone()
        p_run[define._SET_INDEX_] = slot
        p_run[define._SET_PLAN_] = 1
        p_run[define._PARITY_] = 0
        p_run[define._VERBOSE_] = int(args.backend_verbose)
        p_run[define._MG_USE_INIT_GUESS_] = int(use_init_guess)
        initialized = False
        try:
            qcu.applyInitQcu(s, p_run, av)
            initialized = True
            x = torch.empty_like(b_eo)
            if initial is None:
                x.zero_()
            else:
                x.copy_(initial)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            qcu.applyCloverMultigridQcu(
                x, b_eo, g, ce, coo, cei, coi, s, p_run)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            return x, wall, _parse_last_history()
        finally:
            if initialized:
                p_run[define._SET_INDEX_] = slot
                qcu.applyEndQcu(s, p_run)

    x_mg, mg_time, history = run_once(2, False)
    result = {
        "lat": lat, "mass": args.mass, "atol": args.atol,
        "levels": args.levels, "E": level_es, "nvi": args.nvi,
        "mg_grid": args.mg_grid, "restart": args.restart,
        "coarse_max_iter": args.cmi, "coarse_tol_factor": args.ctf,
        "max_iter": args.max_iter, "mu_pre": args.mu_pre,
        "gcr": bool(args.gcr), "deflate": bool(args.deflate),
        "warm_requested": bool(args.warm),
        "mg_wall_s": mg_time,
        "rel_diff_vs_bistabcg": float((torch.linalg.norm(
            (x_mg - x_ref).ravel()) / torch.linalg.norm(x_ref.ravel())).item()),
        "history_len": len(history),
        "history_final": history[-1] if history else None,
    }

    if args.warm:
        x_warm, warm_time, warm_history = run_once(3, True, x_mg)
        result.update({
            "warm_wall_s": warm_time,
            "warm_rel_diff_vs_bistabcg": float((torch.linalg.norm(
                (x_warm - x_ref).ravel()) / torch.linalg.norm(x_ref.ravel())).item()),
            "warm_history_len": len(warm_history),
            "warm_history_final": warm_history[-1] if warm_history else None,
            "warm_semantics": "ignored_by_run_gcr" if args.gcr else "enabled",
        })
        x_save = x_warm
    else:
        x_save = x_mg

    # 与 quda 对照共享的 Python full-operator 真残差；只构造一次 U/clover。
    U_full = None
    clover_full = None
    try:
        from pyqcu import dslash, tools
        kappa = 1.0 / (2.0 * args.mass + 8.0)
        U_full = tools.poooxyzt2oooxyzt(g)
        clover_full = dslash.make_clover(
            U_full, kappa=torch.tensor([kappa], dtype=torch.float32))
        result["true_residual_rel"] = _full_residual_rel(
            x_mg, b_eo, g, args.mass, U=U_full, clover=clover_full)
        if args.warm:
            result["warm_true_residual_rel"] = _full_residual_rel(
                x_save, b_eo, g, args.mass, U=U_full, clover=clover_full)
    except Exception as exc:  # 真残差失败不掩盖 C++ 求解结果
        result["true_residual_error"] = repr(exc)

    label = args.label
    if label is None:
        label = f"{args.levels}l"
        if args.gcr:
            label += "_gcr"
        if args.deflate:
            label += "_deflate"
        if args.warm:
            label += "_warm"
    result["label"] = label
    save_result("qcu_clover_mg", result)
    save_result("qcu_mg_" + _slug(label), result)
    np.savez_compressed(OUT / "qcu_clover_mg.npz", x_eo=x_save.cpu().numpy())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
