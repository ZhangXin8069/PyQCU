"""dev87 MG C++ 组件实测：transfer 与窄/宽粗 dslash。

本脚本只调用 ``qcu`` 的 Cython/CUDA 入口，并将同一输入交给 Python
参考实现逐元素比较。它不把缓存文件本身当作测试结果：缓存只提供已经
正交化的 null vector 与 Galerkin 33-tensor。

运行（需先 ``source ./env.sh``）：
  python examples/qcu/dev87/component_cuda.py

默认使用 data/ 中 16x32x32x48、E=12、nvi=1 的资产；结果写入
examples/qcu/dev87/out/component_cuda.json。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import LAT_DEFAULT, load_stencil, pick_v100  # noqa: E402
from pyqcu import tools  # noqa: E402
from pyqcu.cuda import define, qcu  # noqa: E402


OUT = HERE / "out"
E = 12
MG_GRID = (2, 2, 2, 2)


def _params(fine_lat, coarse_lat, device_type):
    """构造只供组件 kernel 使用的独立参数/指针张量。"""
    p = define.params.clone()
    a = define.argv.clone()
    s = define.set_ptrs.clone()
    p[define._LAT_X_] = int(fine_lat[0])
    p[define._LAT_Y_] = int(fine_lat[1])
    p[define._LAT_Z_] = int(fine_lat[2])
    p[define._LAT_T_] = int(fine_lat[3])
    p[define._LAT_XYZT_] = int(np.prod(fine_lat))
    p[define._GRID_X_] = p[define._GRID_Y_] = 1
    p[define._GRID_Z_] = p[define._GRID_T_] = 1
    p[define._NODE_RANK_] = 0
    p[define._NODE_SIZE_] = 1
    p[define._DATA_TYPE_] = define._LAT_C64_
    p[define._SET_INDEX_] = 0
    p[define._SET_PLAN_] = 1
    p[define._VERBOSE_] = 0
    p[define._MG_LEVEL1_E_] = E
    p[define._MG_LEVEL1_X_] = int(coarse_lat[0])
    p[define._MG_LEVEL1_Y_] = int(coarse_lat[1])
    p[define._MG_LEVEL1_Z_] = int(coarse_lat[2])
    p[define._MG_LEVEL1_T_] = int(coarse_lat[3])
    a[define._MASS_] = 0.05
    a[define._ATOL_] = 1e-6
    # device_type 只用于让调用点明确该张量属于哪一设备；当前数据类型固定 c64。
    del device_type
    return p, a, s


def _rel(a, b):
    """返回 L2 相对差及最大绝对差。"""
    d = (a - b).reshape(-1)
    den = torch.linalg.norm(b.reshape(-1)).item()
    return (float(torch.linalg.norm(d).item() / max(den, 1e-30)),
            float(d.abs().max().item()))


def _time_call(fn, warmup=2, repeat=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples)), float(np.min(samples))


def main():
    dev_id = pick_v100()
    device = torch.device(f"cuda:{dev_id}")
    full_lat = tuple(LAT_DEFAULT)
    fine_lat = (full_lat[0], full_lat[1], full_lat[2], full_lat[3] // 2)
    coarse_lat = tuple(n // b for n, b in zip(fine_lat, MG_GRID))
    print(f"[component-cuda] device={torch.cuda.get_device_name(dev_id)}")
    print(f"[component-cuda] fine_odd={fine_lat} coarse_odd={coarse_lat}", flush=True)

    lonv, hnn, hdg, sit = load_stencil(list(full_lat), E=E, nvi=1,
                                        device=device)
    # 10 维块布局的连续内存就是 kernel 所需的 [E,e,Xf,Yf,Zf,Tf]。
    if tuple(lonv.shape) != (E, 12, 8, 2, 16, 2, 16, 2, 12, 2):
        raise ValueError(f"unexpected lonv shape: {tuple(lonv.shape)}")
    p, a, s = _params(fine_lat, coarse_lat, device)
    qcu.applyInitQcu(s, p, a)
    initialized = True

    result = {
        "lat": list(full_lat),
        "fine_odd": list(fine_lat),
        "coarse_odd": list(coarse_lat),
        "E": E,
        "cache": "data/L16x32x32x48_lv1_E12_nvi1_t1e-2.h5 (or t0.01 alias)",
        "device": dev_id,
        "device_name": torch.cuda.get_device_name(dev_id),
    }
    try:
        # CPU 生成后拷贝，避免在不带 sm_60 PyTorch kernel 的 P100 上误触发
        # torch.randn/arange CUDA kernel；本脚本的单卡规范是 V100。
        nf = int(np.prod(fine_lat))
        nc = int(np.prod(coarse_lat))
        base_f = torch.arange(E * nf, dtype=torch.float32).reshape(E, *fine_lat)
        fine = (base_f + 0.125j * (base_f.remainder(17.0) - 8.0)).to(
            torch.complex64).to(device)
        base_c = torch.arange(E * nc, dtype=torch.float32).reshape(E, *coarse_lat)
        coarse = (base_c + 0.25j * (base_c.remainder(13.0) - 6.0)).to(
            torch.complex64).to(device)

        coarse_cpp = torch.empty_like(coarse)
        fine_cpp = torch.empty_like(fine)
        coarse_ref = tools.restrict(lonv, fine)
        fine_ref = tools.prolong(lonv, coarse)

        def restrict_call():
            qcu.applyMultigridRestrictQcu(coarse_cpp, fine, lonv, s, p)

        def prolong_call():
            qcu.applyMultigridProLongQcu(fine_cpp, coarse, lonv, s, p)

        restrict_call()
        prolong_call()
        torch.cuda.synchronize()
        r_rel, r_max = _rel(coarse_cpp, coarse_ref)
        p_rel, p_max = _rel(fine_cpp, fine_ref)
        r_ms, r_min_ms = _time_call(restrict_call)
        p_ms, p_min_ms = _time_call(prolong_call)

        rp_cpp = torch.empty_like(fine)
        rp_coarse_cpp = torch.empty_like(coarse)
        qcu.applyMultigridProLongQcu(rp_cpp, coarse, lonv, s, p)
        qcu.applyMultigridRestrictQcu(rp_coarse_cpp, rp_cpp, lonv, s, p)
        torch.cuda.synchronize()
        rp_rel, rp_max = _rel(rp_coarse_cpp, coarse)
        result["restrict"] = {
            "l2_rel": r_rel, "max_abs": r_max, "median_ms": r_ms,
            "min_ms": r_min_ms,
            "pass": bool(r_rel < 1e-5),
        }
        result["prolong"] = {
            "l2_rel": p_rel, "max_abs": p_max, "median_ms": p_ms,
            "min_ms": p_min_ms,
            "pass": bool(p_rel < 1e-5),
        }
        result["restrict_prolong_identity"] = {
            "l2_rel": rp_rel, "max_abs": rp_max,
            "pass": bool(rp_rel < 1e-5),
        }

        # 同一粗输入同时走窄/宽 C++ kernel 与 Python 参考；窄核只包含
        # sitting + 8 nearest terms，宽核再加 24 个 diagonal terms。
        coarse_in = (torch.sin(base_c * 0.013) +
                     1j * torch.cos(base_c * 0.017)).to(torch.complex64).to(device)
        narrow_cpp = torch.empty_like(coarse_in)
        wide_cpp = torch.empty_like(coarse_in)
        narrow_ref = torch.einsum("EeXYZT,eXYZT->EXYZT", sit, coarse_in).clone()
        for d in range(4):
            fwd = torch.roll(coarse_in, shifts=-1, dims=d + 1)
            bwd = torch.roll(coarse_in, shifts=1, dims=d + 1)
            narrow_ref += torch.einsum("EeXYZT,eXYZT->EXYZT", hnn[0, d], fwd)
            narrow_ref += torch.einsum("EeXYZT,eXYZT->EXYZT", hnn[1, d], bwd)
        wide_ref = tools.apply_stencil(hnn, hdg, sit, coarse_in)

        def narrow_call():
            qcu.applyMultigridCoarseDslashQcu(
                narrow_cpp, coarse_in,
                torch.stack((hnn[0], hnn[1]), dim=0), sit, s, p)

        def wide_call():
            qcu.applyMultigridCoarseDslashWideQcu(
                wide_cpp, coarse_in, sit, hnn, hdg, s, p)

        # stack() 每次调用窄核会产生临时张量；预先固定指针，计时才只包含
        # 组件调用本身。
        hopping = torch.stack((hnn[0], hnn[1]), dim=0).contiguous()

        def narrow_call_fixed():
            qcu.applyMultigridCoarseDslashQcu(
                narrow_cpp, coarse_in, hopping, sit, s, p)

        narrow_call_fixed()
        wide_call()
        torch.cuda.synchronize()
        n_rel, n_max = _rel(narrow_cpp, narrow_ref)
        w_rel, w_max = _rel(wide_cpp, wide_ref)
        n_ms, n_min_ms = _time_call(narrow_call_fixed)
        w_ms, w_min_ms = _time_call(wide_call)
        result["coarse_dslash_narrow"] = {
            "l2_rel": n_rel, "max_abs": n_max, "median_ms": n_ms,
            "min_ms": n_min_ms, "pass": bool(n_rel < 1e-5),
        }
        result["coarse_dslash_wide"] = {
            "l2_rel": w_rel, "max_abs": w_max, "median_ms": w_ms,
            "min_ms": w_min_ms, "pass": bool(w_rel < 1e-5),
        }
        checks = [result[k]["pass"] for k in (
            "restrict", "prolong", "restrict_prolong_identity",
            "coarse_dslash_narrow", "coarse_dslash_wide")]
        result["pass"] = bool(all(checks))
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    finally:
        if initialized:
            p[define._SET_INDEX_] = 0
            qcu.applyEndQcu(s, p)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "component_cuda.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[result] {OUT / 'component_cuda.json'}")
    return result


if __name__ == "__main__":
    r = main()
    raise SystemExit(0 if r.get("pass") else 1)
