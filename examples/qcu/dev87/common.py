"""dev87 公共设施：设备选择、统一 gauge/nullvec 加载、PyQCU<->QDP 布局转换、结果落盘。

数据约定（继承 dev84）：data/gauge_{lat}_m{mass}_seed{seed}_c64.h5 的
  g  = [2(parity),3,3,4,x,y,z,T/2] c64（奇偶压缩布局，t 方向对半）
  fi = [2,4,3,x,y,z,T/2] c64（随机源场）
GPU 规约（v20260825 指令17）：单卡测试用 V100-32GB；多卡测试用 P100×2。
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from pyqcu import tools
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params as mod_params, argv as mod_argv, set_ptrs as mod_set_ptrs
from pyqcu.cuda import qcu

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"
LAT_DEFAULT = [16, 32, 32, 48]
MASS_DEFAULT = 0.05
ATOL_DEFAULT = 1e-6
SIGMA_DEFAULT = 0.1
SEED_DEFAULT = 42


def pick_v100():
    """按 v20260825 指令17 选 V100-32GB（torch 枚举本机为 cuda:0）。"""
    for i in range(torch.cuda.device_count()):
        if "V100" in torch.cuda.get_device_name(i):
            torch.cuda.set_device(i)
            return i
    raise RuntimeError("V100 not found")


def gauge_tag(lat, mass, seed=SEED_DEFAULT):
    return f"gauge_{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_m{mass}_seed{seed}_c64.h5"


def nv_tag(lat, E, nvi, suf=""):
    return f"L{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_lv1_E{E}_nvi{nvi}{suf}_t1e-2.h5"


def load_gauge_h5(lat, mass=MASS_DEFAULT, seed=SEED_DEFAULT, device="cuda"):
    """读 data/*.h5 -> g_dev[2,3,3,4,x,y,z,T/2]。"""
    import h5py
    path = DATA_DIR / gauge_tag(lat, mass, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(str(path), "r") as f:
        g_np = f["g"][...]
    return torch.from_numpy(g_np).to(device)


def load_stencil(lat, E=12, nvi=1, suf="", device="cuda"):
    """读 data/*_lv1_*.h5 -> (lonv,hnn,hdg,sit) 33-tensor 粗算子资产。"""
    path = DATA_DIR / nv_tag(lat, E, nvi, suf)
    if not path.exists():
        raise FileNotFoundError(path)
    return tuple(tools.load_tensor_h5(str(path), dataset=k, device=device)
                 for k in ("lonv", "hnn", "hdg", "sit"))


def full_gauge_numpy(g_dev):
    """PyQCU 奇偶布局 -> 全格点 numpy [3,3,4,X,Y,Z,T]。"""
    U_full = tools.poooxyzt2oooxyzt(g_dev)
    return U_full.detach().cpu().contiguous().numpy()


def full_to_qdp(u_full):
    """全格点 [3,3,4,X,Y,Z,T] -> QDP 序 numpy (4,Lt,Lz,Ly,Lx,3,3)（供 pyquda loadGauge）。

    色指标行/列次序与 dagger 约定的最终核对由 cmp_dslash 数值对照闭环裁决。
    """
    u = np.asarray(u_full)
    u = np.ascontiguousarray(np.transpose(u, (2, 6, 5, 4, 3, 0, 1)))
    return u


def make_clover_tensors(g_dev, lat, mass=MASS_DEFAULT):
    """经 C++ applyCloversQcu 构建 (ce,cei,coo,coi)。调用方负责生命周期。"""
    p = mod_params.clone()
    a = mod_argv.clone()
    s = mod_set_ptrs.clone()
    dt = define._LAT_C64_
    Lx, Ly, Lz, Lt = lat
    p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
    p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    p[define._GRID_X_] = p[define._GRID_Y_] = p[define._GRID_Z_] = p[define._GRID_T_] = 1
    p[define._NODE_RANK_] = 0; p[define._NODE_SIZE_] = 1
    p[define._DATA_TYPE_] = dt
    av = a.to(dtype=define.dtype(dt).to_real())
    av[define._MASS_] = mass; av[define._SIGMA_] = SIGMA_DEFAULT
    ls = define.lat_shape(p)
    ce = torch.empty([4, 3, 4, 3] + ls, dtype=torch.complex64, device=g_dev.device)
    cei = torch.empty_like(ce); coo = torch.empty_like(ce); coi = torch.empty_like(ce)
    idx = int(p[define._SET_INDEX_].item())
    p[define._SET_INDEX_] = idx; p[define._SET_PLAN_] = 2; p[define._PARITY_] = 0
    qcu.applyInitQcu(s, p, av)
    qcu.applyCloversQcu(ce, cei, g_dev, s, p)
    p[define._SET_INDEX_] = idx + 1; p[define._PARITY_] = 1
    qcu.applyInitQcu(s, p, av)
    qcu.applyCloversQcu(coo, coi, g_dev, s, p)
    for j in (idx, idx + 1):
        p[define._SET_INDEX_] = j
        qcu.applyEndQcu(s, p)
    p[define._SET_INDEX_] = idx + 2
    return ce, cei, coo, coi, s, p, av


def save_result(name, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    payload = dict(payload)
    payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"[result] {path}")
    return path
