"""pyquda 对比套件公共工具：维度排布转换 + h5 I/O + 指标计算。

以实际代码为准的维度排布（2026-08-28 实测核对）：

- pyqcu（纯 Python 后端，pyqcu/dslash/_wilson.py、pyqcu/tools/_define.py）：
  gauge   [c,c,d,x,y,z,t] = (3,3,4,Lx,Ly,Lz,Lt)，xyzt 正序，无奇偶轴
  fermion [s,c,x,y,z,t]   = (4,3,Lx,Ly,Lz,Lt)
  奇偶排布 poooxyzt：[p, prefix..., Lx,Ly,Lz,Lt//2] —— 最后一个空间维 t 被切分
  （oooxyzt2poooxyzt 按 (x+y+z+t)%2 棋盘格，p=0 为偶奇偶）
- pyquda 0.3.2（QUDA 1.1.0，pyquda/field.py）：
  gauge   [d,q,t,z,y,x/2,c,c] = (4,2,Lt,Lz,Ly,Lx//2,3,3)
  fermion [q,t,z,y,x/2,s,c]   = (2,Lt,Lz,Ly,Lx//2,4,3)
  —— 时空维 tzyx 倒序，最后一个空间维 x 被奇偶切分（q=0 为偶奇偶）

两侧奇偶切分维不同：pyqcu 切 t（最后轴），pyquda 切 x（最后轴）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import h5py

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "examples" / "data" / "pyquda_cmp"
OUT_DIR = Path(__file__).resolve().parent / "out"

MASS = 0.05
KAPPA_PYQCU = 1.0 / (2.0 * MASS + 8.0)  # 1/(2m+8)，dev87 锚定
LAT_DEFAULT = [8, 8, 8, 16]


# ---------------------------------------------------------------- h5 I/O
def save_h5(path: Path, overwrite: bool = True, **datasets) -> None:
    """每调用独立 File 句柄（多进程安全）。键/值 -> dataset 名/数组。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w" if overwrite else "a") as f:
        for k, v in datasets.items():
            f.create_dataset(k, data=np.asarray(v))
    print(f"[h5] wrote {path}")


def load_h5(path: Path, names) -> dict:
    path = Path(path)
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k]) for k in names}


# ------------------------------------------------- 棋盘格奇偶切分（x 维，pyquda 约定）
def evenodd_split_x(arr: np.ndarray, lat, x_axis: int) -> np.ndarray:
    """棋盘格奇偶切分 X 轴：arr[.., T,Z,Y,X, ..] -> arr[.., q, T,Z,Y,X/2, ..]。

    x_axis 为 X 轴号（X 前必为连续的 T,Z,Y 三轴）。q=0 为偶奇偶
    ((x+y+z+t)%2==0)，与 pyquda field.lexico / QUDA 约定一致。
    """
    Lx, Ly, Lz, Lt = lat
    t_axis = x_axis - 3
    pref = arr.shape[:t_axis]
    suf = arr.shape[x_axis + 1:]
    T, Z, Y = arr.shape[t_axis], arr.shape[t_axis + 1], arr.shape[t_axis + 2]
    x = np.arange(Lx)
    ev_idx = np.flatnonzero(x % 2 == 0)
    od_idx = np.flatnonzero(x % 2 == 1)
    out = np.zeros(pref + (2, T, Z, Y, Lx // 2) + suf, dtype=arr.dtype)
    sl = (slice(None),) * t_axis
    sl_suf = (slice(None),) * len(suf)
    for t in range(T):
        for z in range(Z):
            for y in range(Y):
                src = arr[sl + (t, z, y, slice(None)) + sl_suf]
                xi = len(sl)
                if (y + z + t) % 2 == 0:
                    out[sl + (0, t, z, y, slice(None)) + sl_suf] = np.take(src, ev_idx, axis=xi)
                    out[sl + (1, t, z, y, slice(None)) + sl_suf] = np.take(src, od_idx, axis=xi)
                else:
                    out[sl + (0, t, z, y, slice(None)) + sl_suf] = np.take(src, od_idx, axis=xi)
                    out[sl + (1, t, z, y, slice(None)) + sl_suf] = np.take(src, ev_idx, axis=xi)
    return out


def evenodd_merge_x(arr: np.ndarray, lat, x_axis: int) -> np.ndarray:
    """逆变换：arr[.., q, T,Z,Y,X/2, ..] -> arr[.., T,Z,Y,X, ..]（x_axis 为恢复后 X 轴号）。"""
    Lx, Ly, Lz, Lt = lat
    t_axis = x_axis - 3
    pref = arr.shape[:t_axis]
    suf = arr.shape[t_axis + 5:]
    T, Z, Y = arr.shape[t_axis + 1], arr.shape[t_axis + 2], arr.shape[t_axis + 3]
    x = np.arange(Lx)
    ev_idx = np.flatnonzero(x % 2 == 0)
    od_idx = np.flatnonzero(x % 2 == 1)
    out = np.zeros(pref + (T, Z, Y, Lx) + suf, dtype=arr.dtype)
    sl = (slice(None),) * t_axis
    sl_suf = (slice(None),) * len(suf)
    for t in range(T):
        for z in range(Z):
            for y in range(Y):
                if (y + z + t) % 2 == 0:
                    out[sl + (t, z, y, ev_idx) + sl_suf] = arr[sl + (0, t, z, y, slice(None)) + sl_suf]
                    out[sl + (t, z, y, od_idx) + sl_suf] = arr[sl + (1, t, z, y, slice(None)) + sl_suf]
                else:
                    out[sl + (t, z, y, ev_idx) + sl_suf] = arr[sl + (1, t, z, y, slice(None)) + sl_suf]
                    out[sl + (t, z, y, od_idx) + sl_suf] = arr[sl + (0, t, z, y, slice(None)) + sl_suf]
    return out


# ------------------------------------------------- pyqcu <-> pyquda 排布
def pyqcu_gauge_to_quda(u_p: np.ndarray, lat) -> np.ndarray:
    """pyqcu [3,3,4,X,Y,Z,T] -> pyquda [4,2,T,Z,Y,X/2,3,3]。"""
    u = np.ascontiguousarray(np.transpose(u_p, (2, 6, 5, 4, 3, 0, 1)))  # [d,T,Z,Y,X,c,c]
    return evenodd_split_x(u, lat, x_axis=4)


def quda_gauge_to_pyqcu(u_q: np.ndarray, lat) -> np.ndarray:
    """pyquda [4,2,T,Z,Y,X/2,3,3] -> pyqcu [3,3,4,X,Y,Z,T]。"""
    u = evenodd_merge_x(u_q, lat, x_axis=4)  # [d,T,Z,Y,X,c,c]
    return np.ascontiguousarray(np.transpose(u, (5, 6, 0, 4, 3, 2, 1)))


def pyqcu_fermion_to_quda(b_p: np.ndarray, lat) -> np.ndarray:
    """pyqcu [4,3,X,Y,Z,T] -> pyquda [2,T,Z,Y,X/2,4,3]。"""
    b = np.ascontiguousarray(np.transpose(b_p, (5, 4, 3, 2, 0, 1)))  # [T,Z,Y,X,s,c]
    return evenodd_split_x(b, lat, x_axis=3)


def quda_fermion_to_pyqcu(b_q: np.ndarray, lat) -> np.ndarray:
    """pyquda [2,T,Z,Y,X/2,4,3] -> pyqcu [4,3,X,Y,Z,T]。"""
    b = evenodd_merge_x(b_q, lat, x_axis=3)  # [T,Z,Y,X,s,c]
    return np.ascontiguousarray(np.transpose(b, (4, 5, 3, 2, 1, 0)))


# ------------------------------------------------- 指标
def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    nb = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / (nb if nb else 1.0))


def linreg_scale(a: np.ndarray, b: np.ndarray) -> tuple:
    """最小二乘标量 c：min ||a - c*b||/||b||；返回 (c, rel)。"""
    ra, rb = a.ravel(), b.ravel()
    c = float(np.vdot(rb, ra) / np.vdot(rb, rb))
    return c, float(np.linalg.norm(ra - c * rb) / np.linalg.norm(rb))


def save_json(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    payload = dict(payload)
    payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    print(f"[result] {path}")
    return path


def parse_cg_iterations(text: str):
    """解析 QUDA stdout 中的逐迭代残差行，返回 [(iter, residual), ...]。"""
    import re

    rows = []
    for line in text.splitlines():
        m = re.search(r"CG:.*?iter\s*=\s*(\d+).*?residual\s*=\s*([0-9.eE+-]+)", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    return rows