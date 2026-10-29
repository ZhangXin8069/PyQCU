#!/usr/bin/env python3
"""PyQCU(C++ Clover MultiGrid) 对照原生 QUDA 的 pion 两点关联函数入口。

物理口径
--------
对每个时间片 ``tsrc`` 在空间原点放置全 12 分量点源
``b[s, c, 0, 0, 0, tsrc] = 1``（与 ``pyqcu.lattice.point_source`` 的
``spin=None/color=None`` 约定一致），求解 Clover-Wilson 系统
``D x = b``，得到 12 分量传播子 ``Q[(s0,c0), s, c, x, y, z, t]``，再按下式
构造动量投影的介子两点函数（基脚本 ``main_pion_C2.py`` 的 GM-loop 结构，
其中 gamma 只作用在 spin 指标、color 指标在源汇两端闭合）::

    anti[a,d,j,i,x] = sum_{b,c} g5[a,b] conj(Q[(b,j), c, i, x]) g5[c,d]
    C2[a,e,t]       = sum_x phase(x) sum_{b,c,d,i,j}
                      anti[a,b,j,i,x] * gamma[b,c] * Q[(d,i), c, j, x] * gamma[d,e]
    C2_pion(t)      = sum_a C2[a,a,t]

两套后端的归一化不同（已在单位规范场下逐点实测）：
``D_QUDA = (m+4) * D_PyQCU``，即 ``x_PyQCU = (m+4) * x_QUDA = (2*kappa)^-1 x_QUDA``；
本脚本不盲缩放，而是先用最小二乘拟合复数标度 ``c``，再检查去除 ``c`` 后的
相对残差，``c`` 与 ``1/(2*kappa)`` 的偏差也会写入记录。

排布转换（QUDA/QDP 与 PyQCU 不同）
----------------------------------
* PyQCU: gauge ``[3,3,4,X,Y,Z,T]``、fermion ``[4,3,X,Y,Z,T]``，棋盘按
  ``(x+y+z+t)%2`` 在 **t 轴**压缩（``tools.oooxyzt2poooxyzt``）。
* QUDA/QDP: gauge ``[4,T,Z,Y,X,3,3]``、fermion ``[2,T,Z,Y,X/2,4,3]``，
  棋盘在 **x 轴**压缩。
两者往返使用 ``pyqcu.testing.quda.common`` 的显式转换函数，运行前先做
round-trip 门禁（误差必须为 0），禁止直接 reshape 冒充转换。

进程隔离
--------
libqcu.so 与 libquda.so 不能在同一进程加载，因此默认 ``compare`` 角色会
以子进程方式分别执行 QCU 与 QUDA 求解（共享 HDF5 输入/输出）。

用法
----
::

    # 全流程：PyQCU 生成测试 gauge → QCU MG → QUDA MG → 对照
    source ./env.sh
    python pyqcu/testing/quda/test_pion_c2.py --lat 8 8 8 16 --ntsrc 1

    # 指定已有 gauge（PyQCU 排布 [3,3,4,X,Y,Z,T] 的 HDF5 或 .npy）
    python pyqcu/testing/quda/test_pion_c2.py --gauge data/my_gauge.h5

    # 多卡（一线程一卡，QCU 侧；QUDA 侧可用 --quda-mpi 走 MPI 进程网格）
    python pyqcu/testing/quda/test_pion_c2.py --qcu-devices 0 1 --qcu-threads 2

    # 只做 CPU 可复现部分
    python pyqcu/testing/quda/test_pion_c2.py --role check

环境提示
--------
* P100(sm_60) 上 torch 2.10 无 SASS 时会失败，请先 ``source
  pyqcu/testing/qcu/strict/quda_comparison/p100_env.sh``；
* QUDA 的 half/sloppy 精度在本机 clover 路径不可用，脚本显式把
  cuda/sloppy/precondition/refinement_sloppy/eigensolver 全部设为所选精度；
* P100 无 tensor core，QUDA MMA 粗层内核不可用，脚本在 CC<7.0 时自动关闭
  ``setup_use_mma/dslash_use_mma/transfer_use_mma``；
* QCU C++ 后端时间方向为周期边界，故 QUDA 默认 ``t_boundary=+1``；
  ``--t-boundary -1`` 只允许在 ``--role quda`` 单独运行时使用。

产物（默认 ``data/testing/pion_c2/<run_id>/``）：``inputs/gauge_xyzt.h5``、
``qcu/solutions*.h5``、``quda/solutions*.h5``、两侧 ``metrics.json`` 与汇总
``record.json``（含求解总迭代次数、总耗时、真残差、标度与 C2 差异）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = REPO / "data" / "testing" / "pion_c2"
COMPONENTS = 12


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------
class RunFailure(RuntimeError):
    """可解释的失败（用于 fail-closed 输出）。"""


class StepTimer:
    """同步计时：只统计被测段，避免把 setup/转换算进求解耗时。"""

    def __init__(self, sync=None) -> None:
        self.sync = sync
        self.timings: List[Tuple[str, float]] = []

    def measure(self, label: str, fn, *args, **kwargs):
        if self.sync is not None:
            self.sync()
        started = time.perf_counter()
        out = fn(*args, **kwargs)
        if self.sync is not None:
            self.sync()
        self.timings.append((label, time.perf_counter() - started))
        return out

    def add(self, label: str, seconds: float) -> None:
        self.timings.append((label, float(seconds)))

    def total(self, prefix: str = "") -> float:
        return float(sum(v for k, v in self.timings if k.startswith(prefix)))


def precision_spec(name: str) -> Dict[str, Any]:
    """精度必须两端一致：single↔complex64↔QUDA_SINGLE，double↔complex128↔QUDA_DOUBLE。"""
    if name == "single":
        return {"name": "single", "np": np.complex64, "real_np": np.float32,
                "torch": "complex64", "torch_real": "float32", "quda": "QUDA_SINGLE_PRECISION",
                "bytes": 8}
    if name == "double":
        return {"name": "double", "np": np.complex128, "real_np": np.float64,
                "torch": "complex128", "torch_real": "float64", "quda": "QUDA_DOUBLE_PRECISION",
                "bytes": 16}
    raise RunFailure(f"未知精度 {name!r}（可选 single/double）")


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.ascontiguousarray(contiguous).tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"不可序列化类型: {type(value)!r}")


def parse_tsrc(text: str, lt: int) -> List[int]:
    if text == "all":
        return list(range(lt))
    values = [int(v) for v in text.split(",") if v.strip()]
    for value in values:
        if not 0 <= value < lt:
            raise RunFailure(f"tsrc={value} 超出 [0,{lt})")
    return values


def run_dir_from_args(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    return DEFAULT_RUN_ROOT / args.run_id


# --------------------------------------------------------------------------
# 排布转换（复用 pyqcu/testing/quda/common.py 的审计过的转换，另加压缩布局）
# --------------------------------------------------------------------------
def layout_roundtrip_large(lat: Tuple[int, int, int, int]) -> Dict[str, float]:
    """转换门禁：gauge/fermion 往返必须 bit-exact（transpose/reshape 组合）。"""
    from pyqcu.testing.quda.common import (
        pyqcu_fermion_to_quda,
        pyqcu_gauge_to_quda,
        quda_fermion_to_pyqcu,
        quda_gauge_to_pyqcu,
    )

    rng = np.random.default_rng(20261014)
    u = (rng.normal(size=(3, 3, 4, *lat)) + 1j * rng.normal(size=(3, 3, 4, *lat))).astype(np.complex64)
    f = (rng.normal(size=(4, 3, *lat)) + 1j * rng.normal(size=(4, 3, *lat))).astype(np.complex64)
    return {
        "gauge_error": float(np.max(np.abs(quda_gauge_to_pyqcu(pyqcu_gauge_to_quda(u)) - u))),
        "fermion_error": float(np.max(np.abs(quda_fermion_to_pyqcu(pyqcu_fermion_to_quda(f)) - f))),
    }


def pyqcu_gauge_to_qcu_compact(u: np.ndarray) -> np.ndarray:
    """PyQCU 全格点 [3,3,4,X,Y,Z,T] → QCU/C++ 压缩 [2,3,3,4,X,Y,Z,T/2]。"""
    import torch
    from pyqcu import tools

    if u.shape[:3] != (3, 3, 4):
        raise RunFailure(f"PyQCU gauge 必须为 [3,3,4,X,Y,Z,T]，收到 {u.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(u))
    return np.asarray(tools.oooxyzt2poooxyzt(tensor)).copy()


def qcu_compact_to_pyqcu_gauge(compact: np.ndarray) -> np.ndarray:
    import torch
    from pyqcu import tools

    return np.asarray(tools.poooxyzt2oooxyzt(torch.from_numpy(np.ascontiguousarray(compact)))).copy()


# --------------------------------------------------------------------------
# gauge 生成 / 读取 / 保存
# --------------------------------------------------------------------------
def prepare_gauge(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """无输入 gauge 时用 PyQCU 生成 SU(3) 测试 gauge，并（可选）stout 平滑。"""
    spec = precision_spec(args.precision)
    lat = tuple(int(v) for v in args.lat)
    out_path = run_dir / "inputs" / "gauge_xyzt.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {"lattice_xyzt": list(lat), "precision": spec["name"]}

    if args.gauge:
        source = Path(args.gauge).expanduser()
        if not source.is_file():
            raise RunFailure(f"找不到输入 gauge: {source}")
        if source.suffix == ".npy":
            u = np.load(source)
        else:
            import h5py

            with h5py.File(source, "r") as handle:
                key = args.gauge_dataset if args.gauge_dataset in handle else next(iter(handle.keys()))
                u = np.asarray(handle[key][...])
        meta.update({"gauge_source": str(source), "gauge_origin": "user-input"})
    else:
        import torch
        from pyqcu import lattice

        dtype = getattr(torch, spec["torch"])
        u_t = torch.empty([3, 3, 4, *lat], dtype=dtype)
        lattice.generate_gauge_field(u_t, sigma=float(args.gauge_sigma), seed=int(args.gauge_seed))
        ok = lattice.check_su3(u_t, tol=1e-3, verbose=False)
        u = np.asarray(u_t)
        meta.update({
            "gauge_source": "pyqcu.lattice.generate_gauge_field",
            "gauge_origin": "pyqcu-generated",
            "gauge_seed": int(args.gauge_seed),
            "gauge_sigma": float(args.gauge_sigma),
            "su3_check": bool(ok),
        })
        if not ok:
            raise RunFailure("PyQCU 生成的 gauge 未通过 SU(3) 检查，拒绝继续")

    if tuple(u.shape[-4:]) != lat:
        raise RunFailure(f"gauge 格点 {tuple(u.shape[-4:])} 与 --lat {lat} 不一致")
    if u.shape[:3] != (3, 3, 4):
        raise RunFailure(f"gauge 前缀维必须为 [3,3,4]，收到 {u.shape}")
    u = np.ascontiguousarray(u.astype(spec["np"], copy=False))

    if args.stout_steps > 0:
        import torch
        from pyqcu.torch import smear

        dtype = getattr(torch, spec["torch"])
        u_t = torch.from_numpy(u)
        u = np.asarray(smear.stout_smear(u_t, nstep=int(args.stout_steps),
                                         rho=float(args.stout_rho))).astype(spec["np"], copy=False)
        meta.update({"stout_steps": int(args.stout_steps), "stout_rho": float(args.stout_rho)})
    else:
        meta.update({"stout_steps": 0, "stout_rho": None})

    import h5py

    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("data", data=u)
        handle.attrs["layout"] = "xyzt_cxyzt"  # [color,color,dir,X,Y,Z,T]
        handle.attrs["precision"] = spec["name"]
    meta["gauge_sha256"] = sha256_array(u)
    meta["gauge_h5"] = str(out_path)
    write_json(run_dir / "inputs" / "gauge_meta.json", meta)
    print(f"[gauge] {meta['gauge_origin']} lattice={lat} dtype={u.dtype} sha256={meta['gauge_sha256'][:16]}…")
    return meta


def load_gauge_xyzt(path: Path, precision: str) -> np.ndarray:
    import h5py

    spec = precision_spec(precision)
    with h5py.File(path, "r") as handle:
        key = "data" if "data" in handle else next(iter(handle.keys()))
        return np.ascontiguousarray(handle[key][...].astype(spec["np"], copy=False))


def point_source_xyzt(lat: Sequence[int], tsrc: int, component: Optional[int],
                      dtype) -> np.ndarray:
    """点源 [4,3,X,Y,Z,T]。

    ``component=None`` 表示该时空点全部 spin/color 分量同时置 1（PyQCU
    ``lattice.point_source`` 默认约定）；否则只填 ``component = spin*3+color``。
    """
    source = np.zeros((4, 3, *lat), dtype=dtype)
    if component is None:
        source[:, :, 0, 0, 0, tsrc] = 1.0
    else:
        spin, color = divmod(int(component), 3)
        source[spin, color, 0, 0, 0, tsrc] = 1.0
    return source


def momentum_phase(lat: Sequence[int], mom: Sequence[int], grid_coord=(0, 0, 0, 0),
                   local_shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """e^{-2*pi*i p·x/L}，x 为全局坐标（多 rank 时按 rank 偏移）。"""
    lx, ly, lz, _ = (list(local_shape) if local_shape is not None else list(lat))
    gx, gy, gz, _ = grid_coord
    x = gx * lx + np.arange(lx)
    y = gy * ly + np.arange(ly)
    z = gz * lz + np.arange(lz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.exp(-2j * np.pi * (mom[0] * xx / lat[0] + mom[1] * yy / lat[1] +
                                 mom[2] * zz / lat[2]))


# --------------------------------------------------------------------------
# pion 两点函数收缩
# --------------------------------------------------------------------------
def meson_c2(prop: np.ndarray, gamma: np.ndarray, g5: np.ndarray,
             phase: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """按模块 docstring 的公式计算 C2。

    prop: [12, 4, 3, X, Y, Z, T]（第 0 轴 = (s0,c0) 源分量，row-major）。
    phase: [X,Y,Z] 动量相位（None 表示零动量）。
    """
    if prop.ndim != 7 or prop.shape[1] != 4 or prop.shape[2] != 3 or prop.shape[0] != COMPONENTS:
        raise RunFailure(f"propagator 必须为 [12,4,3,X,Y,Z,T]，收到 {prop.shape}")
    q = prop.reshape(4, 3, 4, 3, *prop.shape[3:])                   # [s0,c0,s,c,X,Y,Z,T]
    anti = np.einsum("ab,bjcixyzt,cd->adjixyzt", g5, q.conj(), g5)  # [a,d,源color,汇color]
    anti = np.swapaxes(anti, 2, 3)                                  # [a,d,汇color,源color]
    # [c,d,j,i,X,Y,Z,T] = [汇 spin, 源 spin, 汇 color, 源 color]
    p2 = np.transpose(q, (2, 0, 3, 1, 4, 5, 6, 7))
    weights = np.ones(prop.shape[3:6]) if phase is None else phase
    # anti[a,b,j,i] → Γ1[b,c] → quark[c,d,j,i] → Γ2[d,e]，最后对 a 取迹。
    c2 = np.einsum("abjixyzt,bc,cdjixyzt,de,xyz->aet", anti, gamma, p2, gamma, weights)
    return {"c2_spin": c2, "c2_pion": np.einsum("aat->t", c2)}


def meson_c2_reference(prop: np.ndarray, gamma: np.ndarray, g5: np.ndarray,
                       phase: Optional[np.ndarray] = None) -> np.ndarray:
    """逐点循环的独立实现，仅用于 CPU 门禁交叉验证 ``meson_c2`` 的指标配对。"""
    x_dim, y_dim, z_dim, t_dim = prop.shape[3:]
    out = np.zeros((4, 4, t_dim), dtype=np.complex128)
    weights = np.ones((x_dim, y_dim, z_dim)) if phase is None else phase
    for t in range(t_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(z_dim):
                    w = weights[x, y, z]
                    for a in range(4):
                        for e in range(4):
                            total = 0.0j
                            for b in range(4):
                                for c in range(4):
                                    for d in range(4):
                                        for i in range(3):
                                            for j in range(3):
                                                # 约定：第 3 指标 = 汇 color，第 4 指标 = 源 color
                                                # anti[a,b,j,i] = sum_{b2,c2} g5[a,b2] conj(Q[(b2,i),c2,j]) g5[c2,b]
                                                anti = sum(g5[a, b2] *
                                                           np.conj(prop[b2 * 3 + i, c2, j, x, y, z, t]) *
                                                           g5[c2, b]
                                                           for b2 in range(4) for c2 in range(4))
                                                # C2 中配对的传播子项：[c,d,j,i] = Q[(d,i), c, j]
                                                quark = prop[d * 3 + i, c, j, x, y, z, t]
                                                total += anti * gamma[b, c] * quark * gamma[d, e]
                            out[a, e, t] += w * total
    return out


def pyqcu_gammas() -> Tuple[np.ndarray, np.ndarray]:
    from pyqcu import lattice

    return np.asarray(lattice.gamma_5).astype(np.complex128), np.asarray(lattice.gamma_5).astype(np.complex128)


def reference_true_residual(prop_row: np.ndarray, source: np.ndarray, gauge: np.ndarray,
                            kappa: float) -> Dict[str, float]:
    """用纯 PyTorch(PyQCU) 参考算子独立复算 ||D x - b|| / ||b||（CPU，不依赖两后端）。"""
    import torch
    from pyqcu.torch import dslash

    x = torch.from_numpy(np.asarray(prop_row)).to(torch.complex64)
    u = torch.from_numpy(np.asarray(gauge)).to(torch.complex64)
    b = torch.from_numpy(np.asarray(source)).to(torch.complex64)
    u0 = torch.Tensor([1.0])
    clover = dslash.make_clover(u, kappa=torch.Tensor([kappa]), u_0=u0).to(torch.complex64)
    resid = dslash.give_wilson(x, u, kappa, u0, True) + dslash.give_clover(x, clover) - b
    norm_b = float(torch.linalg.vector_norm(b))
    return {"residual": float(torch.linalg.vector_norm(resid)) / norm_b,
            "norm_x": float(torch.linalg.vector_norm(x)), "norm_b": norm_b}


def propagator_true_residual(propagator: np.ndarray, gauge: np.ndarray, kappa: float,
                             lat: Sequence[int], tsrc: int,
                             max_rows: int = 12) -> Dict[str, float]:
    """整条传播子的最坏真残差。

    12 分量模式逐行对各自单分量点源复核；1 分量模式对应“全 12 分量置 1”的点源。
    ``max_rows`` 限制参与复核的行数（大格点上按需抽样，降低非求解开销）。
    """
    if propagator.shape[0] == COMPONENTS:
        rows = list(range(COMPONENTS))[:max(1, int(max_rows))]
        sources = [point_source_xyzt(lat, tsrc, row, np.complex64) for row in rows]
    else:
        rows = [0]
        sources = [point_source_xyzt(lat, tsrc, None, np.complex64)]
    worst = 0.0
    for row, source in zip(rows, sources):
        worst = max(worst, reference_true_residual(propagator[row], source, gauge, kappa)["residual"])
    return {"residual": worst, "rows_checked": len(rows)}


# --------------------------------------------------------------------------
# HDF5 结果读写
# --------------------------------------------------------------------------
def save_solution(path: Path, propagators: Dict[int, np.ndarray], meta: Dict[str, Any]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for tsrc, array in sorted(propagators.items()):
            handle.create_dataset(f"tsrc_{tsrc}", data=array)
        for key, value in meta.items():
            handle.attrs[key] = value if not isinstance(value, (list, dict)) else json.dumps(value)


def load_solutions(path: Path) -> Dict[int, np.ndarray]:
    import h5py

    out: Dict[int, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            if key.startswith("tsrc_"):
                out[int(key.split("_")[1])] = np.asarray(handle[key][...])
    return out


# --------------------------------------------------------------------------
# QUDA 侧
# --------------------------------------------------------------------------
def run_quda(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """原生 pyquda(QUDA) Clover MultiGrid 求解,每个 tsrc × 12 分量独立求解。"""
    import cupy as cp
    import threading
    import pyquda
    from pyquda.dirac import CloverWilsonDirac
    from pyquda.enum_quda import QudaBoolean, QudaPrecision
    from pyquda.field import LatticeFermion, LatticeGauge, LatticeInfo

    from pyqcu.testing.quda.common import pyqcu_gauge_to_quda

    spec = precision_spec(args.precision)
    lat = tuple(int(v) for v in args.lat)
    grid = tuple(int(v) for v in args.mpi_grid)
    # 同一 run-dir 复用（换格子/换 MPI 规模）时必须先清掉本角色旧产物，
    # 否则 compare 会把不同格点的文件混在一起（实测 T=16 与 T=32 混入）。
    worker_index = int(getattr(args, "quda_worker_index", 0))
    quda_workers = int(getattr(args, "quda_workers", 1))
    self_label = (worker_index if quda_workers > 1
                  else int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    # 仅清理本 worker 自己的旧产物：多 worker 顺序启动时不得互删（实测会被抹掉）。
    for pattern in (f"solutions_rank{self_label}.h5", f"metrics_rank{self_label}.json",
                    f"c2_rank{self_label}.npz"):
        for stale in (run_dir / "quda").glob(pattern):
            stale.unlink()

    # 输出文件名标签：worker 模式下用 worker 序号（每个单 rank 进程的 MPI rank 都是 0，
    # 不区分会互相覆盖——实测只留下最后一个 worker 的 tsrc）。
    rank_label = (worker_index if quda_workers > 1
                  else int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    all_tsrc = parse_tsrc(args.tsrc, lat[3])
    if int(getattr(args, "quda_workers", 1)) > 1:
        idx = int(getattr(args, "quda_worker_index", 0))
        tsrc_list = [t for pos, t in enumerate(all_tsrc) if pos % int(args.quda_workers) == idx]
    else:
        tsrc_list = all_tsrc
    gauge = load_gauge_xyzt(run_dir / "inputs" / "gauge_xyzt.h5", args.precision)

    if args.t_boundary < 0 and args.role == "compare":
        raise RunFailure("QCU C++ 后端为周期时间边界，对照模式不允许 --t-boundary -1")

    tuning_dir = Path(args.quda_resource) if args.quda_resource else run_dir / "quda" / "tune"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    tuning_was_cached = any(tuning_dir.iterdir()) if tuning_dir.is_dir() else False
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count < 1:
        raise RunFailure("未检测到 CUDA 设备")
    device_id = int(os.environ.get("PYQCU_QUDA_DEVICE",
                                   args.quda_device + worker_index))
    if not 0 <= device_id < device_count:
        raise RunFailure(f"QUDA device {device_id} 不可见（可见 {device_count} 张）")
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    mpi_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    if mpi_size > 1:
        # 多 rank 时由 PyQUDA 按 local rank 绑定设备，避免所有 rank 抢同一张卡。
        device_id = int(os.environ.get("PYQCU_QUDA_DEVICE", device_id + local_rank))
        if not 0 <= device_id < device_count:
            raise RunFailure(f"QUDA rank {local_rank} 绑定的 device {device_id} 不可见")
        cp.cuda.Device(device_id).use()
    else:
        cp.cuda.Device(device_id).use()
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    compute_capability = (int(props["major"]), int(props["minor"]))

    pyquda.init(grid_size=list(grid), latt_size=list(lat), backend="cupy", backend_target="cuda",
                resource_path=str(tuning_dir), enable_nvshmem=False,
                enable_tuning=not args.quda_no_tuning, enable_device_memory_pool=True,
                enable_pinned_memory_pool=True)
    info = LatticeInfo(list(lat), t_boundary=int(args.t_boundary), anisotropy=1.0)
    coord = pyquda.getGridCoord()
    local_shape = (info.Lx, info.Ly, info.Lz, info.Lt)
    spatial_origin = tuple(int(v) for v in coord[:3])
    if spatial_origin != (0, 0, 0):
        raise RunFailure(f"点源要求空间原点所在的 rank 负责收缩；当前 grid_coord={tuple(coord)}")

    gx, gy, gz, gt = (int(v) for v in coord)
    local_gauge = np.ascontiguousarray(
        gauge[:, :, :, gx * info.Lx:(gx + 1) * info.Lx, gy * info.Ly:(gy + 1) * info.Ly,
              gz * info.Lz:(gz + 1) * info.Lz, gt * info.Lt:(gt + 1) * info.Lt])
    qdp = pyqcu_gauge_to_quda(local_gauge).astype(np.complex128, copy=False)
    gauge_field = LatticeGauge(info, 4, cp.asarray(info.evenodd(qdp, True)))

    blocks = [[int(v) for v in block] for block in args.mg_block]
    dirac = CloverWilsonDirac(info, float(args.mass), float(args.tol), int(args.max_iter),
                              clover_csw=1.0, multigrid=blocks)
    precision = getattr(QudaPrecision, spec["quda"])
    dirac.setPrecision(cuda=precision, sloppy=precision, precondition=precision,
                       refinement_sloppy=precision, eigensolver=precision)
    if args.quda_nvec is not None and dirac.multigrid is not None \
            and dirac.multigrid.param is not None:
        param = dirac.multigrid.param
        column = [int(v) for v in param.n_vec]
        param.n_vec = [int(args.quda_nvec)] * len(column)
        observed = [int(v) for v in param.n_vec]
        if observed != [int(args.quda_nvec)] * len(column):
            raise RunFailure(f"QUDA n_vec 写回失败：{observed}")
    mg_enabled_mma = compute_capability[0] >= 7 and not args.quda_no_mma
    # 公平性记录（只读）：QUDA coarse_color = n_vec * (fine_spin / spin_block_size)。
    # 不对 QUDA 参数做运行中改写——改写顺序/整列语义出错会被 QUDA 拒绝
    # （实测写入 cycle_type 列表 → "Multigrid cycle type 0 not supported"）。
    alignment: Dict[str, Any] = {}
    if dirac.multigrid is not None and dirac.multigrid.param is not None:
        param = dirac.multigrid.param
        try:
            spin_block = [int(v) for v in param.spin_block_size]
            coarse_spin = 4 // max(1, spin_block[0]) if spin_block else 2
            n_vec = [int(v) for v in param.n_vec]
            alignment = {
                "quda_n_level": int(param.n_level),
                "quda_n_vec": n_vec,
                "quda_spin_block_size": spin_block,
                "quda_coarse_color_level0": n_vec[0] * coarse_spin,
                "quda_cycle_type": [str(v) for v in param.cycle_type],
                "quda_nu_pre": [int(v) for v in param.nu_pre],
                "quda_nu_post": [int(v) for v in param.nu_post],
                "quda_geo_block_size": [[int(v) for v in row] for row in param.geo_block_size],
                "qcu_mg_levels": int(args.mg_levels),
                "qcu_mg_dof": [int(v) for v in args.mg_dof],
                "qcu_mg_block": [[int(v) for v in b] for b in args.mg_block],
            }
            alignment["coarse_color_match"] = bool(
                alignment["quda_coarse_color_level0"] == int(args.mg_dof[1]))
            alignment["block_match"] = bool(
                list(alignment["quda_geo_block_size"][0][:4]) == [int(v) for v in args.mg_block[0]])
            alignment["levels_match"] = bool(int(param.n_level) == int(args.mg_levels))
        except Exception as exc:
            alignment = {"alignment_error": repr(exc)}
    if dirac.multigrid is not None and dirac.multigrid.param is not None and not mg_enabled_mma:
        param = dirac.multigrid.param
        for field in ("setup_use_mma", "dslash_use_mma", "transfer_use_mma"):
            column = getattr(param, field)
            setattr(param, field, [QudaBoolean.QUDA_BOOLEAN_FALSE] * len(column))

    sync = lambda: cp.cuda.runtime.deviceSynchronize()
    # 峰值显存采样（目标要求"直至达到显存上限"需可审计的数字）
    mem_peak = {"used_bytes": 0, "total_bytes": 0, "samples": 0}
    _stop = threading.Event()

    def _sample_memory() -> None:
        while not _stop.is_set():
            try:
                free_b, total_b = cp.cuda.Device(device_id).mem_info
                used = int(total_b) - int(free_b)
                mem_peak["used_bytes"] = max(mem_peak["used_bytes"], used)
                mem_peak["total_bytes"] = int(total_b)
                mem_peak["samples"] += 1
            except Exception:
                pass
            _stop.wait(0.2)

    sampler = threading.Thread(target=_sample_memory, daemon=True)
    sampler.start()
    timer = StepTimer(sync=sync)
    timer.measure("loadGauge", dirac.loadGauge, gauge_field)

    # MG/求解器口径：迭代数语义与 QCU 不同，必须随结果一起记录。
    mg_config: Dict[str, Any] = {}
    try:
        param = dirac.multigrid.param if dirac.multigrid is not None else None
        if param is not None:
            mg_config = {
                "n_level": int(param.n_level),
                "n_vec": [int(v) for v in param.n_vec],
                "cycle_type": str(param.cycle_type),
                "nu_pre": [int(v) for v in param.nu_pre],
                "nu_post": [int(v) for v in param.nu_post],
                "coarse_solver": str(param.coarse_solver),
                "setup_type": str(param.setup_type),
            }
        mg_config["inv_type"] = str(dirac.invert_param.inv_type)
        mg_config["tol"] = float(dirac.invert_param.tol)
        mg_config["maxiter"] = int(dirac.invert_param.maxiter)
    except Exception as exc:  # 记录失败不阻断求解
        mg_config["probe_error"] = repr(exc)

    warmup_seconds = 0.0
    if args.quda_warmup > 0:
        # 首轮 QUDA 会重调所有 kernel（日志 All kernels will be re-tuned），
        # 必须与计时求解分离，否则耗时不可信。
        warmup_source = np.zeros((4, 3, *local_shape), dtype=spec["np"])
        warmup_source[:, :, 0, 0, 0, max(0, (tsrc_list[0] - gt * info.Lt))] = 1.0
        tzyxsc = np.ascontiguousarray(np.transpose(warmup_source.astype(np.complex128),
                                                   (5, 4, 3, 2, 0, 1)))
        started = time.perf_counter()
        for _ in range(int(args.quda_warmup)):
            dirac.invert(LatticeFermion(info, cp.asarray(info.evenodd(tzyxsc, False))))
        sync()
        warmup_seconds = time.perf_counter() - started
        timer.add("warmup", warmup_seconds)

    propagators: Dict[int, np.ndarray] = {}
    iterations: Dict[int, int] = {}
    solve_seconds: Dict[int, float] = {}
    true_residuals: Dict[int, float] = {}
    patterns = [(spin, color) for spin in range(4) for color in range(3)] \
        if args.components == 12 else [None]
    local_phase = momentum_phase(lat, args.mom, coord, local_shape)
    c2_store: Dict[int, np.ndarray] = {}
    c2_pion_store: Dict[int, np.ndarray] = {}
    g5, gamma = pyqcu_gammas()

    for tsrc in tsrc_list:
        local_t = tsrc - gt * info.Lt
        owns_tsrc = gt * info.Lt <= tsrc < (gt + 1) * info.Lt
        if owns_tsrc:
            local_t = tsrc - gt * info.Lt
        else:
            # 非属主 rank 仍须参加同一集合调用序列（QUDA invert 是集合操作，
            # 跳过会导致属主 rank 在 MG 粗层通信中永久等待），源取本地 t=0。
            local_t = 0
        solved: List[np.ndarray] = []
        worst_residual = 0.0
        for pattern in patterns:
            source = np.zeros((4, 3, *local_shape), dtype=spec["np"])
            if pattern is None:
                source[:, :, 0, 0, 0, local_t] = 1.0
            else:
                spin, color = pattern
                source[spin, color, 0, 0, 0, local_t] = 1.0
            tzyxsc = np.ascontiguousarray(np.transpose(source.astype(np.complex128),
                                                       (5, 4, 3, 2, 0, 1)))
            rhs = LatticeFermion(info, cp.asarray(info.evenodd(tzyxsc, False)))
            started = time.perf_counter()
            solution = dirac.invert(rhs)
            sync()
            elapsed = time.perf_counter() - started
            solve_seconds[tsrc] = solve_seconds.get(tsrc, 0.0) + elapsed
            iterations[tsrc] = iterations.get(tsrc, 0) + int(dirac.invert_param.iter)
            worst_residual = max(
                worst_residual,
                float(np.asarray(dirac.invert_param.true_res).ravel()[0]))
            lexico = np.asarray(info.lexico(cp.asnumpy(solution.data), False))
            solved.append(np.ascontiguousarray(np.transpose(lexico, (4, 5, 3, 2, 1, 0))))
        propagator = np.stack(solved, axis=0).astype(spec["np"])
        if not owns_tsrc:
            continue
        propagators[tsrc] = propagator
        true_residuals[tsrc] = float(worst_residual)
        if args.components == COMPONENTS and not args.skip_c2:
            contraction = meson_c2(propagator, gamma, g5, local_phase)
            c2_store[tsrc] = contraction["c2_spin"]
            c2_pion_store[tsrc] = contraction["c2_pion"]

    _stop.set()
    sampler.join(timeout=2.0)

    # 多 rank：所有 rank 按同一 tsrc 序列参加 gather（集合调用必须对称，
    # 否则属主 rank 会在 gather 上永久等待——实测 rank1 无 tsrc 直接退出即死锁），
    # 之后统一 Barrier，再由 rank0 落全局解。
    if mpi_size > 1:
        from mpi4py import MPI as _MPI

        comm = _MPI.COMM_WORLD
        rank = comm.Get_rank()
        for tsrc in tsrc_list:
            local = propagators.get(tsrc)
            parts = comm.gather((list(coord), local), root=0)
            if rank == 0:
                merged = None
                for grid_coord_part, part in parts:
                    if part is None:
                        continue
                    if merged is None:
                        merged = np.zeros((part.shape[0], 4, 3, *lat), dtype=spec["np"])
                    gx, gy, gz, gt = (int(v) for v in grid_coord_part)
                    lx, ly, lz, lt = info.Lx, info.Ly, info.Lz, info.Lt
                    merged[:, :, gx * lx:(gx + 1) * lx, gy * ly:(gy + 1) * ly,
                           gz * lz:(gz + 1) * lz, gt * lt:(gt + 1) * lt] = part
                if merged is not None:
                    propagators[tsrc] = merged
            elif rank != 0:
                propagators.pop(tsrc, None)
        comm.Barrier()

    out_dir = run_dir / "quda"
    save_solution(out_dir / f"solutions_rank{rank_label}.h5",
                  propagators, {"role": "quda", "precision": spec["name"], "grid": list(grid),
                                "grid_coord": list(coord), "t_boundary": int(args.t_boundary)})
    if c2_store:
        np.savez_compressed(
            out_dir / f"c2_rank{rank_label}.npz",
            **{f"c2_spin_{k}": v for k, v in c2_store.items()},
            **{f"c2_pion_{k}": v for k, v in c2_pion_store.items()})
    metrics = {
        "backend": "quda",
        "quda_version": getattr(pyquda, "__version__", "unknown"),
        "precision": spec["name"],
        "lattice": list(lat), "mpi_grid": list(grid), "grid_coord": list(coord),
        "mass": float(args.mass), "kappa": float(1.0 / (2 * args.mass + 8.0)),
        "clover_csw": 1.0, "t_boundary": int(args.t_boundary),
        "multigrid": blocks, "mma": mg_enabled_mma,
        "compute_capability": list(compute_capability),
        "load_gauge_seconds": dict(timer.timings).get("loadGauge"),
        "multigrid_config": mg_config,
        "mg_alignment": alignment,
        "tuning_enabled": not bool(args.quda_no_tuning),
        "tuning_dir": str(tuning_dir),
        "tuning_was_cached": bool(tuning_was_cached),
        "warmup_seconds": float(warmup_seconds),
        "warmup_solves": int(args.quda_warmup),
        "peak_device_memory_bytes": int(mem_peak["used_bytes"]),
        "device_memory_total_bytes": int(mem_peak["total_bytes"]),
        "memory_samples": int(mem_peak["samples"]),
        "solve_total_iterations": int(sum(iterations.values())),
        "solve_total_seconds": float(sum(solve_seconds.values())),
        "quda_internal_secs_per_tsrc": {
            tsrc: float(getattr(dirac.invert_param, "secs", 0.0)) for tsrc in solve_seconds},
        "solve_seconds_per_tsrc": solve_seconds,
        "iterations_per_tsrc": iterations,
        "quda_true_residual_per_tsrc": true_residuals,
        "components": int(args.components),
        "role": args.role,
    }
    write_json(out_dir / f"metrics_rank{rank_label}.json",
               metrics)
    print(f"[quda] tsrc={tsrc_list} iters={metrics['solve_total_iterations']} "
          f"solve={metrics['solve_total_seconds']:.3f}s", flush=True)
    # 产物已落盘后再清理：本机 QUDA 在 MPI 2-rank 下 finalize/collective 清理
    # 会挂住（实测 >20 min），因此多 rank 默认跳过清理并直接退出，保证驱动
    # 进程能继续对照；需要完整清理时显式 --quda-cleanup。
    if args.quda_cleanup:
        try:
            dirac.freeGauge()
            if dirac.multigrid is not None:
                dirac.multigrid.destroy()
        except Exception as exc:
            print(f"[quda] cleanup 异常（已忽略）: {exc!r}", flush=True)
    elif mpi_size > 1:
        print(f"[quda] rank {rank_label}: 跳过 QUDA 清理（避免 MPI finalize 挂起），"
              "产物已写盘", flush=True)
        sys.stdout.flush()
        # 不做 MPI.Barrier/Finalize：rank 之间完成任务时间不对称，先到达者
        # Finalize 会让后到者在 Barrier 上永久等待（实测）。直接退出即可，
        # 各 rank 产物已在退出前落盘，驱动侧以产物齐全判定成功。
        os._exit(0)
    return metrics


# --------------------------------------------------------------------------
# QCU 侧
# --------------------------------------------------------------------------
def _qcu_configure(params, argv, lat, mass, tol, levels, dof_list, block, torch_dtype):
    import pyqcu.cuda.define as define

    params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_], params[define._LAT_T_] = lat
    params[define._LAT_XYZT_] = lat[0] * lat[1] * lat[2] * lat[3]
    params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[define._GRID_T_] = 1, 1, 1, 1
    params[define._PARITY_] = 0
    params[define._NODE_RANK_] = 0
    params[define._NODE_SIZE_] = 1
    params[define._DAGGER_] = 0
    params[define._MAX_ITER_] = 1000
    params[define._DATA_TYPE_] = define.epytd(torch_dtype)
    params[define._VERBOSE_] = 0
    params[define._SEED_] = 42
    params[define._TEST_IN_CPU_] = 0
    params[define._MG_NUM_LEVEL_] = int(levels)
    if levels >= 2:
        params[define._MG_LEVEL1_E_] = dof_list[1]
        params[define._MG_LEVEL1_X_] = lat[0] // block[0]
        params[define._MG_LEVEL1_Y_] = lat[1] // block[1]
        params[define._MG_LEVEL1_Z_] = lat[2] // block[2]
        params[define._MG_LEVEL1_T_] = lat[3] // (2 * block[3])
        params[define._MG_LEVEL1_MAX_ITER_] = 200
        params[define._MG_LEVEL1_DATA_TYPE_] = params[define._DATA_TYPE_]
        params[define._MG_LEVEL1_NUM_RESTART_] = 10
    if levels >= 3:
        params[define._MG_LEVEL2_E_] = dof_list[2]
        params[define._MG_LEVEL2_X_] = lat[0] // (block[0] * block[0])
        params[define._MG_LEVEL2_Y_] = lat[1] // (block[1] * block[1])
        params[define._MG_LEVEL2_Z_] = lat[2] // (block[2] * block[2])
        params[define._MG_LEVEL2_T_] = lat[3] // (4 * block[3])
        params[define._MG_LEVEL2_MAX_ITER_] = 200
        params[define._MG_LEVEL2_DATA_TYPE_] = params[define._DATA_TYPE_]
        params[define._MG_LEVEL2_NUM_RESTART_] = 3
    argv[define._MASS_] = float(mass)
    argv[define._ATOL_] = float(tol)
    argv[define._SIGMA_] = 0.1
    if levels >= 2:
        argv[define._MG_LEVEL1_ATOL_] = float(tol) * 1e3
    if levels >= 3:
        argv[define._MG_LEVEL2_ATOL_] = float(tol) * 1e3
    return argv.to(dtype=define.real_dtype(define.dtype(params[define._DATA_TYPE_])))


def _qcu_build_coarse(args, full_gauge_cpu, mass, primary_device):
    """构建（或按 gauge 指纹复用）C++ MG 粗层算子。"""
    import torch
    from pyqcu.torch import dslash
    from pyqcu.cuda._multi_gpu import _coarse_cache_paths, build_schur_levels

    lat = [int(v) for v in args.lat]
    kappa = 1.0 / (2.0 * mass + 8.0)
    dtype = getattr(torch, precision_spec(args.precision)["torch"])
    u_full = torch.from_numpy(np.asarray(full_gauge_cpu)).to(dtype=dtype, device=primary_device)
    clover = dslash.make_clover(u_full, kappa=torch.Tensor([kappa])).to(dtype)
    op = dslash.operator(U=u_full, clover_term=clover, kappa=torch.Tensor([kappa]),
                         support_parity=True, verbose=False)
    coarse_cache = None
    if args.mg_cache_dir:
        base = Path(args.mg_cache_dir).expanduser()
        fingerprint = sha256_array(np.asarray(full_gauge_cpu))
        # 粗层算子只在同一 gauge 下有效：缓存必须落在按 gauge 指纹隔离的子目录，
        # 否则会误加载仓库里为别的 gauge 预生成的 h5（静默给出错误的粗空间）。
        if args.reuse_coarse:
            coarse_cache = str(base / fingerprint[:16])
            Path(coarse_cache).mkdir(parents=True, exist_ok=True)
    coarse = build_schur_levels(op, op.matvec_parity, int(args.mg_levels), args.mg_dof,
                                [int(v) for v in args.mg_block[0]], lat, args.mg_dof[1],
                                dtype, primary_device, nv_iters=int(args.mg_nv_iters),
                                use_cache=coarse_cache is not None, cache_dir=coarse_cache,
                                verbose=bool(args.verbose))
    return list(zip(*coarse)) if coarse and coarse[0] else []


def run_qcu(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """PyQCU C++ Clover MultiGrid 求解（一线程一卡，tsrc 轮转分派）。"""
    import torch
    from pyqcu import tools
    from pyqcu.cuda import qcu
    import pyqcu.cuda.define as define
    from pyqcu.cuda.define import argv as module_argv
    from pyqcu.cuda.define import params as module_params
    from pyqcu.cuda.define import set_ptrs as module_set_ptrs

    spec = precision_spec(args.precision)
    dtype = getattr(torch, spec["torch"])
    lat = [int(v) for v in args.lat]
    for pattern in ("solutions_dev*.h5", "c2_dev*.npz", "metrics.json"):
        for stale in (run_dir / "qcu").glob(pattern):
            stale.unlink()
    tsrc_list = parse_tsrc(args.tsrc, lat[3])
    gauge_full = load_gauge_xyzt(run_dir / "inputs" / "gauge_xyzt.h5", args.precision)
    gauge_compact = pyqcu_gauge_to_qcu_compact(gauge_full)

    devices = [int(v) for v in args.qcu_devices] or list(range(torch.cuda.device_count()))
    if not devices or not torch.cuda.is_available():
        raise RunFailure("QCU 求解需要可见 CUDA 设备")
    if args.qcu_threads is not None:
        devices = devices[: int(args.qcu_threads)]
    primary = torch.device(f"cuda:{devices[0]}")
    torch.cuda.set_device(devices[0])
    torch.cuda.synchronize()
    for device_id in devices:
        try:
            torch.cuda.reset_peak_memory_stats(device_id)
        except Exception:
            pass

    log_dir = run_dir / "qcu" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "clover_multigrid.log"
    os.environ["QCU_LOG_DIR"] = str(log_dir)

    timer = StepTimer(sync=lambda: torch.cuda.synchronize())
    coarse = timer.measure("coarse_setup", _qcu_build_coarse, args, gauge_full,
                           float(args.mass), primary)
    if args.mg_levels >= 2 and not coarse:
        raise RunFailure("粗层算子构建失败（--mg-levels>=2 需要非空 coarse）")

    shared = {
        "g": torch.from_numpy(gauge_compact).to(primary),
        "coarse_cpu": [(t[0].detach().cpu(), t[1].detach().cpu(), t[2].detach().cpu(), t[3].detach().cpu())
                       for t in coarse],
    }
    patterns = [(spin, color) for spin in range(4) for color in range(3)] \
        if args.components == 12 else [None]

    def worker(device_id: int, indices: List[int]) -> Dict[str, Any]:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        params = module_params.clone()
        argv = module_argv.clone()
        set_ptrs = module_set_ptrs.clone()
        argv = _qcu_configure(params, argv, lat, float(args.mass), float(args.tol),
                              int(args.mg_levels), args.mg_dof, args.mg_block[0], dtype)
        g = shared["g"].to(device)
        ls = define.lat_shape(params)
        # Clover 项由 C++ 后端按注入的 gauge 现场构造（每张卡各自一份）。
        clover_even = torch.empty([4, 3, 4, 3, *ls], dtype=dtype, device=device)
        clover_even_inv = torch.empty_like(clover_even)
        clover_odd = torch.empty_like(clover_even)
        clover_odd_inv = torch.empty_like(clover_even)
        params[define._SET_INDEX_] = 0
        params[define._SET_PLAN_] = 2
        for parity, out, out_inv in ((0, clover_even, clover_even_inv),
                                     (1, clover_odd, clover_odd_inv)):
            params[define._PARITY_] = parity
            qcu.applyInitQcu(set_ptrs, params, argv)
            qcu.applyCloversQcu(out, out_inv, g, set_ptrs, params)
            qcu.applyEndQcu(set_ptrs, params)
        coarse_dev = []
        for level, (lonv, hnn, hdg, sit) in enumerate(shared["coarse_cpu"]):
            tensors = [t.to(device) for t in (lonv, hnn, hdg, sit)]
            coarse_dev.extend(tensors)
            base = 30 + 4 * level
            for offset, tensor in enumerate(tensors):
                set_ptrs[base + offset] = tensor.data_ptr()

        phases = momentum_phase(lat, args.mom)
        g5, gamma = pyqcu_gammas()
        results: Dict[int, np.ndarray] = {}
        iterations: Dict[int, int] = {}
        seconds: Dict[int, float] = {}
        c2_spin: Dict[int, np.ndarray] = {}
        c2_pion: Dict[int, np.ndarray] = {}
        for tsrc in indices:
            solved = []
            total_it = 0
            total_s = 0.0
            for pattern in patterns:
                source = np.zeros((4, 3, *lat), dtype=spec["np"])
                if pattern is None:
                    source[:, :, 0, 0, 0, tsrc] = 1.0
                else:
                    spin, color = pattern
                    source[spin, color, 0, 0, 0, tsrc] = 1.0
                rhs = _compact_source(source, tools).to(device)
                out = torch.empty_like(rhs)
                params[define._SET_INDEX_] = 0
                params[define._SET_PLAN_] = 1
                params[define._VERBOSE_] = 0
                qcu.applyInitQcu(set_ptrs, params, argv)
                torch.cuda.synchronize()
                started = time.perf_counter()
                qcu.applyCloverMultigridQcu(out, rhs, g, clover_even, clover_odd,
                                            clover_even_inv,
                                            clover_odd_inv, set_ptrs, params)
                torch.cuda.synchronize()
                total_s += time.perf_counter() - started
                qcu.applyEndQcu(set_ptrs, params)
                total_it += _last_total_iterations(log_path)
                full = np.asarray(tools.poooxyzt2oooxyzt(out.cpu()))
                solved.append(full)
            propagator = np.stack(solved, axis=0).astype(spec["np"])
            results[tsrc] = propagator
            iterations[tsrc] = int(total_it)
            seconds[tsrc] = float(total_s)
            if args.components == COMPONENTS and not args.skip_c2:
                contraction = meson_c2(propagator, gamma, g5, phases)
                c2_spin[tsrc] = contraction["c2_spin"]
                c2_pion[tsrc] = contraction["c2_pion"]
        return {"device": device_id, "propagators": results, "iterations": iterations,
                "seconds": seconds, "c2_spin": c2_spin, "c2_pion": c2_pion,
                "coarse_ops": coarse_dev}

    assignments = {device_id: [] for device_id in devices}
    for position, tsrc in enumerate(tsrc_list):
        assignments[devices[position % len(devices)]].append(tsrc)

    replica: Dict[int, Dict[str, Any]] = {}
    if len(devices) > 1 and args.qcu_verify:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(worker, device_id, [tsrc_list[0]]): device_id for device_id in devices}
            for future, device_id in futures.items():
                replica[device_id] = future.result()

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {pool.submit(worker, device_id, sorted(tsrc)): device_id
                   for device_id, tsrc in assignments.items() if tsrc}
        outputs = [future.result() for future in futures]

    peak_memory = {}
    for device_id in devices:
        try:
            peak_memory[str(device_id)] = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_id)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_id)),
                "total_bytes": int(torch.cuda.get_device_properties(device_id).total_memory),
            }
        except Exception as exc:
            peak_memory[str(device_id)] = {"error": repr(exc)}

    out_dir = run_dir / "qcu"
    all_iterations: Dict[int, int] = {}
    all_seconds: Dict[int, float] = {}
    for output in outputs:
        all_iterations.update(output["iterations"])
        all_seconds.update(output["seconds"])
        save_solution(out_dir / f"solutions_dev{output['device']}.h5", output["propagators"],
                      {"role": "qcu", "precision": spec["name"], "device": output["device"],
                       "mg_levels": int(args.mg_levels), "mg_dof": list(args.mg_dof)})
        if output["c2_spin"]:
            np.savez_compressed(out_dir / f"c2_dev{output['device']}.npz",
                                **{f"c2_spin_{k}": v for k, v in output["c2_spin"].items()},
                                **{f"c2_pion_{k}": v for k, v in output["c2_pion"].items()})

    consistency = {}
    for device_id, output in replica.items():
        reference = replica[devices[0]]["propagators"][tsrc_list[0]]
        current = output["propagators"][tsrc_list[0]]
        consistency[str(device_id)] = float(
            np.linalg.norm(current - reference) / (np.linalg.norm(reference) + 1e-30))

    metrics = {
        "backend": "qcu",
        "precision": spec["name"],
        "lattice": lat, "mass": float(args.mass), "kappa": float(1.0 / (2 * args.mass + 8.0)),
        "mg_levels": int(args.mg_levels), "mg_dof": list(args.mg_dof),
        "mg_block": [list(b) for b in args.mg_block],
        "devices": devices, "threads": len(devices),
        "coarse_setup_seconds": dict(timer.timings).get("coarse_setup"),
        "coarse_cache": bool(args.mg_cache_dir and args.reuse_coarse),
        "solve_total_iterations": int(sum(all_iterations.values())),
        "solve_total_seconds": float(sum(all_seconds.values())),
        "iterations_per_tsrc": all_iterations,
        "solve_seconds_per_tsrc": all_seconds,
        "multi_gpu_consistency": consistency,
        "components": int(args.components),
        "peak_memory": peak_memory,
        "role": args.role,
    }
    write_json(out_dir / "metrics.json", metrics)
    print(f"[qcu ] tsrc={tsrc_list} iters={metrics['solve_total_iterations']} "
          f"solve={metrics['solve_total_seconds']:.3f}s coarse_setup="
          f"{metrics['coarse_setup_seconds']:.3f}s")
    return metrics


def _compact_source(source_xyzt: np.ndarray, tools):
    """PyQCU 全格点 [4,3,X,Y,Z,T] → C++ [2,4,3,X,Y,Z,T/2]。"""
    import torch

    return tools.oooxyzt2poooxyzt(torch.from_numpy(np.ascontiguousarray(source_xyzt)))


def _last_total_iterations(log_path: Path) -> int:
    """读取 C++ MG 日志中最近一次 ``Total iterations: N``（每次求解一行）。"""
    if not Path(log_path).is_file():
        raise RunFailure(f"缺少 QCU MG 日志：{log_path}")
    last = None
    with Path(log_path).open("r", errors="ignore") as handle:
        for line in handle:
            marker = "Total iterations:"
            if marker in line:
                try:
                    last = int(line.split(marker, 1)[1].strip().split()[0])
                except (IndexError, ValueError):
                    continue
    if last is None:
        raise RunFailure(f"日志中没有 Total iterations 记录：{log_path}")
    return int(last)


# --------------------------------------------------------------------------
# 对照与汇总
# --------------------------------------------------------------------------
def _gather_solutions(directory: Path, pattern: str) -> Dict[int, np.ndarray]:
    merged: Dict[int, np.ndarray] = {}
    for path in sorted(directory.glob(pattern)):
        for tsrc, array in load_solutions(path).items():
            if tsrc in merged:
                raise RunFailure(f"tsrc={tsrc} 在多个文件中重复: {path}")
            merged[tsrc] = array
    return merged


def compare(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    from pyqcu.testing.quda.common import pyqcu_fermion_to_quda

    spec = precision_spec(args.precision)
    gauge = load_gauge_xyzt(run_dir / "inputs" / "gauge_xyzt.h5", args.precision).astype(np.complex128)
    kappa = 1.0 / (2.0 * args.mass + 8.0)
    expected_scale = 1.0 / (2.0 * kappa)
    lat = [int(v) for v in args.lat]
    qcu_solutions = _gather_solutions(run_dir / "qcu", "solutions_dev*.h5")
    quda_solutions = _gather_solutions(run_dir / "quda", "solutions_rank*.h5")
    if not qcu_solutions or not quda_solutions:
        raise RunFailure("缺少 qcu/quda 传播子，无法对照")

    requested = parse_tsrc(args.tsrc, int(args.lat[3]))
    comparison: Dict[str, Any] = {"per_tsrc": {}, "lattice": list(args.lat),
                                  "requested_tsrc": requested}
    scales: List[float] = []
    gamma, g5 = pyqcu_gammas()
    for tsrc in sorted(set(qcu_solutions) & set(quda_solutions)):
        x_qcu = np.asarray(qcu_solutions[tsrc]).astype(np.complex128)
        x_quda = np.asarray(quda_solutions[tsrc]).astype(np.complex128)
        if x_qcu.shape != x_quda.shape:
            raise RunFailure(f"tsrc={tsrc} 形状不一致 {x_qcu.shape} vs {x_quda.shape}")
        numerator = np.vdot(x_quda.ravel(), x_qcu.ravel())
        denominator = np.vdot(x_quda.ravel(), x_quda.ravel())
        scale = numerator / denominator
        after = np.linalg.norm(scale * x_quda - x_qcu) / (np.linalg.norm(x_qcu) + 1e-30)
        if args.skip_residual:
            reference = scaled_reference = {"residual": 0.0, "rows_checked": 0}
        else:
            reference = propagator_true_residual(x_qcu, gauge, kappa, lat, tsrc,
                                                 args.residual_rows)
            scaled_reference = propagator_true_residual(
                (scale * x_quda).astype(spec["np"]), gauge, kappa, lat, tsrc,
                args.residual_rows)
        phase = momentum_phase(args.lat, args.mom)
        if args.components == COMPONENTS and not args.skip_c2:
            c2_qcu = meson_c2(x_qcu.astype(spec["np"]), gamma, g5, phase)
            c2_quda = meson_c2((scale * x_quda).astype(spec["np"]), gamma, g5, phase)
            c2_diff: Optional[float] = float(
                np.linalg.norm(c2_qcu["c2_pion"] - c2_quda["c2_pion"]) /
                (np.linalg.norm(c2_qcu["c2_pion"]) + 1e-30))
            c2_qcu_store, c2_quda_store = c2_qcu["c2_pion"], c2_quda["c2_pion"]
        else:
            c2_diff, c2_qcu_store, c2_quda_store = None, None, None
        scales.append(scale)
        # 统一转到 QUDA/QDP 排布 [2,T,Z,Y,X/2,4,3] 做最终差异对比
        # （spin/color、tzyx 轴序与 x 轴棋盘压缩都由 common.py 的显式转换给出）。
        scaled_qcu = x_qcu.astype(spec["np"])
        scaled_quda = (scale * x_quda).astype(spec["np"])
        qdp_qcu = np.stack([pyqcu_fermion_to_quda(row) for row in scaled_qcu], axis=0)
        qdp_quda = np.stack([pyqcu_fermion_to_quda(row) for row in scaled_quda], axis=0)
        delta = qdp_qcu - qdp_quda
        qdp_diff = {
            "layout": "qdp_[row,parity,T,Z,Y,X/2,spin,color]",
            "shape": list(qdp_qcu.shape),
            "relative_l2": float(np.linalg.norm(delta.ravel()) /
                                 (np.linalg.norm(qdp_qcu.ravel()) + 1e-30)),
            "max_abs": float(np.max(np.abs(delta))),
            "per_row_relative_l2": [
                float(np.linalg.norm(delta[row].ravel()) /
                      (np.linalg.norm(qdp_qcu[row].ravel()) + 1e-30))
                for row in range(qdp_qcu.shape[0])],
            "qcu_norm": float(np.linalg.norm(qdp_qcu.ravel())),
            "quda_scaled_norm": float(np.linalg.norm(qdp_quda.ravel())),
        }
        compare_dir = run_dir / "compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        import h5py

        with h5py.File(compare_dir / f"compare_qdp_tsrc{tsrc}.h5", "w") as handle:
            handle.create_dataset("qcu_qdp", data=qdp_qcu)
            handle.create_dataset("quda_qdp_scaled", data=qdp_quda)
            handle.create_dataset("difference", data=delta)
            for key, value in qdp_diff.items():
                handle.attrs[key] = value if not isinstance(value, (list, dict)) else json.dumps(value)
        comparison["per_tsrc"][str(tsrc)] = {
            "scale_real": float(scale.real), "scale_imag": float(scale.imag),
            "scale_residual": float(after),
            "qcu_true_residual": reference["residual"],
            "quda_scaled_true_residual": scaled_reference["residual"],
            "c2_pion_relative_difference": c2_diff,
            "qdp_layout_difference": qdp_diff,
        }
        if c2_qcu_store is not None:
            (run_dir / "c2").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run_dir / "c2" / f"tsrc_{tsrc}.npz",
                                qcu_c2_pion=c2_qcu_store, quda_c2_pion=c2_quda_store)

    scale_mean = float(np.mean([abs(s) for s in scales]))
    scale_spread = float(np.std([abs(s) for s in scales]))
    comparison["scale_expected"] = float(expected_scale)
    comparison["scale_mean"] = scale_mean
    comparison["scale_spread"] = scale_spread
    comparison["scale_relative_error"] = abs(scale_mean - expected_scale) / expected_scale
    comparison["status"] = "PASS"
    if comparison["scale_relative_error"] > args.scale_tol:
        comparison["status"] = "FAIL"
    if any(item["scale_residual"] > args.compare_tol for item in comparison["per_tsrc"].values()):
        comparison["status"] = "FAIL"
    # 数值差异对比为必要项：QDP 排布下的 relL2 / maxabs 必须显式过闸，不允许跳过。
    qdp_failures = {
        tsrc: item["qdp_layout_difference"]
        for tsrc, item in comparison["per_tsrc"].items()
        if item.get("qdp_layout_difference", {}).get("relative_l2", 1.0) > args.compare_tol
        or item.get("qdp_layout_difference", {}).get("max_abs", 1.0) > args.max_abs_tol}
    comparison["qdp_layout_gate"] = {"thresholds": {"relative_l2": args.compare_tol,
                                                   "max_abs": args.max_abs_tol},
                                    "failures": list(qdp_failures),
                                    "required": True}
    if qdp_failures:
        comparison["status"] = "FAIL"
    if any(item["qcu_true_residual"] > args.residual_tol for item in comparison["per_tsrc"].values()):
        comparison["status"] = "FAIL"
    covered = sorted(int(k) for k in comparison["per_tsrc"])
    comparison["tsrc_coverage"] = {"requested": requested, "covered": covered,
                                   "complete": covered == sorted(requested)}
    if covered != sorted(requested):
        comparison["status"] = "FAIL"
    comparison["multi_gpu"] = _multi_gpu_report(run_dir)
    return comparison


def _multi_gpu_report(run_dir: Path) -> Dict[str, Any]:
    """多卡环节汇总：QCU 一线程一卡（含跨卡一致性）+ QUDA MPI 分 rank 明细。"""
    report: Dict[str, Any] = {}
    qcu_path = run_dir / "qcu" / "metrics.json"
    if qcu_path.is_file():
        payload = json.loads(qcu_path.read_text())
        report["qcu"] = {
            "devices": payload.get("devices"),
            "threads": payload.get("threads"),
            "iterations_per_tsrc": payload.get("iterations_per_tsrc"),
            "solve_seconds_per_tsrc": payload.get("solve_seconds_per_tsrc"),
            "consistency_relative_l2": payload.get("multi_gpu_consistency"),
        }
    quda_metrics = _merge_quda_metrics(run_dir / "quda")
    ranks = quda_metrics.get("ranks", [])
    if ranks:
        report["quda"] = {
            "mpi_ranks": len(ranks),
            "per_rank_iterations": [r.get("solve_total_iterations") for r in ranks],
            "per_rank_seconds": [r.get("solve_total_seconds") for r in ranks],
            "aggregate_iterations": int(sum(r.get("solve_total_iterations", 0) for r in ranks)),
            "wall_seconds": max((r.get("solve_total_seconds", 0.0) for r in ranks), default=0.0),
        }
    report["active"] = bool(report.get("qcu", {}).get("threads", 1) > 1 or
                            report.get("quda", {}).get("mpi_ranks", 1) > 1)
    return report


# --------------------------------------------------------------------------
# 角色编排
# --------------------------------------------------------------------------
def run_check(args: argparse.Namespace) -> Dict[str, Any]:
    """CPU 可复现门禁：排布往返 + 收缩恒等式（不依赖 CUDA/QUDA）。"""
    lat = tuple(int(v) for v in args.lat)
    # 门禁固定用小格点，避免大格点上为非求解环节分配整场内存。
    report = {"roundtrip": layout_roundtrip_large((4, 4, 4, 4))}
    rng = np.random.default_rng(7)
    prop = (rng.normal(size=(COMPONENTS, 4, 3, *lat)) +
            1j * rng.normal(size=(COMPONENTS, 4, 3, *lat))).astype(np.complex64)
    g5, _ = pyqcu_gammas()
    small = lat[0] <= 2
    zero = meson_c2(prop, g5, g5)
    report["meson_c2"] = {
        "c2_pion_finite": bool(np.all(np.isfinite(zero["c2_pion"]))),
        "c2_pion_norm": float(np.linalg.norm(zero["c2_pion"])),
        "linearity_error": None,
        "loop_match_error": None,
    }
    if small:
        loop = meson_c2_reference(prop, g5, g5)
        report["meson_c2"]["loop_match_error"] = float(np.max(np.abs(loop - zero["c2_spin"])))
    doubled = meson_c2(2.0 * prop, g5, g5)["c2_pion"]
    report["meson_c2"]["linearity_error"] = float(
        np.linalg.norm(doubled - 4.0 * zero["c2_pion"]) /
        (np.linalg.norm(zero["c2_pion"]) + 1e-30))
    report["status"] = "PASS" if (report["roundtrip"]["gauge_error"] == 0.0 and
                                  report["roundtrip"]["fermion_error"] == 0.0 and
                                  report["meson_c2"]["c2_pion_finite"] and
                                  report["meson_c2"]["linearity_error"] < 1e-4 and
                                  (report["meson_c2"]["loop_match_error"] is None or
                                   report["meson_c2"]["loop_match_error"] < 1e-4)) else "FAIL"
    return report


def _wait_rank_artifacts(run_dir: Path, ranks: int, timeout: float) -> int:
    """等待各 rank 的 metrics 落盘；mpirun 可能先于存活 rank 返回。"""
    deadline = time.time() + max(1.0, float(timeout))
    found = 0
    while time.time() < deadline:
        found = len(list((run_dir / "quda").glob("metrics_rank*.json")))
        if found >= ranks:
            return found
        time.sleep(0.5)
    return len(list((run_dir / "quda").glob("metrics_rank*.json")))

def spawn(args: argparse.Namespace, run_dir: Path, role: str) -> None:
    """以子进程执行指定后端（libqcu 与 libquda 必须进程隔离）。"""
    command = [sys.executable, str(Path(__file__).resolve()), "--role", role,
               "--run-dir", str(run_dir), "--lat", *[str(v) for v in args.lat],
               "--mass", str(args.mass), "--tol", str(args.tol),
               "--max-iter", str(args.max_iter), "--precision", args.precision,
               "--tsrc", args.tsrc, "--components", str(args.components),
               "--mom", *[str(v) for v in args.mom],
               "--mg-levels", str(args.mg_levels), "--mg-nv-iters", str(args.mg_nv_iters),
               "--qcu-devices", *[str(v) for v in args.qcu_devices],
               "--quda-device", str(args.quda_device), "--t-boundary", str(args.t_boundary),
               "--mpi-grid", *[str(v) for v in args.mpi_grid],
               "--mg-dof", *[str(v) for v in args.mg_dof],
               "--no-doc"]
    for block in args.mg_block:
        command += ["--mg-block", *[str(v) for v in block]]
    if args.mg_cache_dir:
        command += ["--mg-cache-dir", args.mg_cache_dir, "--reuse-coarse"]
    if args.quda_no_mma:
        command.append("--quda-no-mma")
    if args.skip_c2:
        command.append("--skip-c2")
    if args.skip_residual:
        command.append("--skip-residual")
    command += ["--residual-rows", str(args.residual_rows)]
    if args.quda_no_tuning:
        command.append("--quda-no-tuning")
    else:
        command.append("--quda-tuning")
    if args.quda_cleanup:
        command.append("--quda-cleanup")
    command += ["--quda-warmup", str(args.quda_warmup)]
    if args.quda_resource:
        command += ["--quda-resource", str(args.quda_resource)]
    if args.qcu_verify:
        command.append("--qcu-verify")
    else:
        command.append("--no-qcu-verify")
    if args.verbose:
        command.append("--verbose")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if role == "quda" and args.quda_mpi > 1:
        command = ["mpirun", "-np", str(args.quda_mpi)] + command
        env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    print(f"[spawn] {' '.join(command)}")
    if role == "quda" and args.quda_mpi > 1:
        # 本机 QUDA 在 MPI 下即使各 rank 已写盘退出，mpirun 仍可能不返回；
        # 只要全部 rank 的 metrics 已落盘就视为完成（科学产物完整）。
        try:
            result = subprocess.run(command, cwd=REPO, env=env, timeout=args.quda_mpi_timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-x", "mpirun"], check=False)
            found = _wait_rank_artifacts(run_dir, int(args.quda_mpi), 30.0)
            if found >= int(args.quda_mpi):
                print(f"[spawn] mpirun 未退出（超时 {args.quda_mpi_timeout}s），但 {found} 个 rank "
                      "产物已落盘 → 继续对照", flush=True)
                return
            raise RunFailure(f"quda MPI 超时且仅 {found}/{args.quda_mpi} 个 rank 落盘")
        if result.returncode != 0:
            # mpirun 可能因先退出 rank 的 improper termination 先返回；等存活 rank 落盘
            found = _wait_rank_artifacts(run_dir, int(args.quda_mpi), args.quda_mpi_timeout)
            if found >= int(args.quda_mpi):
                print(f"[spawn] mpirun 退出码 {result.returncode}，但 {found} 个 rank 产物完整 → 继续",
                      flush=True)
                return
            raise RunFailure(f"quda 子进程退出码 {result.returncode}")
        return
    result = subprocess.run(command, cwd=REPO, env=env)
    if result.returncode != 0:
        raise RunFailure(f"{role} 子进程退出码 {result.returncode}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", default="compare",
                        choices=["compare", "gen", "qcu", "quda", "check"])
    parser.add_argument("--run-id", default=time.strftime("pionC2_%Y%m%d_%H%M%S"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--lat", nargs=4, type=int, default=(8, 8, 8, 16), metavar=("X", "Y", "Z", "T"))
    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--precision", choices=["single", "double"], default="single")
    parser.add_argument("--gauge", default=None, help="输入 gauge（HDF5 [3,3,4,X,Y,Z,T] 或 .npy）")
    parser.add_argument("--gauge-dataset", default="data")
    parser.add_argument("--gauge-seed", type=int, default=42)
    parser.add_argument("--gauge-sigma", type=float, default=0.1)
    parser.add_argument("--stout-steps", type=int, default=0)
    parser.add_argument("--stout-rho", type=float, default=0.125)
    parser.add_argument("--tsrc", default="0", help="时间源列表（逗号分隔）或 all")
    parser.add_argument("--components", type=int, choices=[1, 12], default=12,
                        help="每个 tsrc 的点源分量数（12 = 完整传播子）")
    parser.add_argument("--mom", nargs=3, type=int, default=(0, 0, 0), metavar=("PX", "PY", "PZ"))
    parser.add_argument("--t-boundary", type=int, choices=[1, -1], default=1)
    parser.add_argument("--mg-levels", type=int, default=2)
    parser.add_argument("--mg-block", nargs=4, type=int, action="append",
                        help="每层粗化块（可重复，长度必须为 mg-levels-1）")
    parser.add_argument("--mg-dof", nargs="+", type=int, default=(12, 48, 48, 48, 48))
    parser.add_argument("--mg-nv-iters", type=int, default=2)
    parser.add_argument("--mg-cache-dir", default=None,
                        help="粗层算子缓存目录；仅在 --reuse-coarse 且 gauge 指纹一致时复用")
    parser.add_argument("--reuse-coarse", action="store_true")
    parser.add_argument("--qcu-devices", nargs="*", type=int, default=[],
                        help="QCU 一线程一卡设备列表（默认全部可见设备）")
    parser.add_argument("--qcu-threads", type=int, default=None)
    parser.add_argument("--qcu-verify", action="store_true", default=True,
                        help="多卡时在每个设备上复算同一 tsrc 并报告一致性")
    parser.add_argument("--no-qcu-verify", dest="qcu_verify", action="store_false")
    parser.add_argument("--quda-device", type=int, default=0)
    parser.add_argument("--quda-mpi", type=int, default=1, help="QUDA 侧 MPI rank 数（集合调用路径）")
    parser.add_argument("--quda-nvec", type=int, default=None,
                        help="QUDA 每层 null vector 数（粗层 color = n_vec*4/spin_block；"
                             "与 QCU --mg-dof 对齐时使用）")
    parser.add_argument("--quda-workers", type=int, default=1,
                        help="QUDA 独立进程数（每进程一张卡 + tsrc 分片，规避 MPI 集合调用）")
    parser.add_argument("--quda-worker-index", type=int, default=0, help="内部使用：worker 序号")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="多卡环节：QCU 多线程多卡 + 跨卡一致性，QUDA 走 MPI 进程网格")
    parser.add_argument("--mpi-grid", nargs=4, type=int, default=None,
                        help="QUDA 进程网格，默认按 --quda-mpi 平铺到 t 轴")
    parser.add_argument("--quda-resource", default="")
    parser.add_argument("--quda-no-tuning", dest="quda_no_tuning", action="store_true", default=True,
                        help="禁用 QUDA autotuning（默认，保证公平/可复现）")
    parser.add_argument("--quda-tuning", dest="quda_no_tuning", action="store_false",
                        help="启用 QUDA autotuning（默认关闭）")
    parser.add_argument("--quda-mpi-timeout", type=float, default=180.0,
                        help="QUDA MPI 子进程等待上限（超时但各 rank 产物完整则继续）")
    parser.add_argument("--quda-cleanup", action="store_true",
                        help="求解后执行 QUDA freeGauge/multigrid.destroy（MPI 下可能挂起，默认关闭）")
    parser.add_argument("--quda-warmup", type=int, default=1,
                        help="计时前额外求解次数（吸收 QUDA 首次 kernel autotuning）")
    parser.add_argument("--quda-no-mma", action="store_true")
    parser.add_argument("--scale-tol", type=float, default=1e-3)
    parser.add_argument("--compare-tol", type=float, default=1e-2)
    parser.add_argument("--max-abs-tol", type=float, default=1e-3,
                        help="QDP 排布下 |Δx| 绝对差上限（数值差异对比为必要项）")
    parser.add_argument("--residual-tol", type=float, default=1e-4,
                        help="两后端解在 PyQCU 参考算子下的真残差上限")
    parser.add_argument("--skip-residual", action="store_true",
                        help="跳过纯 Python 真残差复核（大格点上省去非求解开销）")
    parser.add_argument("--residual-rows", type=int, default=12,
                        help="12 分量模式下参与真残差复核的行数（抽样上限）")
    parser.add_argument("--skip-c2", action="store_true",
                        help="跳过 pion 收缩（只做求解迭代数/耗时/解对照）")
    parser.add_argument("--no-doc", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    args.lat = tuple(int(v) for v in args.lat)
    args.mom = tuple(int(v) for v in args.mom)
    if args.mpi_grid is None:
        grid = [1, 1, 1, 1]
        grid[3] = int(args.quda_mpi)
        args.mpi_grid = tuple(grid)
    if args.mg_block is None:
        args.mg_block = [[2, 2, 2, 2] for _ in range(max(1, args.mg_levels - 1))]
    if len(args.mg_block) != max(1, args.mg_levels - 1):
        raise SystemExit(f"--mg-block 需要 {args.mg_levels - 1} 组，收到 {len(args.mg_block)}")
    if args.mg_levels < 2:
        raise SystemExit("MultiGrid 对照要求 --mg-levels >= 2")
    if args.multi_gpu:
        # 多卡环节：QCU 需要 >=2 张同构卡；QUDA 需要 >=2 MPI rank。
        args.qcu_verify = True
        qcu_devices = args.qcu_devices or []
        if len(qcu_devices) < 2 and args.quda_mpi < 2:
            raise SystemExit(
                "--multi-gpu 需要 --qcu-devices 至少 2 张卡，或 --quda-mpi >= 2；"
                "本机若 libqcu/libquda SASS 覆盖不同架构，请分别指定可用设备。")
        if args.quda_mpi >= 2:
            grid = [1, 1, 1, 1]
            grid[3] = int(args.quda_mpi)
            args.mpi_grid = tuple(grid)
    run_dir = run_dir_from_args(args)

    if not args.no_doc:
        print("=== PyQCU(C++ MG) vs 原生 QUDA：pion C2 对照 ===")
        print(json.dumps({"role": args.role, "run_dir": str(run_dir), "lat": args.lat,
                          "mass": args.mass, "precision": args.precision, "tsrc": args.tsrc,
                          "components": args.components, "mom": args.mom,
                          "qcu_devices": args.qcu_devices, "quda_mpi": args.quda_mpi,
                          "mpi_grid": list(args.mpi_grid), "mg_levels": args.mg_levels,
                          "mg_block": args.mg_block, "mg_dof": list(args.mg_dof)},
                         ensure_ascii=False, default=_json_default))

    started = time.perf_counter()
    try:
        if args.role == "check":
            report = run_check(args)
            print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
            return 0 if report["status"] == "PASS" else 1
        if args.role == "gen":
            prepare_gauge(args, run_dir)
            return 0
        if args.role == "qcu":
            run_qcu(args, run_dir)
            return 0
        if args.role == "quda":
            run_quda(args, run_dir)
            return 0

        prepare_gauge(args, run_dir)
        if args.mg_cache_dir is None:
            args.mg_cache_dir = str(run_dir / "qcu" / "nullvec_cache")
        args.reuse_coarse = True
        spawn(args, run_dir, "qcu")
        if int(args.quda_workers) > 1:
            for worker_index in range(int(args.quda_workers)):
                command = [sys.executable, str(Path(__file__).resolve()), "--role", "quda",
                           "--run-dir", str(run_dir), "--lat", *[str(v) for v in args.lat],
                           "--mass", str(args.mass), "--tol", str(args.tol),
                           "--max-iter", str(args.max_iter), "--precision", args.precision,
                           "--tsrc", args.tsrc, "--components", str(args.components),
                           "--mom", *[str(v) for v in args.mom], "--t-boundary", str(args.t_boundary),
                           "--mpi-grid", "1", "1", "1", "1",
                           "--mg-levels", str(args.mg_levels),
                           "--mg-nv-iters", str(args.mg_nv_iters),
                           "--quda-device", str(args.quda_device),
                           "--quda-workers", str(args.quda_workers),
                           "--quda-worker-index", str(worker_index),
                           *(["--quda-nvec", str(args.quda_nvec)] if args.quda_nvec else []),
                           "--quda-warmup", str(args.quda_warmup),
                           "--mg-dof", *[str(v) for v in args.mg_dof], "--no-doc"]
                for block in args.mg_block:
                    command += ["--mg-block", *[str(v) for v in block]]
                env = dict(os.environ)
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["PYQCU_QUDA_DEVICE"] = str(int(args.quda_device) + worker_index)
                print(f"[spawn-worker {worker_index}] CUDA device {env['PYQCU_QUDA_DEVICE']}", flush=True)
                result = subprocess.run(command, cwd=REPO, env=env)
                if result.returncode != 0:
                    raise RunFailure(f"quda worker {worker_index} 退出码 {result.returncode}")
        else:
            spawn(args, run_dir, "quda")
        record = compare(args, run_dir)
        record.update({"run_id": args.run_id, "run_dir": str(run_dir),
                       "elapsed_seconds": time.perf_counter() - started,
                       "qcu_metrics": json.loads((run_dir / "qcu" / "metrics.json").read_text()),
                       "quda_metrics": _merge_quda_metrics(run_dir / "quda")})
        write_json(run_dir / "record.json", record)
        _print_summary(record)
        return 0 if record["status"] == "PASS" else 1
    except RunFailure as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


def _merge_quda_metrics(directory: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"ranks": []}
    for path in sorted(directory.glob("metrics_rank*.json")):
        payload = json.loads(path.read_text())
        merged["ranks"].append(payload)
    if merged["ranks"]:
        merged["solve_total_iterations"] = int(
            sum(rank.get("solve_total_iterations", 0) for rank in merged["ranks"]))
        merged["solve_total_seconds"] = float(
            max(rank.get("solve_total_seconds", 0.0) for rank in merged["ranks"]))
        merged["precision"] = merged["ranks"][0].get("precision")
    return merged


def _print_summary(record: Dict[str, Any]) -> None:
    qcu = record.get("qcu_metrics", {})
    quda = record.get("quda_metrics", {})
    print("\n=== 求解对照汇总 ===")
    print(f"状态            : {record['status']}")
    print(f"格点/精度       : {record['lattice']} / {qcu.get('precision')}")
    print(f"QCU  总迭代/耗时: {qcu.get('solve_total_iterations')} / "
          f"{qcu.get('solve_total_seconds'):.3f} s（粗层 setup "
          f"{qcu.get('coarse_setup_seconds'):.3f} s）")
    print(f"QUDA 总迭代/耗时: {quda.get('solve_total_iterations')} / "
          f"{quda.get('solve_total_seconds'):.3f} s")
    print(f"标度 c (期望 {record['scale_expected']:.6f}): {record['scale_mean']:.6f} "
          f"± {record['scale_spread']:.2e}，相对偏差 {record['scale_relative_error']:.2e}")
    ranks = quda.get("ranks") or [{}]
    mg = ranks[0].get("multigrid_config", {})
    if mg:
        print(f"QUDA MG 口径    : inv={mg.get('inv_type')} n_level={mg.get('n_level')} "
              f"n_vec={mg.get('n_vec')} cycle={mg.get('cycle_type')} "
              f"nu={mg.get('nu_pre')}/{mg.get('nu_post')} coarse={mg.get('coarse_solver')}")
    print(f"QUDA warmup/tune: {quda.get('warmup_seconds', 0.0):.3f} s "
          f"（tuning 复用={ranks[0].get('tuning_was_cached')}）；"
          "迭代数语义：QUDA=外层 GCR 迭代，QCU=其 MG 重启/外层迭代，不可直接等价。")
    for tsrc, item in record["per_tsrc"].items():
        c2_text = ("—" if item["c2_pion_relative_difference"] is None
                   else f"{item['c2_pion_relative_difference']:.3e}")
        qdp = item.get("qdp_layout_difference", {})
        print(f"  tsrc={tsrc}: 标度残差 {item['scale_residual']:.3e}；"
              f"QCU 真残差 {item['qcu_true_residual']:.3e}；"
              f"QUDA 标度后真残差 {item['quda_scaled_true_residual']:.3e}；"
              f"C2 相对差 {c2_text}")
    if qdp:
            print(f"           QDP 排布差异: relL2={qdp['relative_l2']:.3e} "
                  f"maxabs={qdp['max_abs']:.3e} shape={qdp['shape']}")
    multi = record.get("multi_gpu") or {}
    if multi:
        qcu_multi = multi.get("qcu", {})
        quda_multi = multi.get("quda", {})
        print(f"多卡环节        : active={multi.get('active')} | "
              f"QCU threads={qcu_multi.get('threads')} devices={qcu_multi.get('devices')} "
              f"跨卡一致性={qcu_multi.get('consistency_relative_l2')} | "
              f"QUDA ranks={quda_multi.get('mpi_ranks')} "
              f"逐 rank 迭代={quda_multi.get('per_rank_iterations')} "
              f"逐 rank 耗时={quda_multi.get('per_rank_seconds')}")


def test_pion_c2_layouts() -> None:
    """pytest 入口：仅 CPU 可复现门禁，GPU 流程由 main() 显式触发。"""
    report = run_check(argparse.Namespace(lat=(4, 4, 4, 4)))
    assert report["status"] == "PASS", report


if __name__ == "__main__":
    raise SystemExit(main())
