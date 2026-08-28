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


def parse_complex_dtype(value):
    """解析 dev87 CLI/调用方使用的复数精度名称。

    QCU 的数据类型协议只支持 ``complex64`` 和 ``complex128``；这里把
    用户输入统一成 ``(torch.dtype, params code)``，避免 Python 层某处
    只改了张量 dtype、另一处仍把 ``_DATA_TYPE_`` 留在 c64 的半混合状态。
    """
    if isinstance(value, torch.dtype):
        if value == torch.complex64:
            return value, define._LAT_C64_
        if value == torch.complex128:
            return value, define._LAT_C128_
    name = str(value).strip().lower()
    aliases = {
        "c64": (torch.complex64, define._LAT_C64_),
        "complex64": (torch.complex64, define._LAT_C64_),
        "fp32": (torch.complex64, define._LAT_C64_),
        "c128": (torch.complex128, define._LAT_C128_),
        "complex128": (torch.complex128, define._LAT_C128_),
        "fp64": (torch.complex128, define._LAT_C128_),
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError("dtype must be one of c64/complex64 or c128/complex128") from exc


def _real_dtype_for_code(data_type):
    if int(data_type) == define._LAT_C64_:
        return torch.float32
    if int(data_type) == define._LAT_C128_:
        return torch.float64
    raise ValueError(f"unsupported QCU complex data type code: {data_type}")


def process_grid(grid=None):
    """返回与 C++ rank 映射一致的 4D process grid ``[x,y,z,t]``。"""
    values = tools.give_grid_size() if grid is None else list(grid)
    values = [int(v) for v in values]
    if len(values) != 4 or any(v <= 0 for v in values):
        raise ValueError(f"process grid must contain four positive integers: {values}")
    return values


def grid_index_for_rank(grid, rank):
    """按 ``np.arange(size).reshape(grid)`` 取得 C++ 使用的 rank 坐标。"""
    grid = process_grid(grid)
    size = int(np.prod(grid))
    rank = int(rank)
    if rank < 0 or rank >= size:
        raise ValueError(f"rank {rank} is outside process grid {grid}")
    return [int(v) for v in np.unravel_index(rank, tuple(grid))]


def local_geometry(lat, grid=None, rank=None, require_even=False):
    """返回 ``(local_lat, starts, grid_index)``，轴顺序均为 ``[X,Y,Z,T]``。

    奇偶压缩需要每个 local physical 轴保持 checkerboard 的局部周期约定；
    因此调用方在准备 fine 场时应要求四个 local 轴为偶数。粗层本身不做
    checkerboard 压缩，允许最后一层出现尺寸 1，故默认不强制偶数。
    """
    global_lat = [int(v) for v in lat]
    if len(global_lat) != 4 or any(v <= 0 for v in global_lat):
        raise ValueError(f"lattice must contain four positive integers: {global_lat}")
    grid = process_grid(grid)
    if any(global_lat[d] % grid[d] != 0 for d in range(4)):
        raise ValueError(f"lattice {global_lat} is not divisible by process grid {grid}")
    rank = define.rank if rank is None else int(rank)
    index = grid_index_for_rank(grid, rank)
    local_lat = [global_lat[d] // grid[d] for d in range(4)]
    if require_even and any(v % 2 for v in local_lat):
        raise ValueError(
            f"local fine lattice must be even in every axis for checkerboard MPI: "
            f"global={global_lat}, grid={grid}, local={local_lat}")
    starts = [index[d] * local_lat[d] for d in range(4)]
    return local_lat, starts, index


def _slice_last4(array, starts, sizes):
    if array.ndim < 4:
        raise ValueError(f"expected at least four lattice axes, got shape={tuple(array.shape)}")
    slices = [slice(None)] * (array.ndim - 4)
    slices.extend(slice(int(starts[d]), int(starts[d] + sizes[d])) for d in range(4))
    return array[tuple(slices)].contiguous()


def global_parity_to_local(parity, global_lat, grid=None, rank=None,
                           device=None, dtype=None):
    """将全局奇偶布局转换为本 rank 的 local 奇偶布局。

    不能直接对 ``[parity,X,Y,Z,T/2]`` 的最后四轴做普通 block slice：
    每个 parity 页的压缩 t 索引依赖 ``x+y+z``，而 rank block 的 checkerboard
    原点可能不同。先恢复全格点、切 physical block、再重新压缩，保证边界
    与 C++ local coordinate convention 一致。
    """
    global_lat = [int(v) for v in global_lat]
    if tuple(int(v) for v in parity.shape[-4:]) != tuple(
            global_lat[:3] + [global_lat[3] // 2]):
        raise ValueError(
            f"parity shape {tuple(parity.shape)} does not match global lattice {global_lat}")
    local_lat, starts, _ = local_geometry(
        global_lat, grid=grid, rank=rank, require_even=True)
    full = tools.poooxyzt2oooxyzt(parity)
    local_full = _slice_last4(full, starts, local_lat)
    local = tools.oooxyzt2poooxyzt(local_full)
    if dtype is not None:
        local = local.to(dtype=dtype)
    if device is not None:
        local = local.to(device=device)
    return local.contiguous()


def pick_v100():
    """按 v20260825 指令17 选 V100-32GB；QCU_DEVICE_ID 可显式覆盖。"""
    env = os.environ.get("QCU_DEVICE_ID")
    if env is not None:
        i = int(env); torch.cuda.set_device(i)
        return i
    for i in range(torch.cuda.device_count()):
        if "V100" in torch.cuda.get_device_name(i):
            torch.cuda.set_device(i)
            return i
    raise RuntimeError("V100 not found")


def gauge_tag(lat, mass, seed=SEED_DEFAULT):
    return f"gauge_{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_m{mass}_seed{seed}_c64.h5"


def nv_tag(lat, E, nvi, suf="", level=1):
    return (f"L{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_lv{level}_E{E}"
            f"_nvi{nvi}{suf}_t1e-2.h5")


def load_gauge_h5(lat, mass=MASS_DEFAULT, seed=SEED_DEFAULT, device="cuda",
                  dtype=None):
    """读 data/*.h5 -> g_dev[2,3,3,4,x,y,z,T/2]。"""
    import h5py
    path = DATA_DIR / gauge_tag(lat, mass, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(str(path), "r") as f:
        g_np = f["g"][...]
    out = torch.from_numpy(g_np)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out.to(device)


def load_local_gauge_h5(lat, mass=MASS_DEFAULT, seed=SEED_DEFAULT,
                        grid=None, rank=None, device="cuda", dtype=None):
    """只向设备上传本 rank 的 gauge block。

    HDF5 gauge 是奇偶压缩格式，局部读取前需要经过全格点重排；全局临时
    副本只留在 CPU，设备端和后续 C++ operator 均只保留 local block。
    """
    torch_dtype = dtype
    if torch_dtype is None:
        torch_dtype = torch.complex64
    g_global = load_gauge_h5(lat, mass, seed, device="cpu")
    return global_parity_to_local(
        g_global, lat, grid=grid, rank=rank, device=device, dtype=torch_dtype)


def _stencil_path(lat, E, nvi=1, suf="", level=1):
    path = DATA_DIR / nv_tag(lat, E, nvi, suf, level=level)
    if not path.exists() and not suf:
        legacy = DATA_DIR / (
            f"L{lat[0]}x{lat[1]}x{lat[2]}x{lat[3]}_lv{level}_E{E}"
            f"_nvi{nvi}_t0.01.h5"
        )
        if legacy.exists():
            path = legacy
    return path


def load_stencil(lat, E=12, nvi=1, suf="", device="cuda", level=1,
                 dtype=None):
    """读取指定层的 33-tensor 粗算子缓存。

    ``level`` 是从细层到粗层的 transition 编号：1 表示 level-0→1，
    2 表示 level-1→2。兼容历史缓存中的 ``t0.01`` 文件名。
    """
    path = _stencil_path(lat, E, nvi, suf, level)
    if not path.exists():
        raise FileNotFoundError(path)
    tensors = tuple(tools.load_tensor_h5(str(path), dataset=k, device=device)
                    for k in ("lonv", "hnn", "hdg", "sit"))
    if dtype is not None:
        tensors = tuple(t.to(dtype=dtype).contiguous() for t in tensors)
    return tensors


def load_stencil_local(lat, E=12, nvi=1, suf="", grid=None, rank=None,
                       device="cuda", dtype=None, level=1):
    """从粗算子 HDF5 缓存直接读取本 rank 的 local transition。

    缓存轴约定：
      ``lonv=[E,e,Xc,x,Yc,y,Zc,z,Tc,t]``；
      ``hnn=[2,4,E,E,Xc,Yc,Zc,Tc]``；
      ``hdg=[2,2,6,E,E,Xc,Yc,Zc,Tc]``；
      ``sit=[E,E,Xc,Yc,Zc,Tc]``。
    """
    import h5py

    path = _stencil_path(lat, E, nvi, suf, level)
    if not path.exists():
        raise FileNotFoundError(path)
    grid = process_grid(grid)
    rank = define.rank if rank is None else int(rank)
    axes = {"lonv": (2, 4, 6, 8),
            "hnn": (4, 5, 6, 7),
            "hdg": (5, 6, 7, 8),
            "sit": (2, 3, 4, 5)}
    with h5py.File(str(path), "r") as f:
        lonv_shape = tuple(int(v) for v in f["lonv"].shape)
        global_coarse = [lonv_shape[a] for a in axes["lonv"]]
        local_coarse, starts, _ = local_geometry(
            global_coarse, grid=grid, rank=rank, require_even=False)
        result = []
        for name in ("lonv", "hnn", "hdg", "sit"):
            ds = f[name]
            if any(int(ds.shape[a]) != global_coarse[d]
                   for d, a in enumerate(axes[name])):
                raise ValueError(
                    f"inconsistent {name} lattice axes in {path}: shape={ds.shape}")
            selection = [slice(None)] * ds.ndim
            for d, axis in enumerate(axes[name]):
                selection[axis] = slice(starts[d], starts[d] + local_coarse[d])
            arr = np.asarray(ds[tuple(selection)])
            tensor = torch.from_numpy(arr)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            result.append(tensor.to(device=device).contiguous())
    return tuple(result)


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


def make_clover_tensors(g_dev, lat, mass=MASS_DEFAULT, grid=None, rank=None,
                        dtype=None, data_type=None):
    """经 C++ applyCloversQcu 构建 (ce,cei,coo,coi)。调用方负责生命周期。"""
    if data_type is None:
        if dtype is None:
            dtype, data_type = torch.complex64, define._LAT_C64_
        else:
            dtype, data_type = parse_complex_dtype(dtype)
    else:
        data_type = int(data_type)
        expected_dtype = torch.complex64 if data_type == define._LAT_C64_ else torch.complex128
        if dtype is None:
            dtype = expected_dtype
        else:
            dtype, parsed_code = parse_complex_dtype(dtype)
            if parsed_code != data_type:
                raise ValueError("dtype and data_type describe different precisions")
    if data_type not in (define._LAT_C64_, define._LAT_C128_):
        raise ValueError(f"unsupported Clover dtype code: {data_type}")
    grid = process_grid(grid)
    rank = define.rank if rank is None else int(rank)
    if int(np.prod(grid)) != define.size:
        raise ValueError(f"process grid {grid} does not match MPI size {define.size}")
    g_dev = g_dev.to(dtype=dtype).contiguous()
    p = mod_params.clone()
    a = mod_argv.clone()
    s = mod_set_ptrs.clone()
    dt = int(data_type)
    Lx, Ly, Lz, Lt = lat
    p[define._LAT_X_] = Lx; p[define._LAT_Y_] = Ly; p[define._LAT_Z_] = Lz; p[define._LAT_T_] = Lt
    p[define._LAT_XYZT_] = Lx * Ly * Lz * Lt
    p[define._GRID_X_], p[define._GRID_Y_], p[define._GRID_Z_], p[define._GRID_T_] = grid
    p[define._NODE_RANK_] = rank; p[define._NODE_SIZE_] = int(np.prod(grid))
    p[define._DATA_TYPE_] = dt
    av = a.to(dtype=_real_dtype_for_code(dt))
    av[define._MASS_] = mass; av[define._SIGMA_] = SIGMA_DEFAULT
    ls = define.lat_shape(p)
    ce = torch.empty([4, 3, 4, 3] + ls, dtype=dtype, device=g_dev.device)
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
