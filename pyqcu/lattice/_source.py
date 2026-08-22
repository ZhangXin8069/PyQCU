import math
from typing import Optional, List, Union
import torch
import pyqcu.cann as _torch

"""源场构造 — 参考实现 PyQUDA pyquda_utils/source.py（point/wall/volume/momentum）。

布局约定：费米子场 [4, 3, Lx, Ly, Lz, Lt] = [spin, color, x, y, z, t]，时空维最后 4 轴。
spin/color 为 None 时表示该自由度维全填（对 spin/color 不做 δ 选择）。
"""


def _give_latt(latt_size: List[int], dtype: torch.dtype, device) -> torch.Tensor:
    return _torch.zeros([4, 3, *latt_size], dtype=dtype, device=device)


def point_source(latt_size: List[int], t_srce: List[int], spin: Optional[int] = None,
                 color: Optional[int] = None, dtype: torch.dtype = torch.complex128,
                 device=torch.device('cpu')) -> torch.Tensor:
    """点源：时空 δ(x - x_src)，spin/color 任选（None → 全填）。

    Args:
        latt_size: [Lx, Ly, Lz, Lt]
        t_srce: 源坐标 [x, y, z, t]
    Returns:
        [4, 3, Lx, Ly, Lz, Lt] 源场
    """
    b = _give_latt(latt_size, dtype, device)
    if spin is None:
        b[:, :, t_srce[0], t_srce[1], t_srce[2], t_srce[3]] = 1.0
    elif color is None:
        b[spin, :, t_srce[0], t_srce[1], t_srce[2], t_srce[3]] = 1.0
    else:
        b[spin, color, t_srce[0], t_srce[1], t_srce[2], t_srce[3]] = 1.0
    return b


def wall_source(latt_size: List[int], t_srce: int, spin: Optional[int] = None,
                color: Optional[int] = None, dtype: torch.dtype = torch.complex128,
                device=torch.device('cpu')) -> torch.Tensor:
    """墙源：固定时间片全空间填充，spin/color 任选（None → 全填）。"""
    b = _give_latt(latt_size, dtype, device)
    if spin is None:
        b[:, :, :, :, :, t_srce] = 1.0
    elif color is None:
        b[spin, :, :, :, :, t_srce] = 1.0
    else:
        b[spin, color, :, :, :, t_srce] = 1.0
    return b


def volume_source(latt_size: List[int], spin: Optional[int] = None,
                  color: Optional[int] = None, dtype: torch.dtype = torch.complex128,
                  device=torch.device('cpu')) -> torch.Tensor:
    """体积源：全时空填充，spin/color 任选（None → 全填）。"""
    b = _give_latt(latt_size, dtype, device)
    if spin is None:
        b[:, :] = 1.0
    elif color is None:
        b[spin, :] = 1.0
    else:
        b[spin, color] = 1.0
    return b


def z2_source(latt_size: List[int], seed: Optional[int] = None,
              dtype: torch.dtype = torch.complex128, device=torch.device('cpu'),
              verbose: bool = False) -> torch.Tensor:
    """Z2 随机噪声源：每自由度独立取 ±1（实值），用于随机源平均。

    Args:
        seed: 随机种子（None → 不重置全局种子）
    """
    if seed is not None:
        torch.manual_seed(seed)
        if isinstance(device, torch.device) and device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    sign = torch.randint(0, 2, [4, 3, *latt_size], device=device) * 2.0 - 1.0
    b = sign.to(dtype)
    if verbose:
        print(f"PYQCU::LATTICE::SOURCE:\n Z2 source generated with seed={seed}, norm={_torch.norm(b)}")
    return b


def momentum_source(latt_size: List[int], mode: List[int], t_srce: Optional[int] = None,
                    spin: Optional[int] = None, color: Optional[int] = None,
                    dtype: torch.dtype = torch.complex128,
                    device=torch.device('cpu')) -> torch.Tensor:
    """平面波动量源：exp(i·2π·(n·x)/L)，mode=[nx, ny, nz, nt] 整数模式号。

    t_srce=None → 全时空（volume 型动量源）；int → 固定时间片（wall 型动量源）。
    """
    Lx, Ly, Lz, Lt = latt_size
    nx, ny, nz, nt = mode
    x = torch.arange(Lx, device=device).view(-1, 1, 1, 1).to(torch.float64)
    y = torch.arange(Ly, device=device).view(1, -1, 1, 1).to(torch.float64)
    z = torch.arange(Lz, device=device).view(1, 1, -1, 1).to(torch.float64)
    t = torch.arange(Lt, device=device).view(1, 1, 1, -1).to(torch.float64)
    two_pi_over_l = [2.0 * math.pi / float(L) for L in latt_size]
    phase = torch.exp(1j * (nx * x * two_pi_over_l[0] + ny * y * two_pi_over_l[1]
                            + nz * z * two_pi_over_l[2] + nt * t * two_pi_over_l[3])).to(dtype)
    if t_srce is not None:
        b = wall_source(latt_size, t_srce, spin, color, dtype, device)
    else:
        b = volume_source(latt_size, spin, color, dtype, device)
    if spin is None:
        b[:, :] *= phase.unsqueeze(0).unsqueeze(0)
    elif color is None:
        b[spin, :] *= phase.unsqueeze(0)
    else:
        b[spin, color] *= phase
    return b


def fermion_source(latt_size: List[int], kind: str, t_srce: Union[List[int], int, None] = None,
                   spin: Optional[int] = None, color: Optional[int] = None,
                   mode: Optional[List[int]] = None, seed: Optional[int] = None,
                   dtype: torch.dtype = torch.complex128,
                   device=torch.device('cpu')) -> torch.Tensor:
    """统一分派入口（PyQUDA source() 语义）：kind ∈ {point, wall, volume, momentum, z2}。

    point → t_srce=[x,y,z,t]；wall → t_srce=int；momentum → mode 必填；
    z2 → seed 可选；volume → 其余忽略。
    """
    kind = kind.lower()
    if kind == 'point':
        assert isinstance(t_srce, list), "point source requires t_srce=[x,y,z,t]"
        return point_source(latt_size, t_srce, spin, color, dtype, device)
    elif kind == 'wall':
        assert isinstance(t_srce, int), "wall source requires t_srce=int"
        return wall_source(latt_size, t_srce, spin, color, dtype, device)
    elif kind == 'volume':
        return volume_source(latt_size, spin, color, dtype, device)
    elif kind == 'momentum':
        assert mode is not None, "momentum source requires mode=[nx,ny,nz,nt]"
        assert t_srce is None or isinstance(t_srce, int), \
            "momentum source requires t_srce=None or int"
        return momentum_source(latt_size, mode, t_srce, spin, color, dtype, device)
    elif kind == 'z2':
        return z2_source(latt_size, seed, dtype, device)
    raise ValueError(f"PYQCU::LATTICE::SOURCE:\n unknown source kind '{kind}'")
