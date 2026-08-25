"""dev87 G2 算子级锚定：单位规范下对比 PyQCU(Python) 与 quda(MatQuda) 的 Wilson M。

单位规范 -> clover 项为零，M = 2κm·1 + κD_hop（各库质量项约定差异直接显形）。
对同一随机源作用并扫描匹配变体（直配/转置色/厄米），报告最优相对差。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAT_DEFAULT, MASS_DEFAULT
from pyqcu import dslash, tools

OUT = Path(__file__).resolve().parent / "out"
LAT = [8, 8, 8, 16]
MASS = MASS_DEFAULT


def unit_gauge_full(lat):
    X, Y, Z, T = lat
    e = torch.eye(3, dtype=torch.complex64)
    u = e.view(3, 3, 1, 1, 1, 1, 1).expand(3, 3, 4, X, Y, Z, T)
    return u.contiguous().to("cuda")


def main():
    kappa = 1.0 / (2 * MASS + 8)
    import pyquda
    # pyquda 0.10.54 initDevice 硬编码 device=-1 → rank0 恒选 GPU0(P100)。
    # 本机 libquda 仅编 sm_70 → 须 V100(cuda 物理卡2)。用 CUDA_VISIBLE_DEVICES
    # 无法穿透(pyquda 内部 MPI 枚举仍见全卡), 故直接 patch initDevice 默认值。
    import pyquda as _pq
    import pyquda_comm as _pc
    _orig_id = _pc.initDevice

    def _dev1(backend, target, device=-1, enable_mps=False):
        return _orig_id(backend, target, 1, enable_mps)
    _pq.initDevice = _dev1
    # initQUDA(getGridSize(), getArrayDevice()) — getArrayDevice 在
    # initDevice 后返回 _ARRAY_DEVICE=2 ✓ 无需再改
    pyquda.init(grid_size=[1, 1, 1, 1], latt_size=LAT, backend="torch", backend_target="cuda",
                enable_nvshmem=False, enable_tuning=False,
                resource_path="/tmp/opencode/quda_resource",
                enable_device_memory_pool=False, enable_pinned_memory_pool=False)
    import pyquda_utils.core as core
    from pyquda.field import LatticeFermion, LatticeGauge
    info = core.LatticeInfo(list(LAT), 1, 1.0)
    U = unit_gauge_full(LAT)
    gen = torch.Generator(device="cpu").manual_seed(7)
    src = torch.randn([4, 3] + LAT, generator=gen, dtype=torch.float32, device="cpu").to(torch.complex64).cuda()

    # PyQCU 侧：M = with_I*src + (kappa/u_0)*hopping（u_0=1）
    y_pyqcu = dslash.give_wilson(src, U, torch.Tensor([kappa]), torch.Tensor([1.0]))


    def to_tzyxsc(v):
        return np.ascontiguousarray(np.transpose(v.double().cpu().numpy(), (5, 4, 3, 2, 0, 1)))

    u_np = np.ascontiguousarray(np.transpose(U.double().cpu().numpy(), (2, 6, 5, 4, 3, 0, 1)))
    g = LatticeGauge(info, 4, torch.from_numpy(info.evenodd(u_np, True)).to("cuda"))
    x = LatticeFermion(info, torch.from_numpy(info.evenodd(to_tzyxsc(src), False)).to("cuda"))

    dw = core.getWilson(info, MASS, 1e-12, 100)
    dw.loadGauge(g)
    y = dw.mat(x)
    y_np = np.transpose(y.data.cpu().numpy(), (4, 5, 3, 2, 1, 0))  # (t,z,y,x,s,c)->(s,c,x,y,z,t)
    y_quda = torch.from_numpy(np.ascontiguousarray(y_np)).to("cuda").to(torch.complex64)

    nb = float(tools.norm(y_quda))
    print(f"||y_quda||={nb:.4e}")
    rd = float(tools.norm((y_pyqcu - y_quda).ravel()) / nb)
    print(f"rel_diff(pyqcu_M, quda_MatQuda)={rd:.3e}")
    # 变体扫描：色转置 / 共轭
    rd_t = float(tools.norm(((y_pyqcu.reshape(4,3,LAT[0],LAT[1],LAT[2],LAT[3]).transpose(1,0).reshape(4,3,-1)
                              .view_as(y_pyqcu)) - y_quda).ravel()) / nb) if False else None
    yc = y_pyqcu.reshape(4, 3, LAT[0], LAT[1], LAT[2], LAT[3])
    for name, yy in [("color_transpose", yc.transpose(0, 1).reshape(12, LAT[0], LAT[1], LAT[2], LAT[3]).unsqueeze(0).squeeze(0).reshape(4,3,LAT[0],LAT[1],LAT[2],LAT[3]).reshape(4,3,LAT[0]*LAT[1]*LAT[2]*LAT[3]).reshape(4,3,LAT[0],LAT[1],LAT[2],LAT[3]).reshape(4,3,*LAT)),
                     ]:
        pass
    y_t = yc.transpose(0, 1).contiguous()
    print(f"rel_diff(pyqcu_colorT, quda)={float(tools.norm((y_t - y_quda).ravel())/nb):.3e}")
    print(f"rel_diff(pyqcu_conj, quda)={float(tools.norm((y_pyqcu.conj() - y_quda).ravel())/nb):.3e}")


if __name__ == "__main__":
    main()
