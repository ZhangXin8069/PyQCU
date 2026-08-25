"""dev87 G4.1 约定锚定：用 Python 参考全算子 M 对双方解做残差交叉验证。

r_x = b - M x 的相对范数判定各侧解在"同一算子定义"下是否自洽：
  - 若 quda 解与 PyQCU 解都在 Python-M 下残差小 -> 算子同义，差异来自求解器容差
  - 若仅一侧小 -> 该侧与 Python-M 约定一致，另一侧存在尺度/布局错位
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAT_DEFAULT, MASS_DEFAULT, load_gauge_h5
from pyqcu import dslash, tools

OUT = Path(__file__).resolve().parent / "out"


def full_res(x_full_dev, U_full, cl, src_full, kappa):
    """dev84 配方：||give_wilson+give_clover - b|| / ||b||。"""
    r = dslash.give_wilson(x_full_dev, U_full, kappa, True) \
        + dslash.give_clover(x_full_dev, cl) - src_full
    return float(tools.norm(r) / tools.norm(src_full))


KAPPA = None


def main():
    lat = LAT_DEFAULT
    mass = MASS_DEFAULT
    kappa = 1.0 / (2 * mass + 8)
    g_dev = load_gauge_h5(lat, mass, device="cuda")
    U_full = tools.poooxyzt2oooxyzt(g_dev)
    cl = dslash.make_clover(U_full, kappa=kappa)

    npz_b = np.load(OUT / "qcu_clover_solve.npz")
    b_eo = torch.from_numpy(npz_b["b_eo"]).to("cuda")
    src = tools.poooxyzt2oooxyzt(b_eo)
    x_qcu = tools.poooxyzt2oooxyzt(torch.from_numpy(npz_b["x_eo"]).to("cuda"))

    print(f"rel_res(dev84配方, x_qcu_cpp)={full_res(x_qcu, U_full, cl, src, kappa):.3e}")

    f = OUT / "quda_clover_solve.npz"
    if f.exists():
        x_quda_np = np.load(f)["x_scxyzt"]
        x_quda = torch.from_numpy(x_quda_np).to("cuda").to(torch.complex64)
        print(f"rel_res(dev84配方, x_quda)   ={full_res(x_quda, U_full, cl, src, kappa):.3e}")
        rd = float(tools.norm((x_quda - x_qcu).ravel()) / tools.norm(x_qcu))
        print(f"rel_diff(x_quda, x_qcu_cpp)={rd:.3e}")


if __name__ == "__main__":
    main()
