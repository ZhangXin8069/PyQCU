"""dev87 G5/G6/G7 组件级诊断：null 向量质量、P/R 一致性、Galerkin 一致性。

对 data/ 缓存的 33-tensor stencil（16×32×32×48, E12, nvi1）做标准验收，
结果落 out/component_diag.json 并供报告 §8 引用。仅 PyQCU 侧（quda 组件
不暴露同层接口，结构性差异已在矩阵注明）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAT_DEFAULT, MASS_DEFAULT, load_gauge_h5, load_stencil, pick_v100
from pyqcu import dslash, tools
from pyqcu.testing import verify_nullvecs

OUT = Path(__file__).resolve().parent / "out"
LAT = LAT_DEFAULT
MASS = MASS_DEFAULT


def main():
    dev = pick_v100()
    print(f"[diag] device={torch.cuda.get_device_name(dev)}")
    g = load_gauge_h5(LAT, MASS, device="cuda")
    lonv, hnn, hdg, sit = load_stencil(LAT, E=12, nvi=1, device="cuda")
    U = tools.poooxyzt2oooxyzt(g)
    cl = dslash.make_clover(U, kappa=1.0 / (2 * MASS + 8))
    op = dslash.operator(U=U, clover_term=cl, kappa=torch.Tensor([1.0 / (2 * MASS + 8)]),
                         verbose=False, support_parity=True)
    S = op.matvec_parity

    lat_fine_odd = [LAT[0], LAT[1], LAT[2], LAT[3] // 2]
    mg = [2, 2, 2, 2]
    lat_coarse_odd = [d // m for d, m in zip(lat_fine_odd, mg)]

    t0 = time.perf_counter()
    res = verify_nullvecs(S, lonv, lat_fine_odd, lat_coarse_odd,
                          n_sample=4, stencil=(hnn, hdg, sit), verbose=True)
    dt = time.perf_counter() - t0

    payload = {"lat": LAT, "E": 12, "nvi": 1, "diag_wall_s": dt,
               **{k: (float(v) if torch.is_tensor(v) or isinstance(v, (int, float, np.floating)) else v)
                  for k, v in res.items()
                  if isinstance(v, (int, float, np.floating, torch.Tensor))}}
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "component_diag.json").write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
