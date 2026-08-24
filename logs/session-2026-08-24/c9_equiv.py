import traceback
import time
import torch
import pyqcu.cann as _torch
import pyqcu.dslash as dslash
import pyqcu.lattice as lattice
from pyqcu.tools._multigrid import (local_orthogonalize, BatchedLocalSchur,
                                    build_stencil_local, build_stencil)

CUDA = torch.device('cuda')
DT = torch.complex64

try:
    LAT_F = [16, 16, 16, 8]
    COARSE = [8, 8, 8, 4]
    E = 12
    U = _torch.zeros(size=[3, 3, 4] + [16, 16, 16, 16], dtype=DT, device=CUDA)
    lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
    clover = _torch.zeros(size=[4, 3, 4, 3] + [16, 16, 16, 16], dtype=DT, device=CUDA)
    op = dslash.operator(U=U, kappa=torch.Tensor([0.125]),
                         clover_term=clover, support_parity=True, verbose=False)

    nv = _torch.randn(size=[E, E] + LAT_F, dtype=DT, device=CUDA)
    lonv = local_orthogonalize(null_vecs=nv, coarse_lat_size=COARSE, verbose=False)

    t0 = time.time()
    lsch = BatchedLocalSchur(op, *LAT_F, W=10)
    hnn_l, hdg_l, sit_l = build_stencil_local(lsch, lonv, E, LAT_F, COARSE,
                                              DT, CUDA, verbose=False)
    print(f"[C9] local build {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    hnn_r, hdg_r, sit_r = build_stencil(op.matvec_parity, lonv, E, E,
                                        LAT_F, COARSE, DT, CUDA, verbose=False)
    print(f"[C9] full-grid ref build {time.time()-t0:.1f}s", flush=True)

    for name, a, b in (("sit", sit_l, sit_r), ("hop_nn", hnn_l, hnn_r),
                       ("hop_diag", hdg_l, hdg_r)):
        da = float((a - b).abs().max().item())
        nb = float(b.abs().max().item())
        rel = da / max(nb, 1e-30)
        print(f"[C9] {name}: max|loc-ref|={da:.3e} ref_max={nb:.3e} rel={rel:.2e}", flush=True)
        assert rel < 1e-4, f"{name} mismatch rel={rel:.2e}"
    print("[C9][SUMMARY] EQUIVALENCE PASS — bug35 修复后 local 版与全格参考数学一致", flush=True)
except Exception:
    traceback.print_exc()
    print("[C9][SUMMARY] FAIL")
