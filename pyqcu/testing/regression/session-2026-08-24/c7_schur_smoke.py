import traceback
import torch
import pyqcu.cann as _torch
import pyqcu.tools as tools
import pyqcu.dslash as dslash
import pyqcu.lattice as lattice

CUDA = torch.device('cuda')
DT = torch.complex64

try:
    LAT_F = [16, 16, 16, 8]        # lat_fine_odd 约定: [Lx,Ly,Lz,Lt//2]
    COARSE = [8, 8, 8, 4]
    E = 12
    U = _torch.zeros(size=[3, 3, 4] + [16, 16, 16, 16], dtype=DT, device=CUDA)
    lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
    clover = _torch.zeros(size=[4, 3, 4, 3] + [16, 16, 16, 16], dtype=DT, device=CUDA)
    op = dslash.operator(U=U, kappa=torch.Tensor([0.125]),
                         clover_term=clover, support_parity=True, verbose=False)

    from pyqcu.tools._multigrid import (local_orthogonalize, BatchedLocalSchur,
                                        build_stencil_local, apply_stencil)
    nv = _torch.randn(size=[E, E] + LAT_F, dtype=DT, device=CUDA)
    lonv = local_orthogonalize(null_vecs=nv, coarse_lat_size=COARSE, verbose=False)
    print(f"[C7] lonv shape={tuple(lonv.shape)} finite={bool(torch.isfinite(lonv.real).all())}", flush=True)

    lsch = BatchedLocalSchur(op, *LAT_F, W=10)
    hnn, hdg, sit = build_stencil_local(lsch, lonv, E, LAT_F, COARSE,
                                        DT, CUDA, verbose=False)
    for name, t in (("hop_nn", hnn), ("hop_diag", hdg), ("sit", sit)):
        nz = float(t.abs().max().item())
        fin = bool(torch.isfinite(t.real).all() and torch.isfinite(t.imag).all())
        assert fin and nz > 0, f"{name} finite={fin} max_abs={nz}"
        print(f"[C7] {name}: shape={tuple(t.shape)} max_abs={nz:.3e}", flush=True)

    v_c = _torch.randn(size=[E, *COARSE], dtype=DT, device=CUDA)
    out = apply_stencil(hnn, hdg, sit, v_c)
    assert out.shape == v_c.shape and torch.isfinite(out.real).all()
    print(f"[C7] apply_stencil ok shape={tuple(out.shape)} norm={float(tools.norm(out)):.3e}", flush=True)

    # 确定性: 同 lonv 二次构建逐元素一致
    hnn2, hdg2, sit2 = build_stencil_local(lsch, lonv, E, LAT_F, COARSE,
                                           DT, CUDA, verbose=False)
    dmax = max(float((a - b).abs().max().item()) for a, b in ((hnn, hnn2), (hdg, hdg2), (sit, sit2)))
    assert dmax == 0.0, dmax
    print(f"[C7] deterministic rebuild diff=0", flush=True)
    print("[C7][SUMMARY] ALL PASS")
except Exception:
    traceback.print_exc()
    print("[C7][SUMMARY] FAIL")
