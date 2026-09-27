import traceback
import torch
import pyqcu.cann as _torch
import pyqcu.tools as tools
import pyqcu.dslash as dslash
import pyqcu.lattice as lattice

CUDA = torch.device('cuda')
DT = torch.complex64

try:
    LAT_F = [16, 16, 16, 8]
    COARSE = [8, 8, 8, 4]
    E = 12
    kappa = 0.125
    U = _torch.zeros(size=[3, 3, 4] + [16, 16, 16, 16], dtype=DT, device=CUDA)
    lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
    clover = _torch.zeros(size=[4, 3, 4, 3] + [16, 16, 16, 16], dtype=DT, device=CUDA)
    op = dslash.operator(U=U, kappa=torch.Tensor([kappa]),
                         clover_term=clover, support_parity=True, verbose=False)

    from pyqcu.tools._multigrid import (local_orthogonalize, BatchedLocalSchur,
                                        build_stencil_local)
    from pyqcu.testing import verify_nullvecs

    nv = _torch.randn(size=[E, E] + LAT_F, dtype=DT, device=CUDA)
    lonv_blk = local_orthogonalize(null_vecs=nv, coarse_lat_size=COARSE, verbose=False)
    # verify_nullvecs 的 restrict/prolong/ortho 分支均要求 10 维块结构(C8 误传全局布局为崩溃根因)
    g = lonv_blk

    lsch = BatchedLocalSchur(op, *LAT_F, W=10)
    hnn, hdg, sit = build_stencil_local(lsch, lonv_blk, E, LAT_F, COARSE,
                                        DT, CUDA, verbose=False)

    def S_matvec(v_o):
        # 奇偶 Schur 补 — kappa 已吸收进 hopping 组件(matvec_parity 官方口径,
        # 与 C9 全格参考等价性一致;勿再显式乘 k^2)
        return op.matvec_parity(v_o)

    diag = verify_nullvecs(S=S_matvec, lonv=g, lat_fine=LAT_F,
                           lat_coarse=COARSE, n_sample=4,
                           stencil=(hnn, hdg, sit), verbose=True)
    gd = diag.get("galerkin_rel_diff")
    print(f"[C8][GALERKIN] rel_diff={gd}", flush=True)
    assert gd is not None and float(gd) < 1e-3, diag
    print(f"[C8][SUMMARY] ALL PASS null_ratios={['%.3f' % r for r in diag['null_ratios']]} "
          f"ortho={diag['ortho_offdiag_max']:.1e}", flush=True)
except Exception:
    traceback.print_exc()
    print("[C8][SUMMARY] FAIL")
