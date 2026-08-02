#!/usr/bin/env python3
"""null_vecs 正确性验证。

对照 pyqcu/tools/_multigrid.py 的参考实现，验证：
  1. 零模质量：||A·v|| / ||v||（应明显小于最大本征值，越小越好）
  2. 块内正交性：<v_i, v_j> ≈ δ_ij（local_orthogonalize 后的 lonv）
  3. C++ restrict / prolong 与 Python einsum 版本逐元素一致
  4. C++ 33-tensor coarse dslash 与 Python A_c = P^T S P 一致

用法：source ./env.sh && CUDA_VISIBLE_DEVICES=0 python examples/qcu/mg_v4_verify_nv.py
"""
import torch, os, sys
from pyqcu import tools, dslash
import pyqcu.cuda.define as define
from pyqcu.cuda import qcu
from pyqcu.cuda.define import params, argv, set_ptrs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg_nullvec_cache import load_coarse_ops
from mg_stencil_build import apply_stencil
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    device = torch.device("cuda")
    Lx,Ly,Lz,Lt = 8,8,8,16
    MASS=0.05; KAPPA=1.0/(2*MASS+8); ATOL=1e-6; DT=define._LAT_C64_; dt=define.dtype(DT)
    E=48
    ls=[Lx,Ly,Lz,Lt]
    # ---- build the Schur operator (gauge via C++ gauss, same as the bench) ----
    _csm = _load("csm", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "conftest.schur.multigrid.py"))
    build_config = _csm.build_config
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,2,[12,E],[2,2,2,2],10,15,1e5,DT)
    g=torch.zeros([2,3,3,4]+define.lat_shape(params),dtype=dt,device=device)
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    qcu_U=tools.poooxyzt2oooxyzt(g)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    op = dslash.operator(U=qcu_U, clover_term=ref_cl, kappa=torch.Tensor([KAPPA]),
                         support_parity=True, verbose=False)
    S = op.matvec_parity
    lat_fine=[Lx,Ly,Lz,Lt//2]      # odd lattice [8,8,8,8]
    lat_coarse=[x//2 for x in lat_fine]
    lonv, hnn, hdg, sit = load_coarse_ops(42, ls, 1, E, 2, "c64", device)

    # ---- 1. null-vector quality ----
    print("="*70); print("[1] null-vector quality  ||S·v||/||v||")
    for k in range(min(4, E)):
        Av = S(lonv[k].reshape([12]+lat_fine)).reshape(E, -1)
        # lonv[k] is shape [E, 12, X,Y,Z,T] (coarse dof × fine dof); for the
        # quality check project each coarse column onto the fine operator.
        # We instead compute ||P^T S P e||/||P^T P e|| style: use A_c column.
        ratio = torch.linalg.norm(Av) / torch.linalg.norm(lonv[k])
        print(f"  vector {k}: ||S·lonv||/||lonv|| = {ratio.item():.4e}")
    # reference: spectrum estimate via power iteration on S
    v = torch.randn([12]+lat_fine, dtype=dt, device=device); v/=torch.linalg.norm(v)
    for _ in range(20):
        w = S(v).flatten(); vf = v.flatten()
        lam = torch.real(torch.vdot(w, vf))
        v = w.reshape(v.shape)/torch.linalg.norm(w)
    print(f"  (largest |λ| of S ≈ {abs(lam.item()):.4e})  →  null ratio should be ≪ this")

    # ---- 2. local orthonormality ----
    print("="*70); print("[2] local orthonormality  <v_i, v_j> within a coarse block")
    # lonv shape [E, e, X,x,Y,y,Z,z,T,t] (blocked) -> reshape to [E, 12, X,Y,Z,T]
    X,Y,Z,T = lat_coarse; x,y,z,t = [lat_fine[d]//lat_coarse[d] for d in range(4)]
    vb = lonv.reshape(E, 12, X, x, Y, y, Z, z, T, t)
    # block (0,0,0,0): local_dim = 12*x*y*z*t
    block = vb[:, :, 0, :, 0, :, 0, :, 0, :].reshape(E, -1)  # [E, local_dim]
    G = block @ block.conj().T
    off = G - torch.eye(E, dtype=dt, device=device)
    print(f"  block(0,0,0,0) off-diagonal max |<vi,vj>-δ| = {off.abs().max().item():.3e}")
    print(f"  block(0,0,0,0) diag min {torch.diag(G).real.min().item():.6f} max {torch.diag(G).real.max().item():.6f}")

    # ---- 3. C++ restrict/prolong vs Python ----
    print("="*70); print("[3] C++ restrict/prolong vs Python einsum")
    fine_vec = torch.randn([12]+lat_fine, dtype=dt, device=device)
    r_py = tools.restrict(local_ortho_null_vecs=lonv, fine_vec=fine_vec)
    # C++ restrict: need params for the fine lattice + null_vecs pointer
    params[define._LAT_X_]=lat_fine[0]; params[define._LAT_Y_]=lat_fine[1]
    params[define._LAT_Z_]=lat_fine[2]; params[define._LAT_T_]=lat_fine[3]
    params[define._MG_LEVEL1_X_]=X; params[define._MG_LEVEL1_Y_]=Y
    params[define._MG_LEVEL1_Z_]=Z; params[define._MG_LEVEL1_T_]=T
    params[define._MG_LEVEL1_E_]=E; params[define._MG_NUM_LEVEL_]=12
    out = torch.zeros([E,X,Y,Z,T], dtype=dt, device=device)
    qcu.applyMultigridRestrictQcu(out, fine_vec, lonv, set_ptrs, params)
    print(f"  restrict max|C++-py| = {(out-r_py).abs().max().item():.3e}  rel={((out-r_py).abs().max()/r_py.abs().max()).item():.3e}")
    coarse_vec = torch.randn([E,X,Y,Z,T], dtype=dt, device=device)
    p_py = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=coarse_vec)
    out2 = torch.zeros([12]+lat_fine, dtype=dt, device=device)
    params[define._MG_NUM_LEVEL_]=12
    qcu.applyMultigridProLongQcu(out2, coarse_vec, lonv, set_ptrs, params)
    print(f"  prolong max|C++-py| = {(out2-p_py).abs().max().item():.3e}  rel={((out2-p_py).abs().max()/p_py.abs().max()).item():.3e}")

    # ---- 4. C++ coarse dslash (33-tensor) vs Python A_c ----
    print("="*70); print("[4] C++ 33-tensor coarse dslash vs Python A_c = P^T S P")
    src_c = torch.randn([E,X,Y,Z,T], dtype=dt, device=device)
    # Python A_c via operator-free Galerkin: A_c·e = P^T S P e
    def Ac(v):
        f = tools.prolong(local_ortho_null_vecs=lonv, coarse_vec=v)
        return tools.restrict(local_ortho_null_vecs=lonv, fine_vec=S(f))
    ref = Ac(src_c)
    # C++ stencil application (same layout as the kernel)
    cu = apply_stencil(hnn, hdg, sit, src_c)
    rel = (cu-ref).abs().max().item() / ref.abs().max().item()
    print(f"  coarse dslash rel diff = {rel:.3e}")

if __name__=="__main__":
    main()
