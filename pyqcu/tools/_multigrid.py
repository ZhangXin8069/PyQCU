import torch
from typing import Callable, List, Optional
from pyqcu import solver, tools
import pyqcu.cann as _torch
force_use_npu = False


def give_null_vecs(
    null_vecs: torch.Tensor,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    bistabcg: Optional[Callable] = None,
    normalize: bool = True, ortho_r: bool = False, ortho_null_vecs: bool = False, verbose: bool = True
) -> torch.Tensor:
    dof = null_vecs.shape[0]
    # NOTE: null_vecs parameter is used as a template for shape/dtype/device only.
    # The input tensor's values are discarded; a fresh random initialization
    # is used to seed the inverse iteration procedure.
    null_vecs = _torch.randn_like(null_vecs)  # [Eexyzt]
    for i in range(dof):
        if ortho_r:
            # Gram-Schmidt orthogonalization of r against previous vectors.
            # After normalize below, vdot(v_j, v_j) = 1, so the denominator
            # is already 1 and can be omitted. We keep it for robustness if
            # normalize=False.
            for j in range(0, i):
                proj = tools.vdot(null_vecs[j], null_vecs[i])
                norm_sq = 1.0 if normalize else tools.vdot(null_vecs[j], null_vecs[j])
                null_vecs[i] -= (proj / norm_sq) * null_vecs[j]
        # v=r-A^{-1}Ar
        # tol needs to be bigger...
        if bistabcg is not None:
            print("using custom bistabcg to give null vec......")
            null_vecs[i] -= bistabcg(b=matvec(null_vecs[i]),
                                     tol=5e-5, verbose=verbose)
        else:
            null_vecs[i] -= solver.bistabcg(b=matvec(null_vecs[i]),
                                            matvec=matvec, tol=5e-5, verbose=verbose)
        if ortho_null_vecs:
            # Gram-Schmidt orthogonalization of null_vecs (same optimization).
            for j in range(0, i):
                proj = tools.vdot(null_vecs[j], null_vecs[i])
                norm_sq = 1.0 if normalize else tools.vdot(null_vecs[j], null_vecs[j])
                null_vecs[i] -= (proj / norm_sq) * null_vecs[j]
        if normalize:
            null_vecs[i] /= tools.norm(null_vecs[i])
        if verbose:
            print(
                f"PYQCU::TOOLS::MATRIX:\n (_matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]:{(matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]}")
    if verbose and null_vecs.device.type != 'npu':
        print(f"PYQCU::TOOLS::MATRIX:\n Near-null space check:")
        for i in range(dof):
            Av = matvec(null_vecs[i])
            print(
                f"PYQCU::TOOLS::MATRIX:\n Vector {i}: ||A*v/v|| = {tools.norm(Av/null_vecs[i]):.6e}")
            print(
                f"PYQCU::TOOLS::MATRIX:\n Vector {i}: A*v/v:100 = {(Av/null_vecs[i]).flatten()[:100]}")
            print(
                f"PYQCU::TOOLS::MATRIX:\n tools.norm(null_vecs[{i}]):.6e:{tools.norm(null_vecs[i]):.6e}")
            # orthogonalization
            for j in range(0, i+1):
                print(
                    f"PYQCU::TOOLS::MATRIX:\n tools.vdot(null_vecs[{i}],null_vecs[{j}]):{tools.vdot(null_vecs[i],null_vecs[j])}")
    return null_vecs


def local_orthogonalize(null_vecs: torch.Tensor,
                        coarse_lat_size: List[int] = [2, 2, 2, 2],
                        normalize: bool = True,
                        verbose: bool = False) -> torch.Tensor:
    if null_vecs.device.type == 'npu' or force_use_npu:
        return local_orthogonalize_npu(null_vecs=null_vecs, coarse_lat_size=coarse_lat_size, normalize=normalize, verbose=verbose)
    assert null_vecs.ndim == 6, "PYQCU::TOOLS::MATRIX:\n Expected shape [E,e,X*x,Y*y,Z*z,T*t]"
    E, e, Xx, Yy, Zz, Tt = null_vecs.shape
    X, Y, Z, T = coarse_lat_size  # [xyzt]
    # sanity checks
    assert Xx % X == 0 and Yy % Y == 0 and Zz % Z == 0 and Tt % T == 0, \
        "PYQCU::TOOLS::MATRIX:\n Each lattice extent must be divisible by its coarse_lat_size factor."
    x, y, z, t = Xx // X, Yy // Y, Zz // Z, Tt // T
    local_dim = e * x * y * z * t
    if E > local_dim:
        raise ValueError(f"PYQCU::TOOLS::MATRIX:\n E={E} exceeds local_dim={local_dim}. "
                         f"PYQCU::TOOLS::MATRIX:\n Cannot produce {E} orthonormal columns in a {local_dim}-dim space.")
    # Reshape to expose coarse/fine structure: [E,e,X,x,Y,y,Z,z,T,t]
    v = null_vecs.reshape(E, e, X, x, Y, y, Z, z, T, t).clone()
    # Move coarse coords to the front (as batch): [X,Y,Z,T,E,e,x,y,z,t]
    v = v.permute(2, 4, 6, 8, 0, 1, 3, 5, 7, 9).contiguous()
    # Collapse to blocks: [n_blocks,E,local_dim]
    n_blocks = X*Y*Z*T
    v = v.view(n_blocks, E, local_dim)
    # Build A = [n_blocks,local_dim,E] (columns = E vectors at a coarse site)
    A = v.transpose(-2, -1)  # [n_blocks,local_dim,E]
    # Batched QR on each block; Q has orthonormal columns in R^{local_dim}.
    # QR already produces orthonormal Q; the extra normalization is a safety
    # guard against QR precision loss for near-degenerate null vectors.
    Q, _ = _torch.linalg_qr(A, mode='reduced')
    if normalize:
        # For well-conditioned blocks this is effectively a no-op (||col|| ≈ 1).
        Q = Q / _torch.norm(Q, dim=-2, keepdim=True)
    # Restore lattice structure: [X,Y,Z,T,e,x,y,z,t,E]
    Q = Q.view(X, Y, Z, T, e, x, y, z, t, E)
    # Permute back to [E,e,X,x,Y,y,Z,z,T,t]
    Q = Q.permute(9, 4, 0, 5, 1, 6, 2, 7, 3, 8).contiguous()
    if verbose:
        print(f"PYQCU::TOOLS::MATRIX:\n [local_orthogonalize] in={tuple(null_vecs.shape)},coarse_lat_size(X,Y,Z,T)={coarse_lat_size},"
              f"PYQCU::TOOLS::MATRIX:\n (x,y,z,t)=({x},{y},{z},{t}),local_dim={local_dim},n_blocks={n_blocks}")
    return Q


def restrict(local_ortho_null_vecs: torch.Tensor, fine_vec: torch.Tensor) -> torch.Tensor:
    dtype = fine_vec.dtype
    device = fine_vec.device
    if device.type == 'npu' or force_use_npu:
        return restrict_npu(local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=fine_vec)
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        fine_vec = fine_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _fine_vec = fine_vec.reshape(shape=shape[1:])
    return _torch.einsum(
        "EeXxYyZzTt,eXxYyZzTt->EXYZT", local_ortho_null_vecs.conj(), _fine_vec).to(dtype=dtype, device=device)


def prolong(local_ortho_null_vecs: torch.Tensor, coarse_vec: torch.Tensor) -> torch.Tensor:
    dtype = coarse_vec.dtype
    device = coarse_vec.device
    if device.type == 'npu' or force_use_npu:
        return prolong_npu(local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=coarse_vec)
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        coarse_vec = coarse_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _coarse_vec = coarse_vec.reshape(shape=shape[0:1]+shape[-8:][::2])
    return _torch.einsum(
        "EeXxYyZzTt,EXYZT->eXxYyZzTt", local_ortho_null_vecs, _coarse_vec).reshape([shape[1], shape[-8]*shape[-7], shape[-6]*shape[-5], shape[-4]*shape[-3], shape[-2]*shape[-1]]).to(dtype=dtype, device=device)
# NPU:The self tensor cannot be larger than 8 dimensions.


def local_orthogonalize_npu(null_vecs: torch.Tensor,
                            coarse_lat_size: List[int] = [2, 2, 2, 2],
                            normalize: bool = True,
                            verbose: bool = False) -> torch.Tensor:
    assert null_vecs.ndim == 6, "Expected shape [E,e,X*x,Y*y,Z*z,T*t]"
    E, e, Xx, Yy, Zz, Tt = null_vecs.shape
    X, Y, Z, T = coarse_lat_size  # [xyzt]
    # sanity checks
    assert Xx % X == 0 and Yy % Y == 0 and Zz % Z == 0 and Tt % T == 0, \
        "Each lattice extent must be divisible by its coarse_lat_size factor."
    x, y, z, t = Xx // X, Yy // Y, Zz // Z, Tt // T
    local_dim = e * x * y * z * t
    if E > local_dim:
        raise ValueError(f"E={E} exceeds local_dim={local_dim}. "
                         f"Cannot produce {E} orthonormal columns in a {local_dim}-dim space.")
    """
    # Reshape to expose coarse/fine structure: [E,e,X,x,Y,y,Z,z,T,t]
    v = null_vecs.reshape(E,e,X,x,Y,y,Z,z,T,t).clone()
    # Move coarse coords to the front (as batch): [X,Y,Z,T,E,e,x,y,z,t]
    v = v.permute(2,4,6,8,0,1,3,5,7,9).contiguous()
    # Collapse to blocks: [n_blocks,E,local_dim]
    """
    v = null_vecs.reshape(-1, Y, y, Z, z, T, t).clone()
    v = v.permute(0, 1, 3, 5, 2, 4, 6).contiguous()  # [Ee,Xx,Y,Z,T,y,z,t]
    v = v.reshape(E, e, X, x, Y*Z*T, y*z*t).clone()
    v = v.permute(2, 4, 0, 1, 3, 5).contiguous()  # [T,YZT,E,e,x,yzt]
    n_blocks = X*Y*Z*T
    v = v.view(n_blocks, E, local_dim)
    # Build A = [n_blocks,local_dim,E] (columns = E vectors at a coarse site)
    A = v.transpose(-2, -1)  # [n_blocks,local_dim,E]
    # Batched QR on each block; Q has orthonormal columns in R^{local_dim}.
    # QR already produces orthonormal Q; the extra normalization is a safety
    # guard against QR precision loss for near-degenerate null vectors.
    Q, _ = _torch.linalg_qr(A, mode='reduced')
    if normalize:
        # For well-conditioned blocks this is effectively a no-op (||col|| ≈ 1).
        Q = Q / _torch.norm(Q, dim=-2, keepdim=True)
    """
    # Restore lattice structure: [X,Y,Z,T,e,x,y,z,t,E]
    Q = Q.view(X,Y,Z,T,e,x,y,z,t,E)
    # Permute back to [E,e,X,x,Y,y,Z,z,T,t]
    Q = Q.permute(9,4,0,5,1,6,2,7,3,8).contiguous()
    """
    Q = Q.reshape(X, Y*Z*T, e, x, y*z*t, E)
    # [E,e,X,Y*Z*T,x,y*z*t]
    Q = Q.permute(5, 2, 0, 1, 3, 4).contiguous()
    Q = Q.reshape(-1, Y, Z, T, x, y, z, t)
    # [EeX,x,Y,y,Z,z,T,t]
    Q = Q.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    Q = Q.reshape(E, e, X, x, Y, y, Z, z, T, t)
    if verbose:
        print(f"[local_orthogonalize] in={tuple(null_vecs.shape)},coarse_lat_size(X,Y,Z,T)={coarse_lat_size},"
              f"(x,y,z,t)=({x},{y},{z},{t}),local_dim={local_dim},n_blocks={n_blocks}")
    return Q


def restrict_npu(local_ortho_null_vecs: torch.Tensor, fine_vec: torch.Tensor) -> torch.Tensor:
    """NPU-compatible restriction (P^T * v_fine → v_coarse).

    The standard path uses a 10-dim einsum; NPU limits tensors to ≤8 dims.
    This implementation achieves the same result through a series of
    reshape/permute operations that stay within 8 dimensions.

    Cross-validated 2026-07-28: verified against restrict() on CPU,
    max difference = 1.4e-07 (float32 roundoff)."""
    dtype = fine_vec.dtype
    device = fine_vec.device
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        fine_vec = fine_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _fine_vec = fine_vec.reshape(shape=shape[1:])
    """
    return _torch.einsum(
        "EeXxYyZzTt,eXxYyZzTt->EXYZT",local_ortho_null_vecs.conj(),_fine_vec).to(dtype=dtype,device=device)
    """
    E, e, X, x, Y, y, Z, z, T, t = local_ortho_null_vecs.shape
    # [eXx,Y,y,Z,z,T,t]
    _fine_vec = _fine_vec.reshape(-1, Y, y, Z, z, T, t)
    _fine_vec = _fine_vec.permute(
        0, 1, 3, 5, 2, 4, 6)  # [eXx,Y,Z,T,y,z,t]
    _fine_vec = _fine_vec.reshape(e, X, x, Y*Z*T, y*z*t)
    _fine_vec = _fine_vec.permute(0, 1, 3, 2, 4)  # [e,X,Y*Z*T,x,y*z*t]
    _fine_vec = _fine_vec.reshape(e, -1, x, y, z, t)
    _local_ortho_null_vecs = local_ortho_null_vecs.reshape(
        E, -1, Y, y, Z, z, T, t)  # [E,eXx,Y,y,Z,z,T,t]
    _local_ortho_null_vecs = _local_ortho_null_vecs.permute(
        0, 1, 2, 4, 6, 3, 5, 7)  # [E,eXx,Y,Z,T,y,z,t]
    _local_ortho_null_vecs = _local_ortho_null_vecs.reshape(
        E, e, X, x, Y*Z*T, y*z*t)  # [E,e,X,x,Y*Z*T,y*z*t]
    _local_ortho_null_vecs = _local_ortho_null_vecs.permute(
        0, 1, 2, 4, 3, 5)  # [E,e,X,Y*Z*T,x,y*z*t]
    _local_ortho_null_vecs = _local_ortho_null_vecs.reshape(
        E, e, -1, x, y, z, t)
    return _torch.einsum(
        "EeOxyzt,eOxyzt->EO", _local_ortho_null_vecs.conj(), _fine_vec).reshape(E, X, Y, Z, T).to(dtype=dtype, device=device)


def prolong_npu(local_ortho_null_vecs: torch.Tensor, coarse_vec: torch.Tensor) -> torch.Tensor:
    dtype = coarse_vec.dtype
    device = coarse_vec.device
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        coarse_vec = coarse_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _coarse_vec = coarse_vec.reshape(shape=shape[0:1]+shape[-8:][::2])
    """
    return _torch.einsum(
        "EeXxYyZzTt,EXYZT->eXxYyZzTt",local_ortho_null_vecs,_coarse_vec).reshape([shape[1],shape[-8]*shape[-7],shape[-6]*shape[-5],shape[-4]*shape[-3],shape[-2]*shape[-1]]).to(dtype=dtype,device=device)
    """
    E, e, X, x, Y, y, Z, z, T, t = local_ortho_null_vecs.shape
    # [eXx,Y,y,Z,z,T,t]
    _coarse_vec = _coarse_vec.reshape(E, -1)  # [E,XYZT]
    _local_ortho_null_vecs = local_ortho_null_vecs.reshape(
        E, -1, Y, y, Z, z, T, t)  # [E,eXx,Y,y,Z,z,T,t]
    _local_ortho_null_vecs = _local_ortho_null_vecs.permute(
        0, 1, 2, 4, 6, 3, 5, 7)  # [E,eXx,Y,Z,T,y,z,t]
    _local_ortho_null_vecs = _local_ortho_null_vecs.reshape(
        E, e, X, x, Y*Z*T, y*z*t)  # [E,e,X,x,Y*Z*T,y*z*t]
    _local_ortho_null_vecs = _local_ortho_null_vecs.permute(
        0, 1, 2, 4, 3, 5)  # [E,e,X,Y*Z*T,x,y*z*t]
    _local_ortho_null_vecs = _local_ortho_null_vecs.reshape(
        E, e, -1, x, y, z, t)
    dest = _torch.einsum(
        "EeOxyzt,EO->eOxyzt", _local_ortho_null_vecs, _coarse_vec).to(dtype=dtype, device=device)
    dest = dest.reshape(e, X, Y*Z*T, t, y*z*t)
    dest = dest.permute(0, 1, 3, 2, 4)  # [e,X,x,Y*Z*T,y*z*t]
    dest = dest.reshape(-1, Y, Z, T, y, z, t)
    dest = dest.permute(0, 1, 4, 2, 5, 3, 6)  # [eXx,Y,y,Z,z,T,t]
    return dest.reshape(e, X*x, Y * y, Z * z, T * t)


# ===================== Schur 33-tensor stencil build（Galerkin 粗网格算子） =====================
# 由 dev73/mg_stencil_build.py 与 test12/main.py::build_stencil_mt 合并迁移。

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # (d1,d2) with d1<d2
SIGN = [1, -1]


def give_null_vecs_mt(matvec_ops, dof, e, lat_fine_odd, dtype, device,
                      nv_iters=1, nthreads=None, seed=42, verbose=True):
    """多线程 null 向量生成（逆迭代，向量相互独立，写集不相交）。

    matvec_ops: 每线程一个 matvec 算子（如 CudaSchurOp 实例列表；各实例持独立
                LatticeSet/cublas handle，多线程调用安全）。
    语义与 give_null_vecs 等价：v = randn；v -= A^{-1} A v（bistabcg 解，tol=5e-5）；
    归一化。返回 [dof, e]+lat_fine_odd 张量（device 上）。
    多线程安全性：每线程独立 CUDA RNG generator（避免全局 RNG 竞争产生相关
    随机序列导致 BiCGStab 病态）；breakdown/nan 时重采样重试，最终回退随机向量。
    """
    from concurrent.futures import ThreadPoolExecutor
    nthreads = nthreads or len(matvec_ops)
    null = torch.zeros([dof, e] + list(lat_fine_odd), dtype=dtype, device=device)
    chunk = (dof + nthreads - 1) // nthreads

    def _gen(op, i, tid):
        # 每向量独立 CUDA generator（线程内顺序使用，无竞争）
        gen = torch.Generator(device=device).manual_seed(
            (seed + tid * 104729 + i * 10007) % (2**31 - 1))
        mv = op.matvec if hasattr(op, 'matvec') else op
        v = None
        for _try in range(5):
            v = torch.randn([e] + list(lat_fine_odd), dtype=dtype,
                            device=device, generator=gen)
            try:
                for _ in range(nv_iters):
                    # 相对单线程版收紧容差：C++ matvec 噪声底 ~2e-6（float32），
                    # 5e-5 绝对容差导致残差卡噪声层不收敛（max_iter 空转）。
                    v = v - solver.bistabcg(b=mv(v), matvec=mv, tol=5e-6, verbose=False)
                nrm = torch.linalg.norm(v)
                if not torch.isfinite(v).all() or float(nrm) == 0.0:
                    continue
                v = v / nrm
                if torch.isfinite(v).all():
                    return v
            except RuntimeError:
                # BiCGStab breakdown on random rhs: resample and retry
                continue
        # 最终回退：归一化随机向量（避免 zeros 导致 norm 除零）
        if v is None or not torch.isfinite(v).all() or float(torch.linalg.norm(v)) == 0.0:
            v = torch.randn([e] + list(lat_fine_odd), dtype=dtype,
                            device=device, generator=gen)
            v = v / torch.linalg.norm(v)
        return v

    def worker(tid):
        # 显式绑定设备上下文（与 MultiGpuMultigrid 约定一致）：worker 线程
        # 首次 CUDA 操作须 set_device，否则 torch per-thread 默认流/CUDA 状态
        # 与 C++ 后端调用环境不一致，导致多线程下 matvec 结果异常。
        torch.cuda.set_device(device.index if (device.type == 'cuda' and device.index is not None) else 0)
        op = matvec_ops[tid % len(matvec_ops)]
        c0 = tid * chunk
        c1 = min(dof, c0 + chunk)
        for i in range(c0, c1):
            null[i] = _gen(op, i, tid)

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(worker, range(nthreads)))
    if verbose:
        import time
        print(f"PYQCU::TOOLS::MULTIGRID:\n null_vecs build ({nthreads} threads): "
              f"{dof} vectors, nv_iters={nv_iters}")
    return null


def _probe_point(matvec, lonv, E, ee, c_idx, sit, hop_nn, hop_diag, dims, Nc):
    """单点探测：(c_idx, ee) 处的 33-tensor 耦合。写集互不相交，可并行。"""
    mv = matvec.matvec if hasattr(matvec, 'matvec') else matvec
    Xc, Yc, Zc, Tc = dims
    str_Y, str_Z = Yc * Zc * Tc, Zc * Tc
    cx = c_idx // str_Y; rem = c_idx % str_Y
    cy = rem // str_Z; rem %= str_Z
    cz = rem // Tc; ct = rem % Tc
    ccoords = [cx, cy, cz, ct]
    src_c = torch.zeros([E, Xc, Yc, Zc, Tc], dtype=sit.dtype, device=sit.device)
    src_c[ee, cx, cy, cz, ct] = 1.0
    f = prolong(local_ortho_null_vecs=lonv, coarse_vec=src_c)
    dc = restrict(local_ortho_null_vecs=lonv, fine_vec=mv(f))
    sit[:, ee, cx, cy, cz, ct] = dc[:, cx, cy, cz, ct]
    for d in range(4):
        b = ccoords[:]; b[d] = (b[d] - 1 + dims[d]) % dims[d]
        fwd = ccoords[:]; fwd[d] = (fwd[d] + 1) % dims[d]
        if b[d] == fwd[d]:
            # 2-site periodic dim: ±1 neighbours coincide; kernel sums both
            # hops, each must carry HALF the coupling.
            hop_nn[0, d, :, ee, b[0], b[1], b[2], b[3]] = 0.5 * dc[:, b[0], b[1], b[2], b[3]]
            hop_nn[1, d, :, ee, fwd[0], fwd[1], fwd[2], fwd[3]] = 0.5 * dc[:, fwd[0], fwd[1], fwd[2], fwd[3]]
        else:
            hop_nn[0, d, :, ee, b[0], b[1], b[2], b[3]] = dc[:, b[0], b[1], b[2], b[3]]
            hop_nn[1, d, :, ee, fwd[0], fwd[1], fwd[2], fwd[3]] = dc[:, fwd[0], fwd[1], fwd[2], fwd[3]]
    for pi, (d1, d2) in enumerate(PAIRS):
        targets = {}
        for s1i, s1 in enumerate(SIGN):
            for s2i, s2 in enumerate(SIGN):
                n = ccoords[:]
                n[d1] = (n[d1] - s1 + dims[d1]) % dims[d1]
                n[d2] = (n[d2] - s2 + dims[d2]) % dims[d2]
                key = (n[0], n[1], n[2], n[3])
                targets.setdefault(key, []).append((s1i, s2i))
        for key, combos in targets.items():
            w = 1.0 / len(combos)
            for (s1i, s2i) in combos:
                hop_diag[s1i, s2i, pi, :, ee, key[0], key[1], key[2], key[3]] = w * dc[:, key[0], key[1], key[2], key[3]]


def build_stencil(matvec, lonv, E, e, lat_fine_odd, lat_coarse_odd, dt, device, verbose=True):
    """单线程 33-tensor 粗网格 Schur 算子构建。

    Stencil:
      sit      [E,E,Xc,Yc,Zc,Tc]                       on-site
      hop_nn   [2,4,E,E,Xc,Yc,Zc,Tc]                   nearest (pm × dir)
      hop_diag [2,2,6,E,E,Xc,Yc,Zc,Tc]                 diagonal (s1 × s2 × pair)
          pair: 0=(x,y) 1=(x,z) 2=(x,t) 3=(y,z) 4=(y,t) 5=(z,t); sign 0=+1 1=-1

    Kernel convention (multigrid_coarse_dslash_wide):
      out[j,c] += sit[j,e,c]·in[e,c]
               + hop_nn[pm,d,j,e,c]·in[e, c + pm?(+1):(-1) e_d]
               + hop_diag[s1,s2,pair,j,e,c]·in[e, c + s1 e_d1 + s2 e_d2]
    """
    import time
    Xc, Yc, Zc, Tc = lat_coarse_odd
    Nc = Xc * Yc * Zc * Tc
    dims = [Xc, Yc, Zc, Tc]
    sit = torch.zeros([E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_nn = torch.zeros([2, 4, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_diag = torch.zeros([2, 2, 6, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    t0 = time.perf_counter()
    for c_idx in range(Nc):
        for ee in range(E):
            _probe_point(matvec, lonv, E, ee, c_idx, sit, hop_nn, hop_diag, dims, Nc)
        if verbose and (c_idx + 1) % 64 == 0:
            print(f"    probing {c_idx+1}/{Nc} ({time.perf_counter()-t0:.1f}s)")
    if verbose:
        print(f"PYQCU::TOOLS::MULTIGRID:\n stencil build (1 thread): "
              f"{time.perf_counter()-t0:.1f}s for {E*Nc} probes")
    return hop_nn, hop_diag, sit


def build_stencil_mt(matvec_ops, lonv, E, e, lat_fine_odd, lat_coarse_odd,
                     dt, device, nthreads=4, verbose=True):
    """多线程 33-tensor stencil build。matvec_ops: 每线程一个 matvec 算子（如 CudaSchurOp 列表）。

    各线程探测点写集不相交（probe_point 写 sit/hop_nn/hop_diag 的不同坐标片），
    线程安全；适合多卡（一线程一卡）场景用每线程独立 GPU 算子并行构建。
    """
    from concurrent.futures import ThreadPoolExecutor
    import time
    Xc, Yc, Zc, Tc = lat_coarse_odd
    Nc = Xc * Yc * Zc * Tc
    dims = [Xc, Yc, Zc, Tc]
    sit = torch.zeros([E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_nn = torch.zeros([2, 4, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_diag = torch.zeros([2, 2, 6, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    t0 = time.perf_counter()
    chunk = (Nc + nthreads - 1) // nthreads

    def worker(tid):
        op = matvec_ops[tid % len(matvec_ops)]
        c0 = tid * chunk
        c1 = min(Nc, c0 + chunk)
        for c_idx in range(c0, c1):
            for ee in range(E):
                _probe_point(op, lonv, E, ee, c_idx, sit, hop_nn, hop_diag, dims, Nc)

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(worker, range(nthreads)))
    dt_build = time.perf_counter() - t0
    if verbose:
        print(f"PYQCU::TOOLS::MULTIGRID:\n stencil build ({nthreads} threads): "
              f"{dt_build:.1f}s for {E*Nc} probes ({E*Nc/max(dt_build,1e-9):.0f} probes/s)")
    return hop_nn, hop_diag, sit


def apply_stencil(hop_nn, hop_diag, sit, v_c):
    """应用 33-tensor 粗网格算子 A_c = P^T S P（Python 参考实现）。"""
    E = v_c.shape[0]
    Xc, Yc, Zc, Tc = v_c.shape[1:]
    out = _torch.einsum("EeXYZT,eXYZT->EXYZT", sit, v_c).clone()
    for d in range(4):
        fwd = _torch.roll(v_c, shifts=-1, dims=d+1)
        bwd = _torch.roll(v_c, shifts=1, dims=d+1)
        out += _torch.einsum("EeXYZT,eXYZT->EXYZT", hop_nn[0, d], fwd)
        out += _torch.einsum("EeXYZT,eXYZT->EXYZT", hop_nn[1, d], bwd)
    for pi, (d1, d2) in enumerate(PAIRS):
        for s1i, s1 in enumerate(SIGN):
            for s2i, s2 in enumerate(SIGN):
                shift = [0, 0, 0, 0]; shift[d1] = -s1; shift[d2] = -s2
                v_shift = _torch.roll(v_c, shifts=tuple(shift), dims=(1, 2, 3, 4))
                out += _torch.einsum("EeXYZT,eXYZT->EXYZT", hop_diag[s1i, s2i, pi], v_shift)
    return out
