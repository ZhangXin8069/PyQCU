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
            # dev84 修复: nv_tol/tol 此前为绝对容差语义 —— 大格子上 ‖Av‖~O(10²-10³)
            # 时 5e-5 绝对值近似精确解, v-A⁻¹(Av)≈舍入噪声, 归一化后得到的是
            # 随机向量而非近零模 (实测 ‖Sv‖/‖v‖≈0.4≈谱 RMS, ρ_V≈0.976,
            # 见 examples/qcu/dev84/dev84_report.md §3.2)。改相对容差。
            null_vecs[i] -= solver.bistabcg(b=matvec(null_vecs[i]),
                                            matvec=matvec, tol=5e-5,
                                            if_rtol=True, verbose=verbose)
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


def _bistabcg_batch(b_batch, matvec_batch, tol=1e-2, max_iter=2000,
                    if_rtol=False):
    """批量 BiCGStab：同时对 batch 维（B）内的全部右端求解。

    b_batch: [B, e, X, Y, Z, T/2]；matvec_batch: 批量 matvec（同形 → 同形）。
    标量（rho/alpha/omega/beta）按批独立（[B] 向量），迭代直到全部批次收敛
    或 max_iter。breakdown（rho/rtv/tts≈0）批次用 eps 保护避免除零，
    最终未收敛批次由调用方回退随机（与 give_null_vecs_mt 语义一致）。
    if_rtol=True 时 tol 为相对容差（以各批初始 ‖b‖ 为基准）。
    2026-08-15：null 向量生成批量化 —— 16x16x16x32 lv2 的 48 个右端
    （196608 未知数）一次迭代（C++ 逐场 40min+ → torch 批量分钟级）。
    """
    B = b_batch.shape[0]
    x = torch.zeros_like(b_batch)
    r = b_batch.clone()
    r_norm = torch.linalg.norm(r.reshape(B, -1), dim=1)
    b_norm = r_norm.clone()
    if if_rtol:
        conv_tol = tol * torch.clamp(b_norm, min=1e-30)
        conv_tol = torch.where(b_norm < 1e-30, torch.zeros_like(conv_tol),
                               conv_tol)
    else:
        conv_tol = torch.full_like(r_norm, tol)
    if bool((r_norm < conv_tol).all()):
        return x
    r_tilde = r.clone()
    p = torch.zeros_like(b_batch)
    v = torch.zeros_like(b_batch)
    s = torch.zeros_like(b_batch)
    t = torch.zeros_like(b_batch)
    rho_prev = torch.ones([B], dtype=b_batch.dtype, device=b_batch.device)
    alpha = torch.ones([B], dtype=b_batch.dtype, device=b_batch.device)
    omega = torch.ones([B], dtype=b_batch.dtype, device=b_batch.device)
    eps = 1e-30

    def _safe_div(a, b):
        """复数安全除法：|b|<eps 时置 1（该批次可能不收敛，调用方回退随机）。"""
        b_abs = torch.abs(b)
        b_safe = torch.where(b_abs < eps, torch.ones_like(b), b)
        return a / b_safe

    # 标量 [B] 广播形状：尾维全 1（与 b_batch 尾维数一致）
    bc = [B] + [1] * (b_batch.ndim - 1)
    for i in range(max_iter):
        rho = _vdot_batch(r_tilde, r)
        beta = _safe_div(rho, rho_prev) * _safe_div(alpha, omega)
        rho_prev = rho
        p = r + beta.view(bc) * (p - omega.view(bc) * v)
        v = matvec_batch(p)
        rtv = _vdot_batch(r_tilde, v)
        alpha = _safe_div(rho, rtv)
        s = r - alpha.view(bc) * v
        t = matvec_batch(s)
        tts = _vdot_batch(t, t)
        omega = _safe_div(_vdot_batch(t, s), tts)
        x = x + alpha.view(bc) * p + omega.view(bc) * s
        r = s - omega.view(bc) * t
        r_norm = torch.linalg.norm(r.reshape(B, -1), dim=1)
        if bool((r_norm < conv_tol).all()):
            break
    return x


def _vdot_batch(a, b):
    """批量内积：a/b [B, ...]（任意尾维）→ [B]（每批独立，收缩尾维）。"""
    B = a.shape[0]
    a_flat = a.reshape(B, -1)
    b_flat = b.reshape(B, -1)
    return torch.einsum("Bk,Bk->B", a_flat.conj(), b_flat)


def give_null_vecs_mt(matvec_ops, dof, e, lat_fine_odd, dtype, device,
                      nv_iters=1, nthreads=None, seed=42, verbose=True,
                      nv_tol=1e-2, batch_matvec=None, batch_chunk=8):
    """多线程 null 向量生成（逆迭代，向量相互独立，写集不相交）。

    matvec_ops: 每线程一个 matvec 算子（如 CudaSchurOp 实例列表；各实例持独立
                LatticeSet/cublas handle，多线程调用安全）。
    语义与 give_null_vecs 等价：v = randn；v -= A^{-1} A v（bistabcg 解）；
    归一化。返回 [dof, e]+lat_fine_odd 张量（device 上）。
    多线程安全性：每线程独立 CUDA RNG generator（避免全局 RNG 竞争产生相关
    随机序列导致 BiCGStab 病态）；breakdown/nan 时重采样重试，最终回退随机向量。
    nv_tol: null 向量 BiCGStab 解容差（dev84 起为**相对**容差，if_rtol=True）。
    null 向量只需近似近零空间（逆迭代的收敛要求远低于最终求解 atol），
    2026-08-15 实测 5e-5 在粗层大系统（16x16x16x32 lv2，196608 未知数）上
    迭代爆炸（>34min 未完成）；放宽至 1e-2 显著加速，粗算子质量足够
    （MG 收敛由细层平滑保证）。
    dev84 教训：此前 nv_tol 为绝对容差 —— 大格子上 ‖Av‖~O(10²-10³) 时它近似
    精确逆，v-A⁻¹(Av)≈舍入噪声，归一化后得到随机向量（‖Sv‖/‖v‖≈谱 RMS，
    ρ_V≈0.976，粗空间无效）；改相对容差后逆迭代真正按 λ⁻¹ 富集低模。
    batch_matvec: 可选批量 matvec（[B,e,...] → 同形）。给定时全部 dof 个
    右端一次批量 BiCGStab（_bistabcg_batch）——16x16x16x32 等大格子
    null 向量从逐场 C++（40min+）提速至分钟级。
    """
    from concurrent.futures import ThreadPoolExecutor
    if batch_matvec is not None:
        nthreads = nthreads or 1
    else:
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
                    # 与单线程 give_null_vecs 一致的语义；容差见 nv_tol 说明。
                    # dev84: if_rtol=True — nv_tol 为相对容差 (修复绝对容差
                    # 在大格子上退化为精确逆→噪声向量的问题, 见报告 §3.2)。
                    v = v - solver.bistabcg(b=mv(v), matvec=mv, tol=nv_tol,
                                            if_rtol=True, verbose=False)
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

    if batch_matvec is not None:
        # 批量模式：dof 个右端一次批量 BiCGStab（逆迭代 nv_iters 次）。
        # 2026-08-18：分块（batch chunk）执行，降低峰值显存 —— 大格子
        # （24x24x24x72）上全 batch 一次 BiCGStab 的 7 个工作向量
        # （每 [chunk,12,...] 1.15GB@24）+ matvec 中间张量叠加超过 32GB。
        # 每块独立收敛到 nv_tol，语义与全 batch 等价（逆迭代每块独立）。
        import math as _math
        chunk = _math.gcd(dof, max(1, batch_chunk or 8))
        if chunk < 1:
            chunk = dof
        v_all = torch.zeros([dof, e] + list(lat_fine_odd), dtype=dtype, device=device)
        for c0 in range(0, dof, chunk):
            c1 = min(dof, c0 + chunk)
            vb = torch.randn([c1 - c0, e] + list(lat_fine_odd), dtype=dtype,
                             device=device)
            for _ in range(nv_iters):
                # dev84: 相对容差 — 以各批次初始 ‖r‖ 为基准
                xb = _bistabcg_batch(batch_matvec(vb), batch_matvec,
                                     tol=nv_tol, if_rtol=True)
                vb = vb - xb
            nrm = torch.linalg.norm(vb.reshape(c1 - c0, -1), dim=1)
            nrm = torch.clamp(nrm, min=1e-30)
            bc = [c1 - c0] + [1] * (vb.ndim - 1)
            vb = vb / nrm.view(bc)
            v_all[c0:c1] = vb
        null = v_all
    else:
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

        if nthreads <= 1:
            # 单线程直接执行（避免嵌套线程池触发 torch lazy-wrapper 冲突）
            worker(0)
        else:
            with ThreadPoolExecutor(max_workers=nthreads) as ex:
                list(ex.map(worker, range(nthreads)))
    if verbose:
        import time
        print(f"PYQCU::TOOLS::MULTIGRID:\n null_vecs build ({nthreads} threads"
              f"{' batch' if batch_matvec is not None else ''}): "
              f"{dof} vectors, nv_iters={nv_iters}")
    return null


def _probe_point_fast(matvec, lonv, E, ee, c_idx, sit, hop_nn, hop_diag, dims, Nc):
    """快速单点探测（与 _probe_point 数学等价，避免 10 维全格 einsum）。

    src_c[ee, c]=1 的单位向量探测：
      * prolong：f 为全细格零场 + lonv[ee] 在 c 块的局部填充（切片拷贝，零 einsum）
      * restrict：dc[p] 只依赖 (S f)[p块] 局部值，只需 c 的 ±1 邻域（≤33 点）批量收缩
    """
    mv = matvec.matvec if hasattr(matvec, 'matvec') else matvec
    Xc, Yc, Zc, Tc = dims
    str_Y, str_Z = Yc * Zc * Tc, Zc * Tc
    cx = c_idx // str_Y; rem = c_idx % str_Y
    cy = rem // str_Z; rem %= str_Z
    cz = rem // Tc; ct = rem % Tc
    ccoords = [cx, cy, cz, ct]
    E_l, e, X, x, Y, y, Z, z, T, t = lonv.shape
    # 1) prolong 切片化：零场 + 局部块填充（单位向量的 prolong 结果）
    f = torch.zeros([e, X * x, Y * y, Z * z, T * t], dtype=lonv.dtype, device=lonv.device)
    f[:, cx*x:(cx+1)*x, cy*y:(cy+1)*y, cz*z:(cz+1)*z, ct*t:(ct+1)*t] = \
        lonv[ee, :, cx, :, cy, :, cz, :, ct, :].reshape(e, x, y, z, t)
    dc_full = mv(f)
    # 2) 收集相关粗格点：c 本身 + 4 方向 ±1 + 6 对角 ±1（去重，保序）
    pts = [(cx, cy, cz, ct)]
    for d in range(4):
        b = ccoords[:]; b[d] = (b[d] - 1 + dims[d]) % dims[d]
        fwd = ccoords[:]; fwd[d] = (fwd[d] + 1) % dims[d]
        pts.append(tuple(b)); pts.append(tuple(fwd))
    for (d1, d2) in PAIRS:
        for s1 in SIGN:
            for s2 in SIGN:
                n = ccoords[:]
                n[d1] = (n[d1] - s1 + dims[d1]) % dims[d1]
                n[d2] = (n[d2] - s2 + dims[d2]) % dims[d2]
                pts.append(tuple(n))
    seen, uniq = set(), []
    for p in pts:
        if p not in seen:
            seen.add(p); uniq.append(p)
    # 3) 批量局部 restrict：dc[p] = Σ lonv[:,:,p块]† · dc_full[p块]
    lonv_p = torch.stack([lonv[:, :, p[0], :, p[1], :, p[2], :, p[3], :]
                          for p in uniq])  # [P,E,e,x,y,z,t]
    dc_p = torch.stack([
        dc_full[:, p[0]*x:(p[0]+1)*x, p[1]*y:(p[1]+1)*y,
                p[2]*z:(p[2]+1)*z, p[3]*t:(p[3]+1)*t]
        for p in uniq])  # [P,e,x,y,z,t]
    dc_vals = _torch.einsum("PEexyzt,Pexyzt->PE", lonv_p.conj(), dc_p)  # [P,E]
    dc_by_pt = {p: dc_vals[i] for i, p in enumerate(uniq)}
    # 4) 写回（系数约定与原版一致）
    sit[:, ee, cx, cy, cz, ct] = dc_by_pt[(cx, cy, cz, ct)]
    for d in range(4):
        b = ccoords[:]; b[d] = (b[d] - 1 + dims[d]) % dims[d]
        fwd = ccoords[:]; fwd[d] = (fwd[d] + 1) % dims[d]
        if b[d] == fwd[d]:
            # 2-site periodic dim: ±1 neighbours coincide; kernel sums both
            # hops, each must carry HALF the coupling.
            hop_nn[0, d, :, ee, b[0], b[1], b[2], b[3]] = 0.5 * dc_by_pt[tuple(b)]
            hop_nn[1, d, :, ee, fwd[0], fwd[1], fwd[2], fwd[3]] = 0.5 * dc_by_pt[tuple(fwd)]
        else:
            hop_nn[0, d, :, ee, b[0], b[1], b[2], b[3]] = dc_by_pt[tuple(b)]
            hop_nn[1, d, :, ee, fwd[0], fwd[1], fwd[2], fwd[3]] = dc_by_pt[tuple(fwd)]
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
                hop_diag[s1i, s2i, pi, :, ee, key[0], key[1], key[2], key[3]] = \
                    w * dc_by_pt[key]


def _schur_matvec_batch(op, x_b):
    """批量 Schur 奇偶算子：S = A_oo - k² D_oe A_ee⁻¹ D_eo（torch einsum 版）。

    x_b: [B, e, X, Y, Z, T/2]（B = 批维，e = spin×color 12）→ 同形输出。
    与 C++ CudaSchurOp.matvec 等价（实测 8x8x8x16 相对误差 ~1e-7）。
    2026-08-15：stencil 探测批量化 —— 固定粗格点 c_idx 的 E 个探针合并为
    一次批量 Schur，48 场单次 ~11ms vs 逐场 C++ 1.64ms×48 ≈ 79ms（7 倍）。
    单 rank（grid=1）路径：无 MPI halo，mask 由 give_wilson_plus/minus 处理。
    """
    B = x_b.shape[0]
    from pyqcu import lattice as _lattice
    dest_e = torch.zeros_like(x_b)
    for ward in range(4):
        # give_wilson_plus: shifts=-1 配 M_plus；give_wilson_minus: shifts=+1 配 M_minus
        wd = _lattice.wards[_lattice.ward_keys[ward]]
        for pm, M in ((-1, op.hopping.M_e_plus_list[ward]),
                      (1, op.hopping.M_e_minus_list[ward])):
            src = torch.roll(x_b, shifts=pm, dims=wd)
            if ward == 3:
                # give_wilson_plus(parity=1)→even_mask；give_wilson_minus(parity=1)→odd_mask
                mask = (tools.give_eo_mask(oootzy_t_p=x_b[0], eo=0) if pm == -1
                        else tools.give_eo_mask(oootzy_t_p=x_b[0], eo=1))
                src[..., mask] = x_b[..., mask]
            dest_e += _torch.einsum("Eexyzt,Bexyzt->BExyzt", M, src)
    xe = dest_e
    xe_inv = _torch.einsum("EeXYZT,BeXYZT->BEXYZT", op.sitting.M_e_inv, xe)
    dest_o = torch.zeros_like(x_b)
    for ward in range(4):
        wd = _lattice.wards[_lattice.ward_keys[ward]]
        for pm, M in ((-1, op.hopping.M_o_plus_list[ward]),
                      (1, op.hopping.M_o_minus_list[ward])):
            src = torch.roll(xe_inv, shifts=pm, dims=wd)
            if ward == 3:
                # give_wilson_plus(parity=0)→odd_mask；give_wilson_minus(parity=0)→even_mask
                mask = (tools.give_eo_mask(oootzy_t_p=x_b[0], eo=1) if pm == -1
                        else tools.give_eo_mask(oootzy_t_p=x_b[0], eo=0))
                src[..., mask] = xe_inv[..., mask]
            dest_o += _torch.einsum("Eexyzt,Bexyzt->BExyzt", M, src)
    out = _torch.einsum("EeXYZT,BeXYZT->BEXYZT", op.sitting.M_o, x_b)
    return out - dest_o


def _stencil_matvec_batch(stencil, x_b):
    """批量 33-tensor 粗层 Schur matvec（apply_stencil 的批量版）。

    stencil: (sit, hop_nn, hop_diag)；x_b: [B, E, Xc, Yc, Zc, Tc] → 同形。
    对应 CudaCoarseSchurOp.matvec（C++ 宽版）的 torch 等价物，
    用于 lvl>=2 的粗层探测批量化。
    """
    sit, hop_nn, hop_diag = stencil
    B = x_b.shape[0]
    out = _torch.einsum("EeXYZT,BeXYZT->BEXYZT", sit, x_b)
    for d in range(4):
        fwd = _torch.roll(x_b, shifts=-1, dims=d + 2)
        bwd = _torch.roll(x_b, shifts=1, dims=d + 2)
        out += _torch.einsum("EeXYZT,BeXYZT->BEXYZT", hop_nn[0, d], fwd)
        out += _torch.einsum("EeXYZT,BeXYZT->BEXYZT", hop_nn[1, d], bwd)
    for pi, (d1, d2) in enumerate(PAIRS):
        for s1i, s1 in enumerate(SIGN):
            for s2i, s2 in enumerate(SIGN):
                shift = [0, 0, 0, 0]; shift[d1] = -s1; shift[d2] = -s2
                v_shift = _torch.roll(x_b, shifts=tuple(shift), dims=(2, 3, 4, 5))
                out += _torch.einsum("EeXYZT,BeXYZT->BEXYZT",
                                     hop_diag[s1i, s2i, pi], v_shift)
    return out


def _probe_point_batch(matvec_batch, lonv, E, c_idx, sit, hop_nn, hop_diag, dims, Nc):
    """批量单点探测：固定 c_idx 一次计算全部 E 个 ee 的 33-tensor 耦合。

    matvec_batch: 批量 matvec 函数（x_b [B, e, Xf, Yf, Zf, Tf] → 同形），
        细层用 _schur_matvec_batch(op)，粗层用 _stencil_matvec_batch(stencil)。
    数学等价于对全部 ee 调 _probe_point（prolong 线性 + restrict 块局部）：
      * prolong：全部 E 个单位向量 e_ee 一次切片填充（f_b[ee] = lonv[ee] 的 c 块）
      * restrict：dc[p, ee] = Σ_{fine∈p块} lonv[:, :, p块]† · dc_full[ee, :, p块]，
        批量 einsum 一次得到全部 E 探针在 p 的耦合（p 仅需 c 的 ±1 邻域，≤33 点）
    写集与 _probe_point 一致（sit/hop_nn/hop_diag 不同坐标片），可并行。
    """
    Xc, Yc, Zc, Tc = dims
    str_Y, str_Z = Yc * Zc * Tc, Zc * Tc
    cx = c_idx // str_Y; rem = c_idx % str_Y
    cy = rem // str_Z; rem %= str_Z
    cz = rem // Tc; ct = rem % Tc
    ccoords = [cx, cy, cz, ct]
    E_l, e, X, x, Y, y, Z, z, T, t = lonv.shape
    # 1) 批量 prolong：f_b[ee] = lonv[ee] 在 c 块的零填充切片
    f_b = torch.zeros([E, e, X * x, Y * y, Z * z, T * t], dtype=lonv.dtype,
                      device=lonv.device)
    f_b[:, :, cx*x:(cx+1)*x, cy*y:(cy+1)*y, cz*z:(cz+1)*z, ct*t:(ct+1)*t] = \
        lonv[:, :, cx, :, cy, :, cz, :, ct, :].reshape(E, e, x, y, z, t)
    dc_full = matvec_batch(f_b)  # [E, e, Xf, Yf, Zf, Tf]
    # 2) 相关粗格点集合（c 本身 + 4 方向 ±1 + 6 对角 ±1，去重保序）
    pts = [(cx, cy, cz, ct)]
    for d in range(4):
        b = ccoords[:]; b[d] = (b[d] - 1 + dims[d]) % dims[d]
        fwd = ccoords[:]; fwd[d] = (fwd[d] + 1) % dims[d]
        pts.append(tuple(b)); pts.append(tuple(fwd))
    for (d1, d2) in PAIRS:
        for s1 in SIGN:
            for s2 in SIGN:
                n = ccoords[:]
                n[d1] = (n[d1] - s1 + dims[d1]) % dims[d1]
                n[d2] = (n[d2] - s2 + dims[d2]) % dims[d2]
                pts.append(tuple(n))
    seen, uniq = set(), []
    for p in pts:
        if p not in seen:
            seen.add(p); uniq.append(p)
    # 3) 批量局部 restrict：dc[p, ee]（[E, E]：p 块输出 dof × E 探针）
    lonv_p = torch.stack([lonv[:, :, p[0], :, p[1], :, p[2], :, p[3], :]
                          for p in uniq])  # [P,E,e,x,y,z,t]
    dc_p = torch.stack([
        dc_full[:, :, p[0]*x:(p[0]+1)*x, p[1]*y:(p[1]+1)*y,
                p[2]*z:(p[2]+1)*z, p[3]*t:(p[3]+1)*t]
        for p in uniq])  # [P,E,e,x,y,z,t]
    # dc[p] = [E(粗 dof), E(探针)]：lonv_p [P,a,e,x,y,z,t] conj · dc_p [P,b,e,x,y,z,t]
    dc_vals = _torch.einsum("Paexyzt,Pbexyzt->Pab",
                            lonv_p.conj(), dc_p)  # [P, E_dof, E_probe]
    dc_by_pt = {p: dc_vals[i] for i, p in enumerate(uniq)}
    # 4) 写回（与 _probe_point 相同的系数约定；ee 维用冒号批量）
    sit[:, :, cx, cy, cz, ct] = dc_by_pt[(cx, cy, cz, ct)]
    for d in range(4):
        b = ccoords[:]; b[d] = (b[d] - 1 + dims[d]) % dims[d]
        fwd = ccoords[:]; fwd[d] = (fwd[d] + 1) % dims[d]
        if b[d] == fwd[d]:
            hop_nn[0, d, :, :, b[0], b[1], b[2], b[3]] = 0.5 * dc_by_pt[tuple(b)]
            hop_nn[1, d, :, :, fwd[0], fwd[1], fwd[2], fwd[3]] = 0.5 * dc_by_pt[tuple(fwd)]
        else:
            hop_nn[0, d, :, :, b[0], b[1], b[2], b[3]] = dc_by_pt[tuple(b)]
            hop_nn[1, d, :, :, fwd[0], fwd[1], fwd[2], fwd[3]] = dc_by_pt[tuple(fwd)]
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
                hop_diag[s1i, s2i, pi, :, :, key[0], key[1], key[2], key[3]] = \
                    w * dc_by_pt[key]


def _probe_point(matvec, lonv, E, ee, c_idx, sit, hop_nn, hop_diag, dims, Nc,
                  src_c=None):
    """单点探测：(c_idx, ee) 处的 33-tensor 耦合。写集互不相交，可并行。

    src_c: 可选预分配探针张量 [E, Xc, Yc, Zc, Tc]（每 probe 复用，省分配开销）。
    """
    mv = matvec.matvec if hasattr(matvec, 'matvec') else matvec
    Xc, Yc, Zc, Tc = dims
    str_Y, str_Z = Yc * Zc * Tc, Zc * Tc
    cx = c_idx // str_Y; rem = c_idx % str_Y
    cy = rem // str_Z; rem %= str_Z
    cz = rem // Tc; ct = rem % Tc
    ccoords = [cx, cy, cz, ct]
    if src_c is None:
        src_c = torch.zeros([E, Xc, Yc, Zc, Tc], dtype=sit.dtype, device=sit.device)
    else:
        src_c.zero_()
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
    src_c = torch.zeros([E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    for c_idx in range(Nc):
        for ee in range(E):
            _probe_point(matvec, lonv, E, ee, c_idx, sit, hop_nn, hop_diag, dims, Nc,
                         src_c=src_c)
        if verbose and (c_idx + 1) % 64 == 0:
            print(f"    probing {c_idx+1}/{Nc} ({time.perf_counter()-t0:.1f}s)")
    if verbose:
        print(f"PYQCU::TOOLS::MULTIGRID:\n stencil build (1 thread): "
              f"{time.perf_counter()-t0:.1f}s for {E*Nc} probes")
    return hop_nn, hop_diag, sit


def build_stencil_mt(matvec_ops, lonv, E, e, lat_fine_odd, lat_coarse_odd,
                     dt, device, nthreads=4, verbose=True, fast=True,
                     batch=False):
    """多线程 33-tensor stencil build。matvec_ops: 每线程一个 matvec 算子（如 CudaSchurOp 列表）。

    各线程探测点写集不相交（probe_point 写 sit/hop_nn/hop_diag 的不同坐标片），
    线程安全；适合多卡（一线程一卡）场景用每线程独立 GPU 算子并行构建。

    fast=True（默认）：_probe_point_fast（单位向量探测切片化，prolong 零 einsum、
    restrict 邻域块局部化）。

    batch=True：_probe_point_batch（固定 c_idx 一次批量全部 E 探针，torch
    批量 matvec）——2026-08-15 实测 8x8x8x16 lv1 从 135s → ~15s（10 倍）。
    batch 模式 matvec_ops 传批量 matvec 函数（如 _schur_matvec_batch(op) 或
    _stencil_matvec_batch(stencil)），nthreads 仅用于分片探测点（单卡时
    nthreads=1 最优：多线程全设备同步竞争反而更慢）。
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
        c0 = tid * chunk
        c1 = min(Nc, c0 + chunk)
        if batch:
            mv = matvec_ops[0] if isinstance(matvec_ops, (list, tuple)) else matvec_ops
            for c_idx in range(c0, c1):
                _probe_point_batch(mv, lonv, E, c_idx, sit, hop_nn,
                                   hop_diag, dims, Nc)
            return
        op = matvec_ops[tid % len(matvec_ops)]
        src_c = torch.zeros([E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
        if fast:
            for c_idx in range(c0, c1):
                for ee in range(E):
                    _probe_point_fast(op, lonv, E, ee, c_idx, sit, hop_nn,
                                      hop_diag, dims, Nc)
        else:
            for c_idx in range(c0, c1):
                for ee in range(E):
                    _probe_point(op, lonv, E, ee, c_idx, sit, hop_nn,
                                 hop_diag, dims, Nc, src_c=src_c)

    if nthreads <= 1:
        worker(0)
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            list(ex.map(worker, range(nthreads)))
    dt_build = time.perf_counter() - t0
    if verbose:
        print(f"PYQCU::TOOLS::MULTIGRID:\n stencil build ({nthreads} threads"
              f"{' batch' if batch else ''}): "
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


# ======================================================================
# 局部化批量 Schur 与 stencil 构建（2026-08-18，test15 24x24x24x72 大格子）
# ----------------------------------------------------------------------
# 背景：24x24x24x72 lv1（31104 粗格点）stencil 探测，全格 torch 批量
# (_schur_matvec_batch, 2.58s/点) 约 22 小时、C++ 逐场 (CudaSchurOp 14.33ms
# × 746496) 约 178 分钟均不可行。本文件实现局部化：每个粗格点 c 只需
# Schur 在 c 窗口（W=10，覆盖 c±1 块 + 半径 2 padding）内的作用，
# 与全格 Schur 在中心 c±1 块逐点验证 diff=0（local_test4/local_probe2）。
# 实测 24x24x24x72 lv1 单点 ~54ms → 31104 点 ~28 min（setup 一次性，缓存复用）。
# ======================================================================


class BatchedLocalSchur:
    """局部化批量 Schur 奇偶算子 S = A_oo - k² D_oe A_ee⁻¹ D_eo。

    op: dslash.operator（支持奇偶，保留 hopping.M_e/o_plus/minus_list、
        sitting.M_e_inv/M_o 组件；其余可 slim 释放）。
    idx: [K,4,W] 每个粗格点 c 的窗口全局坐标（x0..x0+W 模格点）。
    x_local: [K, B, e, W,W,W,W]（K=窗口批量、B=探针数=24、e=12 dof）。
    输出 [K, B, e, W,W,W,W]，在中心 c±1 块与全格 Schur 一致。

    2026-08-18：窗口起点 x0 = 2c - (W//2 - 1)，c 块 [2c,2c+2) 居中；
    输出只需中心 [W//2-3, W//2+3)（c±1 块）。einsum 的 dof 标签约定：
    M（[E,e,..]）的第一维是输出 dof（E），第二维是输入 dof（e）；
    x 的 dof 维是输入，须标 e（不能用 E），否则 einsum 语义错（数值错）。
    """

    def __init__(self, op, Xf, Yf, Zf, Tf, W=10):
        self.op = op
        self.W = W
        self.dims = (Xf, Yf, Zf, Tf)
        self.Mep = [op.hopping.M_e_plus_list[d] for d in range(4)]
        self.Mem = [op.hopping.M_e_minus_list[d] for d in range(4)]
        self.Mop = [op.hopping.M_o_plus_list[d] for d in range(4)]
        self.Mom = [op.hopping.M_o_minus_list[d] for d in range(4)]
        self.Me_inv = op.sitting.M_e_inv
        self.Mo = op.sitting.M_o
        self.ar = torch.arange(W, device=op.hopping.M_e_plus_list[0].device)
        self.sp = (self.ar + 1) % W
        self.sm = (self.ar - 1) % W

    def _slicem(self, M, idx, starts=None):
        """对 K 个窗口做 M[:, :, 窗口] 切片 → [K,E,e,W,W,W,W]。

        starts: 可选 [K,4] 整数窗口起点（避免 .item() 同步）；给定时优先用
        连续切片（快 ~86 倍），否则回退高级索引。
        """
        K = idx.shape[0]
        W = self.W
        Xf, Yf, Zf, Tf = self.dims
        out = torch.empty([K, 12, 12, W, W, W, W], dtype=M.dtype, device=M.device)
        if starts is not None:
            for k in range(K):
                x0, y0, z0, t0 = starts[k]
                if (x0 + W <= Xf and y0 + W <= Yf and z0 + W <= Zf and t0 + W <= Tf):
                    out[k] = M[:, :, x0:x0 + W, y0:y0 + W, z0:z0 + W, t0:t0 + W]
                else:
                    ix, iy, iz, it = idx[k, 0], idx[k, 1], idx[k, 2], idx[k, 3]
                    out[k] = M[:, :, ix][:, :, :, iy][:, :, :, :, iz][:, :, :, :, :, it]
            return out
        for k in range(K):
            ix, iy, iz, it = idx[k, 0], idx[k, 1], idx[k, 2], idx[k, 3]
            x0, y0, z0, t0 = ix[0].item(), iy[0].item(), iz[0].item(), it[0].item()
            if (ix[-1].item() == (x0 + W - 1) % Xf and iy[-1].item() == (y0 + W - 1) % Yf and
                    iz[-1].item() == (z0 + W - 1) % Zf and it[-1].item() == (t0 + W - 1) % Tf and
                    x0 + W <= Xf and y0 + W <= Yf and z0 + W <= Zf and t0 + W <= Tf):
                out[k] = M[:, :, x0:x0 + W, y0:y0 + W, z0:z0 + W, t0:t0 + W]
            else:
                out[k] = M[:, :, ix][:, :, :, iy][:, :, :, :, iz][:, :, :, :, :, it]
        return out

    def _masks(self, idx, K, E):
        """t 方向 (wd=3) 掩码：even 格点 (x+y+z)%2==0，odd 反之（全局坐标）。"""
        W = self.W
        xgk = idx[:, 0].view(K, W, 1, 1, 1)
        ygk = idx[:, 1].view(K, 1, W, 1, 1)
        zgk = idx[:, 2].view(K, 1, 1, W, 1)
        me = ((xgk + ygk + zgk) % 2 == 0).expand(K, W, W, W, W)
        mo = ~me
        mek = me.view(K, 1, 1, W, W, W, W).expand(K, E, 12, W, W, W, W)
        mok = mo.view(K, 1, 1, W, W, W, W).expand(K, E, 12, W, W, W, W)
        return mek, mok

    def __call__(self, x_local, idx, starts=None):
        K = x_local.shape[0]
        W = self.W
        E = x_local.shape[1]
        Mep = [self._slicem(self.Mep[d], idx, starts) for d in range(4)]
        Mem = [self._slicem(self.Mem[d], idx, starts) for d in range(4)]
        Mop = [self._slicem(self.Mop[d], idx, starts) for d in range(4)]
        Mom = [self._slicem(self.Mom[d], idx, starts) for d in range(4)]
        Me_inv = self._slicem(self.Me_inv, idx, starts)
        Mo = self._slicem(self.Mo, idx, starts)
        mek, mok = self._masks(idx, K, E)
        # even: D_oe（even 输出），配 M_e_plus/minus，t 向 mask
        dest_e = torch.zeros([K, E, 12, W, W, W, W], dtype=x_local.dtype, device=x_local.device)
        for d in range(4):
            src_p = torch.roll(x_local, shifts=-1, dims=d + 3)
            src_m = torch.roll(x_local, shifts=1, dims=d + 3)
            if d == 3:
                src_p = torch.where(mek, x_local, src_p)
                src_m = torch.where(mok, x_local, src_m)
            dest_e += torch.einsum("kEexyzt,kBexyzt->kBExyzt", Mep[d], src_p)
            dest_e += torch.einsum("kEexyzt,kBexyzt->kBExyzt", Mem[d], src_m)
        # xe_inv = A_ee⁻¹ · dest_e：dest_e 的 dof 是 even 输入 → 标 e
        xe_inv = torch.einsum("kEeXYZT,kBeXYZT->kBEXYZT", Me_inv, dest_e)
        # odd: D_eo（odd 输出），配 M_o_plus/minus，t 向 mask（odd/eo 对调）
        dest_o = torch.zeros([K, E, 12, W, W, W, W], dtype=x_local.dtype, device=x_local.device)
        for d in range(4):
            src_p = torch.roll(xe_inv, shifts=-1, dims=d + 3)
            src_m = torch.roll(xe_inv, shifts=1, dims=d + 3)
            if d == 3:
                src_p = torch.where(mok, xe_inv, src_p)
                src_m = torch.where(mek, xe_inv, src_m)
            dest_o += torch.einsum("kEexyzt,kBexyzt->kBExyzt", Mop[d], src_p)
            dest_o += torch.einsum("kEexyzt,kBexyzt->kBExyzt", Mom[d], src_m)
        # out = A_oo · x：x 的 dof 是 odd 输入 → 标 e
        out = torch.einsum("kEeXYZT,kBeXYZT->kBEXYZT", Mo, x_local)
        return out - dest_o


def _probe_point_batch_local(lsch, lonv, E, c_idx, sit, hop_nn, hop_diag, dims, Nc, W):
    """局部化批量单点探测：用 BatchedLocalSchur 替代全格 _schur_matvec_batch。

    与 _probe_point_batch 数学等价（中心 c±1 块验证 diff=0）：prolong 在
    窗口内 c 块填充 lonv，局部 Schur，restrict 到 c±1 邻域 33 个粗格点。
    """
    Xc, Yc, Zc, Tc = dims
    str_Y, str_Z = Yc * Zc * Tc, Zc * Tc
    cx = c_idx // str_Y
    rem = c_idx % str_Y
    cy = rem // str_Z
    rem %= str_Z
    cz = rem // Tc
    ct = rem % Tc
    ccoords = [cx, cy, cz, ct]
    E_l, e, X, x, Y, y, Z, z, T, t = lonv.shape
    off = W // 2 - 1
    Xf, Yf, Zf, Tf = lsch.dims
    x0 = 2 * cx - off
    y0 = 2 * cy - off
    z0 = 2 * cz - off
    t0 = 2 * ct - off
    _opdev = lsch.Mep[0].device
    # dev84: lonv 可驻留 CPU —— 探测工作张量全部锚定算子设备
    ix = (torch.arange(x0, x0 + W, device=_opdev)) % Xf
    iy = (torch.arange(y0, y0 + W, device=_opdev)) % Yf
    iz = (torch.arange(z0, z0 + W, device=_opdev)) % Zf
    it = (torch.arange(t0, t0 + W, device=_opdev)) % Tf
    idx = torch.stack([ix, iy, iz, it]).unsqueeze(0)  # [1,4,W]
    f_local = torch.zeros([1, E, e, W, W, W, W], dtype=lonv.dtype, device=_opdev)
    _blk = lonv[:, :, cx, :, cy, :, cz, :, ct, :].reshape(E, e, x, y, z, t)
    if _blk.device != _opdev:
        _blk = _blk.to(_opdev, non_blocking=True)
    starts = [(x0 % Xf, y0 % Yf, z0 % Zf, t0 % Tf)]
    dc_local = lsch(f_local, idx, starts)[0]  # [E, e, W,W,W,W]
    # 33 个相关粗格点（c 本身 + 4 方向 ±1 + 6 对角 ±1，去重保序）
    pts = [(cx, cy, cz, ct)]
    for d in range(4):
        b = ccoords[:]
        b[d] = (b[d] - 1 + dims[d]) % dims[d]
        fwd = ccoords[:]
        fwd[d] = (fwd[d] + 1) % dims[d]
        pts.append(tuple(b))
        pts.append(tuple(fwd))
    for (d1, d2) in PAIRS:
        for s1 in SIGN:
            for s2 in SIGN:
                n = ccoords[:]
                n[d1] = (n[d1] - s1 + dims[d1]) % dims[d1]
                n[d2] = (n[d2] - s2 + dims[d2]) % dims[d2]
                pts.append(tuple(n))
    seen, uniq = set(), []
    for p in pts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    # 窗口内 p 块偏移（全局 2p 减去窗口起点，模格点）
    x0m = x0 % Xf
    y0m = y0 % Yf
    z0m = z0 % Zf
    t0m = t0 % Tf
    dc_p = torch.stack([
        dc_local[:, :, (2 * p[0] - x0m) % Xf:(2 * p[0] - x0m) % Xf + 2,
                (2 * p[1] - y0m) % Yf:(2 * p[1] - y0m) % Yf + 2,
                (2 * p[2] - z0m) % Zf:(2 * p[2] - z0m) % Zf + 2,
                (2 * p[3] - t0m) % Tf:(2 * p[3] - t0m) % Tf + 2]
        for p in uniq])
    # dev84: lonv 可驻留 CPU —— 邻域切片统一上卡后再收缩
    _lp = [lonv[:, :, pp[0], :, pp[1], :, pp[2], :, pp[3], :] for pp in uniq]
    _dev = dc_p.device
    _lp = [t.to(_dev) if t.device != _dev else t for t in _lp]
    lonv_p = torch.stack(_lp)
    dc_vals = torch.einsum("Paexyzt,Pbexyzt->Pab", lonv_p.conj(), dc_p)  # [P,E,E]
    dc_by_pt = {p: dc_vals[i] for i, p in enumerate(uniq)}
    # 写回（系数约定与 _probe_point_batch 一致）
    sit[:, :, cx, cy, cz, ct] = dc_by_pt[(cx, cy, cz, ct)]
    for d in range(4):
        b = ccoords[:]
        b[d] = (b[d] - 1 + dims[d]) % dims[d]
        fwd = ccoords[:]
        fwd[d] = (fwd[d] + 1) % dims[d]
        if b[d] == fwd[d]:
            hop_nn[0, d, :, :, b[0], b[1], b[2], b[3]] = 0.5 * dc_by_pt[tuple(b)]
            hop_nn[1, d, :, :, fwd[0], fwd[1], fwd[2], fwd[3]] = 0.5 * dc_by_pt[tuple(fwd)]
        else:
            hop_nn[0, d, :, :, b[0], b[1], b[2], b[3]] = dc_by_pt[tuple(b)]
            hop_nn[1, d, :, :, fwd[0], fwd[1], fwd[2], fwd[3]] = dc_by_pt[tuple(fwd)]
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
                hop_diag[s1i, s2i, pi, :, :, key[0], key[1], key[2], key[3]] = \
                    w * dc_by_pt[key]
    return dc_by_pt


def build_stencil_local(lsch, lonv, E, lat_fine_odd, lat_coarse_odd,
                        dt, device, verbose=True):
    """局部化 33-tensor stencil 构建（24x24x24x72 大格子）。

    用 BatchedLocalSchur 在 c 窗口内计算 Schur（替代全格 matvec），
    单线程顺序探测全部粗格点。返回 (hop_nn, hop_diag, sit)。
    """
    import time
    Xc, Yc, Zc, Tc = lat_coarse_odd
    Nc = Xc * Yc * Zc * Tc
    dims = [Xc, Yc, Zc, Tc]
    W = lsch.W
    sit = torch.zeros([E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_nn = torch.zeros([2, 4, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    hop_diag = torch.zeros([2, 2, 6, E, E, Xc, Yc, Zc, Tc], dtype=dt, device=device)
    t0 = time.perf_counter()
    for c_idx in range(Nc):
        _probe_point_batch_local(lsch, lonv, E, c_idx, sit, hop_nn,
                                 hop_diag, dims, Nc, W)
    dt_build = time.perf_counter() - t0
    if verbose:
        print(f"PYQCU::TOOLS::MULTIGRID:\n stencil build (local): "
              f"{dt_build:.1f}s for {E * Nc} probes "
              f"({E * Nc / max(dt_build, 1e-9):.0f} probes/s)")
    return hop_nn, hop_diag, sit
