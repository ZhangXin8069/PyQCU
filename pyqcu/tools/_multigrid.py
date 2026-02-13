import torch
from typing import Callable, Tuple
from pyqcu import _torch, solver, tools
disable_patch_npu = False


def give_null_vecs(
    null_vecs: torch.Tensor,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    normalize: bool = True, ortho_r: bool = False, ortho_null_vecs: bool = False, verbose: bool = True
) -> torch.Tensor:
    dof = null_vecs.shape[0]
    null_vecs = _torch.randn_like(null_vecs)  # [Eexyzt]
    for i in range(dof):
        if ortho_r:
            # The orthogonalization of r
            for j in range(0, i):
                null_vecs[i] -= tools.vdot(null_vecs[j], null_vecs[i])/tools.vdot(
                    null_vecs[j], null_vecs[j])*null_vecs[j]
        # v=r-A^{-1}Ar
        # tol needs to be bigger...
        null_vecs[i] -= solver.bistabcg(b=matvec(null_vecs[i]),
                                        matvec=matvec, tol=5e-5, verbose=verbose)
        if ortho_null_vecs:
            # The orthogonalization of null_vecs
            for j in range(0, i):
                null_vecs[i] -= tools.vdot(null_vecs[j], null_vecs[i])/tools.vdot(
                    null_vecs[j], null_vecs[j])*null_vecs[j]
        if normalize:
            null_vecs[i] /= tools.norm(null_vecs[i])
        if verbose:
            print(
                f"PYQCU::TOOLS::MATRIX:\n (_matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]:{(matvec(null_vecs[i])/null_vecs[i]).flatten()[:10]}")
    if verbose:
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
    return null_vecs.clone()


def local_orthogonalize(null_vecs: torch.Tensor,
                        coarse_lat_size: Tuple[int, int,
                                               int, int] = (2, 2, 2, 2),
                        normalize: bool = True,
                        verbose: bool = False) -> torch.Tensor:
    if null_vecs.device.type == 'npu' or disable_patch_npu:
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
    # Batched QR on each block; Q has orthonormal columns in R^{local_dim}
    # Use reduced mode: Q: [n_blocks,local_dim,E],R: [n_blocks,E,E]
    Q, _ = _torch.linalg_qr(A, mode='reduced')
    if normalize:
        # Normalize each column vector explicitly
        Q = Q / _torch.norm(Q, dim=-2, keepdim=True)
    # Restore lattice structure: [X,Y,Z,T,e,x,y,z,t,E]
    Q = Q.view(X, Y, Z, T, e, x, y, z, t, E)
    # Permute back to [E,e,X,x,Y,y,Z,z,T,t]
    Q = Q.permute(9, 4, 0, 5, 1, 6, 2, 7, 3, 8).contiguous().clone()
    if verbose:
        print(f"PYQCU::TOOLS::MATRIX:\n [local_orthogonalize] in={tuple(null_vecs.shape)},coarse_lat_size(X,Y,Z,T)={coarse_lat_size},"
              f"PYQCU::TOOLS::MATRIX:\n (x,y,z,t)=({x},{y},{z},{t}),local_dim={local_dim},n_blocks={n_blocks}")
    return Q.clone()


def restrict(local_ortho_null_vecs: torch.Tensor, fine_vec: torch.Tensor) -> torch.Tensor:
    dtype = fine_vec.dtype
    device = fine_vec.device
    if device.type == 'npu' or disable_patch_npu:
        return restrict_npu(local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=fine_vec)
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        fine_vec = fine_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _fine_vec = fine_vec.reshape(shape=shape[1:]).clone()
    return _torch.einsum(
        "EeXxYyZzTt,eXxYyZzTt->EXYZT", local_ortho_null_vecs.conj(), _fine_vec).clone().to(dtype=dtype, device=device)


def prolong(local_ortho_null_vecs: torch.Tensor, coarse_vec: torch.Tensor) -> torch.Tensor:
    dtype = coarse_vec.dtype
    device = coarse_vec.device
    if device.type == 'npu' or disable_patch_npu:
        return prolong_npu(local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=coarse_vec)
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        coarse_vec = coarse_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _coarse_vec = coarse_vec.reshape(shape=shape[0:1]+shape[-8:][::2]).clone()
    return _torch.einsum(
        "EeXxYyZzTt,EXYZT->eXxYyZzTt", local_ortho_null_vecs, _coarse_vec).reshape([shape[1], shape[-8]*shape[-7], shape[-6]*shape[-5], shape[-4]*shape[-3], shape[-2]*shape[-1]]).clone().to(dtype=dtype, device=device)
# NPU:The self tensor cannot be larger than 8 dimensions.


def local_orthogonalize_npu(null_vecs: torch.Tensor,
                            coarse_lat_size: Tuple[int, int,
                                                   int, int] = (2, 2, 2, 2),
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
    # Batched QR on each block; Q has orthonormal columns in R^{local_dim}
    # Use reduced mode: Q: [n_blocks,local_dim,E],R: [n_blocks,E,E]
    Q, _ = _torch.linalg_qr(A, mode='reduced')
    if normalize:
        # Normalize each column vector explicitly
        Q = Q / _torch.norm(Q, dim=-2, keepdim=True)
    """
    # Restore lattice structure: [X,Y,Z,T,e,x,y,z,t,E]
    Q = Q.view(X,Y,Z,T,e,x,y,z,t,E)
    # Permute back to [E,e,X,x,Y,y,Z,z,T,t]
    Q = Q.permute(9,4,0,5,1,6,2,7,3,8).contiguous().clone()
    """
    Q = Q.reshape(X, Y*Z*T, e, x, y*z*t, E)
    # [E,e,X,Y*Z*T,x,y*z*t]
    Q = Q.permute(5, 2, 0, 1, 3, 4).contiguous().clone()
    Q = Q.reshape(-1, Y, Z, T, x, y, z, t)
    # [EeX,x,Y,y,Z,z,T,t]
    Q = Q.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().clone()
    Q = Q.reshape(E, e, X, x, Y, y, Z, z, T, t)
    if verbose:
        print(f"[local_orthogonalize] in={tuple(null_vecs.shape)},coarse_lat_size(X,Y,Z,T)={coarse_lat_size},"
              f"(x,y,z,t)=({x},{y},{z},{t}),local_dim={local_dim},n_blocks={n_blocks}")
    return Q


def restrict_npu(local_ortho_null_vecs: torch.Tensor, fine_vec: torch.Tensor) -> torch.Tensor:
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
        "EeXxYyZzTt,eXxYyZzTt->EXYZT",local_ortho_null_vecs.conj(),_fine_vec).clone().to(dtype=dtype,device=device)
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
        "EeOxyzt,eOxyzt->EO", _local_ortho_null_vecs.conj(), _fine_vec).reshape(E, x, y, z, t).clone().to(dtype=dtype, device=device)


def prolong_npu(local_ortho_null_vecs: torch.Tensor, coarse_vec: torch.Tensor) -> torch.Tensor:
    dtype = coarse_vec.dtype
    device = coarse_vec.device
    _dtype = local_ortho_null_vecs.dtype
    _device = local_ortho_null_vecs.device
    if dtype != _dtype or device != _device:
        coarse_vec = coarse_vec.to(dtype=_dtype, device=_device)
    shape = local_ortho_null_vecs.shape
    _coarse_vec = coarse_vec.reshape(shape=shape[0:1]+shape[-8:][::2]).clone()
    """
    return _torch.einsum(
        "EeXxYyZzTt,EXYZT->eXxYyZzTt",local_ortho_null_vecs,_coarse_vec).reshape([shape[1],shape[-8]*shape[-7],shape[-6]*shape[-5],shape[-4]*shape[-3],shape[-2]*shape[-1]]).clone().to(dtype=dtype,device=device)
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
