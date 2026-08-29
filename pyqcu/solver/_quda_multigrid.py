"""QUDA 风格的聚合多重网格参考实现。

本模块对应 QUDA ``Transfer``、``DiracCoarse`` 和 ``MG`` 的算法骨架，
使用 PyTorch 张量实现，定位是 CPU/CUDA 上可验证的参考路径。它与旧的
``solver.multigrid`` 并列存在：旧类保留原有的扁平 ``E`` 自由度/33 点实现，
本模块则显式保留

* fine-spin -> coarse-spin 的映射（Wilson/Clover: 4 -> 2）；
* 每个 aggregate、每个 coarse-spin block 的重复 CGS 正交化；
* ``P``、``R=P^dagger`` 和 ``D_c=R D_f P``；
* 局部 ``X``、正反向 ``Y``、``X^{-1}``、``Yhat``；
* 每个层级的 even/odd Schur complement；
* MR 平滑、递归 V-cycle 和 flexible GMRES 外迭代。

张量的规范布局为 ``[dof, X, Y, Z, T]``，最后四轴始终是时空轴。为了
兼容 PyQCU 现有的 ``[spin, color, ...]`` 输入，公开的求解器也接受并返回
``[spin, color, X, Y, Z, T]``。粗算子构造默认可关闭，因为对大格子逐列
探测 ``RDP`` 的代价是 O(粗格点数 × 粗自由度)；关闭时 ``apply`` 仍严格
执行矩阵自由的 ``R(D(P(x)))``。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pyqcu.cann as _torch
from pyqcu import dslash, tools


Tensor = Any
Coord = Tuple[int, int, int, int]
Shape4 = Tuple[int, int, int, int]
BlockKey = Tuple[int, int, int, int]


def _shape4(shape: Sequence[int]) -> Shape4:
    if len(shape) != 4:
        raise ValueError(f"需要四维时空尺寸，得到 {tuple(shape)}")
    result = tuple(int(x) for x in shape)
    if any(x <= 0 for x in result):
        raise ValueError(f"时空尺寸必须为正数，得到 {result}")
    return result  # type: ignore[return-value]


def _site_index(coord: Coord, shape: Shape4) -> int:
    index = 0
    for value, extent in zip(coord, shape):
        index = index * extent + value
    return index


def _all_coords(shape: Shape4) -> Iterable[Coord]:
    return product(*(range(extent) for extent in shape))  # type: ignore[return-value]


def _zero_key(ndim: int = 4) -> BlockKey:
    return (0,) * ndim  # type: ignore[return-value]


def _signed_displacement(value: int, extent: int) -> int:
    """把模 ``extent`` 的位移折叠为可读的 signed displacement。"""
    if extent == 1:
        return 0
    value %= extent
    if value > extent // 2:
        value -= extent
    return value


def _roll_field(field: Tensor, displacement: BlockKey) -> Tensor:
    """返回 ``out[c] = field[c + displacement]`` 的周期平移。"""
    result = field
    for dim, delta in enumerate(displacement):
        if delta:
            result = _torch.roll(result, shifts=-delta, dims=1 + dim)
    return result


def _roll_site_tensor(field: Tensor, shift: BlockKey) -> Tensor:
    """对 ``[*, X, Y, Z, T]`` 张量按输出索引 ``c <- c-shift`` 平移。"""
    result = field
    for dim, delta in enumerate(shift):
        if delta:
            result = _torch.roll(result, shifts=delta, dims=2 + dim)
    return result


def _matvec_block(matrix: Tensor, vector: Tensor) -> Tensor:
    """批量局部矩阵作用，矩阵布局 ``[out, in, X, Y, Z, T]``。"""
    return _torch.einsum("ijxyzt,jxyzt->ixyzt", matrix, vector)


def _matmul_site(left: Tensor, right: Tensor) -> Tensor:
    """批量局部矩阵乘法，矩阵布局均为 ``[row, col, X, Y, Z, T]``。"""
    return _torch.einsum("ijxyzt,jkxyzt->ikxyzt", left, right)


def _adjoint_site(matrix: Tensor) -> Tensor:
    return matrix.conj().transpose(0, 1)


def _sum_block_axes(value: Tensor) -> Tensor:
    """对 ``[Cx,bx,Cy,by,Cz,bz,Ct,bt]`` 的 block 内轴求和。"""
    result = value
    for dim in (7, 5, 3, 1):
        result = result.sum(dim=dim)
    return result


def _relative_norm(numerator: Tensor, denominator: Tensor, floor: float = 1e-30) -> float:
    num = float(_torch.norm(numerator).item())
    den = float(_torch.norm(denominator).item())
    return num / max(den, floor)


def _call_matvec(operator: Any, value: Tensor) -> Tensor:
    if hasattr(operator, "apply"):
        return operator.apply(value)
    if hasattr(operator, "matvec"):
        return operator.matvec(value)
    return operator(value)


def _normalise_null_shape(null_vectors: Tensor, fine_spin: int,
                          fine_color: int, fine_shape: Shape4) -> Tensor:
    """统一为 ``[Nvec, fine_spin, fine_color, X, Y, Z, T]``。"""
    expected = tuple(fine_shape)
    if null_vectors.ndim == 6:
        if tuple(null_vectors.shape[-4:]) != expected:
            raise ValueError(
                f"null vectors 时空尺寸 {tuple(null_vectors.shape[-4:])} != {expected}")
        if int(null_vectors.shape[1]) != fine_spin * fine_color:
            raise ValueError(
                f"null vectors 自由度 {null_vectors.shape[1]} != "
                f"{fine_spin}*{fine_color}")
        return null_vectors.reshape(
            int(null_vectors.shape[0]), fine_spin, fine_color, *expected)
    if null_vectors.ndim == 7:
        if tuple(null_vectors.shape[-4:]) != expected:
            raise ValueError(
                f"null vectors 时空尺寸 {tuple(null_vectors.shape[-4:])} != {expected}")
        if tuple(int(x) for x in null_vectors.shape[1:3]) != (fine_spin, fine_color):
            raise ValueError(
                f"null vectors spin/color={tuple(null_vectors.shape[1:3])} != "
                f"({fine_spin}, {fine_color})")
        return null_vectors
    raise ValueError(
        "null vectors 必须是 [Nvec, fine_dof, X,Y,Z,T] 或 "
        "[Nvec, fine_spin, fine_color, X,Y,Z,T]")


def _cg_orthogonalise(matrix: Tensor, passes: int,
                      normalize: bool = True) -> Tensor:
    """QUDA block_orthogonalize.cuh 的重复 CGS 核心。

    ``matrix`` 为 ``[n_block, local_dim, nvec]``。第一个 pass 使用 B，
    后续 pass 使用上一 pass 的 V；每个向量先对前面向量做 CGS，再归一化。
    """
    if passes < 1:
        raise ValueError(f"n_block_ortho 必须 >= 1，得到 {passes}")
    n_block, local_dim, nvec = (int(matrix.shape[0]), int(matrix.shape[1]),
                                int(matrix.shape[2]))
    current = matrix.clone()
    eps = 1e-30
    for pass_index in range(passes):
        work = matrix.clone() if pass_index == 0 else current.clone()
        new = _torch.zeros_like(work)
        for j in range(nvec):
            vector = work[:, :, j].clone()
            for i in range(j):
                coefficient = _torch.einsum(
                    "nk,nk->n", new[:, :, i].conj(), vector)
                vector = vector - coefficient.unsqueeze(1) * new[:, :, i]
            if normalize:
                norm = _torch.norm(vector, dim=1)
                safe_norm = norm.clone()
                safe_norm[safe_norm < eps] = 1.0
                new[:, :, j] = vector / safe_norm.unsqueeze(1)
            else:
                new[:, :, j] = vector
        current = new
    return current


class QudaTransfer:
    """QUDA aggregate transfer ``P/R``。

    对 Wilson/Clover，典型参数为 ``fine_spin=4``、``fine_color=3``、
    ``coarse_spin=2``、``spin_block_size=2``。``nvec`` 是每个 coarse-spin
    block 的颜色数，因此 coarse 总自由度是 ``2*nvec``，与旧实现把它
    直接称为 ``E`` 的约定不同。
    """

    def __init__(self, null_vectors: Tensor, fine_shape: Sequence[int],
                 fine_spin: int = 4, fine_color: int = 3,
                 coarse_spin: Optional[int] = None,
                 block_size: Sequence[int] = (2, 2, 2, 2),
                 spin_block_size: Optional[int] = None,
                 n_block_ortho: int = 2,
                 normalize: bool = True,
                 verbose: bool = False):
        self.fine_shape = _shape4(fine_shape)
        self.fine_spin = int(fine_spin)
        self.fine_color = int(fine_color)
        if self.fine_spin <= 0 or self.fine_color <= 0:
            raise ValueError("fine_spin/fine_color 必须为正数")
        self.block_size = _shape4(block_size)
        if any(n % b for n, b in zip(self.fine_shape, self.block_size)):
            raise ValueError(
                f"fine_shape={self.fine_shape} 不能被 block_size={self.block_size} 整除")
        self.coarse_shape = tuple(n // b for n, b in
                                  zip(self.fine_shape, self.block_size))
        if coarse_spin is None:
            coarse_spin = 2 if self.fine_spin >= 2 else 2
        self.coarse_spin = int(coarse_spin)
        if self.coarse_spin <= 0:
            raise ValueError("coarse_spin 必须为正数")
        if spin_block_size is None:
            spin_block_size = 0 if self.fine_spin == 1 and self.coarse_spin == 2 else self.fine_spin // self.coarse_spin
        self.spin_block_size = int(spin_block_size)
        self.spin_map_table = [
            [self.spin_map(spin, parity) for parity in (0, 1)]
            for spin in range(self.fine_spin)
        ]
        if self.spin_block_size == 0:
            if not (self.fine_spin == 1 and self.coarse_spin == 2):
                raise ValueError("spin_block_size=0 只用于 staggered 1->2 映射")
        elif self.fine_spin != self.coarse_spin * self.spin_block_size:
            raise ValueError(
                f"fine_spin={self.fine_spin} != coarse_spin*spin_block_size="
                f"{self.coarse_spin * self.spin_block_size}")

        self.B = _normalise_null_shape(
            null_vectors, self.fine_spin, self.fine_color, self.fine_shape)
        self.nvec = int(self.B.shape[0])
        if self.nvec <= 0:
            raise ValueError("至少需要一个 null vector")
        if self.spin_block_size == 0:
            aggregate_volume = prod(self.block_size)
            if aggregate_volume % 2:
                raise ValueError(
                    "staggered 1->2 聚合要求 block 内 fine site 数为偶数")
            local_dim = self.fine_color * (aggregate_volume // 2)
        else:
            local_dim = (self.spin_block_size * self.fine_color *
                         prod(self.block_size))
        if self.nvec > local_dim:
            raise ValueError(
                f"nvec={self.nvec} 超过每个 coarse-spin aggregate 的局部维数 "
                f"{local_dim}")
        self.coarse_dof = self.coarse_spin * self.nvec
        self.fine_dof = self.fine_spin * self.fine_color
        if self.coarse_spin > 2 and self.spin_block_size == 0:
            raise ValueError("staggered coarse_spin 目前必须为 2")

        self.fine_to_coarse: List[int] = []
        self.coarse_to_fine: List[List[int]] = [
            [] for _ in range(prod(self.coarse_shape))]
        for coord in _all_coords(self.fine_shape):
            coarse_coord = tuple(c // b for c, b in zip(coord, self.block_size))
            coarse_index = _site_index(coarse_coord, self.coarse_shape)  # type: ignore[arg-type]
            fine_index = _site_index(coord, self.fine_shape)
            self.fine_to_coarse.append(coarse_index)
            self.coarse_to_fine[coarse_index].append(fine_index)

        self.V = self._block_orthogonalise(
            passes=int(n_block_ortho), normalize=normalize)
        self.n_block_ortho = int(n_block_ortho)
        self.verbose = bool(verbose)
        if self.verbose:
            print(
                "PYQCU::SOLVER::QUDA_MG::TRANSFER:\n "
                f"fine={self.fine_shape}, coarse={self.coarse_shape}, "
                f"spin {self.fine_spin}->{self.coarse_spin}, nvec={self.nvec}, "
                f"block={self.block_size}, CGS passes={self.n_block_ortho}")

    def spin_map(self, fine_spin: int, parity: int = 0) -> int:
        if self.spin_block_size == 0:
            return int(parity)
        return int(fine_spin) // self.spin_block_size

    def _block_orthogonalise(self, passes: int, normalize: bool) -> Tensor:
        """按 QUDA 的 aggregate/chirality 组织 B 并做重复 CGS。"""
        dtype = self.B.dtype
        device = self.B.device
        V = _torch.zeros(
            size=[self.fine_spin, self.fine_color, self.coarse_spin,
                  self.nvec, *self.fine_shape], dtype=dtype, device=device)

        # Wilson/Clover fast path: 同一 coarse-spin block 内所有 fine spin
        # 属于相同的 aggregate，直接 reshape/permute 到批量矩阵。
        if self.spin_block_size != 0:
            bx, by, bz, bt = self.block_size
            cx, cy, cz, ct = self.coarse_shape
            for coarse_spin in range(self.coarse_spin):
                s0 = coarse_spin * self.spin_block_size
                s1 = s0 + self.spin_block_size
                block = self.B[:, s0:s1].reshape(
                    self.nvec, self.spin_block_size, self.fine_color,
                    cx, bx, cy, by, cz, bz, ct, bt)
                block = block.permute(
                    3, 5, 7, 9, 0, 1, 2, 4, 6, 8, 10).contiguous()
                local_dim = self.spin_block_size * self.fine_color * prod(self.block_size)
                matrix = block.reshape(-1, self.nvec, local_dim).transpose(1, 2)
                # 上面的 transpose 把 [block,nvec,local] 变为
                # [block,local,nvec]，与 CGS 核心一致。
                matrix = _cg_orthogonalise(matrix, passes, normalize=normalize)
                matrix = matrix.transpose(1, 2).reshape(
                    cx, cy, cz, ct, self.nvec, self.spin_block_size,
                    self.fine_color, bx, by, bz, bt)
                block = matrix.permute(
                    4, 5, 6, 0, 7, 1, 8, 2, 9, 3, 10).contiguous()
                block = block.reshape(
                    self.nvec, self.spin_block_size, self.fine_color,
                    *self.fine_shape)
                V[s0:s1, :, coarse_spin] = block.permute(
                    1, 2, 0, 3, 4, 5, 6)
            return V

        # Staggered 1->2 的 slow reference path：每个 coarse-spin 只收集
        # aggregate 内对应 checkerboard parity 的 fine site。
        for coarse_coord in _all_coords(self.coarse_shape):
            ranges = [
                range(c * b, (c + 1) * b)
                for c, b in zip(coarse_coord, self.block_size)
            ]
            fine_sites = list(product(*ranges))
            for coarse_spin in range(self.coarse_spin):
                entries: List[Tensor] = []
                for coord in fine_sites:
                    parity = sum(coord) & 1
                    if parity != coarse_spin:
                        continue
                    for color in range(self.fine_color):
                        entries.append(self.B[(slice(None), 0, color, *coord)])
                if not entries:
                    continue
                matrix = _torch.stack(entries, dim=0).unsqueeze(0)
                matrix = _cg_orthogonalise(
                    matrix, passes, normalize=normalize)[0]
                cursor = 0
                for coord in fine_sites:
                    if (sum(coord) & 1) != coarse_spin:
                        continue
                    for color in range(self.fine_color):
                        V[(0, color, coarse_spin, slice(None), *coord)] = matrix[cursor]
                        cursor += 1
        return V

    def _coarse_structured(self, coarse: Tensor) -> Tensor:
        if coarse.ndim == 5 and int(coarse.shape[0]) == self.coarse_dof:
            return coarse.reshape(self.coarse_spin, self.nvec, *self.coarse_shape)
        if (coarse.ndim == 6 and
                tuple(int(x) for x in coarse.shape[:2]) ==
                (self.coarse_spin, self.nvec) and
                tuple(int(x) for x in coarse.shape[-4:]) == self.coarse_shape):
            return coarse
        raise ValueError(
            f"coarse field 应为 [{self.coarse_dof},*coarse_shape] 或 "
            f"[{self.coarse_spin},{self.nvec},*coarse_shape]，得到 {tuple(coarse.shape)}")

    def _fine_structured(self, fine: Tensor) -> Tensor:
        if fine.ndim == 5 and int(fine.shape[0]) == self.fine_dof:
            return fine.reshape(self.fine_spin, self.fine_color, *self.fine_shape)
        if (fine.ndim == 6 and
                tuple(int(x) for x in fine.shape[:2]) ==
                (self.fine_spin, self.fine_color) and
                tuple(int(x) for x in fine.shape[-4:]) == self.fine_shape):
            return fine
        raise ValueError(
            f"fine field 应为 [{self.fine_dof},*fine_shape] 或 "
            f"[{self.fine_spin},{self.fine_color},*fine_shape]，得到 {tuple(fine.shape)}")

    def prolong_spin_color(self, coarse: Tensor) -> Tensor:
        dtype = coarse.dtype
        device = coarse.device
        coarse_sc = self._coarse_structured(coarse)
        if coarse_sc.dtype != self.V.dtype or coarse_sc.device != self.V.device:
            coarse_sc = coarse_sc.to(dtype=self.V.dtype, device=self.V.device)
        expanded = coarse_sc
        for dim, block in enumerate(self.block_size):
            expanded = expanded.repeat_interleave(block, dim=2 + dim)
        fine = _torch.zeros(
            size=[self.fine_spin, self.fine_color, *self.fine_shape],
            dtype=coarse_sc.dtype, device=coarse_sc.device)
        for spin in range(self.fine_spin):
            coarse_spin = self.spin_map(spin, 0)
            if self.spin_block_size == 0:
                # full-field staggered P 的 coarse-spin 由 fine-site parity 决定；
                # 其 slow path 用显式 site assignment 保持语义清楚。
                for coord in _all_coords(self.fine_shape):
                    for color in range(self.fine_color):
                        fine[(spin, color, *coord)] = _torch.einsum(
                            "v,v->", self.V[(spin, color, sum(coord) & 1,
                                              slice(None), *coord)],
                            expanded[(sum(coord) & 1, slice(None), *coord)])
            else:
                for color in range(self.fine_color):
                    fine[spin, color] = _torch.einsum(
                        "vxyzt,vxyzt->xyzt",
                        self.V[spin, color, coarse_spin], expanded[coarse_spin])
        return fine.to(dtype=dtype, device=device)

    def prolong(self, coarse: Tensor) -> Tensor:
        return self.prolong_spin_color(coarse).reshape(self.fine_dof, *self.fine_shape)

    P = prolong

    def restrict_spin_color(self, fine: Tensor) -> Tensor:
        dtype = fine.dtype
        device = fine.device
        fine_sc = self._fine_structured(fine)
        if fine_sc.dtype != self.V.dtype or fine_sc.device != self.V.device:
            fine_sc = fine_sc.to(dtype=self.V.dtype, device=self.V.device)
        coarse = _torch.zeros(
            size=[self.coarse_spin, self.nvec, *self.coarse_shape],
            dtype=fine_sc.dtype, device=fine_sc.device)
        bx, by, bz, bt = self.block_size
        cx, cy, cz, ct = self.coarse_shape
        for coarse_spin in range(self.coarse_spin):
            if self.spin_block_size == 0:
                spins = list(range(self.fine_spin))
            else:
                spins = [s for s in range(self.fine_spin)
                         if self.spin_map(s, 0) == coarse_spin]
            for vector in range(self.nvec):
                accum: Optional[Tensor] = None
                for spin in spins:
                    for color in range(self.fine_color):
                        product_field = self.V[spin, color, coarse_spin, vector].conj() * fine_sc[spin, color]
                        if self.spin_block_size == 0:
                            # A staggered coarse-spin component owns only one
                            # checkerboard parity inside each aggregate.
                            value = _torch.zeros(
                                size=self.coarse_shape, dtype=product_field.dtype,
                                device=product_field.device)
                            for coarse_coord in _all_coords(self.coarse_shape):
                                ranges = [
                                    range(c * b, (c + 1) * b)
                                    for c, b in zip(coarse_coord, self.block_size)
                                ]
                                for coord in product(*ranges):
                                    if (sum(coord) & 1) == coarse_spin:
                                        value[coarse_coord] = value[coarse_coord] + product_field[coord]
                        else:
                            blocked = product_field.reshape(
                                cx, bx, cy, by, cz, bz, ct, bt)
                            value = _sum_block_axes(blocked)
                        accum = value if accum is None else accum + value
                if accum is not None:
                    coarse[coarse_spin, vector] = accum
        return coarse.to(dtype=dtype, device=device)

    def restrict(self, fine: Tensor) -> Tensor:
        return self.restrict_spin_color(fine).reshape(self.coarse_dof, *self.coarse_shape)

    R = restrict

    def orthogonality_error(self) -> float:
        """返回所有 aggregate/coarse-spin Gram 矩阵的最大非单位误差。"""
        worst = 0.0
        for coarse in _all_coords(self.coarse_shape):
            ranges = [
                range(c * b, (c + 1) * b)
                for c, b in zip(coarse, self.block_size)
            ]
            sites = list(product(*ranges))
            for coarse_spin in range(self.coarse_spin):
                gram = _torch.zeros(
                    size=[self.nvec, self.nvec], dtype=self.V.dtype,
                    device=self.V.device)
                if self.spin_block_size == 0:
                    sites_for_spin = [site for site in sites
                                      if (sum(site) & 1) == coarse_spin]
                    spins = range(self.fine_spin)
                else:
                    sites_for_spin = sites
                    spins = [spin for spin in range(self.fine_spin)
                             if self.spin_map(spin, 0) == coarse_spin]
                for spin in spins:
                    for color in range(self.fine_color):
                        local = _torch.stack([
                            self.V[(spin, color, coarse_spin, vector, *site)]
                            for vector in range(self.nvec)
                            for site in sites_for_spin
                        ]).reshape(self.nvec, len(sites_for_spin))
                        gram = gram + _torch.einsum(
                            "ik,jk->ij", local.conj(), local)
                identity = _torch.eye(
                    self.nvec, dtype=self.V.dtype, device=self.V.device)
                worst = max(worst, float(_torch.norm(gram - identity).item()))
        return worst


@dataclass
class _Level:
    index: int
    operator: Any
    transfer: Optional[QudaTransfer]
    spin: int
    color: int
    shape: Shape4


class _FineOperator:
    """把 PyQCU fine dslash 或用户 callable 统一为 flat-field operator。"""

    def __init__(self, matvec: Callable[[Tensor], Tensor], shape: Shape4,
                 spin: int, color: int,
                 diagonal: Optional[Tensor] = None,
                 adjoint: Optional[Callable[[Tensor], Tensor]] = None):
        self._matvec = matvec
        self.shape = shape
        self.spin = int(spin)
        self.color = int(color)
        self.dof = self.spin * self.color
        self._diagonal = diagonal
        self._adjoint = adjoint
        self._diagonal_inv: Dict[Tuple[Any, Any], Tensor] = {}

    def apply(self, value: Tensor) -> Tensor:
        return _call_matvec(self._matvec, value)

    matvec = apply

    def adjoint_apply(self, value: Tensor) -> Tensor:
        if self._adjoint is None:
            raise RuntimeError("fine operator 未提供 adjoint")
        return _call_matvec(self._adjoint, value)

    def diagonal(self, reference: Optional[Tensor] = None) -> Tensor:
        if self._diagonal is not None:
            return self._diagonal
        if reference is None:
            raise RuntimeError("缺少 reference，无法构造 fine identity diagonal")
        return _torch.eye(self.dof, dtype=reference.dtype,
                          device=reference.device).reshape(
                              self.dof, self.dof, 1, 1, 1, 1).expand(
                                  self.dof, self.dof, *self.shape)

    def diagonal_apply(self, value: Tensor) -> Tensor:
        dtype = value.dtype
        device = value.device
        matrix = self.diagonal(value)
        work = value
        if work.dtype != matrix.dtype or work.device != matrix.device:
            work = work.to(dtype=matrix.dtype, device=matrix.device)
        return _matvec_block(matrix, work).to(dtype=dtype, device=device)

    def diagonal_inverse(self, reference: Tensor) -> Tensor:
        matrix = self.diagonal(reference)
        key = (matrix.device, matrix.dtype)
        if key not in self._diagonal_inv:
            site_matrix = matrix.permute(2, 3, 4, 5, 0, 1).reshape(-1, self.dof, self.dof)
            inverse = _torch.linalg_inv(site_matrix)
            self._diagonal_inv[key] = inverse.reshape(
                *self.shape, self.dof, self.dof).permute(4, 5, 0, 1, 2, 3).contiguous()
        return self._diagonal_inv[key]

    def diagonal_inv_apply(self, value: Tensor) -> Tensor:
        dtype = value.dtype
        device = value.device
        matrix = self.diagonal_inverse(value)
        work = value
        if work.dtype != matrix.dtype or work.device != matrix.device:
            work = work.to(dtype=matrix.dtype, device=matrix.device)
        return _matvec_block(matrix, work).to(dtype=dtype, device=device)


class QudaCoarseOperator:
    """Galerkin coarse operator and QUDA-style ``X/Y/Yhat`` metadata。

    ``blocks[d]`` 使用目标点约定
    ``(D_c x)(q) += blocks[d](q) x(q+d)``。因此 ``Y_forward`` 与
    ``Y_backward`` 是便于矩阵自由动作验证的目标点系数；
    ``Yhat_backward`` 则额外转换成 QUDA 的 link 存储位置 ``q-d``，供
    ``preconditioned_apply`` 按 backward gather 取其伴随。这样既保留
    Python 参考实现的直接 ``RDP`` 语义，也不会丢失 QUDA backward link
    的坐标与左右乘法约定。
    """

    def __init__(self, transfer: QudaTransfer, fine_operator: Any,
                 materialize: bool = False, support_tol: float = 1e-12,
                 max_materialize_elements: int = 50_000_000,
                 verbose: bool = False):
        self.transfer = transfer
        self.fine_operator = fine_operator
        self.shape = transfer.coarse_shape
        self.spin = transfer.coarse_spin
        self.color = transfer.nvec
        self.dof = transfer.coarse_dof
        self.support_tol = float(support_tol)
        self.max_materialize_elements = int(max_materialize_elements)
        if self.max_materialize_elements <= 0:
            raise ValueError("max_materialize_elements 必须为正数")
        self.verbose = bool(verbose)
        self.blocks: Optional[Dict[BlockKey, Tensor]] = None
        self.X: Optional[Tensor] = None
        self.X_inv: Optional[Tensor] = None
        self.Y_forward: Optional[List[Tensor]] = None
        self.Y_backward: Optional[List[Tensor]] = None
        self.Yhat_forward: Optional[List[Tensor]] = None
        self.Yhat_backward: Optional[List[Tensor]] = None
        self._dense: Optional[Tensor] = None
        if materialize:
            self.build()

    def apply(self, value: Tensor) -> Tensor:
        if value.ndim != 5 or int(value.shape[0]) != self.dof:
            raise ValueError(
                f"粗场应为 [{self.dof}, X,Y,Z,T]，得到 {tuple(value.shape)}")
        if self.blocks is None:
            fine = self.transfer.prolong(value)
            result = _call_matvec(self.fine_operator, fine)
            return self.transfer.restrict(result)
        return self.apply_decomposed(value)

    matvec = apply

    def apply_decomposed(self, value: Tensor) -> Tensor:
        if self.blocks is None:
            raise RuntimeError("coarse operator 尚未 materialize")
        dtype = value.dtype
        device = value.device
        work = value
        if work.dtype != self.transfer.V.dtype or work.device != self.transfer.V.device:
            work = work.to(dtype=self.transfer.V.dtype, device=self.transfer.V.device)
        result = _torch.zeros_like(work)
        for displacement, block in self.blocks.items():
            result = result + _matvec_block(block, _roll_field(work, displacement))
        return result.to(dtype=dtype, device=device)

    def adjoint_apply(self, value: Tensor) -> Tensor:
        if self.blocks is None:
            if not hasattr(self.fine_operator, "adjoint_apply"):
                raise RuntimeError("fine operator 未提供 adjoint，且 coarse 未 materialize")
            fine = self.transfer.prolong(value)
            result = self.fine_operator.adjoint_apply(fine)
            return self.transfer.restrict(result)
        dtype = value.dtype
        device = value.device
        work = value
        if work.dtype != self.transfer.V.dtype or work.device != self.transfer.V.device:
            work = work.to(dtype=self.transfer.V.dtype, device=self.transfer.V.device)
        result = _torch.zeros_like(work)
        for displacement, block in self.blocks.items():
            coefficient = _roll_site_tensor(_adjoint_site(block), displacement)
            result = result + _matvec_block(
                coefficient, _roll_field(work, tuple(-x for x in displacement)))
        return result.to(dtype=dtype, device=device)

    Mdag = adjoint_apply

    def build(self) -> "QudaCoarseOperator":
        """逐列探测 ``RDP`` 并按 coarse displacement 组织 block。"""
        if self.blocks is not None:
            return self
        one_block_elements = self.dof * self.dof * prod(self.shape)
        if one_block_elements > self.max_materialize_elements:
            raise MemoryError(
                "逐列 materialize coarse operator 的一个位移块需要 "
                f"{one_block_elements:,} 个元素，超过上限 "
                f"{self.max_materialize_elements:,}；请使用 "
                "materialize_coarse=False 且 use_parity=False，或提高 "
                "max_materialize_elements")
        dtype = self.transfer.V.dtype
        device = self.transfer.V.device
        blocks: Dict[BlockKey, Tensor] = {}
        zero = _zero_key()
        blocks[zero] = _torch.zeros(
            size=[self.dof, self.dof, *self.shape], dtype=dtype, device=device)
        stored_elements = one_block_elements
        nsite = prod(self.shape)
        probes = 0
        for source_coord in _all_coords(self.shape):
            source_index = _site_index(source_coord, self.shape)
            for source_dof in range(self.dof):
                probe = _torch.zeros(
                    size=[self.dof, *self.shape], dtype=dtype, device=device)
                probe[(source_dof, *source_coord)] = 1.0
                image = self._apply_matrix_free(probe)
                for target_coord in _all_coords(self.shape):
                    vector = image[(slice(None), *target_coord)]
                    if float(_torch.norm(vector).item()) <= self.support_tol:
                        continue
                    displacement = tuple(
                        _signed_displacement(source_coord[d] - target_coord[d],
                                             self.shape[d])
                        for d in range(4))
                    key = displacement  # type: ignore[assignment]
                    if key not in blocks:
                        if stored_elements + one_block_elements > self.max_materialize_elements:
                            raise MemoryError(
                                "materialize coarse operator 的位移块总存储量将达到 "
                                f"{stored_elements + one_block_elements:,} 个元素，"
                                f"超过上限 {self.max_materialize_elements:,}；请使用 "
                                "materialize_coarse=False 且 use_parity=False，或提高 "
                                "max_materialize_elements")
                        blocks[key] = _torch.zeros(
                            size=[self.dof, self.dof, *self.shape],
                            dtype=image.dtype, device=image.device)
                        stored_elements += one_block_elements
                    blocks[key][(slice(None), source_dof, *target_coord)] = vector
                probes += 1
            if self.verbose and (source_index + 1) % max(1, nsite // 8) == 0:
                print(
                    "PYQCU::SOLVER::QUDA_MG::COARSE:\n "
                    f"Galerkin probes {probes}/{nsite * self.dof}")
        self.blocks = blocks
        self.X = blocks[zero]
        self._build_links()
        if self.verbose:
            print(
                "PYQCU::SOLVER::QUDA_MG::COARSE:\n "
                f"support displacements={sorted(blocks)}, probes={probes}")
        return self

    def _apply_matrix_free(self, value: Tensor) -> Tensor:
        fine = self.transfer.prolong(value)
        image = _call_matvec(self.fine_operator, fine)
        return self.transfer.restrict(image)

    def _link(self, displacement: BlockKey) -> Tensor:
        if self.blocks is None:
            raise RuntimeError("coarse operator 尚未 materialize")
        canonical = tuple(
            _signed_displacement(value, self.shape[dim])
            for dim, value in enumerate(displacement))
        if canonical in self.blocks:
            return self.blocks[canonical]  # type: ignore[index]
        return _torch.zeros(
            size=[self.dof, self.dof, *self.shape], dtype=self.transfer.V.dtype,
            device=self.transfer.V.device)

    def _build_links(self) -> None:
        if self.blocks is None or self.X is None:
            raise RuntimeError("coarse blocks 未构造")
        self.Y_forward = []
        self.Y_backward = []
        for dim in range(4):
            plus = [0, 0, 0, 0]
            minus = [0, 0, 0, 0]
            plus[dim] = 1
            minus[dim] = -1
            plus_key = tuple(plus)  # type: ignore[assignment]
            minus_key = tuple(minus)  # type: ignore[assignment]
            forward = self._link(plus_key)
            backward = self._link(minus_key)
            # coarse extent=2 时 +1/-1 是同一邻居；QUDA 的两个方向各
            # 持有一份 link，避免由 RDP 的合并 block 在 apply 时重复计数。
            if self.shape[dim] == 2 and plus_key == tuple(-x for x in minus_key):
                forward = 0.5 * forward
                backward = 0.5 * backward
            self.Y_forward.append(forward)
            self.Y_backward.append(backward)

        matrix = self.X.permute(2, 3, 4, 5, 0, 1).reshape(-1, self.dof, self.dof)
        inverse = _torch.linalg_inv(matrix)
        self.X_inv = inverse.reshape(
            *self.shape, self.dof, self.dof).permute(4, 5, 0, 1, 2, 3).contiguous()
        self.Yhat_forward = []
        self.Yhat_backward = []
        for dim in range(4):
            assert self.Y_forward is not None and self.Y_backward is not None
            self.Yhat_forward.append(_matmul_site(self.X_inv, self.Y_forward[dim]))
            # QUDA stores a backward link at q-mu.  ``Y_backward`` below is
            # kept in the action-coefficient convention (indexed by the
            # destination q), so its equivalent stored link is obtained by
            # moving the coefficient to q-mu and taking the adjoint.  The
            # right factor is X^{-dagger}(q), not X^{-dagger}(q-mu).
            backward_storage = _roll_site_tensor(
                _adjoint_site(self.Y_backward[dim]), tuple(
                    -1 if i == dim else 0 for i in range(4)))
            source_xinv = _roll_site_tensor(self.X_inv, tuple(
                -1 if i == dim else 0 for i in range(4)))
            self.Yhat_backward.append(_matmul_site(
                backward_storage, _adjoint_site(source_xinv)))

    @property
    def dense_matrix(self) -> Tensor:
        if self.blocks is None:
            self.build()
        if self._dense is not None:
            return self._dense
        assert self.blocks is not None
        n = self.dof * prod(self.shape)
        dense = _torch.zeros(
            size=[n, n], dtype=self.transfer.V.dtype, device=self.transfer.V.device)
        for displacement, block in self.blocks.items():
            for target in _all_coords(self.shape):
                source = tuple((target[d] + displacement[d]) % self.shape[d]
                               for d in range(4))
                target_site = _site_index(target, self.shape)
                source_site = _site_index(source, self.shape)
                row = slice(target_site, n, prod(self.shape))
                col = slice(source_site, n, prod(self.shape))
                dense[row, col] = block[(slice(None), slice(None), *target)]
        self._dense = dense
        return dense

    def diagonal_apply(self, value: Tensor) -> Tensor:
        if self.X is None:
            self.build()
        assert self.X is not None
        dtype = value.dtype
        device = value.device
        work = value
        if work.dtype != self.X.dtype or work.device != self.X.device:
            work = work.to(dtype=self.X.dtype, device=self.X.device)
        return _matvec_block(self.X, work).to(dtype=dtype, device=device)

    def diagonal_inv_apply(self, value: Tensor) -> Tensor:
        if self.X_inv is None:
            self.build()
        assert self.X_inv is not None
        dtype = value.dtype
        device = value.device
        work = value
        if work.dtype != self.X_inv.dtype or work.device != self.X_inv.device:
            work = work.to(dtype=self.X_inv.dtype, device=self.X_inv.device)
        return _matvec_block(self.X_inv, work).to(dtype=dtype, device=device)

    def preconditioned_apply(self, value: Tensor) -> Tensor:
        """应用 QUDA ``DiracCoarsePC::Dslash`` 的非对角部分。

        本参考实现的 ``Y_forward/Y_backward`` 是按 ``D_c`` 的目标点存储
        的、已经包含原算子系数的 action coefficient，而非 QUDA 内部
        ``Y[d]``/``Y[d+4]`` 的 gauge-link 命名。因此这里把两种存储约定
        连接起来后，结果应满足

        ``preconditioned_apply(x) = X^{-1} (D_c-X) x``。

        这正是 QUDA 的 ``DiracCoarsePC::Dslash`` 在非 dagger 路径上表达的
        邻点动作；局部 ``X`` 不在该接口中重复加入。
        """
        if self.X is None or self.Yhat_forward is None or self.Yhat_backward is None:
            self.build()
        assert self.Yhat_forward is not None and self.Yhat_backward is not None
        dtype = value.dtype
        device = value.device
        work = value
        if work.dtype != self.transfer.V.dtype or work.device != self.transfer.V.device:
            work = work.to(dtype=self.transfer.V.dtype, device=self.transfer.V.device)
        result = _torch.zeros_like(work)
        for dim in range(4):
            displacement = tuple(1 if i == dim else 0 for i in range(4))
            result = result + _matvec_block(
                self.Yhat_forward[dim], _roll_field(work, displacement))
            # Yhat_backward is stored at q-mu and the coarse dslash uses its
            # adjoint at the target q.  The roll therefore first fetches the
            # link at q-mu, then the site adjoint restores the action
            # coefficient X^{-1}(q) Y_backward(q).
            storage_shift = tuple(1 if i == dim else 0 for i in range(4))
            backward_coefficient = _adjoint_site(_roll_site_tensor(
                self.Yhat_backward[dim], storage_shift))
            result = result + _matvec_block(
                backward_coefficient,
                _roll_field(work, tuple(-x for x in displacement)))
        return result.to(dtype=dtype, device=device)

    preconditioned_hopping_apply = preconditioned_apply

    def preconditioned_full_apply(self, value: Tensor) -> Tensor:
        """返回 ``X^{-1}D_c``，即局部项加上 PC hopping。"""
        # X^{-1} X is the identity, while preconditioned_apply contains
        # X^{-1}(D_c-X).
        return value + self.preconditioned_apply(value)



class Checkerboard:
    """全场与 compact even/odd field 之间的确定性映射。"""

    def __init__(self, shape: Sequence[int]):
        self.shape = _shape4(shape)
        self.indices = {
            0: [_site_index(c, self.shape) for c in _all_coords(self.shape)
                if (sum(c) & 1) == 0],
            1: [_site_index(c, self.shape) for c in _all_coords(self.shape)
                if (sum(c) & 1) == 1],
        }
        if len(self.indices[0]) != len(self.indices[1]):
            raise ValueError(
                f"checkerboard 要求偶数体积，shape={self.shape} 的两 parity 数目不同")

    @property
    def volume(self) -> int:
        return len(self.indices[0])

    def extract(self, field: Tensor, parity: int) -> Tensor:
        if field.ndim != 5 or tuple(int(x) for x in field.shape[1:]) != self.shape:
            raise ValueError(f"field shape 应为 [dof,{self.shape}]，得到 {tuple(field.shape)}")
        return field.reshape(int(field.shape[0]), -1)[:, self.indices[int(parity)]]

    def embed(self, compact: Tensor, parity: int, dof: int) -> Tensor:
        if compact.ndim != 2 or int(compact.shape[0]) != dof or int(compact.shape[1]) != self.volume:
            raise ValueError(
                f"compact field 应为 [{dof},{self.volume}]，得到 {tuple(compact.shape)}")
        field = _torch.zeros(
            size=[dof, *self.shape], dtype=compact.dtype, device=compact.device)
        field.reshape(dof, -1)[:, self.indices[int(parity)]] = compact
        return field


class ParitySchurOperator:
    """任意 full coarse operator 的 even/odd Schur 补。

    约定消去 even 分量，返回 odd Schur：

    ``S_o = D_oo - D_oe X_e^{-1} D_eo``，
    ``b_o^S = b_o - D_oe X_e^{-1} b_e``。
    """

    def __init__(self, operator: Any):
        self.operator = operator
        self.shape = _shape4(operator.shape)
        self.dof = int(operator.dof)
        self.checkerboard = Checkerboard(self.shape)

    def _coupling(self, compact: Tensor, source_parity: int,
                  target_parity: int) -> Tensor:
        full = self.checkerboard.embed(compact, source_parity, self.dof)
        image = self.operator.apply(full)
        return self.checkerboard.extract(image, target_parity)

    def apply(self, odd: Tensor) -> Tensor:
        deo = self._coupling(odd, 1, 0)
        even_input = self.checkerboard.embed(deo, 0, self.dof)
        even_inverse = self.operator.diagonal_inv_apply(even_input)
        doe = self._coupling(
            self.checkerboard.extract(even_inverse, 0), 0, 1)
        odd_full = self.checkerboard.embed(odd, 1, self.dof)
        doo = self.checkerboard.extract(
            self.operator.diagonal_apply(odd_full), 1)
        return doo - doe

    matvec = apply

    def rhs(self, full_rhs: Tensor) -> Tensor:
        b_even = self.checkerboard.extract(full_rhs, 0)
        b_odd = self.checkerboard.extract(full_rhs, 1)
        even_input = self.checkerboard.embed(b_even, 0, self.dof)
        even_inverse = self.operator.diagonal_inv_apply(even_input)
        correction = self._coupling(
            self.checkerboard.extract(even_inverse, 0), 0, 1)
        return b_odd - correction

    def reconstruct(self, full_rhs: Tensor, odd_solution: Tensor) -> Tensor:
        b_even = self.checkerboard.extract(full_rhs, 0)
        odd_full = self.checkerboard.embed(odd_solution, 1, self.dof)
        deo = self.checkerboard.extract(self.operator.apply(odd_full), 0)
        even_input = self.checkerboard.embed(b_even - deo, 0, self.dof)
        even_solution = self.operator.diagonal_inv_apply(even_input)
        result = _torch.zeros_like(full_rhs)
        result.reshape(self.dof, -1)[:, self.checkerboard.indices[0]] = even_solution.reshape(self.dof, -1)[:, self.checkerboard.indices[0]]
        result.reshape(self.dof, -1)[:, self.checkerboard.indices[1]] = odd_solution
        return result

    def mr_correction(self, full_rhs: Tensor, steps: int = 1) -> Tensor:
        """以 compact odd Schur 做 ``steps`` 次 MR，返回 full correction。"""
        rhs = self.rhs(full_rhs)
        odd = _torch.zeros_like(rhs)
        for _ in range(max(0, int(steps))):
            residual = rhs - self.apply(odd)
            image = self.apply(residual)
            denominator = tools.vdot(image, image)
            if float(_torch.abs(denominator).item()) < 1e-30:
                break
            alpha = tools.vdot(image, residual) / denominator
            odd = odd + alpha * residual
        return self.reconstruct(full_rhs, odd)

    def solve(self, full_rhs: Tensor, tol: float = 1e-8,
              max_iter: int = 100, restart: int = 20,
              verbose: bool = False) -> Tensor:
        from ._gmres import fgmres
        rhs = self.rhs(full_rhs)
        odd = fgmres(rhs, self.apply, tol=tol, max_iter=max_iter,
                     restart=restart, if_rtol=True, verbose=verbose)
        return self.reconstruct(full_rhs, odd)


class QudaMultigrid:
    """可切换的 QUDA 风格递归聚合 MG。

    参数要点：

    ``fine_matvec`` 采用 flat ``[fine_spin*fine_color, ...]`` 输入；也可以
    直接给 ``U``/``clover_term``，此时复用 PyQCU 的 Wilson/Clover dslash。
    ``null_vectors`` 是每个过渡层的列表，元素为 ``[Nvec, fine_dof, ...]``；
    不提供时按 ``nvec_list`` 生成确定性随机起点，适合结构验证，不等同于
    QUDA 的 CG/CA-CG/Krylov null-vector setup。
    ``dof_list`` 仍可使用旧项目的总 coarse DOF 约定，例如 ``[12,24,24]``；
    本实现会将粗层总 DOF 除以 2 得到每个 coarse-spin block 的 ``Nvec``。
    ``materialize_coarse=True`` 是小格验证模式，会逐列显式保存 ``RDP`` 的
    位移块；大格点应关闭它以使用矩阵自由动作（并同时关闭 ``use_parity``，
    因为奇偶 Schur 需要已构造的每层 ``X/Y``）。
    """

    def __init__(self, U: Optional[Tensor] = None,
                 clover_term: Optional[Tensor] = None,
                 fine_matvec: Optional[Callable[[Tensor], Tensor]] = None,
                 fine_adjoint: Optional[Callable[[Tensor], Tensor]] = None,
                 fine_diagonal: Optional[Tensor] = None,
                 kappa: Optional[Tensor] = None,
                 u_0: Optional[Tensor] = None,
                 lat_size: Optional[Sequence[int]] = None,
                 fine_spin: int = 4, fine_color: int = 3,
                 null_vectors: Optional[Sequence[Tensor] | Tensor] = None,
                 nvec_list: Optional[Sequence[int]] = None,
                 dof_list: Optional[Sequence[int]] = None,
                 block_size: Sequence[int] | Sequence[Sequence[int]] = (2, 2, 2, 2),
                 max_level: Optional[int] = None,
                 n_block_ortho: int = 2,
                 materialize_coarse: bool = True,
                 use_parity: bool = True,
                 nu_pre: int = 2, nu_post: int = 2,
                 coarse_max_iter: int = 100,
                 coarse_tol: float = 1e-8,
                 tol: float = 1e-8, max_iter: int = 100,
                 restart: int = 20,
                 direct_solve_max: int = 4096,
                 max_materialize_elements: int = 50_000_000,
                 seed: int = 42, verbose: bool = False):
        if U is None and fine_matvec is None:
            raise ValueError("U 与 fine_matvec 至少提供一个")
        if U is not None:
            inferred_shape = tuple(int(x) for x in U.shape[-4:])
        elif lat_size is not None:
            inferred_shape = tuple(int(x) for x in lat_size)
        else:
            raise ValueError("使用 fine_matvec 时必须提供 lat_size")
        self.fine_shape = _shape4(inferred_shape)
        self.fine_spin = int(fine_spin)
        self.fine_color = int(fine_color)
        self.fine_dof = self.fine_spin * self.fine_color
        self.U = U
        self.clover_term = clover_term
        self.kappa = kappa
        self.u_0 = u_0
        self.fine_dslash = None
        self.verbose = bool(verbose)
        self.materialize_coarse = bool(materialize_coarse)
        self.use_parity = bool(use_parity)
        self.n_block_ortho = int(n_block_ortho)
        if self.n_block_ortho < 1:
            raise ValueError("n_block_ortho 必须 >= 1")
        self.nu_pre = max(0, int(nu_pre))
        self.nu_post = max(0, int(nu_post))
        self.coarse_max_iter = int(coarse_max_iter)
        self.coarse_tol = float(coarse_tol)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.restart = int(restart)
        self.direct_solve_max = int(direct_solve_max)
        self.max_materialize_elements = int(max_materialize_elements)
        if self.max_materialize_elements <= 0:
            raise ValueError("max_materialize_elements 必须为正数")
        self.seed = int(seed)
        self._setup_done = False

        if fine_matvec is None:
            assert U is not None
            dslash_kwargs: Dict[str, Any] = {"U": U, "verbose": verbose}
            if clover_term is not None:
                dslash_kwargs["clover_term"] = clover_term
            if kappa is not None:
                dslash_kwargs["kappa"] = kappa
            if u_0 is not None:
                dslash_kwargs["u_0"] = u_0
            fine_dslash = dslash.operator(**dslash_kwargs)
            self.fine_dslash = fine_dslash
            fine_matvec = fine_dslash.matvec
            diagonal = self._fine_diagonal_from_dslash(fine_dslash, U)
        else:
            diagonal = fine_diagonal
            if diagonal is not None:
                expected = (self.fine_dof, self.fine_dof, *self.fine_shape)
                if tuple(int(x) for x in diagonal.shape) != expected:
                    raise ValueError(
                        f"fine_diagonal shape 应为 {expected}，得到 {tuple(diagonal.shape)}")
            elif self.use_parity:
                raise ValueError(
                    "自定义 fine_matvec 在 use_parity=True 时必须提供 "
                    "fine_diagonal；否则无法定义局部块的逆")
        self._fine = _FineOperator(
            fine_matvec, self.fine_shape, self.fine_spin, self.fine_color,
            diagonal=diagonal, adjoint=fine_adjoint)
        self._null_vectors = self._normalise_null_list(null_vectors)
        self._nvec_list = self._infer_nvec_list(nvec_list, dof_list)
        self._block_sizes = self._normalise_block_sizes(block_size)
        transitions = self._number_of_transitions(max_level)
        self._transition_count = transitions
        if self.use_parity and transitions and not self.materialize_coarse:
            raise ValueError(
                "use_parity=True 需要已 materialize 的每层 X/Y；对于大格点请 "
                "设置 materialize_coarse=True，或关闭 use_parity 使用矩阵自由模式")
        if len(self._block_sizes) < transitions:
            raise ValueError("block_size 数量少于层间过渡数")
        if len(self._nvec_list) < transitions:
            raise ValueError("nvec_list/dof_list 数量少于层间过渡数")

        self.levels: List[_Level] = []
        self.transfers: List[QudaTransfer] = []
        self.operators: List[Any] = []
        self.parity_operators: List[ParitySchurOperator] = []

    @staticmethod
    def _fine_diagonal_from_dslash(operator: Any, U: Tensor) -> Tensor:
        if getattr(operator.sitting, "clover_term", None) is None:
            identity = _torch.eye(12, dtype=U.dtype, device=U.device)
            return identity.reshape(12, 12, 1, 1, 1, 1).expand(12, 12, *U.shape[-4:]).clone()
        return operator.sitting.M

    @staticmethod
    def _normalise_null_list(null_vectors: Optional[Sequence[Tensor] | Tensor]) -> List[Tensor]:
        if null_vectors is None:
            return []
        if hasattr(null_vectors, "ndim"):
            return [null_vectors]  # type: ignore[list-item]
        return list(null_vectors)

    def _infer_nvec_list(self, nvec_list: Optional[Sequence[int]],
                         dof_list: Optional[Sequence[int]]) -> List[int]:
        if nvec_list is not None:
            values = [int(x) for x in nvec_list]
        elif dof_list is not None:
            values = []
            for total in list(dof_list)[1:]:
                total = int(total)
                if total % 2:
                    raise ValueError(f"QUDA Wilson/Clover coarse 总 DOF 必须为偶数，得到 {total}")
                values.append(total // 2)
        elif self._null_vectors:
            values = [int(v.shape[0]) for v in self._null_vectors]
        else:
            values = [4]
        if any(value <= 0 for value in values):
            raise ValueError(f"nvec_list 必须为正数，得到 {values}")
        return values

    def _normalise_block_sizes(self, block_size: Sequence[int] | Sequence[Sequence[int]]) -> List[Shape4]:
        values = list(block_size)
        if not values:
            return []
        if isinstance(values[0], int):
            one = _shape4(values)  # type: ignore[arg-type]
            return [one]
        return [_shape4(value) for value in values]  # type: ignore[arg-type]

    def _number_of_transitions(self, max_level: Optional[int]) -> int:
        available = max(len(self._null_vectors), len(self._nvec_list))
        if max_level is None:
            return max(1, available) if available else 1
        transitions = max(0, int(max_level) - 1)
        return transitions

    def _random_null(self, operator: Any, nvec: int,
                     dtype: Any, device: Any) -> Tensor:
        shape = [nvec, operator.spin, operator.color, *operator.shape]
        # cann.randn 当前不接受 generator；用逐层固定 seed 的 CPU/CUDA
        # 全局调用即可保证单线程 setup 的可复现性，MultiGPU 路径由调用方
        # 显式传 null_vectors 管理独立 RNG。
        return _torch.randn(size=shape, dtype=dtype, device=device)

    def setup(self) -> "QudaMultigrid":
        if self._setup_done:
            return self
        # ``cann`` 负责屏蔽 NPU/复数差异；将种子也放在兼容层内，避免
        # 纯 Python 求解器重新直接依赖 torch。显式 null_vectors 时不会
        # 生成随机向量，但仍保持后续随机诊断的可复现起点。
        _torch.manual_seed(self.seed)
        current: Any = self._fine
        self.operators = [current]
        self.levels = []
        for level in range(self._transition_count):
            nvec = self._nvec_list[level]
            if level < len(self._null_vectors):
                null = self._null_vectors[level]
            else:
                reference = self._reference_field(current)
                null = self._random_null(current, nvec, reference.dtype, reference.device)
            transfer = QudaTransfer(
                null_vectors=null, fine_shape=current.shape,
                fine_spin=current.spin, fine_color=current.color,
                coarse_spin=2, block_size=self._block_sizes[level],
                n_block_ortho=self.n_block_ortho, verbose=self.verbose)
            coarse = QudaCoarseOperator(
                transfer, current,
                materialize=self.materialize_coarse,
                max_materialize_elements=self.max_materialize_elements,
                verbose=self.verbose)
            self.transfers.append(transfer)
            self.operators.append(coarse)
            self.levels.append(_Level(
                index=level, operator=current, transfer=transfer,
                spin=current.spin, color=current.color, shape=current.shape))
            current = coarse
        self.levels.append(_Level(
            index=self._transition_count, operator=current, transfer=None,
            spin=current.spin, color=current.color, shape=current.shape))
        if self.use_parity:
            self.parity_operators = [ParitySchurOperator(op)
                                     for op in self.operators]
        self._setup_done = True
        return self

    init = setup

    @staticmethod
    def _reference_field(operator: Any) -> Tensor:
        if isinstance(operator, _FineOperator) and operator._diagonal is not None:
            dtype = operator._diagonal.dtype
            device = operator._diagonal.device
        elif isinstance(operator, QudaCoarseOperator):
            dtype = operator.transfer.V.dtype
            device = operator.transfer.V.device
        else:
            dtype = device = None
        if dtype is None:
            # coarse transfer 总能从上层 V 找到 dtype/device；fine 无 diagonal
            # 的用户 callable 需要在第一次 solve 前给出 null_vectors。
            raise ValueError("随机 null vectors 需要显式 null_vectors 或 fine operator diagonal")
        return _torch.zeros(size=[operator.dof, *operator.shape], dtype=dtype, device=device)

    def apply(self, value: Tensor) -> Tensor:
        self.setup()
        flat, shape_kind = self._to_flat(value, self._fine)
        result = self.operators[0].apply(flat)
        return self._from_flat(result, shape_kind, self.fine_spin, self.fine_color)

    matvec = apply

    def _to_flat(self, value: Tensor, level_operator: Any) -> Tuple[Tensor, str]:
        if value.ndim == 5 and int(value.shape[0]) == level_operator.dof:
            return value, "flat"
        if (value.ndim == 6 and int(value.shape[0]) == level_operator.spin and
                int(value.shape[1]) == level_operator.color):
            return value.reshape(level_operator.dof, *level_operator.shape), "spin_color"
        raise ValueError(
            f"场应为 [{level_operator.dof},*shape] 或 "
            f"[{level_operator.spin},{level_operator.color},*shape]，得到 {tuple(value.shape)}")

    @staticmethod
    def _from_flat(value: Tensor, shape_kind: str, spin: int = 4,
                   color: int = 3) -> Tensor:
        if shape_kind == "flat":
            return value
        return value.reshape(spin, color, *value.shape[-4:])

    def _mr_smooth(self, level: int, rhs: Tensor, solution: Tensor,
                   steps: int) -> Tensor:
        operator = self.operators[level]
        x = solution
        for _ in range(max(0, steps)):
            residual = rhs - operator.apply(x)
            if self.use_parity:
                correction = self.parity_operators[level].mr_correction(residual, steps=1)
                x = x + correction
                continue
            image = operator.apply(residual)
            denominator = tools.vdot(image, image)
            if float(_torch.abs(denominator).item()) < 1e-30:
                break
            alpha = tools.vdot(image, residual) / denominator
            x = x + alpha * residual
        return x

    def _coarse_solve(self, level: int, rhs: Tensor) -> Tensor:
        operator = self.operators[level]
        if self.use_parity:
            # QUDA 在启用 MATPC 时把每层的粗求解也落到目标 parity 的
            # Schur 系统；这样 max_level=1 时 finest 层同样走完整的
            # parity 路径，而不是只在平滑阶段使用 checkerboard。
            return self.parity_operators[level].solve(
                rhs, tol=self.coarse_tol, max_iter=self.coarse_max_iter,
                restart=min(self.restart, max(1, self.coarse_max_iter)),
                verbose=self.verbose)
        n = int(rhs.numel())
        if (isinstance(operator, QudaCoarseOperator) and
                operator.blocks is not None and n <= self.direct_solve_max):
            dense = operator.dense_matrix
            solution = _torch.linalg_solve(dense, rhs.reshape(-1))
            return solution.reshape(rhs.shape)
        from ._gmres import fgmres
        return fgmres(rhs, operator.apply, tol=self.coarse_tol,
                      max_iter=self.coarse_max_iter,
                      restart=min(self.restart, max(1, self.coarse_max_iter)),
                      if_rtol=True, verbose=self.verbose)

    def v_cycle(self, rhs: Tensor, level: int = 0) -> Tensor:
        self.setup()
        operator = self.operators[level]
        if level == len(self.operators) - 1:
            return self._coarse_solve(level, rhs)
        x = _torch.zeros_like(rhs)
        x = self._mr_smooth(level, rhs, x, self.nu_pre)
        residual = rhs - operator.apply(x)
        coarse_rhs = self.transfers[level].restrict(residual)
        coarse_error = self.v_cycle(coarse_rhs, level + 1)
        x = x + self.transfers[level].prolong(coarse_error)
        x = self._mr_smooth(level, rhs, x, self.nu_post)
        return x

    def solve(self, b: Tensor, x0: Optional[Tensor] = None) -> Tensor:
        self.setup()
        b_flat, shape_kind = self._to_flat(b, self._fine)
        guess = None
        if x0 is not None:
            guess, guess_kind = self._to_flat(x0, self._fine)
            if guess_kind != shape_kind:
                raise ValueError("b 与 x0 的 spin/color 布局必须一致")
        from ._gmres import fgmres
        history: List[float] = []
        solution = fgmres(
            b_flat, self.operators[0].apply,
            tol=self.tol, max_iter=self.max_iter, restart=self.restart,
            x0=guess,
            precond=lambda residual: self.v_cycle(residual),
            if_rtol=True, verbose=self.verbose, history=history)
        self.convergence_history = history
        if shape_kind == "spin_color":
            return solution.reshape(self.fine_spin, self.fine_color, *self.fine_shape)
        return solution

    def solve_parity(self, b: Tensor, tol: Optional[float] = None,
                     max_iter: Optional[int] = None) -> Tensor:
        """直接以 finest Schur 求解并重构 full 解，用于 parity 对照验证。"""
        self.setup()
        if not self.use_parity:
            raise RuntimeError("solve_parity 要求 use_parity=True")
        b_flat, shape_kind = self._to_flat(b, self._fine)
        schur = self.parity_operators[0]
        from ._gmres import fgmres
        odd_rhs = schur.rhs(b_flat)
        odd = fgmres(
            odd_rhs, schur.apply,
            tol=self.tol if tol is None else float(tol),
            max_iter=self.max_iter if max_iter is None else int(max_iter),
            restart=self.restart, if_rtol=True, verbose=self.verbose)
        result = schur.reconstruct(b_flat, odd)
        if shape_kind == "spin_color":
            return result.reshape(self.fine_spin, self.fine_color, *self.fine_shape)
        return result

    def diagnostics(self, seed_field: Optional[Tensor] = None) -> Dict[str, float]:
        """执行小格验证所需的 P/R、Galerkin、伴随和 Schur 诊断。"""
        self.setup()
        if not self.transfers:
            raise RuntimeError("至少需要一个 coarse 层才能运行 MG diagnostics")
        if seed_field is None:
            if self._fine._diagonal is not None:
                dtype = self._fine._diagonal.dtype
                device = self._fine._diagonal.device
            else:
                dtype = self.transfers[0].V.dtype
                device = self.transfers[0].V.device
            seed_field = _torch.randn(
                size=[self.fine_dof, *self.fine_shape],
                dtype=dtype, device=device)
        fine = seed_field
        result: Dict[str, float] = {}
        transfer = self.transfers[0]
        coarse = _torch.randn(
            size=[transfer.coarse_dof, *transfer.coarse_shape],
            dtype=fine.dtype, device=fine.device)
        result["transfer_RP"] = _relative_norm(
            transfer.restrict(transfer.prolong(coarse)) - coarse, coarse)
        lhs = tools.vdot(transfer.prolong(coarse), fine)
        rhs = tools.vdot(coarse, transfer.restrict(fine))
        result["transfer_adjoint"] = float(
            _torch.abs(lhs - rhs).item()) / max(
                float(_torch.norm(transfer.prolong(coarse)).item()) *
                float(_torch.norm(fine).item()), 1e-30)
        result["transfer_projection"] = _relative_norm(
            transfer.prolong(transfer.restrict(fine)), fine)
        coarse_operator = self.operators[1]
        direct = transfer.restrict(self.operators[0].apply(transfer.prolong(coarse)))
        result["galerkin_RDP"] = _relative_norm(coarse_operator.apply(coarse) - direct, direct)
        if self.use_parity:
            result["transfer_block_ortho"] = transfer.orthogonality_error()
            # Schur 的重构误差使用一个随机 full rhs；只测试块代数，不要求
            # 外层迭代收敛。
            schur = self.parity_operators[0]
            odd_trial = _torch.randn(
                size=[self.fine_dof, schur.checkerboard.volume],
                dtype=fine.dtype, device=fine.device)
            reconstructed = schur.reconstruct(fine, odd_trial)
            residual = fine - self.operators[0].apply(reconstructed)
            result["schur_reconstruct_even"] = float(
                _torch.norm(schur.checkerboard.extract(residual, 0)).item())
        return result


# 公开别名：保留小写风格以便与旧 solver.multigrid 并列，也提供 Python
# 常见的 CamelCase 名称。旧类本身不被替换。
quda_multigrid = QudaMultigrid
QUDAMultigrid = QudaMultigrid


__all__ = [
    "QudaTransfer", "QudaCoarseOperator", "Checkerboard",
    "ParitySchurOperator", "QudaMultigrid", "quda_multigrid",
    "QUDAMultigrid",
]
