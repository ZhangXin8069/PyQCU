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

显式 ``setup_operator="schur"`` 时，首层采用 QUDA MATPC 的紧凑 odd
布局 ``[dof, X, Y, Z, T/2]``，其后的层级直接对 ``R S_o P`` 做 Galerkin
粗化；这条新路径与保留的旧 ``use_parity=True`` full-coarse 实现并列。

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
from pyqcu import dslash, lattice, tools


Tensor = Any
Coord = Tuple[int, int, int, int]
Shape4 = Tuple[int, int, int, int]
BlockKey = Tuple[int, int, int, int]
_QCU_DIAGONAL_PAIRS = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
)


class QcuStrictAssetBinding:
    """拥有 strict CUDA 资产并把稳定指针绑定到 ``set_ptrs``。

    绑定对象的生命周期必须覆盖所有后端调用。默认只持有运行期必需的
    ``Yhat/(X,X^-1)`` 以及递归层间 ``V``；raw ``Y`` 留作 setup/诊断资产，
    不进入求解期显存。``close`` 只清除仍指向本对象的槽，避免覆盖后来者。
    """

    def __init__(self, set_ptrs: Tensor, assets: Sequence[Dict[str, Any]],
                 start_level: int = 1, retain_raw_links: bool = False):
        from pyqcu.cuda import define

        self.set_ptrs = set_ptrs
        self.start_level = int(start_level)
        self.retain_raw_links = bool(retain_raw_links)
        self._owned: List[Tensor] = []
        self._bound: List[Tuple[int, int]] = []
        self._closed = False
        self.omitted_raw_bytes = 0

        level_count = len(assets) + 1
        if (level_count < 2 or level_count > 5 or self.start_level < 1 or
                self.start_level >= level_count):
            raise ValueError(
                "strict QCU binding 要求 2..5 层且 start_level 指向 coarse 层")
        if int(set_ptrs.numel()) < define._SET_PTRS_SIZE_:
            raise ValueError("set_ptrs 长度不足 strict 四槽资产区")
        if getattr(set_ptrs.device, "type", str(set_ptrs.device)) != "cpu":
            raise ValueError("set_ptrs 必须是 CPU int64 张量")

        first_transition = self.start_level - 1
        for transition in range(first_transition, len(assets)):
            base = (define._SET_PTRS_STRICT_COARSE_BASE_ +
                    transition * define._SET_PTRS_STRICT_STRIDE_)
            for slot in range(define._SET_PTRS_STRICT_STRIDE_):
                set_ptrs[base + slot] = 0

        def bind(slot: int, tensor: Optional[Tensor], label: str) -> None:
            if tensor is None:
                raise ValueError(f"strict QCU 资产 {label} 缺失")
            if not tensor.is_contiguous():
                raise ValueError(f"strict QCU 资产 {label} 必须 contiguous")
            pointer = int(tensor.data_ptr())
            if pointer == 0:
                raise ValueError(f"strict QCU 资产 {label} 指针为空")
            set_ptrs[slot] = pointer
            self._owned.append(tensor)
            self._bound.append((slot, pointer))

        for transition in range(first_transition, len(assets)):
            asset = assets[transition]
            base = (define._SET_PTRS_STRICT_COARSE_BASE_ +
                    transition * define._SET_PTRS_STRICT_STRIDE_)
            preconditioned = asset.get("preconditioned_links")
            onsite = asset.get("onsite_pair")
            bind(base + define._SET_PTRS_STRICT_PRECONDITIONED_LINKS_,
                 preconditioned, f"transition {transition} Yhat")
            bind(base + define._SET_PTRS_STRICT_ONSITE_PAIR_,
                 onsite, f"transition {transition} onsite_pair")
            raw = asset.get("raw_links")
            if self.retain_raw_links:
                bind(base + define._SET_PTRS_STRICT_RAW_LINKS_, raw,
                     f"transition {transition} raw Y")
            else:
                # raw Y 与 Yhat 形状/精度完全相同，可由后者精确计量省下的显存。
                self.omitted_raw_bytes += (
                    int(preconditioned.numel()) *
                    int(preconditioned.element_size()))
            # C++ 在 level L 递归到 L+1 时读取 transition L 的 V。
            if transition >= self.start_level:
                bind(base + define._SET_PTRS_STRICT_NULL_,
                     asset.get("null_vectors"),
                     f"transition {transition} null_vectors")

        unique = {int(tensor.data_ptr()): tensor for tensor in self._owned}
        self.resident_bytes = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in unique.values())

    @property
    def closed(self) -> bool:
        return self._closed

    def memory_report(self) -> Dict[str, int]:
        return {
            "resident_bytes": int(self.resident_bytes),
            "omitted_raw_bytes": int(self.omitted_raw_bytes),
            "bound_tensor_count": len(self._owned),
        }

    def close(self) -> None:
        if self._closed:
            return
        for slot, pointer in self._bound:
            if int(self.set_ptrs[slot]) == pointer:
                self.set_ptrs[slot] = 0
        self._bound.clear()
        self._owned.clear()
        self._closed = True

    def __enter__(self) -> "QcuStrictAssetBinding":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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


def _roll_field_batch(field: Tensor, displacement: BlockKey) -> Tensor:
    """Batched variant for ``[B,dof,X,Y,Z,T]`` fields."""
    result = field
    for dim, delta in enumerate(displacement):
        if delta:
            result = _torch.roll(result, shifts=-delta, dims=2 + dim)
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


def _matvec_block_batch(matrix: Tensor, vector: Tensor) -> Tensor:
    """Site-matrix action on ``[B,dof,X,Y,Z,T]`` fields."""
    return _torch.einsum("ijxyzt,bjxyzt->bixyzt", matrix, vector)


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


def _global_orthogonalise(vectors: Tensor, normalize: bool = True) -> Tensor:
    """对一组全场向量做重复 CGS 的全局正交化。

    QUDA 在 ``generateNullVectors`` 中可以先做全局正交化，setup 求解后
    默认还会再做一次；这与 ``QudaTransfer`` 内部的 aggregate-local
    正交化是两个不同层次。这里使用 ``tools.vdot``，因此在 MPI 运行时
    内积仍然是全局内积，而不是只在当前 rank 上正交化。
    """
    if vectors.ndim < 2:
        raise ValueError("null vectors 至少需要 [Nvec, ...] 两个维度")
    result = vectors.clone()
    eps = 1e-30
    for current_index in range(int(result.shape[0])):
        vector = result[current_index].clone()
        for previous_index in range(current_index):
            coefficient = tools.vdot(result[previous_index], vector)
            vector = vector - coefficient * result[previous_index]
        if normalize:
            norm = float(tools.norm(vector))
            if norm <= eps:
                raise ValueError(
                    f"null vector {current_index} 在全局正交化后退化，无法归一化")
            vector = vector / norm
        result[current_index] = vector
    return result


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

    def to_qcu_blocked(self, dtype: Any = None, device: Any = None) -> Tensor:
        """导出 C++ QCU transfer kernel 所需的 blocked ``V`` 布局。

        Python 参考实现把正交化后的基存为
        ``[fine_spin, fine_color, coarse_spin, nvec, X, Y, Z, T]``，而
        ``multigrid_restrict/prolong`` 按 C-order 读取
        ``[E, e, Xc, bx, Yc, by, Zc, bz, Tc, bt]``。这里的 ``E`` 是
        ``coarse_spin*nvec``，``e`` 是 ``fine_spin*fine_color``；只做
        轴重排和每个物理轴的拆分，不改变数值或共轭约定。

        返回值始终 contiguous，可直接作为
        ``qcu.applyMultigridRestrictQcu``/
        ``qcu.applyMultigridProLongQcu`` 的 ``null_vecs`` 参数。默认保留
        正交化基的 dtype/device；显式传入时用于跨精度或跨设备导出。
        """
        cx, cy, cz, ct = self.coarse_shape
        bx, by, bz, bt = self.block_size
        blocked = self.V.permute(2, 3, 0, 1, 4, 5, 6, 7).reshape(
            self.coarse_dof, self.fine_dof,
            cx, bx, cy, by, cz, bz, ct, bt)
        if dtype is not None or device is not None:
            blocked = blocked.to(
                dtype=blocked.dtype if dtype is None else dtype,
                device=blocked.device if device is None else device)
        return blocked.contiguous()

    @property
    def qcu_blocked_shape(self) -> Tuple[int, ...]:
        """返回 ``to_qcu_blocked`` 的静态 shape，便于 C++ ABI 守卫。"""
        cx, cy, cz, ct = self.coarse_shape
        bx, by, bz, bt = self.block_size
        return (self.coarse_dof, self.fine_dof,
                cx, bx, cy, by, cz, bz, ct, bt)

    # 语义别名：调用方可按“导出 QCU null vector”或“转换布局”理解。
    as_qcu_null_vectors = to_qcu_blocked

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

    def prolong_parity(self, coarse: Tensor, parity: int) -> Tensor:
        """把完整 coarse 场延拓到指定 fine parity。

        这对应 QUDA ``Transfer::setSiteSubset(PARITY, parity)`` 后的 ``P``：
        ``V`` 与输入 coarse 场仍定义在完整格上，仅输出 fine 场是单 parity。
        返回 checkerboard 紧凑布局 ``[fine_dof, fine_volume/2]``。
        """
        parity = int(parity)
        if parity not in (0, 1):
            raise ValueError(f"parity 必须是 0/1，得到 {parity}")
        checkerboard = Checkerboard(self.fine_shape)
        return checkerboard.extract(self.prolong(coarse), parity)

    def restrict_parity(self, fine: Tensor, parity: int) -> Tensor:
        """把指定 fine parity 限制到完整 coarse 场。

        ``fine`` 可为完整 ``[fine_dof,*fine_shape]``，也可为 checkerboard
        紧凑布局 ``[fine_dof,fine_volume/2]``。完整输入只读取指定 parity；
        输出始终为 ``[coarse_dof,*coarse_shape]``，不会永久缩减粗格几何。
        """
        parity = int(parity)
        if parity not in (0, 1):
            raise ValueError(f"parity 必须是 0/1，得到 {parity}")
        checkerboard = Checkerboard(self.fine_shape)
        if (fine.ndim == 2 and int(fine.shape[0]) == self.fine_dof and
                int(fine.shape[1]) == checkerboard.volume):
            full = checkerboard.embed(fine, parity, self.fine_dof)
        elif (fine.ndim == 5 and int(fine.shape[0]) == self.fine_dof and
              tuple(int(x) for x in fine.shape[-4:]) == self.fine_shape):
            compact = checkerboard.extract(fine, parity)
            full = checkerboard.embed(compact, parity, self.fine_dof)
        else:
            raise ValueError(
                "single-parity fine field 应为 "
                f"[{self.fine_dof},{checkerboard.volume}] 或 "
                f"[{self.fine_dof},*{self.fine_shape}]，得到 {tuple(fine.shape)}")
        return self.restrict(full)

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
                 adjoint: Optional[Callable[[Tensor], Tensor]] = None,
                 batch_matvec: Optional[Callable[[Tensor], Tensor]] = None):
        self._matvec = matvec
        self.shape = shape
        self.spin = int(spin)
        self.color = int(color)
        self.dof = self.spin * self.color
        self._diagonal = diagonal
        self._adjoint = adjoint
        self._batch_matvec = batch_matvec
        self._diagonal_inv: Dict[Tuple[Any, Any], Tensor] = {}

    def apply(self, value: Tensor) -> Tensor:
        return _call_matvec(self._matvec, value)

    matvec = apply

    def batch_apply(self, value: Tensor) -> Tensor:
        expected = (self.dof, *self.shape)
        if value.ndim != 6 or tuple(int(x) for x in value.shape[1:]) != expected:
            raise ValueError(
                f"batch fine field 应为 [B,{self.dof},*{self.shape}]，"
                f"得到 {tuple(value.shape)}")
        if self._batch_matvec is None:
            return _torch.stack([self.apply(item) for item in value], dim=0)
        result = _call_matvec(self._batch_matvec, value)
        if tuple(int(x) for x in result.shape) != tuple(int(x) for x in value.shape):
            raise ValueError("fine_batch_matvec 必须保持输入 shape")
        return result

    matvec_batch = batch_apply

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

    def diagonal_inv_apply_batch(self, value: Tensor) -> Tensor:
        dtype = value.dtype
        device = value.device
        matrix = self.diagonal_inverse(value)
        work = value
        if work.dtype != matrix.dtype or work.device != matrix.device:
            work = work.to(dtype=matrix.dtype, device=matrix.device)
        return _matvec_block_batch(matrix, work).to(dtype=dtype, device=device)

    def diagonal_adjoint_apply(self, value: Tensor) -> Tensor:
        """应用局部对角块的伴随。"""
        dtype = value.dtype
        device = value.device
        matrix = _adjoint_site(self.diagonal(value))
        work = value
        if work.dtype != matrix.dtype or work.device != matrix.device:
            work = work.to(dtype=matrix.dtype, device=matrix.device)
        return _matvec_block(matrix, work).to(dtype=dtype, device=device)

    def diagonal_inverse_adjoint_apply(self, value: Tensor) -> Tensor:
        """应用 ``(D_diag^{-1})^dagger``，用于 compact Schur 的伴随。"""
        dtype = value.dtype
        device = value.device
        matrix = _adjoint_site(self.diagonal_inverse(value))
        work = value
        if work.dtype != matrix.dtype or work.device != matrix.device:
            work = work.to(dtype=matrix.dtype, device=matrix.device)
        return _matvec_block(matrix, work).to(dtype=dtype, device=device)


class _LeftPreconditionedOperator:
    """完整场上的 ``X^{-1}D``，即 QUDA 粗化 PC 算子时使用的一阶对象。

    它仍作用于 full lattice；与 parity Schur ``I-Hhat_pq Hhat_qp`` 不同，
    支撑只是一跳，因此 ``R (X^{-1}D) P`` 仍可保存为粗层 ``X/Y``。
    """

    def __init__(self, operator: Any):
        self.operator = operator
        self.shape = _shape4(operator.shape)
        self.spin = int(operator.spin)
        self.color = int(operator.color)
        self.dof = int(operator.dof)

    def apply(self, value: Tensor) -> Tensor:
        if hasattr(self.operator, "preconditioned_full_apply"):
            return self.operator.preconditioned_full_apply(value)
        return self.operator.diagonal_inv_apply(
            _call_matvec(self.operator, value))

    matvec = apply
    preconditioned_full_apply = apply

    def batch_apply(self, value: Tensor) -> Tensor:
        if hasattr(self.operator, "preconditioned_full_apply_batch"):
            return self.operator.preconditioned_full_apply_batch(value)
        if hasattr(self.operator, "batch_apply"):
            image = self.operator.batch_apply(value)
        else:
            image = _torch.stack([
                _call_matvec(self.operator, item) for item in value], dim=0)
        if hasattr(self.operator, "diagonal_inv_apply_batch"):
            return self.operator.diagonal_inv_apply_batch(image)
        return _torch.stack([
            self.operator.diagonal_inv_apply(item) for item in image], dim=0)

    matvec_batch = batch_apply

    def adjoint_apply(self, value: Tensor) -> Tensor:
        if not hasattr(self.operator, "adjoint_apply"):
            raise RuntimeError("源算子未提供 adjoint")
        if not hasattr(self.operator, "diagonal_inverse_adjoint_apply"):
            raise RuntimeError("源算子未提供 diagonal inverse adjoint")
        transformed = self.operator.diagonal_inverse_adjoint_apply(value)
        return self.operator.adjoint_apply(transformed)

    def diagonal_apply(self, value: Tensor) -> Tensor:
        return value

    diagonal_inv_apply = diagonal_apply
    diagonal_adjoint_apply = diagonal_apply
    diagonal_inverse_adjoint_apply = diagonal_apply


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
        self.Y_backward_storage: Optional[List[Tensor]] = None
        self.Yhat_forward: Optional[List[Tensor]] = None
        self.Yhat_backward: Optional[List[Tensor]] = None
        self._strict_packed_assets: Optional[Dict[str, Any]] = None
        self._dense: Optional[Tensor] = None
        if materialize:
            self.build()

    def apply(self, value: Tensor) -> Tensor:
        if value.ndim != 5 or int(value.shape[0]) != self.dof:
            raise ValueError(
                f"粗场应为 [{self.dof}, X,Y,Z,T]，得到 {tuple(value.shape)}")
        if self.blocks is None:
            if self._strict_packed_assets is not None:
                return self.diagonal_apply(
                    self.preconditioned_full_apply(value))
            fine = self.transfer.prolong(value)
            result = _call_matvec(self.fine_operator, fine)
            return self.transfer.restrict(result)
        return self.apply_decomposed(value)

    matvec = apply

    def batch_apply(self, value: Tensor) -> Tensor:
        expected = (self.dof, *self.shape)
        if value.ndim != 6 or tuple(int(x) for x in value.shape[1:]) != expected:
            raise ValueError(
                f"batch coarse field 应为 [B,{self.dof},*{self.shape}]，"
                f"得到 {tuple(value.shape)}")
        if self.blocks is None:
            if self._strict_packed_assets is not None:
                return self.diagonal_apply_batch(
                    self.preconditioned_full_apply_batch(value))
            return _torch.stack([self.apply(item) for item in value], dim=0)
        dtype = value.dtype
        device = value.device
        work = value
        if (work.dtype != self.transfer.V.dtype or
                work.device != self.transfer.V.device):
            work = work.to(
                dtype=self.transfer.V.dtype, device=self.transfer.V.device)
        result = _torch.zeros_like(work)
        for displacement, block in self.blocks.items():
            result = result + _matvec_block_batch(
                block, _roll_field_batch(work, displacement))
        return result.to(dtype=dtype, device=device)

    matvec_batch = batch_apply

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

    def to_qcu_stencil(self, dtype: Any = None, device: Any = None,
                       strict: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        """导出 QCU 宽 33 点粗算子的 ``(sit, hop_nn, hop_diag)``。

        ``blocks[d]`` 的约定是
        ``(D_c x)(q) += blocks[d](q) x(q+d)``。QCU 宽核的存储顺序为：

        * ``sit[E,E,X,Y,Z,T]``：``d=(0,0,0,0)``；
        * ``hop_nn[2,4,E,E,X,Y,Z,T]``：``pm=0`` 读取 ``q+mu``，
          ``pm=1`` 读取 ``q-mu``；
        * ``hop_diag[2,2,6,E,E,X,Y,Z,T]``：符号 0/1 分别表示
          ``+1/-1``，pair 顺序为 ``xy,xz,xt,yz,yt,zt``。

        粗格某一维为 2 时，``+1`` 与 ``-1`` 指向同一邻点；同一系数会
        被平均分到所有等价槽位，因而 QCU kernel 的求和仍与
        ``blocks`` 完全一致。尺寸为 1 的维度直接折入等价的较低阶位移
        （全零位移进入 ``sit``）。

        ``strict=True`` 时，遇到 33 点模板无法表示的非零位移会抛出异常，
        避免把更宽的 Galerkin 支撑静默丢失。返回 tensor 均 contiguous，
        可直接传给 ``qcu.applyMultigridCoarseDslashWideQcu``；其顺序与
        ``CudaCoarseSchurOp`` 的 stencil 参数一致。
        """
        if self.blocks is None:
            self.build()
        assert self.blocks is not None

        base_dtype = self.transfer.V.dtype if dtype is None else dtype
        base_device = self.transfer.V.device if device is None else device
        shape = self.shape
        E = self.dof
        sit = _torch.zeros(
            size=[E, E, *shape], dtype=base_dtype, device=base_device)
        hop_nn = _torch.zeros(
            size=[2, 4, E, E, *shape], dtype=base_dtype, device=base_device)
        hop_diag = _torch.zeros(
            size=[2, 2, 6, E, E, *shape], dtype=base_dtype, device=base_device)
        pair_index = {pair: index for index, pair in
                      enumerate(_QCU_DIAGONAL_PAIRS)}

        def nonzero_block(block: Tensor) -> bool:
            # build() already filters at support_tol, but an explicitly supplied
            # blocks dictionary may contain zero/near-zero unsupported entries.
            return float(_torch.abs(block).max().item()) > self.support_tol

        for displacement, block in self.blocks.items():
            canonical = tuple(
                _signed_displacement(value, shape[dim])
                for dim, value in enumerate(displacement))
            if not nonzero_block(block):
                continue

            axes = [dim for dim, value in enumerate(canonical) if value != 0]
            if any(abs(canonical[dim]) != 1 for dim in axes) or len(axes) > 2:
                if strict:
                    raise ValueError(
                        "QCU 宽 33 点 stencil 无法表示非零位移 "
                        f"{displacement}（canonical={canonical}）")
                continue

            value = block.to(dtype=base_dtype, device=base_device)
            if not axes:
                # This also handles a dimension of extent one: all physical
                # shifts along that axis are the same site and are already
                # combined in the materialized block.
                sit += value
                continue

            # For an extent-two periodic dimension both signs are the same
            # physical neighbour.  Split one block across the duplicate QCU
            # sign slots; for larger dimensions the sign is unique.
            sign_options = []
            for dim in axes:
                if shape[dim] == 2:
                    sign_options.append((0, 1))
                else:
                    sign_options.append((0 if canonical[dim] == 1 else 1,))
            multiplicity = 1
            for options in sign_options:
                multiplicity *= len(options)
            value = value / multiplicity

            for choices in product(*sign_options):
                if len(axes) == 1:
                    hop_nn[choices[0], axes[0]] += value
                else:
                    first, second = axes
                    pair = pair_index[(first, second)]
                    hop_diag[choices[0], choices[1], pair] += value

        return sit.contiguous(), hop_nn.contiguous(), hop_diag.contiguous()

    # Short alias used by callers that treat the result as a packed asset.
    qcu_stencil = to_qcu_stencil

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

    def _validate_strict_nearest_neighbor_support(self) -> None:
        """Reject support that the strict QUDA X/Y ABI cannot represent."""
        if self.blocks is None:
            return
        for displacement, block in self.blocks.items():
            if float(_torch.abs(block).max().item()) <= self.support_tol:
                continue
            canonical = tuple(
                _signed_displacement(value, self.shape[dim])
                for dim, value in enumerate(displacement))
            axes = [dim for dim, value in enumerate(canonical) if value != 0]
            if (len(axes) > 1 or
                    any(abs(canonical[dim]) != 1 for dim in axes)):
                raise ValueError(
                    "strict QUDA X/Y 仅支持 onsite 与 ±axis 最近邻；"
                    f"检测到 displacement={displacement} "
                    f"(canonical={canonical})")

    def _build_links(self) -> None:
        if self.blocks is None or self.X is None:
            raise RuntimeError("coarse blocks 未构造")
        self._validate_strict_nearest_neighbor_support()
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
        self.Y_backward_storage = []
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
            self.Y_backward_storage.append(backward_storage)
            source_xinv = _roll_site_tensor(self.X_inv, tuple(
                -1 if i == dim else 0 for i in range(4)))
            self.Yhat_backward.append(_matmul_site(
                backward_storage, _adjoint_site(source_xinv)))

    def to_qcu_strict_assets(self, dtype: Any = None,
                             device: Any = None,
                             include_raw_links: bool = True
                             ) -> Dict[str, Any]:
        """导出 strict coarse level 的 QUDA ``X/Y/Yhat`` 四槽资产。

        返回张量均采用 C-order、时空轴在末尾：

        * ``raw_links[2,4,E,E,X,Y,Z,T]``：原始 ``Y``；
        * ``preconditioned_links[2,4,E,E,X,Y,Z,T]``：``Yhat``；
        * ``onsite_pair[2,E,E,X,Y,Z,T]``：依次为 ``X``、``X^{-1}``。

        ``pm=0`` 是目标点 ``q`` 的 forward link。``pm=1`` 是 QUDA 的
        backward-link 存储：对目标点 ``q`` 的后向动作读取 ``q-mu`` 处
        link 并取矩阵伴随。这样 CUDA 可直接复现 ``DiracCoarse`` 与
        ``DiracCoarsePC`` 的非 dagger gather 约定，而无须在 setup 后
        再转置整个场。
        """
        if (self._strict_packed_assets is not None and
                (not include_raw_links or
                 self._strict_packed_assets.get("raw_links") is not None)):
            cached = self._strict_packed_assets
            preconditioned = cached["preconditioned_links"]
            onsite = cached["onsite_pair"]
            target_dtype = preconditioned.dtype if dtype is None else dtype
            target_device = preconditioned.device if device is None else device
            return {
                "raw_links": (
                    cached.get("raw_links").to(
                        dtype=target_dtype, device=target_device).contiguous()
                    if include_raw_links else None),
                "preconditioned_links": preconditioned.to(
                    dtype=target_dtype, device=target_device).contiguous(),
                "onsite_pair": onsite.to(
                    dtype=target_dtype, device=target_device).contiguous(),
            }
        if (self.X is None or self.X_inv is None or
                self.Y_forward is None or self.Y_backward_storage is None or
                self.Yhat_forward is None or self.Yhat_backward is None):
            self.build()
        assert self.X is not None and self.X_inv is not None
        assert self.Y_forward is not None
        assert self.Y_backward_storage is not None
        assert self.Yhat_forward is not None and self.Yhat_backward is not None

        preconditioned_links = _torch.stack([
            _torch.stack(self.Yhat_forward, dim=0),
            _torch.stack(self.Yhat_backward, dim=0),
        ], dim=0)
        onsite_pair = _torch.stack([self.X, self.X_inv], dim=0)

        target_dtype = preconditioned_links.dtype if dtype is None else dtype
        target_device = preconditioned_links.device if device is None else device
        result: Dict[str, Any] = {
            "raw_links": None,
            "preconditioned_links": preconditioned_links.to(
                dtype=target_dtype, device=target_device).contiguous(),
            "onsite_pair": onsite_pair.to(
                dtype=target_dtype, device=target_device).contiguous(),
        }
        if include_raw_links:
            raw_links = _torch.stack([
                _torch.stack(self.Y_forward, dim=0),
                _torch.stack(self.Y_backward_storage, dim=0),
            ], dim=0)
            result["raw_links"] = raw_links.to(
                dtype=target_dtype, device=target_device).contiguous()
        return result

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

    def diagonal_apply_batch(self, value: Tensor) -> Tensor:
        if self.X is None:
            self.build()
        assert self.X is not None
        dtype = value.dtype
        device = value.device
        work = value
        if work.dtype != self.X.dtype or work.device != self.X.device:
            work = work.to(dtype=self.X.dtype, device=self.X.device)
        return _matvec_block_batch(
            self.X, work).to(dtype=dtype, device=device)

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

    def diagonal_inv_apply_batch(self, value: Tensor) -> Tensor:
        if self.X_inv is None:
            self.build()
        assert self.X_inv is not None
        dtype = value.dtype
        device = value.device
        work = value
        if work.dtype != self.X_inv.dtype or work.device != self.X_inv.device:
            work = work.to(dtype=self.X_inv.dtype, device=self.X_inv.device)
        return _matvec_block_batch(
            self.X_inv, work).to(dtype=dtype, device=device)

    def diagonal_adjoint_apply(self, value: Tensor) -> Tensor:
        if self.X is None:
            self.build()
        assert self.X is not None
        dtype = value.dtype
        device = value.device
        matrix = _adjoint_site(self.X)
        work = value
        if work.dtype != matrix.dtype or work.device != matrix.device:
            work = work.to(dtype=matrix.dtype, device=matrix.device)
        return _matvec_block(matrix, work).to(dtype=dtype, device=device)

    def diagonal_inverse_adjoint_apply(self, value: Tensor) -> Tensor:
        if self.X_inv is None:
            self.build()
        assert self.X_inv is not None
        dtype = value.dtype
        device = value.device
        matrix = _adjoint_site(self.X_inv)
        work = value
        if work.dtype != matrix.dtype or work.device != matrix.device:
            work = work.to(dtype=matrix.dtype, device=matrix.device)
        return _matvec_block(matrix, work).to(dtype=dtype, device=device)

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

    def preconditioned_apply_batch(self, value: Tensor) -> Tensor:
        """Batched ``X^{-1}(D-X)`` action for colored Galerkin setup."""
        if (self.X is None or self.Yhat_forward is None or
                self.Yhat_backward is None):
            self.build()
        assert self.Yhat_forward is not None and self.Yhat_backward is not None
        dtype = value.dtype
        device = value.device
        work = value
        if (work.dtype != self.transfer.V.dtype or
                work.device != self.transfer.V.device):
            work = work.to(
                dtype=self.transfer.V.dtype, device=self.transfer.V.device)
        result = _torch.zeros_like(work)
        for dim in range(4):
            displacement = tuple(1 if i == dim else 0 for i in range(4))
            result = result + _matvec_block_batch(
                self.Yhat_forward[dim],
                _roll_field_batch(work, displacement))
            storage_shift = tuple(1 if i == dim else 0 for i in range(4))
            backward_coefficient = _adjoint_site(_roll_site_tensor(
                self.Yhat_backward[dim], storage_shift))
            result = result + _matvec_block_batch(
                backward_coefficient,
                _roll_field_batch(
                    work, tuple(-x for x in displacement)))
        return result.to(dtype=dtype, device=device)

    def preconditioned_full_apply(self, value: Tensor) -> Tensor:
        """返回 ``X^{-1}D_c``，即局部项加上 PC hopping。"""
        # X^{-1} X is the identity, while preconditioned_apply contains
        # X^{-1}(D_c-X).
        return value + self.preconditioned_apply(value)

    def preconditioned_full_apply_batch(self, value: Tensor) -> Tensor:
        return value + self.preconditioned_apply_batch(value)



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


class QudaMatPCOperator:
    """QUDA ``DiracCoarsePC`` 的逐层对称 even/odd Schur 算子。

    对 ``D=X+H`` 先定义完整场左预处理算子
    ``Dhat=X^{-1}D=I+Hhat``，再在目标 parity ``p`` 上作用

    ``M_p = I - Hhat_pq Hhat_qp``，``q=1-p``。

    该对象只负责当前层的求解/平滑；其底层 ``operator`` 和下一层 transfer
    仍保留完整格几何。这正是 QUDA 将 ``DiracCoarse`` 与
    ``DiracCoarsePC`` 分开的语义。
    """

    def __init__(self, operator: Any, parity: int = 0):
        self.operator = operator
        self.shape = _shape4(operator.shape)
        self.spin = int(operator.spin)
        self.color = int(operator.color)
        self.dof = int(operator.dof)
        self.parity = int(parity)
        if self.parity not in (0, 1):
            raise ValueError(f"parity 必须是 0/1，得到 {self.parity}")
        self.other_parity = 1 - self.parity
        self.checkerboard = Checkerboard(self.shape)
        self._dense: Optional[Tensor] = None

    def _validate_compact(self, value: Tensor) -> Tensor:
        if (value.ndim != 2 or int(value.shape[0]) != self.dof or
                int(value.shape[1]) != self.checkerboard.volume):
            raise ValueError(
                f"MATPC field 应为 [{self.dof},{self.checkerboard.volume}]，"
                f"得到 {tuple(value.shape)}")
        return value

    def _preconditioned_full(self, value: Tensor) -> Tensor:
        if hasattr(self.operator, "preconditioned_full_apply"):
            return self.operator.preconditioned_full_apply(value)
        return self.operator.diagonal_inv_apply(
            _call_matvec(self.operator, value))

    def _hopping(self, compact: Tensor, source_parity: int,
                 target_parity: int) -> Tensor:
        """应用 ``X_target^{-1} H_target,source``。"""
        full = self.checkerboard.embed(
            self._validate_compact(compact), source_parity, self.dof)
        image = self._preconditioned_full(full)
        return self.checkerboard.extract(image, target_parity)

    def apply(self, value: Tensor) -> Tensor:
        value = self._validate_compact(value)
        first = self._hopping(
            value, self.parity, self.other_parity)
        second = self._hopping(
            first, self.other_parity, self.parity)
        return value - second

    matvec = apply

    def adjoint_apply(self, value: Tensor) -> Tensor:
        """参考路径的精确伴随；小格 setup 通过显式 MATPC 矩阵实现。"""
        value = self._validate_compact(value)
        dense = self.dense_matrix
        return (dense.conj().transpose(0, 1) @ value.reshape(-1)).reshape_as(value)

    @property
    def dense_matrix(self) -> Tensor:
        if self._dense is not None:
            return self._dense
        reference = self.reference_field()
        n = int(reference.numel())
        dense = _torch.zeros(
            size=[n, n], dtype=reference.dtype, device=reference.device)
        for column in range(n):
            probe = _torch.zeros_like(reference)
            probe.reshape(-1)[column] = 1.0
            dense[:, column] = self.apply(probe).reshape(-1)
        self._dense = dense
        return dense

    def rhs(self, full_rhs: Tensor) -> Tensor:
        """构造对称预处理右端 ``X_p^-1(b_p-H_pq X_q^-1 b_q)``。"""
        if (full_rhs.ndim != 5 or int(full_rhs.shape[0]) != self.dof or
                tuple(int(x) for x in full_rhs.shape[-4:]) != self.shape):
            raise ValueError(
                f"full rhs 应为 [{self.dof},*{self.shape}]，得到 {tuple(full_rhs.shape)}")
        b_target = self.checkerboard.extract(full_rhs, self.parity)
        b_other = self.checkerboard.extract(full_rhs, self.other_parity)

        other_full = self.checkerboard.embed(
            b_other, self.other_parity, self.dof)
        other_inverse = self.operator.diagonal_inv_apply(other_full)
        other_inverse_compact = self.checkerboard.extract(
            other_inverse, self.other_parity)
        correction = self._hopping(
            other_inverse_compact, self.other_parity, self.parity)

        target_full = self.checkerboard.embed(
            b_target, self.parity, self.dof)
        target_inverse = self.operator.diagonal_inv_apply(target_full)
        return (self.checkerboard.extract(target_inverse, self.parity) -
                correction)

    def reconstruct(self, full_rhs: Tensor, target_solution: Tensor) -> Tensor:
        """由目标 parity 解恢复另一 parity，并精确满足被消去方程。"""
        target_solution = self._validate_compact(target_solution)
        if (full_rhs.ndim != 5 or int(full_rhs.shape[0]) != self.dof or
                tuple(int(x) for x in full_rhs.shape[-4:]) != self.shape):
            raise ValueError(
                f"full rhs 应为 [{self.dof},*{self.shape}]，得到 {tuple(full_rhs.shape)}")

        target_full = self.checkerboard.embed(
            target_solution, self.parity, self.dof)
        coupling = self.checkerboard.extract(
            _call_matvec(self.operator, target_full), self.other_parity)
        b_other = self.checkerboard.extract(full_rhs, self.other_parity)
        other_rhs = self.checkerboard.embed(
            b_other - coupling, self.other_parity, self.dof)
        other_solution = self.operator.diagonal_inv_apply(other_rhs)

        result = target_full
        result.reshape(self.dof, -1)[:, self.checkerboard.indices[self.other_parity]] = (
            other_solution.reshape(self.dof, -1)[:,
                           self.checkerboard.indices[self.other_parity]])
        return result

    def reference_field(self) -> Tensor:
        if isinstance(self.operator, _FineOperator) and self.operator._diagonal is not None:
            dtype = self.operator._diagonal.dtype
            device = self.operator._diagonal.device
        elif isinstance(self.operator, QudaCoarseOperator):
            dtype = self.operator.transfer.V.dtype
            device = self.operator.transfer.V.device
        else:
            raise RuntimeError("无法推断 MATPC reference field 的 dtype/device")
        return _torch.zeros(
            size=[self.dof, self.checkerboard.volume],
            dtype=dtype, device=device)

    def solve(self, rhs: Tensor, tol: float = 1e-8,
              max_iter: int = 100, restart: int = 20,
              direct_solve_max: int = 4096,
              verbose: bool = False) -> Tensor:
        rhs = self._validate_compact(rhs)
        if int(rhs.numel()) <= int(direct_solve_max):
            solution = _torch.linalg_solve(self.dense_matrix, rhs.reshape(-1))
            return solution.reshape_as(rhs)
        from ._gmres import fgmres
        return fgmres(
            rhs, self.apply, tol=tol, max_iter=max_iter,
            restart=min(int(restart), max(1, int(max_iter))),
            if_rtol=True, verbose=verbose)


class CompactParityLayout:
    """把完整格点的一个 checkerboard 映射为 ``[X,Y,Z,T/2]``。

    PyQCU/QCU 的奇子格布局不是把完整场简单截掉一半，而是固定前三个
    坐标后沿最后一个坐标配对：

    ``t_full = 2*t_compact + ((parity - x - y - z) mod 2)``。

    该顺序与 ``tools.oooxyzt2poooxyzt`` 的 mask 展平顺序一致，因此既能
    连接完整 Wilson/Clover 算子，也能直接交给 QCU 的奇子格接口。
    """

    def __init__(self, full_shape: Sequence[int]):
        self.full_shape = _shape4(full_shape)
        if self.full_shape[-1] % 2:
            raise ValueError(
                "compact parity 布局要求最后一个格点维度为偶数，"
                f"得到 full_shape={self.full_shape}")
        self.compact_shape = (self.full_shape[0], self.full_shape[1],
                              self.full_shape[2], self.full_shape[3] // 2)
        self.indices: Dict[int, List[int]] = {}
        for parity in (0, 1):
            indices: List[int] = []
            for x, y, z, t_compact in _all_coords(self.compact_shape):
                t_full = 2 * t_compact + ((parity - x - y - z) & 1)
                indices.append(_site_index((x, y, z, t_full), self.full_shape))
            self.indices[parity] = indices

    @property
    def volume(self) -> int:
        return prod(self.compact_shape)

    def extract(self, field: Tensor, parity: int) -> Tensor:
        parity = int(parity)
        if parity not in self.indices:
            raise ValueError(f"parity 必须为 0 或 1，得到 {parity}")
        if field.ndim != 5 or tuple(int(x) for x in field.shape[1:]) != self.full_shape:
            raise ValueError(
                f"完整 field shape 应为 [dof,{self.full_shape}]，得到 {tuple(field.shape)}")
        return field.reshape(int(field.shape[0]), -1)[:, self.indices[parity]].reshape(
            int(field.shape[0]), *self.compact_shape)

    def extract_vectors(self, fields: Tensor, parity: int) -> Tensor:
        parity = int(parity)
        if parity not in self.indices:
            raise ValueError(f"parity 必须为 0 或 1，得到 {parity}")
        if fields.ndim != 6 or tuple(int(x) for x in fields.shape[-4:]) != self.full_shape:
            raise ValueError(
                f"完整 null vectors shape 应为 [Nvec,dof,{self.full_shape}]，"
                f"得到 {tuple(fields.shape)}")
        nvec, dof = int(fields.shape[0]), int(fields.shape[1])
        return fields.reshape(nvec * dof, -1)[:, self.indices[parity]].reshape(
            nvec, dof, *self.compact_shape)

    def embed(self, compact: Tensor, parity: int, dof: int) -> Tensor:
        parity = int(parity)
        if parity not in self.indices:
            raise ValueError(f"parity 必须为 0 或 1，得到 {parity}")
        if (compact.ndim != 5 or int(compact.shape[0]) != int(dof) or
                tuple(int(x) for x in compact.shape[1:]) != self.compact_shape):
            raise ValueError(
                f"compact field 应为 [{dof},{self.compact_shape}]，得到 {tuple(compact.shape)}")
        field = _torch.zeros(
            size=[int(dof), *self.full_shape], dtype=compact.dtype,
            device=compact.device)
        field.reshape(int(dof), -1)[:, self.indices[parity]] = compact.reshape(
            int(dof), -1)
        return field


class CompactParityOperator:
    """完整算子的单奇子格 Schur 补，采用 QCU 紧凑布局。

    给定完整块矩阵

    ``D = [[D_ee, D_eo], [D_oe, D_oo]]``，默认消去 even，暴露
    ``S_o = D_oo - D_oe D_ee^{-1} D_eo``，输入输出均为
    ``[dof, X, Y, Z, T/2]``。这与 ``dslash.operator.matvec_parity`` 和
    ``CudaSchurOp`` 的代数语义相同，但保留可供 Python 层 Galerkin
    transfer 使用的四维紧凑几何。
    """

    def __init__(self, operator: Any, parity: int = 1):
        self.operator = operator
        self.full_shape = _shape4(operator.shape)
        self.layout = CompactParityLayout(self.full_shape)
        self.parity = int(parity)
        if self.parity not in (0, 1):
            raise ValueError(f"parity 必须为 0 或 1，得到 {parity}")
        self.eliminated_parity = 1 - self.parity
        self.shape = self.layout.compact_shape
        self.dof = int(operator.dof)
        # 以 spin=1/color=dof 进入 QudaTransfer，避免再次引入 coarse
        # chirality block；compact Wilson Schur 的 12 个分量是一个整体。
        self.spin = 1
        self.color = self.dof

    def _validate(self, value: Tensor) -> Tensor:
        if value.ndim != 5 or int(value.shape[0]) != self.dof or tuple(
                int(x) for x in value.shape[1:]) != self.shape:
            raise ValueError(
                f"compact field 应为 [{self.dof},{self.shape}]，得到 {tuple(value.shape)}")
        return value

    def _diag_adjoint_apply(self, value: Tensor) -> Tensor:
        if hasattr(self.operator, "diagonal_adjoint_apply"):
            return self.operator.diagonal_adjoint_apply(value)
        matrix = _adjoint_site(self.operator.diagonal(value))
        return _matvec_block(matrix, value)

    def _diag_inverse_adjoint_apply(self, value: Tensor) -> Tensor:
        if hasattr(self.operator, "diagonal_inverse_adjoint_apply"):
            return self.operator.diagonal_inverse_adjoint_apply(value)
        matrix = _adjoint_site(self.operator.diagonal_inverse(value))
        return _matvec_block(matrix, value)

    def _coupling(self, compact: Tensor, source_parity: int,
                  target_parity: int) -> Tensor:
        full = self.layout.embed(compact, source_parity, self.dof)
        return self.layout.extract(self.operator.apply(full), target_parity)

    def apply(self, value: Tensor) -> Tensor:
        value = self._validate(value)
        eliminated_image = self._coupling(
            value, self.parity, self.eliminated_parity)
        eliminated_full = self.layout.embed(
            eliminated_image, self.eliminated_parity, self.dof)
        eliminated_inverse = self.operator.diagonal_inv_apply(eliminated_full)
        cross = self.layout.extract(
            self.operator.apply(eliminated_inverse), self.parity)
        target_full = self.layout.embed(value, self.parity, self.dof)
        diagonal = self.layout.extract(
            self.operator.diagonal_apply(target_full), self.parity)
        return diagonal - cross

    matvec = apply

    def adjoint_apply(self, value: Tensor) -> Tensor:
        """应用 Schur 补的严格伴随，而非把 ``S`` 当作 Hermitian。"""
        value = self._validate(value)
        target_full = self.layout.embed(value, self.parity, self.dof)
        # (D_pq)^dagger v：完整 D^dagger 作用在 p 子格，取 q 分量。
        pq_adjoint = self.layout.extract(
            self.operator.adjoint_apply(target_full), self.eliminated_parity)
        pq_adjoint_full = self.layout.embed(
            pq_adjoint, self.eliminated_parity, self.dof)
        eliminated_inverse_adjoint = self._diag_inverse_adjoint_apply(
            pq_adjoint_full)
        result = self.layout.extract(
            self.operator.adjoint_apply(eliminated_inverse_adjoint),
            self.parity)
        diagonal_adjoint = self.layout.extract(
            self._diag_adjoint_apply(target_full), self.parity)
        return diagonal_adjoint - result

    Mdag = adjoint_apply

    def rhs(self, full_rhs: Tensor) -> Tensor:
        if (full_rhs.ndim != 5 or int(full_rhs.shape[0]) != self.dof or
                tuple(int(x) for x in full_rhs.shape[1:]) != self.full_shape):
            raise ValueError(
                f"完整 rhs 应为 [{self.dof},{self.full_shape}]，得到 {tuple(full_rhs.shape)}")
        target_rhs = self.layout.extract(full_rhs, self.parity)
        eliminated_rhs = self.layout.extract(full_rhs, self.eliminated_parity)
        eliminated_full = self.layout.embed(
            eliminated_rhs, self.eliminated_parity, self.dof)
        eliminated_inverse = self.operator.diagonal_inv_apply(eliminated_full)
        correction = self.layout.extract(
            self.operator.apply(eliminated_inverse), self.parity)
        return target_rhs - correction

    def reconstruct(self, full_rhs: Tensor, target_solution: Tensor) -> Tensor:
        target_solution = self._validate(target_solution)
        rhs = full_rhs
        if (rhs.ndim != 5 or int(rhs.shape[0]) != self.dof or
                tuple(int(x) for x in rhs.shape[1:]) != self.full_shape):
            raise ValueError(
                f"完整 rhs 应为 [{self.dof},{self.full_shape}]，得到 {tuple(rhs.shape)}")
        eliminated_rhs = self.layout.extract(rhs, self.eliminated_parity)
        target_full = self.layout.embed(target_solution, self.parity, self.dof)
        coupling = self.layout.extract(
            self.operator.apply(target_full), self.eliminated_parity)
        eliminated_input = self.layout.embed(
            eliminated_rhs - coupling, self.eliminated_parity, self.dof)
        eliminated_solution = self.layout.extract(
            self.operator.diagonal_inv_apply(eliminated_input),
            self.eliminated_parity)
        return (self.layout.embed(target_solution, self.parity, self.dof) +
                self.layout.embed(eliminated_solution,
                                  self.eliminated_parity, self.dof))

    def reference_field(self) -> Tensor:
        diagonal = getattr(self.operator, "_diagonal", None)
        if diagonal is None:
            raise RuntimeError(
                "compact parity 随机 null vectors 需要 fine_diagonal")
        return _torch.zeros(
            size=[self.dof, *self.shape], dtype=diagonal.dtype,
            device=diagonal.device)


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
    显式传入时默认视为已经准备好的向量。没有向量时按 ``nvec_list``
    生成随机起点；若显式选择 ``setup_method``，则会按 QUDA 的
    ``NULL_VECTOR_SETUP``/``TEST_VECTOR_SETUP`` 语义迭代更新这些起点。
    ``setup_method`` 支持 ``random``、``inverse``、``test``、``cg``、
    ``ca-cg``、``krylov``/``gcr``。其中 ``cg`` 与 ``ca-cg`` 默认作用于
    正规算子 ``D^dagger D``，其余 Krylov setup 默认作用于原始算子。
    ``setup_operator="schur"`` 会切换到 QUDA MATPC 风格的紧凑奇子格层级：
    首层先构造 ``S_o``，之后每层直接对当前粗 Schur 算子做 ``R S_o P``；
    此模式的 ``null_vectors`` 形状为 ``[Nvec, fine_dof, X,Y,Z,T/2]``，
    也接受完整场并自动抽取 odd 分量。``solve`` 仍接收完整 rhs，并在
    返回前重构 eliminated parity。
    ``dof_list`` 仍可使用旧项目的总 coarse DOF 约定，例如 ``[12,24,24]``；
    普通 full 模式会将粗层总 DOF 除以 2 得到每个 coarse-spin block 的
    ``Nvec``；compact Schur 模式则把每个条目直接作为粗自由度。
    ``materialize_coarse=True`` 是小格验证模式，会逐列显式保存 ``RDP`` 的
    位移块；大格点应关闭它以使用矩阵自由动作（并同时关闭 ``use_parity``，
    因为奇偶 Schur 需要已构造的每层 ``X/Y``）。
    """

    def __init__(self, U: Optional[Tensor] = None,
                 clover_term: Optional[Tensor] = None,
                 fine_matvec: Optional[Callable[[Tensor], Tensor]] = None,
                 fine_batch_matvec: Optional[Callable[[Tensor], Tensor]] = None,
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
                 setup_method: str = "random",
                 setup_iters: Optional[int] = None,
                 setup_tol: float = 5e-6,
                 setup_max_iter: int = 500,
                 setup_krylov: int = 4,
                 setup_operator: str = "auto",
                 setup_type: Optional[str] = None,
                 setup_pre_orthonormalize: bool = False,
                 setup_post_orthonormalize: bool = True,
                 direct_solve_max: int = 4096,
                 max_materialize_elements: int = 50_000_000,
                 strict_galerkin_mode: str = "column",
                 strict_galerkin_column_batch: int = 4,
                 strict_galerkin_projection_batch: int = 4,
                 strict_galerkin_max_workspace_bytes: Optional[int] = 512 << 20,
                 strict_galerkin_check_support: bool = True,
                 seed: int = 42, verbose: bool = False,
                 hierarchy_mode: str = "legacy",
                 coarse_grid_solution_type: str | Sequence[str] = "matpc",
                 smoother_solve_type: str | Sequence[str] = "direct_pc",
                 target_parity: int = 0):
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
        self.setup_method = self._normalise_setup_method(setup_method)
        if setup_iters is None:
            # 保持旧接口的默认行为：random 不做迭代；显式选择 setup
            # 算法时默认执行一次 QUDA 风格 setup。
            setup_iters = 0 if self.setup_method == "random" else 1
        self.setup_iters = int(setup_iters)
        self.setup_tol = float(setup_tol)
        self.setup_max_iter = int(setup_max_iter)
        self.setup_krylov = int(setup_krylov)
        self.setup_operator = self._normalise_setup_operator(setup_operator)
        self.setup_type = self._normalise_setup_type(setup_type)
        self.hierarchy_mode = self._normalise_hierarchy_mode(hierarchy_mode)
        self._strict_quda = self.hierarchy_mode == "strict"
        self.target_parity = int(target_parity)
        if self.target_parity not in (0, 1):
            raise ValueError(
                f"target_parity 必须是 0/1，得到 {self.target_parity}")
        # ``setup_operator='schur'`` 是旧 compact RSP 层级的入口。严格
        # QUDA 模式将 setup 算子选择与层级几何分离，始终保留 full fields。
        self._compact_parity = (
            self.setup_operator == "schur" and not self._strict_quda)
        if self._compact_parity and not self.use_parity:
            raise ValueError(
                "setup_operator='schur' 要求 use_parity=True，以固定 odd Schur 语义")
        if self._strict_quda and not self.use_parity:
            raise ValueError(
                "hierarchy_mode='strict' 要求 use_parity=True")
        self.setup_pre_orthonormalize = bool(setup_pre_orthonormalize)
        self.setup_post_orthonormalize = bool(setup_post_orthonormalize)
        if self.setup_iters < 0:
            raise ValueError("setup_iters 必须 >= 0")
        if self.setup_tol <= 0:
            raise ValueError("setup_tol 必须 > 0")
        if self.setup_max_iter <= 0:
            raise ValueError("setup_max_iter 必须 > 0")
        if self.setup_krylov <= 0:
            raise ValueError("setup_krylov 必须 > 0")
        self.direct_solve_max = int(direct_solve_max)
        self.max_materialize_elements = int(max_materialize_elements)
        if self.max_materialize_elements <= 0:
            raise ValueError("max_materialize_elements 必须为正数")
        galerkin_mode = str(strict_galerkin_mode).strip().lower().replace("_", "-")
        galerkin_aliases = {
            "auto": "auto",
            "column": "column",
            "legacy": "column",
            "site": "site-batch",
            "site-batch": "site-batch",
            "colored": "colored",
            "colour": "colored",
        }
        if galerkin_mode not in galerkin_aliases:
            raise ValueError(
                "strict_galerkin_mode 可选 auto/column/site-batch/colored")
        self.strict_galerkin_mode = galerkin_aliases[galerkin_mode]
        self.strict_galerkin_column_batch = int(strict_galerkin_column_batch)
        self.strict_galerkin_projection_batch = int(
            strict_galerkin_projection_batch)
        if min(self.strict_galerkin_column_batch,
               self.strict_galerkin_projection_batch) <= 0:
            raise ValueError("strict Galerkin batch size 必须为正数")
        self.strict_galerkin_max_workspace_bytes = (
            None if strict_galerkin_max_workspace_bytes is None else
            int(strict_galerkin_max_workspace_bytes))
        if (self.strict_galerkin_max_workspace_bytes is not None and
                self.strict_galerkin_max_workspace_bytes <= 0):
            raise ValueError(
                "strict_galerkin_max_workspace_bytes 必须为正数或 None")
        self.strict_galerkin_check_support = bool(
            strict_galerkin_check_support)
        self.seed = int(seed)
        self._setup_done = False
        self.setup_history: List[Dict[str, Any]] = []
        self._fine_adjoint_kind = "explicit" if fine_adjoint is not None else None

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
            if fine_batch_matvec is None:
                fine_batch_matvec = getattr(fine_dslash, "matvec_batch", None)
            diagonal = self._fine_diagonal_from_dslash(fine_dslash, U)
            if fine_adjoint is None and self.fine_spin == 4:
                # Wilson/Clover 满足 gamma5-Hermiticity：D^dagger = gamma5 D
                # gamma5。这样 CG/CA-CG setup 不需要用户重复写伴随算子。
                fine_adjoint = self._gamma5_hermitian_adjoint(
                    fine_matvec, self.fine_spin, self.fine_color,
                    self.fine_shape)
                self._fine_adjoint_kind = "gamma5"
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
            diagonal=diagonal, adjoint=fine_adjoint,
            batch_matvec=fine_batch_matvec)
        self._null_vectors = self._normalise_null_list(null_vectors)
        self._fine_full = self._fine
        self._fine_compact: Optional[CompactParityOperator] = None
        if self._compact_parity:
            self._fine_compact = CompactParityOperator(self._fine_full, parity=1)
            self._null_vectors = self._normalise_compact_null_list(
                self._null_vectors)
        self._hierarchy_fine = (
            self._fine_compact if self._fine_compact is not None else self._fine)
        self._nvec_list = self._infer_nvec_list(nvec_list, dof_list)
        self._block_sizes = self._normalise_block_sizes(block_size)
        transitions = self._number_of_transitions(max_level)
        self._transition_count = transitions
        if self._strict_quda:
            if self.fine_spin == 1:
                raise ValueError(
                    "strict standard-staggered/KD 语义尚未实现；当前 strict "
                    "入口只支持 Wilson/Clover 或 generic spin-2 算子")
            self._validate_strict_geometry(transitions)
        level_count = max(1, transitions + 1)
        self.coarse_grid_solution_types = self._normalise_level_modes(
            coarse_grid_solution_type, level_count, "coarse_grid_solution_type",
            {"mat": "mat", "full": "mat", "matpc": "matpc", "pc": "matpc"})
        self.smoother_solve_types = self._normalise_level_modes(
            smoother_solve_type, level_count, "smoother_solve_type",
            {"direct": "direct", "full": "direct",
             "direct-pc": "direct_pc", "direct_pc": "direct_pc",
             "pc": "direct_pc", "matpc": "direct_pc"})
        for level in range(level_count):
            if (self.coarse_grid_solution_types[level] == "matpc" and
                    self.smoother_solve_types[level] != "direct_pc"):
                raise ValueError(
                    "coarse_grid_solution_type='matpc' 要求同层 "
                    f"smoother_solve_type='direct_pc'（level={level}）")
            if (self._strict_quda and
                    (self.coarse_grid_solution_types[level] != "matpc" or
                     self.smoother_solve_types[level] != "direct_pc")):
                raise ValueError(
                    "当前 strict runtime 仅完整实现全层 "
                    "coarse_grid_solution_type='matpc' + "
                    "smoother_solve_type='direct_pc'；"
                    f"level={level} 得到 "
                    f"{self.coarse_grid_solution_types[level]}/"
                    f"{self.smoother_solve_types[level]}")
        if (self.use_parity and not self._compact_parity and transitions and
                not self.materialize_coarse):
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
        self.matpc_operators: List[QudaMatPCOperator] = []
        self.coarsening_operators: List[Any] = []
        self.strict_setup_stats: List[Dict[str, Any]] = []
        self._cuda_runtime_sealed = False
        self._cuda_runtime_seal_report: Dict[str, Any] = {}

    @staticmethod
    def _normalise_hierarchy_mode(mode: str) -> str:
        value = str(mode).strip().lower().replace("_", "-")
        aliases = {
            "legacy": "legacy",
            "compact": "legacy",
            "rsp": "legacy",
            "quda": "strict",
            "quda-strict": "strict",
            "strict": "strict",
        }
        if value not in aliases:
            raise ValueError(
                f"不支持的 hierarchy_mode={mode!r}；可选 legacy/strict")
        return aliases[value]

    @staticmethod
    def _normalise_level_modes(value: str | Sequence[str], count: int,
                               name: str,
                               aliases: Dict[str, str]) -> List[str]:
        raw = [value] if isinstance(value, str) else list(value)
        if not raw:
            raise ValueError(f"{name} 不能为空")
        if len(raw) == 1:
            raw = raw * count
        if len(raw) < count:
            raise ValueError(
                f"{name} 数量 {len(raw)} 少于层数 {count}")
        result: List[str] = []
        for item in raw[:count]:
            key = str(item).strip().lower().replace("_", "-")
            if key not in aliases:
                raise ValueError(
                    f"不支持的 {name}={item!r}；可选 {sorted(set(aliases.values()))}")
            result.append(aliases[key])
        return result

    def _validate_strict_geometry(self, transitions: int) -> None:
        """Apply QUDA's checkerboard-safe coarse-extent invariant."""
        shape = self.fine_shape
        for level in range(transitions):
            if level >= len(self._block_sizes):
                raise ValueError("block_size 数量少于层间过渡数")
            block = self._block_sizes[level]
            if any(extent % width for extent, width in zip(shape, block)):
                raise ValueError(
                    f"strict level {level}: shape={shape} 不能被 "
                    f"block_size={block} 整除")
            coarse = tuple(
                extent // width for extent, width in zip(shape, block))
            invalid = [
                (axis, extent) for axis, extent in enumerate(coarse)
                if extent < 2 or extent % 2
            ]
            if invalid:
                raise ValueError(
                    "strict QUDA 每次粗化后的各维 extent 必须为偶数且至少 2；"
                    f"level={level}, coarse_shape={coarse}, invalid={invalid}")
            shape = coarse  # type: ignore[assignment]

    @staticmethod
    def _normalise_setup_method(method: str) -> str:
        value = str(method).strip().lower().replace("_", "-")
        aliases = {
            "null": "inverse",
            "null-vector": "inverse",
            "null-vector-setup": "inverse",
            "bicgstab": "inverse",
            "test-vector": "test",
            "test-vector-setup": "test",
            "cacg": "ca-cg",
            "ca-cg": "ca-cg",
            "fgmres": "krylov",
        }
        value = aliases.get(value, value)
        allowed = {"random", "inverse", "test", "cg", "ca-cg", "krylov", "gcr"}
        if value not in allowed:
            raise ValueError(
                f"不支持的 setup_method={method!r}；可选值为 "
                "random/inverse/test/cg/ca-cg/krylov/gcr")
        return value

    @staticmethod
    def _normalise_setup_operator(operator: str) -> str:
        value = str(operator).strip().lower().replace("_", "-")
        aliases = {
            "d": "full",
            "dirac": "full",
            "normal-op": "normal",
            "dagger-d": "normal",
            "dd": "normal",
            "matpc": "schur",
            "pc": "schur",
        }
        value = aliases.get(value, value)
        if value not in {"auto", "full", "normal", "schur"}:
            raise ValueError(
                f"不支持的 setup_operator={operator!r}；可选值为 auto/full/normal/schur")
        return value

    @staticmethod
    def _normalise_setup_type(setup_type: Optional[str]) -> str:
        if setup_type is None:
            return "auto"
        value = str(setup_type).strip().lower().replace("_", "-")
        aliases = {
            "null-vector": "null",
            "null-vector-setup": "null",
            "test-vector": "test",
            "test-vector-setup": "test",
        }
        value = aliases.get(value, value)
        if value not in {"auto", "null", "test"}:
            raise ValueError(
                f"不支持的 setup_type={setup_type!r}；可选值为 auto/null/test")
        return value

    @staticmethod
    def _gamma5_apply(value: Tensor, spin: int, color: int,
                      shape: Shape4) -> Tensor:
        if spin != 4:
            raise ValueError("gamma5-Hermiticity 目前只支持 fine_spin=4")
        matrix = lattice.gamma_5.to(device=value.device, dtype=value.dtype)
        structured = value.reshape(spin, color, *shape)
        result = _torch.einsum("ab,bcxyzt->acxyzt", matrix, structured)
        return result.reshape(spin * color, *shape)

    @classmethod
    def _gamma5_hermitian_adjoint(cls, matvec: Callable[[Tensor], Tensor],
                                  spin: int, color: int,
                                  shape: Shape4) -> Callable[[Tensor], Tensor]:
        def adjoint(value: Tensor) -> Tensor:
            transformed = cls._gamma5_apply(value, spin, color, shape)
            result = _call_matvec(matvec, transformed)
            return cls._gamma5_apply(result, spin, color, shape)
        return adjoint

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

    def _normalise_compact_null_list(self, null_vectors: List[Tensor]) -> List[Tensor]:
        """把 compact/full 两种输入统一成 ``[Nvec,dof,X,Y,Z,T/2]``。"""
        if self._fine_compact is None:
            return null_vectors
        layout = self._fine_compact.layout
        normalised: List[Tensor] = []
        for index, vectors in enumerate(null_vectors):
            # Only the first transition is defined on the original full
            # lattice.  Vectors for later transitions already live on the
            # preceding compact coarse lattice and have a different DOF;
            # leave their per-level layout to QudaTransfer validation.
            if index != 0:
                normalised.append(vectors)
                continue
            value = vectors
            if value.ndim == 7:
                if tuple(int(x) for x in value.shape[1:3]) != (
                        self.fine_spin, self.fine_color):
                    raise ValueError(
                        f"compact null_vectors[{index}] 的 spin/color 不匹配："
                        f"得到 {tuple(value.shape[1:3])}，期望 "
                        f"({self.fine_spin},{self.fine_color})")
                value = value.reshape(
                    int(value.shape[0]), self.fine_dof, *value.shape[-4:])
            if value.ndim != 6 or int(value.shape[1]) != self.fine_dof:
                raise ValueError(
                    f"compact null_vectors[{index}] 必须为 [Nvec,{self.fine_dof},...]")
            tail = tuple(int(x) for x in value.shape[-4:])
            if tail == layout.compact_shape:
                normalised.append(value)
            elif tail == self.fine_shape:
                normalised.append(layout.extract_vectors(value, self._fine_compact.parity))
            else:
                raise ValueError(
                    f"compact null_vectors[{index}] 时空尺寸 {tail} 不匹配 "
                    f"compact={layout.compact_shape} 或 full={self.fine_shape}")
        return normalised

    def _infer_nvec_list(self, nvec_list: Optional[Sequence[int]],
                         dof_list: Optional[Sequence[int]]) -> List[int]:
        if nvec_list is not None:
            values = [int(x) for x in nvec_list]
        elif dof_list is not None:
            values = []
            for total in list(dof_list)[1:]:
                total = int(total)
                if self._compact_parity:
                    values.append(total)
                else:
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

    def _setup_solver_kind(self) -> str:
        if self.setup_method in {"inverse", "test"}:
            return "bicgstab"
        if self.setup_method == "cg":
            return "cg"
        if self.setup_method == "ca-cg":
            return "ca-cg"
        if self.setup_method in {"krylov", "gcr"}:
            return "fgmres"
        return "random"

    def _setup_operator_kind(self, solver_kind: str, level: int = 0) -> str:
        if self.setup_operator == "schur":
            # 首层已经是 compact S_o；其后的 operator 是在 odd 几何上
            # 递归生成的 Galerkin coarse operator，不应再次做 checkerboard。
            if level == 0:
                return "normal" if solver_kind in {"cg", "ca-cg"} else "schur"
            return "normal" if solver_kind in {"cg", "ca-cg"} else "full"
        if self.setup_operator == "normal":
            return "normal"
        if self.setup_operator == "full":
            return "full"
        # QUDA 的 CG/CA-CG setup 用 DiracMdagM；BiCGStab/GCR 直接作用于
        # 非厄米 Dirac 算子。auto 正是这一按 solver 类型的默认映射。
        return "normal" if solver_kind in {"cg", "ca-cg"} else "full"

    @staticmethod
    def _cg_setup_solve(rhs: Tensor, matvec: Callable[[Tensor], Tensor],
                        tol: float, max_iter: int,
                        x0: Optional[Tensor] = None) -> Tensor:
        """稳健的复数 CG，用于 ``D^dagger D`` null-vector setup。"""
        x = _torch.zeros_like(rhs) if x0 is None else x0.clone()
        residual = rhs - matvec(x)
        initial_norm = float(_torch.norm(residual).item())
        if initial_norm <= max(float(tol), 1e-30):
            return x
        direction = residual.clone()
        rr = tools.vdot(residual, residual)
        for _ in range(max(0, int(max_iter))):
            image = matvec(direction)
            denominator = tools.vdot(direction, image)
            if float(_torch.abs(denominator).item()) <= 1e-30:
                break
            alpha = rr / denominator
            x = x + alpha * direction
            residual = residual - alpha * image
            residual_norm = float(_torch.norm(residual).item())
            if residual_norm <= float(tol):
                break
            rr_next = tools.vdot(residual, residual)
            if float(_torch.abs(rr).item()) <= 1e-30:
                break
            beta = rr_next / rr
            direction = residual + beta * direction
            rr = rr_next
        return x

    def _setup_linear_solve(self, rhs: Tensor, matvec: Callable[[Tensor], Tensor],
                            solver_kind: str, tol: float) -> Tensor:
        """以统一的绝对阈值调用 setup solver。

        QUDA 对零右端的 NULL_VECTOR_SETUP 会以初始 ``||A B||`` 作为
        relative residual 的基准。这里把 null setup 改写成求解
        ``A delta = A B``，所以所有 solver 都接收非零右端和同一个绝对
        阈值；数值上与 ``B <- B-delta`` 完全等价，而且避免了通用
        FGMRES/CG 对零 RHS 的特殊早停路径。
        """
        if solver_kind == "cg":
            return self._cg_setup_solve(
                rhs, matvec, tol=tol, max_iter=self.setup_max_iter)
        if solver_kind == "ca-cg":
            from ._cacg import cacg
            try:
                return cacg(
                    rhs, matvec, tol=tol, max_iter=self.setup_max_iter,
                    x0=None, n_krylov=self.setup_krylov,
                    if_rtol=False, verbose=False)
            except RuntimeError:
                # 幂基在病态/低维 toy operator 上可能塌缩；CG 是同一正规
                # 算子的稳定 fallback，不能让 setup 因为表示方式失败。
                return self._cg_setup_solve(
                    rhs, matvec, tol=tol, max_iter=self.setup_max_iter)
        if solver_kind == "fgmres":
            from ._gmres import fgmres
            return fgmres(
                rhs, matvec, tol=tol, max_iter=self.setup_max_iter,
                restart=min(self.setup_krylov, max(1, self.setup_max_iter)),
                x0=None, if_rtol=False, verbose=False)

        from ._bistabcg import bistabcg
        try:
            return bistabcg(
                rhs, matvec, tol=tol, max_iter=self.setup_max_iter,
                x0=None, if_rtol=False, verbose=False)
        except RuntimeError:
            # BiCGStab 在精确一步收敛（例如单位阵）时会遇到 t=0 的
            # breakdown；QUDA 会把它视作 lucky convergence。用 GMRES
            # 完成同一个线性问题，保留 setup 的结果而不是丢弃该向量。
            from ._gmres import fgmres
            return fgmres(
                rhs, matvec, tol=tol, max_iter=self.setup_max_iter,
                restart=min(self.setup_krylov, max(1, self.setup_max_iter)),
                x0=None, if_rtol=False, verbose=False)

    def _setup_vectors(self, operator: Any, vectors: Tensor,
                       level: int, solver_kind: str,
                       operator_kind: str, setup_type: str) -> Tensor:
        """执行一个层级的 QUDA 风格 null-vector setup。"""
        if self.setup_iters == 0:
            return vectors

        if operator_kind == "normal":
            if (not hasattr(operator, "adjoint_apply") or
                    (isinstance(operator, _FineOperator) and
                     operator._adjoint is None)):
                raise RuntimeError(
                    "正规算子 setup 需要 fine_adjoint 或可构造的 coarse adjoint；"
                    "自定义 fine_matvec 不会自动假定 gamma5-Hermiticity")

            def apply_setup(value: Tensor) -> Tensor:
                image = _call_matvec(operator, value)
                return _call_matvec(operator.adjoint_apply, image)
        else:
            apply_setup = lambda value: _call_matvec(operator, value)

        current = vectors.clone()
        for iteration in range(self.setup_iters):
            if self.setup_pre_orthonormalize:
                current = _global_orthogonalise(current)
            updated = _torch.zeros_like(current)
            relative_residuals: List[float] = []
            for vector_index in range(int(current.shape[0])):
                basis = current[vector_index]
                field = basis.reshape(operator.dof, *operator.shape)
                if setup_type == "test":
                    rhs = field
                    rhs_norm = float(_torch.norm(rhs).item())
                    if rhs_norm <= 1e-30:
                        solution = _torch.zeros_like(field)
                        relative_residuals.append(0.0)
                    else:
                        threshold = self.setup_tol * rhs_norm
                        solution = self._setup_linear_solve(
                            rhs, apply_setup, solver_kind, threshold)
                        residual = rhs - apply_setup(solution)
                        relative_residuals.append(
                            float(_torch.norm(residual).item()) / rhs_norm)
                else:
                    # NULL_VECTOR_SETUP: solve A*delta=A*B from zero and
                    # retain the relaxed residual B-delta. This is the same
                    # initial-guess relaxation used by QUDA's solver.
                    image = apply_setup(field)
                    image_norm = float(_torch.norm(image).item())
                    if image_norm <= 1e-30:
                        solution = field.clone()
                        relative_residuals.append(0.0)
                    else:
                        threshold = self.setup_tol * image_norm
                        correction = self._setup_linear_solve(
                            image, apply_setup, solver_kind, threshold)
                        solution = field - correction
                        residual = image - apply_setup(correction)
                        relative_residuals.append(
                            float(_torch.norm(residual).item()) / image_norm)
                updated[vector_index] = solution.reshape_as(basis)
            current = updated
            if self.setup_post_orthonormalize:
                current = _global_orthogonalise(current)
            self.setup_history.append({
                "level": level,
                "iteration": iteration + 1,
                "method": self.setup_method,
                "solver": solver_kind,
                "operator": operator_kind,
                "setup_type": setup_type,
                "relative_residual": relative_residuals,
            })
        return current

    def setup(self) -> "QudaMultigrid":
        if self._cuda_runtime_sealed:
            raise RuntimeError(
                "hierarchy 已 seal 为 CUDA runtime；Python setup 资产已释放")
        if self._setup_done:
            return self
        # ``cann`` 负责屏蔽 NPU/复数差异；将种子也放在兼容层内，避免
        # 纯 Python 求解器重新直接依赖 torch。显式 null_vectors 时不会
        # 生成随机向量，但仍保持后续随机诊断的可复现起点。
        _torch.manual_seed(self.seed)
        self.setup_history = []
        solver_kind = self._setup_solver_kind()
        if solver_kind != "random":
            if self.setup_type == "auto":
                setup_type = "test" if self.setup_method == "test" else "null"
            else:
                setup_type = self.setup_type
        else:
            setup_type = "null"
        current: Any = self._hierarchy_fine
        self.operators = [current]
        self.levels = []
        self.transfers = []
        self.coarsening_operators = []
        self.strict_setup_stats = []
        for level in range(self._transition_count):
            nvec = self._nvec_list[level]
            if level < len(self._null_vectors):
                null = self._null_vectors[level].clone()
            else:
                reference = self._reference_field(current)
                null = self._random_null(current, nvec, reference.dtype, reference.device)
            if solver_kind != "random":
                operator_kind = self._setup_operator_kind(solver_kind, level=level)
                null = self._setup_vectors(
                    current, null, level, solver_kind, operator_kind, setup_type)
                # 保存 setup 后的逻辑向量，便于后续诊断/调用方检查；
                # QudaTransfer 内部还会按 aggregate 做一次局部正交化。
                if level < len(self._null_vectors):
                    self._null_vectors[level] = null
                else:
                    self._null_vectors.append(null)
            else:
                operator_kind = (
                    "schur" if self._compact_parity else "full")
            transfer = QudaTransfer(
                null_vectors=null, fine_shape=current.shape,
                fine_spin=current.spin, fine_color=current.color,
                coarse_spin=1 if self._compact_parity else 2,
                spin_block_size=1 if self._compact_parity else None,
                block_size=self._block_sizes[level],
                n_block_ortho=self.n_block_ortho, verbose=self.verbose)
            coarsening_operator: Any = current
            if (self._strict_quda and
                    self.coarse_grid_solution_types[level] == "matpc" and
                    self.smoother_solve_types[level] == "direct_pc"):
                # QUDA coarsens the full-field first-order operator X^{-1}D;
                # parity Schur is created separately for smoothing/solving.
                coarsening_operator = _LeftPreconditionedOperator(current)
            batched_strict_setup = (
                self._strict_quda and self.materialize_coarse and
                self.strict_galerkin_mode != "column")
            coarse = QudaCoarseOperator(
                transfer, coarsening_operator,
                materialize=(self.materialize_coarse and
                             not batched_strict_setup),
                max_materialize_elements=self.max_materialize_elements,
                verbose=self.verbose)
            if batched_strict_setup:
                from pyqcu.tools._strict_galerkin import (
                    build_strict_galerkin,
                    build_strict_galerkin_colored,
                    strict_galerkin_colored_memory_model,
                    strict_galerkin_memory_model,
                )
                if hasattr(coarsening_operator, "batch_apply"):
                    batch_apply = coarsening_operator.batch_apply
                else:
                    batch_apply = lambda values: _torch.stack([
                        _call_matvec(coarsening_operator, item)
                        for item in values
                    ], dim=0)
                setup_mode = self.strict_galerkin_mode
                if setup_mode == "auto":
                    site_batch = min(
                        self.strict_galerkin_projection_batch,
                        prod(transfer.coarse_shape))
                    common_model = {
                        "coarse_dof": transfer.coarse_dof,
                        "fine_dof": transfer.fine_dof,
                        "fine_shape": transfer.fine_shape,
                        "coarse_shape": transfer.coarse_shape,
                        "block_size": transfer.block_size,
                        "element_size": int(transfer.V.element_size()),
                        "include_raw_links": False,
                        "retain_blocks": False,
                    }
                    site_model = strict_galerkin_memory_model(
                        **common_model, site_batch_size=site_batch)
                    colored_model = strict_galerkin_colored_memory_model(
                        **common_model,
                        column_batch_size=min(
                            transfer.coarse_dof,
                            self.strict_galerkin_column_batch),
                        projection_site_batch_size=site_batch)
                    budget = self.strict_galerkin_max_workspace_bytes
                    site_fits = (
                        budget is None or
                        site_model["workspace_upper_bytes"] <= budget)
                    setup_mode = (
                        "site-batch" if site_fits and
                        site_model["operator_calls"] <=
                        colored_model["operator_calls"] else "colored")
                if setup_mode == "colored":
                    setup_result = build_strict_galerkin_colored(
                        transfer,
                        batch_apply,
                        column_batch_size=self.strict_galerkin_column_batch,
                        projection_site_batch_size=(
                            self.strict_galerkin_projection_batch),
                        check_fine_support=self.strict_galerkin_check_support,
                        include_raw_links=False,
                        retain_blocks=False,
                        max_workspace_bytes=(
                            self.strict_galerkin_max_workspace_bytes),
                        verbose=self.verbose,
                    )
                else:
                    setup_result = build_strict_galerkin(
                        transfer,
                        batch_apply,
                        site_batch_size=self.strict_galerkin_projection_batch,
                        check_fine_support=self.strict_galerkin_check_support,
                        include_raw_links=False,
                        retain_blocks=False,
                        max_workspace_bytes=(
                            self.strict_galerkin_max_workspace_bytes),
                        verbose=self.verbose,
                    )
                setup_result.install(coarse)
                setup_result.stats["requested_probe_mode"] = (
                    self.strict_galerkin_mode)
                setup_result.stats["effective_probe_mode"] = setup_mode
                setup_result.stats["requested_column_batch_size"] = (
                    self.strict_galerkin_column_batch)
                setup_result.stats["requested_projection_site_batch_size"] = (
                    self.strict_galerkin_projection_batch)
                setup_result.stats["max_workspace_bytes"] = (
                    self.strict_galerkin_max_workspace_bytes)
                self.strict_setup_stats.append(dict(setup_result.stats))
            self.transfers.append(transfer)
            self.coarsening_operators.append(coarsening_operator)
            self.operators.append(coarse)
            self.levels.append(_Level(
                index=level, operator=current, transfer=transfer,
                spin=current.spin, color=current.color, shape=current.shape))
            current = coarse
        self.levels.append(_Level(
            index=self._transition_count, operator=current, transfer=None,
            spin=current.spin, color=current.color, shape=current.shape))
        if self._strict_quda:
            self.matpc_operators = [
                QudaMatPCOperator(op, parity=self.target_parity)
                for op in self.operators]
            # 保留 raw Schur 对象仅供与旧 full-coarse 路径对照；严格 V-cycle
            # 使用上面的对称左预处理 MATPC 对象。
            self.parity_operators = [ParitySchurOperator(op)
                                     for op in self.operators]
        elif self.use_parity and not self._compact_parity:
            self.parity_operators = [ParitySchurOperator(op)
                                     for op in self.operators]
        elif self._compact_parity:
            # 保留原始 full Schur 入口供 solve_parity/diagnostics 使用；
            # 紧凑 hierarchy 自身不再对 coarse operator 二次 checkerboard。
            self.parity_operators = [ParitySchurOperator(self._fine_full)]
        self._setup_done = True
        return self

    @staticmethod
    def _unique_tensor_storage_bytes(values: Iterable[Any]) -> Tuple[int, int]:
        """Return unique storage bytes/count for tensor-like values."""
        storages: Dict[Tuple[str, int], int] = {}
        for value in values:
            if value is None or not hasattr(value, "data_ptr"):
                continue
            try:
                storage = value.untyped_storage()
                pointer = int(storage.data_ptr())
                nbytes = int(storage.nbytes())
            except (AttributeError, RuntimeError):
                pointer = int(value.data_ptr())
                nbytes = int(value.numel()) * int(value.element_size())
            device = str(getattr(value, "device", "unknown"))
            storages[(device, pointer)] = max(
                storages.get((device, pointer), 0), nbytes)
        return sum(storages.values()), len(storages)

    def seal_cuda_runtime(self, *, runtime_assets_bound: bool = False
                          ) -> Dict[str, Any]:
        """Detach Python setup tensors after QCU has bound stable packed assets.

        This is intentionally destructive: Python ``apply/solve/verify`` and a
        second asset export are disabled.  The C++ binding must already own the
        packed ``V/Yhat/(X,X^-1)`` tensors, hence the explicit acknowledgement.
        The returned byte count describes detached storage references; actual
        allocator reduction can be smaller when the caller retains aliases.
        """
        if self._cuda_runtime_sealed:
            return dict(self._cuda_runtime_seal_report)
        if not runtime_assets_bound:
            raise RuntimeError(
                "seal_cuda_runtime 要求 runtime_assets_bound=True，"
                "避免释放仍未绑定的 QCU 资产")
        if not self._strict_quda or not self._setup_done:
            raise RuntimeError("seal_cuda_runtime 要求已 setup 的 strict hierarchy")

        detached: List[Any] = list(self._null_vectors)
        for transfer in self.transfers:
            detached.extend((getattr(transfer, "B", None),
                             getattr(transfer, "V", None)))
        detached.append(getattr(self._fine, "_diagonal", None))
        detached.extend(getattr(self._fine, "_diagonal_inv", {}).values())
        detached_bytes, detached_count = self._unique_tensor_storage_bytes(detached)

        for transfer in self.transfers:
            transfer.B = None
            transfer.V = None
        self._null_vectors = []
        self._fine._diagonal = None
        self._fine._diagonal_inv.clear()
        self._fine._matvec = None
        self._fine._adjoint = None
        self._fine._batch_matvec = None
        self.U = self.clover_term = self.kappa = self.u_0 = None
        self.fine_dslash = None
        for operator in self.operators[1:]:
            if isinstance(operator, QudaCoarseOperator):
                operator.blocks = None
                operator.X = operator.X_inv = None
                operator.Y_forward = operator.Y_backward = None
                operator.Y_backward_storage = None
                operator.Yhat_forward = operator.Yhat_backward = None
                operator._strict_packed_assets = None
                operator._dense = None
                operator.fine_operator = None
        self.levels = []
        self.transfers = []
        self.operators = []
        self.coarsening_operators = []
        self.parity_operators = []
        self.matpc_operators = []
        self._cuda_runtime_sealed = True
        self._cuda_runtime_seal_report = {
            "sealed": True,
            "detached_setup_storage_bytes": int(detached_bytes),
            "detached_setup_storage_count": int(detached_count),
            "note": "allocator delta may be smaller when caller retains aliases",
        }
        return dict(self._cuda_runtime_seal_report)

    init = setup

    @staticmethod
    def _reference_field(operator: Any) -> Tensor:
        if isinstance(operator, _FineOperator) and operator._diagonal is not None:
            dtype = operator._diagonal.dtype
            device = operator._diagonal.device
        elif isinstance(operator, CompactParityOperator):
            return operator.reference_field()
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
        # ``operators[0]`` is compact ``S_o`` in MATPC mode, whereas the
        # public ``apply`` API has historically meant the full fine operator.
        # Keep that contract stable; the compact action is exposed separately
        # through ``apply_compact`` and is used internally by ``v_cycle``.
        if self._compact_parity:
            result = self._fine_full.apply(flat)
        else:
            result = self.operators[0].apply(flat)
        return self._from_flat(result, shape_kind, self.fine_spin, self.fine_color)

    matvec = apply

    def apply_compact(self, value: Tensor) -> Tensor:
        """应用 MATPC 层级首层的 compact ``S_o``。

        该入口只在 ``setup_operator='schur'`` 时存在语义；full 模式仍应
        使用 ``apply``。输入输出布局均为
        ``[fine_dof, X, Y, Z, T/2]``。
        """
        self.setup()
        if not self._compact_parity or self._fine_compact is None:
            raise RuntimeError(
                "apply_compact 要求 setup_operator='schur'")
        return self._fine_compact.apply(value)

    matvec_compact = apply_compact

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

    @staticmethod
    def _strict_mr_smooth(matpc: QudaMatPCOperator, rhs: Tensor,
                          solution: Tensor, steps: int) -> Tensor:
        """在当前层的 compact MATPC 空间做固定步数 MR。"""
        x = solution
        for _ in range(max(0, int(steps))):
            residual = rhs - matpc.apply(x)
            image = matpc.apply(residual)
            denominator = tools.vdot(image, image)
            if float(_torch.abs(denominator).item()) < 1e-30:
                break
            alpha = tools.vdot(image, residual) / denominator
            x = x + alpha * residual
        return x

    def _strict_v_cycle(self, rhs: Tensor, level: int) -> Tensor:
        """复现 QUDA 的 MATPC V-cycle 字段语义。

        当前层可接收 full rhs（递归层入口）或已准备好的单 parity rhs
        （最外层 MATPC 预条件器入口）。restriction 只读取目标 parity，
        但下一层 rhs 始终恢复为完整 coarse lattice。
        """
        matpc = self.matpc_operators[level]
        full_input = (
            rhs.ndim == 5 and int(rhs.shape[0]) == matpc.dof and
            tuple(int(x) for x in rhs.shape[-4:]) == matpc.shape)
        if full_input:
            prepared_rhs = matpc.rhs(rhs)
        else:
            prepared_rhs = matpc._validate_compact(rhs)

        if level == len(self.operators) - 1:
            target_solution = matpc.solve(
                prepared_rhs, tol=self.coarse_tol,
                max_iter=self.coarse_max_iter, restart=self.restart,
                direct_solve_max=self.direct_solve_max,
                verbose=self.verbose)
        else:
            target_solution = _torch.zeros_like(prepared_rhs)
            target_solution = self._strict_mr_smooth(
                matpc, prepared_rhs, target_solution, self.nu_pre)
            residual = prepared_rhs - matpc.apply(target_solution)

            transfer = self.transfers[level]
            coarse_rhs = transfer.restrict_parity(
                residual, self.target_parity)
            coarse_error = self._strict_v_cycle(coarse_rhs, level + 1)
            correction = transfer.prolong_parity(
                coarse_error, self.target_parity)
            target_solution = target_solution + correction
            target_solution = self._strict_mr_smooth(
                matpc, prepared_rhs, target_solution, self.nu_post)

        if full_input:
            return matpc.reconstruct(rhs, target_solution)
        return target_solution

    def _mr_smooth(self, level: int, rhs: Tensor, solution: Tensor,
                   steps: int) -> Tensor:
        operator = self.operators[level]
        x = solution
        for _ in range(max(0, steps)):
            residual = rhs - operator.apply(x)
            if self._compact_parity:
                # Every operator in a compact MATPC hierarchy already acts on
                # the target odd geometry.  Applying another checkerboard
                # Schur complement here would halve the coarse lattice a
                # second time and is not QUDA's recursive MATPC path.
                image = operator.apply(residual)
                denominator = tools.vdot(image, image)
                if float(_torch.abs(denominator).item()) < 1e-30:
                    break
                alpha = tools.vdot(image, residual) / denominator
                x = x + alpha * residual
                continue
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
        if self._compact_parity:
            # Compact mode has already eliminated the fine even sites.  Coarse
            # levels are Galerkin operators on that odd geometry, so solve
            # them directly instead of constructing a second checkerboard
            # Schur complement.
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
        if self._strict_quda:
            return self._strict_v_cycle(rhs, level)
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

    def qcu_transition_assets(self, dtype: Any = None, device: Any = None,
                              strict: bool = True,
                              materialize: bool = False) -> List[Dict[str, Any]]:
        """导出每条 fine-to-coarse transition 的 QCU 可消费资产。

        每个返回项包含 ``null_vectors``（QCU blocked ``P`` 布局）以及
        ``sitting``、``hop_nn``、``hop_diag``（宽 33 点粗算子），并附带
        两端几何/自由度元数据。返回项的 ``level`` 从 0 开始，对应
        ``operators[level] -> operators[level + 1]``；compact 模式的
        ``operator_kind`` 首层为 ``compact_schur``，后续为
        ``compact_galerkin``，避免把粗层误认为再次 checkerboard Schur。

        ``QudaCoarseOperator`` 处于 matrix-free 状态时，默认只读而不触发
        逐列 materialize；若确实需要从通用 fine operator 生成精确 33 点
        资产，调用方必须显式传 ``materialize=True``。大格推荐使用
        ``pyqcu.tools.build_stencil_mt`` 的批量/局部构建器生成 stencil，
        再交给 CUDA 驱动，以免把 O(N_site*N_dof) 的 reference 探测误当成
        可扩展 setup 路径。
        """
        self.setup()
        assets: List[Dict[str, Any]] = []
        for level, transfer in enumerate(self.transfers):
            coarse = self.operators[level + 1]
            if not isinstance(coarse, QudaCoarseOperator):
                raise RuntimeError(
                    f"transition {level} 的 coarse operator 类型不支持 QCU 导出")
            if coarse.blocks is None and not materialize:
                raise RuntimeError(
                    f"transition {level} 仍是 matrix-free；QCU stencil 导出会触发"
                    "逐列 materialize。请显式设置 materialize=True，或使用"
                    " tools.build_stencil_mt/build_stencil_local 构建批量 stencil")
            sitting, hop_nn, hop_diag = coarse.to_qcu_stencil(
                dtype=dtype, device=device, strict=strict)
            null_vectors = transfer.to_qcu_blocked(dtype=dtype, device=device)
            assets.append({
                "level": level,
                "fine_shape": transfer.fine_shape,
                "fine_full_shape": self.fine_shape,
                "coarse_shape": transfer.coarse_shape,
                "fine_dof": transfer.fine_dof,
                "coarse_dof": transfer.coarse_dof,
                "compact_parity": self._compact_parity,
                "parity": (self._fine_compact.parity
                            if self._fine_compact is not None else None),
                "eliminated_parity": (
                    self._fine_compact.eliminated_parity
                    if self._fine_compact is not None else None),
                "operator_kind": (
                    "compact_schur" if self._compact_parity and level == 0
                    else "compact_galerkin" if self._compact_parity
                    else "full"),
                "null_vectors": null_vectors,
                "sitting": sitting,
                "hop_nn": hop_nn,
                "hop_diag": hop_diag,
                "stencil": (sitting, hop_nn, hop_diag),
            })
        return assets

    # 面向调用方的同义入口；保留一个明确的主名称，避免与旧 MG API 混淆。
    export_qcu_assets = qcu_transition_assets

    def qcu_strict_transition_assets(
            self, dtype: Any = None, device: Any = None,
            materialize: bool = False, include_raw_links: bool = True,
            runtime_start_level: Optional[int] = None
            ) -> List[Dict[str, Any]]:
        """导出 strict hierarchy 的四槽后端资产。

        每条 transition 的固定槽序为 ``V/raw Y/Yhat/(X,Xinv)``。与旧
        33 点 compact-Schur 四槽不同，这里的两端几何都是完整 lattice；
        single-parity 只在运行 ``R/P`` 时选择 fine site，不改变资产尺寸。
        """
        self.setup()
        if not self._strict_quda:
            raise RuntimeError(
                "qcu_strict_transition_assets 仅适用于 hierarchy_mode='strict'")
        slot_order = (
            "null_vectors", "raw_links", "preconditioned_links",
            "onsite_pair")
        assets: List[Dict[str, Any]] = []
        for level, transfer in enumerate(self.transfers):
            coarse = self.operators[level + 1]
            if not isinstance(coarse, QudaCoarseOperator):
                raise RuntimeError(
                    f"transition {level} 的 coarse operator 类型不支持 strict 导出")
            if (coarse.blocks is None and
                    coarse._strict_packed_assets is None and not materialize):
                raise RuntimeError(
                    f"transition {level} 仍是 matrix-free；strict 资产导出会触发"
                    "逐列 materialize。请显式设置 materialize=True，或使用"
                    "批量 Galerkin 构建器生成 X/Y/Yhat")
            coarse_assets = coarse.to_qcu_strict_assets(
                dtype=dtype, device=device,
                include_raw_links=include_raw_links)
            include_null = (
                runtime_start_level is None or level >= runtime_start_level)
            assets.append({
                "level": level,
                "fine_shape": transfer.fine_shape,
                "coarse_shape": transfer.coarse_shape,
                "fine_dof": transfer.fine_dof,
                "coarse_dof": transfer.coarse_dof,
                "fine_full_geometry": True,
                "coarse_full_geometry": True,
                "target_parity": self.target_parity,
                "operator_kind": "quda_full_preconditioned",
                "slot_order": slot_order,
                "null_vectors": (transfer.to_qcu_blocked(
                    dtype=dtype, device=device) if include_null else None),
                **coarse_assets,
            })
        return assets

    export_qcu_strict_assets = qcu_strict_transition_assets

    def bind_qcu_strict_assets(
            self, set_ptrs: Tensor, dtype: Any = None, device: Any = None,
            start_level: int = 1, retain_raw_links: bool = False,
            materialize: bool = False) -> QcuStrictAssetBinding:
        """导出并绑定 strict CUDA 运行期资产，返回显式生命周期句柄。

        ``start_level=1`` 表示由首个 coarse 层进入递归；因此 fine→level-1
        的 ``V`` 不在该 coarse hierarchy 内驻留。raw ``Y`` 默认省略，诊断
        原语需要时可显式 ``retain_raw_links=True``。
        """
        assets = self.qcu_strict_transition_assets(
            dtype=dtype, device=device, materialize=materialize,
            include_raw_links=retain_raw_links,
            runtime_start_level=int(start_level))
        return QcuStrictAssetBinding(
            set_ptrs, assets, start_level=start_level,
            retain_raw_links=retain_raw_links)

    def solve(self, b: Tensor, x0: Optional[Tensor] = None) -> Tensor:
        self.setup()
        b_flat, shape_kind = self._to_flat(b, self._fine_full)
        guess = None
        if x0 is not None:
            guess, guess_kind = self._to_flat(x0, self._fine_full)
            if guess_kind != shape_kind:
                raise ValueError("b 与 x0 的 spin/color 布局必须一致")
        from ._gmres import fgmres
        history: List[float] = []
        if self._strict_quda:
            matpc = self.matpc_operators[0]
            pc_rhs = matpc.rhs(b_flat)
            pc_guess = None
            if guess is not None:
                pc_guess = matpc.checkerboard.extract(
                    guess, self.target_parity)
            target_solution = fgmres(
                pc_rhs, matpc.apply,
                tol=self.tol, max_iter=self.max_iter, restart=self.restart,
                x0=pc_guess,
                precond=lambda residual: self._strict_v_cycle(residual, 0),
                if_rtol=True, verbose=self.verbose, history=history)
            solution = matpc.reconstruct(b_flat, target_solution)
        elif self._compact_parity:
            assert self._fine_compact is not None
            odd_rhs = self._fine_compact.rhs(b_flat)
            odd_guess = None
            if guess is not None:
                odd_guess = self._fine_compact.layout.extract(
                    guess, self._fine_compact.parity)
            odd_solution = fgmres(
                odd_rhs, self._fine_compact.apply,
                tol=self.tol, max_iter=self.max_iter, restart=self.restart,
                x0=odd_guess,
                precond=lambda residual: self.v_cycle(residual),
                if_rtol=True, verbose=self.verbose, history=history)
            solution = self._fine_compact.reconstruct(b_flat, odd_solution)
        else:
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
        b_flat, shape_kind = self._to_flat(b, self._fine_full)
        if self._strict_quda:
            schur = self.matpc_operators[0]
        elif self._compact_parity:
            assert self._fine_compact is not None
            schur: Any = self._fine_compact
        else:
            schur = self.parity_operators[0]
        from ._gmres import fgmres
        parity_rhs = schur.rhs(b_flat)
        parity_solution = fgmres(
            parity_rhs, schur.apply,
            tol=self.tol if tol is None else float(tol),
            max_iter=self.max_iter if max_iter is None else int(max_iter),
            restart=self.restart, if_rtol=True, verbose=self.verbose)
        result = schur.reconstruct(b_flat, parity_solution)
        if shape_kind == "spin_color":
            return result.reshape(self.fine_spin, self.fine_color, *self.fine_shape)
        return result

    def diagnostics(self, seed_field: Optional[Tensor] = None) -> Dict[str, float]:
        """执行小格验证所需的 P/R、Galerkin、伴随和 Schur 诊断。"""
        self.setup()
        if not self.transfers:
            raise RuntimeError("至少需要一个 coarse 层才能运行 MG diagnostics")
        if seed_field is None:
            if self._fine_full._diagonal is not None:
                dtype = self._fine_full._diagonal.dtype
                device = self._fine_full._diagonal.device
            else:
                dtype = self.transfers[0].V.dtype
                device = self.transfers[0].V.device
            seed_field = _torch.randn(
                size=[self.fine_dof, *self.fine_shape],
                dtype=dtype, device=device)
        if self._compact_parity:
            assert self._fine_compact is not None
            if (seed_field.ndim == 5 and
                    int(seed_field.shape[0]) == self.fine_dof and
                    tuple(int(x) for x in seed_field.shape[1:]) == self.fine_shape):
                full_seed = seed_field
                fine = self._fine_compact.layout.extract(
                    full_seed, self._fine_compact.parity)
            elif (seed_field.ndim == 5 and
                  int(seed_field.shape[0]) == self.fine_dof and
                  tuple(int(x) for x in seed_field.shape[1:]) == self._fine_compact.shape):
                full_seed = self._fine_compact.layout.embed(
                    seed_field, self._fine_compact.parity, self.fine_dof)
                fine = seed_field
            else:
                raise ValueError(
                    "compact diagnostics seed_field 应为 full 或 odd compact 布局")
        else:
            full_seed = seed_field
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
        projected = transfer.prolong(transfer.restrict(fine))
        projected_twice = transfer.prolong(transfer.restrict(projected))
        result["transfer_projection"] = _relative_norm(
            projected_twice - projected, projected)
        result["transfer_projection_capture"] = (
            float(_torch.norm(projected).item()) /
            max(float(_torch.norm(fine).item()), 1e-30))
        coarse_operator = self.operators[1]
        galerkin_source = (
            self.coarsening_operators[0]
            if self._strict_quda else self.operators[0])
        direct = transfer.restrict(
            galerkin_source.apply(transfer.prolong(coarse)))
        result["galerkin_RDP"] = _relative_norm(coarse_operator.apply(coarse) - direct, direct)
        if self.use_parity:
            result["transfer_block_ortho"] = transfer.orthogonality_error()
            # Schur 的重构误差使用一个随机 full rhs；只测试块代数，不要求
            # 外层迭代收敛。
            if self._strict_quda:
                schur = self.matpc_operators[0]
                target_trial = _torch.randn(
                    size=[self.fine_dof, schur.checkerboard.volume],
                    dtype=fine.dtype, device=fine.device)
                reconstructed = schur.reconstruct(full_seed, target_trial)
                residual = full_seed - self._fine_full.apply(reconstructed)
                eliminated = schur.checkerboard.extract(
                    residual, schur.other_parity)
                result["schur_reconstruct_even"] = _relative_norm(
                    eliminated, full_seed)
            elif self._compact_parity:
                assert self._fine_compact is not None
                schur: Any = self._fine_compact
                odd_trial = _torch.randn(
                    size=[self.fine_dof, *schur.shape],
                    dtype=fine.dtype, device=fine.device)
                reconstructed = schur.reconstruct(full_seed, odd_trial)
                residual = full_seed - self._fine_full.apply(reconstructed)
                eliminated = schur.layout.extract(
                    residual, schur.eliminated_parity)
                result["schur_reconstruct_even"] = _relative_norm(
                    eliminated, full_seed)
            else:
                schur = self.parity_operators[0]
                odd_trial = _torch.randn(
                    size=[self.fine_dof, schur.checkerboard.volume],
                    dtype=fine.dtype, device=fine.device)
                reconstructed = schur.reconstruct(full_seed, odd_trial)
                residual = full_seed - self.operators[0].apply(reconstructed)
                result["schur_reconstruct_even"] = float(
                    _torch.norm(schur.checkerboard.extract(residual, 0)).item())
        return result


class QudaStrictMultigrid(QudaMultigrid):
    """严格保持 QUDA full-coarse/逐层 MATPC 语义的便捷入口。

    ``QudaMultigrid`` 的 legacy/compact 行为继续保留；本类只固定
    ``hierarchy_mode='strict'``，其余构造参数完全相同。
    """

    def __init__(self, *args: Any, **kwargs: Any):
        requested = kwargs.pop("hierarchy_mode", "strict")
        if self._normalise_hierarchy_mode(requested) != "strict":
            raise ValueError(
                "QudaStrictMultigrid 不接受非 strict hierarchy_mode")
        kwargs.setdefault("strict_galerkin_mode", "auto")
        super().__init__(*args, hierarchy_mode="strict", **kwargs)


# 公开别名：保留小写风格以便与旧 solver.multigrid 并列，也提供 Python
# 常见的 CamelCase 名称。旧类本身不被替换。
quda_multigrid = QudaMultigrid
QUDAMultigrid = QudaMultigrid
quda_strict_multigrid = QudaStrictMultigrid
QUDAStrictMultigrid = QudaStrictMultigrid


__all__ = [
    "QudaTransfer", "QudaCoarseOperator", "QudaMatPCOperator", "Checkerboard",
    "CompactParityLayout", "CompactParityOperator", "ParitySchurOperator",
    "QcuStrictAssetBinding",
    "QudaMultigrid", "quda_multigrid", "QUDAMultigrid",
    "QudaStrictMultigrid", "quda_strict_multigrid", "QUDAStrictMultigrid",
]
