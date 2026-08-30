"""Batched local Galerkin setup for strict QUDA-style coarse operators.

The strict hierarchy coarsens a *full-lattice*, nearest-neighbour operator

``D_c = R D_f P``

before a parity Schur complement is formed.  This module therefore never
checkerboards the coarse geometry.  It probes ``K`` coarse sites at a time
and all ``E = coarse_spin * nvec`` columns at each site in one batched
operator call.  Prolongation fills only the selected aggregates and
restriction contracts only their one-hop target aggregates.

The implementation is intentionally a direct-import prototype.  It is not
exported from :mod:`pyqcu.tools` until the production fine operator has a
stable batched-call interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from typing import Sequence, Tuple

import pyqcu.cann as _torch


Tensor = Any
Coord = Tuple[int, int, int, int]
Shape4 = Tuple[int, int, int, int]
BlockKey = Tuple[int, int, int, int]

STRICT_GALERKIN_SCHEMA = "pyqcu.strict_galerkin"
STRICT_GALERKIN_SCHEMA_VERSION = 1


def _shape4(value: Sequence[int], name: str) -> Shape4:
    if len(value) != 4:
        raise ValueError(f"{name} 必须有四个维度，得到 {tuple(value)}")
    result = tuple(int(x) for x in value)
    if any(x <= 0 for x in result):
        raise ValueError(f"{name} 各维必须为正数，得到 {result}")
    return result  # type: ignore[return-value]


def _all_coords(shape: Shape4) -> Iterable[Coord]:
    return product(*(range(extent) for extent in shape))  # type: ignore[return-value]


def _signed_displacement(value: int, extent: int) -> int:
    if extent == 1:
        return 0
    value %= extent
    if value > extent // 2:
        value -= extent
    return value


def _displacement(source: Coord, target: Coord, shape: Shape4) -> BlockKey:
    return tuple(
        _signed_displacement(source[d] - target[d], shape[d])
        for d in range(4)
    )  # type: ignore[return-value]


def _target_entries(source: Coord, shape: Shape4) -> List[Tuple[Coord, BlockKey]]:
    """Return unique target aggregates reached by a one-hop coarse stencil."""
    ordered: Dict[Coord, BlockKey] = {}

    def add(target: Coord) -> None:
        key = _displacement(source, target, shape)
        previous = ordered.get(target)
        if previous is not None and previous != key:
            raise RuntimeError(
                f"周期折叠产生冲突位移：source={source}, target={target}, "
                f"{previous} != {key}")
        ordered[target] = key

    add(source)
    for dim in range(4):
        for step in (-1, 1):
            target = list(source)
            target[dim] = (target[dim] + step) % shape[dim]
            add(tuple(target))  # type: ignore[arg-type]
    return list(ordered.items())


def _fine_slices(coord: Coord, block_size: Shape4) -> Tuple[slice, ...]:
    return tuple(
        slice(coord[d] * block_size[d], (coord[d] + 1) * block_size[d])
        for d in range(4)
    )


def _blocked_site(blocked: Tensor, coord: Coord) -> Tensor:
    index: Tuple[Any, ...] = (
        slice(None), slice(None),
        coord[0], slice(None), coord[1], slice(None),
        coord[2], slice(None), coord[3], slice(None),
    )
    return blocked[index]


def _adjoint_site(matrix: Tensor) -> Tensor:
    return matrix.conj().transpose(0, 1)


def _matmul_site(left: Tensor, right: Tensor) -> Tensor:
    return _torch.einsum("ijxyzt,jkxyzt->ikxyzt", left, right)


def _roll_site(matrix: Tensor, shift: BlockKey) -> Tensor:
    result = matrix
    for dim, amount in enumerate(shift):
        if amount:
            result = _torch.roll(result, shifts=amount, dims=2 + dim)
    return result


def _call_batch(operator: Any, value: Tensor) -> Tensor:
    if hasattr(operator, "batch_apply"):
        return operator.batch_apply(value)
    if hasattr(operator, "matvec_batch"):
        return operator.matvec_batch(value)
    if callable(operator):
        return operator(value)
    raise TypeError("batch_matvec 必须可调用或提供 batch_apply/matvec_batch")


def _action_links(
    blocks: Mapping[BlockKey, Tensor], shape: Shape4, template: Tensor
) -> Tuple[List[Tensor], List[Tensor]]:
    """Convert canonical target-point blocks to forward/backward actions."""
    forward: List[Tensor] = []
    backward: List[Tensor] = []
    zero = _torch.zeros_like(template)
    origin: Coord = (0, 0, 0, 0)
    for dim, extent in enumerate(shape):
        if extent == 1:
            # Wrapped hops have already collapsed into the canonical X block.
            forward.append(zero)
            backward.append(zero)
            continue
        plus_target = list(origin)
        plus_target[dim] = (plus_target[dim] - 1) % extent
        plus_key = _displacement(
            origin, tuple(plus_target), shape)  # type: ignore[arg-type]
        plus = blocks.get(plus_key, zero)
        if extent == 2:
            # q+mu and q-mu are the same physical source.  The strict kernel
            # sums both slots, so each receives half of the combined block.
            forward.append(0.5 * plus)
            backward.append(0.5 * plus)
            continue
        minus_target = list(origin)
        minus_target[dim] = (minus_target[dim] + 1) % extent
        minus_key = _displacement(
            origin, tuple(minus_target), shape)  # type: ignore[arg-type]
        forward.append(plus)
        backward.append(blocks.get(minus_key, zero))
    return forward, backward


def strict_galerkin_memory_model(
    *, coarse_dof: int, fine_dof: int, fine_shape: Sequence[int],
    coarse_shape: Sequence[int], block_size: Sequence[int],
    site_batch_size: int, element_size: int,
    include_raw_links: bool = True, retain_blocks: bool = False,
) -> Dict[str, int]:
    """Return tensor-storage counts; operator-internal temporaries are excluded."""
    fine = _shape4(fine_shape, "fine_shape")
    coarse = _shape4(coarse_shape, "coarse_shape")
    block = _shape4(block_size, "block_size")
    E = int(coarse_dof)
    e = int(fine_dof)
    K = int(site_batch_size)
    s = int(element_size)
    if min(E, e, K, s) <= 0:
        raise ValueError("dof、site_batch_size、element_size 必须为正数")
    Vf, Vc, vb = prod(fine), prod(coarse), prod(block)
    points = len(_target_entries((0, 0, 0, 0), coarse))
    nblocks = len({key for _, key in _target_entries((0, 0, 0, 0), coarse)})

    packed_units = 10 + (8 if include_raw_links else 0)
    packed_elements = packed_units * E * E * Vc
    retained_block_elements = nblocks * E * E * Vc if retain_blocks else 0
    transfer_elements = E * e * Vf
    # Conservative simultaneous builder workspace: input + output full fields,
    # local V/image gathers, projected matrices, and one real support mask.
    batch_field_elements = 2 * K * E * e * Vf
    local_complex_elements = 2 * K * points * E * e * vb + K * points * E * E
    support_mask_bytes = K * Vf * max(1, s // 2)
    workspace_bytes = (batch_field_elements + local_complex_elements) * s + support_mask_bytes
    return {
        "fine_volume": Vf,
        "coarse_volume": Vc,
        "aggregate_volume": vb,
        "support_points": points,
        "canonical_blocks": nblocks,
        "packed_asset_elements": packed_elements,
        "packed_asset_bytes": packed_elements * s,
        "retained_block_elements": retained_block_elements,
        "retained_block_bytes": retained_block_elements * s,
        "transfer_elements": transfer_elements,
        "transfer_bytes": transfer_elements * s,
        "workspace_upper_bytes": workspace_bytes,
        "cache_bytes": (packed_elements + transfer_elements) * s,
    }


def apply_strict_links(links: Tensor, value: Tensor,
                       onsite: Optional[Tensor] = None) -> Tensor:
    """Apply QUDA-stored forward/backward links on a full coarse field."""
    if links.ndim != 8 or tuple(int(x) for x in links.shape[:2]) != (2, 4):
        raise ValueError("links 应为 [2,4,E,E,X,Y,Z,T]")
    E = int(links.shape[2])
    shape = tuple(int(x) for x in links.shape[-4:])
    if tuple(int(x) for x in value.shape) != (E, *shape):
        raise ValueError(f"value 应为 [{E},*{shape}]，得到 {tuple(value.shape)}")
    result = (_torch.zeros_like(value) if onsite is None else
              _torch.einsum("ijxyzt,jxyzt->ixyzt", onsite, value))
    for dim in range(4):
        result = result + _torch.einsum(
            "ijxyzt,jxyzt->ixyzt", links[0, dim],
            _torch.roll(value, shifts=-1, dims=1 + dim))
        stored_at_source = _torch.roll(
            links[1, dim], shifts=1, dims=2 + dim)
        result = result + _torch.einsum(
            "ijxyzt,jxyzt->ixyzt", _adjoint_site(stored_at_source),
            _torch.roll(value, shifts=1, dims=1 + dim))
    return result


@dataclass
class StrictGalerkinResult:
    """Packed strict assets plus optional canonical blocks for integration."""

    fine_shape: Shape4
    coarse_shape: Shape4
    block_size: Shape4
    coarse_spin: int
    nvec: int
    fine_dof: int
    raw_links: Optional[Tensor]
    preconditioned_links: Tensor
    onsite_pair: Tensor
    blocks: Optional[Dict[BlockKey, Tensor]]
    stats: Dict[str, Any]

    @property
    def coarse_dof(self) -> int:
        return int(self.coarse_spin * self.nvec)

    @property
    def X(self) -> Tensor:
        return self.onsite_pair[0]

    @property
    def X_inv(self) -> Tensor:
        return self.onsite_pair[1]

    def assets(self, include_raw_links: bool = True) -> Dict[str, Optional[Tensor]]:
        if include_raw_links and self.raw_links is None:
            raise RuntimeError("构造时 include_raw_links=False，raw Y 不可用")
        return {
            "raw_links": self.raw_links if include_raw_links else None,
            "preconditioned_links": self.preconditioned_links,
            "onsite_pair": self.onsite_pair,
        }

    def apply_raw(self, value: Tensor) -> Tensor:
        if self.raw_links is None:
            raise RuntimeError("raw Y 未保留")
        return apply_strict_links(self.raw_links, value, self.X)

    def apply_preconditioned_hopping(self, value: Tensor) -> Tensor:
        return apply_strict_links(self.preconditioned_links, value)

    def apply_matpc(self, compact: Tensor, parity: int = 0) -> Tensor:
        """Apply ``I-Hhat_pq Hhat_qp``; parity cropping occurs only here."""
        parity = int(parity)
        if parity not in (0, 1):
            raise ValueError("parity 必须为 0 或 1")
        if any(extent > 1 and extent % 2 for extent in self.coarse_shape):
            raise ValueError("周期 checkerboard 要求每个非平凡 coarse extent 为偶数")
        target = [i for i, coord in enumerate(_all_coords(self.coarse_shape))
                  if (sum(coord) & 1) == parity]
        other = [i for i, coord in enumerate(_all_coords(self.coarse_shape))
                 if (sum(coord) & 1) != parity]
        E = self.coarse_dof
        if tuple(int(x) for x in compact.shape) != (E, len(target)):
            raise ValueError(
                f"MATPC field 应为 [{E},{len(target)}]，得到 {tuple(compact.shape)}")

        full = _torch.zeros(
            size=[E, *self.coarse_shape], dtype=compact.dtype,
            device=compact.device)
        full.reshape(E, -1)[:, target] = compact
        first_full = self.apply_preconditioned_hopping(full)
        first = first_full.reshape(E, -1)[:, other]

        other_full = _torch.zeros_like(full)
        other_full.reshape(E, -1)[:, other] = first
        second = self.apply_preconditioned_hopping(other_full)
        return compact - second.reshape(E, -1)[:, target]

    def cache_payload(self, null_vectors: Tensor) -> Dict[str, Any]:
        """Return one-dict payload suitable for ``tools.save_dict_h5``."""
        expected = (
            self.coarse_dof, self.fine_dof,
            self.coarse_shape[0], self.block_size[0],
            self.coarse_shape[1], self.block_size[1],
            self.coarse_shape[2], self.block_size[2],
            self.coarse_shape[3], self.block_size[3],
        )
        if tuple(int(x) for x in null_vectors.shape) != expected:
            raise ValueError(
                f"null_vectors 应为 blocked shape {expected}，"
                f"得到 {tuple(null_vectors.shape)}")
        return {
            "schema": {
                "name": STRICT_GALERKIN_SCHEMA,
                "version": STRICT_GALERKIN_SCHEMA_VERSION,
                "layout": "C-order; matrix,row-col,...xyzt",
                "parity_scope": "R/P views and MATPC only; assets are full lattice",
                "backward_storage": "link at q-mu; dagger on gather",
                "coarse_spin": int(self.coarse_spin),
                "nvec": int(self.nvec),
                "coarse_dof": int(self.coarse_dof),
                "fine_dof": int(self.fine_dof),
                "dtype": str(self.onsite_pair.dtype),
                "raw_links_present": self.raw_links is not None,
                "fine_shape": self.fine_shape,
                "coarse_shape": self.coarse_shape,
                "block_size": self.block_size,
            },
            "null_vectors": null_vectors,
            "raw_links": self.raw_links,
            "preconditioned_links": self.preconditioned_links,
            "onsite_pair": self.onsite_pair,
        }

    def install(self, coarse_operator: Any) -> None:
        """Install the result into a lazy ``QudaCoarseOperator`` instance."""
        if self.blocks is None:
            raise RuntimeError("install 要求 build 时 retain_blocks=True")
        if (tuple(int(x) for x in coarse_operator.shape) != self.coarse_shape or
                int(coarse_operator.dof) != self.coarse_dof):
            raise ValueError("coarse_operator 的 shape/dof 与构造结果不一致")
        forward, backward = _action_links(
            self.blocks, self.coarse_shape, self.blocks[(0, 0, 0, 0)])
        coarse_operator.blocks = self.blocks
        coarse_operator.X = self.blocks[(0, 0, 0, 0)]
        coarse_operator.X_inv = self.X_inv
        coarse_operator.Y_forward = forward
        coarse_operator.Y_backward = backward
        if self.raw_links is None:
            storage = []
            for dim in range(4):
                shift = tuple(-1 if i == dim else 0 for i in range(4))
                storage.append(_roll_site(_adjoint_site(backward[dim]), shift))
            coarse_operator.Y_backward_storage = storage
        else:
            coarse_operator.Y_backward_storage = [self.raw_links[1, d]
                                                   for d in range(4)]
        coarse_operator.Yhat_forward = [self.preconditioned_links[0, d]
                                        for d in range(4)]
        coarse_operator.Yhat_backward = [self.preconditioned_links[1, d]
                                         for d in range(4)]
        coarse_operator._dense = None


def build_strict_galerkin(
    transfer: Any,
    batch_matvec: Callable[[Tensor], Tensor] | Any,
    *,
    site_batch_size: int = 4,
    support_rtol: float = 1e-6,
    support_atol: float = 1e-12,
    check_fine_support: bool = True,
    include_raw_links: bool = True,
    retain_blocks: bool = False,
    max_workspace_bytes: Optional[int] = None,
    verbose: bool = False,
) -> StrictGalerkinResult:
    """Build strict ``X/Y/Yhat`` without column-wise full-basis calls.

    ``batch_matvec`` must map ``[B,e,X,Y,Z,T]`` to the same shape without
    coupling the batch entries.  The operator must be local/on-site plus
    nearest-neighbour on the full fine lattice.  With ``check_fine_support``
    enabled, every probe image is required to vanish outside the source
    aggregate and its eight one-hop target aggregates.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    if MPI is not None and int(MPI.COMM_WORLD.Get_size()) != 1:
        raise RuntimeError(
            "strict Galerkin batch 原型仅证明单 MPI rank；跨 rank source/R "
            "汇聚尚未实现")

    fine_shape = _shape4(transfer.fine_shape, "transfer.fine_shape")
    coarse_shape = _shape4(transfer.coarse_shape, "transfer.coarse_shape")
    block_size = _shape4(transfer.block_size, "transfer.block_size")
    if any(coarse_shape[d] * block_size[d] != fine_shape[d] for d in range(4)):
        raise ValueError("fine_shape != coarse_shape * block_size")
    if int(transfer.coarse_spin) != 2:
        raise ValueError(
            "strict full-coarse 原型要求 coarse_spin=2；compact parity "
            "coarse_spin=1 不得用于 X/Y setup")
    if int(getattr(transfer, "spin_block_size", 0)) == 0:
        raise ValueError("当前 strict 原型不支持 staggered spin/parity 映射")

    blocked = transfer.to_qcu_blocked()
    device_type = getattr(blocked.device, "type", str(blocked.device))
    if device_type not in ("cpu", "cuda"):
        raise RuntimeError(
            "strict Galerkin blocked-V 原型仅支持 CPU/CUDA；10 维布局未证明 NPU")
    E = int(transfer.coarse_dof)
    e = int(transfer.fine_dof)
    nvec = int(transfer.nvec)
    if tuple(int(x) for x in blocked.shape[:2]) != (E, e):
        raise ValueError("blocked V 的 E/e 维与 transfer 元数据不一致")
    if E != 2 * nvec:
        raise ValueError(f"coarse_spin=2 时应有 E=2*nvec，得到 E={E}, nvec={nvec}")

    requested_batch = int(site_batch_size)
    if requested_batch <= 0:
        raise ValueError("site_batch_size 必须为正数")
    support_points = len(_target_entries((0, 0, 0, 0), coarse_shape))
    element_size = int(blocked.element_size())
    if max_workspace_bytes is not None:
        one_site = strict_galerkin_memory_model(
            coarse_dof=E, fine_dof=e, fine_shape=fine_shape,
            coarse_shape=coarse_shape, block_size=block_size,
            site_batch_size=1, element_size=element_size,
            include_raw_links=include_raw_links,
            retain_blocks=retain_blocks,
        )["workspace_upper_bytes"]
        allowed = int(max_workspace_bytes) // max(1, one_site)
        if allowed < 1:
            raise MemoryError(
                f"max_workspace_bytes={max_workspace_bytes} 小于单粗点估算 {one_site}")
        requested_batch = min(requested_batch, allowed)

    coords = list(_all_coords(coarse_shape))
    Kmax = min(requested_batch, len(coords))
    zero_key: BlockKey = (0, 0, 0, 0)
    keys = {key for _, key in _target_entries((0, 0, 0, 0), coarse_shape)}
    blocks: Dict[BlockKey, Tensor] = {
        key: _torch.zeros(
            size=[E, E, *coarse_shape], dtype=blocked.dtype,
            device=blocked.device)
        for key in keys
    }
    if zero_key not in blocks:
        raise RuntimeError("strict stencil 缺少 canonical zero block")

    t0 = perf_counter()
    calls = 0
    worst_leakage = 0.0
    worst_scale = 0.0
    for start in range(0, len(coords), Kmax):
        sources = coords[start:start + Kmax]
        K = len(sources)
        fine = _torch.zeros(
            size=[K, E, e, *fine_shape], dtype=blocked.dtype,
            device=blocked.device)
        entries_by_source: List[List[Tuple[Coord, BlockKey]]] = []
        for k, source in enumerate(sources):
            source_slices = _fine_slices(source, block_size)
            fine[(k, slice(None), slice(None), *source_slices)] = _blocked_site(
                blocked, source)
            entries_by_source.append(_target_entries(source, coarse_shape))

        flat_fine = fine.reshape(K * E, e, *fine_shape)
        image = _call_batch(batch_matvec, flat_fine)
        calls += 1
        expected_shape = (K * E, e, *fine_shape)
        if tuple(int(x) for x in image.shape) != expected_shape:
            raise ValueError(
                f"batch_matvec 输出应为 {expected_shape}，得到 {tuple(image.shape)}")
        if image.dtype != blocked.dtype or image.device != blocked.device:
            raise ValueError("batch_matvec 必须保持 blocked V 的 dtype/device")
        image = image.reshape(K, E, e, *fine_shape)

        if check_fine_support:
            coarse_mask = _torch.zeros(
                size=[K, *coarse_shape], dtype=blocked.real.dtype,
                device=blocked.device)
            for k, entries in enumerate(entries_by_source):
                for target, _ in entries:
                    coarse_mask[(k, *target)] = 1.0
            fine_mask = coarse_mask
            for dim, extent in enumerate(block_size):
                fine_mask = fine_mask.repeat_interleave(extent, dim=1 + dim)
            magnitude = _torch.abs(image)
            scale = float(magnitude.max().item()) if image.numel() else 0.0
            leakage = float((magnitude * (1.0 - fine_mask[:, None, None])).max().item())
            worst_leakage = max(worst_leakage, leakage)
            worst_scale = max(worst_scale, scale)
            threshold = float(support_atol) + float(support_rtol) * scale
            if leakage > threshold:
                raise ValueError(
                    "batch_matvec 违反 full nearest-neighbour 支撑："
                    f"outside={leakage:.3e} > {threshold:.3e}; "
                    "Schur/宽 stencil 不得作为 strict X/Y 输入")

        v_rows: List[Tensor] = []
        image_rows: List[Tensor] = []
        for k, entries in enumerate(entries_by_source):
            v_rows.append(_torch.stack([
                _blocked_site(blocked, target) for target, _ in entries
            ], dim=0))
            image_rows.append(_torch.stack([
                image[(k, slice(None), slice(None),
                       *_fine_slices(target, block_size))]
                for target, _ in entries
            ], dim=0))
        v_local = _torch.stack(v_rows, dim=0)
        image_local = _torch.stack(image_rows, dim=0)
        projected = _torch.einsum(
            "kpae...,kpbe...->kpab", v_local.conj(), image_local)

        for k, entries in enumerate(entries_by_source):
            for point, (target, key) in enumerate(entries):
                blocks[key][(slice(None), slice(None), *target)] = projected[k, point]

    X = blocks[zero_key]
    site_matrix = X.permute(2, 3, 4, 5, 0, 1).reshape(-1, E, E)
    inverse = _torch.linalg_inv(site_matrix)
    X_inv = inverse.reshape(
        *coarse_shape, E, E).permute(4, 5, 0, 1, 2, 3).contiguous()
    forward, backward = _action_links(blocks, coarse_shape, X)

    raw_links: Optional[Tensor]
    if include_raw_links:
        raw_links = _torch.zeros(
            size=[2, 4, E, E, *coarse_shape], dtype=blocked.dtype,
            device=blocked.device)
    else:
        raw_links = None
    preconditioned = _torch.zeros(
        size=[2, 4, E, E, *coarse_shape], dtype=blocked.dtype,
        device=blocked.device)
    for dim in range(4):
        shift = tuple(-1 if axis == dim else 0 for axis in range(4))
        backward_storage = _roll_site(_adjoint_site(backward[dim]), shift)
        source_xinv = _roll_site(X_inv, shift)
        if raw_links is not None:
            raw_links[0, dim] = forward[dim]
            raw_links[1, dim] = backward_storage
        preconditioned[0, dim] = _matmul_site(X_inv, forward[dim])
        preconditioned[1, dim] = _matmul_site(
            backward_storage, _adjoint_site(source_xinv))
    onsite_pair = _torch.stack([X, X_inv], dim=0).contiguous()
    if raw_links is not None:
        raw_links = raw_links.contiguous()
    preconditioned = preconditioned.contiguous()

    elapsed = perf_counter() - t0
    memory = strict_galerkin_memory_model(
        coarse_dof=E, fine_dof=e, fine_shape=fine_shape,
        coarse_shape=coarse_shape, block_size=block_size,
        site_batch_size=Kmax, element_size=element_size,
        include_raw_links=include_raw_links, retain_blocks=retain_blocks,
    )
    stats: Dict[str, Any] = {
        "scalar_columns": len(coords) * E,
        "operator_calls": calls,
        "site_batch_size": Kmax,
        "coarse_sites": len(coords),
        "coarse_dof": E,
        "fine_dof": e,
        "support_points": support_points,
        "worst_fine_support_leakage": worst_leakage,
        "worst_image_scale": worst_scale,
        "elapsed_seconds": elapsed,
        "memory": memory,
    }
    if verbose:
        print(
            "PYQCU::TOOLS::STRICT_GALERKIN:\n "
            f"{len(coords) * E} columns in {calls} batched calls "
            f"(K={Kmax}), {elapsed:.3f}s")

    return StrictGalerkinResult(
        fine_shape=fine_shape,
        coarse_shape=coarse_shape,
        block_size=block_size,
        coarse_spin=2,
        nvec=nvec,
        fine_dof=e,
        raw_links=raw_links,
        preconditioned_links=preconditioned,
        onsite_pair=onsite_pair,
        blocks=blocks if retain_blocks else None,
        stats=stats,
    )


__all__ = [
    "STRICT_GALERKIN_SCHEMA",
    "STRICT_GALERKIN_SCHEMA_VERSION",
    "StrictGalerkinResult",
    "apply_strict_links",
    "build_strict_galerkin",
    "strict_galerkin_memory_model",
]
