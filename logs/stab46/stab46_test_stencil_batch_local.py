"""局部批量 stencil 探测的 CPU 等价性回归。

该测试不需要 CUDA：用确定性的块对角 coarse/fine 组件构造
``BatchedLocalSchur``，比较 K=1 参考路径与 K>1 批量路径，样本包含
周期边界和内部粗点。
"""

from types import SimpleNamespace

import torch

from pyqcu.tools._multigrid import (
    BatchedLocalSchur,
    _probe_point_batch_local,
    _probe_points_batch_local,
)


def _matrix(scale, fine):
    eye = torch.eye(12, dtype=torch.complex64).reshape(12, 12, 1, 1, 1, 1)
    return (eye * scale).expand(12, 12, *fine).contiguous()


def _materialized_schur(lsch, x_local, idx, starts):
    """Reference formula retaining the six operator matrix groups."""
    K, E = x_local.shape[:2]
    Mep = [lsch._slicem(lsch.Mep[d], idx, starts) for d in range(4)]
    Mem = [lsch._slicem(lsch.Mem[d], idx, starts) for d in range(4)]
    Mop = [lsch._slicem(lsch.Mop[d], idx, starts) for d in range(4)]
    Mom = [lsch._slicem(lsch.Mom[d], idx, starts) for d in range(4)]
    Me_inv = lsch._slicem(lsch.Me_inv, idx, starts)
    Mo = lsch._slicem(lsch.Mo, idx, starts)
    mek, mok = lsch._masks(idx, K, E)

    dest_e = torch.zeros_like(x_local)
    for d in range(4):
        src_p = torch.roll(x_local, shifts=-1, dims=d + 3)
        src_m = torch.roll(x_local, shifts=1, dims=d + 3)
        if d == 3:
            src_p = torch.where(mek, x_local, src_p)
            src_m = torch.where(mok, x_local, src_m)
        dest_e += torch.einsum(
            "kEexyzt,kBexyzt->kBExyzt", Mep[d], src_p
        )
        dest_e += torch.einsum(
            "kEexyzt,kBexyzt->kBExyzt", Mem[d], src_m
        )
    xe_inv = torch.einsum(
        "kEeXYZT,kBeXYZT->kBEXYZT", Me_inv, dest_e
    )

    dest_o = torch.zeros_like(x_local)
    for d in range(4):
        src_p = torch.roll(xe_inv, shifts=-1, dims=d + 3)
        src_m = torch.roll(xe_inv, shifts=1, dims=d + 3)
        if d == 3:
            src_p = torch.where(mok, xe_inv, src_p)
            src_m = torch.where(mek, xe_inv, src_m)
        dest_o += torch.einsum(
            "kEexyzt,kBexyzt->kBExyzt", Mop[d], src_p
        )
        dest_o += torch.einsum(
            "kEexyzt,kBexyzt->kBExyzt", Mom[d], src_m
        )
    out = torch.einsum(
        "kEeXYZT,kBeXYZT->kBEXYZT", Mo, x_local
    )
    return out - dest_o


def test_batched_local_probe_matches_single_point():
    fine = (8, 8, 8, 8)
    coarse = (4, 4, 4, 4)
    E = 3
    W = 6
    hopping = SimpleNamespace(
        M_e_plus_list=[_matrix(0.7 + 0.03 * d, fine) for d in range(4)],
        M_e_minus_list=[_matrix(0.5 + 0.02 * d, fine) for d in range(4)],
        M_o_plus_list=[_matrix(0.4 + 0.01 * d, fine) for d in range(4)],
        M_o_minus_list=[_matrix(0.3 + 0.02 * d, fine) for d in range(4)],
    )
    sitting = SimpleNamespace(
        M_e_inv=_matrix(1.1, fine),
        M_o=_matrix(1.2, fine),
    )
    lsch = BatchedLocalSchur(
        SimpleNamespace(hopping=hopping, sitting=sitting), *fine, W=W
    )
    torch.manual_seed(7)
    lonv = torch.randn(
        (E, 12, 4, 2, 4, 2, 4, 2, 4, 2), dtype=torch.complex64
    )
    centers = [0, 1, 4, 17, 63]
    shape_s = (E, E, *coarse)
    shape_n = (2, 4, E, E, *coarse)
    shape_d = (2, 2, 6, E, E, *coarse)
    reference = [torch.zeros(shape_s, dtype=torch.complex64),
                 torch.zeros(shape_n, dtype=torch.complex64),
                 torch.zeros(shape_d, dtype=torch.complex64)]
    for c_idx in centers:
        _probe_point_batch_local(
            lsch, lonv, E, c_idx, reference[0], reference[1], reference[2],
            list(coarse), coarse[0] * coarse[1] * coarse[2] * coarse[3], W
        )
    batched = [torch.zeros_like(value) for value in reference]
    _probe_points_batch_local(
        lsch, lonv, E, centers, batched[0], batched[1], batched[2],
        list(coarse), coarse[0] * coarse[1] * coarse[2] * coarse[3], W
    )
    for actual, expected in zip(batched, reference):
        assert torch.equal(actual, expected)
        assert bool(torch.isfinite(actual).all().item())

    idx = torch.stack([torch.arange(-2, 4) % n for n in fine])
    idx = idx.unsqueeze(0).repeat(2, 1, 1)
    starts = [(6, 6, 6, 6), (0, 0, 0, 0)]
    x_local = torch.randn(
        (2, E, 12, W, W, W, W),
        generator=torch.Generator().manual_seed(11),
        dtype=torch.complex64,
    )
    expected = _materialized_schur(lsch, x_local, idx, starts)
    actual = lsch(x_local, idx, starts)
    assert torch.equal(actual, expected)
