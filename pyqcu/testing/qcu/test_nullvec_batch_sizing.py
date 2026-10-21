"""Data-free checks for adaptive null-vector BiCGStab batching."""

from pyqcu.tools._multigrid import _nullvec_batch_chunk_from_free


def test_nullvec_batch_chunk_uses_request_when_memory_is_ample() -> None:
    assert _nullvec_batch_chunk_from_free(
        dof=12, field_numel=12 * 24 * 24 * 24 * 32, itemsize=8,
        free_bytes=8 << 30, requested=6) == 6


def test_nullvec_batch_chunk_shrinks_before_out_of_memory() -> None:
    assert _nullvec_batch_chunk_from_free(
        dof=12, field_numel=12 * 24 * 24 * 24 * 32, itemsize=8,
        free_bytes=int(2.25 * (1 << 30)), requested=6) == 2


def test_nullvec_batch_chunk_never_exceeds_dof_or_one() -> None:
    assert _nullvec_batch_chunk_from_free(
        dof=5, field_numel=12 * 8**4, itemsize=8,
        free_bytes=8 << 30, requested=6) == 5
    assert _nullvec_batch_chunk_from_free(
        dof=12, field_numel=12 * 24 * 24 * 24 * 32, itemsize=8,
        free_bytes=0, requested=6) == 1
