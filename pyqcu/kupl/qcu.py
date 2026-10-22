"""ctypes binding for the independent KUPL CPU QCU implementation.

This module exposes the same flat Python ABI as the CUDA bridge. ``params`` is
int32[58] for geometry, rank-grid, dtype, plan, parity, solver controls, and the
current set slot. ``argv`` is real[8] for mass, tolerances, sigma, and ``csw``.
``set_ptrs`` is int64[100]: ``[0:30)`` is ordinary transaction scratch while
``[30:60)``, ``[60:80)``, and slot 80 hold reserved multigrid coarse/strict
assets. Ordinary calls select the current slot before ``applyInitQcu`` and must
keep it unchanged through operations/getters until ``applyEndQcu``. Strict
multigrid keeps one fixed private slot alive across its dedicated setup, solve,
getter, and teardown sequence.

Parity-split fields use ``[X, Y, Z, T/2]`` as their final four axes: fermions
are ``[2, 4, 3, X, Y, Z, T/2]``, gauge links are
``[2, 3, 3, 4, X, Y, Z, T/2]``, and Clover terms are
``[4, 3, 4, 3, X, Y, Z, T/2]``. ``params[_PARITY_]`` selects the even or odd
target where required.

ctypes passes only raw 64-bit addresses. This module checks contiguity but
cannot recover shape, dtype, or device from ``data_ptr()``, so callers must pass
exact ABI-compatible CPU tensors and retain every tensor until the C++ call
returns. CUDA production remains one Python process per MPI rank and device;
KUPL uses the same process/rank partitioning for CPU/OpenMP work.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any


_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_LIB = _REPO / "cpp/kupl/qcu/build/libqcu.so"


def library_path() -> Path:
    requested = os.environ.get("PYQCU_KUPL_LIB")
    path = Path(requested).expanduser() if requested else _DEFAULT_LIB
    if not path.is_file():
        raise FileNotFoundError(
            f"KUPL QCU library not found at {path}; build "
            "cpp/kupl/qcu or set PYQCU_KUPL_LIB")
    return path


_LIBRARY = ctypes.CDLL(str(library_path()), mode=ctypes.RTLD_LOCAL)
_LL = ctypes.c_longlong
_I = ctypes.c_int
_ULL = ctypes.c_ulonglong
_ULL_P = ctypes.POINTER(_ULL)
_I_P = ctypes.POINTER(_I)
_D = ctypes.c_double
_D_P = ctypes.POINTER(_D)

_SIGNATURES = {
    "applyInitQcu": (_LL, _LL, _LL),
    "applyEndQcu": (_LL, _LL),
    "applyWilsonDslashQcu": (_LL, _LL, _LL, _LL, _LL),
    "applyWilsonBistabCgQcu": (_LL, _LL, _LL, _LL, _LL),
    "applyWilsonBistabCgDslashQcu": (_LL, _LL, _LL, _LL, _LL),
    "applyWilsonCgQcu": (_LL, _LL, _LL, _LL, _LL),
    "applyWilsonCgDslashQcu": (_LL, _LL, _LL, _LL, _LL),
    "testWilsonDslashQcu": (_LL, _LL, _LL, _LL, _LL),
    "applyCloverDslashQcu": (_LL, _LL, _LL, _LL, _LL),
    "testCloverDslashQcu": (_LL, _LL, _LL, _LL, _LL),
    "applyCloverQcu": (_LL, _LL, _LL, _LL),
    "applyCloversQcu": (_LL, _LL, _LL, _LL, _LL),
    "applyDslashQcu": (_LL, _LL, _LL, _LL, _LL, _LL),
    "applyLaplacianQcu": (_LL, _LL, _LL, _LL, _LL),
    "applyGaussGaugeQcu": (_LL, _LL, _LL),
    "applyMultigridRestrictQcu": (_LL,) * 5,
    "applyMultigridProLongQcu": (_LL,) * 5,
    "applyMultigridCoarseDslashQcu": (_LL,) * 6,
    "applyMultigridCoarseDslashWideQcu": (_LL,) * 7,
    "applyCloverBistabCgQcu": (_LL,) * 9,
    "getCloverBistabCgIterationsQcu": (_LL, _LL),
    "applyCloverBistabCgDslashQcu": (_LL,) * 9,
    "applyCloverBistabCgPrepareQcu": (_LL,) * 9,
    "applyCloverBistabCgReconstructQcu": (_LL,) * 10,
    "applyMultigridStrictCoarseQcu": (_LL,) * 6 + (_I,) * 6,
    "applyMultigridStrictMatPCQcu": (_LL,) * 6 + (_I,) * 6,
    "applyMultigridStrictFineMatPCQcu": (_LL,) * 9 + (_I,),
    "applyMultigridStrictPrepareQcu": (_LL,) * 7 + (_I,) * 6,
    "applyMultigridStrictReconstructQcu": (_LL,) * 8 + (_I,) * 6,
    "applyMultigridStrictRestrictQcu": (_LL,) * 5 + (_I,) * 11,
    "applyMultigridStrictProLongQcu": (_LL,) * 5 + (_I,) * 11,
    "applyMultigridStrictVCycleQcu": (_LL,) * 4 + (_I, _ULL_P),
    "applyMultigridStrictInitQcu": (_LL,) * 2 + (_I, _ULL_P),
    "applyMultigridStrictEndQcu": (_LL,) * 2,
    "applyMultigridStrictFgmresQcu": (
        (_LL,) * 10 + (_I,) * 13 + (_D,) + (_I,) * 2 +
        (_ULL, _I_P, _I_P, _D_P, _ULL_P)),
    "applyCloverMultigridQcu": (_LL,) * 9,
    "verifyCloverMultigridQcu": (_LL,) * 9,
}

_STATUS_FUNCTIONS = {
    "getCloverBistabCgIterationsQcu",
    "applyCloverBistabCgPrepareQcu",
    "applyCloverBistabCgReconstructQcu",
    "applyMultigridStrictCoarseQcu",
    "applyMultigridStrictMatPCQcu",
    "applyMultigridStrictFineMatPCQcu",
    "applyMultigridStrictPrepareQcu",
    "applyMultigridStrictReconstructQcu",
    "applyMultigridStrictRestrictQcu",
    "applyMultigridStrictProLongQcu",
    "applyMultigridStrictVCycleQcu",
    "applyMultigridStrictInitQcu",
    "applyMultigridStrictEndQcu",
    "applyMultigridStrictFgmresQcu",
    "verifyCloverMultigridQcu",
}

for _name, _argtypes in _SIGNATURES.items():
    _function = getattr(_LIBRARY, _name)
    _function.argtypes = _argtypes
    _function.restype = ctypes.c_int if _name in _STATUS_FUNCTIONS else None


def _pointer(tensor: Any) -> int:
    """Return a dense tensor's address for a raw ctypes call.

    ``data_ptr()`` does not validate ABI shape/dtype/device and does not keep
    the storage alive. The caller must provide a contiguous tensor of the exact
    expected ABI type on host-accessible memory and retain it for the duration
    of the native call.
    """
    if not tensor.is_contiguous():
        raise ValueError("KUPL QCU tensors must be contiguous")
    return int(tensor.data_ptr())


def applyInitQcu(set_ptrs, params, argv) -> None:
    """Allocate resources in ``params[_SET_INDEX_]`` and fill ``set_ptrs``.

    Select an ordinary slot in ``[0:30)`` before calling. Keep that index fixed
    through the operation and any getters, then release it with
    :func:`applyEndQcu` before selecting the next slot.
    """
    _LIBRARY.applyInitQcu(_pointer(set_ptrs), _pointer(params), _pointer(argv))


def applyEndQcu(set_ptrs, params) -> None:
    """Release the backend resource at the current ``_SET_INDEX_`` slot."""
    _LIBRARY.applyEndQcu(_pointer(set_ptrs), _pointer(params))


def applyWilsonDslashQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyWilsonDslashQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def applyWilsonBistabCgQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyWilsonBistabCgQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def applyWilsonBistabCgDslashQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyWilsonBistabCgDslashQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def applyWilsonCgQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyWilsonCgQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def applyWilsonCgDslashQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyWilsonCgDslashQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def testWilsonDslashQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.testWilsonDslashQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def applyCloverDslashQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyCloverDslashQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def testCloverDslashQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.testCloverDslashQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def applyCloverQcu(clover, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyCloverQcu(
        _pointer(clover), _pointer(gauge), _pointer(set_ptrs), _pointer(params))


def applyCloversQcu(clover, clover_inv, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyCloversQcu(
        _pointer(clover), _pointer(clover_inv), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def applyDslashQcu(out, src, gauge, clover, set_ptrs, params) -> None:
    _LIBRARY.applyDslashQcu(
        _pointer(out), _pointer(src), _pointer(gauge), _pointer(clover),
        _pointer(set_ptrs), _pointer(params))


def applyLaplacianQcu(out, src, gauge, set_ptrs, params) -> None:
    _LIBRARY.applyLaplacianQcu(
        _pointer(out), _pointer(src), _pointer(gauge),
        _pointer(set_ptrs), _pointer(params))


def applyGaussGaugeQcu(gauge, set_ptrs, params) -> None:
    _LIBRARY.applyGaussGaugeQcu(
        _pointer(gauge), _pointer(set_ptrs), _pointer(params))


def applyMultigridRestrictQcu(coarse_out, fine_in, null_vectors, set_ptrs,
                              params) -> None:
    _LIBRARY.applyMultigridRestrictQcu(
        _pointer(coarse_out), _pointer(fine_in), _pointer(null_vectors),
        _pointer(set_ptrs), _pointer(params))


def applyMultigridProLongQcu(fine_out, coarse_in, null_vectors, set_ptrs,
                             params) -> None:
    _LIBRARY.applyMultigridProLongQcu(
        _pointer(fine_out), _pointer(coarse_in), _pointer(null_vectors),
        _pointer(set_ptrs), _pointer(params))


def applyMultigridCoarseDslashQcu(out, src, hopping, sitting, set_ptrs,
                                  params) -> None:
    _LIBRARY.applyMultigridCoarseDslashQcu(
        _pointer(out), _pointer(src), _pointer(hopping), _pointer(sitting),
        _pointer(set_ptrs), _pointer(params))


def applyMultigridCoarseDslashWideQcu(out, src, sitting, hop_nn, hop_diag,
                                      set_ptrs, params) -> None:
    _LIBRARY.applyMultigridCoarseDslashWideQcu(
        _pointer(out), _pointer(src), _pointer(sitting), _pointer(hop_nn),
        _pointer(hop_diag), _pointer(set_ptrs), _pointer(params))


def applyCloverBistabCgQcu(out, src, gauge, ce, coo, cei, coi,
                           set_ptrs, params) -> None:
    _LIBRARY.applyCloverBistabCgQcu(
        _pointer(out), _pointer(src), _pointer(gauge), _pointer(ce),
        _pointer(coo), _pointer(cei), _pointer(coi),
        _pointer(set_ptrs), _pointer(params))


def getCloverBistabCgIterationsQcu(set_ptrs, params) -> int:
    return int(_LIBRARY.getCloverBistabCgIterationsQcu(
        _pointer(set_ptrs), _pointer(params)))


def applyCloverBistabCgDslashQcu(out, src, gauge, ce, coo, cei, coi,
                                 set_ptrs, params) -> None:
    _LIBRARY.applyCloverBistabCgDslashQcu(
        _pointer(out), _pointer(src), _pointer(gauge), _pointer(ce),
        _pointer(coo), _pointer(cei), _pointer(coi),
        _pointer(set_ptrs), _pointer(params))


def applyCloverBistabCgPrepareQcu(compact_rhs, full_rhs, gauge, ce, coo, cei,
                                  coi, set_ptrs, params) -> int:
    return int(_LIBRARY.applyCloverBistabCgPrepareQcu(
        _pointer(compact_rhs), _pointer(full_rhs), _pointer(gauge),
        _pointer(ce), _pointer(coo), _pointer(cei), _pointer(coi),
        _pointer(set_ptrs), _pointer(params)))


def applyCloverBistabCgReconstructQcu(full_out, full_rhs, target_solution,
                                      gauge, ce, coo, cei, coi, set_ptrs,
                                      params) -> int:
    return int(_LIBRARY.applyCloverBistabCgReconstructQcu(
        _pointer(full_out), _pointer(full_rhs), _pointer(target_solution),
        _pointer(gauge), _pointer(ce), _pointer(coo), _pointer(cei),
        _pointer(coi), _pointer(set_ptrs), _pointer(params)))


def applyMultigridStrictCoarseQcu(fermion_out, fermion_in, links,
                                  onsite_pair, set_ptrs, params, E, X, Y, Z,
                                  T, onsite_index) -> int:
    return int(_LIBRARY.applyMultigridStrictCoarseQcu(
        _pointer(fermion_out), _pointer(fermion_in), _pointer(links),
        _pointer(onsite_pair), _pointer(set_ptrs), _pointer(params),
        E, X, Y, Z, T, onsite_index))


def applyMultigridStrictMatPCQcu(fermion_out, fermion_in, links, scratch,
                                 set_ptrs, params, E, X, Y, Z, T,
                                 parity) -> int:
    return int(_LIBRARY.applyMultigridStrictMatPCQcu(
        _pointer(fermion_out), _pointer(fermion_in), _pointer(links),
        _pointer(scratch), _pointer(set_ptrs), _pointer(params),
        E, X, Y, Z, T, parity))


def applyMultigridStrictFineMatPCQcu(
        fermion_out, fermion_in, gauge, ce, coo, cei, coi, set_ptrs, params,
        parity) -> int:
    return int(_LIBRARY.applyMultigridStrictFineMatPCQcu(
        _pointer(fermion_out), _pointer(fermion_in), _pointer(gauge),
        _pointer(ce), _pointer(coo), _pointer(cei), _pointer(coi),
        _pointer(set_ptrs), _pointer(params), parity))


def applyMultigridStrictPrepareQcu(
        fermion_out, full_rhs, links, onsite_pair, scratch, set_ptrs, params,
        E, X, Y, Z, T, parity) -> int:
    return int(_LIBRARY.applyMultigridStrictPrepareQcu(
        _pointer(fermion_out), _pointer(full_rhs), _pointer(links),
        _pointer(onsite_pair), _pointer(scratch), _pointer(set_ptrs),
        _pointer(params), E, X, Y, Z, T, parity))


def applyMultigridStrictReconstructQcu(
        full_out, full_rhs, target_solution, links, onsite_pair, scratch,
        set_ptrs, params, E, X, Y, Z, T, parity) -> int:
    return int(_LIBRARY.applyMultigridStrictReconstructQcu(
        _pointer(full_out), _pointer(full_rhs), _pointer(target_solution),
        _pointer(links), _pointer(onsite_pair), _pointer(scratch),
        _pointer(set_ptrs), _pointer(params), E, X, Y, Z, T, parity))


def applyMultigridStrictRestrictQcu(
        coarse_out, fine_in, null_vectors, set_ptrs, params, E, e, Xf, Yf,
        Zf, Tf, Xc, Yc, Zc, Tc, parity) -> int:
    return int(_LIBRARY.applyMultigridStrictRestrictQcu(
        _pointer(coarse_out), _pointer(fine_in), _pointer(null_vectors),
        _pointer(set_ptrs), _pointer(params), E, e, Xf, Yf, Zf, Tf, Xc, Yc,
        Zc, Tc, parity))


def applyMultigridStrictProLongQcu(
        fine_out, coarse_in, null_vectors, set_ptrs, params, E, e, Xf, Yf,
        Zf, Tf, Xc, Yc, Zc, Tc, parity) -> int:
    return int(_LIBRARY.applyMultigridStrictProLongQcu(
        _pointer(fine_out), _pointer(coarse_in), _pointer(null_vectors),
        _pointer(set_ptrs), _pointer(params), E, e, Xf, Yf, Zf, Tf, Xc, Yc,
        Zc, Tc, parity))


def applyMultigridStrictVCycleQcu(full_out, full_rhs, set_ptrs, params,
                                  start_level, allocated_bytes) -> int:
    return int(_LIBRARY.applyMultigridStrictVCycleQcu(
        _pointer(full_out), _pointer(full_rhs), _pointer(set_ptrs),
        _pointer(params), start_level, allocated_bytes))


def applyMultigridStrictInitQcu(set_ptrs, params, start_level,
                                allocated_bytes) -> int:
    """Initialize the reusable strict hierarchy in its fixed reserved slot.

    Unlike an ordinary transaction, the strict setup remains live across many
    operation/getter calls and ``_SET_INDEX_`` must remain fixed. Tear it down
    with :func:`applyMultigridStrictEndQcu` and then the enclosing
    :func:`applyEndQcu`.
    """
    return int(_LIBRARY.applyMultigridStrictInitQcu(
        _pointer(set_ptrs), _pointer(params), start_level, allocated_bytes))


def applyMultigridStrictEndQcu(set_ptrs, params) -> int:
    """Release the fixed-slot strict hierarchy before ``applyEndQcu``."""
    return int(_LIBRARY.applyMultigridStrictEndQcu(
        _pointer(set_ptrs), _pointer(params)))


def applyMultigridStrictFgmresQcu(
        full_out, full_rhs, gauge, ce, coo, cei, coi, fine_null_vectors,
        set_ptrs, params, fine_dims, coarse_dims, element_bytes, restart,
        max_iter, tolerance, nu_pre, nu_post, max_workspace_bytes,
        iterations, converged, final_true_residual, allocated_bytes) -> int:
    return int(_LIBRARY.applyMultigridStrictFgmresQcu(
        _pointer(full_out), _pointer(full_rhs), _pointer(gauge), _pointer(ce),
        _pointer(coo), _pointer(cei), _pointer(coi),
        _pointer(fine_null_vectors), _pointer(set_ptrs), _pointer(params),
        *fine_dims, *coarse_dims, element_bytes, restart, max_iter,
        tolerance, nu_pre, nu_post, max_workspace_bytes, iterations,
        converged, final_true_residual, allocated_bytes))


def applyCloverMultigridQcu(fermion_out, fermion_in, gauge, ce, coo, cei,
                            coi, set_ptrs, params) -> None:
    _LIBRARY.applyCloverMultigridQcu(
        _pointer(fermion_out), _pointer(fermion_in), _pointer(gauge),
        _pointer(ce), _pointer(coo), _pointer(cei), _pointer(coi),
        _pointer(set_ptrs), _pointer(params))


def verifyCloverMultigridQcu(fermion_out, fermion_in, gauge, ce, coo, cei,
                             coi, set_ptrs, params) -> int:
    return int(_LIBRARY.verifyCloverMultigridQcu(
        _pointer(fermion_out), _pointer(fermion_in), _pointer(gauge),
        _pointer(ce), _pointer(coo), _pointer(cei), _pointer(coi),
        _pointer(set_ptrs), _pointer(params)))
