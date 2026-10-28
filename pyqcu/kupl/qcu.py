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

ctypes passes only raw 64-bit addresses. This module rejects non-contiguous or
non-CPU tensors, while callers remain responsible for ABI-compatible shapes,
dtypes, and lifetime retention. CUDA production remains one Python process per
MPI rank and device; KUPL uses the same process/rank partitioning for
CPU/OpenMP work.
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
    "applyMultigridLocalOrthogonalizeQcu": (_LL,) * 2 + (_I,) * 11,
    "applyMultigridBuildNullVectorsQcu": (
        (_LL,) * 8 + (_I,) * 11 + (_ULL, _D)),
    "applyMultigridBuildStencilLocalQcu": (_LL,) * 11 + (_I,) * 12,
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
    "applyMultigridLocalOrthogonalizeQcu",
    "applyMultigridBuildNullVectorsQcu",
    "applyMultigridBuildStencilLocalQcu",
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
    the storage alive. The caller must provide a contiguous torch CPU tensor or
    a host-array adapter of the exact expected ABI type and retain it for the
    duration of the native call.
    """
    if hasattr(tensor, "device"):
        if tensor.device.type != "cpu":
            raise ValueError("KUPL QCU tensors must be CPU tensors")
    else:
        # The simple benchmark adapters wrap host NumPy arrays without
        # exposing a torch.device.  Keep that stable CPU ABI, but do not
        # accept arbitrary pointer-like objects.
        array = getattr(tensor, "array", None)
        flags = getattr(array, "flags", None)
        if flags is None or not bool(flags.c_contiguous):
            raise ValueError(
                "KUPL QCU adapters must wrap a contiguous host array")
    if not tensor.is_contiguous():
        raise ValueError("KUPL QCU tensors must be contiguous")
    return int(tensor.data_ptr())


def _scalar_buffer_pointer(
        tensor: Any, name: str, expected_dtypes: set[str],
        pointer_type: Any) -> Any:
    """Expose one writable tensor element through an exact ctypes pointer."""
    if int(tensor.numel()) != 1:
        raise ValueError(f"KUPL {name} must contain exactly one element")
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name not in expected_dtypes:
        allowed = "/".join(sorted(expected_dtypes))
        raise TypeError(f"KUPL {name} must have dtype {allowed}")
    address = _pointer(tensor)
    return ctypes.cast(ctypes.c_void_p(address), pointer_type)


def _strict_fgmres_geometry(
        full_out: Any, full_rhs: Any, gauge: Any, clover_even: Any,
        fine_null_vectors: Any, fine_dims: Any,
        coarse_dims: Any) -> tuple[int, int]:
    """Derive the two DOF counts omitted from the public Python signature."""
    fine = tuple(int(value) for value in fine_dims)
    coarse = tuple(int(value) for value in coarse_dims)
    if len(fine) != 4 or any(value <= 0 for value in fine):
        raise ValueError("KUPL fine_dims must contain four positive integers")
    if len(coarse) != 4 or any(value <= 0 for value in coarse):
        raise ValueError("KUPL coarse_dims must contain four positive integers")
    if fine[3] % 2:
        raise ValueError("KUPL strict FGMRES fine T extent must be even")

    expected_full = (2, 4, 3, fine[0], fine[1], fine[2], fine[3] // 2)
    if tuple(full_rhs.shape) != expected_full:
        raise ValueError(
            f"KUPL strict FGMRES rhs must have shape {expected_full}")
    if tuple(full_out.shape) != expected_full:
        raise ValueError("KUPL strict FGMRES output/rhs shapes must match")
    if int(full_rhs.shape[1] * full_rhs.shape[2]) != 12:
        raise ValueError("KUPL strict FGMRES requires 12 fine components")
    compact_shape = (*fine[:3], fine[3] // 2)
    expected_gauge = (2, 3, 3, 4, *compact_shape)
    expected_clover = (4, 3, 4, 3, *compact_shape)
    if tuple(gauge.shape) != expected_gauge:
        raise ValueError(
            f"KUPL strict FGMRES gauge must have shape {expected_gauge}")
    if tuple(clover_even.shape) != expected_clover:
        raise ValueError(
            f"KUPL strict FGMRES Clover terms must have shape {expected_clover}")

    if fine_null_vectors.ndim != 10:
        raise ValueError("KUPL strict null vectors require a 10-D blocked layout")
    if int(fine_null_vectors.shape[1]) != 12:
        raise ValueError("KUPL strict null vectors require 12 fine components")
    blocked_coarse = tuple(
        int(fine_null_vectors.shape[2 + 2 * axis]) for axis in range(4))
    expected_fine = tuple(
        int(blocked_coarse[axis] * fine_null_vectors.shape[3 + 2 * axis])
        for axis in range(4))
    if blocked_coarse != coarse:
        raise ValueError(
            "KUPL strict null-vector coarse geometry does not match coarse_dims")
    if expected_fine != fine:
        raise ValueError(
            "KUPL strict null-vector blocking does not match fine_dims")

    reference_dtype = full_rhs.dtype
    data_fields = (
        full_out, full_rhs, fine_null_vectors)
    if any(field.dtype != reference_dtype for field in data_fields):
        raise TypeError("KUPL strict fields and null vectors must share dtype")
    return 12, int(fine_null_vectors.shape[0])


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


def applyMultigridLocalOrthogonalizeQcu(
        flat_in, blocked_out, E: int, e: int, Xf: int, Yf: int, Zf: int,
        Tf: int, Xc: int, Yc: int, Zc: int, Tc: int, data_type: int) -> int:
    if tuple(flat_in.shape) != (E, e, Xf, Yf, Zf, Tf):
        raise ValueError("flat null-vector shape does not match geometry")
    flat_size = flat_in.numel() if hasattr(flat_in, "numel") else flat_in.size
    blocked_size = (blocked_out.numel() if hasattr(blocked_out, "numel")
                    else blocked_out.size)
    if flat_size != blocked_size:
        raise ValueError("flat input and blocked output sizes differ")
    return int(_LIBRARY.applyMultigridLocalOrthogonalizeQcu(
        _pointer(flat_in), _pointer(blocked_out), E, e, Xf, Yf, Zf, Tf,
        Xc, Yc, Zc, Tc, data_type))


def applyMultigridBuildNullVectorsQcu(
        blocked_out, gauge, clover_even, clover_odd, clover_even_inv,
        clover_odd_inv, set_ptrs, params, E: int, e: int, Xf: int, Yf: int,
        Zf: int, Tf: int, Xc: int, Yc: int, Zc: int, Tc: int,
        nv_iters: int = 1, seed: int = 42, nv_tol: float = 1.0e-2) -> int:
    expected = (E, e, Xc, Xf // Xc, Yc, Yf // Yc, Zc, Zf // Zc, Tc,
                Tf // Tc)
    if tuple(blocked_out.shape) != expected:
        raise ValueError("blocked null-vector shape does not match geometry")
    if E <= 0 or e <= 0 or min(Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc) <= 0:
        raise ValueError("null-vector dimensions must be positive")
    if Xf % Xc or Yf % Yc or Zf % Zc or Tf % Tc:
        raise ValueError("fine null-vector geometry must be block-divisible")
    if nv_iters < 0 or nv_tol <= 0.0:
        raise ValueError("null-vector iteration controls are invalid")
    return int(_LIBRARY.applyMultigridBuildNullVectorsQcu(
        _pointer(blocked_out), _pointer(gauge), _pointer(clover_even),
        _pointer(clover_odd), _pointer(clover_even_inv),
        _pointer(clover_odd_inv), _pointer(set_ptrs), _pointer(params), E, e,
        Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, nv_iters, seed, nv_tol))


def applyMultigridBuildStencilLocalQcu(
        sitting, hop_nn, hop_diag, null_vectors, gauge, clover_even,
        clover_odd, clover_even_inv, clover_odd_inv, set_ptrs, params,
        E: int, e: int, Xf: int, Yf: int, Zf: int, Tf: int, Xc: int,
        Yc: int, Zc: int, Tc: int, window: int = 8,
        batch_size: int = 8) -> int:
    coarse_shape = (E, E, Xc, Yc, Zc, Tc)
    if tuple(sitting.shape) != coarse_shape:
        raise ValueError("sitting stencil shape does not match geometry")
    if tuple(hop_nn.shape) != (2, 4, *coarse_shape):
        raise ValueError("nearest stencil shape does not match geometry")
    if tuple(hop_diag.shape) != (2, 2, 6, *coarse_shape):
        raise ValueError("diagonal stencil shape does not match geometry")
    expected_null = (
        E, e, Xc, Xf // Xc, Yc, Yf // Yc, Zc, Zf // Zc, Tc, Tf // Tc)
    if tuple(null_vectors.shape) != expected_null:
        raise ValueError("null-vector stencil shape does not match geometry")
    if not (Xf % Xc == Yf % Yc == Zf % Zc == Tf % Tc == 0):
        raise ValueError("fine stencil geometry must be block-divisible")
    if window <= 0 or batch_size <= 0:
        raise ValueError("stencil window and batch size must be positive")
    return int(_LIBRARY.applyMultigridBuildStencilLocalQcu(
        _pointer(sitting), _pointer(hop_nn), _pointer(hop_diag),
        _pointer(null_vectors), _pointer(gauge), _pointer(clover_even),
        _pointer(clover_odd), _pointer(clover_even_inv),
        _pointer(clover_odd_inv), _pointer(set_ptrs), _pointer(params), E, e,
        Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, window, batch_size))


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
    allocated_pointer = _scalar_buffer_pointer(
        allocated_bytes, "allocated_bytes", {"int64", "uint64"}, _ULL_P)
    return int(_LIBRARY.applyMultigridStrictVCycleQcu(
        _pointer(full_out), _pointer(full_rhs), _pointer(set_ptrs),
        _pointer(params), start_level, allocated_pointer))


def applyMultigridStrictInitQcu(set_ptrs, params, start_level,
                                allocated_bytes) -> int:
    """Initialize the reusable strict hierarchy in its fixed reserved slot.

    Unlike an ordinary transaction, the strict setup remains live across many
    operation/getter calls and ``_SET_INDEX_`` must remain fixed. Tear it down
    with :func:`applyMultigridStrictEndQcu` and then the enclosing
    :func:`applyEndQcu`.
    """
    allocated_pointer = _scalar_buffer_pointer(
        allocated_bytes, "allocated_bytes", {"int64", "uint64"}, _ULL_P)
    return int(_LIBRARY.applyMultigridStrictInitQcu(
        _pointer(set_ptrs), _pointer(params), start_level, allocated_pointer))


def applyMultigridStrictEndQcu(set_ptrs, params) -> int:
    """Release the fixed-slot strict hierarchy before ``applyEndQcu``."""
    return int(_LIBRARY.applyMultigridStrictEndQcu(
        _pointer(set_ptrs), _pointer(params)))


def applyMultigridStrictFgmresQcu(
        full_out, full_rhs, gauge, ce, coo, cei, coi, fine_null_vectors,
        set_ptrs, params, fine_dims, coarse_dims, element_bytes, restart,
        max_iter, tolerance, nu_pre, nu_post, max_workspace_bytes,
        iterations, converged, final_true_residual, allocated_bytes) -> int:
    fine_E, coarse_E = _strict_fgmres_geometry(
        full_out, full_rhs, gauge, ce, fine_null_vectors,
        fine_dims, coarse_dims)
    reference_dtype = full_rhs.dtype
    if any(field.dtype != reference_dtype for field in (gauge, ce, coo, cei, coi)):
        raise TypeError("KUPL strict FGMRES fields must share one complex dtype")
    if any(tuple(field.shape) != tuple(ce.shape) for field in (coo, cei, coi)):
        raise ValueError("KUPL strict FGMRES Clover tensor shapes must match")
    if int(element_bytes) != int(full_rhs.element_size()):
        raise ValueError("KUPL strict FGMRES element_bytes does not match rhs")

    iterations_pointer = _scalar_buffer_pointer(
        iterations, "iterations", {"int32"}, _I_P)
    converged_pointer = _scalar_buffer_pointer(
        converged, "converged", {"int32"}, _I_P)
    residual_pointer = _scalar_buffer_pointer(
        final_true_residual, "final_true_residual", {"float64"}, _D_P)
    allocated_pointer = _scalar_buffer_pointer(
        allocated_bytes, "allocated_bytes", {"int64", "uint64"}, _ULL_P)
    return int(_LIBRARY.applyMultigridStrictFgmresQcu(
        _pointer(full_out), _pointer(full_rhs), _pointer(gauge), _pointer(ce),
        _pointer(coo), _pointer(cei), _pointer(coi),
        _pointer(fine_null_vectors), _pointer(set_ptrs), _pointer(params),
        fine_E, *fine_dims, coarse_E, *coarse_dims, element_bytes, restart,
        max_iter, tolerance, nu_pre, nu_post, max_workspace_bytes,
        iterations_pointer, converged_pointer, residual_pointer,
        allocated_pointer))


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
