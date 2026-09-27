"""Initialize QMP before importing PyQUDA or QUDA."""
from __future__ import annotations

import atexit
import ctypes
import os
from pathlib import Path
from typing import Sequence

_QMP_HOLD: list[object] = []


def initialize_qmp(topology: Sequence[int] = (1, 1, 1, 1)) -> dict:
    """Initialize the QMP transport and declare a logical topology."""
    dims = tuple(int(value) for value in topology)
    if len(dims) != 4 or any(value <= 0 for value in dims):
        raise ValueError("QMP topology must contain four positive extents")

    prefix = os.environ.get("QUDA_INSTALL") or os.environ.get("QUDA_PATH")
    if not prefix:
        raise RuntimeError(
            "QUDA_INSTALL or QUDA_PATH is required to locate libqmp.so; "
            "source pyqcu/testing/qcu/strict/quda_comparison/quda_env.sh")
    library = Path(prefix).expanduser().resolve() / "lib" / "libqmp.so"
    if not library.is_file():
        raise RuntimeError(f"missing QMP library: {library}")

    try:
        qmp = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        qmp.QMP_is_initialized.argtypes = []
        qmp.QMP_is_initialized.restype = ctypes.c_int
        qmp.QMP_logical_topology_is_declared.argtypes = []
        qmp.QMP_logical_topology_is_declared.restype = ctypes.c_int
        qmp.QMP_declare_logical_topology.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        qmp.QMP_declare_logical_topology.restype = ctypes.c_int
        qmp.QMP_get_logical_number_of_dimensions.argtypes = []
        qmp.QMP_get_logical_number_of_dimensions.restype = ctypes.c_int
        qmp.QMP_get_logical_dimensions.argtypes = []
        qmp.QMP_get_logical_dimensions.restype = ctypes.POINTER(ctypes.c_int)
        qmp.QMP_finalize_msg_passing.argtypes = []
        qmp.QMP_finalize_msg_passing.restype = None
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"cannot load QMP runtime {library}: {exc}") from exc

    initialized_here = qmp.QMP_is_initialized() != 1
    provided_value = None
    if initialized_here:
        qmp.QMP_init_msg_passing.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        qmp.QMP_init_msg_passing.restype = ctypes.c_int
        argc = ctypes.c_int(1)
        argv_storage = (ctypes.c_char_p * 2)(b"pyqcu-pyquda-runner", None)
        argv = ctypes.cast(argv_storage, ctypes.POINTER(ctypes.c_char_p))
        provided = ctypes.c_int(-1)
        status = int(qmp.QMP_init_msg_passing(
            ctypes.byref(argc), ctypes.byref(argv), 1,
            ctypes.byref(provided)))
        if status != 0 or qmp.QMP_is_initialized() != 1:
            raise RuntimeError(
                f"QMP initialization failed: status={status}, "
                f"provided={provided.value}")
        provided_value = int(provided.value)
        atexit.register(qmp.QMP_finalize_msg_passing)
        _QMP_HOLD.extend((qmp, argv_storage, argv))
    else:
        _QMP_HOLD.append(qmp)

    if not qmp.QMP_logical_topology_is_declared():
        topology_dims = (ctypes.c_int * 4)(*dims)
        status = int(qmp.QMP_declare_logical_topology(topology_dims, 4))
        if status != 0 or not qmp.QMP_logical_topology_is_declared():
            raise RuntimeError(
                f"QMP topology declaration failed: status={status}, dims={dims}")

    ndim = int(qmp.QMP_get_logical_number_of_dimensions())
    dims_ptr = qmp.QMP_get_logical_dimensions()
    if ndim <= 0 or not bool(dims_ptr):
        raise RuntimeError(f"QMP topology readback failed: ndim={ndim}")
    resolved = tuple(int(dims_ptr[index]) for index in range(ndim))
    if resolved != dims:
        raise RuntimeError(
            f"QMP topology mismatch: expected {dims}, got {resolved}")
    return {
        "library": str(library),
        "initialized_here": initialized_here,
        "thread_level_provided": provided_value,
        "logical_topology": list(resolved),
    }
