#!/usr/bin/env python3
"""Check a QCU clover field or a packed QUDA clover array."""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from ._common import (qcu_clover_to_quda, quda_clover_to_qcu, read_array,
                          read_h5_metadata, resolve_path)
except ImportError:
    from _common import (qcu_clover_to_quda, quda_clover_to_qcu, read_array,
                         read_h5_metadata, resolve_path)


def _self_test() -> np.ndarray:
    rng = np.random.default_rng(7)
    block = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    block = block + block.conj().T
    matrix = np.zeros((12, 12), dtype=np.complex128)
    indices = np.array([0, 1, 2, 9, 10, 11])
    matrix[np.ix_(indices, indices)] = block
    return np.broadcast_to(matrix.reshape(4, 3, 4, 3, 1, 1, 1, 1),
                           (4, 3, 4, 3, 2, 1, 1, 1)).copy()


def _matrix_roundtrip(clover: np.ndarray) -> tuple[np.ndarray, float]:
    packed = qcu_clover_to_quda(clover)
    restored = quda_clover_to_qcu(packed)
    return packed, float(np.max(np.abs(restored - clover)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--backend", choices=("auto", "qcu", "quda"), default="auto")
    parser.add_argument("--dataset", default="data")
    args = parser.parse_args(argv)

    metadata = {}
    if args.input is None:
        clover = _self_test()
        source = "self-test"
    else:
        path = resolve_path(args.input, resolve_path(None, args.input))
        source = str(path)
        raw = read_array(path, args.dataset)
        backend = args.backend
        if backend == "auto":
            backend = "qcu" if raw.ndim == 8 and raw.shape[:4] == (4, 3, 4, 3) else "quda"
        if backend == "qcu":
            clover = raw
            if path.suffix in (".h5", ".hdf5"):
                metadata = read_h5_metadata(path)
        else:
            clover = quda_clover_to_qcu(raw)

    packed, error = _matrix_roundtrip(clover)
    finite = bool(np.isfinite(clover).all())
    status = "PASS" if finite and error <= 1e-6 else "FAIL"
    print(json.dumps({
        "source": source, "qcu_shape": list(clover.shape),
        "quda_shape": list(packed.shape), "dtype": str(clover.dtype),
        "finite": finite, "roundtrip_max_error": error,
        "metadata": metadata, "status": status,
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
