#!/usr/bin/env python3
"""Check a QCU fermion field or QUDA compressed QDP fermion field."""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from ._common import (qcu_fermion_to_qdp, qdp_fermion_to_qcu, read_array,
                          resolve_path)
except ImportError:
    from _common import (qcu_fermion_to_qdp, qdp_fermion_to_qcu, read_array,
                         resolve_path)


def _self_test() -> np.ndarray:
    return np.zeros((4, 3, 2, 2, 2, 4), dtype=np.complex64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--backend", choices=("auto", "qcu", "quda"), default="auto")
    parser.add_argument("--dataset", default="data")
    args = parser.parse_args(argv)

    if args.input is None:
        qcu = _self_test()
        source = "self-test"
    else:
        path = resolve_path(args.input, resolve_path(None, args.input))
        source = str(path)
        raw = read_array(path, args.dataset)
        backend = args.backend
        if backend == "auto":
            backend = "qcu" if raw.ndim == 6 and raw.shape[:2] == (4, 3) else "quda"
        qcu = raw if backend == "qcu" else qdp_fermion_to_qcu(raw)

    finite = bool(np.isfinite(qcu).all())
    qdp = qcu_fermion_to_qdp(qcu)
    restored = qdp_fermion_to_qcu(qdp)
    error = float(np.max(np.abs(restored - qcu)))
    status = "PASS" if finite and error <= 1e-6 else "FAIL"
    print(json.dumps({
        "source": source, "shape": list(qcu.shape), "dtype": str(qcu.dtype),
        "finite": finite, "roundtrip_max_error": error,
        "norm": float(np.linalg.norm(qcu.ravel())), "status": status,
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
