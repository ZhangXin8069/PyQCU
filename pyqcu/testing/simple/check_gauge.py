#!/usr/bin/env python3
"""Check a QCU gauge or QUDA/QDP gauge field and its round-trip layout."""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from ._common import (qcu_gauge_to_qdp, qdp_gauge_to_qcu, read_h5,
                          read_qdp_gauge, resolve_path)
except ImportError:
    from _common import (qcu_gauge_to_qdp, qdp_gauge_to_qcu, read_h5,
                         read_qdp_gauge, resolve_path)


def _unitarity_error(gauge: np.ndarray) -> float:
    links = np.transpose(gauge, (2, 3, 4, 5, 6, 0, 1)).reshape(-1, 3, 3)
    eye = np.eye(3, dtype=links.dtype)
    return float(np.max(np.linalg.norm(links @ links.conj().transpose(0, 2, 1) - eye,
                                       axis=(-2, -1))) / np.sqrt(3.0))


def _self_test() -> tuple[np.ndarray, float]:
    gauge = np.zeros((3, 3, 4, 2, 2, 2, 4), dtype=np.complex128)
    gauge[0, 0] = gauge[1, 1] = gauge[2, 2] = 1.0
    return gauge, 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="QCU HDF5 or QUDA .lime/.npy")
    parser.add_argument("--backend", choices=("auto", "qcu", "quda"), default="auto")
    parser.add_argument("--dataset", default="data")
    args = parser.parse_args(argv)

    if args.input is None:
        qcu, expected_roundtrip = _self_test()
        source = "self-test"
    else:
        source_path = resolve_path(args.input, resolve_path(None, args.input))
        source = str(source_path)
        backend = args.backend
        if backend == "auto":
            backend = "quda" if source_path.suffix == ".lime" else "auto"
        if backend == "quda":
            qdp = read_qdp_gauge(source_path)
            qcu = qdp_gauge_to_qcu(qdp)
        elif backend == "qcu":
            qcu = read_h5(source_path, args.dataset)
        else:
            raw = read_h5(source_path, args.dataset) if source_path.suffix in (".h5", ".hdf5") \
                else np.load(source_path, allow_pickle=False)
            qcu = raw if raw.ndim == 7 and raw.shape[:3] == (3, 3, 4) \
                else qdp_gauge_to_qcu(raw)
        expected_roundtrip = 0.0 if backend == "qcu" else float(np.max(np.abs(
            qdp_gauge_to_qcu(qcu_gauge_to_qdp(qcu)) - qcu)))

    roundtrip = float(np.max(np.abs(qdp_gauge_to_qcu(qcu_gauge_to_qdp(qcu)) - qcu)))
    finite = bool(np.isfinite(qcu).all())
    unitarity = _unitarity_error(qcu)
    status = "PASS" if finite and roundtrip <= max(expected_roundtrip, 1e-6) and unitarity < 5e-3 else "FAIL"
    print(json.dumps({
        "source": source, "shape": list(qcu.shape), "dtype": str(qcu.dtype),
        "finite": finite, "roundtrip_max_error": roundtrip,
        "unitarity_error": unitarity, "status": status,
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
