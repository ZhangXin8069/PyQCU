#!/usr/bin/env python3
"""Convert a PyQCU clover HDF5 field to QUDA's packed real layout."""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from ._common import (qcu_clover_to_quda, read_h5, read_h5_metadata,
                          resolve_path, write_array)
except ImportError:
    from _common import (qcu_clover_to_quda, read_h5, read_h5_metadata,
                         resolve_path, write_array)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="clover_24x24x24x64_c64.h5")
    parser.add_argument("--output", default="clover_quda_24x24x24x64_packed.npz")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    source = resolve_path(args.input, resolve_path(None, args.input))
    qcu = read_h5(source)
    source_metadata = read_h5_metadata(source)
    packed = qcu_clover_to_quda(qcu)
    output = resolve_path(args.output, source.parent / args.output)
    if not args.dry_run:
        write_array(output, packed, metadata={
            "layout": "[q,t,z,y,x/2,chirality,36]",
            "source": str(source),
            "csw": source_metadata.get("csw"),
            "mass": source_metadata.get("mass"),
            "note": "PACKED_CLOVER_ORDER",
        })
    print(json.dumps({
        "input": str(source), "output": None if args.dry_run else str(output),
        "qcu_shape": list(qcu.shape), "quda_shape": list(packed.shape),
        "dtype": str(packed.dtype), "norm": float(np.linalg.norm(packed.ravel())),
        "dry_run": bool(args.dry_run),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
