#!/usr/bin/env python3
"""Convert a full PyQCU fermion HDF5 field to QUDA's compressed QDP layout."""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from ._common import qcu_fermion_to_qdp, read_h5, resolve_path, write_array
except ImportError:
    from _common import qcu_fermion_to_qdp, read_h5, resolve_path, write_array


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="source_24x24x24x64_c64.h5")
    parser.add_argument("--output", default="source_quda_24x24x24x64_qdp.npy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    source = resolve_path(args.input, resolve_path(None, args.input))
    qcu = read_h5(source)
    qdp = qcu_fermion_to_qdp(qcu)
    output = resolve_path(args.output, source.parent / args.output)
    if not args.dry_run:
        write_array(output, qdp, metadata={
            "layout": "[q,t,z,y,x/2,spin,color]",
            "source": str(source),
        })
    print(json.dumps({
        "input": str(source), "output": None if args.dry_run else str(output),
        "qcu_shape": list(qcu.shape), "quda_shape": list(qdp.shape),
        "dtype": str(qdp.dtype), "norm": float(np.linalg.norm(qdp.ravel())),
        "dry_run": bool(args.dry_run),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
