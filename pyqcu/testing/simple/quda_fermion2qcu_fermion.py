#!/usr/bin/env python3
"""Convert a QUDA compressed QDP fermion field to full PyQCU layout."""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from ._common import qdp_fermion_to_qcu, read_array, resolve_path, write_h5
except ImportError:
    from _common import qdp_fermion_to_qcu, read_array, resolve_path, write_h5


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="source_quda_24x24x24x64_qdp.npy")
    parser.add_argument("--output", default="source_24x24x24x64_c64.h5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    source = resolve_path(args.input, resolve_path(None, args.input))
    qdp = read_array(source)
    qcu = qdp_fermion_to_qcu(qdp)
    output = resolve_path(args.output, source.parent / args.output)
    if not args.dry_run:
        write_h5(output, qcu, metadata={
            "layout": "[spin,color,x,y,z,t]",
            "quda_layout": "[q,t,z,y,x/2,spin,color]",
            "source": str(source),
        })
    print(json.dumps({
        "input": str(source), "output": None if args.dry_run else str(output),
        "quda_shape": list(qdp.shape), "qcu_shape": list(qcu.shape),
        "dtype": str(qcu.dtype), "norm": float(np.linalg.norm(qcu.ravel())),
        "dry_run": bool(args.dry_run),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
