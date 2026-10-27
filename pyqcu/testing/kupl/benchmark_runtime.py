#!/usr/bin/env python3
"""Measure KUPL init and first-Clover latency/RSS without solver iterations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

import pyqcu.cann as _torch
from pyqcu import lattice, tools
from pyqcu.cuda import define
from pyqcu.cuda import QcuRuntime
from pyqcu.kupl import qcu


def _status_bytes(name: str) -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _memory() -> dict[str, int | None]:
    lifetime_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "rss_bytes": _status_bytes("VmRSS"),
        "lifetime_hwm_bytes": _status_bytes("VmHWM") or lifetime_kib * 1024,
        "anonymous_rss_bytes": _status_bytes("RssAnon"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_delta(after: dict[str, int | None],
               before: dict[str, int | None],
               field: str) -> int | None:
    left = after.get(field)
    right = before.get(field)
    return None if left is None or right is None else left - right


def _validate_payload(payload: dict[str, object]) -> None:
    if payload.get("status") != "PASS":
        raise RuntimeError("runtime benchmark did not complete")
    memory = payload.get("memory")
    if not isinstance(memory, dict):
        raise RuntimeError("runtime benchmark memory schema is missing")
    for stage in ("before_init", "after_init", "after_inputs",
                  "after_first_clover"):
        value = memory.get(stage)
        if not isinstance(value, dict) or "rss_bytes" not in value or \
                "lifetime_hwm_bytes" not in value:
            raise RuntimeError(f"runtime benchmark stage {stage} is incomplete")
    deltas = memory.get("delta_from_before_init")
    if not isinstance(deltas, dict) or "after_init" not in deltas or \
            "after_inputs" not in deltas or \
            "after_first_clover" not in deltas:
        raise RuntimeError("runtime benchmark RSS deltas are missing")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", nargs=4, type=int, default=(16, 16, 16, 32))
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--csw", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if any(value <= 0 or value % 2 for value in args.lat):
        parser.error("--lat must contain four positive even extents")

    os.environ["KUPL_OMP_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    dtype = (_torch.complex64 if args.dtype == "c64"
             else _torch.complex128)
    compact = (*args.lat[:3], args.lat[3] // 2)

    runtime = QcuRuntime.from_lattice(
        args.lat, mass=args.mass, csw=args.csw, dtype=dtype, device="cpu",
        plan=define._SET_PLAN2_, parity=0, grid=(1, 1, 1, 1),
        qcu_module=qcu)
    baseline = _memory()

    started = time.perf_counter()
    qcu.applyInitQcu(runtime.set_ptrs, runtime.params, runtime.argv)
    init_seconds = time.perf_counter() - started
    after_init = _memory()

    try:
        started = time.perf_counter()
        full_gauge = _torch.torch.empty((3, 3, 4, *args.lat), dtype=dtype)
        lattice.generate_gauge_field(
            full_gauge, sigma=0.12, seed=args.seed, verbose=False)
        gauge = tools.oooxyzt2poooxyzt(full_gauge).contiguous()
        generator = _torch.torch.Generator().manual_seed(args.seed + 1)
        source = _torch.torch.complex(
            _torch.torch.randn(
                (4, 3, *compact), generator=generator, dtype=_torch.float32),
            _torch.torch.randn(
                (4, 3, *compact), generator=generator, dtype=_torch.float32),
        ).to(dtype).contiguous()
        output = _torch.torch.empty_like(source)
        prepare_inputs_seconds = time.perf_counter() - started
        after_inputs = _memory()

        started = time.perf_counter()
        qcu.applyCloverDslashQcu(
            output, source, gauge, runtime.set_ptrs, runtime.params)
        first_clover_seconds = time.perf_counter() - started
        after_clover = _memory()
    finally:
        qcu.applyEndQcu(runtime.set_ptrs, runtime.params)

    library_name = Path(qcu._LIBRARY._name).resolve()
    payload = {
        "status": "PASS",
        "library": str(library_name),
        "library_sha256": _sha256(library_name),
        "lattice": list(args.lat),
        "local_shape": list(compact),
        "dtype": args.dtype,
        "threads": int(_torch.torch.get_num_threads()),
        "kupl_omp_threads": os.environ.get("KUPL_OMP_THREADS"),
        "kupl_numa_bind": os.environ.get("KUPL_NUMA_BIND"),
        "kupl_memory_policy": os.environ.get("KUPL_MEMORY_POLICY"),
        "kupl_hugepage": os.environ.get("KUPL_HUGEPAGE"),
        "kupl_fused_bicg_dots": os.environ.get("KUPL_FUSED_BICG_DOTS"),
        "prepare_inputs_seconds": prepare_inputs_seconds,
        "init_seconds": init_seconds,
        "first_clover_seconds": first_clover_seconds,
        "memory": {
            "before_init": baseline,
            "after_init": after_init,
            "after_inputs": after_inputs,
            "after_first_clover": after_clover,
            "delta_from_before_init": {
                "after_init": {
                    "rss_bytes": _rss_delta(after_init, baseline, "rss_bytes"),
                    "anonymous_rss_bytes": _rss_delta(
                        after_init, baseline, "anonymous_rss_bytes"),
                },
                "after_inputs": {
                    "rss_bytes": _rss_delta(
                        after_inputs, baseline, "rss_bytes"),
                    "anonymous_rss_bytes": _rss_delta(
                        after_inputs, baseline, "anonymous_rss_bytes"),
                },
                "after_first_clover": {
                    "rss_bytes": _rss_delta(
                        after_clover, baseline, "rss_bytes"),
                    "anonymous_rss_bytes": _rss_delta(
                        after_clover, baseline, "anonymous_rss_bytes"),
                },
            },
        },
    }
    _validate_payload(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
