#!/usr/bin/env python3
"""Run one isolated CUDA QCU Legacy or Strict MG benchmark side.

The case files are canonical QCU HDF5 tensors.  Each invocation owns one
Python process, one MPI rank, and one CUDA device.  Legacy asset construction
and solve can be run as separate subcommands so MG initialization time is not
hidden in steady-solve timing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


_LEGACY_CACHE_MANIFEST_ATTR = "pyqcu_legacy_mg_manifest"
_LEGACY_CACHE_MANIFEST_SCHEMA = "pyqcu.legacy_mg_cache.v1"


def _validate_full_block(block: Sequence[int]) -> tuple[int, int, int, int]:
    """Return a validated full-lattice MG block ``(bx, by, bz, bt)``."""
    values = tuple(int(value) for value in block)
    if len(values) != 4:
        raise ValueError(f"full block must contain four extents, got {values}")
    if values[3] <= 0 or values[3] % 2:
        raise ValueError(
            "full block time extent bt must be a positive even integer, "
            f"got {values[3]}")
    if any(value <= 0 for value in values[:3]):
        raise ValueError(f"full block extents must be positive, got {values}")
    return values  # type: ignore[return-value]


def _legacy_parity_block(
        full_block: Sequence[int]) -> tuple[int, int, int, int]:
    """Map a full-lattice block to Legacy's parity-compressed block."""
    bx, by, bz, bt = _validate_full_block(full_block)
    return bx, by, bz, bt // 2


def _legacy_cache_block_from_lonv_shape(
        shape: Sequence[int], *, level: int | None = None) -> tuple[int, int, int, int]:
    """Recover the blocked Legacy coarsening width from a 10-D ``lonv`` shape."""
    values = tuple(int(value) for value in shape)
    location = "" if level is None else f" at level {level}"
    if len(values) != 10:
        raise ValueError(
            f"Legacy lonv{location} must be 10-D, got shape {values}")
    block = tuple(values[index] for index in (3, 5, 7, 9))
    if any(value <= 0 for value in block):
        raise ValueError(
            f"Legacy lonv{location} has invalid block {block}: {values}")
    return block  # type: ignore[return-value]


def _validate_legacy_cache_block(
        shape: Sequence[int], expected: Sequence[int], level: int,
        ) -> tuple[int, int, int, int]:
    """Validate one Legacy cache's physical block mapping, failing closed."""
    actual = _legacy_cache_block_from_lonv_shape(shape, level=level)
    expected_block = tuple(int(value) for value in expected)
    if actual != expected_block:
        raise ValueError(
            f"Legacy coarse cache level {level} block mismatch: "
            f"actual={actual} expected={expected_block}")
    return actual


def _legacy_cache_physical_shape(
        shape: Sequence[int], *, level: int | None = None,
        ) -> tuple[int, int, int, int]:
    """Recover the full-lattice coarse shape represented by a Legacy cache."""
    values = tuple(int(value) for value in shape)
    location = "" if level is None else f" at level {level}"
    if len(values) != 10:
        raise ValueError(
            f"Legacy lonv{location} must be 10-D, got shape {values}")
    physical = (values[2], values[4], values[6], 2 * values[8])
    if any(value <= 0 for value in physical):
        raise ValueError(
            f"Legacy lonv{location} has invalid physical coarse shape "
            f"{physical}: {values}")
    return physical  # type: ignore[return-value]


def _physical_coarse_shapes(
        lattice: Sequence[int], full_block: Sequence[int],
        levels: int) -> list[tuple[int, int, int, int]]:
    """Return full-lattice coarse shapes after each hierarchy transition."""
    fine = tuple(int(value) for value in lattice)
    block = _validate_full_block(full_block)
    if len(fine) != 4 or any(value <= 0 for value in fine):
        raise ValueError(
            f"lattice must contain four positive extents, got {fine}")
    count = int(levels) - 1
    if count < 0:
        raise ValueError(f"levels must be positive, got {levels}")
    shapes: list[tuple[int, int, int, int]] = []
    shape = fine
    for level in range(1, count + 1):
        if any(extent % width for extent, width in zip(shape, block)):
            raise ValueError(
                f"physical coarsening level {level} cannot divide "
                f"shape={shape} by full block={block}")
        shape = tuple(extent // width for extent, width in zip(shape, block))
        shapes.append(shape)  # type: ignore[arg-type]
    return shapes


def _legacy_cache_manifest(
        *, level: int, lattice: Sequence[int], full_block: Sequence[int],
        legacy_block: Sequence[int], physical_shape: Sequence[int],
        gauge_sha256: str, mass: float, csw: float, coarse_dof: int,
        nv_iters: int, local_stencil: bool,
        local_stencil_batch_size: int, null_batch_chunk: int | None,
        ) -> dict[str, Any]:
    """Return the provenance stored beside a generated Legacy cache."""
    return {
        "schema": _LEGACY_CACHE_MANIFEST_SCHEMA,
        "level": int(level),
        "lattice": [int(value) for value in lattice],
        "block_size_full": [int(value) for value in full_block],
        "legacy_parity_block": [int(value) for value in legacy_block],
        "physical_coarse_shape": [int(value) for value in physical_shape],
        "gauge_sha256": str(gauge_sha256),
        "mass": float(mass),
        "csw": float(csw),
        "precision": "c64",
        "coarse_dof": int(coarse_dof),
        "nv_iters": int(nv_iters),
        "local_stencil": bool(local_stencil),
        "local_stencil_batch_size": int(local_stencil_batch_size),
        "null_batch_chunk": (
            None if null_batch_chunk is None else int(null_batch_chunk)),
    }


def _legacy_cache_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(manifest), sort_keys=True, separators=(",", ":"))


def _decode_legacy_cache_manifest(
        raw: Any, *, path: Path, level: int) -> dict[str, Any]:
    """Decode a cache provenance attribute or fail with a precise error."""
    if raw is None:
        raise ValueError(
            f"Legacy coarse cache level {level} lacks provenance attribute "
            f"{_LEGACY_CACHE_MANIFEST_ATTR!r}: {path}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        manifest = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Legacy coarse cache level {level} has invalid provenance: "
            f"{path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Legacy coarse cache level {level} provenance must be an object: "
            f"{path}")
    return manifest


def _validate_legacy_cache_manifest(
        actual: Mapping[str, Any], expected: Mapping[str, Any], *,
        path: Path, level: int) -> dict[str, Any]:
    """Require exact cache provenance for every decision-bearing input."""
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            matches = (
                isinstance(actual_value, (int, float)) and
                float(actual_value) == expected_value)
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches.append(
                f"{key}: actual={actual_value!r} expected={expected_value!r}")
    if mismatches:
        raise ValueError(
            f"Legacy coarse cache level {level} provenance mismatch: {path}: "
            + "; ".join(mismatches))
    return dict(actual)


def _strict_hierarchy_transitions(
        transfers: Sequence[Any], lattice: Sequence[int],
        full_block: Sequence[int], levels: int) -> list[dict[str, Any]]:
    """Validate the materialized Strict hierarchy against the shared geometry."""
    expected_shapes = _physical_coarse_shapes(lattice, full_block, levels)
    expected_block = _validate_full_block(full_block)
    if len(transfers) != len(expected_shapes):
        raise ValueError(
            f"Strict hierarchy has {len(transfers)} transitions, expected "
            f"{len(expected_shapes)} for levels={levels}")
    transitions: list[dict[str, Any]] = []
    for level, (transfer, expected_shape) in enumerate(
            zip(transfers, expected_shapes), start=1):
        actual_block = tuple(int(value) for value in transfer.block_size)
        actual_fine = tuple(int(value) for value in transfer.fine_shape)
        actual_coarse = tuple(int(value) for value in transfer.coarse_shape)
        if actual_block != expected_block:
            raise ValueError(
                f"Strict hierarchy level {level} block mismatch: "
                f"actual={actual_block} expected={expected_block}")
        if actual_coarse != expected_shape:
            raise ValueError(
                f"Strict hierarchy level {level} coarse shape mismatch: "
                f"actual={actual_coarse} expected={expected_shape}")
        transitions.append({
            "level": level,
            "block": list(actual_block),
            "fine_shape": list(actual_fine),
            "coarse_shape": list(actual_coarse),
        })
    return transitions


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _json_mad(values: list[float]) -> float | None:
    if not values:
        return None
    center = statistics.median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _depth_warnings(levels: int, variant: str, lattice: Sequence[int],
                    block: Sequence[int]) -> list[str]:
    from pyqcu._multigrid_warnings import (
        MultigridDepthWarning,
        warn_excessive_multigrid_levels,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MultigridDepthWarning)
        warn_excessive_multigrid_levels(
            levels, variant=variant, backend="cuda",
            context="benchmark", lattice=lattice, block=block,
            process_grid=(1, 1, 1, 1))
    return [str(item.message) for item in caught
            if issubclass(item.category, MultigridDepthWarning)]


def _load_case(gauge_path: Path, source_path: Path, lat: tuple[int, ...],
               device: Any):
    import torch
    from pyqcu import tools
    from pyqcu.testing.simple._common import read_h5

    gauge_full = torch.from_numpy(read_h5(gauge_path)).to(
        device=device, dtype=torch.complex64).contiguous()
    source_full = torch.from_numpy(read_h5(source_path)).to(
        device=device, dtype=torch.complex64).contiguous()
    if tuple(int(value) for value in gauge_full.shape[-4:]) != lat:
        raise ValueError(f"gauge shape {gauge_full.shape[-4:]} != {lat}")
    if tuple(int(value) for value in source_full.shape[-4:]) != lat:
        raise ValueError(f"source shape {source_full.shape[-4:]} != {lat}")
    gauge = tools.oooxyzt2poooxyzt(gauge_full).contiguous()
    rhs = tools.oooxyzt2poooxyzt(source_full).contiguous()
    return gauge_full, source_full, gauge, rhs


def _fine_clover(gauge_full: Any, mass: float, csw: float, device: Any):
    import torch
    from pyqcu.torch import dslash

    kappa = 1.0 / (2.0 * mass + 8.0)
    kappa_t = torch.tensor([kappa], dtype=torch.float32, device=device)
    u0 = torch.ones(1, dtype=torch.float32, device=device)
    clover = dslash.make_clover(
        gauge_full, kappa=kappa_t, u_0=u0, csw=csw)
    return clover, kappa, kappa_t, u0


def _fine_operator(gauge_full: Any, lat: tuple[int, ...], mass: float,
                   csw: float, device: Any, *, parity_only: bool = False):
    from pyqcu.torch import dslash

    clover, kappa, kappa_t, u0 = _fine_clover(
        gauge_full, mass, csw, device)
    operator = dslash.operator(
        U=gauge_full, clover_term=clover, kappa=kappa_t, u_0=u0,
        support_parity=True, parity_only=parity_only, verbose=False)
    return operator, clover, kappa, kappa_t, u0


def _true_residual(output: Any, rhs: Any, gauge: Any, gauge_full: Any,
                   clover: Any, mass: float,
                   operator: Any | None = None) -> float:
    import torch
    from pyqcu import tools
    from pyqcu.torch import dslash

    del gauge
    kappa = 1.0 / (2.0 * mass + 8.0)
    solution_full = tools.poooxyzt2oooxyzt(output)
    rhs_full = tools.poooxyzt2oooxyzt(rhs)
    del operator
    u0 = torch.ones(1, dtype=torch.float32, device=solution_full.device)
    residual = dslash.give_wilson(
        solution_full, gauge_full, kappa, u0, with_I=True)
    residual = residual + dslash.give_clover(solution_full, clover) - rhs_full
    denominator = max(float(torch.linalg.vector_norm(rhs_full)), 1.0e-30)
    return float(torch.linalg.vector_norm(residual) / denominator)


def _memory_snapshot(device: Any) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _process_peak_rss_bytes() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _DeviceMemorySampler:
    """Sample device-wide usage so native QCU allocations are included."""

    def __init__(self, device: Any, interval_seconds: float = 0.05):
        self.device = device
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._baseline = self._used_bytes()
        self._peak = self._baseline
        self._samples = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="qcu-device-memory-sampler", daemon=True)

    def _used_bytes(self) -> int:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return int(total_bytes) - int(free_bytes)

    def _sample(self) -> None:
        used_bytes = self._used_bytes()
        with self._lock:
            self._baseline = min(self._baseline, used_bytes)
            self._peak = max(self._peak, used_bytes)
            self._samples += 1

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                self._sample()
        except BaseException as exc:
            self._error = exc

    def start(self) -> "_DeviceMemorySampler":
        self._sample()
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("device memory sampler failed") from self._error
        with self._lock:
            return {
                "scope": "device-wide cudaMemGetInfo; includes CUDA context and native QCU allocations",
                "sample_interval_seconds": self.interval_seconds,
                "sample_count": self._samples,
                "baseline_used_bytes": self._baseline,
                "peak_used_bytes": self._peak,
                "peak_delta_over_baseline_bytes": max(
                    0, self._peak - self._baseline),
            }


def _combine_device_peaks(
        peaks: Sequence[Mapping[str, Any] | None]) -> dict[str, Any] | None:
    """Combine disjoint phase samplers without losing either high-water mark."""
    present = [dict(peak) for peak in peaks if peak]
    if not present:
        return None
    baseline = min(int(peak["baseline_used_bytes"]) for peak in present)
    observed_peak = max(int(peak["peak_used_bytes"]) for peak in present)
    return {
        "scope": "combined disjoint phase samplers",
        "sample_interval_seconds": min(
            float(peak["sample_interval_seconds"]) for peak in present),
        "sample_count": sum(int(peak["sample_count"]) for peak in present),
        "baseline_used_bytes": baseline,
        "peak_used_bytes": observed_peak,
        "peak_delta_over_baseline_bytes": max(0, observed_peak - baseline),
    }


def _resource_payload(memory: Mapping[str, Any], *,
                      cache_bytes: int | None = None,
                      null_vector_bytes: int | None = None,
                      strict_runtime_memory: Mapping[str, Any] | None = None,
                      device_peak: Mapping[str, Any] | None = None,
                      setup_device_peak: Mapping[str, Any] | None = None,
                      solve_device_peak: Mapping[str, Any] | None = None,
                      ) -> dict[str, Any]:
    strict_memory = {} if strict_runtime_memory is None else {
        "asset_resident_bytes": int(
            strict_runtime_memory.get("asset_resident_bytes", 0)),
        "fused_workspace_resident_bytes": int(
            strict_runtime_memory.get(
                "fused_workspace_resident_bytes", 0)),
        "fused_workspace_planned_bytes": int(
            strict_runtime_memory.get(
                "fused_workspace_planned_bytes", 0)),
        "coarse_workspace_bytes": int(
            strict_runtime_memory.get("coarse_workspace_bytes", 0)),
        "fine_transfer_bytes": int(
            strict_runtime_memory.get("fine_transfer_bytes", 0)),
        "borrowed_gauge_clover_bytes": int(
            strict_runtime_memory.get(
                "borrowed_gauge_clover_bytes", 0)),
        "lattice_known_device_bytes": int(
            strict_runtime_memory.get("lattice_known_device_bytes", 0)),
        "runtime_owned_bytes": int(
            strict_runtime_memory.get("accounted_owned_bytes", 0)),
        "known_live_device_bytes": int(
            strict_runtime_memory.get("known_live_device_bytes", 0)),
        "setup_detached_storage_bytes": int(
            strict_runtime_memory.get(
                "setup_detached_storage_bytes", 0)),
        "setup_allocator_released_bytes": int(
            strict_runtime_memory.get(
                "setup_allocator_released_bytes", 0)),
    }
    return {
        "schema": "pyqcu.mg_resources.v1",
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "torch_peak_allocated_bytes": int(
            memory.get("peak_allocated_bytes", 0)),
        "torch_peak_reserved_bytes": int(
            memory.get("peak_reserved_bytes", 0)),
        "legacy_cache_bytes": cache_bytes,
        "strict_null_vector_bytes": null_vector_bytes,
        "strict_runtime": strict_memory,
        "device": {} if device_peak is None else dict(device_peak),
        "device_setup": (
            {} if setup_device_peak is None else dict(setup_device_peak)),
        "device_solve": (
            {} if solve_device_peak is None else dict(solve_device_peak)),
        "memory_detail": dict(memory),
    }


def _common_payload(args: argparse.Namespace) -> dict[str, Any]:
    from pyqcu.testing.simple._common import read_h5_metadata

    gauge = Path(args.gauge).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    return {
        "side": args.side,
        "lattice": list(args.lat),
        "block_size_full": list(args.block),
        "legacy_parity_block": list(args.legacy_parity_block),
        "mass": float(args.mass),
        "csw": float(args.csw),
        "precision": "c64",
        "gauge": {"path": str(gauge), "sha256": _sha256(gauge),
                  "metadata": read_h5_metadata(gauge)},
        "source": {"path": str(source), "sha256": _sha256(source),
                   "metadata": read_h5_metadata(source)},
    }


def _build_legacy(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import torch
    from pyqcu.cuda._multi_gpu import build_schur_levels

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    device_sampler = _DeviceMemorySampler(device).start()
    started = time.perf_counter()
    gauge_full, _, gauge, _ = _load_case(
        Path(args.gauge), Path(args.source), tuple(args.lat), device)
    case_load = time.perf_counter() - started
    started = time.perf_counter()
    operator = _fine_operator(
        gauge_full, tuple(args.lat), args.mass, args.csw, device,
        parity_only=True)[0]
    del gauge, _
    torch.cuda.empty_cache()
    fine_operator_prepare = time.perf_counter() - started
    fine_prepare = case_load + fine_operator_prepare
    started = time.perf_counter()
    assets = build_schur_levels(
        operator, operator.matvec_parity, 2, [12, args.coarse_dof],
        list(args.legacy_parity_block), list(args.lat), args.coarse_dof,
        torch.complex64,
        device, nv_iters=args.nv_iters, use_cache=False, verbose=args.verbose,
        mass=args.mass, csw=args.csw, local_stencil=True,
        local_stencil_batch_size=args.local_stencil_batch,
        null_batch_chunk=args.null_batch_chunk)
    torch.cuda.synchronize(device)
    asset_build = time.perf_counter() - started
    lonv, hnn, hdg, sit = (item[0] for item in assets)
    actual_cache_block = _validate_legacy_cache_block(
        lonv.shape, args.legacy_parity_block, 1)
    actual_physical_shape = _legacy_cache_physical_shape(
        lonv.shape, level=1)
    physical_coarse_shapes = _physical_coarse_shapes(
        args.lat, args.block, 2)
    expected_physical_shape = physical_coarse_shapes[0]
    if actual_physical_shape != expected_physical_shape:
        raise ValueError(
            "Legacy coarse cache level 1 physical shape mismatch: "
            f"actual={actual_physical_shape} expected={expected_physical_shape}")
    gauge_sha256 = _sha256(Path(args.gauge).expanduser().resolve())
    manifest = _legacy_cache_manifest(
        level=1, lattice=args.lat, full_block=args.block,
        legacy_block=actual_cache_block,
        physical_shape=actual_physical_shape, gauge_sha256=gauge_sha256,
        mass=args.mass, csw=args.csw, coarse_dof=args.coarse_dof,
        nv_iters=args.nv_iters, local_stencil=True,
        local_stencil_batch_size=args.local_stencil_batch,
        null_batch_chunk=args.null_batch_chunk)
    output = Path(args.cache_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_write_started = time.perf_counter()
    with h5py.File(output, "w") as handle:
        for name, tensor in (("lonv", lonv), ("hnn", hnn),
                             ("hdg", hdg), ("sit", sit)):
            handle.create_dataset(name, data=tensor.detach().cpu().numpy())
        handle.attrs[_LEGACY_CACHE_MANIFEST_ATTR] = (
            _legacy_cache_manifest_json(manifest))
    cache_write = time.perf_counter() - cache_write_started
    payload = _common_payload(args)
    memory = _memory_snapshot(device)
    device_peak = device_sampler.stop()
    payload.update({
        "operation": "legacy_asset_build",
        "status": "PASS",
        "case_load_seconds": case_load,
        "fine_operator_prepare_seconds": fine_operator_prepare,
        "fine_prepare_seconds": fine_prepare,
        "mg_asset_build_seconds": asset_build,
        "cache_write_seconds": cache_write,
        "mg_init_seconds": asset_build,
        "mg_init_scope": (
            "cold coarse asset generation only; excludes case load, fine "
            "operator preparation, HDF5 cache write, and solve runtime init"),
        "coarse_dof": args.coarse_dof,
        "mg_grid": list(args.legacy_parity_block),
        "actual_cache_block": list(actual_cache_block),
        "actual_cache_blocks": [
            {"level": 1, "block": list(actual_cache_block),
             "physical_coarse_shape": list(actual_physical_shape)}],
        "physical_coarse_shapes": [
            list(shape) for shape in physical_coarse_shapes],
        "physical_coarse_shape": list(physical_coarse_shapes[-1]),
        "nv_iters": args.nv_iters,
        "local_stencil": True,
        "local_stencil_batch_size": args.local_stencil_batch,
        "null_batch_chunk": args.null_batch_chunk,
        "cache": {"path": str(output), "sha256": _sha256(output),
                  "size_bytes": output.stat().st_size,
                  "manifest_sha256": _sha256_text(
                      _legacy_cache_manifest_json(manifest))},
        "asset_shapes": {
            "lonv": list(lonv.shape), "hnn": list(hnn.shape),
            "hdg": list(hdg.shape), "sit": list(sit.shape)},
        "memory": memory,
        "resources": _resource_payload(
            memory, cache_bytes=output.stat().st_size,
            device_peak=device_peak),
    })
    return payload


def _legacy_run_once(qcu: Any, define: Any, set_ptrs: Any, params: Any,
                     argv: Any, output: Any, fields: tuple[Any, ...],
                     device: Any) -> tuple[float, dict[str, Any]]:
    import torch

    output.zero_()
    params[define._MG_USE_INIT_GUESS_] = 0
    source = fields[1].clone()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    qcu.applyCloverMultigridQcu(
        output, source, fields[0], fields[3], fields[4], fields[5],
        fields[6], set_ptrs, params)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, _memory_snapshot(device)


def _parse_stop_report(text: str) -> dict[str, str | int | bool | float]:
    """Parse the final machine-readable Legacy stop report in ``text``."""
    matches = re.findall(r"FINAL STOP:\s*([^\n]+)", text)
    if not matches:
        return {}
    values: dict[str, str] = dict(
        item.split("=", 1) for item in matches[-1].split() if "=" in item)
    result: dict[str, str | int | bool | float] = {
        "reason": values.get("reason", "unknown"),
        "converged": values.get("converged") == "true",
    }
    for key in (
            "full_gate_checks", "correction_attempts", "corrections_accepted",
            "corrections_rejected"):
        if key in values:
            result[key] = int(values[key])
    for key in ("full_relative", "tol"):
        if key in values:
            result[key] = float(values[key])
    return result


def _parse_prof_sections(text: str) -> dict[str, float | int]:
    """Parse the final ``PROF_SECTIONS`` counters in ``text``."""
    matches = re.findall(r"PROF_SECTIONS:\s*([^\n]+)", text)
    if not matches:
        return {}
    values: dict[str, float | int] = {}
    for token in matches[-1].split():
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        if raw.endswith("ms"):
            raw = raw[:-2]
        if key.startswith("n_") or key.endswith("_iters") or key.endswith("_checks"):
            try:
                values[key] = int(raw)
            except ValueError:
                pass
        else:
            try:
                values[key] = float(raw)
            except ValueError:
                pass
    return values


def _legacy_coarse_execution(
        prof: Mapping[str, float | int],
        stop_report: Mapping[str, str | int | bool | float],
        ) -> dict[str, Any]:
    """Report observed Legacy correction work, never configured inference."""
    attempted_present = "n_vcycles" in prof
    reported_present = "correction_attempts" in stop_report
    accepted_present = "corrections_accepted" in stop_report
    rejected_present = "corrections_rejected" in stop_report
    attempted = int(prof["n_vcycles"]) if attempted_present else None
    reported_attempts = (
        int(stop_report["correction_attempts"])
        if reported_present else None)
    accepted = (
        int(stop_report["corrections_accepted"])
        if accepted_present else None)
    rejected = (
        int(stop_report["corrections_rejected"])
        if rejected_present else None)
    partition_consistent = (
        None if attempted is None or accepted is None or rejected is None else
        attempted == accepted + rejected)
    counters_consistent = (
        attempted is not None and reported_attempts is not None and
        attempted == reported_attempts and partition_consistent is True)
    if counters_consistent:
        effective_mode = (
            "fine_only" if attempted == 0 else "coarse_correction")
    else:
        effective_mode = "unverified"
    return {
        "attempted_vcycles": attempted,
        "reported_correction_attempts": reported_attempts,
        "accepted_vcycles": accepted,
        "rejected_vcycles": rejected,
        "coarse_iterations": (
            int(prof["coarse_iters"]) if "coarse_iters" in prof else None),
        "vcycle_seconds": (
            float(prof["vcycle"]) / 1000.0 if "vcycle" in prof else None),
        "counters_consistent": counters_consistent,
        "correction_partition_consistent": partition_consistent,
        "executed": None if not counters_consistent else attempted > 0,
        "effective_mode": effective_mode,
    }


def _strict_coarse_execution(outer_iterations: Sequence[int]) -> dict[str, Any]:
    """Report Strict backend-returned outer work, not hierarchy configuration."""
    values = [int(value) for value in outer_iterations]
    return {
        "kind": "strict_recursive_vcycle",
        "outer_iterations": values,
        "top_level_vcycle_calls": values,
        "source": "C++ applyMultigridStrictFgmresQcu returned iteration counts",
        "evidence": (
            "the backend calls precondition_fine once per returned nonzero "
            "outer FGMRES iteration"),
        "trace_verified": False,
        "effective_mode": "recursive_vcycle" if any(values) else "zero_work",
    }


def _parse_legacy_run_text(text: str) -> dict[str, Any]:
    """Parse exactly one solve fragment captured from the Legacy log."""
    iterations = re.findall(r"Total iterations:\s*(\d+)", text)
    residuals = re.findall(
        r"FINAL TRUE residual \(full-op\).*?relative\s*=\s*([0-9.eE+-]+)",
        text, re.S)
    stop_report = _parse_stop_report(text)
    prof = _parse_prof_sections(text)
    return {
        "iterations": int(iterations[-1]) if iterations else None,
        "native_true_residual": (
            float(residuals[-1]) if residuals else None),
        "stop_report": stop_report,
        "prof": prof,
        "coarse_execution": _legacy_coarse_execution(prof, stop_report),
        "complete": bool(iterations and residuals and stop_report and prof),
    }


def _read_log_growth(
        log_path: Path, offset: int) -> tuple[str, int]:
    """Read only bytes appended by one solve, preventing stale-log evidence."""
    if not log_path.is_file():
        return "", int(offset)
    size = log_path.stat().st_size
    if size < offset:
        raise RuntimeError(
            f"Legacy MG log shrank during benchmark: {log_path} "
            f"offset={offset} size={size}")
    with log_path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read()
    return payload.decode("utf-8", errors="replace"), int(offset + len(payload))


def _run_legacy(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import numpy as np
    import torch
    from pyqcu import tools
    from pyqcu.cuda import define, qcu
    from pyqcu.testing.qcu.strict.quda_comparison.common import make_clover_tensors

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    setup_device_sampler = _DeviceMemorySampler(device).start()
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QCU_LOG_DIR"] = str(log_dir)
    started = time.perf_counter()
    gauge_full, _, gauge, rhs = _load_case(
        Path(args.gauge), Path(args.source), tuple(args.lat), device)
    case_load = time.perf_counter() - started
    levels = int(args.levels)
    multigrid_warnings = _depth_warnings(
        levels, "legacy", tuple(args.lat), args.legacy_parity_block)
    physical_coarse_shapes = _physical_coarse_shapes(
        args.lat, args.block, levels)
    cache_paths = [
        Path(value).expanduser().resolve()
        for value in (args.cache, *args.cache_level)
    ]
    if len(cache_paths) < levels - 1:
        raise ValueError(
            f"levels={levels} requires {levels - 1} coarse caches, "
            f"got {len(cache_paths)}")
    cache_paths = cache_paths[:levels - 1]
    gauge_sha256 = _sha256(Path(args.gauge).expanduser().resolve())
    started = time.perf_counter()
    assets_by_level = []
    actual_cache_blocks = []
    cache_records = []
    for level_index, cache_path in enumerate(cache_paths, start=1):
        expected_physical_shape = physical_coarse_shapes[level_index - 1]
        expected_manifest = _legacy_cache_manifest(
            level=level_index, lattice=args.lat, full_block=args.block,
            legacy_block=args.legacy_parity_block,
            physical_shape=expected_physical_shape,
            gauge_sha256=gauge_sha256, mass=args.mass, csw=args.csw,
            coarse_dof=args.coarse_dof, nv_iters=args.nv_iters,
            local_stencil=True,
            local_stencil_batch_size=args.local_stencil_batch,
            null_batch_chunk=args.null_batch_chunk)
        with h5py.File(cache_path, "r") as handle:
            actual_cache_block = _validate_legacy_cache_block(
                handle["lonv"].shape, args.legacy_parity_block, level_index)
            actual_physical_shape = _legacy_cache_physical_shape(
                handle["lonv"].shape, level=level_index)
            if actual_physical_shape != expected_physical_shape:
                raise ValueError(
                    f"Legacy coarse cache level {level_index} physical shape "
                    f"mismatch: actual={actual_physical_shape} "
                    f"expected={expected_physical_shape}")
            manifest = _decode_legacy_cache_manifest(
                handle.attrs.get(_LEGACY_CACHE_MANIFEST_ATTR),
                path=cache_path, level=level_index)
            manifest = _validate_legacy_cache_manifest(
                manifest, expected_manifest, path=cache_path,
                level=level_index)
            for name in ("lonv", "hnn", "hdg", "sit"):
                dataset = handle[name]
                if dataset.dtype != np.dtype(np.complex64):
                    raise ValueError(
                        f"Legacy coarse cache level {level_index} dataset "
                        f"{name!r} dtype={dataset.dtype}, expected complex64: "
                        f"{cache_path}")
            actual_cache_blocks.append({
                "level": level_index,
                "path": str(cache_path),
                "block": list(actual_cache_block),
                "physical_coarse_shape": list(actual_physical_shape),
            })
            cache_records.append({
                "level": level_index,
                "path": str(cache_path),
                "sha256": _sha256(cache_path),
                "size_bytes": cache_path.stat().st_size,
                "manifest": manifest,
                "manifest_sha256": _sha256_text(
                    _legacy_cache_manifest_json(manifest)),
            })
            assets_by_level.append([
                torch.from_numpy(handle[name][...]).to(
                    device=device, dtype=torch.complex64).contiguous()
                for name in ("lonv", "hnn", "hdg", "sit")
            ])
    torch.cuda.synchronize(device)
    cache_load = time.perf_counter() - started
    started = time.perf_counter()
    define.params[define._VERBOSE_] = 0
    ce, cei, coo, coi, set_ptrs, params, argv = make_clover_tensors(
        gauge, tuple(args.lat), args.mass, dtype=torch.complex64,
        data_type=define._LAT_C64_, csw=args.csw)
    torch.cuda.synchronize(device)
    clover_tensor_prepare = time.perf_counter() - started
    argv[define._ATOL_] = float(args.tol)
    params[define._SET_PLAN_] = define._SET_PLAN1_
    params[define._PARITY_] = 0
    params[define._MAX_ITER_] = int(args.max_iter)
    params[define._VERBOSE_] = 1
    params[define._MG_USE_DEFLATE_] = int(args.deflate)
    params[define._MG_NUM_LEVEL_] = levels
    set_ptrs[define._SET_PTRS_COARSE_BASE_:] = 0
    for level_index, assets in enumerate(assets_by_level, start=1):
        lonv = assets[0]
        if lonv.ndim != 10:
            raise ValueError(
                f"level {level_index} lonv shape is {tuple(lonv.shape)}, "
                "expected the 10-D blocked Legacy layout")
        base = define._MG_LEVEL1_E_ + 8 * (level_index - 1)
        params[base + 0] = int(lonv.shape[0])
        params[base + 1] = int(lonv.shape[2])
        params[base + 2] = int(lonv.shape[4])
        params[base + 3] = int(lonv.shape[6])
        params[base + 4] = int(lonv.shape[8])
        params[base + 5] = int(args.coarse_max_iter)
        params[base + 6] = define._LAT_C64_
        params[base + 7] = int(args.coarse_restart)
        argv[define._MG_LEVEL1_ATOL_ + level_index - 1] = float(
            args.tol * args.coarse_tol_factor * (3 ** (level_index - 1)))
        pointer_base = define._SET_PTRS_COARSE_BASE_ + 4 * (level_index - 1)
        for offset, tensor in enumerate(assets):
            set_ptrs[pointer_base + offset] = tensor.data_ptr()
            tensor.record_stream(torch.cuda.current_stream(device))
    # make_clover_tensors leaves SET_INDEX at the next reusable slot.
    torch.cuda.synchronize(device)
    init_started = time.perf_counter()
    qcu.applyInitQcu(set_ptrs, params, argv)
    torch.cuda.synchronize(device)
    runtime_init = time.perf_counter() - init_started
    setup_device_peak = setup_device_sampler.stop()
    output = torch.empty_like(rhs)
    warmups = []
    steady = []
    iterations = []
    runtime_reinit_seconds = []
    run_diagnostics = []
    log_path = log_dir / "clover_multigrid.log"
    log_offset = log_path.stat().st_size if log_path.is_file() else 0
    solve_device_sampler = _DeviceMemorySampler(device).start()
    final_result = None
    native_residual = None
    python_residual = None
    try:
        for index in range(args.warmups + args.repeats):
            if index > 0:
                reinit_started = time.perf_counter()
                qcu.applyEndQcu(set_ptrs, params)
                qcu.applyInitQcu(set_ptrs, params, argv)
                torch.cuda.synchronize(device)
                runtime_reinit_seconds.append(
                    time.perf_counter() - reinit_started)
            run_log_offset = log_offset
            elapsed, _mem = _legacy_run_once(
                qcu, define, set_ptrs, params, argv, output,
                (gauge, rhs, output, ce, coo, cei, coi), device)
            log_fragment, log_offset = _read_log_growth(
                log_path, run_log_offset)
            diagnostics = _parse_legacy_run_text(log_fragment)
            diagnostics["coarse_execution"]["source"] = (
                f"{log_path}:{run_log_offset}-{log_offset}")
            run_diagnostics.append(diagnostics)
            (warmups if index < args.warmups else steady).append(elapsed)
            iterations.append(diagnostics["iterations"])
    finally:
        solve_device_peak = solve_device_sampler.stop()
        qcu.applyEndQcu(set_ptrs, params)
        # Native cache tensors and runtime scratch must leave the device before
        # preparing the independent Torch residual on capacity-bound cases.
        del assets, lonv, tensor
        del assets_by_level
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    residual_prepare_started = time.perf_counter()
    clover, _kappa, _kappa_t, _u0 = _fine_clover(
        gauge_full, args.mass, args.csw, device)
    torch.cuda.synchronize(device)
    residual_prepare = time.perf_counter() - residual_prepare_started
    python_residual = float(_true_residual(
        output, rhs, gauge, gauge_full, clover, args.mass))
    final_diagnostics = run_diagnostics[-1] if run_diagnostics else {}
    native_residual = final_diagnostics.get("native_true_residual")
    final_coarse_execution = dict(
        final_diagnostics.get("coarse_execution", {}))
    final_result = (native_residual if native_residual is not None
                    else python_residual)
    payload = _common_payload(args)
    memory = _memory_snapshot(device)
    device_peak = _combine_device_peaks(
        (setup_device_peak, solve_device_peak))
    stop_report = dict(final_diagnostics.get("stop_report", {}))
    native_pass = (
        native_residual is not None and native_residual <= args.tol and
        stop_report.get("converged") is True and
        stop_report.get("tol") == float(args.tol))
    independent_pass = python_residual is not None and python_residual <= args.tol
    all_runs_complete = (
        len(run_diagnostics) == args.warmups + args.repeats and
        all(report.get("complete") is True for report in run_diagnostics))
    coarse_execution_pass = (
        not args.require_coarse_execution or
        (final_coarse_execution.get("executed") is True))
    status_pass = (
        native_pass and independent_pass and all_runs_complete and
        coarse_execution_pass)
    if args.solution_output:
        solution_path = Path(args.solution_output).expanduser().resolve()
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(solution_path, "w") as handle:
            handle.create_dataset(
                "data",
                data=tools.poooxyzt2oooxyzt(output).detach().cpu().numpy())
    payload.update({
        "operation": "legacy_solve",
        "status": "PASS" if status_pass else "FAIL",
        "case_load_seconds": case_load,
        "fine_prepare_seconds": case_load,
        "fine_prepare_scope": (
            "gauge/source HDF5 load and layout conversion; solve-side Legacy "
            "does not rebuild the fine coarse-basis operator"),
        "residual_operator_prepare_seconds": residual_prepare,
        "cache_load_seconds": cache_load,
        "clover_tensor_prepare_seconds": clover_tensor_prepare,
        "setup_seconds_prebuilt_assets": (
            case_load + cache_load + clover_tensor_prepare + runtime_init),
        "setup_scope_prebuilt_assets": (
            "case load + validated cache load + native Clover tensors + "
            "applyInitQcu; excludes cold coarse-cache generation"),
        "mg_init_seconds": runtime_init,
        "mg_init_scope": "applyInitQcu LatticeSet/MG runtime allocation; excludes asset build",
        "runtime_reinit_seconds": runtime_reinit_seconds,
        "steady_scope": "solve only; LatticeSet runtime is rebuilt between samples outside the timer to prevent cross-solve state reuse",
        "warmup_seconds": warmups,
        "steady_seconds": steady,
        "steady_median_seconds": _json_median(steady),
        "steady_mad_seconds": _json_mad(steady),
        "iterations": iterations,
        "outer_iterations": iterations,
        "outer_iteration_semantics": (
            "Legacy fine BiCGStab iterations parsed from each solve's exact "
            "log fragment; not coarse V-cycle count"),
        "final_true_residual": final_result,
        "native_true_residual": native_residual,
        "independent_true_residual": python_residual,
        "native_residual_pass": native_pass,
        "independent_residual_pass": independent_pass,
        "residual_source": (
            "C++ full-operator true residual from Legacy MG log"
            if native_residual is not None else
            "Python full-operator residual fallback"),
        "coarse_dof": args.coarse_dof,
        "levels": levels,
        "multigrid_warnings": multigrid_warnings,
        "coarse_caches": [str(path) for path in cache_paths],
        "cache_records": cache_records,
        "actual_cache_block": actual_cache_blocks[0]["block"],
        "actual_cache_blocks": actual_cache_blocks,
        "physical_coarse_shapes": [
            list(shape) for shape in physical_coarse_shapes],
        "physical_coarse_shape": list(physical_coarse_shapes[-1]),
        "coarse_max_iter": args.coarse_max_iter,
        "coarse_restart": args.coarse_restart,
        "deflate": bool(args.deflate),
        "coarse_execution": final_coarse_execution,
        "coarse_execution_required": bool(args.require_coarse_execution),
        "coarse_execution_pass": coarse_execution_pass,
        "solve_reports": [
            {
                "iterations": report.get("iterations"),
                "native_true_residual": report.get("native_true_residual"),
                "stop_report": report.get("stop_report"),
                "coarse_execution": report.get("coarse_execution"),
                "complete": report.get("complete"),
            }
            for report in run_diagnostics
        ],
        "stop_gate": {
            "native_scope": "full Wilson/Clover relative residual",
            "native_converged": stop_report.get("converged"),
            "native_tolerance": stop_report.get("tol"),
            "independent_scope": "full Wilson/Clover relative residual",
            "independent_tolerance": args.tol,
            "acceptance": "native converged at tol and independent residual <= tol",
        },
        "all_runs_complete": all_runs_complete,
        "stop_report": stop_report,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "memory": memory,
        "solution_output": (
            str(Path(args.solution_output).expanduser().resolve())
            if args.solution_output else None),
        "resources": _resource_payload(
            memory, cache_bytes=sum(
                int(record["size_bytes"]) for record in cache_records),
            device_peak=device_peak,
            setup_device_peak=setup_device_peak,
            solve_device_peak=solve_device_peak),
    })
    return payload


def _run_strict(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import numpy as np
    import torch
    from pyqcu import tools
    from pyqcu.torch import dslash
    from pyqcu.cuda import define
    from pyqcu.cuda import StrictMultigrid as CudaStrictMultigrid
    from pyqcu.torch.solver import StrictMultigrid

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    setup_device_sampler = _DeviceMemorySampler(device).start()
    started = time.perf_counter()
    gauge_full, _, gauge, rhs = _load_case(
        Path(args.gauge), Path(args.source), tuple(args.lat), device)
    case_load = time.perf_counter() - started
    started = time.perf_counter()
    _operator, clover, kappa, kappa_t, u0 = _fine_operator(
        gauge_full, tuple(args.lat), args.mass, args.csw, device)
    fine_operator_prepare = time.perf_counter() - started
    fine_prepare = case_load + fine_operator_prepare
    null_path = Path(args.nullvec).expanduser().resolve()
    null_load_started = time.perf_counter()
    with h5py.File(null_path, "r") as handle:
        null_dataset = handle[args.null_dataset]
        if null_dataset.dtype != np.dtype(np.complex64):
            raise ValueError(
                f"Strict null-vector dataset {args.null_dataset!r} has "
                f"dtype={null_dataset.dtype}, expected complex64: {null_path}")
        null = torch.from_numpy(null_dataset[...]).to(
            device=device, dtype=torch.complex64).contiguous()
    torch.cuda.synchronize(device)
    null_vector_load_seconds = time.perf_counter() - null_load_started
    available_nvec = int(null.shape[0])
    nvec = available_nvec if args.nvec is None else int(args.nvec)
    if nvec <= 0 or nvec > available_nvec:
        raise ValueError(
            f"selected nvec={nvec} is outside available range 1..{available_nvec}")
    null = null[:nvec].contiguous()
    levels = int(args.levels)
    multigrid_warnings = _depth_warnings(
        levels, "strict", tuple(args.lat), args.block)
    physical_coarse_shapes = _physical_coarse_shapes(
        args.lat, args.block, levels)
    hierarchy = StrictMultigrid(
        U=gauge_full, clover_term=clover, kappa=kappa_t, u_0=u0,
        lat_size=tuple(args.lat), null_vectors=[null], nvec_list=[nvec],
        dof_list=[12] + [2 * nvec] * (levels - 1),
        block_size=[tuple(args.block)] * (levels - 1), max_level=levels,
        propagate_null_vectors=levels > 2,
        materialize_coarse=True, use_parity=True, target_parity=1,
        nu_pre=1, nu_post=1, coarse_max_iter=args.coarse_max_iter,
        coarse_tol=args.tol * args.coarse_tol_factor,
        restart=args.restart, max_iter=args.max_iter, tol=args.tol,
        setup_method="random", setup_iters=0, strict_galerkin_mode="colored",
        verbose=False)
    hierarchy_started = time.perf_counter()
    hierarchy.setup()
    torch.cuda.synchronize(device)
    hierarchy_setup = time.perf_counter() - hierarchy_started
    hierarchy_transitions = _strict_hierarchy_transitions(
        hierarchy.transfers, args.lat, args.block, levels)
    from pyqcu.testing.qcu.strict.quda_comparison.common import make_clover_tensors
    clover_started = time.perf_counter()
    define.params[define._VERBOSE_] = 0
    ce, cei, coo, coi, set_ptrs, params, argv = make_clover_tensors(
        gauge, tuple(args.lat), args.mass, dtype=torch.complex64,
        data_type=define._LAT_C64_, csw=args.csw)
    torch.cuda.synchronize(device)
    clover_tensor_prepare = time.perf_counter() - clover_started
    argv[define._ATOL_] = float(args.tol)
    params[define._MAX_ITER_] = int(args.max_iter)
    solver_started = time.perf_counter()
    solver = CudaStrictMultigrid(
        hierarchy, argv, gauge, ce, coo, cei, coi, params,
        restart=args.restart, max_krylov_bytes=args.max_krylov_bytes,
        release_setup_assets=True, verbose=False)
    torch.cuda.synchronize(device)
    runtime_init = time.perf_counter() - solver_started
    setup_device_peak = setup_device_sampler.stop()
    warmups = []
    steady = []
    iterations = []
    residuals = []
    converged_samples = []
    final_result = None
    output = torch.empty_like(rhs)
    solve_device_sampler = _DeviceMemorySampler(device).start()
    try:
        for index in range(args.warmups + args.repeats):
            rhs_work = rhs.clone()
            started = time.perf_counter()
            solver.solve(rhs_work, out=output)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            (warmups if index < args.warmups else steady).append(elapsed)
            iterations.append(int(solver.iterations))
            residuals.append(float(solver.final_residual))
            converged_samples.append(bool(solver.converged))
        final_result = float(_true_residual(
            output, rhs, gauge, gauge_full, clover, args.mass, _operator))
        runtime_memory = solver.memory_report()
        # Capture allocator high-water marks while the benchmark-owned objects
        # are still alive and before close() releases the Strict runtime.
        allocator_memory = _memory_snapshot(device)
    finally:
        solve_device_peak = solve_device_sampler.stop()
        solver.close()
    payload = _common_payload(args)
    device_peak = _combine_device_peaks(
        (setup_device_peak, solve_device_peak))
    memory = {**runtime_memory, "torch_allocator": allocator_memory}
    if args.solution_output:
        solution_path = Path(args.solution_output).expanduser().resolve()
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(solution_path, "w") as handle:
            handle.create_dataset(
                "data",
                data=tools.poooxyzt2oooxyzt(output).detach().cpu().numpy())
    all_runs_complete = (
        len(iterations) == args.warmups + args.repeats and
        len(converged_samples) == len(iterations) and
        all(converged_samples))
    status_pass = (
        final_result is not None and final_result <= args.tol and
        all_runs_complete)
    payload.update({
        "operation": "strict_solve",
        "status": "PASS" if status_pass else "FAIL",
        "case_load_seconds": case_load,
        "fine_operator_prepare_seconds": fine_operator_prepare,
        "fine_prepare_seconds": fine_prepare,
        "null_vector_load_seconds": null_vector_load_seconds,
        "clover_tensor_prepare_seconds": clover_tensor_prepare,
        "hierarchy_setup_seconds": hierarchy_setup,
        "runtime_init_seconds": runtime_init,
        "setup_seconds_prebuilt_assets": (
            case_load + fine_operator_prepare + null_vector_load_seconds +
            clover_tensor_prepare + hierarchy_setup + runtime_init),
        "setup_scope_prebuilt_assets": (
            "case load + canonical null-basis load + fine operator + native "
            "Clover tensors + Strict hierarchy setup + runtime binding; "
            "excludes cold null-vector generation"),
        "mg_init_seconds": hierarchy_setup + runtime_init,
        "mg_init_scope": "strict Galerkin hierarchy/asset setup plus runtime binding; excludes null-vector file creation",
        "null_vector_path": str(null_path),
        "null_vector_sha256": _sha256(null_path),
        "null_vector_size_bytes": null_path.stat().st_size,
        "nvec": nvec,
        "null_vector_source_nvec": available_nvec,
        "null_vector_indices": list(range(nvec)),
        "coarse_dof": 2 * nvec,
        "levels": levels,
        "multigrid_warnings": multigrid_warnings,
        "actual_block_size": hierarchy_transitions[0]["block"],
        "actual_hierarchy_transitions": hierarchy_transitions,
        "actual_cache_blocks": [],
        "physical_coarse_shapes": [
            list(shape) for shape in physical_coarse_shapes],
        "physical_coarse_shape": list(physical_coarse_shapes[-1]),
        "warmup_seconds": warmups,
        "steady_seconds": steady,
        "steady_median_seconds": _json_median(steady),
        "steady_mad_seconds": _json_mad(steady),
        "iterations": iterations,
        "outer_iterations": iterations,
        "outer_iteration_semantics": (
            "Strict right-preconditioned FGMRES outer iterations returned by "
            "the CUDA backend; not total smoother or coarse Krylov iterations"),
        "backend_final_residuals": residuals,
        "backend_final_residual_scope": "absolute MATPC residual norm",
        "backend_converged": converged_samples,
        "final_true_residual": final_result,
        "residual_source": "Python full Wilson/Clover reconstruction",
        "coarse_execution": _strict_coarse_execution(iterations),
        "solve_reports": [
            {
                "outer_iterations": int(iteration),
                "backend_final_residual": float(residual),
                "backend_converged": bool(converged),
            }
            for iteration, residual, converged in zip(
                iterations, residuals, converged_samples)
        ],
        "stop_gate": {
            "native_scope": "odd-parity Clover MATPC relative residual in C++",
            "native_residual_reported": "absolute MATPC residual norm",
            "native_converged": converged_samples,
            "independent_scope": "full Wilson/Clover relative residual",
            "independent_tolerance": args.tol,
            "acceptance": "native converged each solve and independent residual <= tol",
        },
        "all_runs_complete": all_runs_complete,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "restart": args.restart,
        "coarse_max_iter": args.coarse_max_iter,
        "memory": memory,
        "solution_output": (
            str(Path(args.solution_output).expanduser().resolve())
            if args.solution_output else None),
        "resources": _resource_payload(
            allocator_memory, null_vector_bytes=null_path.stat().st_size,
            strict_runtime_memory=runtime_memory, device_peak=device_peak,
            setup_device_peak=setup_device_peak,
            solve_device_peak=solve_device_peak),
    })
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gauge", required=True)
    common.add_argument("--source", required=True)
    common.add_argument("--lat", nargs=4, type=int, required=True)
    common.add_argument(
        "--block", nargs=4, type=int, default=(2, 2, 2, 2),
        metavar=("BX", "BY", "BZ", "BT"),
        help="full-lattice coarsening block; BT must be positive and even")
    common.add_argument("--mass", type=float, required=True)
    common.add_argument("--csw", type=float, required=True)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--tol", type=float, default=1e-6)
    common.add_argument("--max-iter", type=int, default=200)
    common.add_argument("--warmups", type=int, default=2)
    common.add_argument("--repeats", type=int, default=5)
    common.add_argument("--output", required=True)
    common.add_argument("--solution-output")

    build = sub.add_parser("legacy-build", parents=[common])
    build.add_argument("--cache-output", required=True)
    build.add_argument("--coarse-dof", type=int, default=12)
    build.add_argument("--nv-iters", type=int, default=1)
    build.add_argument("--local-stencil-batch", type=int, default=4)
    build.add_argument("--null-batch-chunk", type=int, default=None)
    build.add_argument("--verbose", action="store_true")

    legacy = sub.add_parser("legacy", parents=[common])
    legacy.add_argument("--cache", required=True)
    legacy.add_argument(
        "--cache-level", action="append", default=[],
        help="coarse cache for level 2, 3, ...; repeat in ascending order")
    legacy.add_argument("--levels", type=int, default=2)
    legacy.add_argument("--log-dir", required=True)
    legacy.add_argument("--coarse-dof", type=int, default=12)
    legacy.add_argument("--nv-iters", type=int, default=1)
    legacy.add_argument("--local-stencil-batch", type=int, default=4)
    legacy.add_argument("--null-batch-chunk", type=int, default=None)
    legacy.add_argument("--coarse-max-iter", type=int, default=200)
    legacy.add_argument("--coarse-restart", type=int, default=5)
    legacy.add_argument("--coarse-tol-factor", type=float, default=1e4)
    legacy.add_argument("--deflate", action="store_true",
                        help="attempt the initial Legacy coarse correction")
    legacy.add_argument(
        "--require-coarse-execution", action="store_true",
        help="fail unless the exact solve-log fragment proves a V-cycle ran")

    strict = sub.add_parser("strict", parents=[common])
    strict.add_argument("--nullvec", required=True)
    strict.add_argument("--null-dataset", default="null")
    strict.add_argument("--levels", type=int, default=2)
    strict.add_argument("--nvec", type=int, default=None,
                        help="use the first N vectors from the canonical input")
    strict.add_argument("--restart", type=int, default=20)
    strict.add_argument("--coarse-max-iter", type=int, default=200)
    strict.add_argument("--coarse-tol-factor", type=float, default=1e2)
    strict.add_argument("--max-krylov-bytes", type=int, default=512 << 20)

    args = parser.parse_args(argv)
    try:
        args.block = _validate_full_block(args.block)
        args.legacy_parity_block = _legacy_parity_block(args.block)
    except ValueError as exc:
        parser.error(str(exc))
    args.side = "legacy" if args.command.startswith("legacy") else "strict"
    if hasattr(args, "levels") and not 2 <= int(args.levels) <= 4:
        parser.error("--levels must be in [2,4] for the MG benchmark")
    if not 0.0 < float(args.tol) < 1.0:
        parser.error("--tol must be in (0,1)")
    if int(args.max_iter) <= 0:
        parser.error("--max-iter must be positive")
    if args.command != "legacy-build":
        if int(args.warmups) < 0:
            parser.error("--warmups must be nonnegative")
        if int(args.repeats) <= 0:
            parser.error("--repeats must be positive")
    if args.command in ("legacy-build", "legacy"):
        if int(args.nv_iters) < 0:
            parser.error("--nv-iters must be nonnegative")
        if int(args.local_stencil_batch) <= 0:
            parser.error("--local-stencil-batch must be positive")
        if int(args.coarse_dof) <= 0:
            parser.error("--coarse-dof must be positive")
    if args.command == "legacy-build":
        payload = _build_legacy(args)
    elif args.command == "legacy":
        payload = _run_legacy(args)
    else:
        payload = _run_strict(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
