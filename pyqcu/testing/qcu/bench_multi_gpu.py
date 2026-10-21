import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from pyqcu.cuda._multi_gpu import MultiGpuMultigrid


def _round_optional(value, digits=4):
    return None if value is None else round(value, digits)


def bench(lat, mass=0.05, atol=1e-6, num_levels=2, nthreads=3,
          device_ids=None, num_restart=5, coarse_max_iter=15,
          coarse_tol_factor=1e5, nv_iters=2, verbose=False,
          use_cache=True, cache_dir=None, csw=1.0, seed=42,
          warmups=1, repeats=3, run_reference=True,
          gauge=None, source=None, input_tag=None, verify_residual=False,
          dof_list=None, mg_grid=None, use_deflate=False):
    mg = MultiGpuMultigrid(lat_size=lat, mass=mass, atol=atol,
                           num_levels=num_levels, nthreads=nthreads,
                           dof_list=dof_list, mg_grid=mg_grid,
                           device_ids=device_ids, num_restart=num_restart,
                           coarse_max_iter=coarse_max_iter,
                           coarse_tol_factor=coarse_tol_factor,
                           nv_iters=nv_iters, verbose=verbose,
                           use_cache=use_cache, cache_dir=cache_dir,
                           csw=csw, seed=seed,
                           run_reference=run_reference, gauge=gauge,
                           source=source, input_tag=input_tag,
                           verify_residual=verify_residual,
                           use_deflate=use_deflate)
    for _ in range(warmups):
        mg.solve()
    calls = []
    r = None
    for repeat in range(repeats):
        t0 = time.perf_counter()
        r = mg.solve()
        calls.append({
            'repeat': repeat + 1,
            'total_solve': round(time.perf_counter() - t0, 6),
            'setup_seconds': _round_optional(
                (mg.last_timings or {}).get('setup_seconds')),
            'setup_reused': (mg.last_timings or {}).get('setup_reused'),
            'instrumented_wall_seconds': _round_optional(
                (mg.last_timings or {}).get('wall_seconds')),
            'threads': [{'tid': t['tid'], 'device': t['device'],
                         'ref_time': _round_optional(t['ref_time']),
                         'mg_time': _round_optional(t['mg_time']),
                         'mg_residual_seconds': _round_optional(
                             t.get('mg_residual_seconds')),
                         'mg_residual': t.get('mg_residual')}
                        for t in r['threads']],
        })
    assert r is not None
    threads = r['threads']
    ref_max = (max(t['ref_time'] for t in threads)
               if run_reference else None)
    mg_max = max(t['mg_time'] for t in threads)
    report = {
        'lat': lat, 'nthreads': nthreads, 'device_ids': device_ids,
        'num_restart': num_restart, 'coarse_max_iter': coarse_max_iter,
        'coarse_tol_factor': coarse_tol_factor,
        'threads': [{'tid': t['tid'], 'device': t['device'],
                     'ref_time': _round_optional(t['ref_time']),
                     'mg_time': _round_optional(t['mg_time']),
                     'mg_residual_seconds': _round_optional(
                         t.get('mg_residual_seconds')),
                     'mg_residual': t.get('mg_residual')}
                    for t in threads],
        'multi_ref_wall': None if ref_max is None else round(ref_max, 4),
        'multi_mg_wall': round(mg_max, 4),
        'mg_over_ref_ratio': (None if ref_max is None
                              else round(mg_max / ref_max, 4)),
        'warmups': warmups,
        'repeats': repeats,
        'run_reference': run_reference,
        'verify_residual': verify_residual,
        'input_tag': input_tag,
        'calls': calls,
        'total_solve_median': round(statistics.median(
            call['total_solve'] for call in calls), 6),
    }
    report['memory'] = _memory_snapshot(mg.device_ids)
    mg.close()
    return report


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_snapshot(device_ids):
    import torch

    records = []
    for device_id in device_ids:
        device = torch.device('cuda', int(device_id))
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        records.append({
            'device': int(device_id),
            'name': torch.cuda.get_device_name(device),
            'allocated_bytes': int(torch.cuda.memory_allocated(device)),
            'reserved_bytes': int(torch.cuda.memory_reserved(device)),
            'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
            'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
            'device_free_bytes': int(free_bytes),
            'device_total_bytes': int(total_bytes),
        })
    return records


def _load_real_inputs(gauge_path, lat, seed):
    import torch
    from pyqcu import tools
    from pyqcu.testing.simple._common import (
        lime_metadata, qdp_gauge_to_qcu, read_h5, read_h5_metadata,
        read_qdp_gauge, resolve_path,
    )

    path = resolve_path(gauge_path, resolve_path(None, gauge_path))
    if path.suffix.lower() == '.lime':
        metadata = lime_metadata(path)
        gauge_full = qdp_gauge_to_qcu(read_qdp_gauge(path))
    elif path.suffix.lower() in ('.h5', '.hdf5'):
        metadata = read_h5_metadata(path)
        gauge_full = read_h5(path)
    else:
        raise ValueError('--gauge must be .lime, .h5, or .hdf5')
    if tuple(gauge_full.shape[-4:]) != tuple(lat):
        raise ValueError(
            f'gauge lattice {tuple(gauge_full.shape[-4:])} does not match {tuple(lat)}')
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    shape = (4, 3, *lat)
    source_full = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)
    gauge = tools.oooxyzt2poooxyzt(
        torch.from_numpy(np.ascontiguousarray(gauge_full, dtype=np.complex64)))
    source = tools.oooxyzt2poooxyzt(torch.from_numpy(source_full))
    tag = (
        f'{_sha256_file(path)}:mg-source:{seed}:'
        f'{hashlib.sha256(memoryview(source_full).cast("B")).hexdigest()}'
    )
    return gauge, source, tag, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lat', default='8x8x8x16',
                        help='lattice dimensions formatted as XxYxZxT')
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--device-ids', nargs='+', type=int, default=None)
    parser.add_argument('--num-levels', type=int, default=2)
    parser.add_argument('--dof', nargs='+', type=int, default=None,
                        help='degrees of freedom per level, e.g. 12 24')
    parser.add_argument('--mg-grid', nargs=4, type=int,
                        default=[2, 2, 2, 2])
    parser.add_argument('--mass', type=float, default=None,
                        help='default: gauge metadata, then 0.05')
    parser.add_argument('--csw', type=float, default=None,
                        help='default: gauge metadata, then 1.0')
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-restart', type=int, default=5)
    parser.add_argument('--coarse-max-iter', type=int, default=15)
    parser.add_argument('--coarse-tol-factor', type=float, default=1e5)
    parser.add_argument('--nv-iters', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--no-reference', action='store_true')
    parser.add_argument('--gauge', default=None,
                        help='real LIME/HDF5 gauge; enables residual verification')
    parser.add_argument('--verify-residual', action='store_true')
    parser.add_argument('--deflate', action='store_true',
                        help='apply one initial coarse-space deflation before Krylov')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    lat = [int(value) for value in args.lat.split('x')]
    if len(lat) != 4 or any(value <= 0 for value in lat):
        parser.error('--lat must contain four positive XxYxZxT dimensions')
    if args.nthreads <= 0:
        parser.error('--nthreads must be positive')
    if args.dof is not None and len(args.dof) < args.num_levels:
        parser.error('--dof needs one value per requested level')
    if args.output:
        log_dir = Path(args.output).expanduser().parent / 'backend_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('QCU_LOG_DIR', str(log_dir))
    if args.warmups < 0 or args.repeats <= 0:
        parser.error('--warmups must be non-negative and --repeats positive')
    gauge = source = input_tag = None
    metadata = {}
    if args.gauge is not None:
        gauge, source, input_tag, metadata = _load_real_inputs(
            args.gauge, tuple(lat), args.seed)
        args.verify_residual = True
    mass = (args.mass if args.mass is not None else
            float(metadata.get('reference_mass', metadata.get('mass', 0.05))))
    csw = (args.csw if args.csw is not None else
           float(metadata.get('csw', 1.0)))
    res = bench(
        lat, mass=mass, atol=args.atol, num_levels=args.num_levels,
        nthreads=args.nthreads, device_ids=args.device_ids,
        num_restart=args.num_restart,
        coarse_max_iter=args.coarse_max_iter,
        coarse_tol_factor=args.coarse_tol_factor, nv_iters=args.nv_iters,
        use_cache=not args.no_cache, cache_dir=args.cache_dir,
        csw=csw, seed=args.seed, warmups=args.warmups,
        repeats=args.repeats, run_reference=not args.no_reference,
        gauge=gauge, source=source, input_tag=input_tag,
        verify_residual=args.verify_residual,
        dof_list=args.dof, mg_grid=args.mg_grid,
        use_deflate=args.deflate)
    res['input'] = {
        'gauge': None if args.gauge is None else str(Path(args.gauge).expanduser()),
        'metadata': metadata,
        'mass': mass,
        'csw': csw,
    }
    res['log_dir'] = os.environ.get('QCU_LOG_DIR')

    # The C++ backend logs to stdout; keep the machine-readable result isolated.
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s = json.dumps(res, indent=1)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(s, encoding='utf-8')
        print(f"bench result -> {output}")
    else:
        print(s)
