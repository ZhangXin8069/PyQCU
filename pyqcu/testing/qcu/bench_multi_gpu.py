import argparse
import json
import statistics
import time
from pathlib import Path

from pyqcu.cuda._multi_gpu import MultiGpuMultigrid


def _round_optional(value, digits=4):
    return None if value is None else round(value, digits)


def bench(lat, mass=0.05, atol=1e-6, num_levels=2, nthreads=3,
          device_ids=None, num_restart=5, coarse_max_iter=15,
          coarse_tol_factor=1e5, nv_iters=2, verbose=False,
          use_cache=True, cache_dir=None, csw=1.0, seed=42,
          warmups=1, repeats=3, run_reference=True):
    mg = MultiGpuMultigrid(lat_size=lat, mass=mass, atol=atol,
                           num_levels=num_levels, nthreads=nthreads,
                           device_ids=device_ids, num_restart=num_restart,
                           coarse_max_iter=coarse_max_iter,
                           coarse_tol_factor=coarse_tol_factor,
                           nv_iters=nv_iters, verbose=verbose,
                           use_cache=use_cache, cache_dir=cache_dir,
                           csw=csw, seed=seed,
                           run_reference=run_reference)
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
            'threads': [{'tid': t['tid'], 'device': t['device'],
                         'ref_time': _round_optional(t['ref_time']),
                         'mg_time': _round_optional(t['mg_time'])}
                        for t in r['threads']],
        })
    assert r is not None
    threads = r['threads']
    ref_max = (max(t['ref_time'] for t in threads)
               if run_reference else None)
    mg_max = max(t['mg_time'] for t in threads)
    return {
        'lat': lat, 'nthreads': nthreads, 'device_ids': device_ids,
        'num_restart': num_restart, 'coarse_max_iter': coarse_max_iter,
        'coarse_tol_factor': coarse_tol_factor,
        'threads': [{'tid': t['tid'], 'device': t['device'],
                     'ref_time': _round_optional(t['ref_time']),
                     'mg_time': _round_optional(t['mg_time'])}
                    for t in threads],
        'multi_ref_wall': None if ref_max is None else round(ref_max, 4),
        'multi_mg_wall': round(mg_max, 4),
        'mg_over_ref_ratio': (None if ref_max is None
                              else round(mg_max / ref_max, 4)),
        'warmups': warmups,
        'repeats': repeats,
        'run_reference': run_reference,
        'calls': calls,
        'total_solve_median': round(statistics.median(
            call['total_solve'] for call in calls), 6),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lat', default='8x8x8x16',
                        help='lattice dimensions formatted as XxYxZxT')
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--device-ids', nargs='+', type=int, default=None)
    parser.add_argument('--num-levels', type=int, default=2)
    parser.add_argument('--mass', type=float, default=0.05)
    parser.add_argument('--csw', type=float, default=1.0)
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
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    lat = [int(value) for value in args.lat.split('x')]
    if len(lat) != 4 or any(value <= 0 for value in lat):
        parser.error('--lat must contain four positive XxYxZxT dimensions')
    if args.nthreads <= 0:
        parser.error('--nthreads must be positive')
    if args.warmups < 0 or args.repeats <= 0:
        parser.error('--warmups must be non-negative and --repeats positive')
    res = bench(
        lat, mass=args.mass, atol=args.atol, num_levels=args.num_levels,
        nthreads=args.nthreads, device_ids=args.device_ids,
        num_restart=args.num_restart,
        coarse_max_iter=args.coarse_max_iter,
        coarse_tol_factor=args.coarse_tol_factor, nv_iters=args.nv_iters,
        use_cache=not args.no_cache, cache_dir=args.cache_dir,
        csw=args.csw, seed=args.seed, warmups=args.warmups,
        repeats=args.repeats, run_reference=not args.no_reference)

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
