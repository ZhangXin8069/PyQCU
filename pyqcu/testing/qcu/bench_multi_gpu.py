import time, json
import torch
from pyqcu.cuda._multi_gpu import MultiGpuMultigrid

def bench(lat, mass=0.05, atol=1e-6, num_levels=2, nthreads=3,
          device_ids=None, num_restart=5, coarse_max_iter=15,
          coarse_tol_factor=1e5, nv_iters=2, verbose=False):
    mg = MultiGpuMultigrid(lat_size=lat, mass=mass, atol=atol,
                           num_levels=num_levels, nthreads=nthreads,
                           device_ids=device_ids, num_restart=num_restart,
                           coarse_max_iter=coarse_max_iter,
                           coarse_tol_factor=coarse_tol_factor,
                           nv_iters=nv_iters, verbose=verbose)
    t0 = time.perf_counter()
    r = mg.solve()
    total = time.perf_counter() - t0
    threads = r['threads']
    ref_max = max(t['ref_time'] for t in threads)
    mg_max = max(t['mg_time'] for t in threads)
    return {
        'lat': lat, 'nthreads': nthreads, 'device_ids': device_ids,
        'num_restart': num_restart, 'coarse_max_iter': coarse_max_iter,
        'coarse_tol_factor': coarse_tol_factor,
        'threads': [{'tid': t['tid'], 'device': t['device'],
                     'ref_time': round(t['ref_time'], 4),
                     'mg_time': round(t['mg_time'], 4)} for t in threads],
        'multi_ref_wall': round(ref_max, 4),
        'multi_mg_wall': round(mg_max, 4),
        'speedup_mg_over_ref': round(ref_max / mg_max, 4),
        'total_solve': round(total, 4),
    }

if __name__ == '__main__':
    import sys
    lat = [int(x) for x in sys.argv[1].split('x')]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    res = bench(lat)
    # C++ 后端 verbose 输出走 stdout（lattice set init 等），JSON 结果写 stderr
    # 或指定文件，避免被污染。
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s = json.dumps(res, indent=1)
    if out:
        with open(out, 'w') as f:
            f.write(s)
        print(f"bench result -> {out}", file=sys.stderr)
    else:
        print(s, file=sys.stderr)
