#!/usr/bin/env python3
"""Portable launcher for the unchanged LX2 STREAM v2 probe.

This launcher is a packaging adaptation; the C kernels and plans are unchanged.
It respects the current CPU affinity and has no scheduler or account dependency.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

BASE = Path(__file__).resolve().parent


def cpulist(value):
    result = []
    for item in value.strip().split(','):
        if not item:
            continue
        ends = item.split('-')
        result.extend(range(int(ends[0]), int(ends[-1]) + 1))
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meminfo(node):
    text = Path(f'/sys/devices/system/node/node{node}/meminfo').read_text()
    return {key: int(re.search(r'\b' + key + r':\s+(\d+) kB', text)[1]) * 1024
            for key in ('MemTotal', 'MemFree')}


def summarize(rows):
    samples = [r for r in rows if r['type'] == 'sample']
    if not rows or rows[-1].get('type') != 'done' or not rows[-1].get('correct'):
        raise ValueError('Incomplete probe output')
    if any(not r.get('correct', True) for r in rows):
        raise ValueError('Correctness or placement failure')
    import statistics
    grouped = collections.defaultdict(list)
    for row in samples:
        ns = row['window_end_ns'] - row['window_start_ns']
        if ns <= 0 or len({w['cpu'] for w in row['workers']}) != len(row['workers']):
            raise ValueError('Invalid timing or CPU mapping')
        grouped[(row['case'], row['kernel'])].append(row)
    output = []
    for (case, kernel), values in sorted(grouped.items()):
        rates = [r['logical_bytes'] / (r['window_end_ns'] - r['window_start_ns']) for r in values]
        mean = statistics.mean(rates)
        cv = 100 * statistics.pstdev(rates) / mean if mean else 0
        flags = set()
        if cv > 5:
            flags.add('bandwidth_cv_above_5_percent')
        for row in values:
            ns = row['window_end_ns'] - row['window_start_ns']
            for w in row['workers']:
                if w['start_lag_ns'] > .01 * ns:
                    flags.add('start_lag_above_1_percent_window')
                if w['overrun_ns'] > .01 * ns:
                    flags.add('final_batch_overrun_above_1_percent_window')
                if w['pmu_errno'] or not w['time_enabled_ns'] or not .995 <= w['time_running_ns'] / w['time_enabled_ns'] <= 1:
                    flags.add('pmu_unavailable_or_multiplexed')
        output.append(dict(case=case, kernel=kernel, rounds=len(values),
                           median_GBps=statistics.median(rates), min_GBps=min(rates),
                           max_GBps=max(rates), cv_percent=cv, quality_flags=sorted(flags)))
    if not samples:
        raise ValueError('No samples')
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', choices=['smoke', 'numa', 'hierarchy', 'access', 'cache', 'anchors'], default='smoke')
    ap.add_argument('--output', type=Path, required=True, help='New output directory; never overwritten')
    ap.add_argument('--seconds', type=float, default=.15)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--domain', type=int, default=2, help='smoke only, compute domain 0..15')
    ap.add_argument('--threads', type=int, default=1, help='smoke only, 1..38')
    ap.add_argument('--tier', choices=['DDR', 'HBM'], default='DDR', help='smoke only')
    ap.add_argument('--array-mib', type=int, default=16, help='smoke only, each of three arrays per worker')
    ap.add_argument('--plan-only', action='store_true', help='Validate placement inputs and save plan without running')
    args = ap.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        ap.error('Run on an allocated Linux AArch64 LX2 node')
    if not .1 <= args.seconds <= 2 or not 1 <= args.rounds <= 20:
        ap.error('seconds must be 0.1..2; rounds must be 1..20')
    if not 0 <= args.domain < 16 or not 1 <= args.threads <= 38 or args.array_mib < 1:
        ap.error('Invalid smoke configuration')
    cpus = {d: cpulist(Path(f'/sys/devices/system/node/node{d}/cpulist').read_text()) for d in range(16)}
    if any(len(v) != 38 for v in cpus.values()):
        ap.error('Expected 16 compute NUMA domains with 38 physical CPUs each; inspect topology')
    packages = {}
    for d, values in cpus.items():
        ids = {int(Path(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read_text()) for c in values}
        if len(ids) != 1:
            ap.error('A compute domain spans packages')
        packages[d] = ids.pop()
    if [packages[d] for d in range(16)] != [0] * 8 + [1] * 8:
        ap.error('Unexpected package mapping')
    for d in range(16):
        distances = list(map(int, Path(f'/sys/devices/system/node/node{d}/distance').read_text().split()))
        for t in range(16):
            expected = 10 if d == t else 20 if d // 4 == t // 4 else 30 if d // 8 == t // 8 else 40
            if len(distances) < 32 or distances[t] != expected or distances[t+16] != expected+51:
                ap.error('Unexpected NUMA/die/HBM distance mapping')
    if args.batch == 'smoke':
        d = args.domain
        node = d + (16 if args.tier == 'HBM' else 0)
        cases = [dict(name=f'smoke_d{d}_t{args.threads}_{args.tier}',
                      array_bytes_per_worker=args.array_mib * 1024**2,
                      workers=[[c, node, node, node, d] for c in cpus[d][:args.threads]])]
    else:
        if args.batch == 'anchors':
            from plan import make_plan
        else:
            from stream_plan_base import make_plan
        cases = make_plan(cpus, args.batch)
    allowed = set(os.sched_getaffinity(0))
    required_cpus = {w[0] for c in cases for w in c['workers']}
    if not required_cpus <= allowed:
        ap.error(f'Current CPU affinity excludes {len(required_cpus - allowed)} requested CPUs; use a matching allocation/affinity')
    nodes = sorted({n for c in cases for w in c['workers'] for n in w[1:4]})
    before = {n: meminfo(n) for n in nodes}
    offline = [n for n, info in before.items() if not info['MemTotal']]
    if offline:
        ap.error(f'Skip this faulty configuration/node: memory NUMA offline {offline}')
    peak = collections.defaultdict(int)
    for c in cases:
        use = collections.Counter()
        for w in c['workers']:
            for n in w[1:4]:
                use[n] += c['array_bytes_per_worker']
        for n, count in use.items():
            peak[n] = max(peak[n], count)
    if any(before[n]['MemFree'] < size + 64 * 1024**2 for n, size in peak.items()):
        ap.error('Insufficient per-domain free memory (three allocated arrays plus 64 MiB reserve)')
    binary = BASE / 'stream_probe'
    if not args.plan_only and not binary.is_file():
        ap.error('Run ./build.sh first')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (out / 'cases.txt').write_text(''.join(
        f'{c["name"]} {len(c["workers"])} {c["array_bytes_per_worker"]}\n'
        + ''.join(' '.join(map(str, w)) + '\n' for w in c['workers']) for c in cases))
    command = [str(binary), str(out / 'cases.txt'), str(args.seconds), str(args.rounds)]
    state = dict(batch=args.batch, seconds=args.seconds, rounds=args.rounds,
                 case_count=len(cases), status='planned', allowed_cpus=sorted(allowed),
                 cpus_by_domain=cpus, packages=packages, peak_array_bytes_by_memory_node=dict(peak),
                 memory_before=before, source_sha256=digest(BASE / 'stream_probe.c'),
                 command=command, frequency_policy='unmodified; inspect PMU measurements',
                 launcher='portable packaging adaptation')
    if binary.is_file():
        state['binary_sha256'] = digest(binary)
    def save():
        (out / 'run.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    if args.plan_only:
        print(f'Plan saved: {out} ({len(cases)} configurations)')
        return
    state['status'] = 'running'
    save()
    try:
        with (out / 'samples.jsonl').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
            p = subprocess.run(command, stdout=stdout, stderr=stderr)
        state['returncode'] = p.returncode
        state['memory_after'] = {n: meminfo(n) for n in nodes}
        if p.returncode:
            raise RuntimeError(f'Probe failed: {p.returncode}; see stderr.txt')
        if any(not x['MemTotal'] for x in state['memory_after'].values()):
            raise RuntimeError('Memory domain went offline during run; exclude result')
        rows = [json.loads(line) for line in (out / 'samples.jsonl').read_text().splitlines()]
        summary = summarize(rows)
        if len(summary) != len(cases) * 4 or any(c['rounds'] != args.rounds for c in summary):
            raise RuntimeError('Missing configurations, kernels or rounds')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        state.update(status='completed', raw_sha256=digest(out / 'samples.jsonl'))
        print(f'Completed: {out} ({len(summary)} operation/configuration results)')
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
