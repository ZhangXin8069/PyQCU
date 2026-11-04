#!/usr/bin/env python3
"""Render the bundled historical STREAM measurements without changing statistics."""
import collections
import html
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OPS = ('copy', 'scale', 'add', 'triad')


def span(values):
    lo, hi = min(values), max(values)
    return f'{lo:.2f}' if len(values) == 1 else f'{lo:.2f}–{hi:.2f}'


def main():
    data = json.loads((BASE / 'data/stream-results.json').read_text())
    entries = data['entries']
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in entries:
        for key, c in e['cases'].items():
            assert key == c['name'] + ':' + c['kernel']
            grouped[(e['frequency_GHz'], c['name'])][c['kernel']].append((e, c))
    parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>LX2 STREAM 测试报告</title><style>
body{font:15px/1.65 system-ui,sans-serif;margin:32px;color:#172331;background:#f7f9fb}
main{max-width:1500px;margin:auto}h1{font-size:28px}h2{margin-top:32px}p{max-width:1120px}
table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}
th,td{border:1px solid #dce2e8;padding:7px 10px;text-align:left;white-space:nowrap}
th{background:#e8eef4;position:sticky;top:0}.scroll{overflow:auto;max-height:72vh}
code{background:#e8eef4;padding:2px 5px}.note{border-left:4px solid #406b91;padding-left:16px}
input,select{font:inherit;padding:7px;margin:8px 14px 14px 0}.flag{color:#9c4d00}
</style><main><h1>LX2 STREAM 测试报告</h1>
<p>本报告整理既有实测数据，本次未重新运行节点基准。默认优化参考为 1.55GHz；2.0GHz 单列比较。
文件 <a href="data/stream-results.json">完整数据 JSON</a> 保留每轮数值、统计、线程/内存映射和质量标记。</p>
<h2>方法与口径</h2>
<p>FP64，SVE VL=64B，普通 temporal store；Triad 使用 FMA。每线程独立数组，显式绑核，
mbind 分配；初始化后和测试结束前通过 move_pages 查询每个基页归属。
请求 MADV_HUGEPAGE，不代表已核实实际大页覆盖。硬件预取状态未知，未修改预取或频率设置。</p>
<table><tr><th>操作</th><th>表达式</th><th>逻辑字节/元素</th><th>FLOP/元素</th></tr>
<tr><td>Copy</td><td>z=x</td><td>16</td><td>0</td></tr><tr><td>Scale</td><td>z=3*x</td><td>16</td><td>1</td></tr>
<tr><td>Add</td><td>z=x+y</td><td>24</td><td>1</td></tr><tr><td>Triad</td><td>z=x+3*y</td><td>24</td><td>2</td></tr></table>
<p>每配置每操作 5 轮，每轮共同计时窗口 0.15s；仅统计窗口内完成的 1MiB 输出批次。
带宽 GB/s 为十进制逻辑读写量/窗口时间，不是内存控制器实际传输字节。
这是自定义 STREAM-style 固定窗口测试，不是官方 STREAM reference 的计时流程。
Copy/Scale 活跃两数组，Add/Triad 活跃三数组；程序始终分配三数组。
大数组默认每数组每线程 16MiB；cache 系列另列大小。</p>
<p>拓扑：2 CPU × 2 die/CPU × 4 计算 NUMA/die，每域 38 物理核，配对 HBM 为内存域。
全节点 576 核配置每域使用 36 核。每个 worker 的输入、第二输入及输出可分别放在 DDR/HBM 或远端域。</p>
<p class="note">表中范围为相同配置在各样本内先取 5 轮中位数，再取这些中位数的最小–最大值；
不是置信区间。n 为配置样本数，不保证独立物理节点数。不同频率中的同名 sample 不表示同一物理节点。
两组频率来自不同样本，不能将全部差异解释为纯频率效应。带质量标记的数据保留展示，不静默删去。</p>''']
    parts.append('<h2>数据覆盖</h2><table><tr><th>目标 GHz</th><th>配置批次</th><th>配置×操作结果</th><th>计时轮次</th><th>带质量标记结果</th></tr>')
    for freq in (1.55, 2.0):
        es = [e for e in entries if e['frequency_GHz'] == freq]
        count = sum(len(e['cases']) for e in es)
        rounds = sum(c['total_logical_GBps']['n'] for e in es for c in e['cases'].values())
        flagged = sum(bool(c['quality_flags']) for e in es for c in e['cases'].values())
        parts.append(f'<tr><td>{freq:.2f}</td><td>{len(es)}</td><td>{count}</td><td>{rounds}</td><td>{flagged}</td></tr>')
    parts.append('</table><h2>代表性配置（逻辑 GB/s）</h2>')

    def table(keys, all_rows=False):
        start = '<div class="scroll"><table' + (' id="results"' if all_rows else '') + '><thead><tr>'
        heads = ['GHz', '配置 / 层级', '总线程', '每数组 KiB/线程', 'n', 'Copy', 'Scale', 'Add', 'Triad', '最大轮内 CV %', '有标记结果数']
        out = [start + ''.join('<th>' + h + '</th>' for h in heads) + '</tr></thead><tbody>']
        for freq, name in keys:
            ops = grouped[(freq, name)]
            assert set(ops) == set(OPS)
            c = ops['copy'][0][1]
            assert len({len(ops[o]) for o in OPS}) == 1
            all_cases = [item for values in ops.values() for item in values]
            fields = [freq, name + ' / ' + c['kind'], c['threads'], c['array_bytes_per_worker'] // 1024, len(ops['copy'])]
            fields += [span([row['total_logical_GBps']['median'] for _, row in ops[o]]) for o in OPS]
            fields += [f'{max(row["total_logical_GBps"]["cv_percent"] for _, row in all_cases):.2f}',
                       sum(bool(row['quality_flags']) for _, row in all_cases)]
            out.append(f'<tr data-frequency="{freq}">' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in fields) + '</tr>')
        out.append('</tbody></table></div>')
        return ''.join(out)

    selected = ['numa_d2_t36_DDR', 'numa_d2_t36_HBM', 'die0_t9_DDR', 'spread4_t9_DDR',
                'node_t36_DDR', 'node_t36_HBM', 'split_node_18_18']
    parts.append(table([(f, n) for f in (1.55, 2.0) for n in selected]))
    parts.append('''<h2>结果解读</h2>
<p>同为 36 个计算核，die0_t9_DDR 是同一 die 四域各 9 核，spread4_t9_DDR 是跨四 die 各一域各 9 核。
比较两者可见数据随任务跨 die 分布的价值；不能将单域带宽直接乘 NUMA 数量。
全节点 HBM 相比 DDR 具有更高并行读写服务能力，但工作集、NUMA 位置和实际线程分配必须匹配。</p>
<p>split_* 表示一部分线程仅访问 DDR、另一部分仅访问 HBM；跨 tier 的单线程混合访问另看 cross_tier_*。
split 的总吞吐不能直接作为任意固定 DDR/HBM 工作量比例的完成速度；完整数据保留两侧分量。
相同总容量下，两数组与三数组的 cache 驻留不同；cache_* 系列不能充当大数组 DRAM 带宽。</p>
<p>质量标记包括轮内 CV 超过 5%、起跑滞后超过窗口 1%、末批超时超过窗口 1%。
最大 CV 列取该行所有操作/样本的最大值；标记数按配置×操作样本计数，详见 JSON quality_flags。
PMU 有效频率与 sysfs 报告值分别保留，分析频率时不要只看 sysfs。</p>
<h2>全部配置</h2><p>remote_*：same_die / other_die / other_cpu；cross_tier_* 的三个字母依次为 x、y、z 的 DDR/HBM 层。
Copy/Scale 不读取 y，y 的分配位置不应增加其带宽流量。完整 worker 映射、分域带宽及每轮值在 JSON 中。</p>
<label>频率 <select id="frequency"><option value="">全部</option><option>1.55</option><option value="2.0">2.0</option></select></label>
<label>配置筛选 <input id="search" placeholder="如 node_t36、remote、cache"></label>''')
    parts.append(table(sorted(grouped), True))
    parts.append('<h2>样本与源码版本</h2><p>v1 与 v2 的四个计算内核相同；v2 在线程数超过 72 时，将共同起跑前等待从 10ms 改为 100ms。1.55GHz 的早期 numa 批次使用 v1，其他列出的批次使用 v2。</p><table><tr><th>GHz</th><th>sample</th><th>batch</th><th>probe</th><th>配置×操作</th></tr>')
    for e in entries:
        parts.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in [e['frequency_GHz'], e['sample'], e['batch'], e['probe_version'], len(e['cases'])]) + '</tr>')
    parts.append('''</table><p>源码：<a href="stream_probe.c">v2</a> / <a href="stream_probe_v1.c">v1</a>；
<a href="README.txt">编译、运行与结果说明</a>。本次提供的便携启动脚本去除了部署依赖，未经新的 LX2 节点实跑；
C 源码和测试计划与原测试留存文件按字节核对一致。</p></main><script>
function filter(){const q=document.getElementById('search').value.toLowerCase();
const f=document.getElementById('frequency').value;
document.querySelectorAll('#results tbody tr').forEach(r=>{
r.hidden=!!((f&&Number(r.dataset.frequency)!==Number(f))||!r.textContent.toLowerCase().includes(q));});}
document.getElementById('search').addEventListener('input',filter);
document.getElementById('frequency').addEventListener('change',filter);
</script></html>''')
    (BASE / 'report.html').write_text('\n'.join(parts))
    print(f'Report: {len(entries)} batches, {len(grouped)} distinct frequency/configuration rows')


if __name__ == '__main__':
    main()
