LX2 STREAM 源码、运行脚本与测试报告

1. 文件

stream_probe.c       原实测 v2 源码，316 行，按字节保留。
stream_probe_v1.c    早期 v1 源码；1.55GHz 初始 numa 批次使用它。
stream_plan_base.py  原完整测试计划：numa / hierarchy / access / cache。
plan.py             原多节点 anchors 计划，101 个配置。
build.sh            按原编译选项编译 v2，编译器路径由 CC 或 PATH 指定。
run.sh              设置原 OpenMP 等待策略，运行便携启动器。
run_stream.py       本次整理的便携启动器，无个人路径、队列或作业系统依赖。
report.html         既有测试报告，含两组频率、代表配置与完整配置筛选表。
data/stream-results.json  既有完整数据，含每轮值、分域分量、物理映射及质量标记。
make_report.py      从上述 JSON 重新生成同一报告。
SHA256SUMS          文件校验清单。

本次导出未重新测量节点性能。C 源码和测试计划保持原样；build/run 包装脚本
为本次便携整理版本，已离线检查，但尚未在 LX2 上以这个包装版本重跑。

2. 编译

在已分配的 Linux AArch64 LX2 节点上加载 BiSheng/LLVM 的编译器与 OpenMP
运行时环境。原测试使用 BiSheng Enterprise 5.2.0 / clang 19.1.7。
确保 clang 在 PATH，OpenMP 动态库可由运行时找到；也可指定 CC 为编译器路径。

    ./build.sh

等价编译参数：
    clang -O3 -fopenmp -march=armv9-a+sme+sme-f64f64 \
      -msve-vector-bits=scalable stream_probe.c -o stream_probe

实际内核使用普通 SVE；程序要求所有 worker 的 SVE VL 为 64B。
编译参数保留原测试设置。直接运行要求 mbind/move_pages 查询可用；PMU 不可用
时 C 程序仍可能完成带宽测试，summary 会标记，不能声称已验证实际 PMU 频率。

3. 运行

启动器不会申请节点或扩大 CPU affinity。先在匹配的计算资源分配中运行；
按该机器的 CPU / die / NUMA / HBM 拓扑校验，并检查每个目标域容量。
HBM 掉线或内存不足时停止该配置，换健康节点或改为有效 DDR 配置。
输出目录必须不存在，避免覆盖已有测量。

单域单核 DDR 小范围检查（每数组 16MiB）：
    ./run.sh --batch smoke --domain 2 --threads 1 --tier DDR --output results/smoke-ddr

单域 36 核 HBM：
    ./run.sh --batch smoke --domain 2 --threads 36 --tier HBM --output results/smoke-hbm

完整原计划，先仅生成并检查，不执行：
    ./run.sh --batch hierarchy --output results/hierarchy-plan --plan-only

执行各维度，每配置自动运行 Copy/Scale/Add/Triad，各 5 轮、每轮 0.15s：
    ./run.sh --batch numa --output results/numa
    ./run.sh --batch hierarchy --output results/hierarchy
    ./run.sh --batch access --output results/access
    ./run.sh --batch cache --output results/cache
    ./run.sh --batch anchors --output results/anchors

--seconds 可设 0.1..2；--rounds 可设 1..20。
--domain / --threads / --tier / --array-mib 仅用于 smoke。
numa 测 16 域及核心数扩展；hierarchy 测 die / CPU / 全节点 / 跨 die；
access 测远访、DDR/HBM 分流及跨 tier；cache 测 4KiB 至 16MiB 的工作集。
默认是每数组每 worker 16MiB：576 worker 时实际分配三数组共 27GiB。
程序不调频。默认目标 1.55GHz；2.0GHz 需要在相应频率的实际资源上另跑，
不能仅修改结果标签或用比例缩放。

run.sh 使用 OMP_DYNAMIC=FALSE、OMP_PROC_BIND=FALSE、OMP_WAIT_POLICY=PASSIVE、
KMP_BLOCKTIME=0；每个 worker 由 C 程序显式绑核。

4. 自定义 cases.txt

每个配置一行：
    case_name worker_count array_bytes_per_worker
后接 worker_count 行：
    cpu_id x_memory_node y_memory_node z_memory_node cpu_domain_group

无注释、名称不含空格。每配置 1..576 worker，CPU 不重复且须在当前 affinity 内。
内存节点 0..15 为 DDR、16..31 为配对 HBM；group 0..15 用于计算域分组。
数组大小至少 4096 字节，满足 4096 及 8*VL 对齐；小于 1MiB 时须整除 1MiB，
大于等于 1MiB 时须为 1MiB 的整数倍。示例（先确认 CPU 0 可用且位于域 0）：

    local_ddr_1 1 16777216
    0 0 0 0 0

直接调用：
    ./stream_probe cases.txt 0.15 5 > samples.jsonl 2> stderr.txt

直接调用不会执行启动器的容量/拓扑预检查或生成 summary。

5. 计时与数据口径

自定义 STREAM-style fixed-window 探针，非官方 STREAM reference 计时流程。
FP64：Copy z=x；Scale z=3*x；Add z=x+y；Triad z=x+3*y（FMA）。
每元素逻辑字节分别 16/16/24/24，FLOP 分别 0/1/1/2。
只计共同窗口内完整结束的 1MiB 输出批次，记录起跑滞后和末批超时。
先热身，分配/初始化/正确性检查/页归属检查不计入带宽窗口。
这是稳态带宽，不是含初始化的完整算子端到端时间，也不是 repack 带宽。
固定窗口内各 worker 可完成不同的工作量，分流吞吐不能直接代替固定等分任务耗时。
MADV_HUGEPAGE 是申请建议，本探针不证明实际大页覆盖；预取状态未知且未修改。

GB/s = logical_bytes / (window_end_ns - window_start_ns)。
输出 payload GB/s 为写出字节/窗口 ns，Copy/Scale 逻辑带宽为 payload 的 2 倍，
Add/Triad 为 3 倍。逻辑带宽不包含推断出来的写分配等额外物理流量。

新运行输出：cases.json/txt、run.json、samples.jsonl、stderr.txt、summary.json。
summary 含各配置/操作的轮次中位数、范围、CV 与质量标记；原始 samples 保留
每 worker 的读写字节、PMU、页位置相关记录、起跑滞后和超时，便于进一步分析。
该便携 summary 是简明汇总，不等同于原测试的数据审核工具。

6. 报告与校验

直接打开 report.html；完整数据为 data/stream-results.json。
报告范围是各样本内 5 轮中位数的最小–最大值，不是置信区间。
sample 是频率组内配置样本标签，不能跨频率连接为相同物理节点。
保留质量标记，尤其注意分流 Copy 的波动与大线程组起跑滞后。
v1 与 v2 的四个计算内核相同；v2 在线程数 >72 时将起跑提前量从 10ms 改为
100ms。报告有每批次对应版本。旧数据未改用新脚本生成的值。

重新生成报告：
    python3 -B make_report.py
检查文件完整性（Linux）：
    sha256sum -c SHA256SUMS

v2 C SHA256:
6fa2e0f311a763e05461e73966a738c64a9c56b395d8da172b3a75c037c286de
v1 C SHA256:
b4d3207e0f653eb934a0342b32e4e04a572fef3421bc139c840d2b9e309ae94c
