# dev87 Strict FGMRES 性能/显存热点审计

日期：2026-08-31

## 范围与结论

本次仅使用已完成的短 smoke、一次内存态 `torch.profiler` 和静态代码计数；没有继续运行长时间 profiler 或正式 benchmark，也没有修改核心代码。

当前最可信的第一主导项是粗层 BiCGStab 的迭代量及其重复的粗层 parity hopping，而不是 fine dslash：`strict_hopping_parity_kernel` 占 4 次 FGMRES 调用 CUDA kernel 时间约 73.9%，平均约 598 次发射/solve；粗层 BiCGStab 的 `p`/`s` kernel 各约 148 次/solve。需要注意，这证明了“时间集中在哪里”，尚不能单独证明高迭代数的根因是某个具体粗算子或预条件器参数。

## 已完成的快速测量

### 1. Strict fast gate

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B examples/qcu/dev87/run_strict_fast.py \
  --only cuda-fused-fgmres --fail-fast
```

退出码：`0`；结果：`3 passed`；runner 总耗时：`5.2195 s`。出现 P100/sm60 PyTorch warning，但 V100 路径通过。

### 2. PyQCU Strict smoke

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B \
  examples/qcu/dev87/bench_strict_vs_quda.py \
  --profile smoke --side pyqcu --cache-expect hit --repeats 1
```

退出码：`0`；外层耗时：`40.4163 s`。

| 项目 | 结果 |
|---|---:|
| cache-hit setup | `14.9545 s` |
| cache restore | `13.2713 s` |
| 非 cache setup | `1.6832 s` |
| 两次 warmup | `2.10025 s`、`3.10384 s` |
| steady | `2.06452 s` |
| FGMRES outer iterations | `11` |
| 真残差 | `3.6013e-7` |

#### 显存

| 阶段/对象 | 字节 | GiB |
|---|---:|---:|
| setup device-wide peak | `11,219,046,400` | `10.4486` |
| first solve device-wide peak | `11,722,362,880` | `10.9173` |
| first solve 增量 | `503,316,480` | `0.46875` |
| PyTorch setup allocated/reserved peak | `7,331,774,976` / `9,149,874,176` | — |
| PyTorch first-solve allocated/reserved peak | `7,407,272,448` / `9,149,874,176` | — |
| resident packed assets | `4,076,863,488` | — |
| 其中 fine transfer | `1,811,939,328` | — |
| 其中 coarse assets | `2,264,924,160` | — |
| fused FGMRES workspace | `509,607,936` | — |
| coarse workspace | `42,483,712` | — |

raw `Y` 的 `1,811,939,328 B` 是逻辑省略量，不是当前 resident 显存；当前 resident packed assets 已包含对应的 fine transfer。first-solve 增量约 `0.46875 GiB`，与 lazy FGMRES arena 约 `0.475 GiB` 的量级高度吻合，因此首解峰值的主要新增来源是懒分配的 FGMRES arena，而非再次加载全部层级资产。

相关位置：

- setup/cache restore/seal：`examples/qcu/dev87/bench_strict_vs_quda.py:1837-2020`
- first solve 与显存采样：`examples/qcu/dev87/bench_strict_vs_quda.py:2067-2159`
- 显存字段汇总：`examples/qcu/dev87/bench_strict_vs_quda.py:2167-2198`
- strict 资产绑定及 raw-Y 省略：`pyqcu/solver/_quda_multigrid.py:46-123`
- lazy FGMRES arena：`cpp/cuda/qcu/src/apply_multigrid_strict.cu:1103-1230`
- persistent coarse arena：`cpp/cuda/qcu/src/apply_multigrid_strict.cu:1237-1302`

### 3. QUDA autotune=off smoke

命令：

```bash
QUDA_ENABLE_TUNING=0 \
QUDA_INSTALL=/root/PyQCU/data/quda-qio-install \
QUDA_PATH=/root/PyQCU/data/quda-qio-install \
LD_LIBRARY_PATH=/root/PyQCU/data/quda-qio-install/lib:$LD_LIBRARY_PATH \
python -B examples/qcu/dev87/bench_strict_vs_quda.py \
  --profile smoke --side quda --repeats 1 \
  --quda-nullvec-prefix /root/PyQCU/data/L16x32x32x48_nvec12_quda \
  --quda-nullvec-manifest /root/PyQCU/data/L16x32x32x48_nvec12_quda.conversion.json
```

退出码：`0`；外层耗时：`38.5983 s`；worker wall：`33.2597 s`。

| 项目 | 结果 |
|---|---:|
| setup | `5.58827 s` |
| 两次 warmup | `1.84964 s`、`1.79993 s` |
| steady | `1.78827 s` |
| iterations | `12` |
| 真残差 | `4.1102e-7` |

单次 smoke 的 steady 比值为 PyQCU/QUDA=`1.1545x`（QUDA autotune=off）。已有 autotune=on steady `1.05395 s` 的对照中，PyQCU/QUDA=`1.9588x`；既有 formal PyQCU median `3.09786 s` 对 autotune=on 约 `2.94x`，但样本波动明显，不能把单次 smoke 当作正式公平基线。

QUDA worker 日志中，`invertQuda Total time=12.260 s`，其中 compute=`6.879 s/2042 calls`、file I/O=`3.335 s/61 calls`、init=`0.325 s/887 calls`、preamble=`0.340 s/160 calls`。这是 worker 内部聚合日志，不与 PyQCU 的单个 steady 字段直接等价，尤其不能忽略 I/O 和初始化协议差异。

### 4. 内存态 `torch.profiler`

覆盖 4 次 FGMRES 调用（2 warmup、1 steady、1 probe）；fused FGMRES CUDA 总计：`8.501 s`。

| kernel/事件 | 计数/总时间 | 归一化或占比 |
|---|---:|---:|
| `strict_hopping_parity_kernel` | `2392` 次，`6.284 s` | `598` 次/solve；约 fused CUDA 时间 `73.9%` |
| `strict_bicg_p_kernel` | `592` 次 | `148` 次/solve |
| `strict_bicg_s_kernel` | `592` 次 | `148` 次/solve |
| `strict_bicg_update_kernel` | `560` 次 | `140` 次/solve |
| short update | `32` 次 | `8` 次/solve |
| fine Wilson dslash（`wilson_dslash<float>`） | `392` 次，`173.55 ms` | `98` 次/solve；约 `43.4 ms/solve` |
| restrict | `44` 次，`150.287 ms` | — |
| prolong | `44` 次，`112.888 ms` | — |
| fine MATPC update | `192` 次，`27.211 ms` | — |

另计得 D2H memcpy=`3272` 次、`cudaStreamSynchronize`=`6496` 次（CPU self 约 `1.005 s`），`cudaMemcpyAsync` CPU self 约 `8.768 s`。这些主要指向 dot/dot-pair 的 host scalar round-trip，属于重要次项；由于 profiler 覆盖整个 worker，CPU attribution 不能与 CUDA kernel 时间简单相加。

相关位置：

- 粗 parity hopping kernel：`cpp/cuda/qcu/src/apply_multigrid_strict.cu:139-195`
- coarse `apply_matpc` 的两次 hopping：`cpp/cuda/qcu/src/apply_multigrid_strict.cu:2037-2049`
- coarse BiCGStab 的重复 `apply_matpc`：`cpp/cuda/qcu/src/apply_multigrid_strict.cu:2109-2185`
- fine MATPC：`cpp/cuda/qcu/src/apply_multigrid_strict.cu:1693-1713`
- dot/D2H/sync：`cpp/cuda/qcu/src/apply_multigrid_strict.cu:1459-1545`

## 当前最小优化候选

优先做 coarse `strict_hopping_parity_kernel` 的 launch block-size A/B：当前为 `128` 时，建立一个临时 `256` 线程 block 变体。该实验只改变线程分块/调度；在确认 kernel 是每线程独立输出且没有依赖 block 内归约后，不改变迭代次数、算术表达式和数据语义。预期仅该 kernel 类收益约 `5-20%`，折算整体 steady 约 `4-15%`，只是量级估计，必须实测确认寄存器、occupancy 和访存效果。

当前没有在本轮执行该 A/B，也没有取得 Nsight 的寄存器、occupancy、带宽或 launch replay 数据。最小验证命令（分别对 128 与 256 变体执行；本轮不执行）为：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B \
  examples/qcu/dev87/bench_strict_vs_quda.py \
  --profile smoke --side pyqcu --cache-expect hit --repeats 1
```

每个变体至少记录 `steady`、first-solve peak、最终真残差和 coarse iteration/kernel count；若残差或计数改变，应立即放弃该变体。若 256 导致寄存器溢出或 occupancy 下降，保留 128。host scalar/pinned-buffer 复用是第二候选，但预期收益低于粗层 hopping，且应单独验证 D2H 次数和同步语义。

## 尚未取得的数据与公平性限制

- 没有本轮重新跑的 QUDA autotune=on 成对基线；现有 `1.05395 s` 只是已有记录，不能视为本轮同条件复测。
- 没有正式多重复数、置信区间、固定 GPU 时钟/独占 GPU 条件下的 PyQCU/QUDA 统计结果；因此不能据此宣称最终性能优于或劣于 QUDA。
- 没有 Nsight Compute 的 occupancy、寄存器、L2/DRAM 带宽、实际 launch 配置和 kernel roofline 数据。
- 没有细分 setup/export/bind/seal/first-solve 每个子阶段的 kernel 时间；已有显存峰值覆盖 setup 与 first solve，但不能替代逐阶段时间线。
- 没有多 rank/MPI、不同 lattice size、不同 nvec 或不同 solver tolerance 的泛化数据。
- `torch.profiler` 有自身开销，且 4 次调用为聚合观察；其 kernel 计数适合定位热点，不应直接当作无 profiler 的生产耗时。

## 本轮状态

- 新增本报告文件；未修改 PyQCU 核心实现。
- 未删除文件、未提交、未打 tag、未运行新的长任务。
