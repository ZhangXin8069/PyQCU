# P100 / dual-P100 environment notes

Scope: establish a reproducible cu118 execution path for the two physical
P100-16GB cards. The V100 at physical index 2 is always excluded by
`CUDA_VISIBLE_DEVICES=0,1`. No PyQCU source, C++ source, benchmark, or unrelated
agent file was modified.

## Environment build

The requested Torch build was installed into the allowed data directory:

```bash
pip install --no-cache-dir --target /root/PyQCU/data/p100-site \
  'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu118
pip install --target /root/PyQCU/data/p100-site psutil
```

The first install completed successfully as `torch-2.7.1+cu118`.
`psutil-7.2.2` was added because the dev87 CUDA tests use it in their backend
availability guard. `numpy`, `mpi4py`, and `h5py` remain usable from the system
site and were explicitly imported with the target directory first.

Use:

```bash
source examples/qcu/dev87/p100_env.sh
```

The script sources repository `env.sh`, prepends `data/p100-site` to
`PYTHONPATH`, sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, sets
`CUDA_VISIBLE_DEVICES=0,1`, and exports `QCU_STRICT_DEVICE=0` plus
`QCU_STRICT_DEVICE_COUNT=2`. It was syntax-checked and sourced successfully in
both Bash and Zsh.

Verified runtime:

```text
torch=2.7.1+cu118
torch.version.cuda=11.8
arch_list=sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_37, sm_90
numpy=2.1.2
mpi4py=4.1.1
h5py=3.12.1
```

Physical/visible device mapping:

```text
0  Tesla P100-PCIE-16GB  compute capability 6.0  16384 MiB
1  Tesla P100-PCIE-16GB  compute capability 6.0  16384 MiB
2  Tesla V100-SXM2-32GB  excluded
```

With `CUDA_VISIBLE_DEVICES=0,1`, Torch reports exactly two
`Tesla P100-PCIE-16GB` devices. The system Torch `2.10.0+cu128` has no sm_60
support, so it is not used through `p100_env.sh`.

## Single-P100 verification

Exact minimum test:

```bash
source examples/qcu/dev87/p100_env.sh
CUDA_VISIBLE_DEVICES=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q -rs -p no:cacheprovider \
  examples/qcu/dev87/test_quda_transfer_cuda.py::test_cuda_strict_solver_converges_with_bounded_krylov_arena
```

Observed on logical P100 0 and P100 1:

```text
device: Tesla P100-PCIE-16GB
capability: (6, 0)
result: 1 passed
iterations: 14
final true residual: 2.8319233024376445e-05
solve elapsed: 0.0417 s (GPU 0), 0.0521 s (GPU 1)
torch max allocated: 868864 bytes
torch max reserved: 2097152 bytes
```

Both cards load the rebuilt libqcu sm_60 kernel and execute the strict
V-cycle/complete bounded solve. Logs:

```text
data/p100-logs/p100_single_strict_vcycle.log
data/p100-logs/p100_single1_strict_vcycle.log
```

## Dual-P100 MPI verification

Each MPI rank gets a distinct card with:

```bash
mpirun --allow-run-as-root -np 2 bash -c '
  source /root/PyQCU/examples/qcu/dev87/p100_env.sh
  rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"
  export CUDA_VISIBLE_DEVICES="$rank"
  cd /root/PyQCU
  python3 -m pytest -q -rs -p no:cacheprovider \
    examples/qcu/dev87/test_strict_mpi_preflight.py
'
```

The per-rank mapping was independently observed as rank 0 -> P100 0 and rank 1
-> P100 1. The preflight suite completed successfully with `14 passed,
13 skipped` on both ranks. Log:

```text
data/p100-logs/p100_dual_device_mapping.log
data/p100-logs/p100_dual_preflight.log
```

A minimal end-to-end distributed strict solve was then run with global
`8x4x4x4`, process grid `[2,1,1,1]`, block size `[2,2,2,2]`, restart 4, and
60 outer-iteration budget:

```bash
mpirun --allow-run-as-root -np 2 bash -c '
  source /root/PyQCU/examples/qcu/dev87/p100_env.sh
  rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"
  export CUDA_VISIBLE_DEVICES="$rank"
  cd /root/PyQCU
  python3 examples/qcu/dev87/strict_mpi_solve_probe.py \
    --shape 8 4 4 4 --grid 2 1 1 1 \
    --restart 4 --max-iter 60 --tol 1.0e-6
'
```

Result: both ranks reported `converged=True`, 31 outer iterations, C++ reported
residual `4.084693864569999e-05`, and the independent Python true-relative
residual was `7.828087760101502e-07`. The per-rank Torch peak was
`13,013,504` allocated bytes and `27,262,976` reserved bytes. No OOM or
16GB-limit trigger occurred. Log:
`data/p100-logs/p100_dual_strict_solve_probe_shape8_mem.log`.

An earlier `4x4x4x4` probe was rejected before CUDA because block-2 coarsening
produced coarse shape `(1,2,2,2)`, which violates the strict even-extent
geometry invariant. This is an invalid test configuration, not a P100 failure;
its log is retained as
`data/p100-logs/p100_dual_strict_solve_probe.log`.

## Status and matrix naming

`p100-dual` is available for the minimal distributed strict-MG path: both P100
devices load sm_60 kernels, MPI rank binding is isolated, and the small
distributed solve has an independent residual below `1e-6`. No blocker was
observed in the requested P100 path. The large `16x32x32x48` and c128 matrix
points were intentionally not run here and remain for the main-agent matrix.

Use these matrix device names:

```text
v100-single: physical device 2 only, CUDA_VISIBLE_DEVICES=2
p100-dual:   physical devices 0 and 1, CUDA_VISIBLE_DEVICES=0,1,
             one MPI rank per card through p100_env.sh
```

Machine-readable summary: `data/p100-logs/p100_summary.json`.

## 2026-09-18 最终代码在双 P100 上的复验

分布式 strict 全链路（分布式 Galerkin setup + fine/coarse halo +
distributed FGMRES）在本轮实现定稿后重新在双 P100 上跑过，均使用
`CUDA_VISIBLE_DEVICES=<rank>` 一 rank 一卡：

| 探针 | 结果 |
|---|---|
| `strict_mpi_setup_probe.py --shape 8 8 8 16 --grid 2 1 1 1 --levels 2 --mode colored` | passed；内部点 `0.0`，rank 边界点最大相对误差 `5.52e-07`（c64） |
| `strict_mpi_primitive_probe.py --shape 8 8 8 8 --grid 2 1 1 1 --dof 4 --parity 1` | full `1.54e-07`、compact `2.07e-08` |
| `strict_mpi_solve_probe.py --shape 8 8 8 16 --grid 2 1 1 1 --restart 20 --max-iter 200 --galerkin-mode colored` | converged，28 次外层迭代，独立真相对残差 `6.59e-07` |

仍然存在的 P100 阻塞（与 strict 实现无关）：

1. 收集器 `bench_strict_vs_quda.py --device p100 --mpi-ranks 2` 在 PyQCU
   侧触发 CUDA illegal memory access，位置是**旧版** clover-gauge MPI halo
   （`cpp/cuda/qcu/include/lattice_clover_dslash.h` 的 `_make_mpi` /
   `pick_up_u_*`），不是本轮实现的 strict halo 路径；
2. QUDA 侧完全不可运行：`data/quda-double-install/lib/libquda.so` 只含
   `sm_70` cubin，P100（`sm_60`）无法加载。

因此矩阵里的 `p100-dual` 列只能用上面的独立探针结果支撑，收集器矩阵单元
仍按 blocked 记录，不用替代实现填充。
