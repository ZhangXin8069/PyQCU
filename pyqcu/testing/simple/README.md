# PyQCU / QUDA 最小对照脚本

本目录只保留可单独运行的转换、检查和求解示例；默认数据目录为仓库的
`data/`，可用 `PYQCU_REFERENCE_LIME` 指定仓库外参考组态。QUDA gauge
参考文件为：

```text
data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime
```

## 格式与排布

| 对象 | QCU / PyQCU | QUDA / PyQDP |
|---|---|---|
| gauge | HDF5 `[3,3,4,x,y,z,t]` | `.lime` 或 `.npy` `[dir,t,z,y,x,3,3]` |
| fermion | HDF5 `[spin,color,x,y,z,t]` | `.npy` `[q,t,z,y,x/2,spin,color]` |
| clover | HDF5 `[spin,color,spin,color,x,y,z,t]` | `.npz/.npy` packed `[q,t,z,y,x/2,2,36]` |

QUDA 的 fermion 奇偶轴沿 `x` 压缩，`q=(x+y+z+t)%2`；QCU 的 parity 张量则沿
时间轴压缩。两者必须走本目录的转换函数，不能直接 `reshape`。QCU 的
`argv[_MASS_]` 用于计算 `kappa=1/(2*m+8)`，`argv[_CSW_]` 单独保存
Clover 系数。PyQCU/QCU 与 PyQUDA/QUDA 都使用 `-kappa*csw/8` 的 onsite
约定。CLI 未给 `--csw` 时，QCU/QUDA 示例都会优先读取 LIME 的
`clovCoeff`，`mass` 未显式给出时也会使用 record XML 中的 `Mass`；QCU HDF5
输入则读取同名 HDF5 attrs。上表
参考组态解析为 `mass=-0.277, csw=1.160920226`。若显式覆盖 `csw` 为不同值，
报告会保留 `gauge_csw` 并把 `comparison_valid` 标为 `False`。

QUDA/PyQUDA 求解脚本会在导入 PyQUDA 前按 GPU compute capability 选择并预载
`data/quda-sm70-install` 或 `data/quda-sm60-install` 下的 `lib/libquda.so`
（若存在）；显式 `QUDA_INSTALL` 优先。这避免旧 RUNPATH 或
`LD_LIBRARY_PATH` 中的 sm60 库在 V100 上被误加载。`solve_case.py` /
`compare_three_way.py` 的 `--quda-variant auto|sm60|sm70|default` 可覆盖。
simple 对照把 QUDA 设为 `QUDA_KAPPA_NORMALIZATION`，与 PyQCU 的
`M=I-kappa D` 使用同一方程归一化；PyQUDA 默认 asymmetric normalization
会让解额外相差 `1/(2*kappa)=m+4`。

新示例优先使用 `pyqcu.cuda.QcuRuntime`：

```python
from pyqcu.cuda import QcuRuntime

runtime = QcuRuntime.from_lattice(
    (8, 8, 16, 16), mass=-0.277, csw=1.160920226, plan=2)
runtime.call("applyCloverQcu", clover, gauge)
```

`QcuRuntime` 自动选择空闲 `set_ptrs` 槽位，并按
`applyInitQcu -> operation -> applyEndQcu -> _SET_INDEX_ += 1` 清理；普通
scratch 仅使用 `set_ptrs[0:30]`，保留 coarse 槽位。求解器需要读取
`getCloverBistabCgIterationsQcu` 时使用 `runtime.operation(...)` 上下文。

## 常用命令

```bash
python quda_gauge2qcu_gauge.py --dry-run
python quda_gauge2qcu_gauge.py --output qcu_gauge_24x24x24x64_c64.h5
python qcu_gauge2quda_gauge.py --input qcu_gauge_24x24x24x64_c64.h5
python check_gauge.py --input qcu_gauge_24x24x24x64_c64.h5
python give_gauss_gauge.py --backend torch --lat 8 8 16 16
python qcu_bistabcg.py --gauge "$HOME/windows/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime" \
  --lat 24 24 24 64 --mass -0.277
python quda_bistabcg.py --gauge "$HOME/windows/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime" \
  --lat 24 24 24 64 --mass -0.277
python compare_bistabcg.py \
  --gauge "$HOME/windows/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime" \
  --lat 24 24 24 64 --qcu-device cuda:0
```

`qcu_gauge2quda_gauge.py` 可直接写单文件 SciDAC/QIO `.lime`（带真实 CRC
checksum）；它不实现 MPI part-file，也不替代带 QIO 的 QUDA 生产写入器。
`qcu_*2quda_*` 默认写 QDP NumPy 文件，可直接交给 PyQUDA/QUDA 的 host-field 转换层。`QudaMultigridParam`
的数组属性返回的是副本，修改 `n_vec`、`geo_block_size` 等必须整列写回；QCU
则通过 `params[17:51]`、`argv[3:7]` 和 `set_ptrs[30+4*level]` 传递各层参数。

求解脚本在缺少 CUDA、`libqcu.so` 或 `pyquda` 时会明确输出 `SKIP`，不会把
纯 Python 参考伪装成后端求解。`qcu_*` 与 `quda_*` 也应在不同进程中运行，
避免同时加载 `libqcu.so` 与 `libquda.so`。`compare_bistabcg.py` 会自动以两个
子进程运行、生成同一 RHS，并校验两侧 mass、csw、lattice、真实收敛状态和
归一化解差；每次运行使用独立报告名，避免旧 `PASS` 文件污染本次结果。
合法 `csw=0` 在 QUDA 侧切换到 Wilson Dirac，而不是调用会断言失败的 Clover API。
任一后端崩溃、缺失、返回 `SKIP` 或非零时，聚合结果固定为 `FAIL`；安装的
QUDA kernel 架构与当前 GPU 不匹配时也必须修复 QUDA 构建，不能降级为跳过。
固定精度 QUDA 若开启 Clover reconstruct，遇到严格零 Clover 项（例如单位 gauge）
会报 `max_element appears unset`；验证 Clover 路径应使用非退化规范场。

QUDA 入口自动兼容 `pyquda_utils` 和当前机器上的旧 `PyQuda-0.3.2`
（`pyquda.core` + `cupy`）两套原生 API。旧 API 的 BiStabCG 使用
`QudaInvertParam.iter/secs/gflops/true_res`；Multigrid 默认 `n_vec=24`
并把请求值整列写回 `QudaMultigridParam`。旧 API 不提供可捕获的逐迭代
日志接口时，`residual_history` 保持为空并注明来源。

## KUPL CPU 简易验证

`kupl_bistabcg.py` 和 `kupl_multigrid.py` 通过真实 KUPL `ctypes` ABI 运行。
输入仍为 full QCU `[spin,color,x,y,z,t]`，脚本内部显式转换为 KUPL 的
parity-compressed canonical 布局。求解后使用独立 NumPy Wilson-Clover
重构计算相对真残差，不依赖后端自报值。两个对照脚本分别以 QUDA
BiStabCG 为独立参考：

```bash
PYTHONPATH=. KUPL_OMP_THREADS=16 \
  python pyqcu/testing/simple/compare_kupl_bistabcg.py \
  --lat 8 8 16 16 --mass 0.17 --csw 0.8 --kupl-threads 16

PYTHONPATH=. KUPL_OMP_THREADS=2 \
  python pyqcu/testing/simple/compare_kupl_multigrid.py \
  --lat 4 4 4 4 --mass 0.17 --csw 0.8 --levels 2 \
  --dof 12 2 --mg-grid 2 2 2 2 --kupl-threads 2
```

两级 MG 脚本先用 full-site matvec 构建真正的 33-stencil Galerkin 粗算子，
调用 `verifyCloverMultigridQcu` 校验 R/P 与 Galerkin 关系，再与 QUDA 解比较。
`benchmark_kupl.py` 重复运行 C++ benchmark 并输出中位数/MAD；它分别记录
Clover 两 parity setup、Wilson dslash、Clover dslash 和 Clover Schur dslash：

```bash
PYTHONPATH=. python pyqcu/testing/simple/benchmark_kupl.py \
  --lat 16 16 16 32 --iterations 10 --repeats 5 --threads 16
```

大格 `KUPL_DISABLE_BLOCK_DSLASH=1` 可回退到原 scalar site kernel，用于 A/B
回归；默认仅在 local `half_sites >= 65536` 时启用批量 SIMD 内点/周期面内核。

KUPL `kupl_multigrid.py --restart 0` 关闭 coarse hierarchy，选择 optimized
parity-Schur BiCGStab。在 `24³×64`、`mass=-0.277`、`csw=1.160920226` 上，
它比 `restart=5` 的 legacy coarse correction 快约 `2.8–3.0×`（19 对 47
次迭代）；当前 E12 粗空间在该组态上不产生净收益。

`24³×64` 同窗强扩展（16 个物理核，3–5 次独立冷 solve 的中位数）：
1 rank×4 thr `19.229 s`，1 rank×16 thr `9.397 s`，2×4 `11.490 s`，
4×4（`1×4×1×1`）`7.842 s`，**8×2（`1×4×2×1`）`7.532 s` 为最佳**，
16×1（`1×4×4×1`）`8.299 s`。因此生产默认推荐 8 ranks × 2 threads；
它与 4×4 只差约 `4%`，落在批间噪声内，二者都可部署，但 16 ranks 的
通信/边界占比更高，不再有收益。`solve_case.py` 的
`--kupl-ranks 8 --kupl-threads 2` 会自动得到 `1×4×2×1`。

非 2 幂 rank 数也已测：6×3 `8.249 s`、9×2 `10.640 s`、12×1
（`1×3×2×2`）`7.830 s`、18×1 `11.276 s`。12 ranks 只用 12 个核就接近
8×2，适合需要留核给其它任务的场景；9/18 ranks 的局部面太小，明显变慢。
默认网格启发式对 12 ranks 给出 `1×12×1×1`，实测 `8.040 s`；显式
`--kupl-grid 1 4 3 1` 为 `7.796 s`（快 `3.1%`），`1 3 2 2` 为 `7.830 s`。
12 ranks 建议显式指定网格。更均衡的 `1×2×2×2`（8 ranks）与切 t 的网格
都更慢，默认的 y→z→t→x 顺序对其余 rank 数与测量一致。
不要把 `OMP_PROC_BIND`/`OMP_PLACES` 加在 `--bind-to none` 之外：本实现
由大量短并行区组成，libgomp 每次建组都会重新绑定线程，实测 8×2 从
`7.2–7.4 s` 恶化到 `65–67 s`。`OMP_WAIT_POLICY=active` 也无收益。

`cpp/kupl/qcu/build-lto/` 提供 `-flto -fno-semantic-interposition` 构建，
同一 8×2 用例从 `7.556 s` 降到 `7.196 s`（快 `4.8%`），解逐位一致、
CTest 29/29。用 `--kupl-lib lto` 或 `PYQCU_KUPL_LIB=<build-lto>/libqcu.so`
选择；默认仍加载 `build/libqcu.so`。

多 rank 默认不再需要该开关：`kupl_residual.py` 会用纯 NumPy 在每个 rank
重构完整 Wilson-Clover 算子，通过 `sendrecv` 交换 ±μ 边界面和反向 link，
再 Allreduce 局部残差范数。`kupl_multigrid.py` 在 `mpi_size>1` 时自动走这条
分布式独立残差路径，并把各 rank 的 parity 块按 `np.unravel_index(rank, grid)`
拼回完整 `[spin,color,x,y,z,t]` 解；报告中的 `residual_seconds` 是各 rank
最大值。4-rank 与 1-rank 真残差分别为 `2.0052e-7` 和 `2.0022e-7`（8³×16），
解相对差 `1.18e-7`；`24³×64` 上 4/8/16 ranks 均为 `4.01e-7`，4-rank 与
16-rank 解相对差 `1.40e-7`。节点内跨 rank 已验证到 16 rank（1 thread/rank）；
跨节点/跨 NUMA 需要相同 MPI 实现在多节点复测，本机只有 1 个 NUMA node，
相关数字尚未覆盖。

注意 1 rank 的 `24³×64` 独立残差需要把整个 `24³×64` 算子放在单进程
NumPy 中重构，实测约 `308 s`，远超 solve 本身（约 `19 s`）。生产上应使用
≥4 ranks（`24³×64` 上分布式残差约 `2–3 s`），或明确
`--skip-independent-residual`；
`compare_three_way.py` 的计时样本默认跳过该步。

## 生产调用（推荐）

生产路径与性能对照已拆开，共享同一套输入准备和进程工具：

| 脚本 | 用途 |
|---|---|
| `prepare_case.py` | 从 LIME 切出全格/子体积，生成共享源，写 `case_manifest.json`（含 SHA256） |
| `solve_case.py` | 选择 `cuda` / `quda` / `kupl` 做一次冷启动生产求解，写解和独立真残差报告 |
| `compare_three_way.py` | 仅做性能测量：GPU 时钟预热、warmup、多次计时、中位数 |
| `bench_three_way.py` | 已弃用的兼容别名，转发到 `compare_three_way.py` |
| `_runner.py` | 子进程、设备映射、KUPL 命令构造等共享工具 |

CUDA 生产求解（V100 为 PCI 顺序 `cuda:2`）：

```bash
source ./env.sh
python pyqcu/testing/simple/prepare_case.py \
  --lat 24 24 24 64 --out-dir data/production_L24

python pyqcu/testing/simple/solve_case.py --backend cuda \
  --lat 24 24 24 64 --device cuda:2 --out-dir data/production_L24
```

KUPL 生产求解（8 rank × 2 OpenMP 线程，1×4×2×1 网格；`--kupl-lib lto`
可选，约快 `5%`）：

```bash
python pyqcu/testing/simple/solve_case.py --backend kupl \
  --lat 24 24 24 64 --kupl-ranks 8 --kupl-threads 2 \
  --kupl-lib lto --out-dir data/production_L24
```

QUDA 生产求解：

```bash
python pyqcu/testing/simple/solve_case.py --backend quda \
  --lat 24 24 24 64 --device cuda:2 --out-dir data/production_L24
```

默认输出 `solution_<lat>_<backend>_c64.h5`、`report_<backend>.json` 和
`last_solve_<backend>.json`；`--output/--report` 可覆盖。生产脚本不做 GPU
时钟预热和重复计时，只跑一次；解与报告写完后返回 `0` 仅当状态为 `PASS`。
`--gauge/--source` 可直接复用已有文件，缺省时自动调用 `prepare_case`。
KUPL 生产默认计算分布式独立真残差；明确接受后端 fail-closed 检查时可用
`--skip-independent-residual` 省掉这部分时间。

三个后端应使用同一份 gauge/source：`prepare_case.py` 的 manifest 记录
SHA256。跨后端验证可直接比较 HDF5 解（同一 `[spin,color,x,y,z,t]` 布局）；
在 `8×8×16×16`、mass/csw 相同、`atol=1e-6` 时，CUDA 与 4-rank KUPL 解的
相对差为 `4.5e-7`，最大逐点差 `6.0e-6`；QCU 与 QUDA 的相对差为 `1.5e-6`。

## 三方性能对照

`compare_three_way.py` 串联 CUDA QCU、KUPL QCU 与 QUDA，以 1 次 warmup +
3 次计时汇总中位数、迭代数、真残差和相对 QUDA 倍率：

```bash
source ./env.sh
python pyqcu/testing/simple/compare_three_way.py \
  --lat 24 24 24 64 --device cuda:2 \
  --kupl-ranks 8 --kupl-threads 2 --kupl-lib lto
```

输出 `data/benchmark_three_way/{summary.json,summary.md}`，逐后端 JSON 也在
同一目录。`--lat` 小于参考 LIME 时自动切出子体积；接缝非周期，只用于
尺寸/并行扩展研究。CUDA/QUDA 的 `--warmup/--repeats` 为进程内重复求解，
QUDA 会显式关闭 init-guess/resident-solution，避免热启动污染样本。
脚本默认先做 3 s GPU 矩阵乘法预热（`--gpu-warmup-seconds`），避免 V100
空闲时钟（405 MHz）压低小体积 solve 的计时；小体积建议多次独立调用取中位数。
`--device cuda:N` 在未设置 `CUDA_VISIBLE_DEVICES` 时按 PCI 顺序映射物理卡。
计时模式默认让 KUPL 跳过独立残差（解仍受后端 fail-closed 门控），需要时用
生产脚本单独验证。

### GPU 架构选择（sm_60 / sm_70）

`data/libqcu-sm60/` 与 `data/libqcu-sm70/` 分别存放原生 sm_60、sm_70 构建。
`compare_three_way.py`、`solve_case.py` 用 `--libqcu auto`（默认）按
`nvidia-smi --query-gpu=compute_cap` 选择，并把对应目录前置到子进程的
`LD_LIBRARY_PATH`；`--libqcu sm60|sm70` 可强制 A/B，`default` 保留调用者
当前栈（通常是 `cpp/cuda/qcu` 的 sm_60+PTX 包）。因此 P100 用 sm_60、
V100 用 sm_70，不需要改 `env.sh`。直接运行 `qcu_bistabcg.py` 时没有这层
自动选择，需要手动 `LD_LIBRARY_PATH=<repo>/data/libqcu-sm70:$LD_LIBRARY_PATH`。
P100 的系统 PyTorch 2.10 不支持 sm_60，使用 `data/venv-p100/bin/python`
（torch 2.4.1+cu121）；V100 继续用系统 Python。强制把 sm_70 库跑在 P100 上
会按预期报 `no kernel image is available for execution on the device`。

## 性能记录

求解器默认在 `data/` 写 `performance_<backend>_<solver>.json`，字段只来自
各后端的原生接口或明确定义的估算模型：

- QUDA：`QudaInvertParam.iter/secs/gflops/true_res`；开启 native verbose
  输出后解析外层迭代残差，缺失时保留空历史而不是补值。
- QCU：BiStabCG 使用 `getCloverBistabCgIterationsQcu`；Multigrid 解析
  `QCU_LOG_DIR` 下的 `Iteration N: Residual`、`CONVERGENCE_HISTORY`、
  `Residual(norm2)` 和 `Total iterations`，分别输出逐外层、采样历史、
  V-cycle 与总迭代数；`PROF_SECTIONS` 记录为 `profile_ms`。
- 两者都记录 CUDA event 墙钟、每迭代耗时、显存分配/保留/峰值和设备剩余量。
  QCU 没有 native GFLOPS 时记录明确标注的 Clover-Wilson 算子估算；QUDA
  优先记录 `QudaInvertParam.gflops`。
- 非有限残差或相对真残差超过 `max(atol, precision floor)` 时报告 `FAIL`
  并返回非零退出码；csw 或配置不可比时报告 `WARN` 和
  `comparison_valid=false`。
