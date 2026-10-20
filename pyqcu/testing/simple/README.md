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

QUDA 入口自动兼容 `pyquda_utils` 和当前机器上的旧 `PyQuda-0.3.2`
（`pyquda.core` + `cupy`）两套原生 API。旧 API 的 BiStabCG 使用
`QudaInvertParam.iter/secs/gflops/true_res`；Multigrid 默认 `n_vec=24`
并把请求值整列写回 `QudaMultigridParam`。旧 API 不提供可捕获的逐迭代
日志接口时，`residual_history` 保持为空并注明来源。

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
