# PyQCU / QUDA 最小对照脚本

本目录只保留可单独运行的转换、检查和求解示例；默认数据目录为仓库的
`data/`。QUDA gauge 参考文件为：

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
`argv[_MASS_]` 用于计算 `kappa=1/(2*m+8)`，QUDA/PyQUDA 接口接收 `mass`；
QUDA 的 clover 还要显式传 `csw`，QCU 的 `applyCloversQcu` 固定使用其内置
约定。示例默认 `mass=0.05, csw=1.0`；上表 `.lime` 文件中的参考
`mass=-0.277, csw=1.160920226` 会被 QUDA 侧自动读取，但 QCU 不能匹配该
`csw`，因此 QCU 读取该 `.lime` 时只做 gauge 示例并把 `comparison_valid`
标为 `False`，不会把结果伪装成 Chroma 配置等价解。

## 常用命令

```bash
python quda_gauge2qcu_gauge.py --dry-run
python quda_gauge2qcu_gauge.py --output qcu_gauge_24x24x24x64_c64.h5
python qcu_gauge2quda_gauge.py --input qcu_gauge_24x24x24x64_c64.h5
python check_gauge.py --input qcu_gauge_24x24x24x64_c64.h5
python give_gauss_gauge.py --backend torch --lat 8 8 16 16
```

`qcu_gauge2quda_gauge.py` 可直接写单文件 SciDAC/QIO `.lime`（带真实 CRC
checksum）；它不实现 MPI part-file，也不替代带 QIO 的 QUDA 生产写入器。
`qcu_*2quda_*` 默认写 QDP NumPy 文件，可直接交给 PyQUDA/QUDA 的 host-field 转换层。`QudaMultigridParam`
的数组属性返回的是副本，修改 `n_vec`、`geo_block_size` 等必须整列写回；QCU
则通过 `params[17:51]`、`argv[3:7]` 和 `set_ptrs[30+4*level]` 传递各层参数。

求解脚本在缺少 CUDA、`libqcu.so` 或 `pyquda` 时会明确输出 `SKIP`，不会把
纯 Python 参考伪装成后端求解。`qcu_*` 与 `quda_*` 也应在不同进程中运行，
避免同时加载 `libqcu.so` 与 `libquda.so`。

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
- 非有限残差或达到 `max_iter` 仍未收敛时报告 `FAIL` 并返回非零退出码；
  csw 或配置不可比时报告 `WARN` 和 `comparison_valid=false`。
