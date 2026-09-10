# QUDA 单功能测试约定

每个脚本只验证一个 QUDA 接口或求解器功能；测试必须提供可复现输入，并在可用时用 PyQCU 纯 PyTorch 算子复算相对残差。缺少 `pyquda`、CuPy 或 CUDA 设备时，测试应明确跳过。

当前维护入口为 `test_gauss_gauge.py`、`test_wilson_dslash.py`、`test_clover_dslash.py`、
`test_wilson_bistabcg.py`、`test_wilson_multigrid.py`、`test_clover_bistabcg.py` 和
`test_clover_multigrid.py`。公共 `common.py` 固定 QDP `[d,t,z,y,x,3,3]`、`[q,t,z,y,x/2,s,c]`
与 PyQCU `[... ,x,y,z,t]` 的往返转换；每个入口先运行纯 PyTorch 参考，只有显式具备 pyquda/CUDA
时才扩展到 QUDA 调用，缺依赖必须输出可解释的 skip。

QUDA 与 QCU 不得在同一进程混用；实际 QUDA 调用仅在显式设置
`RUN_QUDA_TESTS=1 QUDA_UNSAFE_INPROCESS=1` 时启用，正式对照应使用 dev87 的双进程运行器。
