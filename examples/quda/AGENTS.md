# QUDA 单功能测试约定

每个脚本只验证一个 QUDA 接口或求解器功能；测试必须提供可复现输入，并在可用时用 PyQCU 纯 PyTorch 算子复算相对残差。缺少 `pyquda`、CuPy 或 CUDA 设备时，测试应明确跳过。
