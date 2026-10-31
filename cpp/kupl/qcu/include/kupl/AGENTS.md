# AGENTS.md — cpp/kupl/qcu/include/kupl

KUPL 的公开声明与内联数值基础。保持该目录独立于 CUDA 源码头文件。

- `define.h` 固定 `params int32[58]`、`argv real[8]`、`set_ptrs int64[100]`
  及各计划/槽位常量；修改时必须同步 `pyqcu/cuda/define.py`、CUDA 头与 ABI 测试。
- `qcu.h` 是 C ABI 的唯一声明面；函数名和签名必须继续与 CUDA 兼容边界一致。
- `lattice_set.h` 只保留生命周期、注册表和门面声明；数值实现放在 `../src/`。
- canonical 布局、MPI halo、NUMA 与 OpenMP 语义以父目录 `../AGENTS.md` 为准；
  不引入 GPU 指针、CUDA stream 或 device 类型。
- 修改头文件后从 `cpp/kupl/qcu` 运行 `ctest --test-dir build --output-on-failure`；
  构建目录不存在时先按父规则构建，不把未执行测试计为通过。
