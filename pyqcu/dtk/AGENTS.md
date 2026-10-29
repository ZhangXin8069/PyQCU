# AGENTS.md — pyqcu.dtk

DCU/ROCm（AMD GPU）运行时包占位。暂无独立 Python 实现。

仅含空 `PASS` 文件作为目录占位。

当前可用 DCU 构建路径位于 `cpp/cuda/qcu/make-dcu.sh` 和
`cpp/cuda/qcu/CMakeLists-dcu.txt`，由 `PYQCU_BUILD_BACKEND=dcu` 选择，产物仍通过
`pyqcu.cuda.qcu` 暴露。不要从这个占位包导入或伪造 `backend="dcu"`。
