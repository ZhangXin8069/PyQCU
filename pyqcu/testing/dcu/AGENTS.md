# AGENTS.md — pyqcu/testing/dcu

AMD DCU / ROCm (HIP) 测试。

## 测试文件

| 文件 | 测试内容 |
|---|---|
| `conftest.py` | 主 DCU 测试入口 |
| `conftest_1.py` | DCU 测试变体 1 |
| `conftest_2.py` | DCU 测试变体 2 |
| `test_install_selection.py` | 无 DCU 硬件的构建选路、install.sh 和 make-dcu.sh 契约 |

## 用法

```bash
python pyqcu/testing/dcu/test_install_selection.py
python pyqcu/testing/dcu/conftest.py
```

第一个入口不需要 DCU；`conftest*.py` 只允许在具有 ROCm/HIP PyTorch 和可见 DCU
的目标环境运行。

TileLang 是 DCU TileLang 测试的可选依赖。缺失时 `conftest_1.py`/`conftest_2.py`
必须 warning 后 SKIP 并以 0 退出，不能在导入阶段抛 traceback。
