---
name: qcu-single-tests
description: examples/qcu 与 examples/quda 的 QCU/QUDA 单功能测试、PyTorch 参考校验和布局转换约定。
---

# QCU/QUDA 单功能测试

## 适用范围

用于维护 `examples/qcu/single_qcu_*.py` 与 `examples/quda/test_*.py`。每个文件只覆盖一个接口或
一个紧密的接口族，输入使用小格点和固定随机种子，先做纯 PyTorch 参考，再按环境选择 CUDA/QUDA
扩展运行。缺少 CUDA、`libqcu.so`、pyquda 或 QIO 资产时必须明确 skip，不能伪造通过。

## QCU 生命周期

控制张量固定为 `params=int32[58]`、`argv=real[7]`、`set_ptrs=int64[100]`。每次调用严格执行：

```text
applyInitQcu(set_ptrs, params, argv)
operation(..., set_ptrs, params)
params[_SET_INDEX_] += 1
applyEndQcu(set_ptrs, params)
```

`params[_SET_PLAN_]` 使用 -2 Laplacian、-1 Gauss、0 Wilson dslash、1 求解器、2 Clover；QCU
数据为 `[2,3,3,4,X,Y,Z,T/2]`、`[2,4,3,X,Y,Z,T/2]`，纯 PyTorch 参考先经
`tools.poooxyzt2oooxyzt` 还原到 `[... ,X,Y,Z,T]`。

## QUDA 排布

PyQUDA/QDP 规范场为 `[4,T,Z,Y,X,3,3]`，费米场为 `[2,T,Z,Y,X/2,4,3]`，棋盘奇偶按
`(x+y+z+t)%2`，与 PyQCU 的 t 压缩布局不同。所有对照脚本必须通过
`examples/quda/common.py` 的显式往返转换，禁止直接 reshape 冒充转换。

## 验收

```bash
python examples/qcu/single_qcu_api.py
python examples/qcu/single_qcu_wilson_dslash.py
pytest examples/quda
git diff --check
```

设置 `QCU_STRICT_NUMERIC=1` 才把跨架构数值偏差升级为失败；默认运行仍记录相对误差，便于诊断
不同 GPU/精度下的归一化差异。新增接口或独立项目完成后，必须依照 `examples/AGENTS.md` 整理
测试入口并更新本技能的接口表。
