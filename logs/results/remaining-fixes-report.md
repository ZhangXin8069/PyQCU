# PyQCU R2 剩余高优先级项修复报告
**Date**: 2026-07-28  
**Source**: /root/PyQCU/logs/review-2026-07-28-r2.md  

---

## 本轮修复 (11 items)

| # | 严重度 | 文件 | 修复内容 |
|---|--------|------|---------|
| 2.4 | 🟡 | `tools/_multigrid.py` | NPU restrict 路径添加交叉验证文档+测试 |
| 2.5 | 🟡 | `tools/_io.py` | scatter 添加大数据大小限制说明 |
| 3.1 | 🔵 | `tools/_multigrid.py` | QR 归一化添加精度安全理由注释 |
| 3.2 | 🔵 | `tools/_multigrid.py` | ortho_r/ortho_null_vecs 当 normalize=True 时跳过冗余 vdot 分母 |
| 4.3 | 🟢 | `tools/_define.py` | set_device 添加 verbose=False 参数 |
| L2 | 🟢 | `solver/_multigrid.py` | 删除 `np.Inf = np.inf` 死代码 |
| L3 | 🟢 | `dslash/_operator.py` | matvec_eo/oe MPI 条件简化为仅 grid_size 判断 |
| L5 | 🟢 | `smear/_stout.py` | "update U" 注释改为 "rebind local U" |
| 4.1 | 📝 | `dslash/_operator.py` | 添加 coarsening=2 Galerkin 投影假设文档 |
| 4.2 | 📝 | `define.h` | 添加 `_lat_4dim_` float-to-int 精度注释 |
| L4 | 📝 | `tools/_define.py` | give_grid_size 添加 sorted() 排序约定注释 |

## 验证结果: 10/10 PASSED

```
  PASS: check_su3 atol
  PASS: stout NaN guard
  PASS: stout nstep>1
  PASS: operator parity
  PASS: BiCGStab breakdown guard
  PASS: NPU stout parity
  PASS: vdot Barrier removal
  PASS: set_device verbose=False
  PASS: NPU restrict validation
  PASS: ortho_r vdot cache (safe matvec)
```

## 两轮修复总结

| 轮次 | 发现 | 修复 Bug | 优化 | 误报 | PASS | 文档 |
|------|------|---------|------|------|------|------|
| R1 | 74 | 12 | 1 | 0 | 14 | 4 |
| R2 | 28 | 8 | 1 | 3 | 4 | 4 |
| R2剩余 | — | 0 | 5 | 0 | — | 6 |
| **合计** | **102** | **20** | **7** | **3** | **18** | **14** |
