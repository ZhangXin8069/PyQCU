# test14 收尾结果报告（2026-08-15）

test14 —— **多线程版（一线程一卡）CUDA C++ MultiGrid 求解器**测试套件，
形式参考 test12/test13（单文件 main.py 子命令入口 + 版本化产物目录 v<ts>/ + 全部 h5py 持久化）。

## 本次工作汇总

1. **套件生成**：`logs/test14/`（main.py / run-local.sh / AGENTS.md），由 test13 演进（
   `MultiGpuMultigrid` 多线程被测对象、多线程 BiStabCG 基准、h5py-only 持久化、P100×2 +
   V100 单线程大格子设备分配）。
2. **粗算子构建加速**（核心改进，相对 test13）—— pyqcu/tools/_multigrid.py +
   pyqcu/cuda/_multi_gpu.py：
   - **nv_tol=1e-2**：null 向量 BiCGStab 容差 5e-5 → 1e-2。5e-5 在粗层大系统
     （16x16x16x32 lv2，196608 未知数）迭代爆炸（>34min 未完成）；1e-2 分钟级。
     小格子 8x8x8x16 质量等价（rel_diff=0）。
   - **批量 stencil 探测**（_probe_point_batch + _schur_matvec_batch +
     _stencil_matvec_batch）：固定 c_idx 一次批量全部 E 探针（torch einsum；
     单位向量 prolong 切片化、restrict 邻域块局部化）。8x8x8x16 lv1
     12288 probes 135.6s → 3.3s（21 倍）；16x16x16x32 lv1（196608 probes）
     ~36min → ~3min。
   - **批量 BiCGStab**（_bistabcg_batch）：null 向量 dof 个右端一次批量迭代
     （标量按批独立 + 复数安全除法），与逐场 solver.bistabcg 等价（err~1e-5）。
     16x16x16x32 lv2 从 40min+ → 83s。
   - 实测 **16x16x16x32 3L 完整构建 86s**（test13 时 1h+ 未完成），求解正确
     （consistency=True），speedup=0.75（大格子 MG coarse solve 开销，历史特性）。
   - 缓存 key 含 `_t{nv_tol}` 后缀，旧 5e-5 缓存自动失效重建。

## 运行结果（v202608152021）

| 阶段 | 结果 |
|---|---|
| verify | 全 PASS（一致性 P100×2 rel=0、独立问题、V100 3L、h5py 4 线程 IO） |
| clean | 8x8x8x16 2L：speedup=1.98（交叉计时，consistency=True） |
| bench | 最佳 8x8x8x16 2L speedup=1.94；V100 8x16x16x16 3L=1.51 |
| sweep | **16/16 ≥ gate=1.5**（test13 为 11/16），best=2.49（L3 r10 ct1e5 cmi15） |
| check | PASS（16/16 ≥ 1.5） |
| budget | 16G/32G 档全 OK |

加速比提升（test13 → test14）：sweep 通过率 11/16 → 16/16，best 2.15 → 2.49。

## 关键文件

- `logs/test14/main.py` — 测试代码（verify/clean/bench/sweep/check/budget/collect/mktable/plots）
- `logs/test14/run-local.sh` — 运行脚本（设备编排 + 版本目录）
- `logs/test14/AGENTS.md` — 复现与比对指南
- `pyqcu/tools/_multigrid.py` — _bistabcg_batch/_schur_matvec_batch/_stencil_matvec_batch/_probe_point_batch
- `pyqcu/cuda/_multi_gpu.py` — build_schur_levels 批量路径（batch_build/nv_tol）
