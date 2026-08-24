# PyQCU Bug Fix Report — 2026-08-24

无人值守 auto 会话（`.auto.2026-08-24-02-37-52.log` 全程留痕，断点恢复自
`.auto.2026-08-24-02-15-09.log`）：debug→optim 交替循环 + 组件全覆盖扫描。
基线 `56e6ee6 (dev84_2_2)` → 终态 `1025a90`，标签 bug31–bug36、test16、dev85、bug37。

## 修复概览

| 标签 | 层级 | 一句话根因 | 定量闭环 |
|------|------|-----------|---------|
| bug31 | Python 测试 | `torch.Tensor(<标量>)` 被新版 PyTorch 移除（×2 处）+ give_null_vecs 诊断降本 | TypeError 消除；MG 测试 9.9s→6.1s（−38%） |
| bug32 | dev84 工具脚本 | `cmd_multi` 对 argparse 默认值 `"1,2"` 执行 `int()` 必崩 | 默认参数即复现 → 双兼容解析后 exit=0 |
| bug33 | smear/_wuppertal | wards 含 t 维（8 邻居 hop）配 (1−6σ) 系数 → 常数场每步 ×(1+2σ) 发散 | U=I 不动点 dev 44.4→6.75e-08；白噪声范数比 12.0→0.030 |
| bug34 | smear/_wuppertal MPI | halo 第二次 Sendrecv recvtag 错用 rank_plus（grid=2 周期退化死锁）+ U 边界切片缺方向轴（7v6 维崩溃） | np=2/4 黄金判据 rel=5.1e-08/3.9e-08 |
| bug35 | tools/_multigrid | e46a4cf 重构遗失 `f_local[...]=_blk` 赋值 → stencil 静默全零 | 中格子双场景 hop_nn=2.2~2.6e-02；C9 等价性 1e-06；Galerkin 5.6e-07 |
| bug36 | tools/_matul | tilelang 0.1.7.post3 四处同名 emitter 缺失 `_legalize_to_buffer_region`（上游包内不一致） | 4096³ fp16 rel=3.6e-04；38.7 TFLOPS=cuBLAS 94% |
| test16 | 回归固化 | Wuppertal 三重不变量测试 `test_smear_wuppertal` + 反模式文档同步 | cpu/cuda 双端 PASS |

另：dev85 澄清 `verify_nullvecs` 块结构布局与 kappa 吸收语义文档；
bug37 修复 benchmark/conftest.py 收集期笔误崩溃。

## 方法论要点（供后续会话复用）

1. **worktree 时间线二分**定位静默回归（bug35：test14 全零→dev79 正常→dev84_1 引入），
   配合资产 mtime 与提交时间交叉验证（缓存 Aug 22 08:46 < 提交 Aug 23 00:35 ⇒ 生产资产无恙）。
2. **同口径双实现对比**优于单边正确性论证（C9：build_stencil_local vs build_stencil 全格参考，
   同 lonv 逐元素对比，rel ~1e-6 即等价铁证）。
3. **tools.norm() 返回 float**（内部已 .item()）——二次 `.item()` 会 AttributeError，
   本会话三踩后文档化至 AGENTS.md 反模式清单。
4. monkey-patch 热修第三方库需**全栈 traceback 定位真实抛点**（bug36 v1 只补一处未中，
   实际运行时走 sm70 架构回退路径的第四处同名类）。

## 遗留清单（非本会话可解）

| 项 | 阻塞点 |
|----|--------|
| NPU 路径回归 | 无昇腾硬件 |
| ~~with_data 参考 HDF5~~ **已重建** | C++ 后端即独立实现源；L16³ Wilson 组 + L8³ clover 组生成器入库（logs/session-2026-08-24/gen_*_ref.py），h5 因 gitignore 由其确定性再生；solver 双后端交叉 rel=8.6e-07 |
| tilelang fp32-gemm sm70 / 64 小 block 数值错 | tilelang 包内模板/fragment 布局问题（上游另案，fp16 主语义不受影响） |
| ~~clover 双实现 ~12% 约定分歧~~ **已澄清为实验伪影** | 对照基线 κ 错配（C++ argv MASS=0⇒内部 κ=0.125 vs 误用 Python κ=1.0）；正确口径下 C++ applyClovers ≡ Python make_clover+add_I，rel=9.5e-09 位级一致 |
| verify_nullvecs Galerkin 分支对非块布局的容错 | 已文档化规避（须传 10 维块结构），转换层属可选增强 |

—— 报告生成：opencode auto 循环（bug31–37 / test16 / dev85 全程实测留痕）
