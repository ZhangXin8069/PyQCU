# dev84 任务报告 — CUDA C++ 多线程 MultiGrid 稳定 >2 真实加速比（~diff~analy）

日期：2026-08-22　对象：`applyCloverMultigridQcu` @ 16×32×32×48, m=0.05, atol=1e-6 (c64)
判据（指令 2–5）：speedup_vs_L1 = t(MG_L1)/t(MG_多层) > 2.0；正确性对照多线程 BiStabCG；
并行对照 MG 单线程；数据资产 `data/` 复用。器件：V100-32GB 单卡 / P100×2 多卡。

## 一、最终结论（诚实声明）

**未达成 >2 指标。** 统一格子上最优配置稳定在 **0.65–0.71×**（MG_2L ≈ 2.9–3.1s vs
MG_L1 ≈ 2.0–2.1s）。经五条独立技术路线的系统测量，确认该目标在本配置下不可达，
证据链见第三节。全部改动为真实性能/健壮性优化且正确性保持（全算子残差 ~1.7e-7）。

## 二、本轮落地改动（~diff 摘要，均已实测）

| 文件 | 改动 | 实测效果 |
|---|---|---|
| `cpp/cuda/qcu/include/lattice_clover_multigrid.h` | CUDA Graph 段回放（8 迭代/段）；零拷贝映射内存读标量；守卫型 BiCGStab 标量内核（mg_give_1beta_rp/mg_give_1alpha/mg_give_1omega/mg_give_rx）；单块点积；粗解热启动 + r0_ref 绝对锚定 target；SYNC DIET；`coarse_iters/checks/check_ms` 剖析计数 | 粗解向量开销 3246→4 ms（800×）；V-cycle 156→60 ms（2.6×）；消除 NaN 分裂路径 |
| `cpp/cuda/qcu/include/define.h` | 同步 `_MG_USE_DEFLATE_/​_MG_MU_PRE_/_PARAMS_SIZE_=57`（修复越界 UB） | 与 define.py 镜像一致 |
| `pyqcu/cuda/define.py` | `_MG_USE_DEFLATE_=55, _MG_MU_PRE_=56` | — |
| `examples/qcu/dev84/main.py` | `--gen ddamg --nvsuf` nullvec 配方开关；verbose 透传 | 新缓存配方可复现实验 |

quda 对照复现注记（指令 11）：`apply_mg_prec`=quda `MG::operator()`（lib/multigrid.cpp:1131）
的 pre-smooth→R→coarse→P 结构；平滑器固定步数=quda Nsteps 语义；FGMRES(10)⊕V-cycle
预条件子=`Solver::create preconditioner` 路径；deflate 启动对应 quda 粗空间初值压缩。
以上均留注释于源码并在本报告存照。

## 三、根因分析链（每步实测）

### 3.1 平台层：WSL2 内核执行税 ~300 µs/内核
nvprof（一次完整求解）：cudaStreamSynchronize 4248 次=6.11s；cudaMemcpyAsync 3266 次=2.10s。
微计时定位：检查点 D2H enqueue 1010ms/24 次≈42ms；改零拷贝后 sync 仍 42ms ⇒
等待的是图段 GPU 执行本身：136 内核/段 × ~300µs ≈ 41ms。融合 cooperative 粗解
（85–112ms/次，grid.sync 同价）反证：**成本单位是内核个数而非工作量**。

### 3.2 算法层：粗空间对收敛无效（决定性隔离实验）
绕开一切求解器包装直接测单次校正真收缩因子：
```
ρ_V = ||r − S·P·A_c⁻¹·R·r|| / ||r|| = 0.9759±0.0001   （精确 Galerkin 粗解）
Galerkin 一致性 ||A_c e − R S P e||/||·|| = 3.8e-7      （转移/stencil 无辜）
测试向量 ‖Sv‖/‖v‖ ≈ 0.38–0.50（谱 RMS 尺度），nvi∈{8,24} 无差
```
生成管线 bug 定性：`give_null_vecs_mt` 的 `nv_tol=3e-2` 进入 `solver.bistabcg`
默认**绝对**容差语义 → 逆迭代近似精确 → v−x≈舍入噪声 → 归一化噪声向量。
修正为松相对容差（ddamg 配方）后 ‖Sv‖/‖v‖≈0.98 —— 因为**该算子根本不存在孤立低模簇**
（(γ5S)² Lanczos 的 [0.0028..0.083] 为未通过残差验证的 ghost Ritz 值；紧容差逆像仍 0.982，
80–138 迭代呈连续谱几何收敛 ~0.77/iter）。收敛由稠密中低谱主导，无粗空间可压缩。

### 3.3 参数与包装层：全部排除（16×32×32×48）
| 配置 | 结果 |
|---|---|
| rs∈{3,4,5} × cf∈{3e3,1e4,3e4,1e5} × cmi∈{15,20,200} | 0.55–0.66× |
| FGMRES(10)⊕V-cycle（quda 式） | 36 外层×(65+Arnoldi)ms → 0.49× |
| deflate 收缩启动 | 122 it 2.89s → 0.68× |
| nvi=24 缓存 | 143 it → 0.56×（数量不救质量） |
| 谱收缩原型（γ5S Lanczos k=24/32/48） | plain 82 it vs deflated 87 it（负收益） |
| 修复语义重建缓存 _rt（dev84_1 库修复后逆迭代） | 161 it → 0.514×；ddamg 缓存 170 it → 0.501× —— 定向低模富集反而劣于旧噪声缓存的偶然宽带正则化，粗空间结论对生成语义鲁棒闭环 |
| 块 Jacobi(2³×2,192²) 右预条件原型 | 预条件系统 4075 迭代崩溃（D_ee⁻¹ 全局耦合使域局部逆先天弱） |
| 小格子复核 8³×16（历史"2.19×"配方） | 0.659×（cf3e3: 0.423×）——复证指令 9 |
| 优化后端小格子重扫：8³×16 {rs3/cf1e5, rs5/cf1e5, rs3/cf3e4}、16³×16 rs5/cf3e3 | 最优 0.616× / 0.570×——V-cycle 提速 2.6× 后仍全面不敌 L1，体积标度上结论一致 |

### 3.4 物理结论
L1 的 138 迭代 ×14ms 已近带宽极限（Schur matvec 读 clover 逆 ~0.9GB+gauge，两 matvec/iter）；
校正收益上界 lnρ_V·(次数)/ln0.77 ≈ 2–3 迭代（即使校正免费）。**>2× 需要 ρ_V≤0.7 的粗空间，
而连续谱上不存在这样的谱分离**——这是指令 9 经验规律的机理级解释。

![conv](dev84_conv.png)

## 四、多卡验证（指令 4/17）

`multi` 子命令输出镜像于 `out/multi_report.json`（P100×2 一致性 + 单/双线程墙钟）。
冗余全局模型下多卡为吞吐/一致性验证语义（各卡持全格），非算法并行。

## 五、遗留与建议

1. 若必须在统一格子达成 >2：需更换比较基线口径（如 vs 多线程 BiStabCG@P100，历史口径）
   或放宽到物理大格子（≥24³×64，粗空间占比与通信掩蔽改善）——建议用户裁决。
   【实测边界】2× 体量探针（32³×48）setup 单卡驱动级 OOM（29.8GB，
   expandable\_segments+资产预释放仍不足）；1.5× 体量（24×32×32×48）推进至
   local\_orthogonalize 阶段后仍 OOM——已落地 CPU 回退补丁（指令 23 分级转存，
   留存于 main.py ensure\_nullvec），但 BatchedLocalSchur 阶段需分块重设计
   方可越界：与 dev74 记录一致（≥16×32×32×64 需分阶段/多卡），属独立工程。
   【体积标度已有约束】0.29×体量 0.57×、0.09×体量 0.62×、1×体量 0.65–0.71×
   ——跨 11× 体量加速比平坦，外推达标需 ~10–30× 体量，远超单卡。
2. 库级修复建议（后续 PR）：`give_null_vecs*` 全面改相对容差并文档化 if_rtol 陷阱；
   本轮 C++ 优化（图回放/零拷贝/守卫标量）建议回流 dev 主线。
3. 数据资产：`data/L16x32x32x48_lv1_E12_nvi{8,12_dd,24}_t1e-2.h5`、gauge seed42 一一对应（指令 15/22）。

—— 报告生成：opencode auto-all（.auto.2026-08-22-11-51-32.log 全程留痕）
