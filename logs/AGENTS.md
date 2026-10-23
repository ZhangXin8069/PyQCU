# AGENTS.md — logs

`logs/` 只保存当前任务尚未形成 tag 归档的日志、图表和阶段报告。阶段结果先进入
待 tag 提交，tag 创建后再用独立清理提交删除工作区报告副本。

## 生命周期

1. 当前任务把 `.json/.log/.tsv/.txt/.stdout`、阶段报告和图表写入 `logs/**`；
2. 验证完成后提交报告并创建对应 tag；
3. 在后续清理提交中删除报告、图表及其它可由 tag 恢复的阶段产物；
4. 后续读取使用 `git show <tag>:<path>`，不恢复永久副本。

`logs/<tag>/**` 不作为报告的长期存放位置。若报告已经进入 tag，工作区仍存在同名副本
即属于格式冲突；若尚未进入 tag，则必须先提交并打 tag，再执行删除。

## 目录职责

- `logs/data/`：从 `data/**` 提升出的当前运行摘要；
- `logs/clover_multigrid.log`：现役 C++ 求解器收敛输出；
- `data/logs/<tag>/`：tag 关联的数据、张量或 cache 实体；
- `data/logs/nullvec_cache/`：共享 null-vector/粗算子缓存实体；
- 测试与脚本实体位于 `pyqcu/testing/**`，不放入 `logs/`。

## 日志索引

### 2026-10-20-big-file-purge - final - 2026-10-20 00:55:24 +0800
- 主任务: Git 历史大文件永久删除与仓库瘦身
- 日志文件: logs/v2026102000.txt
- 文件大小: 5512 bytes
- 触发原因: 任务完成
- 关键内容: 删除 8 个远程历史大文件路径、重写并强制同步 2 个分支和 275 个标签，完成本地与远端可达历史验证。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
### 2026-10-20-tag-dev96 - final - 2026-10-20 01:01:47 +0800
- 主任务: 创建本地 dev96 附注标签
- 日志文件: logs/v2026102001.txt
- 文件大小: 3442 bytes
- 触发原因: 任务完成
- 关键内容: 基于 dev95 创建本地 dev96，记录 Git 历史瘦身；peeled commit 与 HEAD 一致，标签链自检通过，未推送远程。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
### 2026-10-20-form-agents-final - final - 2026-10-20 01:16:53 +0800
- 主任务: PyQCU 2026-10-20 form 根规则托管块同步与本地技能补强
- 日志文件: logs/v2026102001-2.txt
- 文件大小: 7359 bytes
- 触发原因: 任务完成
- 关键内容: 完成 AGENTS form 唯一标记块同步、本地技能例外边界说明、任务文档与全部指定验证，终态未提交。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
### tag-dev97-20261020 - final - 2026-10-20 01:32:20 +0800
- 主任务: 创建 PyQCU dev97 本地附注标签
- 日志文件: logs/v2026102001-3.txt
- 文件大小: 3261 bytes
- 触发原因: 任务完成
- 关键内容: PyQCU dev97 已创建并与 HEAD 对齐；历史链旧问题保留，标签未推送
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
历史读取示例：

```bash
git show dev73:logs/dev73/review-2026-07-28.md
git show dev94:logs/stab33/report_multigrid_python_20260828.tex
```
## dev98-csw-kupl-final - final - 2026-10-20 05:22:40 +0800
- 主任务: PyQCU Csw、QcuRuntime/simple 与独立 KUPL MPI+OpenMP 后端
- 日志文件: logs/v2026102005.txt
- 文件大小: 8205 bytes
- 触发原因: 任务完成
- 关键内容: Csw/接口/简单示例完成；KUPL 24/37 实现，高级入口 fail closed；验证与外部 QUDA 阻塞如实记录
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## quda-sm70-runtime-align - final - 2026-10-20 09:04:09 +0800
- 主任务: 修复 PyQUDA 加载 sm60 QUDA 导致 V100 no kernel image，并对齐 QCU/QUDA mass normalization
- 日志文件: logs/v2026102009-2.txt
- 文件大小: 5477 bytes
- 触发原因: 任务完成
- 关键内容: sm70 预载、env 选择、kappa normalization 与同源数值比较均修复并验证
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## kupl-qcu-structure - final - 2026-10-20 09:18:07 +0800
- 主任务: KUPL QCU 与 CUDA QCU 结构对齐重构及回归
- 日志文件: logs/v2026102009.txt
- 文件大小: 6743 bytes
- 触发原因: 任务完成
- 关键内容: KUPL QCU 分层重构、LatticeSet 门面、37 项 ABI 门禁与 Release/UBSAN 回归
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## kupl-cuda-alignment - final - 2026-10-20 11:51:08 +0800
- 主任务: KUPL QCU 与 CUDA QCU 对齐重构
- 日志文件: logs/v2026102011.txt
- 文件大小: 6565 bytes
- 触发原因: 任务完成
- 关键内容: 37 项 ABI、strict/legacy MG 语义、workspace 账本与回归完成；Release/UBSAN/CTest/pytest/form 全绿。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## 01a1574f-context - final - 2026-10-20 14:12:06 +0800
- 主任务: KUPL independent simple tests and performance optimization
- 日志文件: logs/v2026102014.txt
- 文件大小: 6288 bytes
- 触发原因: 任务完成
- 关键内容: KUPL simple/QUDA correctness plus adaptive SIMD and solver/MG optimization
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## 20261021-kupl-cuda-multigpu - final - 2026-10-21 00:54:57 +0800
- 主任务: KUPL Kunpeng NUMA 与 CUDA A100 多卡 debug/优化
- 日志文件: logs/v2026102100.txt
- 文件大小: 10499 bytes
- 触发原因: 任务完成
- 关键内容: 修复 KUPL 24 核线程契约、验证双 NUMA，优化 CUDA 纯 MG 路径并完成双 A100 与真实 LIME 回归
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## refer-git-rep - final - 2026-10-21 05:10:19 +0800
- 主任务: 整理 refer/git-rep 参考库结构
- 日志文件: logs/v2026102105.txt
- 文件大小: 5523 bytes
- 触发原因: 任务完成
- 关键内容: 参考库改为上游 submodule，34 份自产报告集中归档，form 审计通过
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## docs-progress-dev99-1640349d - final - 2026-10-21 06:31:38 +0800
- 主任务: archive-dev99-to-1640349d-summary
- 日志文件: logs/v2026102106.txt
- 文件大小: 2509 bytes
- 触发原因: 任务完成
- 关键内容: 完整进度总结已保存到 docs，并完成文档格式审计
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## tag-dev100 - final - 2026-10-21 06:36:52 +0800
- 主任务: create-dev100-local-tag
- 日志文件: logs/v2026102106-2.txt
- 文件大小: 3494 bytes
- 触发原因: 任务完成
- 关键内容: 创建并验证本地附注标签 dev100，未推送远程
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## stage1-production-mg-20261021 - stage - 2026-10-21 09:09:23 +0800
- 主任务: 比照 QUDA 迭代 debug/优化 KUPL QCU 与 CUDA QCU，MG 为主、BiCGStab 为辅
- 日志文件: logs/v202610210909.txt
- 文件大小: 10774 bytes
- 触发原因: 阶段成果
- 关键内容: 完成 CUDA/KUPL MG 生命周期、缓存、停止判据与 Schur fallback 修复；生产 24^3x64 c64 三方基线与 KUPL 1/2/4/8/16 线程扩展通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage2-cuda-d2d-mg-20261021 - stage - 2026-10-21 09:14:13 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610210914.txt
- 文件大小: 5027 bytes
- 触发原因: 阶段成果
- 关键内容: nvprof 定位整向量 D2D 拷贝热点；cudaMemcpyAsync 使 CUDA MG 降 20.6%、BiCGStab 降 26.7%，三方生产口径与回归通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage3-cuda-mg-rx-20261021 - stage - 2026-10-21 09:52:08 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610210952.txt
- 文件大小: 5553 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA mg_give_rx 合并访存优化；MG 后端降 21.7%，三方性能与回归通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage4-cuda-lazy-workspace-kupl-lane16-20261021 - stage - 2026-10-21 10:35:20 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610211035.txt
- 文件大小: 6495 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA MG lazy workspace 降 19.4%；KUPL Wilson lane16 微基准降 19.5%；三方回归通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage5-cuda-fused-clover-io-20261021 - stage - 2026-10-21 10:42:36 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610211042.txt
- 文件大小: 4913 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA Clover I/O 融合：MG 再降 6.0%，BiCGStab 再降 6.2%，三方回归通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage6-cuda-clover-subtract-fusion-20261021 - stage - 2026-10-21 10:47:42 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610211047.txt
- 文件大小: 4556 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA Clover 减法融合：MG 再降 1.9%，BiCGStab 再降 2.8%，粗修正与残差通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage7-cuda-fused-wilson-clover-20261021 - stage - 2026-10-21 10:59:34 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610211059.txt
- 文件大小: 5196 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA c64 Wilson+Clover 融合：MG 再降 4.5%，BiCGStab 再降 0.9%，strict 与粗修正通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage8-kupl-production-mpi-20261021 - stage - 2026-10-21 11:15:56 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610211115.txt
- 文件大小: 4635 bytes
- 触发原因: 阶段成果
- 关键内容: KUPL 生产 MPI 分片与强扩展：4x4 y 分解降至 5.439 s，效率 79.9%
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage9-kupl-schur-zero-cache-mpi-20261021 - stage - 2026-10-21 11:17:58 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610211117.txt
- 文件大小: 4319 bytes
- 触发原因: 阶段成果
- 关键内容: KUPL restart=0 零 coarse-cache 冷启动；4x4 MPI 中位 5.635 s、RSS 1.13 GB
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage10-final-verified-state-20261021 - stage - 2026-10-21 11:29:35 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU；MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610211129.txt
- 文件大小: 6473 bytes
- 触发原因: 阶段成果
- 关键内容: 最终复核：CUDA MG 0.1720 s、KUPL 4x4 5.649 s、CUDA BiCG 0.1872 s；测试与构建通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## mg-assets-analysis - final - 2026-10-21 17:21:14 +0800
- 主任务: 解释 QCU 完整 33-tensor 粗层资产
- 日志文件: logs/v2026102117.txt
- 文件大小: 6570 bytes
- 触发原因: 任务完成
- 关键内容: 从物理来源、三数组布局、完整缓存边界和 strict cache 区别说明 QCU 33-tensor 粗层资产；CPU 打包与缓存回归通过。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## mg-cache-comparison - final - 2026-10-21 17:32:32 +0800
- 主任务: 对比 33-tensor cache 与 Strict cache
- 日志文件: logs/v2026102117-2.txt
- 文件大小: 8067 bytes
- 触发原因: 任务完成
- 关键内容: 从粗算子数学表示、几何、张量内容、内存、构建、kernel、缓存完整性和运行期 ABI 对比两种 MG cache；33 项 Strict 与 4 项旧 cache 测试通过。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## skills-mg-cache-doc - final - 2026-10-21 18:01:44 +0800
- 主任务: 将 33-tensor cache 与 Strict cache 对比写入 skills
- 日志文件: logs/v2026102118.txt
- 文件大小: 4452 bytes
- 触发原因: 任务完成
- 关键内容: 在 pyqcu-solver 技能新增 MG cache 对比参考、触发入口与索引同步；27 项缓存回归和源码锚点检查通过，form audit 仅命中未修改的既有文档。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## stage12-cuda-nullvec-kupl-onsite-20261021 - stage - 2026-10-21 18:08:39 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主、BiCGStab 为辅
- 日志文件: logs/v202610211808.txt
- 文件大小: 10179 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA null-vector BiCGStab chunk 6 改善约 29%；KUPL onsite c64 生产 A/B 改善 7.2%，c128 微基准改善约 13.2% 但生产样本不足；完整冷 setup 与三方回归通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage13-cuda-gather-w8-kupl-tile512-20261021 - stage - 2026-10-21 19:25:37 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主、BiCGStab 为辅；并行派发 CUDA/KUPL 子 agent
- 日志文件: logs/v202610211925.txt
- 文件大小: 12573 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA stencil gather+W8 使 cold setup 降 46.85%；KUPL coarse tile512 等使 4-rank MG 降 68.18%；三方报告、CTest 29/29 与 strict tier1 全绿
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage16-paired-reduction-20261021 - stage - 2026-10-21 21:07:57 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主、BiCGStab 为辅；并行执行 CUDA/KUPL 子 agent
- 日志文件: logs/v202610212107.txt
- 文件大小: 11229 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA Clover BiCG 成对归约 2.18%，KUPL 4-rank legacy MG 成对归约 4.19%；CUDA MG 无 V-cycle 回归，三方报告与门禁通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage17-clover-block-gate-20261021 - stage - 2026-10-21 21:29:43 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主、BiCGStab 为辅
- 日志文件: logs/v202610212129.txt
- 文件大小: 7685 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA legacy MG 增加严格 Clover 6x6 结构检测与完整 12x12 回退；生产 MG 后端降低 16.04%，解逐元素相同，三方报告与门禁通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage18-bicg-clover-block-20261021 - stage - 2026-10-21 21:39:33 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主、BiCGStab 为辅
- 日志文件: logs/v202610212139.txt
- 文件大小: 6132 bytes
- 触发原因: 阶段成果
- 关键内容: Clover BiCGStab 6x6 结构门控降低 16.58% 并逐元素保持解；CUDA wall 比 QUDA 快 3.59%，三方报告与 strict 门禁通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage19-fused-norm-rejected-20261021 - stage - 2026-10-21 21:51:46 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主、BiCGStab 为辅
- 日志文件: logs/v202610212151.txt
- 文件大小: 5533 bytes
- 触发原因: 阶段成果
- 关键内容: 融合 r/x 与残差范数候选仅 0.23% 收益已撤销；最终 CUDA BiCG wall 比 QUDA 快 4.91%，三方报告与门禁通过
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage20-sync-chunk-kupl-scaling-20261021 - stage - 2026-10-21 22:20:17 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610212220.txt
- 文件大小: 10416 bytes
- 触发原因: 阶段成果
- 关键内容: Stage20 CUDA 每2迭代收敛检查降低1.80%，CUDA 18对QUDA 19次且wall快6.33%；KUPL c64 4-rank 2.978x/74.5%，c128生产restart0与独立BiCG真残差通过；严格标注MG不可比口径与c128 coarse V-cycle缺口
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage21-mg-pair-reduction-20261021 - stage - 2026-10-21 23:43:50 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610212343.txt
- 文件大小: 7368 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA legacy MG fine成对归约ABBA降低3.01%，20 iterations与生产真残差门禁保持；strict/c64/c128/ABI全绿；Graph与双流overlap候选实测失败已撤销，并完成三方性能报告
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage22-mg-single-dot-20261021 - stage - 2026-10-21 23:55:52 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主，BiCGStab 为辅；CUDA/KUPL 子任务并行
- 日志文件: logs/v202610212355.txt
- 文件大小: 6719 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA legacy MG fine单点积ABBA再降3.38%，相对Stage20累计降5.88%；strict/Clover/ABI/c64/c128全绿；KUPL专项并行调查2/4-rank效率损失并报告三方性能
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage23-parallel-diagnostics-20261022 - stage - 2026-10-22 00:21:56 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主，BiCGStab 为辅；CUDA/KUPL 并行诊断
- 日志文件: logs/v202610220021.txt
- 文件大小: 6903 bytes
- 触发原因: 阶段成果
- 关键内容: KUPL瓶颈分解显示4-rank fine apply占73.2%，5→4 Allreduce候选慢10.67%已撤销；CUDA 6x6特化仅0.19%且无寄存器收益已撤销；两侧门禁通过并给出三方性能
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage24-real-mg-and-kupl-fine-20261022 - stage - 2026-10-22 00:44:28 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：真实 MG、fine dslash 与并行效率
- 日志文件: logs/v202610220044.txt
- 文件大小: 7144 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA E12真实MG各模式均慢于fine-only，deflate残差更低但wall增9.9%；KUPL fine标量边界SIMD候选慢4.60%已撤销；完成三方性能与容量约束报告
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## stage25-bicg-single-dot-20261022 - stage - 2026-10-22 01:04:50 +0800
- 主任务: 持续比照 QUDA 优化 KUPL/CUDA QCU：MG 为主，BiCGStab 为辅
- 日志文件: logs/v202610220104.txt
- 文件大小: 5936 bytes
- 触发原因: 阶段成果
- 关键内容: CUDA standalone Clover BiCG单点积ABBA降低2.85%，18 iterations与真残差保持，相对QUDA wall快10.11%；冷构建null-space无显著变化；完成三方报告和完整门禁
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## dev100-to-worktree-handoff-20261022 - general - 2026-10-22 01:10:47 +0800
- 主任务: dev100 至当前工作区工作总结与续接交接
- 日志文件: logs/v20261022011047.txt
- 文件大小: 6083 bytes
- 触发原因: 一般记录
- 关键内容: 新增dev100至今Markdown总结并生成续接TXT：当前CUDA BiCG 0.120462s、MG 0.13295s，KUPL 4-rank 15.072466s；记录保留优化、撤销方向、门禁证据、风险与下一阶段优先级
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## tag-dev101-20261022 - final - 2026-10-22 01:16:58 +0800
- 主任务: PyQCU dev101 本地开发标签归档
- 日志文件: logs/v2026102201.txt
- 文件大小: 3852 bytes
- 触发原因: 任务完成
- 关键内容: 创建 dev101 注释标签，固定 HEAD 5a21a5e6，验证本地标签链；未提交工作区和未推送状态明确保留
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## form-20261022 - final - 2026-10-22 01:37:19 +0800
- 主任务: PyQCU form 整理：tag 名路径整改与目录索引补齐
- 日志文件: logs/v2026102201-2.txt
- 文件大小: 7243 bytes
- 触发原因: 任务完成
- 关键内容: 重命名 2 组 tag 名路径资产、补齐 refer/docs 索引与 form 例外口径；审计 PASS，豁免 1069 项，tag-chain 回归与 bash -n 通过
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## three-way-benchmark-20261022 - final - 2026-10-22 03:17:11 +0800
- 主任务: CUDA QCU / KUPL QCU / QUDA 三方性能测评与一键脚本
- 日志文件: logs/v2026102203.txt
- 文件大小: 8085 bytes
- 触发原因: 任务完成
- 关键内容: CUDA BiCG 比 QUDA 快13.20%；KUPL restart=0 快2.8-3倍、L24 4-rank 5.530s；报告与bench_three_way.py交付，门禁PASS
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## three-way-benchmark-20261022 - final - 2026-10-22 06:11:42 +0800
- 主任务: CUDA QCU / KUPL QCU / QUDA 三方性能测评、生产脚本拆分与多 rank 真残差
- 日志文件: logs/v2026102206.txt
- 文件大小: 14547 bytes
- 触发原因: 任务完成
- 关键内容: 生产/测量脚本拆分、分布式独立真残差、sm60/sm70、同窗三方测量、并行扩展与 LTO 优化
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## simple-mpi - final - 2026-10-22 15:35:49 +0800
- 主任务: 实现 pyqcu/testing/simple MPI 小格点入口
- 日志文件: logs/v2026102215.txt
- 文件大小: 7764 bytes
- 触发原因: 任务完成
- 关键内容: 新增 KUPL/CUDA MPI 小格点 QCU MG 入口，LIME 兼容并完成两后端 PASS 验证
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## simple-docs - final - 2026-10-22 16:13:12 +0800
- 主任务: 丰富 simple 与 simple/mpi 用户文档并补充 OpenMPI 绑定说明
- 日志文件: logs/v2026102216.txt
- 文件大小: 7482 bytes
- 触发原因: 任务完成
- 关键内容: 重写两目录用户文档，覆盖全部主要 CLI、LIME、报告和 OpenMPI 绑定，并完成实现一致性修复与验证
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## test9-dev103-profiler-summary - final - 2026-10-22 16:29:26 +0800
- 主任务: 汇总 test9 至 dev103 标签区间的 QCU/PyQCU/QUDA 求解器性能记录
- 日志文件: logs/v2026102216-2.txt
- 文件大小: 89489 bytes
- 触发原因: 任务完成
- 关键内容: 340 个唯一性能/失败/备注记录；test27 66 行与 stab48 57 行完整入表；仅汇总不重测
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## qcu-quda-mg-docs - final - 2026-10-22 16:41:15 +0800
- 主任务: 编写 QCU 与 QUDA MultiGrid 算法组成和流程图文档
- 日志文件: logs/v2026102216-3.txt
- 文件大小: 5345 bytes
- 触发原因: 任务完成
- 关键内容: 新增 docs/qcu_quda_multigrid_algorithms.md，含 7 张 Mermaid 流程图、默认参数快照和并排对照
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## qcu-quda-mg-subalgorithm-docs - final - 2026-10-22 16:55:53 +0800
- 主任务: 展开 QCU/QUDA MG 子算法分析、流程图和同类算法优劣
- 日志文件: logs/v2026102216-4.txt
- 文件大小: 5220 bytes
- 触发原因: 任务完成
- 关键内容: 新增 1354 行子算法详解，含 25 张 Mermaid 流程图、52 个 LaTeX 公式块和同类算法权衡
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## test9-dev103-profiler-summary - final - 2026-10-22 17:04:48 +0800
- 主任务: 为 test9 至 dev103 求解器性能汇总补充统一环境信息
- 日志文件: logs/v2026102217.txt
- 文件大小: 103782 bytes
- 触发原因: 任务完成
- 关键内容: 30 张记录表统一环境列；409 个唯一记录 ID、13 个环境定义；X99 未确认项明确保留；仅汇总不重测
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## analysis - final - 2026-10-22 17:05:37 +0800
- 主任务: QCU MG 粗层 gauge/clover 与 dslash 表示
- 日志文件: logs/v2026102217-2.txt
- 文件大小: 5207 bytes
- 触发原因: 任务完成
- 关键内容: 区分 legacy 33-tensor Schur/Galerkin 粗算子与 Strict/QUDA X-Y-Yhat 表示
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## docs - final - 2026-10-22 17:11:14 +0800
- 主任务: 将 QCU/QUDA MG 粗层 dslash 表示调查写入 docs
- 日志文件: logs/v2026102217-3.txt
- 文件大小: 3605 bytes
- 触发原因: 任务完成
- 关键内容: 新增粗层算子表示正式文档，更新索引与算法总览交叉引用，并完成 Markdown 与仓库检查
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## analysis - final - 2026-10-22 17:17:41 +0800
- 主任务: 核查 C++ QCU 与 Torch/PyQCU 当前默认 MG 路径
- 日志文件: logs/v2026102217-4.txt
- 文件大小: 3794 bytes
- 触发原因: 任务完成
- 关键内容: 确认 C++ 与 Torch 均并存 Legacy/Strict，但现有 MG 驱动和构造默认均为 Legacy/非 Strict，Strict 需显式选择
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## analysis - final - 2026-10-22 17:24:42 +0800
- 主任务: 解释 C++ QCU 默认 Legacy MG 而非 Strict-MG 的原因
- 日志文件: logs/v2026102217-5.txt
- 文件大小: 4614 bytes
- 触发原因: 任务完成
- 关键内容: 默认由调用兼容性和工程边界决定，并非统一性能裁决；Strict 的 setup/cache 成本需按重复求解场景摊销
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## analysis - final - 2026-10-22 17:31:25 +0800
- 主任务: 比较 C++ QCU Legacy 与 Strict 的完成度和多卡能力
- 日志文件: logs/v2026102217-6.txt
- 文件大小: 4161 bytes
- 触发原因: 任务完成
- 关键内容: 综合完成度 Legacy 最高；CUDA Legacy/Strict 均支持 MPI 一进程一卡多卡，Legacy 另有单进程多线程多 GPU 驱动
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## analysis - final - 2026-10-22 17:38:57 +0800
- 主任务: 解释 Legacy 单进程多线程多 GPU 驱动
- 日志文件: logs/v2026102217-7.txt
- 文件大小: 3114 bytes
- 触发原因: 任务完成
- 关键内容: MultiGpuMultigrid 是一个进程内每线程绑定一张 GPU 并独立执行完整 MG，属于复制式吞吐并行而非单次 solve 的跨卡分域
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## docs - final - 2026-10-22 17:55:58 +0800
- 主任务: 标注 MultiGpuMultigrid 为测试专用并固化 CUDA 进程-rank-设备拓扑
- 日志文件: logs/v2026102217-8.txt
- 文件大小: 4439 bytes
- 触发原因: 任务完成
- 关键内容: 同步代码注释、AGENTS、skills 与用户文档，明确显式 opt-in 和一句进程-rank-设备生产约定
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## skills - final - 2026-10-22 18:03:49 +0800
- 主任务: 将 CUDA 进程-rank-设备与 MultiGpu 测试专用约定写入 skills
- 日志文件: logs/v2026102218.txt
- 文件大小: 4256 bytes
- 触发原因: 任务完成
- 关键内容: 在 skills 根约定和 15 个相关 skill 中建立一进程一 rank 一设备规则，并同步 AGENTS 与当前 Strict MPI 能力
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## docs-to-skills - final - 2026-10-22 18:36:52 +0800
- 主任务: 提取 docs 参考内容到 skills
- 日志文件: logs/v2026102218-2.txt
- 文件大小: 4469 bytes
- 触发原因: 任务完成
- 关键内容: 新增 pyqcu-docs 文档导航技能、17/17 来源映射与蒸馏工程参考，并同步域技能指针和文档索引
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## go-on-legacy-vs-strict-20261022 - final - 2026-10-22 23:41:36 +0800
- 主任务: 恢复四个中断运行并完成 MG 弃用审计与状态报告
- 日志文件: logs/v2026102223.txt
- 文件大小: 6559 bytes
- 触发原因: 任务完成
- 关键内容: Legacy/Strict CUDA 报告完成；三条只读审计完成；stable HEAD 9e678901；无 GPU/MPI
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## docs-multigrid-audit-20261022 - final - 2026-10-22 23:46:36 +0800
- 主任务: 将 MultiGrid 审计与 benchmark 状态结果正式写入 docs
- 日志文件: logs/v2026102223-2.txt
- 文件大小: 3908 bytes
- 触发原因: 任务完成
- 关键内容: 新增审计文档；更新 docs 索引和来源映射；状态报告交叉引用；Markdown/form/Git 检查通过
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## multigrid-unified-wrapper-20261022 - final - 2026-10-22 23:55:26 +0800
- 主任务: 重构 PyQCU Torch/CUDA/KUPL multigrid 层级与统一接口，并归档 strict/legacy 对比摘要
- 日志文件: logs/v2026102223-3.txt
- 文件大小: 7320 bytes
- 触发原因: 任务完成
- 关键内容: 统一 Multigrid 上层与 StrictMultigrid/LegacyMultigrid 下级；清除 qcu 代码中的 Quda* 命名；CPU/KUPL/V100 回归通过；docs 报告完成
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## legacy-strict-mg-memory-20261023 - final - 2026-10-23 00:09:43 +0800
- 主任务: 比较 QCU Legacy-MG 与 Strict-MG 运行时显存峰值
- 日志文件: logs/v2026102300.txt
- 文件大小: 4437 bytes
- 触发原因: 任务完成
- 关键内容: Legacy-MG 在现有 c64 两级可比记录中峰值更低；Strict 高 32.46% 至 63.25%
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## torch-backend-layout-20261023 - final - 2026-10-23 00:28:34 +0800
- 主任务: 建立 pyqcu/torch 后端命名空间并迁移 Torch 版 qcu 实现，保留公共层与兼容门面
- 日志文件: logs/v2026102300-2.txt
- 文件大小: 5977 bytes
- 触发原因: 任务完成
- 关键内容: Torch dslash/solver/smear/tools 已迁移到 pyqcu/torch；公共 lattice/tools/selector 保留顶层；CPU/KUPL/V100 回归通过
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## docs-conclusions-to-skills-20261023 - final - 2026-10-23 00:31:08 +0800
- 主任务: 将 Legacy/Strict-MG 显存结论写入 docs，并将全部 docs 的稳定结论同步到 skills
- 日志文件: logs/v2026102300-3.txt
- 文件大小: 6065 bytes
- 触发原因: 任务完成
- 关键内容: 新增显存正式文档；22 份 docs 全部进入技能映射与结论覆盖索引；关键版本、测试和启停边界已校正
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## qcu-unified-mg-20261023 - final - 2026-10-23 00:50:10 +0800
- 主任务: 分析 Torch/CUDA/KUPL 的 Legacy 与 Strict MG 并设计统一 PyQCU 接口
- 日志文件: logs/v2026102300-4.txt
- 文件大小: 8674 bytes
- 触发原因: 任务完成
- 关键内容: 完成六实现对比、统一架构设计、Torch Legacy 层级构建回归修复与验证
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## pyqcu-api-p0-20261023 - final - 2026-10-23 01:01:13 +0800
- 主任务: 实现 PyQCU 统一 QCU 架构 P0 公共 API 契约
- 日志文件: logs/v2026102301.txt
- 文件大小: 6355 bytes
- 触发原因: 任务完成
- 关键内容: 新增 pyqcu.api 公共模型、配置、结果、能力、资产清单与 Protocol，并为 Multigrid 增加兼容 solve_result 和最终回归
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## docs-skills-governance-20261023 - final - 2026-10-23 01:14:46 +0800
- 主任务: 全面规范优化 docs 与 skills 目录
- 日志文件: logs/v2026102301-2.txt
- 文件大小: 6494 bytes
- 触发原因: 任务完成
- 关键内容: docs=24、skills=35；建立统一审计器、trigger-only descriptions、唯一 Primary 映射与 pyqcu-api 技能
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## qcu-six-way-benchmark-20261023 - final-uncommitted - 2026-10-23 02:12:41 +0800
- 主任务: 真实 LIME 前缀上的 Torch/CUDA/KUPL 六版本与 QUDA 同配置性能评定
- 日志目录: logs/qcu_six_way_20261023/formal-release
- 摘要文件: summary.json、summary.csv、summary.md、inputs.json、environment.json
- 关键内容: 7 条主路径与 CUDA Legacy fine-only 消融全部 PASS；记录独立真残差、coarse/V-cycle 证据、setup/cache/first workspace/steady 和设备峰值
- 验证: 5 次 steady、同 gauge/source hash、同 c64/tol/max_iter/levels/block/nvec、JSON/CSV 一致性
- 批次状态: uncommitted
## qcu-p1-testing-benchmark-20261023 - final - 2026-10-23 02:31:21 +0800
- 主任务: 统一 QCU P1 接口、整改 testing、拆分 simple、近真实组态六版本性能评定并约定 Legacy-MultiGrid 默认
- 日志文件: logs/v2026102302.txt
- 文件大小: 8997 bytes
- 触发原因: 任务完成
- 关键内容: 完成 LinearOperator/Krylov 统一适配、Legacy 默认约定、注释与测试目录整改、弃用清理、simple 分层和七路径真实 LIME 基准
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## qcu-l24-mg-cache - final - 2026-10-23 02:51:56 +0800
- 主任务: QCU L24 Legacy/Strict/QUDA MG cache 口径与大小
- 日志文件: logs/v2026102302-2.txt
- 文件大小: 4973 bytes
- 触发原因: 任务完成
- 关键内容: 按 c64、两级、block2^4、nvec12 核对 Legacy 实际 cache、Strict schema 逻辑尺寸及 QUDA 无持久粗算子 cache
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## qcu-l24-mg-cache-formula - final - 2026-10-23 02:56:39 +0800
- 主任务: 推导 L24 QCU Legacy/Strict 与 QUDA MG cache 算式
- 日志文件: logs/v2026102302-3.txt
- 文件大小: 5509 bytes
- 触发原因: 任务完成
- 关键内容: 按 V/Vc/Vodd/P/complex64 展开 Legacy 33-tensor、Strict V/Yhat/X 和 QUDA null-vector 字节公式
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## docs-qcu-l24-mg-cache - final - 2026-10-23 03:15:13 +0800
- 主任务: 保存 L24 MG cache 尺寸计算过程到 docs
- 日志文件: logs/v2026102303.txt
- 文件大小: 3480 bytes
- 触发原因: 任务完成
- 关键内容: 新增正式技术参考文档并同步 pyqcu-docs 索引、映射与提取摘要
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## tag-history-query - final - 2026-10-23 09:44:04 +0800
- 主任务: 核查 git tag test27 系列中 QCU 使用 Legacy-MG 还是 Strict-MG
- 日志文件: logs/v2026102309.txt
- 文件大小: 4223 bytes
- 触发原因: 任务完成
- 关键内容: 确认 test27/test27_1/test27_2/test27_3/test27_4 系列使用 pyqcu.cuda.qcu.applyMultigridStrictFgmresQcu，即 Strict-MG；未使用 Legacy-MG。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
