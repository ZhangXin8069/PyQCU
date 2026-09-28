# High-Performance Implementation of Multigrid Solver for Lattice QCD

## 1. 标题页（20 秒）
介绍题目、汇报人和证据基线。强调本报告只主张 MultiGrid，不把结果外推到全部求解器。

## 2. Wilson 与 Clover（35 秒）
先给 H、D_W、κ，再给 Clover onsite 块。说明 C++ 返回的裸核与完整算子不是一个对象。

## 3. 原始 Dslash（30 秒）
按“八邻居、spin projection、SU(3)、累加”讲带宽瓶颈；指出 xyzt 尾轴布局和投影重用。

## 4. 奇偶消元（30 秒）
块矩阵、Schur 补、RHS 与重建。强调只改变未知量组织，真残差仍回到 full operator。

## 5. 通用 MG（35 秒）
null vector 近似低模，局部 QR 建 P/R，Galerkin 生成粗算子，V-cycle 作预条件器。

## 6. 预条件 Bi-CGStab（35 秒）
按表逐行说明 rho、p、v、alpha、s、omega 和 x 更新。强调预条件器 M 与热启动路径。

## 7. PyQCU Strict（35 秒）
D=X+H、Dhat=X^-1 D、D_c=R Dhat P。细层 compact target parity，粗层 full X/Y/Yhat。

## 8. V-cycle 与 FGMRES（35 秒）
左侧是递归 MG 预条件器，右侧是 flexible right-preconditioned GMRES。说明迭代不跨层相加。

## 9. CUDA-C++ 优化（40 秒）
从寄存器、流融合、device scalar、CUDA Graph、setup/cache 讲到 MPI overlap；c128 默认回退。

## 10. 对比协议（30 秒）
72 请求、66 精确、132 side record；1+2+5 phase；真残差与 85% 显存门。

## 11. 全分布（35 秒）
总中位 1.025，但 MG 35/44，trace-off MG 21/22；长尾和 BiCG 0/22 必须一起讲。

## 12. 分组中位（35 秒）
V100 MG-3 最突出，P100 trace-off 温和有利；trace-on 只作诊断。

## 13. 瓶颈与容量（40 秒）
PyQCU 记录中 coarse+other 主导；双 P100 大格 6 项容量替代，不能混入原大格速度比。

## 14. 优势、不足与下一步（35 秒）
优势是 MG 专项、自主可控、显存；不足是总能力和通信成熟度；下一步聚焦粗层同步、通信和 setup。

## 15. 参考文献与复现（25 秒）
列出 quantum-mg、QUDA、DDalphaAMG；快速闸门先于完整矩阵，并简述平台适配状态。

## 16. 508 报告大规模测试（30 秒）
展示 508 报告的 64→1024 进程强扩展曲线、弱扩展差异曲线和 TEST9 原始表。
说明这是早期 DCU 路径成果，MG 当时仍属前期验证，不能与 test27 的 Strict-MG 速度比混合。

## 17. 致谢（10 秒）
一句总结和一个明确行动项，然后进入问答。
