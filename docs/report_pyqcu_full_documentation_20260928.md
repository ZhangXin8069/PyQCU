# PyQCU 全文汇总：Wilson/Clover、Strict MultiGrid、CUDA/MPI 实现与最终性能对照

> 文档版本：2026-09-28<br>
> 合并范围：`docs/` 下当前有效的指南、技术报告、性能报告、算法说明、历史复盘与图表资产<br>
> 源码基线：`test27-7-g6a37d61-dirty`<br>
> 权威性能口径：`test27` 与 `report_multigrid_quda_pyqcu_20260928.tex`<br>
> 证据等级：E1 本机实测；E2 源码级推导或逐行对照；E3 历史结论或外部参考

## 文档控制

### 合并原则

本文不是旧文档的顺序拼接，而是一次带优先级裁决的全文整合。内容重叠时按下述顺序选择：

| 优先级 | 来源 | 使用方式 |
|---:|---|---|
| 1 | 当前工作区源码 | 解决 ABI、分支、默认值、分布式能力和生命周期歧义 |
| 2 | 2026-09-28 报告与当日高性能全文 | 最终性能、覆盖、残差、显存和结论 |
| 3 | 2026-09-27 及更早正式报告 | 实现演进、被撤回口径、历史 A/B 和失败路径 |
| 4 | 分析型文档与计划 | 数学定义、验收设计、流程和验证矩阵 |
| 5 | `.pdf`、`.pptx`、渲染图与重复运行目录 | 派生交付物，只在源文件不可读或图表需单独展示时使用 |

同一文件同时存在 `.tex`、`.md`、`.pdf` 时，优先分析源文件，PDF 仅作为编译或展示产物。同一报告存在 `_2`、`_3`、`_4`、`_5` 等连续版本时，只保留内容上最新的修订；旧版本的独特证据进入历史章节。

### 质量筛选

低信息密度、重复背景、过程日志、仅证明“命令已执行”的输出、无独立证据的推断、中间渲染页和可由源码或表格重建的图，不进入正文。被舍弃内容仍保留原始路径、版本关系或替代关系，便于回溯。

保留标准是满足以下至少一项：

| 类别 | 保留内容 |
|---|---|
| 物理 | 作用量、算子、对称性、离散误差、极限与量纲检查 |
| 数学 | Schur 消元、Galerkin、残差、Krylov 递推、资产定义 |
| 工程 | ABI、线程/rank 隔离、缓存身份、通信、内存和生命周期 |
| 实证 | 运行协议、真残差、时间、迭代、显存、容量替代和错误边界 |
| 可复现 | 命令、入口、输入身份、版本、图和机器可读证据路径 |
| 风险 | 被撤回结论、失败实验、未验证项和外推边界 |

### 当前结论

`test27` 最终矩阵请求 72 个 combined 单元，实际完成 66 个精确单元和 132 条 side record。剩余 6 个 combined 单元仅对应双 P100 的 c64 `16x32x32x48`，因显存或 570 秒单实例时限被明确替代，不使用历史值补齐。

所有 66 个精确单元均满足两侧 `status=ok`、配置哈希一致、输入 bundle 一致、1 cold 加 2 warmup 加 5 steady 完整，并通过独立 full-operator 真残差门。整体稳态时间比定义为

$$
R=\frac{t_{\mathrm{QUDA,steady}}}{t_{\mathrm{PyQCU,steady}}},\qquad R>1\ \text{表示 PyQCU 更快}.
$$

全部 66 个精确单元的逐单元比值中位数为 `1.0250`。MG-2/3 在 `35/44` 个单元中优于 QUDA；仅看正式 trace-off 口径，MG 为 `21/22`。BiCGStab reference 为 `0/22`，因此结论只适用于 MG 专项，不可外推为 PyQCU 全部求解器领先。

PyQCU 与 QUDA 的最大 full true residual 分别为 `7.40e-7` 与 `7.97e-7`，均低于 `5e-6`。保留样本的最大 device-wide 观测显存分别为 `11907.3 MiB` 与 `23063.3 MiB`，均低于设备容量 85% 门限。

### 阅读导航

| 主题 | 章节 |
|---|---|
| 工程结构、布局、ABI、构建与测试 | 1 至 3 |
| Wilson/Clover、Schur、Krylov、MG 数学基础 | 4 至 7 |
| QUDA 粗层资产与矩阵顺序 | 8 |
| PyQCU CUDA、MPI、缓存和生命周期 | 9 至 13 |
| 最终性能、残差、显存与容量边界 | 14 至 18 |
| 实现演进、撤回指标和失败实验 | 19 至 24 |
| 2025 DCU 历史应用测试 | 25 |
| 图表总览、复现和验证矩阵 | 26 至 28 |
| 来源台账与处置规则 | 29 至 30 |

---

## 1. 项目结构与执行层级

PyQCU 是面向格点 QCD 的 Python/Cython/C++ CUDA 库，覆盖 Wilson/Clover Dirac 算子、BiCGStab、FGMRES、MultiGrid、stout smearing、规范场生成和设备后端适配。核心计算在四维 MPI 进程网格上分布。

| 层级 | 路径 | 职责 |
|---|---|---|
| 纯 Python | `pyqcu/dslash/`、`pyqcu/solver/`、`pyqcu/smear/` | PyTorch/兼容层实现，支持 CPU、CUDA 和 NPU 参考路径 |
| Python 工具 | `pyqcu/tools/` | MPI 网格、布局转换、HDF5、线性代数、coarse setup |
| Cython 桥 | `pyqcu/cuda/qcu/` | 将扁平参数和张量指针转换为 C API 调用，释放 GIL |
| C++/CUDA 生产后端 | `cpp/cuda/qcu/` | Wilson/Clover Dslash、Schur、Krylov、Strict MG、halo 和 MPI |
| 多卡驱动 | `pyqcu/cuda/_multi_gpu.py` | 一线程一卡任务并行，不把多个线程误解为单个 RHS 的域分解 |
| 测试入口 | `pyqcu/testing/` | Python、QCU、QUDA、PyQUDA、CPU/NPU/DCU/GPU 等分类测试 |
| 正式文档 | `docs/` | 指南、论文、阶段性报告、对照报告和图表资产 |
| 运行证据 | `data/`、`logs/` | 张量、构建树、机器可读记录、日志和 tag 报告 |

项目有两套重要但语义不同的执行路径。

| 路径 | 求解器 | 并行方式 | 备注 |
|---|---|---|---|
| 纯 Python | PyTorch 实现 | CPU/CUDA/NPU、Python MPI | 可移植原型和算子级验证 |
| C++ CUDA | Cython 绑定后的生产内核 | MPI rank 域分解和 CUDA halo | 正式性能路径 |

所有纯 Python 代码应通过 `pyqcu.cann as _torch` 访问张量后端。尤其 NPU 路径不能硬编码 `import torch`，以避免绕过复数张量拆解兼容层。

## 2. 维度、布局、MPI 与参数 ABI

### 2.1 张量布局

时空维始终放在张量最后四轴，统一记为 `...xyzt`。ward 方向使用负整数索引：`wards['x']=-4`、`wards['y']=-3`、`wards['z']=-2`、`wards['t']=-1`。

| 对象 | 规范布局 |
|---|---|
| 规范场 | `[c,c,d,x,y,z,t]` |
| 奇偶规范场 | `[p,c,c,d,x,y,z,t]` |
| 费米子 | `[s,c,x,y,z,t]` |
| 奇偶费米子 | `[p,s,c,x,y,z,t]` |
| Wilson block | `[s,c,s,c,x,y,z,t]` |
| Clover term | `[s,c,s,c,x,y,z,t]` |

HDF5 内部使用 `zyxt` 序。跨布局转换使用：

| 转换 | 入口 | 含义 |
|---|---|---|
| `ccdxyzt -> ccdptzyx` | `pyqcu.tools.ccdxyzt2ccdptzyx` | 规范场转到 parity、`tzyx` 和内部表示 |
| `scxyzt -> psctzyx` | `pyqcu.tools.scxyzt2psctzyx` | 费米子转到 parity、`tzyx` |
| 分布式 HDF5 写入 | `gridoooxyzt2hdf5oooxyzt` | rank slab 到全局 `xyzt` HDF5 |
| 分布式 HDF5 读取 | `hdf5oooxyzt2gridoooxyzt` | 全局 HDF5 到本地 rank slab |

转换前后必须同时核对 parity、spin-color 顺序、方向索引、复共轭和 HDF5 的 `zyxt` 约定。仅比较张量 shape 不能证明布局兼容。

### 2.2 参数协议

Python 与 C++ 通过三个扁平数组桥接。当前源码基线仍保持 `params` 长度 58 的 ABI。

| 数组 | 类型与长度 | 内容 |
|---|---|---|
| `params` | `int32[58]` | 几何、rank、plan、精度、level、MG 模式和生命周期索引 |
| `argv` | `float[7]` | mass、atol、sigma 和各层容差 |
| `set_ptrs` | `int64[100]` | `LatticeSet`、scratch、coarse asset 和持久 hierarchy 指针 |

当前关键索引如下：

| 索引 | 值 | 语义 |
|---|---:|---|
| `_SET_PLAN_` | `-2` | Laplacian |
| `_SET_PLAN_` | `-1` | Gauss gauge |
| `_SET_PLAN_` | `0` | Wilson dslash |
| `_SET_PLAN_` | `1` | BiStabCG/CG |
| `_SET_PLAN_` | `2` | Clover dslash |
| `_MG_USE_DEFLATE_` | `55` | deflation 开关 |
| `_MG_MU_PRE_` | `56` | 预平滑次数 |
| `_MG_USE_INIT_GUESS_` | `57` | 热启动开关 |
| `_SET_PTRS_STRICT_COARSE_BASE_` | `60` | Strict coarse asset 基址 |
| `_SET_PTRS_STRICT_STRIDE_` | `4` | 每条 transition 的槽位数 |
| `_SET_PTRS_STRICT_HIERARCHY_` | `80` | 持久 Strict hierarchy 句柄 |

Strict coarse asset 每组四槽依次为 blocked basis `V`、raw links `Y`、preconditioned links `Yhat`、onsite pair `(X, X^-1)`。

`pyqcu/cuda/define.py` 与 `cpp/cuda/qcu/include/define.h` 必须同步。当前源码在两处都定义了相同的 58 项 ABI 和 mode bit mask：

| 位 | 含义 |
|---:|---|
| `2` | MR smoother |
| `4` | Chebyshev |
| `8` | CA-GCR |
| `16` | W-cycle |
| `32` | F-cycle |
| `64` | K-cycle |
| `128` | BiCGStab(l) |

### 2.3 调用生命周期

普通 QCU 操作遵循：

```text
applyInitQcu
  -> 一次或多次算子/求解器调用
  -> params[define._SET_INDEX_] += 1
  -> applyEndQcu
```

每次调用之间必须递增 `_SET_INDEX_`。不递增会复用 scratch、工作区和 `LatticeSet` 槽位，错误通常不会在单次 kernel 内部立即暴露。

Strict 持久 hierarchy 是例外：在 hierarchy 的多次 V-cycle/FGMRES 调用间保持同一 `_SET_INDEX_`。其生命周期为

```text
applyMultigridStrictInitQcu
  -> repeated V-cycle / FGMRES
  -> applyMultigridStrictEndQcu
```

### 2.4 多线程多卡

`MultiGpuMultigrid` 使用一线程一卡模型。每个线程必须具备：

| 隔离对象 | 要求 |
|---|---|
| CUDA context | 线程内独立设置 device |
| `params`、`argv`、`set_ptrs` | 独立副本 |
| `_SET_INDEX_` | 各自从 0 计数 |
| hierarchy 与 scratch | 每线程独立 |
| HDF5 句柄 | 每次调用独立 `with h5py.File(...)` |

Cython 函数先持有 GIL 取张量指针，再通过 `with nogil` 调用 C++。pxd 中的 C API 声明必须带 `nogil`，并使用 `qcu_api.pxd` 别名避免和 pyx 的 Python `def` 同名。

当前多卡驱动求的是多个独立任务，而不是把一个 RHS 域分解到多张卡。其可测量指标是吞吐和线程一致性，不是单 RHS 延迟加速。

## 3. 环境、构建、测试与组织

### 3.1 环境

推荐 Python 3.10 或更高版本。核心依赖包括 PyTorch、Cython、mpi4py、h5py、NumPy 和 CUDA toolkit；TileLang 可选。

```bash
source ./env.sh
bash ./build.sh
bash ./install.sh
```

必须保证：

| 变量 | 用途 |
|---|---|
| `PYTHONPATH` | 指向 PyQCU 仓库根 |
| `LD_LIBRARY_PATH` | 找到 `libqcu.so` 和 CUDA/MPI 运行库 |
| `QUDA_PATH` | QUDA 对照构建安装路径 |
| MPI root 权限 | 容器内允许 `mpirun` 时按环境配置 |

### 3.2 常用测试

```bash
cd pyqcu/testing && pytest .
mpirun -np 4 python pyqcu/testing/python/conftest.py
python pyqcu/testing/qcu/single_qcu_api.py
pytest pyqcu/testing/quda
```

顶层 `examples/` 已删除。历史文档中的 `examples/...` 路径全部映射到 `pyqcu/testing/...`，当前命令必须使用新路径。

Profiler 路径为 `pyqcu/testing/profiler/`，典型命令为：

```bash
cd PyQCU/pyqcu/testing/profiler
mpirun -np 1 python -u conftest.py
```

生成的 `trace_*.json` 可载入 `https://ui.perfetto.dev/`。

### 3.3 文件分类

| 类型 | 唯一规范目录 |
|---|---|
| 正式独立文档 | `docs/**` |
| 日志、运行摘要、tag 报告 | `logs/**` |
| HDF5、SO、DAT、CSV、SVG、构建安装树 | `data/**` |
| 测试和其他代码 | `pyqcu/testing/**` |

禁止用软链接或嵌套归档复制资产。LaTeX 辅助文件、逐页渲染、一次性缓存和会话垃圾应删除或保留在明确标记的构建工作区。

---

## 4. Wilson 与 Clover 算子

### 4.1 规范链接与协变平移

规范链接满足 `U_{x,mu} in SU(3)`。正、反向协变平移为

$$
T_{+\mu}\psi_x=U_{x,\mu}\psi_{x+\hat\mu},
\qquad
T_{-\mu}\psi_x=U^\dagger_{x-\hat\mu,\mu}\psi_{x-\hat\mu}.
$$

自由场极限 `U=1` 是检查邻居索引、gamma 约定和边界条件的最低成本测试。

### 4.2 Wilson hopping 核

取 Wilson 参数 `r=1`，定义无单位项和 `-kappa` 因子的 hopping 核

$$
(H\psi)_x=
\sum_{\mu=0}^{3}
\left[
(1-\gamma_\mu)U_{x,\mu}\psi_{x+\hat\mu}
+(1+\gamma_\mu)U^\dagger_{x-\hat\mu,\mu}\psi_{x-\hat\mu}
\right].
$$

Wilson 部分和作用量为

$$
D_W=(m_0+4)I-\frac12H,
\qquad
S_W=\sum_x\bar\psi_xD_W\psi_x,
\qquad
\kappa=\frac{1}{2m_0+8}.
$$

预条件归一化形式为

$$
D_{\mathrm{pc}}=I-\frac{\kappa}{u_0}H,
\qquad
D_W=(m_0+4)D_{\mathrm{pc}}.
$$

当前 C++ 生产路径等价于 `u0=1`。原始 Wilson Dslash 接口返回裸核 `H`；带单位项的 Python 接口返回 `I-(kappa/u0)H`。跨实现比较必须先统一该归一化。

### 4.3 Clover 改进

有向 plaquette 为

$$
P_{\mu\nu}(x)=
U_\mu(x)U_\nu(x+\hat\mu)
U^\dagger_\mu(x+\hat\nu)U^\dagger_\nu(x).
$$

四叶 Clover 和为

$$
Q_{\mu\nu}=P_{\mu\nu}+P_{\nu,-\mu}+P_{-\mu,-\nu}+P_{-\nu,\mu},
\qquad
C_{\mu\nu}=Q_{\mu\nu}-Q^\dagger_{\mu\nu}.
$$

代码级 onsite block 为

$$
T_p=-\frac{\kappa c_{\mathrm{sw}}}{8u_0}
\sum_{\mu<\nu}\gamma_\mu\gamma_\nu C_{\mu\nu},
\qquad
A_p=I_p+T_p.
$$

当前有效 `c_sw=1`。Clover 项不扩大最近邻 stencil，但每个站点从纯颜色 onsite 问题提升为 `12x12` spin-color onsite 问题。实现必须批量构造 Clover、批量求逆并复用 onsite 因子，不能使用逐点通用矩阵逆。

### 4.4 对称性、谱与极限检查

Wilson 算子满足 gamma5-Hermiticity：

$$
D_W^\dagger=\gamma_5D_W\gamma_5.
$$

这说明可构造 Hermitian kernel `gamma5*D`，但不表示原始 Wilson/Clover 算子本身 Hermitian。solver 选择、左右预条件和 dagger 语义必须与此一致。

自由场动量空间形式为

$$
D_W(p)=m_0+\sum_\mu(1-\cos p_\mu)
+i\sum_\mu\gamma_\mu\sin p_\mu.
$$

必须执行：

| 检查 | 预期 |
|---|---|
| 自由场 `U=1` | 与直接 stencil 或动量空间一致 |
| `c_sw=0` | Clover 路径退化为 Wilson |
| 大质量极限 | 谱隙扩大，迭代通常减少 |
| 临界质量 | 条件数恶化，不能把 solver 变慢误判为算子错误 |
| 局部规范变换 | 先平移场再做算子等于先做算子再变换 |
| `gamma5` 关系 | 相对误差处于 dtype 舍入量级 |

条件数采用

$$
\kappa_2(A)=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.
$$

对 `D^dagger D`，条件数为原条件数的平方。源码和报告中必须明确当前求解的是 `D`、`D^dagger D`、full operator 还是 Schur operator。

## 5. 奇偶 Schur 与残差语义

### 5.1 精确 block elimination

将 full 场按 parity 排列：

$$
D=
\begin{pmatrix}
A_e & B_{eo}\\
B_{oe} & A_o
\end{pmatrix},
\qquad
b=
\begin{pmatrix}
b_e\\
b_o
\end{pmatrix}.
$$

消去 odd 分量得到 even Schur 补

$$
S_e=A_e-B_{eo}A_o^{-1}B_{oe},
\qquad
b_e^{S}=b_e-B_{eo}A_o^{-1}b_o,
$$

恢复 odd 分量为

$$
x_o=A_o^{-1}(b_o-B_{oe}x_e).
$$

Clover 情况下 `A_p=I+C_p`；若使用 `D=I-kappa H`，hopping block 包含 `-kappa`，Schur 项中的符号和 `kappa^2` 必须由具体归一化推导。

### 5.2 Schur 语义分类

| 名称 | 定义 | 适用路径 | 主要风险 |
|---|---|---|---|
| full | 完整 `D` 同时作用于两种 parity | full Galerkin、full residual、Strict hierarchy | 存储和 apply 成本高 |
| asymmetric Schur | `A_p-B_pq A_q^-1 B_qp` | MATPC、BiCGStab、GCR、FGMRES | 非 Hermitian，左右次序固定 |
| symmetric Schur | 先做对称缩放或 `A^-1/2` | 希望保留 Hermitian 结构 | 缩放、右端和重建必须成对 |
| coarse-PC | `X_c^-1 D_c` | 粗层 PC apply | 不能和 fine Schur 混用 |

### 5.3 真残差

MATPC 的递推残差只属于压缩子系统。可靠停止条件必须由 full operator 重算：

$$
r_{\mathrm{true}}=b-Dx,
\qquad
\rho_{\mathrm{rel}}=
\frac{\lVert r_{\mathrm{true}}\rVert_2}{\lVert b\rVert_2}.
$$

右端为零时直接返回零解，避免 `1/||b||`。Strict 单 rank 主循环使用相对判据

$$
\lVert r_n\rVert_2^2
<
\mathrm{atol}^2\lVert b\rVert_2^2,
$$

并在预设周期执行 reliable-update，重算 full residual。多 rank 和不同入口是否刷新必须以当前源码为准，不能用单 rank 行为外推。

Schur 求解最小流程：

```text
输入 full b=(b_e,b_o), onset A_e,A_o, hopping B_eo,B_oe
1. y_o = A_o^{-1} b_o
2. b_e^S = b_e - B_eo y_o
3. 解 S_e x_e = b_e^S
4. x_o = A_o^{-1}(b_o - B_oe x_e)
5. 独立重算 r_true = b - D x
6. 只有 full true residual 达标才报告收敛
```

## 6. Krylov 方法与平滑器

### 6.1 算法选择

| 算法 | 算子条件 | 成本 | 常见失败 | 当前状态 |
|---|---|---|---|---|
| CG | SPD | 1 次 apply 和少量归约 | 非 SPD、负分母 | 实现/参考 |
| 多移位 CG | SPD 加标量移位 | 共享 Krylov | 非标量局部项、停止不一致 | Python 实现 |
| BiCG | 非 Hermitian 且需 dagger | 双 matvec/归约 | shadow breakdown | 参考 |
| CGS | 非 Hermitian | 低成本 shadow | 非正规矩阵上误差放大 | 局部正交参考 |
| BiCGStab | 一般非 Hermitian | 2 matvec/迭代 | `rho`、`omega` breakdown | 生产实现 |
| GMRES | 一般非 Hermitian | 长基和归约 | 内存增长、失正交 | 参考/部分实现 |
| FGMRES | 变化预条件器 | 保存 `Z` 与 `V` | 预条件返回异常 | Strict 生产路径 |
| GCR | 一般非 Hermitian | 方向历史和内积 | 历史大、失正交 | QUDA 参考 |
| CA-CG/CA-GCR | 块 Krylov | 少全局同步 | 块条件数恶化 | CA-CG 实现 |
| MR | 平滑或短迭代 | 固定短递推 | dagger 方向约定错误 | 生产 smoother |
| Chebyshev | 已知谱区间 | 固定多项式 | 谱估计错误 | 模式可切换 |

### 6.2 MR 平滑

最小残差步为

$$
\alpha=\frac{\langle v,r\rangle}{\langle v,v\rangle},
\qquad
x\leftarrow x+\alpha r,
\qquad
r\leftarrow r-\alpha v.
$$

实际内核可能使用 `A r` 或 `A^dagger r`。使用前必须核对 `matvec_dag` 和 inner product 的复共轭位置。

### 6.3 BiCGStab

常用递推为

$$
\begin{aligned}
\rho_k&=\langle \tilde r_0,r_k\rangle,\\
\beta_k&=\frac{\rho_k}{\rho_{k-1}}\frac{\alpha_{k-1}}{\omega_{k-1}},\\
p_k&=r_k+\beta_k(p_{k-1}-\omega_{k-1}v_{k-1}),\\
v_k&=Ap_k,\\
\alpha_k&=\rho_k/\langle\tilde r_0,v_k\rangle,\\
s_k&=r_k-\alpha_kv_k,\\
t_k&=As_k,\\
\omega_k&=\langle t_k,s_k\rangle/\langle t_k,t_k\rangle,\\
x_{k+1}&=x_k+\alpha_kp_k+\omega_ks_k,\\
r_{k+1}&=s_k-\omega_kt_k.
\end{aligned}
$$

必须保护所有分母、`rho` 和 `omega`，检测 NaN/Inf，并在可靠更新点用 full operator 刷新残差。

### 6.4 FGMRES

Arnoldi 关系为

$$
AV_m=V_{m+1}\bar H_m.
$$

FGMRES 允许每步预条件器变化：

$$
z_j=M_j^{-1}v_j,
\qquad
w_j=Az_j,
\qquad
x_m=x_0+Z_my.
$$

MG hierarchy、SAP、内层步数或 warm state 变化时，预条件器不是固定线性算子，必须使用 FGMRES 或 flexible-GCR。Strict 实现使用右预条件，并在 restart 边界重算 full true residual。

若 Arnoldi 历史长度为 `m`，融合 FGMRES 工作区量级为

$$
B_{\mathrm{work}}=(2m+5)B_f+2B_c,
$$

其中 `B_f` 是 compact fine 向量，`B_c` 是 full first-coarse 向量。

### 6.5 正交化与 breakdown

MGS 一步为

$$
w=Av_j,
\quad
h_{ij}=\langle v_i,w\rangle,
\quad
w\leftarrow w-h_{ij}v_i,
\quad
h_{j+1,j}=\lVert w\rVert,
\quad
v_{j+1}=w/h_{j+1,j}.
$$

局部 null-space 正交化可使用 CGS、MGS 或 block QR。验收不能只看 setup 成功，而应报告

$$
\epsilon_{\mathrm{orth}}=
\max_{\mathcal A}
\left\lVert V_{\mathcal A}^\dagger V_{\mathcal A}-I\right\rVert.
$$

分布式归约改变浮点求和顺序时，不应要求逐 bit 相同；必须比较残差量级、真残差门和物理结果。

---

## 7. MultiGrid、Galerkin 与 Strict coarse operator

### 7.1 粗空间的物理目标

轻夸克质量使 Dirac 算子接近奇异。Krylov 迭代对低频模态的收敛随体积恶化，即 critical slowing down。MG 将误差拆为

$$
e=e_{\mathrm{high}}+e_{\mathrm{low}}.
$$

平滑器压低高频误差，粗空间表示低频误差。设近零模集合 `B_l` 近似满足

$$
A_lB_l\approx 0.
$$

`B_l` 不是精确核，而是低模空间的数值近似。经过 aggregate 重排和局部正交化后得到延拓 `P_l`，通常取

$$
R_l=P_l^\dagger,
\qquad
P_l^\dagger P_l=I.
$$

粗算子由 Galerkin 投影定义：

$$
A_{l+1}=R_lA_lP_l.
$$

对非 Hermitian 或非正规算子，`P P^dagger != I`。粗空间不能替代细层单位算子，只负责消除 `P` 可表示的低模误差。

### 7.2 显式矩阵为何不可行

若粗自由度为 `E`、粗格点数为 `N_c`，显式稠密矩阵需要

$$
O\left((EN_c)^2\right)
$$

个复数。一个 `N_c≈49152`、`E=24` 的层级有约 1179648 个 coarse 自由度；c64 稠密矩阵约 11.1 TB，c128 约 22.3 TB。生产实现必须保存有限支撑 stencil，并以 matrix-free 或打包 block 资产应用。

### 7.3 粗空间质量与成本

对 aggregate 内的基 `V_A`，检查

$$
\epsilon_{RP}=\lVert RP-I\rVert,
\qquad
\epsilon_{\mathrm{orth}}=
\max_A\lVert V_A^\dagger V_A-I\rVert.
$$

只报告 null-vector 数量不足以证明粗空间质量。当前质量诊断曾给出近零模指标 `||S v||/||v||≈0.31-0.46`，说明向量可用但纯净度有限。

粗层一次 apply 的量级成本为

$$
C_{\mathrm{coarse}}
\sim
V_c(C_XE^2+8C_YE^2)
+C_{\mathrm{reduce}}N_{\mathrm{global\ dot}}.
$$

setup 还包含 null-vector 生成、局部正交、Galerkin probes、onsite inverse 和缓存写入。粗化比增大可减少 `V_c`，但 block 过大会降低 transfer 质量和并行度。增加 `E` 会以 `E^2` 提升粗算子成本，不能只增加 null-vector 数。

### 7.4 Strict split

PyQCU 正式主线为 Strict full-coarse MG。细层使用目标 parity compact 表示，粗层保留完整几何和 full operator 语义。

$$
D_l=X_l+H_l,
$$

其中 `X_l` 是零位移 onsite block，`H_l` 是轴邻近 hopping。左预条件为

$$
\widehat D_l=X_l^{-1}D_l=I+\widehat H_l,
\qquad
\widehat H_l=X_l^{-1}H_l.
$$

粗层算子为

$$
D_{l+1}=R_l\widehat D_lP_l
=R_l(X_l^{-1}D_l)P_l.
$$

运行期资产中，raw link 定义为

$$
Y^{f}_{l,\mu},Y^{b}_{l,\mu},
$$

preconditioned link 为

$$
\widehat Y^{f}_{l,\mu}=X_l^{-1}Y^{f}_{l,\mu},
\qquad
\widehat Y^{b}_{l,\mu}=Y^{b}_{l,\mu}X_l^{-\dagger}.
$$

forward 和 backward 的预条件顺序不同。raw `Y` 主要用于 setup 和诊断；solve 热路径使用 `Yhat`、`X` 和 `X^-1`。

### 7.5 最近邻支持与 fail-closed

Strict coarse stencil 只允许 onsite 和一格 hop：

$$
D_c=X_c+\sum_\mu
\left(
Y_c^{+\mu}T_{+\mu}
+Y_c^{-\mu}T_{-\mu}
\right).
$$

若 Galerkin probe 发现更远 displacement，必须显式失败。不能把宽 stencil 截断成最近邻后继续求解。

### 7.6 V-cycle

一次 V-cycle 为

$$
\begin{aligned}
x_l^{(1)}&=S_l^{\nu_{\mathrm{pre}}}(A_l,b_l,x_l^{(0)}),\\
r_l&=b_l-A_lx_l^{(1)},\\
b_{l+1}&=R_lr_l,\\
e_{l+1}&=\operatorname{Vcycle}(l+1,b_{l+1},0),\\
x_l^{(2)}&=x_l^{(1)}+P_le_{l+1},\\
x_l^{\mathrm{out}}&=S_l^{\nu_{\mathrm{post}}}(A_l,b_l,x_l^{(2)}).
\end{aligned}
$$

最粗层调用专用 coarse solver。外层右预条件 FGMRES 中，V-cycle 返回 `M^-1 r`，FGMRES 必须保存每一步的 `z_j` 并累加 `sum z_j y_j`。

### 7.7 Setup 与资产生命周期

MG setup 至少包含三类资产：

| 类别 | 内容 | 失效条件 |
|---|---|---|
| transfer | null vectors、aggregate、局部正交基、`P/R` | gauge、质量、几何或 blocking 变化 |
| operator | `X`、raw `Y`、`Yhat`、33-tensor stencil | 层级、精度、算子或资产 identity 变化 |
| solver state | 粗层分解、workspace、warm initial guess | hierarchy 或重启协议变化 |

warm start 只改变初值 `x0`，不改变 `A`、`b` 或 stopping criterion。切换 gauge、质量、边界、精度或层级后必须显式失效缓存。

### 7.8 Galerkin 构造

单列 probe 会为每列执行 full-field matvec，成本很高。当前实现使用：

| 策略 | 作用 |
|---|---|
| site/column batch | 一次处理多个 probe，摊薄 Python/C++ 同步 |
| colored source | 不冲突源同批执行，降低峰值内存 |
| batched gather/scatter | 预计算索引和 blocked basis 布局 |
| 原子累加 | 多 fine point 并发写同一 coarse block |
| 缓存 identity | 防止错误复用旧资产 |

历史 c128 冒烟中，将 canonical block staging 从逐点 Python 循环改为批量张量操作后，`K=1` 从 `44.86 s` 降到 `16.67 s`，约 `2.69x`；进一步按源 parity 分桶后，大格 c128 C=4/K=16 冷 setup 从约 `974 s` 降到 `585.28 s`。

## 8. QUDA 粗层资产、矩阵顺序与 MATPC

### 8.1 自由度与几何

fine Wilson/Clover 每站点自由度为

$$
e=N_s^fN_c^f=4\cdot3=12.
$$

coarse spin 为 `N_s^c=2`，null-vector 数为 `n_v`，因此

$$
E=N_s^cn_v.
$$

当 `n_v=12` 时 `E=24`。`X`、`Y`、`Yhat` 是 dense `E x E` 矩阵，不是 `SU(3)` link。

block `b=(b_x,b_y,b_z,b_t)` 给出

$$
X_\mu=\lfloor x_\mu/b_\mu\rfloor,
\qquad
\Lambda_c=(L_x/b_x,L_y/b_y,L_z/b_z,L_t/b_t).
$$

checkerboard parity 为

$$
p(x)=(x+y+z+t)\bmod2.
$$

hopping 改变 parity，onsite 不改变。full coarse asset 保留全格；MATPC 输入才压缩一半。固定前三维时，parity compact 时间坐标为

$$
t_{\rm full}=2t_{\rm half}+[(p-x-y-z)\bmod2].
$$

### 8.2 Transfer

aggregate 内正交基记为 `V`：

$$
(P\phi)_{s_fc_f}(x)=
\sum_{s_c,j}V_{s_fc_f,s_cj}(x)\phi_{s_cj}(X(x)),
$$

$$
(R\psi)_{s_cj}(X)=
\sum_{x\in A_X}\sum_{s_f,c_f}
V_{s_fc_f,s_cj}^*(x)\psi_{s_fc_f}(x),
\qquad
R=P^\dagger.
$$

每个 aggregate 内满足

$$
V_X^\dagger V_X=I_E,
\qquad
P^\dagger P=I_{\rm coarse},
\qquad
PP^\dagger\ne I_{\rm fine}.
$$

### 8.3 `X/Y`

forward link 的构造为

$$
Y_\mu^{f}(X)
=-\kappa\sum_{x\in A_X}
(A V_X(x))^\dagger U_\mu(x)V_{X+\hat\mu}(x+\hat\mu).
$$

backward storage block 为

$$
Y_\mu^{b}(X-\hat\mu)
=-\kappa\sum_{x\in A_X}
V_X(x)^\dagger U_\mu^\dagger(x-\hat\mu)
A V_{X-\hat\mu}(x-\hat\mu).
$$

若 `-\kappa` 被延后到 dslash 的 `dim_collapse`，比较缓存时必须记录系数是否已吸收。

full Clover onsite block 为

$$
X_C(X)=\sum_{x\in A_X}V_X(x)^\dagger C(x)V_X(x).
$$

Wilson-like 分支还会执行

$$
X\leftarrow X+I_E.
$$

coarse-to-coarse 重新投影当前层 `X_l`，不是再次读取 fine Clover 项。

### 8.4 Raw coarse action 与 storage

QUDA coarse kernel 的 storage 关系为

$$
out(x)=
\sum_\mu Y_{-\mu}(x)in(x+\mu)
+Y_\mu^\dagger(x-\mu)in(x-\mu).
$$

用 action 目标点 `q` 写 backward 项：

$$
Y_\mu^{b}(q-\hat\mu)^\dagger z(q-\hat\mu).
$$

若 `-\kappa` 未吸收进 link，则

$$
D_cz=Xz-\kappa H_Yz.
$$

常见错误是把 backward link 存到 `q` 而不是 `q-mu`，或在错误一侧乘 onsite inverse。dense block 不可交换时，这类错误会直接改变结果。

### 8.5 `Yhat` 与预条件

由

$$
X^{-1}(q)Y_\mu^{b}(q-\hat\mu)^\dagger
=
\left[
Y_\mu^{b}(q-\hat\mu)X^{-\dagger}(q)
\right]^\dagger
$$

得到

$$
\widehat Y_\mu^{f}(X)=X^{-1}(X)Y_\mu^{f}(X),
\qquad
\widehat Y_\mu^{b}(X-\hat\mu)
=Y_\mu^{b}(X-\hat\mu)X^{-\dagger}(X).
$$

实现顺序为 batch inverse `X`、交换双向 `Y` ghost、计算 backward `Y X^-dagger`、计算 forward `X^-1 Y`、注入 backward bulk，并只交换 forward ghost。

### 8.6 MATPC 与 prepare/reconstruct

按 parity 排列：

$$
D=
\begin{pmatrix}
A_p&D_{pq}\\
D_{qp}&A_q
\end{pmatrix}.
$$

compact solve 与恢复为

$$
x_q=A_q^{-1}(b_q-D_{qp}x_p),
$$

$$
S_p=A_p-D_{pq}A_q^{-1}D_{qp},
\qquad
M_p=A_p^{-1}S_p,
$$

$$
b_p^{\rm prep}
=A_p^{-1}b_p
-(A_p^{-1}D_{pq})A_q^{-1}b_q,
$$

$$
x_q=X_q^{-1}b_q-(X_q^{-1}D_{qp})x_p.
$$

coarse-PC dslash 只应用 `Yhat` hopping；onsite `X` 只供 prepare、reconstruct 和 raw path。MATPC 输入必须是目标 parity compact field，full field 应被拒绝。

### 8.7 Cycle、smoother 与外层 solver

| 组件 | QUDA 参考语义 | PyQCU 对应 |
|---|---|---|
| 外层 | GCR | 右预条件 FGMRES |
| smoother | CA-GCR、MR、Chebyshev | MR、CG、Chebyshev、CA-GCR 模式 |
| coarse solve | CA-GCR/BiCGStab | BiCGStab/FGMRES |
| cycle | V/W/F/K | V 为主，W/F/K/BiCGStab(l) 模式可切换 |

比较时必须统一：

| 参数 | 必须一致 |
|---|---|
| coarse-tol | 粗层 stopping tolerance |
| `nu_pre`、`nu_post` | 前、后平滑次数 |
| 中间层 maxiter | 每层递归预算 |
| 最粗层 maxiter | 最粗层预算 |
| restart | 外层 Krylov 历史长度 |
| initial guess | zero 或 warm start |
| true residual gate | full operator 相对残差 |

不同 cycle 或不同内层预算下的加速比不能合并。

## 9. CUDA/C++ 实现

### 9.1 四层软件结构

| 层 | 职责 |
|---|---|
| Python | geometry、field、cache identity、solver orchestration |
| Cython | 指针提取、GIL 释放、C API 调用 |
| C++ | hierarchy、workspace、MPI、资产和生命周期 |
| CUDA kernel | Dslash、coarse apply、smoother、Krylov、reduction、halo |

### 9.2 细层 Dslash

Wilson Dslash 每个方向读取正、反邻居。`1±gamma_mu` 将四分量旋量投影到两个独立分量，因此内核先做 spin projection，再做 `SU(3)` 矩阵乘，可降低寄存器和内存压力。

优化包括：

| 优化 | 效果 |
|---|---|
| 时空维放最后 | 相邻点连续，提升合并访存 |
| spin projection 前置 | 减少无效矩阵乘 |
| Clover 批量 inverse | 避免逐点通用 inverse |
| 复数向量化 | 降低 kernel 和内存事务数 |
| cuBLAS/融合 kernel 混合 | 对小向量减少启动开销 |

### 9.3 粗层延迟优化

粗层是 latency-bound，而不是算力-bound。主要措施为：

| 措施 | 说明 |
|---|---|
| device scalar | `alpha`、`beta`、`omega`、`rho` 尽量留在设备端 |
| dot-pair/dot-many | 合并多个 inner product 和 MPI 归约 |
| kernel fusion | 合并更新、正交化和 prolongation 加法 |
| CUDA Graph | 对固定粗层迭代段降低启动开销 |
| reduced host sync | 只在稳定检查点读取 host scalar |
| persistent workspace | 跨 solve 复用固定布局工作区 |

当前源码还使用分块 partial reduction。多 rank 时，局部 partial 结果通过 `MPI_Allreduce` 转成全局标量。源码中仍有个别“single-rank”注释残留，但实际路径已包含 vector halo、link halo、全局 dot、`dot_pair` 和 `dot_many`，应以可执行分支为准，而不是注释文字。

### 9.4 缓存

Strict runtime cache schema 为

```text
pyqcu.strict-runtime-cache
schema_version=2
digest=sha256(pyqcu-logical-tensor-v1)
```

缓存包含：

| transition asset | 内容 |
|---|---|
| fine basis | blocked `V` |
| preconditioned links | `Yhat` |
| onsite pair | `X`、`X^-1` |
| recursive basis | 后续层 `V` |

写入使用：

| 机制 | 目的 |
|---|---|
| 单 HDF5 handle | 避免逐 dataset 重建 |
| 8 MiB 分块 | 限制 host/device 临时占用 |
| logical tensor digest | 防止 shape 相同但内容改变 |
| `identity_json` | 绑定 geometry、dtype、operator 和 hierarchy |
| `metadata_json` | 记录版本和运行元数据 |
| `manifest_json` | 记录每个 tensor 的 shape、dtype、bytes 和 digest |
| writing -> complete | 未完成文件不可作为 cache hit |
| 临时文件加 no-clobber link | 原子发布，不覆盖已有 identity |

加载时先校验 root attrs、JSON canonical encoding、manifest 和每个 tensor digest，再分块 H2D。任何缺失、损坏、非 canonical JSON 或 identity 不符都返回结构化 cache miss。

### 9.5 HDF5 与外存布局

小文件使用 `save_tensor_h5`、`load_tensor_h5`，每次调用独立 File 句柄。嵌套 dict 使用 `save_dict_h5`、`load_dict_h5`，单句柄一次写完全部内容并用临时文件加 `os.replace` 原子替换。

分布式全局文件使用 h5py `driver='mpio'`。当前矩阵 collector 中，多 rank 输入必须按 rank slab 读取：

| 输入 | 分布式读取 |
|---|---|
| gauge/source | checkerboard-compressed 的 `x,y,z,t/2` hyperslab |
| full null vector | `x,y,z,t` hyperslab |
| single rank | 保持全量语义 |

双 rank 小格 round-trip 的 gauge、source、null `max_abs` 为 0。禁用全量读取是避免每个 rank 复制数 GiB host memory 的硬约束。

## 10. 分布式 halo 与 overlap

### 10.1 为什么必须区分三层 halo

当前实现包含：

| halo | 作用 |
|---|---|
| fine vector halo | fine compact/full 向量面交换 |
| coarse vector halo | full coarse vector 的 parity 和方向面交换 |
| axis link halo | `mu` 方向 backward link face 交换 |

粗层算子不是直接读 rank-local 周期邻居，而是拆为

```text
local periodic base
- wrong self-wrap
+ true remote contribution
```

当前源码在 `StrictVectorHalo` 和 `StrictAxisLinkHalo` 中执行 face pack、MPI 和 device ghost 管理。只有当 `params[_GRID_*]` 的乘积等于 `MPI_COMM_WORLD` size，且 `NODE_RANK/NODE_SIZE` 一致时才继续；否则 fail-closed。

### 10.2 P/R coarse halo

aggregate 不跨 rank 时，`P/R` 本身是局部映射。需要通信的是 coarse link 在 rank 边界的 storage 位置：

| 操作 | 边界处理 |
|---|---|
| restrict | coarse 输入置为 halo-valid |
| prolongate | full coarse 输出做边界处理，fine compact 输出复用 Wilson halo |
| `Yhat` backward | 从邻居 rank 的对应 face 读取 `q-mu` storage |

### 10.3 分布式 fused FGMRES

外层右预条件 FGMRES 的每一步 inner product 必须全局：

| 操作 | 实现 |
|---|---|
| dot | 局部 cublas dot 后做全局实数归约 |
| dot pair | 两个内积合并为 count=4 的归约 |
| dot many | Arnoldi 的多个系数一次 Allreduce |

历史缺陷是把全局和只写回 host，而正交化 kernel 继续消费 rank-local partial。结果是 restart 越大越不收敛，例如 restart 20 时 true relative residual 为 `1.9e-2`；修复后同一算例 17 次外层迭代降到 `6.7e-7`。当前 `dot_many` 在 Allreduce 后显式刷新 device coefficient，保证 Hessenberg 和 Arnoldi basis 使用同一组全局系数。

### 10.4 分布式 Galerkin setup

setup 需要全局位移和全局 stencil：

| 环节 | 实现 |
|---|---|
| roll | 每个轴只交换两个 face slab，未分解轴退化为本地 roll |
| stencil | fine 算子对分解维扩一层 rank ghost，再在 padded 格点上做原周期算子 |

验证结果：

| 项目 | 结果 |
|---|---|
| full/compact coarse operator，c64 | 约 `1e-7`，为舍入量级 |
| full/compact coarse operator，c128 | 约 `1e-16` |
| 1/2/4 rank，多轴分解 | 与全局参考一致 |
| setup asset | 2/4 rank 内部点和边界点与单 rank 参考按位一致 |
| 分布式 true residual | `<1e-6` |

### 10.5 Nonblocking overlap

Strict c64 默认启用非阻塞 vector halo 与 local/remote 分区：

$$
\text{active-face pack}
\parallel
\text{local base kernel}
\rightarrow
\text{nonblocking MPI}
\parallel
\text{local tail}
\rightarrow
\text{remote correction}.
$$

输入就绪 event 从主 stream 记录。交换流完成 pack、pinned-host staging 和非阻塞 MPI。主 stream 先计算不含远端项的局部核，再由合并 correction kernel 加 remote forward/backward。

该路径改变 floating-point accumulation order：

| 类型 | 默认 |
|---|---|
| complex64 distributed | overlap 开启 |
| complex128 distributed | overlap 关闭 |
| complex128 显式实验 | `PYQCU_STRICT_OVERLAP_C128=1` |

complex128 大格三层的实验路径曾从 32 个外层迭代退化到 451 个，不能默认开启。当前源码默认 `PYQCU_MPI_OVERLAP=true`，但 complex128 还需额外的 C128 开关。

MPI transport 由 `PYQCU_MPI_DEVICE_AWARE` 控制：

| 值 | 行为 |
|---|---|
| `auto` | 使用 MPI vendor capability，可 CUDA-aware 则直发 device buffer |
| `0/off` | 强制 pinned-host staging |
| `1/on` | 请求 CUDA-aware，但 capability 不支持时仍拒绝 device pointer |

### 10.6 已修复的真实缺陷

| 缺陷 | 后果 | 修复 |
|---|---|---|
| vector halo slot stride 与 face length 混用 | ghost 分量整体错位 | 以最大 face 长度统一 slot stride |
| correction 符号固定 | MATPC 两次 hop 中一次符号翻转 | correction 符号与 base 累加方向一致 |
| `dot_many` 全局系数未回写 device | restart 大时失正交、不收敛 | Allreduce 后 H2D 刷新 |
| coarsest cooperative kernel 与 MPI collective 冲突 | 多 rank 非法或死锁 | 多 rank 回退 host-loop 全局归约 |
| 设备强制绑定 0 | rank 1 跨设备指针访问 | 使用 `PYQCU_MPI_DEVICE_ID` 或 local rank |
| face pack 无上界保护 | 尾线程越界 | 十个 1D/2D pack kernel 在索引解码前检查 |
| link halo 每步重打包 | 静态 link 重复通信 | 按 pointer identity 和 `(parity,dim)` 缓存 |

## 11. 多 GPU、MPI 与输入一致性

### 11.1 一线程一卡

`MultiGpuMultigrid` 的默认语义是：

```text
N threads -> N GPU contexts
each thread -> independent params/argv/set_ptrs
each thread -> same logical problem or independent problem
```

当所有线程求同一输入时，使用线程间解的一致性验证实现。其吞吐扩展不是单 RHS 域分解。

历史测试中，双 P100 两任务吞吐扩展约 `1.951x`，但单任务最慢线程延迟比单 P100 约 `0.976x`。该结果应表述为吞吐收益，不能表述为单解加速。

### 11.2 rank 与 device 绑定

当前策略为：

| 场景 | 设备选择 |
|---|---|
| 显式 `PYQCU_MPI_DEVICE_ID` | 使用显式索引 |
| 多 rank、多可见设备 | fallback 到 local rank |
| 单可见设备 | device 0 |
| 一进程一卡 | 由 `CUDA_VISIBLE_DEVICES` 隔离 |

Python collector 在 C++ 调用前把 Torch 实际 device index 写入环境变量，避免 Torch 逻辑顺序与 physical/local rank 顺序不同。

### 11.3 输入身份

公平比较必须持有：

| 对象 | 必需记录 |
|---|---|
| gauge | 文件名、seed、shape、dtype、逻辑摘要 |
| source | dataset、shape、dtype、逻辑摘要 |
| null vector | canonical、数量、coarse spin、full/compact 语义 |
| QIO/cache | path、identity、manifest、metadata 和 tensor digest |
| binary | `libqcu.so` 路径、SHA256、实际 cubin arch |
| QUDA | library path、precision、reconstruct、MG double 开关 |
| runtime | device UUID、rank/grid、threads、tuning cache |

只有 config 与 input bundle 均一致，combined 单元才可标记 `fair=true`。

## 12. 误差、显存与性能预算

### 12.1 误差分解

总误差可写为

$$
e_{\mathrm{total}}
\lesssim
e_{\mathrm{disc}}
+e_{\mathrm{setup}}
+e_{\mathrm{alg}}
+e_{\mathrm{round}}
+e_{\mathrm{comm}}.
$$

| 项 | 来源 |
|---|---|
| `e_disc` | Wilson 的 `O(a)` 或 Clover 剩余离散误差 |
| `e_setup` | null-space、局部正交和 Galerkin 截断 |
| `e_alg` | Krylov 未完全收敛 |
| `e_round` | mixed precision、归约和递推漂移 |
| `e_comm` | halo、MPI reduction 和布局错误 |

减小 `tol` 只能修复最后一项的一部分，不能修复错误 gamma、parity 或 backward link。

### 12.2 可靠更新

mixed precision 或长递推中，应：

1. 用低精度推进；
2. 每隔固定迭代访问 full operator `r=b-Ax`；
3. 用 true residual 重置递推或重启；
4. 记录刷新前后残差比，检测 NaN/Inf 和突跳。

### 12.3 显存判据

最终矩阵使用 device-wide sampler。该值可能包含同设备其他进程，因此：

| 指标 | 用途 |
|---|---|
| device-wide observed max | 容量门限和调度保护 |
| process-exclusive allocator peak | 用于代码内存优化 |

两者不能混为同一量。`torch.cuda.empty_cache()` 只释放 allocator cache，不证明 hierarchy 或 `set_ptrs` 已释放。

最终精确单元的最大观测结果为：

| 侧 | 最大 full true residual | 最大 device-wide observed memory |
|---|---:|---:|
| PyQCU | `7.40e-7` | `11907.3 MiB` |
| QUDA | `7.97e-7` | `23063.3 MiB` |

## 13. 失败定位顺序与工程反模式

### 13.1 推荐定位顺序

```text
operator -> parity -> transfer -> coarse apply
-> one V-cycle -> outer solver -> full matrix -> performance attribution
```

在 free-field Dslash 尚未通过前修改 MG 层数或 smoother，会把多个误差源混在一起。

| 现象 | 首先检查 | 含义 |
|---|---|---|
| forward/backward 方向错误 | link index、边界 | transport 或 neighbor 反了 |
| gamma5-Hermiticity 失败 | gamma basis、Clover dagger、乘法次序 | 算子约定不一致 |
| Schur 收敛但 full residual 大 | prepare/reconstruct、onsite inverse | 压缩解未恢复 |
| 出现非最近邻 strict support | transfer、action | strict ABI 不适用 |
| 递推残差小但 true residual 不降 | mixed precision、reliable update、归约 | 数值漂移或 residual 混用 |
| 单线程正确、多线程错误 | params/argv/set_ptrs 是否共享 | scratch/index 跨线程复用 |
| rank 增加后漂移大 | global dot、halo、layout | 通信或归约语义错误 |
| 显存持续增长 | hierarchy 引用、HDF5 句柄、workspace | allocator cache 不等于 leak |

### 13.2 已记录反模式

| 反模式 | 风险 |
|---|---|
| `torch.Tensor(<scalar>)` | 新版 PyTorch 抛 TypeError，应写成 `torch.Tensor([x])` |
| 对 `tools.norm()` 再 `.item()` | `norm` 已返回 float |
| 复数 `operator*=` 覆盖前仍读 `_data.x` | 别名和覆盖顺序错误 |
| `cudaMallocAsync` 大小与 kernel 写大小不一致 | 越界或未初始化 |
| 裸 `except:` | 吞掉 KeyboardInterrupt 和真实错误 |
| 逐点 `torch.linalg.inv` | 应批量 inverse |
| 用对象做 truthy 判断 | `__len__` 或协议可能改变语义 |
| stout `nstep>1` 未每步更新 `U` | 数值路径错误 |
| `python_requires < 3.8` | 不再允许 |
| 未做大格收敛回归就重排 coarsest reduction | 可能让 FGMRES 停滞 |

---

## 14. 最终性能协议与覆盖

### 14.1 权威来源

最终性能只采用：

| 层级 | 来源 |
|---|---|
| 运行数据 | `git tag test27` |
| 正式矩阵 | `data/report_multigrid_comprehensive_20260928/final_protocol/` |
| 当前报告 | `docs/report_multigrid_quda_pyqcu_20260928.tex/.pdf` |
| 当前全文 | `docs/High-Performance Implementation of Multigrid Solver for Lattice QCD.tex/.pdf` |
| overlap 风险与 c128 回退 | `docs/report_multigrid_optimized_20260927.tex/.pdf` |

2026-09-27 报告中的 `72/72 exact`、`144/144 side`、MG `38/48` 和总中位 `1.346` 在当时的 600 秒口径内成立，但不是当前最终口径。2026-09-28 加入 85% device-wide 显存门限和 570 秒单实例时限后，正式结果为：

```text
66 exact combined units
132 exact side records
6 explicit capacity substitutions
```

因此不得把 `72/72` 继续写成当前完成数。

### 14.2 测试矩阵

| 轴 | 取值 |
|---|---|
| 实现 | QUDA 1.1.0；PyQCU Strict CUDA/C++ |
| 设备 | V100-SXM2-32GB 单卡；P100-PCIE-16GB 双卡 |
| c64 格点 | `8^3 x 16`、`16^4`、`16 x 32 x 32 x 48` |
| c128 格点 | `8^3 x 16`、`16^4`、`16 x 16 x 32 x 32` |
| trace | on、off |
| solver | BiCGStab reference、MG-2、MG-3 |
| phase | 1 cold、2 warmup、5 steady |
| 性能统计 | steady median，附带 MAD |
| true residual | `||b-Dx|| / ||b|| < 5e-6` |
| 调度限制 | 单 collector 不超过 570 s |
| 显存限制 | device-wide max 低于容量 85% |

固定物理和算法参数：

| 参数 | 值 |
|---|---:|
| mass `m0` | `0.05` |
| null-vector 数 `n_v` | 12 |
| block | `(2,2,2,2)` |
| coarse spin | 2 |
| coarse dof `E` | 24 |
| parity | odd-odd MATPC，target parity `p=1` |
| c64 coarse-tol | `0.003` |
| c128 coarse-tol | `3e-5` |
| P100 / V100 c64 `nu` | 1 |
| V100 c128 `nu` | 2 |

历史 coarse-tol `0.3` 只用于 isolation A/B 和快速迭代，不进入最终矩阵。

### 14.3 运行身份

| 设备 | 运行栈 | PyQCU binary | QUDA |
|---|---|---|---|
| V100 | PyTorch 2.10.0+cu128、CUDA 12.8、PyQUDA 0.10.54 | `cpp/cuda/qcu/libqcu.so`，`sm_70`，SHA256 `81ec6142...a670889` | `quda-double-install`，`sm_70`，QIO/QMP、MG-double、reconstruct 7 |
| P100 | PyTorch 2.7.1+cu118、CUDA 11.8、PyQUDA 0.10.54 | `data/build/libqcu_arch/sm60/libqcu.so`，`sm_60`，SHA256 `e29c1cf0...d0c20e1` | `quda-sm60-install`，`sm_60`，QIO/QMP、MG-double、reconstruct 7 |

QUDA tuning cache 固定在：

```text
data/quda-resource-sm70-20260927
data/quda-resource-sm60-20260927
```

P100 使用 `2x1x1x1` process grid，一线程一卡；V100 使用单 rank。QIO null-vector manifest 与二进制统一放在 `data/qio-matrix`。

### 14.4 Fail-closed 门禁

| 门禁 | 观察值 | 通过标准 |
|---|---:|---|
| combined 精确单元 | 66 | 66/66 |
| side record | 132 | 132/132 |
| 1 cold / 2 warmup / 5 steady | 132/132 | 全部完整 |
| config hash | 66/66 | 全部一致 |
| input bundle | 66/66 | 全部一致 |
| full true residual | 66/66 | 全部 `<5e-6` |
| trace-on MG residual curve | 44/44 | 全部存在 |
| MG level rows | 220/220 | 无缺失字段 |

## 15. 最终性能结果

### 15.1 分组中位时间比

图中和表中的比值统一为

$$
R=\frac{t_{\mathrm{QUDA,steady}}}{t_{\mathrm{PyQCU,steady}}}.
$$

`R>1` 表示 PyQCU 更快。09-28 报告的图注中有一处“PyQCU/QUDA”反向措辞笔误；公式、表头、坐标轴和全部相邻解释均采用 `QUDA/PyQCU`，本文以公式和实际数值为准。

![最终 66 单元与 22 个分组的 QUDA/PyQCU 中位时间比](High-Performance_Implementation_MG_assets/report_chart_assets/overlap_matrix.png)

| 设备 | 精度 | solver | trace | 样本 | median | min | max |
|---|---|---|---:|---:|---:|---:|---:|
| V100 | c64 | BiCGStab | off | 3 | 0.2818 | 0.2355 | 0.3288 |
| V100 | c64 | BiCGStab | on | 3 | 0.3213 | 0.1766 | 0.4102 |
| V100 | c64 | MG-2 | off | 3 | 2.5717 | 1.3999 | 7.1161 |
| V100 | c64 | MG-2 | on | 3 | 1.6785 | 1.0387 | 3.6403 |
| V100 | c64 | MG-3 | off | 3 | 4.7366 | 2.0284 | 8.1391 |
| V100 | c64 | MG-3 | on | 3 | 1.7444 | 1.2602 | 4.1878 |
| V100 | c128 | BiCGStab | off | 3 | 0.2580 | 0.2463 | 0.3573 |
| V100 | c128 | BiCGStab | on | 3 | 0.2397 | 0.2242 | 0.2998 |
| V100 | c128 | MG-2 | off | 3 | 2.0269 | 0.5401 | 7.5586 |
| V100 | c128 | MG-2 | on | 3 | 1.4515 | 1.0124 | 4.9719 |
| V100 | c128 | MG-3 | off | 3 | 3.2174 | 1.8447 | 14.6340 |
| V100 | c128 | MG-3 | on | 3 | 1.3822 | 0.9966 | 4.2991 |
| P100x2 | c64 | BiCGStab | off | 2 | 0.3416 | 0.2403 | 0.4429 |
| P100x2 | c64 | BiCGStab | on | 2 | 0.2368 | 0.2333 | 0.2404 |
| P100x2 | c64 | MG-2 | off | 2 | 1.7773 | 1.4305 | 2.1241 |
| P100x2 | c64 | MG-2 | on | 2 | 0.9341 | 0.8531 | 1.0150 |
| P100x2 | c64 | MG-3 | off | 2 | 1.7944 | 1.2758 | 2.3130 |
| P100x2 | c64 | MG-3 | on | 2 | 0.8282 | 0.7456 | 0.9107 |
| P100x2 | c128 | BiCGStab | off | 3 | 0.2284 | 0.2244 | 0.4334 |
| P100x2 | c128 | BiCGStab | on | 3 | 0.2400 | 0.2386 | 0.4383 |
| P100x2 | c128 | MG-2 | off | 3 | 1.2502 | 1.0349 | 1.2725 |
| P100x2 | c128 | MG-2 | on | 3 | 1.0420 | 0.8120 | 1.1716 |
| P100x2 | c128 | MG-3 | off | 3 | 1.1517 | 1.1061 | 1.4436 |
| P100x2 | c128 | MG-3 | on | 3 | 0.8843 | 0.6831 | 0.9181 |

关键结论：

| 集合 | 结果 |
|---|---|
| 全部 66 个精确单元 | 整体中位 `R=1.0250` |
| MG-2/3 全部 trace 模式 | `35/44` 有利 |
| MG-2/3 正式 trace-off | `21/22` 有利 |
| BiCGStab reference | `0/22` 有利，QUDA 全部更快 |
| V100 c64 MG-3 trace-off | 分组中位 `4.7366` |
| V100 c128 MG-3 trace-off | 分组中位 `3.2174` |
| 双 P100 MG-2/3 trace-off | 中位均大于 1 |
| trace-on 双 P100 部分分组 | 退到 1 以下 |

逐单元分布如下。实心符号为 trace-off，空心符号为 trace-on。极值只是单点，不是置信区间。

![66 个精确单元逐点时间比](High-Performance_Implementation_MG_assets/paper_assets/fig_ratio_distribution.png)

![trace-off 分组中位时间比](High-Performance_Implementation_MG_assets/paper_assets/fig_group_medians.png)

### 15.2 逐单元视图

![66 个 combined unit 的稳态时间比](High-Performance_Implementation_MG_assets/report_chart_assets/mg_speedup_units.png)

V100 的优势最明显，但 c128 MG-2 存在单点 `R=0.54`，c128 MG-3 存在单点 `R=14.63`。这证明结果对格点体积和 Krylov 轨迹敏感，不能只引用最大值。

### 15.3 结论措辞

可以主张：

1. PyQCU Strict MG 在典型 V100 MG 场景中与 QUDA 同量级，且有明显场景优势。
2. 正式 trace-off MG 的 `21/22` 有利支持 MG 专项性能结论。
3. 全部精确结果通过统一真残差门和显存门。

不能主张：

1. PyQCU 全部求解器优于 QUDA。
2. 双 P100 的所有 trace/level 组合都领先。
3. 当前结果可外推到任意 GPU、GPU 图、process grid 或千卡规模。
4. 当前标题中的 Hopper 优化是 Hopper GPU 实测。现有 CUDA 性能证据只覆盖 P100 `sm_60` 和 V100 `sm_70`。

## 16. 阶段归因

![trace-on MG 逐层阶段堆叠分解](High-Performance_Implementation_MG_assets/report_chart_assets/mg_stage_breakdown.png)

阶段份额中位数为：

| 实现 | pre | post | restrict | prolong | coarse | other |
|---|---:|---:|---:|---:|---:|---:|
| PyQCU | 2.1% | 2.0% | 0.5% | 0.5% | 43.9% | 50.9% |
| QUDA | 10.5% | 17.3% | 1.0% | 0.7% | 69.3% | 0.1% |

![两侧记录的阶段份额](High-Performance_Implementation_MG_assets/paper_assets/fig_stage_shares.png)

PyQCU 的 coarse solve 加 `other` 约 95%。这里的 `other` 包含 Arnoldi、fine MATPC、解更新、true-residual 刷新和同步；不同实现的层级映射不同，不能把同名阶段直接跨库相加。

trace-on 开销在不同格点上的历史实测包括：

| 配置 | trace-off | trace-on | 比值 |
|---|---:|---:|---:|
| c64 小格 MG-3 PyQCU | `0.043971 s` | `0.125649 s` | 约 `2.86x` |
| 同点 QUDA | `0.505706 s` | `0.560432 s` | 约 `1.11x` |
| c128 大格历史点 PyQCU | `2.065 s` | `3.31-3.43 s` | 约 `1.6x` |

因此 trace-on 只能用于归因，正式性能必须使用 trace-off。

## 17. 显存、残差与容量边界

### 17.1 峰值显存

![按设备、精度、MG 层数和 trace 模式的 device-wide 峰值显存](High-Performance_Implementation_MG_assets/report_chart_assets/mg_peak_memory.png)

| 侧 | 设备/精度 | 样本 | median MiB | max MiB |
|---|---|---:|---:|---:|
| PyQCU | P100/c64 | 12 | 1852.9 | 2134.9 |
| PyQCU | P100/c128 | 18 | 3150.9 | 7308.9 |
| PyQCU | V100/c64 | 18 | 2677.3 | 11907.3 |
| PyQCU | V100/c128 | 18 | 4005.3 | 9575.3 |
| QUDA | P100/c64 | 12 | 1934.9 | 2628.9 |
| QUDA | P100/c128 | 18 | 3800.9 | 11082.9 |
| QUDA | V100/c64 | 18 | 3587.3 | 23063.3 |
| QUDA | V100/c128 | 18 | 5227.3 | 15739.3 |

该表是 device-wide sampler，不声明为进程独占峰值。

### 17.2 残差

| 侧 | 样本 | median | max |
|---|---:|---:|---:|
| PyQCU | 66 | `1.85e-8` | `7.40e-7` |
| QUDA | 66 | `1.89e-8` | `7.97e-7` |

![全部 trace-on MG 的最细层真残差](High-Performance_Implementation_MG_assets/report_chart_assets/mg_finest_residual.png)

![最大真残差与 device-wide 峰值显存](High-Performance_Implementation_MG_assets/paper_assets/fig_residual_memory.png)

PyQCU 的曲线主要是 FGMRES Arnoldi least-squares estimate，QUDA 的曲线是 GCR iterated residual。两条曲线用于观察趋势，最终通过性只由独立 full-operator true residual 判定。

### 17.3 容量替代

6 个未进入精确矩阵的请求均为双 P100 c64 `16x32x32x48`：

| 请求 | 替代 | 原因 |
|---|---|---|
| L1 trace-off | `16^4` c64 L1 trace-off | 显存或时限 |
| L1 trace-on | `16^4` c64 L1 trace-on | 显存或时限 |
| L2 trace-off | `16^4` c64 L2 trace-off | 显存或时限 |
| L2 trace-on | `16^4` c64 L2 trace-on | 显存或时限 |
| L3 trace-off | `16^4` c64 L3 trace-off | 显存或时限 |
| L3 trace-on | `16^4` c64 L3 trace-on | 显存或时限 |

三次尝试结果：

| 路径 | 峰值 | 设备占比 | 结果 |
|---|---:|---:|---|
| 默认 overlap | 约 `15.9 GiB` | 93.1% | 超过 85% 门 |
| `PYQCU_MPI_OVERLAP=0` | 约 `14.8 GiB` | 86.2% | 超过 85% 门 |
| extended CPU staging | 约 `12.4 GiB` | 72.4% | MG-2 setup 超过 570 s |

替代结果只用于容量趋势，不能与原请求混入公平加速比。

## 18. 限制与下一步

### 18.1 当前限制

| 限制 | 证据与影响 |
|---|---|
| 总中位仅 `1.0250` | 整体优势较弱 |
| BiCGStab `0/22` | 非 MG 路径明显落后 |
| trace-on 有同步开销 | 双 P100 部分分组退化 |
| P100 c64 大格被替代 | setup 时限和显存超门 |
| c128 overlap 默认关闭 | 修改浮点顺序导致迭代轨迹敏感 |
| link halo 仍阻塞 | 只有 pointer identity 缓存 |
| 多 rank true-residual refresh 未启用 | 当前可靠更新能力有限 |
| c128 输入多为 c64 canonical asset | 不是完整 complex128 HDF5 链路 |
| 证据集中 V100 和双 P100 | 不能外推千卡或任意拓扑 |

### 18.2 优化优先级

1. 继续降低 coarse solve、全局归约和 true-residual 检查的同步成本。
2. 评估 persistent request、device-aware MPI 和可选 NVSHMEM。
3. 降低 setup 峰值显存，使 P100 c64 大格进入精确矩阵。
4. 对 c128 重新设计不破坏递推稳定性的 overlap 或 reproducible reduction。
5. 扩展 mixed-coarse precision，但必须先完成同精度性能对照。
6. 把当前协议扩展到更多 process grid 和更大 MPI 规模。

---

## 19. 统一演进时间线

| 日期 | 里程碑 | 当时结论 | 当前处置 |
|---|---|---|---|
| 2025-08-29 | DCU 4 卡、64-1024 进程应用测试 | BiCGStab 大规模强/弱扩展 | 历史能力证据，不与 `test27` 混算 |
| 2026-08-14 | 多线程多卡竞争修复、构建加速和早期 Pareto 分析 | 修复跨线程共享、构建约 `2.8x` | 工程历史，保留反模式 |
| 2026-08-18 | init 到 `stab27` 的变更审计和核心结构分析 | 盘点实现边界和依赖 | 被当前目录结构和 ABI 取代 |
| 2026-08-20 | `24^3x72` MG/BiCGStab 扫描 | 3L 最佳约 `1.107x` | 旧路径，已被后续 Strict 矩阵取代 |
| 2026-08-28 | CUDA/C++ MG 算法族 | MR、Chebyshev、CA-GCR、BiCGStabL、V/W/F/K | 保留为算法分支与历史性能 |
| 2026-08-29 | QUDA MG 复现链 `_1` 到 `_5` | 从 full 到 compact MATPC、多 MPI 拓扑 | 只保留 `_5` 和跨版本差异 |
| 2026-08-30 | 33 点 setup 优化、严格 cold-solve benchmark | 2L 对 1L 无稳定净收益 | 保留关键反例和显存数据 |
| 2026-09-06 | 首次 Strict 大格 QUDA 正式对照 | PyQCU `2.041653 s`，QUDA `2.094841 s` | 时间和 speedup 被 09-09 取代 |
| 2026-09-09 | device-side MR、融合 coarse BiCGStab、消融 | PyQCU `1.966795 s`，speedup `1.065065x` | 历史 Strict 单 rank 最终值 |
| 2026-09-11 | 数值线代、作用量和 MG 审计全文 | 统一数学语义与验证矩阵 | 作为理论参考，不覆盖动态性能 |
| 2026-09-13 | `X/Y/Yhat` 与 storage 专题 | 纠正 backward inverse site 和 PC dslash | 局部代数最终裁决 |
| 2026-09-16 | 分层 trace 和三层早期加速 | 旧 c64 三层 `4.3817x` | 已撤回 |
| 2026-09-17 | aligned 协议、double MG 修复、setup 优化 | c64 三层 `2.0052x` 等 | 历史 aligned 结果，不是最终 66 单元 |
| 2026-09-18 | 分布式 halo、P/R、global FGMRES、分布式 setup | 正确性到舍入水平 | 保留为分布式基础证据 |
| 2026-09-21 | 双 P100 修复、静态 halo cache、全矩阵快照 | `72/72`、MG `36/48` | 旧口径，保留失败优化 |
| 2026-09-27 | nonblocking overlap、c128 回退、600 秒矩阵 | `72/72 exact`、MG `38/48` | A/B 保留，最终覆盖被 09-28 取代 |
| 2026-09-28 | 85% 显存加 570 秒容量感知矩阵 | 66 exact、132 side、6 substitutions | 当前最终性能权威 |

## 20. 早期多线程、多卡与旧 MG 路径

### 20.1 多线程与多卡

早期问题集中在 GIL、CUDA device context 和共享 `params/set_ptrs`：

| 问题 | 修复 |
|---|---|
| Python 线程无法并行进入 C++ | Cython 指针提取后 `with nogil` |
| pxd 缺少 `nogil` 或符号冲突 | 使用 `qcu_api.pxd` 别名并补齐声明 |
| 多线程共享 scratch/index | 每线程克隆 `params/argv/set_ptrs` |
| 构建阶段反复重建 coarse stencil | 每层构建后立即建立下一层 matvec 算子 |
| 逐 probe Python 循环 | batch build 和批量 einsum |

这些改动给出历史构建加速约 `2.8x`，并形成当前一线程一卡约束：

```text
N threads -> N independent GPU contexts
one thread -> one device
one RHS -> not split across those threads
```

### 20.2 `24^3x72` 参数扫描

早期 test15 系列对 `24^3x72` c64 做 18 个 MG 配置扫描。所有版本共用同一组图，最终最有代表性的历史结论为：

| 项目 | 历史结果 |
|---|---|
| 2L speedup 范围 | `0.73-0.84` |
| 3L speedup 范围 | `0.98-1.107` |
| 3L 超过 1 的配置 | `6/9` |
| 最佳单点 | 3L `r30/ct1e3`，约 `1.107x` |
| MG 2L 时间 | 约 `3.5-3.8 s` |
| MG 3L 时间 | 约 `2.53-2.86 s` |
| L1 参考 | `2.808 s` |
| BiCGStab 参考 | `3.197 s` |
| cold 显存预算 | 约 `22 GB` |

该路径后来被奇偶 Schur、full coarse、33 点 stencil 和 Strict full-coarse 取代。早期曲线只用于说明“更多 V-cycle 不等于稳定收益”，不作为当前性能。

## 21. 2026-08-28 至 08-30：算法族、复现链和严格基准

### 21.1 2026-08-28 CUDA/C++ 算法分支

历史测试格点为 `16x32x32x48`，`m=0.05`、`atol=1e-6`、fine c64、block `(2,2,2,2)`。

| 分支 | 时间 | history | 解差 | true residual |
|---|---:|---:|---:|---:|
| Chebyshev | `1.442 s` | 17 | `8.57e-6` | `6.59e-7` |
| CA-GCR(s=4) | `24.262 s` | 26 | `1.87e-6` | `7.34e-7` |
| BiCGStabL(L=2) | `3.105 s` | 58 | `4.85e-6` | `8.02e-7` |
| mixed c64 -> c128 | `5.375 s` | 86 | `5.14e-6` | `7.54e-7` |

小格 `8^3x16` 三层 W/F/K 分别为 `0.229/0.253/0.205 s`。这些结果证明分支闭环，不证明大格三层性能。CA-GCR 的 `24.262 s` 更不能作为通信规避收益。

Chebyshev 历史谱估计采用保守启发式：

$$
\hat\rho=\frac{\lVert Ar\rVert}{\lVert r\rVert},
\qquad
\lambda_{\max}=8\hat\rho,
\qquad
\lambda_{\min}=0.05\lambda_{\max}.
$$

该估计不是严格谱包络证明。CA-GCR 使用四阶块和两遍 MGS，坏块回退 FGMRES；BiCGStabL 固定 `L=2`，每 8 个 block 做 reliable update。

### 21.2 2026-08-29 修订链

08-29 的五个后缀版本是累积修订，不是五份独立结论：

| 版本 | 新增内容 | 替代关系 |
|---|---|---|
| 基础 | CPU `4^4`、c64 小格、2 rank 闭环 | 初始 |
| `_2` | 2/4 rank 与多拓扑 full MG | 第 72 次结束替换为第 71 次收敛 |
| `_3` | null-vector setup 与五类 setup solver | 扩展测试到 21 |
| `_4` | compact MATPC、odd Schur | 替代旧 full 框架 |
| `_5` | 单 rank CUDA production 回归 | 当日最终版 |

08-29 基础等价性：

| 检查 | 结果 |
|---|---:|
| projection | `2.32e-7` |
| restriction/prolongation | `4.70e-7` |
| Galerkin | `6.14e-7` |
| preconditioned operator | `3.71e-8` |
| normal operator | `0` |

compact 代数：

| 检查 | 结果 |
|---|---:|
| compact `RP` | `1.67e-7` |
| transfer adjoint | `6.76e-9` |
| compact block Gram | `4.34e-7` |
| `R S_o P` Galerkin | `7.81e-8` |
| Schur strict adjoint | `5.45e-9` |
| reconstruct | `2.18e-8` |
| MG full residual | `1.02e-7` |
| direct Schur residual | `1.12e-7` |

单 rank CUDA 历史点：

| 格点 / levels | 时间 | full residual | 与参考解差 |
|---|---:|---:|---:|
| `8x8x8x16` 2L | `0.159 s` | `6.88e-7` | `1.72e-5` |
| `8x16x16x16` 2L | `0.234 s` | `2.41e-7` | `2.40e-6` |
| `8x16x16x16` 3L | `0.694 s` | `3.23e-7` | `4.36e-6` |

历史 speedup `2.369x/2.005x/0.455x` 的相对基准是裸 C++ BiStabCG，不是优化后的 1L。

### 21.3 2026-08-30 33 点 stencil 和严格基准

33 点 Schur coarse operator 的 support 为：

```text
1 onsite
8 signed axis neighbors
24 two-axis diagonal neighbors
= 33 coupling positions
```

33 是位置数，不是自由度数。每个位置保存一个 `E x E` dense block。

布局为：

```text
sit:      [E,E,X,Y,Z,T]
hop_nn:   [2,4,E,E,X,Y,Z,T]
hop_diag: [2,2,6,E,E,X,Y,Z,T]
```

父层 `l` 的指针槽为 `30+4l` 到 `30+4l+3`。

历史正确性结果：

| 检查 | 结果 |
|---|---:|
| batch setup | `0.155 -> 0.122 s`，约 `1.266x` |
| streaming 与 materialized Schur | max difference `0` |
| K=1 与 K=4 stencil | max difference `0` |
| 2/4 rank parity round-trip | max/relative `0` |
| 2 rank c64 coarse equivalence | `5.60e-7` |
| 2 rank c128 coarse equivalence | `1.03e-15` |
| 2 rank fine c64、coarse c128 MG | `7.69e-7` |
| Python 回归 | `26 passed` |

严格 cold-solve benchmark 在 V100、`8x8x8x16` 上做了三次中位数：

| 配置 | wall | backend solve | iter | residual | peak / delta |
|---|---:|---:|---:|---:|---:|
| 1L | `0.114211 s` | `0.111100 s` | 92 | `4.36e-7` | `2839 / +679 MiB` |
| 2L-MR/V | `0.120011 s` | `0.108655 s` | 84 | `5.42e-7` | `2840 / +679 MiB` |
| 3L-CG/F | `0.131453 s` | `0.127905 s` | 64 | `4.88e-7` | `2888 / +727 MiB` |

相对 1L：

| 比较 | 结果 |
|---|---:|
| 2L backend solve | `1.023x` |
| 2L 同步 wall | `0.952x` |
| 3L 同步 wall | `0.869x` |

大格 `16x32x32x48`：

| 配置 | wall | iter | residual | peak |
|---|---:|---:|---:|---:|
| BiStabCG | `1.95-1.97 s` | 参考 | 未列 | `5470 MiB` |
| 1L | `1.408-1.428 s` | 84 | `3.96e-7` | `8304 MiB` |
| 2L | `1.424-1.580 s` | 68 | `6.55e-7` | `9628 MiB` |

两点估计的 2L/1L 约 `0.944x`。2L 比 1L 多约 `1324 MiB` peak。

`16^4` 三层在默认、`restart=10/ctf=1e5` 和再加 `cmi=15` 三种配置下均跑满 1000 次并产生 NaN。该阶段不能讨论三层性能。

若使用 mixed precision：

| 路径 | 时间 |
|---|---:|
| c64 -> c128 | `0.6601 s` |
| c128 -> c64 | `0.8654 s` |
| 同精度 c64 | `0.1272 s` |

跨精度约慢 `5.2x/6.8x`。deflate 的 `0.09211 s` 对应 true residual `5.97e-6`，未达到 `1e-6`；warm start 的 `0.006795 s` 是已有近似解，不是 cold solve。

## 22. 2026-09-06 至 09-13：Strict 正式对照与代数纠正

### 22.1 09-06 与 09-09

固定协议为 V100、`16x32x32x48`、c64、2 levels、block `(2,2,2,2)`、`n_v=12`、coarse spin 2、`E=24`、target parity 1、2 warmup 加 5 steady。requested restart 16 受 512 MiB workspace 限制，effective restart 为 4。

| 版本 | PyQCU median | QUDA median | speedup | PyQCU iter | QUDA iter |
|---|---:|---:|---:|---:|---:|
| 09-06 | `2.041653 s` | `2.094841 s` | `1.026051x` | 11 | 37 |
| 09-09 | `1.966795 s` | `2.094764 s` | `1.065065x` | 11 | 37 |

09-09 是 09-06 的直接替代版。09-09 单次外层迭代为：

| 实现 | ms/outer |
|---|---:|
| PyQCU | `178.800` |
| QUDA | `56.615` |

true residual：

| 侧 | max true residual |
|---|---:|
| PyQCU | `3.6013e-7` |
| QUDA | `7.3030e-7` |

09-09 阶段中位数：

| 阶段 | ms | outer 比例 |
|---|---:|---:|
| coarse V-cycle | `158.184` | 88.78% |
| coarsest BiCGStab | `151.301` | 85.07% |
| fine restriction | `3.307` | 1.86% |
| fine pre-smoother | `2.974` | 1.67% |
| fine post-smoother | `2.829` | 1.59% |
| true residual refresh | `3.175` | 1.76% |

09-09 CUDA 消融保留原始绝对时间：

| 配置 | 时间 |
|---|---:|
| coarse hopping block 128 | `2.020994 s` |
| coarse hopping block 256 | `1.926614 s` |
| V100 native `sm_70` | `1.964751 s` |
| barrier-elision | `1.964744 s` |

block 256 保留为当时最优；native sm70 和 barrier-elision 没有稳定收益，后者已回退。原文对百分比基线的描述不够一致，因此本文只引用绝对秒数，不重新构造百分比。

### 22.2 09-13 纠正

09-13 专门纠正粗层 `X/Y/Yhat` 的 storage 和矩阵顺序：

| 旧写法或理解 | 最终裁决 |
|---|---|
| backward `Yhat` 右乘 storage 点 `X^-dagger` | 错误，应右乘目标点 `q` 的 `X^-dagger` |
| PC dslash 还显式乘 onsite `X` | 错误，PC dslash 只做 `Yhat` hopping，`clover=false` |
| 粗层继续读取 fine Clover/Gauge | 错误，后续层使用当前层 `X_l/Y_l` |
| `Yhat` 与 raw `Y` 可互换递归 | 错误，coarse-PC 下一层 coarsen 的是 `Yhat` |

该专题没有运行完整 GPU 逐元素回归，因此只裁决局部代数，不替代全局 solver 实测。

## 23. 2026-09-16 至 09-17：撤回、double MG 与 setup

### 23.1 撤回 `4.3817x`

09-16 报告的 c64 大格三层 `4.3817x` 被撤回。原因是 PyQCU 每个外层迭代执行一次递归 V-cycle，而旧 QUDA 配置在中间层执行多轮内层迭代，收敛预算不对齐。

09-17 重新构建 QUDA，使 `QUDA_PRECISION=12` 同时提供 c64/c128，并开启 `QUDA_MULTIGRID_DOUBLE`。为保证 double coarse MG 可运行，还修复了：

| 修复 | 原因 |
|---|---|
| mixed precision gauge accessor 构造目标 complex 类型 | 类型转换错误 |
| double coarse-link atomic accumulation 使用 double 存储 | 旧路径把 double 字段按 int fixed-point 解释，导致 NaN/norm abort |
| c64 保留 deterministic fixed-point | 避免性能和行为回归 |
| aligned 与 library 两套 QUDA 策略分离 | 防止不同预算混算 |

aligned 统一为：

```text
one outer iteration -> one recursive V-cycle
MR smoother
smoother_tol=0
non-coarsest coarse_solver_maxiter=1
coarsest maxiter=200
```

历史 aligned 单点：

| 点 | PyQCU | QUDA | 比值 | 外迭代 |
|---|---:|---:|---:|---:|
| c64 大格 3L sm70 | `0.655570 s` | `1.314577 s` | `2.0052x` | 14/39 |
| c128 `16^3x16` 3L | `0.380444 s` | `1.230048 s` | `3.2332x` | 44/87 |
| c128 小格 3L | 未列 | 未列 | `12.3768x` | 未列 |
| c128 大格 cache-hit | `2.061594 s` | `16.166481 s` | `7.8417x` | 20/60 |

这些是 09-17 历史点，不是 09-28 的 66 单元最终矩阵。

### 23.2 Setup 优化

原始 colored Galerkin 在 Python 内逐 source、逐 support 点构造 canonical block。优化后：

| 版本 | c128 小格 setup |
|---|---:|
| K=1 Python 逐点 | `44.86 s` |
| K=4 批量 gather/scatter | `16.67 s` |
| 进一步优化 | `14.66 s` |

parity 分桶把大格 `8x16x16x24` 的 source color groups 从 22 降到 16，每层 colored operator calls 从 132 降到 96。大格 c128 C=4/K=16 冷 setup 从约 `974 s` 降到 `585.28 s`，低于同点 QUDA setup `630.74 s`。

C=1 release-before-runtime 冷 setup 为 `2709.45 s`，发布约 `8.89 GB` runtime cache。所有 setup 数字只属于 setup 研究，不进入 solve speedup。

## 24. 2026-09-18 至 09-27：分布式实现、overlap 和失败优化

### 24.1 分布式正确性

09-18 将 Strict 从本地周期算子和本地 dot 补齐为：

| 模块 | 实现 |
|---|---|
| fine/compact/full vector halo | face exchange |
| P/R coarse halo | coarse link 和 output boundary |
| fused FGMRES | global dot、dot pair、dot many |
| Galerkin setup | global roll 加 halo 化 fine operator |

正确性：

| 测试 | c64 | c128 |
|---|---:|---:|
| full coarse operator relative error | `1.6e-7` | `3.0e-16` |
| compact coarse operator relative error | `2.2e-8` | `7.1e-19` |
| 2/4 rank setup internal/boundary | 0 | 0 |
| distributed solve true residual | `<1e-6` | `<1e-6` |

分布式 FGMRES 缺陷修复后，restart 20 的 true relative residual 从 `1.9e-2` 降到 `6.7e-7`，外层迭代从异常状态恢复到 17。

### 24.2 双 P100 根因修复

两个确定性错误已修复：

| 根因 | 修复 |
|---|---|
| C++ 强制 `cudaSetDevice(0)`，rank 1 传入 GPU1 指针 | 使用 `PYQCU_MPI_DEVICE_ID` 或 local rank |
| 1D/2D face pack 对向上取整线程数无上界保护 | 十个 pack kernel 在索引解码前检查 |

修复后原失败命令返回 0，compute-sanitizer memcheck 为 0 errors。

### 24.3 静态 link-halo cache

preconditioned link 在一次 hierarchy 生命周期内不变。按 pointer identity 和 `(parity,dim)` 缓存后：

| 变体 | steady median | MAD | iter | max residual | 相对时间 |
|---|---:|---:|---:|---:|---:|
| cache off | `1.9549 s` | `0.0215 s` | 9 | `7.3721e-7` | 1.000 |
| cache on | `1.0093 s` | `0.0187 s` | 9 | `7.5174e-7` | 0.516 |

该 A/B 使用同一 binary、输入和协议，比值约 `1.937x`。

### 24.4 09-21 全矩阵快照

09-21 当时完成 144 个 side case，72 个 combined 全部 `fair=true`：

| 集合 | median R | wins | losses |
|---|---:|---:|---:|
| MG-2/3 全体 | `1.298` | 36/48 | 12/48 |
| V100 MG-2/3 | `2.492` | 23/24 | 1/24 |
| P100 MG-2/3 | `1.063` | 13/24 | 11/24 |
| BiCGStab reference | `0.243` | 0/24 | 24/24 |

两张历史分布图：

![09-21 快照的逐单元时间比](High-Performance_Implementation_MG_assets/full_document_assets/mg_matrix_20260921_speedup_units.png)

![09-21 快照的逐层阶段时间](High-Performance_Implementation_MG_assets/full_document_assets/mg_matrix_20260921_stage_breakdown.png)

![09-21 快照的逐层迭代数](High-Performance_Implementation_MG_assets/full_document_assets/mg_matrix_20260921_level_iterations.png)

09-21 还记录两条失败优化：

| 实验 | 现象 | 处置 |
|---|---|---|
| triple-dot 合并三个内积 | 大格 c64 FGMRES 外残差长期停在 `1e-3` | 回退 |
| pipelined BiCGStab 延迟 `rho` | 同样停滞，coarse residual 仍约 0.2 | 回退 |

这两个实验说明 reduction reorder 会改变 coarse BiCGStab 递推稳定性，不能只以“减少同步次数”验收。

### 24.5 09-27 nonblocking overlap

独立双 rank 正确性 probe 为双 P100、`8^3x16`、c64、`grid=2x1x1x1`：

| 路径 | 外迭代 | true residual |
|---|---:|---:|
| `PYQCU_MPI_OVERLAP=0` | 27 | `6.4314e-7` |
| `PYQCU_MPI_OVERLAP=1` | 26 | `7.5194e-7` |

大格 c64 MG-2 A/B 使用 coarse-tol `0.3`、`nu=1`、两组反向顺序：

| 路径 | median | MAD | iter median | max residual |
|---|---:|---:|---:|---:|
| off | `4.6544 s` | `0.2816 s` | 15 | `7.83e-7` |
| on | `4.3374 s` | `0.1838 s` | 16 | `7.91e-7` |

运行中位时间的改善为 `6.81%`。

![09-27 overlap off/on A/B](High-Performance_Implementation_MG_assets/full_document_assets/mg_overlap_ab_20260927.png)

c128 大格三层 A/B：

| 路径 | 外迭代 | steady | true residual |
|---|---:|---:|---:|
| 原顺序 / overlap off | 32 | `5.10 s` | `7.55e-7` |
| overlap 实验开启 | 451 | `46.28 s` | `7.61e-7` |

因此 c128 默认关闭 overlap。该结论由当前源码的 `PYQCU_STRICT_OVERLAP_C128` 默认 `false` 支撑。

09-27 的 `72/72 exact`、MG `38/48` 和中位 `1.346` 是旧容量口径。09-28 使用更严格的 85% 显存和 570 秒门限后，正式口径改为 66 exact、132 side、6 substitutions。09-27 overlap A/B 仍可引用，但不能替代最终矩阵。

## 25. 2025-08-29 DCU 历史应用测试

### 25.1 环境与范围

| 项目 | 配置 |
|---|---|
| 处理器 | Hygon C86 7185 32-core |
| 内存 | `8 x 16 GB DDR4 2666 MHz` |
| 网络 | 200 Gb |
| 加速卡 | 4 张 Pre-Wukong DCU |
| 存储 | ParaStor 300 |
| OS | RHEL 7.4，kernel 3.10 |
| MPI | HPC-X 2.4.0 |
| UCX | 1.6.0 |
| 调度 | Gridview 4.6 |

嵌入 Excel 和环境记录与正文名义配置并不完全一致：实际登录节点记录显示 Hygon C86 OPN `7580B`、`2 sockets x 96 cores = 192 CPUs`、12 NUMA nodes、`MemTotal≈500.8 GiB`，并出现 ROCm/HIP、`gfx936`、`sghpc-mpi-gcc-mlnx/25.6` 等路径。该差异说明名义平台表和最终实机环境不能静默合并。复现应以 `lscpu`、`meminfo`、模块列表和原始作业脚本为准。

测试对象是历史上的 Wilson BiCGStab 和 Clover BiCGStab 传播子求解。该时期的 MG 仍是前期验证版本。表内线程数表示 MPI rank/线程规模，不表示 1024 张 GPU。

固定参数：

| 参数 | 值 |
|---|---:|
| mass | `0.05` |
| `tol(x_o)` | `1e-12` |
| `_BLOCK_SIZE_` | 256 |
| 源 | 面源 |
| 最终检查 | 重建另一半解后的 full residual |

### 25.2 关键历史结论

论文资产中的 508 缩放图为：

![508 DCU 历史缩放图](High-Performance_Implementation_MG_assets/full_document_assets/fig_508_scaling.png)

该图对应早期 DCU 4 卡环境，报告 TEST8 c64 大格从 64 到 1024 进程时 Wilson 加速约 `3.119x`、Clover 约 `7.193x`；另一个 c128 系列给出 Wilson `2.211x`、Clover `4.258x`。不同表的基准和总格点不同，不能把 `7.193x` 与 `4.258x` 当成同一分母的连续曲线。

历史结论：

| 结论 | 解释 |
|---|---|
| c64 Wilson 强扩展在 16 线程附近达到约 `7.582x` | 继续到 32 线程回落到 `7.102x` |
| c64 Clover 在 32 线程表内显示 `12.599x`，但该行 NaN | 该单点不能作为可靠结果 |
| c128 在 1024 进程时 Wilson 约 `2.211x`、Clover 约 `4.258x` | 大格强扩展仍需结合通信和内存解释 |
| 弱扩展总体良好 | 通信逐步成为主导 |
| 显存约按 rank 线性复制 | 多 rank 没有减少全量准备资产 |

按原始耗时重算后，结论必须更精确：TEST8 c64 Wilson 在 np512 附近效率约 61.5%，到 np1024 降至约 20.8%；Clover 在 np1024 约 51.4%。TEST9 c128 的 Wilson/Clover 在 np1024 分别约 28.5%/60.7%。弱扩展较好的区间也存在，但高并发和跨节点通信会使 Wilson 路径明显衰减。

TEST6 c64 np32 的 Clover 行在 999 次迭代后出现 NaN，必须从性能统计中剔除并保留为失败证据。c64 个别点最终残差平方约 `1e-11`，高于正文名义 `tol(x_o)=1e-12`，不能宣称通过该声明门。每项基本只有一次运行，证据强度低于当前 `1 cold + 2 warmup + 5 steady` 协议。

### 25.3 原始数据图表

以下图片来自解包的 Office 文档，保留用于历史审计。多数是不同测试表格的截图，不重新解释为当前 Strict-MG 性能。

![508 c64 性能表](<张鑫 508应用测试报告-PyQCU/word/media/image19.png>)

![508 性能分析截图](<张鑫 508应用测试报告-PyQCU/word/media/image20.png>)

![508 profiler 时间线](<张鑫 508应用测试报告-PyQCU/word/media/image21.png>)

![508 原始测试表 22](<张鑫 508应用测试报告-PyQCU/word/media/image22.png>)

![508 原始测试表 23](<张鑫 508应用测试报告-PyQCU/word/media/image23.png>)

![508 原始测试表 24](<张鑫 508应用测试报告-PyQCU/word/media/image24.png>)

![508 c128 强扩展表](<张鑫 508应用测试报告-PyQCU/word/media/image25.png>)

![508 c128 弱扩展表](<张鑫 508应用测试报告-PyQCU/word/media/image26.png>)

![508 c128 总表 27](<张鑫 508应用测试报告-PyQCU/word/media/image27.png>)

![508 c128 总表 28](<张鑫 508应用测试报告-PyQCU/word/media/image28.png>)

![508 综合总表](<张鑫 508应用测试报告-PyQCU/word/media/image29.png>)

### 25.4 适用边界

508 报告证明的是早期 DCU 上 Wilson/Clover BiCGStab 的传播子求解规模、强扩展和弱扩展趋势。它不能证明：

1. 当前 Strict-MG 已在 1024 DCU 上限完成端到端扩展；
2. Pre-Wukong DCU 的性能可外推到现代 NVIDIA/DCU；
3. 1024 MPI rank 等于 1024 个独立 GPU 的聚合峰值；
4. 早期 MG 前期验证版本与 `test27` 的 QUDA 对照可以直接比较。

---

## 26. 图表总览与解释边界

### 26.1 当前权威图

| 图 | 解释 |
|---|---|
| `report_chart_assets/overlap_matrix.png` | 22 个分组和 66 个精确单元的 QUDA/PyQCU 比值 |
| `report_chart_assets/mg_speedup_units.png` | 66 单元逐点分布 |
| `report_chart_assets/mg_stage_breakdown.png` | trace-on 嵌套阶段时间 |
| `report_chart_assets/mg_peak_memory.png` | device-wide 显存中位 |
| `report_chart_assets/mg_finest_residual.png` | full-operator finest residual 与外层迭代 |
| `paper_assets/fig_ratio_distribution.png` | trace on/off 合并分布的论文版 |
| `paper_assets/fig_group_medians.png` | trace-off 分组中位 |
| `paper_assets/fig_stage_shares.png` | 阶段份额，不跨实现直接相加 |
| `paper_assets/fig_residual_memory.png` | 残差与显存 |
| `paper_assets/fig_508_scaling.png` | 2025 DCU 历史缩放 |

对应矢量 PDF 保留在 `High-Performance_Implementation_MG_assets/paper_assets/` 与 `report_chart_assets/`。正文使用 PNG 是为了兼容不直接渲染 PDF 的 Markdown 预览器。

### 26.2 历史性能图

09-21 的最细层残差总览：

![09-21 最细层残差总览](High-Performance_Implementation_MG_assets/full_document_assets/mg_matrix_20260921_finest_residual.png)

09-21 的完整残差曲线。原图较高，适合放大查看。

![09-21 最细层残差曲线](High-Performance_Implementation_MG_assets/full_document_assets/mg_matrix_20260921_residual_curves.png)

09-27 最终协议逐单元比值：

![09-27 72 单元逐单元比值](High-Performance_Implementation_MG_assets/full_document_assets/mg_overlap_matrix_20260927.png)

09-27 的 `mg_level_time_stack.pdf` 是约 27:1 的超宽逐层时间图，不适合缩入单页预览。矢量原件在 `data/report_multigrid_optimized_20260927/final_protocol/report/`，正文保留源路径和结构化阶段表。

### 26.3 算法图

BiCGStab 单列算法图：

![BiCGStab 算法](High-Performance_Implementation_MG_assets/full_document_assets/algorithm_bicgstab.png)

V-cycle 与 FGMRES 嵌套关系：

![V-cycle 与 FGMRES](High-Performance_Implementation_MG_assets/full_document_assets/algorithm_vcycle_fgmres.png)

### 26.4 已舍弃的重复图

以下图不进入正文：

| 类型 | 原因 |
|---|---|
| `mg_report_smoke/partial/final/final2/one/traced/v100_c128` 同组 PDF | 同一自动绘图器的阶段版本，最终图已覆盖 |
| `mg_matrix_final_report` 与 `mg_matrix_final_report_audit_20260927` 重复项 | 后者是被 09-28 最终论文图覆盖的旧矩阵图 |
| `paper_render*`、`beamer_render*`、`presentation_render*` | 可由 TeX 和主图重建的逐页 PNG |
| contact sheet | 只用于人工检查，不承载独立数值 |
| 相同 24³ 图组的六份路径替换副本 | 只有 `_5/final` 代表版保留 |
| 09-06 和 09-09 中相同 residual 曲线的重复截图 | 09-09 保留，09-06 只留来源记录 |

## 27. 复现与验证

### 27.1 构建

```bash
source ./env.sh
bash ./build.sh
bash ./install.sh
```

### 27.2 Strict 快速门禁

```bash
RUNNER=pyqcu/testing/qcu/strict/quda_comparison
python -B "${RUNNER}/run_strict_fast.py" \
  --fail-fast --json strict-fast.json
```

### 27.3 双 rank 真残差 probe

```bash
mpirun -np 2 python -B \
  pyqcu/testing/qcu/strict/quda_comparison/strict_mpi_solve_probe.py \
  --shape 8 8 8 16 \
  --grid 2 1 1 1 \
  --dtype c64 \
  --galerkin-mode colored
```

### 27.4 最终矩阵

```bash
python -B \
  pyqcu/testing/qcu/strict/quda_comparison/bench_mg_matrix_full.py \
  --list
```

正式入口：

| 功能 | 入口 |
|---|---|
| 矩阵编排 | `bench_mg_matrix_full.py` |
| 单侧记录 | `bench_strict_vs_quda.py` |
| 合并审计 | `assemble_final_matrix.py` |
| 图表生成 | `build_mg_report.py` |
| 两 rank solve probe | `strict_mpi_solve_probe.py` |
| 两 rank primitive probe | `strict_mpi_primitive_probe.py` |
| 两 rank setup probe | `strict_mpi_setup_probe.py` |

### 27.5 验收矩阵

#### 算子级

| 编号 | 检查 | 通过条件 |
|---|---|---|
| O1 | 自由场 Dslash | Python、QCU、参考实现一致 |
| O2 | gamma5 Hermiticity | 相对误差处于 dtype 舍入量级 |
| O3 | Clover onsite | 布局、dagger、批量 inverse 一致 |
| O4 | parity round-trip | full -> parity -> full 不变 |
| O5 | Schur reconstruct | full residual 与直接 apply 一致 |
| O6 | transfer | `||RP-I||` 与局部正交可记录 |
| O7 | Galerkin | 逐列、batch、colored 等价 |
| O8 | strict support | 非最近邻 support fail-closed |

#### Solver 级

| 编号 | 场景 | 必须记录 |
|---|---|---|
| S1 | 零 RHS | 返回零解，无 NaN |
| S2 | 已知解 | 与 `x*` 的误差 |
| S3 | breakdown | 分母为零或非有限时返回明确状态 |
| S4 | 递推与 true residual | 周期刷新前后比值 |
| S5 | 变化预条件器 | FGMRES 右预条件语义 |
| S6 | warm start | `x0=0` 与 warm 分开 |
| S7 | mixed precision | 低精度推进和 full residual 刷新 |
| S8 | MPI | rank 变化后的全局范数和可解释漂移 |

#### MG 级

```text
1. 固定 gauge、seed 和 null vectors
2. 保存 layout、dtype、block、nvec、parity、边界和版本
3. 构造 native、legacy/compact、strict hierarchy
4. 检查 RP、局部正交、X inverse 和 support
5. 对同一 full rhs 运行一次 V-cycle
6. 独立计算 full true residual 和误差传播因子
7. 运行外层 solver，记录 setup、solve、通信和显存
```

### 27.6 最小运行记录

```text
commit/tag:
lattice:
boundary:
action:
layout:
dtype:
solver:
mg levels / block / nvec / smoother / coarse solver:
rhs / x0:
stop:
runtime GPU / MPI ranks / threads:
setup / solve / communication / memory:
iterations / residual history / status:
```

缺少这些字段时，可以描述算法，但不能声称两个结果严格可比。

## 28. 冲突裁决表

| 冲突 | 旧内容 | 当前裁决 |
|---|---|---|
| `params` 长度 | 早期文档写 54 | 当前源码和 define 为 58 |
| MG mode | 只写 GCR bit | 当前为兼容 bit mask，新增 MR、Chebyshev、CA-GCR、W/F/K、BiCGStab(l) |
| Strict 单 rank/多 rank | 09-11 静态说明偏向单 rank | 09-18 和当前源码已有分布式 halo、global dot、P/R halo 和 setup |
| Strict asset 递归对象 | raw `Y` | coarse-PC 下一层 coarsen `Yhat` |
| backward `Yhat` | storage 点乘 `X^-dagger` | 目标点 `q` 的 `X^-dagger` |
| PC dslash onsite | 再乘 `X` | 只做 `Yhat` hopping，`clover=false` |
| c64 三层 `4.3817x` | 09-16 正式值 | 预算未对齐，正式撤回 |
| 09-17 `2.0052x` | c64 大格 aligned 单点 | 历史点，不进入最终 66 单元 |
| 09-17 c128 大格 `7.8417x` | cache-hit 单点 | 当前最终 c128 格点改为 `16x16x32x32`，旧值只作历史 |
| `72/72 exact` | 09-27 600 秒口径 | 09-28 改为 66 exact、6 substitutions |
| MG `38/48`、中位 `1.346` | 09-27 | 09-28 为 MG `35/44`、总体 `1.0250` |
| `36/48`、`2.492`、`1.063` | 09-21 | 已被 09-27/09-28 取代 |
| coarse-tol `0.3` | 快速 A/B | 最终 c64 `0.003`、c128 `3e-5` |
| c128 gate | 历史 `5e-8` | 09-28 当前统一门为 `5e-6`，不得宣称当前结果通过 `5e-8` |
| c64/c128 overlap | 统一启用 | c64 默认启用；c128 默认关闭 |
| trace-on 性能 | 阶段诊断值 | 只用于归因 |
| 24³ legacy 3L `1.107x` | 早期路径 | 不与 Strict 性能比较 |
| 08-30 2L/3L 无净收益 | 早期 33 点路径 | 不等于后续 Strict 无效，也不能反指测量错误 |
| 08-29 速度比 `2.369x/2.005x` | 对裸 BiStabCG | 不能对优化后的 1L 基线 |
| 08-29 收敛次数 | 72 | `_2` 之后为 71 |
| `135 s -> 15 s` 与 `135 s -> 48 s` | 不同批量/多线程口径 | 明确区分，不写成同一基准 |
| 09-28 图注 | “PyQCU/QUDA”反向措辞 | 以公式 `R=QUDA/PyQCU`、表头、坐标轴为准 |
| Hopper | 标题中的优化主题 | 实测硬件只有 P100/V100，不是 Hopper GPU 结果 |
| DCU 1024 | 线程/rank 数 | 不是 1024 张 GPU 聚合峰值 |

## 29. 来源台账

### 29.1 规范与运行文档

| 文件 | 处置 |
|---|---|
| `AGENTS.md` | 全文吸收项目约束、命令和反模式 |
| `ORGANIZATION.md` | 全文吸收目录分类与迁移规则 |
| `dims.md` | 全文吸收布局和维度约定 |
| `env.md` | 全文吸收依赖和环境 |
| `install.md` | 全文吸收构建流程 |
| `examples.md` | 全文吸收新测试路径 |
| `profiler.md` | 全文吸收 Perfetto 入口 |
| `form-audit-20260928.md` | 只保留组织/审计结论，不作为物理技术正文 |
| `plans/2026-09-16-multigrid-profiling-benchmark.md` | 保留公平协议和验收设计；任务清单已由报告取代 |

### 29.2 当前权威技术文档

| 文件 | 并入章节 |
|---|---|
| `High-Performance Implementation of Multigrid Solver for Lattice QCD.tex/.pdf` | 4 至 18 |
| `report_multigrid_quda_pyqcu_20260928.tex/.pdf` | 14 至 18 |
| `report_multigrid_optimized_20260927.tex/.pdf` | 10、24 |
| `report_multigrid_distributed_20260918.tex/.pdf` | 9 至 11、24 |
| `report_numerical_linear_algebra_lattice_qcd_20260911.md` | 4 至 13、27 |
| `report_multigrid_quda_pyqcu_20260913.tex/.pdf` | 8、22 |

### 29.3 历史阶段文档

| 文件 | 处置 |
|---|---|
| `report_multigrid_quda_pyqcu_20260917.tex/.pdf` | 保留 aligned、double MG、setup 的历史结论 |
| `report_multigrid_quda_pyqcu_20260916.tex/.pdf` | 保留 trace 设计和 `4.3817x` 撤回链 |
| `report_multigrid_final_20260921.tex/.pdf` | 保留首个完整分布式矩阵、失败 reduction 和热点 |
| `report_multigrid_fast_iteration_20260921.tex/.pdf` | 保留设备绑定、pack 越界和 link cache A/B |
| `report_multigrid_quda_pyqcu_20260909.md/.tex/.pdf` | 保留 device-side MR、融合 coarse BiCGStab 和历史单点 |
| `report_multigrid_quda_pyqcu_20260906.md/.tex/.pdf` | 只作为 09-09 前版；其时间、stage 和局部代数被取代 |
| `report_quda_multigrid_reproduction_20260830.tex/.pdf` | 合并 08-29 五版和 33 点 setup |
| `report_pyqcu_mg_operator_construction_20260830.tex/.pdf` | 保留 Python 构造、C++ 应用边界和 33 点布局 |
| `report_pyqcu_multigrid_cuda_20260828.tex/.pdf` | 保留算法族和早期实现 |
| `data/mg-bench-current-20260830/report.md` | 保留严格 cold-solve 反例和粗层同步热点 |

### 29.4 早期分析文档

| 文件 | 处置 |
|---|---|
| `analy_all_20260814_3.tex/.pdf` | 作为 08-14 最终版；v1/v2 只保留为前版 |
| `analy_pyqcu_20260818.tex/.pdf` | 结构、依赖、生命周期 |
| `pure_pyqcu_20260818.tex/.pdf` | 核心算子、solver、Galerkin 和多线程 |
| `pure_mg_20260818.tex/.pdf` | V-cycle 15 步和 5-stream |
| `diff_init_stab27_20260818.md/.tex/.pdf` | 历史变更统计，只有规模背景 |
| `analy_test15_5_20260820.tex/.pdf` | `24^3x72` 代表版；前五版为同图组路径替换 |
| `pyqcu_cover_20260818.tex/.pdf`、`pyqcu_report_20260818.pdf` | 合成交付件，不替代源稿 |

### 29.5 自动生成与派生文件

| 目录或模式 | 处置 |
|---|---|
| `data/mg_report_*` | 自动阶段图、表、TeX；只取最终代表图 |
| `data/mg_matrix_final_report*` | 09-21 自动报告；保留历史图 |
| `data/report_multigrid_optimized_20260927/final_protocol/report/*` | 09-27 自动图；只保留 overlap 和阶段来源记录 |
| `High-Performance_Implementation_MG_assets/paper_build/` | 生成 |
| `High-Performance_Implementation_MG_assets/beamer_build/` | 生成 |
| `High-Performance_Implementation_MG_assets/speaker_script_build/` | 生成 |
| `*_render/`、`*_contact_sheet.png`、`slides/` | 生成 |
| `*.aux/.log/.out/.toc/.nav/.snm` | 生成缓存 |

这些文件仍可在工作区中保留为构建资产，但不作为独立事实来源。

## 30. 总体结论

PyQCU 已在以下层面形成完整闭环：

| 层面 | 结论 |
|---|---|
| 物理 | Wilson/Clover、gamma5-Hermiticity、Schur 和 MATPC 语义明确 |
| 数学 | P/R、Galerkin、Strict `X/Y/Yhat`、Krylov 和 true residual 有统一定义 |
| 实现 | Python/Cython/C++/CUDA/MPI 分工明确，ABI 和生命周期有回归约束 |
| 分布式 | fine/compact/full、P/R、global FGMRES 和 setup 已验证到舍入水平 |
| CUDA | c64 overlap、静态 link cache、融合 kernel、cache 和显存预算已落地 |
| 性能 | 最终 66 单元矩阵证明 MG 专项优势，非 MG 求解器仍有明显短板 |
| 风险 | c128 overlap、大格 setup 显存、阻塞 link halo 和多 rank reliable update 尚未解决 |

当前最重要的数字关系是：

```text
66 exact combined units
132 exact side records
6 explicit substitutions
R_all_median = 1.0250
MG-2/3 wins = 35/44
trace-off MG wins = 21/22
BiCGStab wins = 0/22
max residual PyQCU = 7.40e-7
max residual QUDA  = 7.97e-7
```

下一阶段不应继续追求小格或单点加速比，而应解决：

1. coarse solve、global reduction 和 reliable update 的同步成本；
2. setup 显存和 P100 c64 大格容量；
3. c128 overlap 的稳定性和 reproducible reduction；
4. CUDA-aware persistent communication 与更大 MPI 拓扑；
5. mixed precision 和更多作用量的扩展；
6. 与 DCU/NPU 后端一致的端到端验证。

---

## 附录 A. 符号表

| 符号 | 含义 |
|---|---|
| `U_mu(x)` | `SU(3)` 规范链接 |
| `H` | Wilson hopping kernel |
| `D_W` | Wilson Dirac operator |
| `kappa` | hopping parameter，`1/(2m0+8)` |
| `A_p` | onsite block，Clover 时为 `I+C_p` |
| `S_p` | parity Schur complement |
| `V` | aggregate 内 null-space basis |
| `P,R` | prolongation、restriction |
| `X` | coarse onsite dense block |
| `Y` | coarse directional link |
| `Yhat` | preconditioned coarse link |
| `E` | coarse dof，`2*n_v` |
| `nu_pre,nu_post` | 前、后 smoothing 次数 |
| `z_j` | FGMRES 预条件后的方向 |
| `r_true` | `b-Dx` full true residual |
| `R` | 性能时间比 `t_QUDA/t_PyQCU` |

## 附录 B. 算法选择

| 问题 | 选择 |
|---|---|
| SPD 的 `D†D` 或 Hermitian coarse operator | CG、CA-CG、多移位 CG |
| 原始 Wilson/Clover 或 asymmetric Schur | BiCGStab、FGMRES、GCR |
| MG/SAP 内层步数变化 | FGMRES、flexible-GCR |
| 局部高频误差 | MR、Chebyshev、Schwarz/SAP |
| 轻质量近零模 | MG、deflation、thick-restart Lanczos |
| full Clover parity solve | asymmetric/symmetric Schur 加 prepare/reconstruct |
| 宽 stencil action | action-specific coarse operator |
| strict nearest-neighbor X/Y | full Galerkin 且 support 必须 fail-closed |

## 附录 C. 最小文件清单

若只需继续开发，最少应保留：

```text
AGENTS.md
ORGANIZATION.md
dims.md
env.md
install.md
examples.md
profiler.md
report_multigrid_quda_pyqcu_20260928.tex/.pdf
High-Performance Implementation of Multigrid Solver for Lattice QCD.tex/.pdf
report_numerical_linear_algebra_lattice_qcd_20260911.md
report_multigrid_quda_pyqcu_20260913.tex/.pdf
report_multigrid_distributed_20260918.tex/.pdf
report_multigrid_optimized_20260927.tex/.pdf
High-Performance_Implementation_MG_assets/paper_assets/
High-Performance_Implementation_MG_assets/report_chart_assets/
张鑫 508应用测试报告-PyQCU/
```

其余历史修订、重复图和构建日志通过 Git 历史、数据目录和来源台账回溯，不必在单篇正文中重复展开。

---

# 附录 D. 完整公式与算法汇编

本附录用于防止公式、推导和算法在总册中被压缩成名称列表。内容按物理对象、线性代数、MG、分布式实现、缓存、性能和 508 历史算法分层。不同路径的归一化、storage 和停止条件分别标明。

## D.1 连续 QCD、格点截断与物理目标

连续欧氏 QCD 拉氏量写为

$$
\mathcal L_{\mathrm{QCD}}
=
\bar\psi_i\left(i\gamma_\mu D^\mu_{ij}-m\delta_{ij}\right)\psi_j
-\frac14F^a_{\mu\nu}F^{a}_{\mu\nu},
$$

其中 `D_mu = partial_mu - i g A_mu^a T^a`，`T^a` 为颜色生成元。格点方法将连续时空替换为

$$
x_\mu=n_\mu a,\qquad n_\mu=0,\ldots,L_\mu-1.
$$

格距和有限体积分别提供紫外、红外截断：

$$
\Lambda_{\mathrm{UV}}\sim\frac{\pi}{a},
\qquad
\Lambda_{\mathrm{IR}}\sim\frac{2\pi}{L}.
$$

### D.1.1 准 PDF

沿 `z` 方向、核子动量 `P_z` 的准 PDF 定义为

$$
\widetilde f(x,P_z,1/a)
=
\int\frac{dz}{4\pi}e^{ixzP_z}
\left\langle P\left|
\bar\psi(z)\gamma_t U(z,0)\psi(0)
\right|P\right\rangle,
$$

其中 `U(z,0)` 是规范链接。有限 `a` 计算得到 quasi-PDF，再通过大动量/连续极限外推。MG/BiCGStab 的任务是得到该矩阵元所需的夸克传播子，不直接完成匹配核或重整化。

### D.1.2 两点关联函数与 Wick 收缩

介子两点函数可写为

$$
C_2(t,t_0)=
\left\langle
\bar\psi(x,t)\Gamma\psi(x,t)
\bar\psi(y,t_0)\Gamma'\psi(y,t_0)
\right\rangle.
$$

对单 flavor 费米子做 Wick 收缩后，代表性项为

$$
C_2(t,t_0)
=-\gamma_5 S_q(x,y)\gamma_5 S_q(y,x),
$$

其中

$$
S_q=D_q^{-1}
$$

是 quark propagator。求解传播子等价于对多个 right-hand side 反复求解 `D_q x=b`。

## D.2 格点规范场、plaquette 与作用量

### D.2.1 规范链接

`U_mu(x) in SU(3)` 连接 `x` 与 `x+mu_hat`。协变平移为

$$
T_{+\mu}\psi(x)=U_\mu(x)\psi(x+\hat\mu),
\qquad
T_{-\mu}\psi(x)=U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu).
$$

局部规范变换为

$$
U_\mu(x)\to G(x)U_\mu(x)G^\dagger(x+\hat\mu),
\qquad
\psi(x)\to G(x)\psi(x).
$$

协变性要求

$$
D[U^G]\psi^G=G\,D[U]\psi.
$$

### D.2.2 Plaquette 与 Clover 场强

有向 plaquette 为

$$
P_{\mu\nu}(x)=
U_\mu(x)U_\nu(x+\hat\mu)
U^\dagger_\mu(x+\hat\nu)U^\dagger_\nu(x).
$$

四叶 Clover 和为

$$
Q_{\mu\nu}
=P_{\mu\nu}+P_{\nu,-\mu}+P_{-\mu,-\nu}+P_{-\nu,\mu}.
$$

反 Hermitian 场强项为

$$
C_{\mu\nu}=Q_{\mu\nu}-Q^\dagger_{\mu\nu}.
$$

代码常用形式可写为

$$
\sigma_{\mu\nu}F_{\mu\nu}
=
2\gamma_\mu\gamma_\nu
\frac14\sum_P\frac12(P-P^\dagger),
$$

其中的 gamma 和 plaquette 顺序必须与源码 convention 一致。

### D.2.3 Wilson hopping kernel

取 `r=1`：

$$
(H\psi)_x=
\sum_{\mu=0}^3
\left[
(1-\gamma_\mu)U_{\mu,x}\psi_{x+\hat\mu}
+(1+\gamma_\mu)U^\dagger_{\mu,x-\hat\mu}\psi_{x-\hat\mu}
\right].
$$

算子与作用量：

$$
D_W=(m_0+4)I-\frac12H,
\qquad
S_W=\sum_x\bar\psi_xD_W\psi_x.
$$

hopping 参数：

$$
\kappa=\frac{1}{2m_0+8},
\qquad
\frac{1}{2\kappa}=m_0+4.
$$

另一种等价形式：

$$
D_W=\frac{1}{2\kappa}(I-\kappa H).
$$

预条件归一化：

$$
D_{\mathrm{pc}}=I-\frac{\kappa}{u_0}H,
\qquad
D_W=(m_0+4)D_{\mathrm{pc}}.
$$

当前 C++ 生产路径使用 `u0=1`。原始 Dslash 接口返回 `H`，不能把 `I-kappa H` 和 `H` 在同一比较中混用。

### D.2.4 Clover onsite block

Clover 项可写为

$$
D_C=D_W+c_{\mathrm{SW}}\frac{i}{4}
\sum_{\mu<\nu}\sigma_{\mu\nu}F_{\mu\nu},
\qquad
\sigma_{\mu\nu}=\frac12[\gamma_\mu,\gamma_\nu].
$$

实现归一化：

$$
T_p=-\frac{\kappa c_{\mathrm{sw}}}{8u_0}
\sum_{\mu<\nu}\gamma_\mu\gamma_\nu C_{\mu\nu},
\qquad
A_p=I_p+T_p.
$$

当前有效 `c_sw=1`。Clover 项只增加 onsite block，不改变 hopping support。

## D.3 Wilson/Clover 的谱、极限与误差

自由场 `U=1` 时，Fourier 模满足

$$
D_W(p)=M(p)I+i\sum_\mu\gamma_\mu\sin p_\mu,
$$

其中

$$
M(p)=m_0+\sum_\mu(1-\cos p_\mu).
$$

因此

$$
D_W^\dagger(p)D_W(p)
=
\left[
M(p)^2+\sum_\mu\sin^2p_\mu
\right]I.
$$

该式给出三项回归：

| 模 | 检查 |
|---|---|
| `p=0` | 质量项 `M(0)=m0` |
| `p->-p` | gamma 项符号反对称 |
| `p_mu≈pi` | Wilson 项抬高 doubler |

Wilson/Clover 满足

$$
D^\dagger=\gamma_5D\gamma_5.
$$

因此 `gamma5*D` 可作为 Hermitian 相关算子，但 `D` 本身不是 Hermitian。

离散误差至少包含

$$
e_{\mathrm{total}}
\lesssim
e_{\mathrm{discrete}}
+e_{\mathrm{setup}}
+e_{\mathrm{alg}}
+e_{\mathrm{round}}
+e_{\mathrm{comm}}.
$$

减小 solver tolerance 只控制 `e_alg`，不能修复 gamma、Clover 或 gauge link 约定。

## D.4 奇偶 Schur、MATPC 与重建

### D.4.1 Full block system

按 parity 排列：

$$
D=
\begin{pmatrix}
A_p&D_{pq}\\
D_{qp}&A_q
\end{pmatrix},
\qquad
b=
\begin{pmatrix}
b_p\\
b_q
\end{pmatrix}.
$$

Schur complement：

$$
S_p=A_p-D_{pq}A_q^{-1}D_{qp}.
$$

若 `D=A-kappa H`，则

$$
S_p=A_p-\kappa^2H_{pq}A_q^{-1}H_{qp}.
$$

对称归一化 MATPC：

$$
M_p=A_p^{-1}S_p
=I-\kappa^2A_p^{-1}H_{pq}A_q^{-1}H_{qp}.
$$

### D.4.2 Prepare 与 reconstruct

通用 `D` block 写法：

$$
b_p^{\mathrm{prep}}
=A_p^{-1}\left(b_p-D_{pq}A_q^{-1}b_q\right),
$$

$$
x_q=A_q^{-1}(b_q-D_{qp}x_p).
$$

若使用 `D=A-kappa H`，符号写成

$$
b_p^{PC}
=A_p^{-1}\left(b_p+\kappa H_{pq}A_q^{-1}b_q\right),
$$

$$
x_q=A_q^{-1}\left(b_q+\kappa H_{qp}x_p\right).
$$

两种写法只差 off-diagonal 符号和 kappa 是否被吸收。

### D.4.3 Schur 求解算法

```text
Algorithm D.1: parity Schur solve
Input:
  full b=(b_p,b_q)
  onsite blocks A_p, A_q
  off-diagonal hopping D_pq, D_qp
  tolerance tau and full residual gate eta

1. solve A_q y_q = b_q
2. b_p^S = b_p - D_pq y_q
3. solve S_p x_p = b_p^S with selected Krylov method
4. reconstruct x_q = A_q^{-1}(b_q - D_qp x_p)
5. form x=(x_p,x_q)
6. compute r_true=b-D x
7. accept only if
      ||r_true||/||b|| <= eta
8. return x, r_true, iterations, breakdown status
```

### D.4.4 残差审计恒等式

若

$$
r=b-Dx,
$$

则精确解满足

$$
x=D^{-1}b.
$$

残差和误差满足

$$
\frac{\lVert x-x_\star\rVert}{\lVert x_\star\rVert}
\le
\kappa_2(D)
\frac{\lVert r\rVert}{\lVert b\rVert},
$$

因此残差小不总是等价于误差小，尤其条件数很大时。Schur residual、Arnoldi estimate 和 GCR iterated residual 均为不同量，必须由 full `r_true` 最终验收。

## D.5 Krylov 方法完整递推

### D.5.1 CG

适用条件是 `A` Hermitian positive definite。

```text
Algorithm D.2: preconditioned CG
Input: A, b, x0, preconditioner M^-1, tolerance tau
r0 = b - A x0
z0 = M^-1 r0
p0 = z0
rho0 = <r0,z0>
for k = 0,1,...:
    v = A p_k
    denom = <p_k,v>
    if denom <= 0 or nonfinite(denom): BREAKDOWN
    alpha = rho_k / denom
    x_{k+1} = x_k + alpha p_k
    r_{k+1} = r_k - alpha v
    if ||r_{k+1}|| <= tau ||b||:
        r_true = b - A x_{k+1}
        return if r_true gate passes
    z_{k+1} = M^-1 r_{k+1}
    rho_{k+1} = <r_{k+1},z_{k+1}>
    beta = rho_{k+1} / rho_k
    p_{k+1} = z_{k+1} + beta p_k
    rho_k = rho_{k+1}
```

对 `D^dagger D` 使用 CG 时，系统条件数从 `kappa_2(D)` 变为 `kappa_2(D)^2`。这可以避免非 SPD 问题，却可能显著增加迭代。

### D.5.2 多移位 CG

同时求解

$$
(A+\sigma_jI)x^{(j)}=b,\qquad j=1,\ldots,J.
$$

```text
Algorithm D.3: multi-shift CG
Input: A, b, shifts sigma_j, x_j=0
r = b
p_j = r
rho = <r,r>
for k = 0,1,...:
    Ap = A p_j                    # same A p_j reused
    for each shift j:
        denom_j = <p_j, Ap + sigma_j p_j>
        alpha_j = rho / denom_j
        x_j = x_j + alpha_j p_j
        r_j = r_j - alpha_j (Ap + sigma_j p_j)
    rho_new = <r_seed,r_seed>     # implementation-dependent seed
    beta = rho_new / rho
    for each shift j:
        p_j = r_j + beta p_j
    rho = rho_new
```

条件：`A` 的基准部分 SPD、移位为标量 `sigma_j I`。Clover 的非平凡 onsite block 或一般 Schur 不能未经验证直接套用。

### D.5.3 BiCG

BiCG 同时维护 `A` 和 `A^dagger` 的 Krylov sequence：

```text
Algorithm D.4: BiCG
r = b - A x
r_tilde = shadow residual
p = r
p_tilde = r_tilde
for k = 0,1,...:
    rho = <r_tilde,r>
    if rho = 0 or nonfinite: BREAKDOWN
    v = A p
    denom = <p_tilde,v>
    if denom = 0 or nonfinite: BREAKDOWN
    alpha = rho/denom
    x = x + alpha p
    r_new = r - alpha v
    r_tilde_new = r_tilde - alpha A^dagger p_tilde
    rho_new = <r_tilde_new,r_new>
    beta = rho_new/rho
    p = r_new + beta p
    p_tilde = r_tilde_new + beta p_tilde
    r = r_new
    r_tilde = r_tilde_new
```

其内存较低，但需要额外 dagger matvec，且 shadow breakdown 可能在非正规矩阵上发生。

### D.5.4 CGS

CGS 去掉显式 shadow vector，并让 residual polynomial 以平方形式递推：

```text
Algorithm D.5: preconditioned CGS
r = b - A x
r_tilde = choose_shadow(r)
rho_old = 1; u = 0; p = 0; q = 0
for k = 0,1,...:
    rho = <r_tilde,r>
    if rho == 0 or nonfinite: BREAKDOWN
    if k == 0:
        u = r
        p = u
    else:
        beta = rho/rho_old
        u = r + beta q
        p = u + beta(q + beta p)
    p_hat = M^-1 p
    v = A p_hat
    denom = <r_tilde,v>
    if denom == 0 or nonfinite: BREAKDOWN
    alpha = rho/denom
    q = u - alpha v
    u_hat = M^-1 (u+q)
    x = x + alpha u_hat
    r = r - alpha A u_hat
    rho_old = rho
```

该递推减少 shadow storage，但 residual polynomial 的平方会放大非正规矩阵上的舍入误差。

### D.5.5 BiCGStab

```text
Algorithm D.6: right-preconditioned BiCGStab with reliable update
Input:
  A, b, x0, preconditioner M^-1
  tolerance tau, max_iter, reliable_period
r = b - A x0
r_shadow = copy(r)
p = 0
v = 0
rho_old = alpha = omega = 1

for k = 0,1,...,max_iter-1:
    rho = <r_shadow,r>
    if nonfinite(rho) or |rho| <= breakdown_floor: BREAKDOWN
    beta = (rho/rho_old) * (alpha/omega)
    p = r + beta*(p - omega*v)

    p_hat = M^-1 p
    v = A p_hat

    denom = <r_shadow,v>
    if nonfinite(denom) or |denom| <= breakdown_floor: BREAKDOWN
    alpha = rho/denom
    s = r - alpha*v

    if ||s|| <= tau ||b||:
        x = x + alpha*p_hat
        r_true = b - A x
        if true_residual_gate(r_true): return x

    s_hat = M^-1 s
    t = A s_hat
    tt = <t,t>
    if nonfinite(tt) or real(tt) <= breakdown_floor: BREAKDOWN
    omega = <t,s>/tt
    if nonfinite(omega) or |omega| <= breakdown_floor: BREAKDOWN

    x = x + alpha*p_hat + omega*s_hat
    r = s - omega*t

    if reliable_update_due(k,reliable_period):
        r = b - A x

    if ||r|| <= tau ||b||:
        r_true = b - A x
        if true_residual_gate(r_true): return x

    rho_old = rho
```

`breakdown_floor` 必须结合 dtype、向量范数和全局归约精度设置，不能用一个与格点规模无关的常数。

### D.5.6 Arnoldi、GMRES 与 FGMRES

Arnoldi 关系：

$$
AV_m=V_{m+1}\bar H_m,
\qquad
\bar H_m\in\mathbb C^{(m+1)\times m}.
$$

MGS 一步：

$$
w=Av_j,
$$

$$
h_{ij}=\langle v_i,w\rangle,
\qquad
w\leftarrow w-h_{ij}v_i,\quad i=0,\ldots,j,
$$

$$
h_{j+1,j}=\lVert w\rVert,
\qquad
v_{j+1}=w/h_{j+1,j}.
$$

```text
Algorithm D.7: right-preconditioned FGMRES(m)
Input: A, b, x0, varying preconditioner M_j^-1, tolerance tau
r0 = b - A x0
beta = ||r0||
if beta == 0: return x0
v0 = r0/beta
g = (beta,0,...,0)^T

for j = 0,1,...,m-1:
    z_j = M_j^-1 v_j
    w = A z_j
    for i = 0,...,j:
        H[i,j] = <v_i,w>
        w = w - H[i,j] v_i
    H[j+1,j] = ||w||
    if H[j+1,j] != 0:
        v_{j+1} = w/H[j+1,j]
    apply_previous_givens(H[:,j])
    choose_givens(H[j,j],H[j+1,j])
    update_givens_rhs(g)
    if |g[j+1]| <= restart_tol:
        break

y = solve_upper_triangular(H[0:j,0:j],g[0:j])
x = x0 + sum_i y_i z_i
r_true = b - A x
if true_residual_gate(r_true): return x
else restart with r_true
```

若 `M_j` 变化，必须保存 `z_j`。不能用普通 GMRES 只保存 `v_j` 后按固定线性预条件器累加。

### D.5.7 GCR

预条件 GCR 的方向同时满足对 residual 的 `A`-inner-product 正交关系。简化算法：

```text
Algorithm D.8: preconditioned GCR
r = b - A x
for k = 0,1,...:
    z = M^-1 r
    v = A z
    for i = 0,...,k-1:
        beta_i = <v,v_i>/<v_i,v_i>
        z = z - beta_i z_i
        v = v - beta_i v_i
    denom = <v,v>
    if denom <= floor: BREAKDOWN
    alpha = <r,v>/denom
    x = x + alpha z
    r = r - alpha v
    z_k = z
    v_k = v
    if true_residual_gate(r): return x
```

历史 memory 随方向数增长，需 restart 或 block 化。

### D.5.8 CA-CG 与 CA-GCR

通信规避思想是以 `s` 步 Krylov basis

$$
\mathcal B_s(A,r)=\{r,Ar,\ldots,A^sr\}
$$

一次形成多个方向，再通过 block inner product 和小系统求解更新：

```text
Algorithm D.9: s-step CA-CG/CA-GCR skeleton
Input: A, r, block size s
1. form B_k = [r, A r, ..., A^s r]
2. local block orthogonalization / QR: B_k = Q_k R_k
3. compute all required block inner products in one reduction
4. solve small s x s least-squares/Hessenberg system
5. update x and r with Q_k coefficients
6. reliable-update full residual
7. reset Q_k when block conditioning or residual drift exceeds gate
```

优点是一次 global sync 摊销多个方向；风险是局部 block 条件数、失正交和 reduced precision。CA-GCR 不是无条件快路径。

### D.5.9 Lanczos 与 thick restart

Hermitian `H` 上的 Lanczos：

$$
Hq_j=\beta_{j-1}q_{j-1}+\alpha_jq_j+\beta_jq_{j+1}.
$$

三对角矩阵 `T_m` 的特征值给出 Ritz 值，Ritz vector 为

$$
y^{(i)}=Q_my_i.
$$

Ritz true residual 为

$$
r_i=Hy^{(i)}-\theta_iy^{(i)}.
$$

thick restart 时保留若干个 Ritz vectors，把新起始向量与其正交化，再继续 Lanczos。验收必须同时报告 `||r_i||`，不能只报告 Ritz 值。

## D.6 平滑器与局部预条件

### D.6.1 Richardson

$$
x_{k+1}=x_k+\omega(b-Ax_k),
\qquad
r_{k+1}=r_k-\omega Ar_k.
$$

误差传播矩阵为

$$
E_\omega=I-\omega A.
$$

稳定条件由 `A` 的谱决定。`omega` 过大或谱估计错误会放大误差。

### D.6.2 MR 平滑

在当前方向 `v` 上最小化新残差：

$$
\alpha_k=
\frac{\langle v_k,r_k\rangle}
{\langle v_k,v_k\rangle},
$$

$$
x_{k+1}=x_k+\alpha_kr_k,
\qquad
r_{k+1}=r_k-\alpha_kv_k.
$$

常见 `v=A r`、`v=A^dagger r` 或另外的 MR 方向。不同 dagger 约定的 alpha 复数相位不同，必须与 C++ kernel 一致。

### D.6.3 Chebyshev

给定谱区间 `[lambda_min,lambda_max]`，令

$$
d=\frac{\lambda_{\max}+\lambda_{\min}}2,
\qquad
c=\frac{\lambda_{\max}-\lambda_{\min}}2.
$$

标准二阶递推可写为

$$
\alpha_0=\frac1d,
$$

$$
\alpha_k=
\frac1{
d-\beta_{k-1}\frac{c^2}{4}
},
\qquad
\beta_k=\left(\frac{c\alpha_k}{2}\right)^2,
$$

$$
x_{k+1}=x_k+\alpha_kp_k,
\qquad
r_{k+1}=r_k-\alpha_kAp_k,
\qquad
p_{k+1}=r_{k+1}+\beta_kp_k.
$$

历史 CUDA MG 路径采用保守估计

$$
\hat\rho=\frac{\lVert Ar\rVert}{\lVert r\rVert},
\qquad
\lambda_{\max}=8\hat\rho,
\qquad
\lambda_{\min}=0.05\lambda_{\max},
$$

这不是严格谱包络证明。谱估计错误会导致过度放大或无效平滑。

### D.6.4 加性 Schwarz

把格点划分为子域 `Omega_i`，加性 Schwarz 预条件为

$$
M^{-1}=
\sum_iR_i^TA_i^{-1}R_i.
$$

乘性 Schwarz 按子域顺序复合

$$
M^{-1}=
I-\prod_i(I-R_i^TA_i^{-1}R_iA).
$$

子域越小，局部求解越便宜，但跨子域耦合越强。重叠越大，平滑质量通常越好，但通信和存储增加。

### D.6.5 SAP

SAP 在 even、odd 或颜色子域之间交替：

```text
Algorithm D.10: two-color SAP smoother
Input: A, b, x
r = b - A x
for color c in {0,1}:
    restrict r and A to subdomain c
    solve local subdomain system approximately
    prolong correction to full field
    update x
    recompute r
return x
```

SAP 更适合最近邻算子和 rank halo；跨 aggregate 耦合仍由 coarse space 处理。

### D.6.6 Smoother 的验收量

对 smoother `S`，每次应用应记录

$$
\rho_S=
\frac{\lVert S r\rVert}{\lVert r\rVert},
$$

并按 Fourier 或实验高频模检查误差传播。平滑次数太少残留高频，太多会把细层 matvec 成本耗尽。

## D.7 MultiGrid 完整算法

### D.7.1 Null-space basis

在每个 aggregate `A_X` 内拼接候选向量

$$
V_X=[v_1,\ldots,v_{n_v}].
$$

目标条件是

$$
V_X^\dagger V_X=I_E,
\qquad
\epsilon_{\mathrm{orth}}=
\max_X\lVert V_X^\dagger V_X-I_E\rVert.
$$

### D.7.2 P/R

$$
(P\phi)(x)=V_X(x)\phi_c(X),
$$

$$
(R\psi)(X)=
\sum_{x\in A_X}V_X(x)^\dagger\psi(x),
\qquad
R=P^\dagger.
$$

因此

$$
P^\dagger P=I_{\mathrm{coarse}},
\qquad
PP^\dagger\ne I_{\mathrm{fine}}
$$
一般成立。

### D.7.3 Galerkin

raw 路径：

$$
D_{l+1}=R_lD_lP_l.
$$

Strict PC 路径：

$$
D_{l+1}=R_lX_l^{-1}D_lP_l.
$$

若 coarse stencil 只支持 onsite 和一跳：

$$
D_{l+1}
=X_{l+1}
+\sum_\mu
\left(
Y_{l+1,\mu}^fT_{+\mu}
+Y_{l+1,\mu}^bT_{-\mu}
\right).
$$

`T_mu` 为 coarse 格点上的周期平移。

### D.7.4 V-cycle

```text
Algorithm D.11: recursive V-cycle
Vcycle(level, b_l, x_l):
    if level == coarsest:
        return coarse_solver(A_l, b_l, x_l)

    x_l = smoother_pre(A_l, b_l, x_l, nu_pre)
    r_l = b_l - A_l x_l
    b_{l+1} = R_l r_l
    e_{l+1} = Vcycle(level+1, b_{l+1}, 0)
    x_l = x_l + P_l e_{l+1}
    x_l = smoother_post(A_l, b_l, x_l, nu_post)
    return x_l
```

外层的右预条件版本只把 `Vcycle` 当作 `M^-1`。

```text
Algorithm D.12: right-preconditioned FGMRES + MG
x = x0
r = b - A x
for restart_cycle = 0,1,...:
    beta = ||r||
    if beta == 0: return x
    v0 = r/beta
    for j = 0,...,m-1:
        z_j = Vcycle(level=0, rhs=v_j, x=0)
        w = A z_j
        for i = 0,...,j:
            H_ij = <v_i,w>
            w = w - H_ij v_i
        H_{j+1,j} = ||w||
        if H_{j+1,j} != 0:
            v_{j+1} = w/H_{j+1,j}
        Givens_QR(H,g)
        if estimated_residual <= tau:
            break
    y = back_solve(H,g)
    x = x + sum_j z_j y_j
    r = b - A x
    if full_true_residual_gate(r):
        return x
```

### D.7.5 W/F/K-cycle

设 `C_l` 表示第 `l` 层 child solve，`V` 表示一次 child call，`F` 表示第一次 F、第二次 V，`K` 表示用固定小 FGMRES 包裹递归预条件器。

```text
V-cycle: child called once
W-cycle: child called twice
F-cycle: first recursive call uses F, second uses V
K-cycle: solve child equation by m-step FGMRES
```

K-cycle 的抽象递推为

$$
x_{l+1}^{(m)}=\operatorname{FGMRES}_m(A_{l+1},b_{l+1};M_{l+1}^{-1}).
$$

当前实现中 K-cycle 使用较小 `m`，不能把外层 FGMRES 和 K-cycle 的 iteration count 混为一个数。

### D.7.6 33 点 legacy/compact stencil

对 Schur operator，两次 hopping 可产生

| 位移类别 | 数量 |
|---|---:|
| onsite | 1 |
| 同轴 `±mu` | 8 |
| 双轴 `±mu±nu` | 24 |
| 总计 | 33 |

算子写为

$$
(A x)(X)
=S(X)x(X)
+\sum_{\mu}\left[
H_{\mu}^{f}(X)x(X+\hat\mu)
+H_{\mu}^{b}(X-\hat\mu)^\dagger x(X-\hat\mu)
\right]
+\sum_{\mu<\nu}\sum_{s_\mu,s_\nu\in\{-1,1\}}
D_{\mu\nu}^{s_\mu s_\nu}(X)x(X+s_\mu\hat\mu+s_\nu\hat\nu).
$$

33 是支持位置数；每个位置承载 `E x E` dense block。

### D.7.7 Strict `X/Y/Yhat`

构造：

$$
X_C(X)=\sum_{x\in A_X}V_X^\dagger C(x)V_X,
$$

$$
Y_\mu^f(X)
=-\kappa\sum_{x\in A_X}
(AV_X(x))^\dagger U_\mu(x)V_{X+\hat\mu}(x+\hat\mu),
$$

$$
Y_\mu^b(X-\hat\mu)
=-\kappa\sum_{x\in A_X}
V_X^\dagger
U_\mu^\dagger(x-\hat\mu)
AV_{X-\hat\mu}(x-\hat\mu).
$$

raw action：

$$
(D_cz)(X)
=X(X)z(X)
+\sum_\mu\left[
Y_\mu^f(X)z(X+\hat\mu)
+Y_\mu^b(X-\hat\mu)^\dagger z(X-\hat\mu)
\right].
$$

precondition：

$$
\widehat Y_\mu^f(X)=X^{-1}(X)Y_\mu^f(X),
$$

$$
\widehat Y_\mu^b(q-\hat\mu)
=Y_\mu^b(q-\hat\mu)X^{-\dagger}(q).
$$

### D.7.8 Strict coarse apply

```text
Algorithm D.13: strict preconditioned coarse hopping
Input:
  target point q
  target parity p_q
  input z on opposite parity p_in=1-p_q
  Yhat_f, Yhat_b storage

out(q) = 0
for mu = 0,...,3:
    out(q) += Yhat_f(q,mu) z(q+mu)
    out(q) += [Yhat_b(q-mu,mu)]^dagger z(q-mu)
return out(q)
```

这是对 target point 的权重读法。优化实现也可以对 source point 展开，但必须保持相同的 storage site、dagger 和 block multiplication order。

### D.7.9 Setup、colored probing 和缓存

```text
Algorithm D.14: distributed Galerkin setup
1. load global gauge/source/null slabs
2. build local periodic views
3. select non-conflicting source colors
4. batch fine operator probes
5. form Galerkin columns
6. check support is onsite + one hop
7. extract X and Yf/Yb
8. batch invert X
9. form Yhat_f=X^-1 Yf and Yhat_b=Yb X^-dagger
10. exchange required coarse/link halos
11. write one complete runtime-cache bundle
12. verify RP, orthogonality, inverse, support and true residual
```

colored batching的目的是一次探测多个互不冲突 source，降低 Python/C++ 调用、同步和峰值内存。它不改变最终 Galerkin 资产 identity。

## D.8 CUDA、MPI 与缓存算法

### D.8.1 Wilson Dslash kernel

```text
Algorithm D.15: Wilson hopping kernel
Input: fermion_in, gauge links, neighbor indexes
for each site x:
    acc = 0
    for mu = 0..3:
        psi_f = fermion_in[x+mu]
        psi_b = fermion_in[x-mu]
        proj_f = spin_project_plus(psi_f, mu)
        proj_b = spin_project_minus(psi_b, mu)
        acc += color_multiply(U(x,mu), proj_f)
        acc += color_multiply(U^dagger(x-mu,mu), proj_b)
    fermion_out[x] = acc
```

实际内核使用 batch、向量化和 register tiling。`1-gamma_mu` 与 `1+gamma_mu` 的成对行关系可复用中间结果。

### D.8.2 Halo pack/exchange/unpack

```text
Algorithm D.16: blocking halo
pack face for each forward/backward direction
post receives
send faces
receive self/peer data
unpack into ghost slots
apply interior terms
apply boundary correction terms
```

```text
Algorithm D.17: overlapped halo
record input-ready event on main stream
on exchange stream:
    wait event
    pack active faces
    stage H2D/D2H as required
    post MPI_Irecv/Isend
launch local base kernel on main stream
wait MPI requests
unpack/upload ghosts
record ghost-ready event
launch local tail/remote correction
```

active face 定义为当前 process grid 中 extent 大于 1 的方向和两个 side；未分解维度不发送。

### D.8.3 全局 inner product

局部向量内积：

$$
\ell=\sum_{x\in\Omega_r}
\langle u(x),v(x)\rangle.
$$

全局结果为

$$
g=\sum_{r=0}^{R-1}\ell_r.
$$

实现必须确保 kernel 消费的 device 标量与 host 记录值一致。`dot_pair` 合并两个内积，`dot_many` 将 Arnoldi 的多个系数放入一个大 Allreduce。

### D.8.4 分布式 FGMRES

```text
Algorithm D.18: distributed Arnoldi column
z_j = MG_precondition(v_j)
w = A z_j
local_coeff[i] = <v_i,w>
global_coeff = Allreduce(local_coeff)
w = w - sum_i global_coeff[i] v_i
beta = global_norm(w)
v_{j+1} = w/beta
update Hessenberg with global_coeff and beta
```

错误模式是 host `Hessenberg` 使用 global coeff，而 device orthogonalization 使用 rank-local coeff。必须在 Allreduce 后显式刷新 device coefficient。

### D.8.5 缓存写入

```text
Algorithm D.19: atomic strict runtime cache publish
1. canonicalize identity, metadata, stats and manifest JSON
2. compute suite SHA256 digests
3. create unique temporary file in destination directory
4. open one h5py File handle
5. write attrs state=writing
6. write each tensor in 8 MiB logical chunks
7. update per-tensor logical SHA256
8. write complete manifest and state=complete
9. flush and fsync file
10. atomically link temporary name to final identity path
11. if destination exists, verify identity; otherwise conflict
12. fsync directory
```

### D.8.6 缓存读取

```text
Algorithm D.20: strict runtime cache hit check
1. open read-only HDF5
2. verify root attribute set
3. verify state=complete
4. parse canonical identity/metadata/stats/manifest JSON
5. verify each JSON digest
6. verify schema and version
7. compare requested identity with cached identity
8. verify tensor manifest shape/dtype/nbytes
9. stream each tensor and recompute logical digest
10. H2D copy into requested device
11. return cache hit only after all checks
```

任何失败都标记为具体 cache miss reason，不静默使用部分文件。

## D.9 性能、误差与扩展性公式

### D.9.1 Solve 速度和加速比

正式性能比：

$$
R=\frac{t_{\mathrm{QUDA,steady}}}{t_{\mathrm{PyQCU,steady}}}.
$$

若比较 PyQCU 两个版本：

$$
S_{\mathrm{PyQCU}}=\frac{t_{\mathrm{old}}}{t_{\mathrm{new}}}.
$$

若比较 PyQCU 相对 QUDA 的加速倍率：

$$
S_{\mathrm{PyQCU/QUDA}}
=\frac{t_{\mathrm{QUDA}}}{t_{\mathrm{PyQCU}}}
=R.
$$

### D.9.2 True residual

$$
r=b-Dx,
\qquad
\rho_{\mathrm{rel}}=
\frac{\lVert r\rVert_2}{\lVert b\rVert_2}.
$$

当前正式门：

$$
\rho_{\mathrm{rel}}<5\times10^{-6}.
$$

Arnoldi least-squares estimate、GCR iterated residual 和 Schur residual 不能替代该门。

### D.9.3 Strong scaling efficiency

固定总问题，线程/rank 从 1 增至 `N`：

$$
E_{\mathrm{strong}}(N)
=\frac{T_1}{N\,T_N}.
$$

若基准不是 1，而是 `N0`，则

$$
E(N_0,N)
=\frac{N_0T_{N_0}}{N\,T_N}.
$$

### D.9.4 Weak scaling efficiency

保持每 rank 工作量近似固定：

$$
E_{\mathrm{weak}}(N)
=\frac{T_1}{T_N}.
$$

若每个新增 rank 也增加格点体积，则必须同时报告局部体积、总问题和通信时间。

### D.9.5 MG break-even

一次 MG 相比 1L direct Krylov 的收益条件：

$$
T_{\mathrm{fine}}(N_{1L}-N_{\mathrm{MG}})
>
N_V\left(
T_R+T_{A_c}+T_P+T_{\mathrm{sync}}
\right).
$$

左边是减少外层迭代节省的 fine matvec 时间，右边是每个 V-cycle 的 restriction、coarse apply、prolongation 和同步成本。该不等式不成立时，MG 即使减少迭代也可能更慢。

### D.9.6 显存模型

对 coarse 自由度 `E`、coarse 体积 `V_c`，dense onsite block 的内存近似为

$$
M_X=O(V_cE^2).
$$

若每层保存 `X`、`X^-1`、raw `Y`、`Yhat`，至少还要乘以方向和资产份数：

$$
M_{\mathrm{levels}}
\sim
M_V+M_X+M_{X^{-1}}+8M_Y+8M_{\hat Y}.
$$

实际峰值还包含 transfer、workspace、MPI staging、parent tensor 和 allocator cache。

## D.10 508 报告的物理公式、求解算法与优化

### D.10.1 格点作用量与 propagator

508 报告采用 Wilson kernel：

$$
2\kappa M[U]_{x,y}
=
\delta_{xy}
-\kappa\sum_\mu\left[
(1-\gamma_\mu)U_\mu(x)\delta_{x+\hat\mu,y}
+(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)
\delta_{x-\hat\mu,y}
\right].
$$

因此

$$
M=D_W.
$$

### D.10.2 Clover 项

$$
\sigma_{\mu\nu}F_{\mu\nu}
=
2\gamma_\mu\gamma_\nu
\frac14\sum_P
\frac12\left(U_P-U_P^\dagger\right).
$$

### D.10.3 Quasi-PDF 与两点函数

$$
\widetilde f(x,P_z,1/a)
=
\int\frac{dz}{4\pi}
e^{ixzP_z}
\left\langle P|
\bar\psi(z)\gamma_tU(z,0)\psi(0)
|P\right\rangle.
$$

$$
C_2(t,t_0)
=
\left\langle
\bar\psi(x,t)\Gamma\psi(x,t)
\bar\psi(y,t_0)\Gamma'\psi(y,t_0)
\right\rangle.
$$

Wick 收缩的代表项：

$$
C_2(t,t_0)
=-\gamma_5S_q(x,y)\gamma_5S_q(y,x),
\qquad
S_q=D_q^{-1}.
$$

### D.10.4 Even-odd Schur

508 报告使用：

$$
A'_{oo}=A_{oo}-\kappa^2D_{oe}A_{ee}^{-1}D_{eo},
$$

$$
b'_o=b_o+\kappa D_{oe}A_{ee}^{-1}b_e.
$$

求解：

$$
A'_{oo}x_o=b'_o,
$$

重建：

$$
x_e=A_{ee}^{-1}(b_e+\kappa D_{eo}x_o).
$$

这相当于当前通用 Schur 公式在特定符号和 kappa 归一化下的写法。

### D.10.5 CG 伪代码

```text
Algorithm D.21: CG used by the historical DCU implementation
x = x0
r = b - A x
p = r
rho = <r,r>
for k = 1..max_iter:
    if sqrt(rho) <= tolerance:
        break
    Ap = A p
    denom = <p,Ap>
    if denom == 0 or nonfinite: BREAKDOWN
    alpha = rho/denom
    x = x + alpha p
    r = r - alpha Ap
    rho_new = <r,r>
    beta = rho_new/rho
    p = r + beta p
    rho = rho_new
```

### D.10.6 BiCGStab 伪代码

```text
Algorithm D.22: historical BiCGStab
x = x0
r = b - A x
r_hat = r
p = 0; v = 0
rho_old = alpha = omega = 1
for k = 1..max_iter:
    rho = <r_hat,r>
    if rho == 0 or nonfinite: BREAKDOWN
    beta = (rho/rho_old)*(alpha/omega)
    p = r + beta*(p - omega*v)
    v = A p
    denom = <r_hat,v>
    if denom == 0 or nonfinite: BREAKDOWN
    alpha = rho/denom
    s = r - alpha v
    x = x + alpha p
    if ||s|| <= tolerance:
        break
    t = A s
    tt = <t,t>
    if tt == 0 or nonfinite: BREAKDOWN
    omega = <t,s>/tt
    x = x + omega s
    r = s - omega t
    rho_old = rho
```

历史实现还需对 `rho`、`omega` 和残差进行 finite guard，并周期性重算 full residual。

### D.10.7 四项核心优化

| 优化 | 原理 |
|---|---|
| Gamma 行复用 | `1±gamma_mu` 的行对关系允许重用中间结果，减少相关计算 |
| SU(3) 重构 | 只读取三行，利用正交和共轭关系构造另外三行，以计算换访存 |
| Even-odd | 求解半体积 Schur 系统，降低问题规模和 Krylov 迭代 |
| 合并访存 | 将时空维放最后、spin-color 连续，使线程访问连续地址 |

### D.10.8 `give_clover` 性能模型

单点内存和浮点量历史估计：

$$
B_{\mathrm{sp}}=1344\ \mathrm{B},\qquad
F_{\mathrm{sp}}=1152\ \mathrm{FLOPs},
$$

$$
B_{\mathrm{dp}}=2688\ \mathrm{B},\qquad
F_{\mathrm{dp}}=1152\ \mathrm{FLOPs}.
$$

报告使用

$$
T=\frac{M}{E}+\frac{F}{F_p}+T_0
$$

拟合得到

$$
E\approx424.258\ \mathrm{GiB/s},
\qquad
F_p\approx1.533\ \mathrm{TFLOP/s}.
$$

该拟合未给出误差条，也未明确单卡/单节点/全系统的归一化。因此只能作为平台诊断，不是硬件峰值测量。

### D.10.9 强扩展和弱扩展重算

若以 `N0` 进程为基准，强扩展效率为

$$
E_{\mathrm{strong}}(N_0,N)
=
\frac{N_0T_{N_0}}{N\,T_N}.
$$

以原始耗时重算的历史趋势大致为：

| 测试 | 路径 | 低并发 | 高并发 |
|---|---|---:|---:|
| TEST6 c64 | Wilson | np2 约 96.5% | np64 约 8.6% |
| TEST6 c64 | Clover | np2 约 95.0% | np16 约 62.0% |
| TEST8 c64 | Wilson | np512 约 61.5% | np1024 约 20.8% |
| TEST8 c64 | Clover | np1024 | 约 51.4% |
| TEST9 c128 | Wilson | np1024 | 约 28.5% |
| TEST9 c128 | Clover | np1024 | 约 60.7% |

弱扩展固定每卡等效格点数时，历史趋势为：

| 精度 | 算子 | 较好区间 | 最低近似 |
|---|---|---:|---:|
| c64 | Wilson | 8.39M sites/GPU 时约 84.8% | 约 34.2% |
| c64 | Clover | 8.39M sites/GPU 时约 81.8% | 约 67.8% |
| c128 | Wilson | 4.19M sites/GPU 时约 62.3% | 约 30.3% |
| c128 | Clover | 4.19M sites/GPU 时约 74.6% | 约 58.2% |

因此正确结论是“小规模低并发较好，高并发时 Wilson 路径明显衰减，Clover 路径相对平缓”，而不是笼统“扩展性良好”。

### D.10.10 508 数据口径警示

| 问题 | 处理 |
|---|---|
| TEST6 c64 np32 Clover 为 NaN | 从加速比剔除，保留失败证据 |
| c64 残差平方约 `1e-11`，声明门为 `1e-12` | 不得宣称该点通过声明门 |
| 每项基本只有一次运行 | 无 MAD/置信区间，低于当前 1/2/5 协议 |
| 正文环境为 32-core/128 GB | 嵌入环境记录为 192-core/约 500.8 GiB，须分栏 |
| 正文旧软件栈与嵌入环境不一致 | 回放时以实际环境脚本和 `lscpu/meminfo` 为准 |
| 原始表“加速比”和耗时比值不完全一致 | 两者都保留并标明定义 |

## D.11 其他格点费米子作用量的对照公式

这些作用量在当前文档中用于理论边界和 QUDA 对照。除 Wilson/Clover 外，不应把它们写成 PyQCU 已完整支持的生产 API。

### D.11.1 Twisted-mass

简并双重态：

$$
D_{\mathrm{tm}}
=D_W(m_0)+i\mu\gamma_5\tau_3.
$$

非简并双重态：

$$
D_{\mathrm{nd}}
=D_W(m_0)
+i\mu_\sigma\gamma_5\tau_1
+\mu_\delta\tau_3.
$$

`tau_i` 作用于 flavor 空间。Even-odd block 必须把 flavor-spin onsite 结构放入 `A_e`、`A_o`。QUDA 参考入口包括 `dirac_twisted_mass.cpp` 和 `dslash_ndeg_twisted_mass.cpp`。

### D.11.2 Twisted-clover

$$
D_{\mathrm{tmC}}
=D_W+C_{\mathrm{SW}}
+i\mu\gamma_5\tau_3.
$$

Clover 与 twist 都是 onsite 项，粗化时应共同进入 `X_c`，不能再把它们当作 hopping 重复加入。

### D.11.3 Staggered

相位为

$$
\eta_\mu(x)=(-1)^{\sum_{\nu<\mu}x_\nu}.
$$

无质量 staggered operator：

$$
(D_{\mathrm{stag}}\chi)(x)
=
\frac12\sum_\mu\eta_\mu(x)
\left[
U_\mu(x)\chi(x+\hat\mu)
-U_\mu^\dagger(x-\hat\mu)\chi(x-\hat\mu)
\right].
$$

asqtad/HISQ 通过 Fat7、Lepage、Naik 等路径扩展 support。宽 stencil 不能直接塞进只接受最近邻 `X/Y` 的 Strict ABI。

### D.11.4 Domain-wall 与 Möbius

Domain-wall 的第五维示意矩阵为

$$
\begin{aligned}
D_{5d}(s,s')
={}&D_W^+(s)\delta_{s,s'}
-P_-\delta_{s+1,s'}
-P_+\delta_{s-1,s'}\\
&+m_f\left(
P_-\delta_{s,0}\delta_{s',L_s-1}
+P_+\delta_{s,L_s-1}\delta_{s',0}
\right),
\end{aligned}
$$

其中

$$
P_\pm=\frac{1\pm\gamma_5}{2}.
$$

有限 `L_s` 产生 residual mass。Möbius 通过第五维系数重参数化，改善 sign-function 逼近效率。第五维会改变站点自由度和 halo 结构。

### D.11.5 Overlap

$$
D_{\mathrm{ov}}(m)
=
\left(1-\frac{am}{2\rho}\right)D_{\mathrm{ov}}(0)+am,
$$

$$
D_{\mathrm{ov}}(0)
=\rho\left[1+\gamma_5\operatorname{sign}(H_W)\right],
\qquad
H_W=\gamma_5(D_W-\rho).
$$

`sign(H_W)` 常用 rational approximation 或 polynomial 近似，每次应用可能包含多次 kernel solve。它不是当前 Wilson/Clover Dslash 的同一成本模型。

### D.11.6 作用量对照

| Action | 自由度/support | 主要数值难点 | 当前状态 |
|---|---|---|---|
| Wilson | spin-color、最近邻 | 轻质量条件数、加性质量重整化 | PyQCU 实现 |
| Clover | Wilson 加 onsite block | 局部 inverse、非平凡 Schur | PyQCU 实现 |
| Twisted-mass | flavor-spin onsite | flavor block、dagger 约定 | QUDA 参考 |
| Twisted-clover | Clover 加 twist | 更大 onsite block、非 Hermitian | QUDA 参考 |
| Staggered | 单分量、phase stencil | taste、rooting 语义 | QUDA 参考 |
| asqtad/HISQ | 宽 stencil | 多路径通信、宽 support | QUDA 参考 |
| Domain-wall | 第五维局部耦合 | `L_s` 成本、残余质量 | QUDA 参考 |
| Möbius | 五维重参数化 | 第五维谱逼近 | QUDA 参考 |
| Overlap | sign function 非局部近似 | 多层嵌套求解 | 理论参考 |

## D.12 端到端可审计求解算法

### D.12.1 输入和不变量

```text
Algorithm D.23: auditable solve input validation
Input:
  U_mu(x), rhs b, mass/kappa, lattice, boundary, dtype, layout

1. verify all lattice dimensions and MPI grid are positive
2. verify spacing axes are the final four trailing axes
3. verify gauge color axes are 3x3
4. verify dtype and precision match params
5. verify SU(3) links:
      max_x |det U_mu(x)-1| << 1
      max_x ||U_mu(x)^dagger U_mu(x)-I|| << 1
6. verify source normalization and boundary conditions
7. verify compiler binary architecture and cache identity
8. fail closed before allocating runtime scratch if any check fails
```

### D.12.2 算子准备

```text
Algorithm D.24: operator preparation
1. build periodic neighbor table
2. load/construct gauge halo
3. construct Wilson hopping H
4. if Clover enabled:
       build plaquette/Clover field
       build T_p and A_p
       batch-invert A_p
5. partition even/odd parity
6. verify operator identities:
       D^dagger = gamma5 D gamma5
       parity round trip
       Schur reconstruction
7. verify free-field and c_sw=0 limits
```

### D.12.3 Full solve

```text
Algorithm D.25: full periodic Wilson/Clover solve
1. choose x0 (zero or warm)
2. compute r0 = b - D x0
3. if ||b|| == 0 return x=0
4. choose solver:
       BiCGStab for general D
       FGMRES for varying preconditioner
       CG only for SPD D^dagger D or Hermitian operator
5. precondition:
       plain = identity
       Schur/MATPC = parity block solve
       MG = V/W/F/K hierarchy
6. iterate until estimate reaches tolerance
7. periodically recompute r_true = b - D x
8. if true residual gate fails:
       reset recurrence or restart Krylov
9. return x, full true residual, iteration ledger, phase times
```

### D.12.4 Strict MG setup solve

```text
Algorithm D.26: strict MG setup and solve
1. load canonical gauge, rhs and full near-null vectors
2. validate block/aggregate geometry and checkerboard parity
3. construct local orthogonal V
4. distribute rank slices and exchange setup halos
5. compute strict D_c = R X^-1 D P
6. extract X, Yf, Yb; fail on wider support
7. batch invert X
8. form Yhat_f=X^-1 Yf, Yhat_b=Yb X^-dagger
9. persist one immutable runtime cache bundle
10. initialize persistent hierarchy and dot/FGMRES workspaces
11. run right-preconditioned FGMRES
12. each Arnoldi column calls one recursive V-cycle
13. use global dot, dot pair and dot many in distributed mode
14. at each restart compute full true residual
15. stop only when full true residual passes
16. close hierarchy and release streams/HDF5/pointers
```

### D.12.5 资源回收

```text
Algorithm D.27: deterministic teardown
1. finish final true-residual and status record
2. stop new kernels and MPI collectives
3. call strict end on hierarchy
4. release coarse assets, null vectors and scratch
5. close HDF5 handles
6. release per-thread stream and device context state
7. clear set_ptrs handles
8. check live allocation baseline and set-index state
```

### D.12.6 可复现运行记录

```text
Algorithm D.28: evidence bundle completion
1. save commit/tag and dirty status
2. save source and binary SHA256
3. save requested/actual CUDA architecture
4. save gauge/source/null identity and bundle hash
5. save MPI grid, rank-device map and thread count
6. save cold, warmup, steady raw timings
7. save iteration and full true residual per sample
8. save phase/level timing and device memory
9. save error codes and substitutions
10. only then mark fair=true
```

该附录保留了当前文档中公认可复用的公式、递推和算法骨架。若源码 convention、容差、storage site 或 parallel reduction order 变化，相关公式必须重新逐项核对，而不是仅更新结论数字。

## D.13 算子等价、Galerkin 与残差审计公式

### D.13.1 算子等价

对两个实现 `A1`、`A2` 和测试向量 `v`，定义

$$
\epsilon_{\mathrm{op}}(v)
=
\frac{\lVert A_1v-A_2v\rVert_2}
{\max(\lVert A_1v\rVert_2,\lVert A_2v\rVert_2,\epsilon_{\mathrm{floor}})}.
$$

该检查至少覆盖：

1. Python 与 C++ Wilson/Clover；
2. full 与 Schur reconstruct；
3. raw `X/Y` 与 preconditioned `Yhat`；
4. 单 rank 与多 rank halo；
5. 不同 dtype 和不同 process grid。

### D.13.2 Transfer 伴随关系

对任意 coarse `z` 和 fine `f`，应满足

$$
\langle Pz,f\rangle
=
\langle z,Rf\rangle,
\qquad
R=P^\dagger.
$$

数值指标：

$$
\epsilon_{PR}
=
\frac{
\left|\langle Pz,f\rangle-\langle z,Rf\rangle\right|
}{
\max(1,
|\langle Pz,f\rangle|,
|\langle z,Rf\rangle|)
}.
$$

### D.13.3 Schur 等价

令 `E` 为 prepare 映射、`J` 为 reconstruct 映射、`S` 为 compact operator，要求

$$
S=EDJ,
\qquad
DJy=\widetilde J Sy.
$$

数值检查：

$$
\epsilon_{\mathrm{Schur}}
=
\frac{\lVert E(DJy)-Sy\rVert_2}
{\max(\lVert Sy\rVert_2,\epsilon_{\mathrm{floor}})}.
$$

同时必须用 reconstruct 后的 full residual 验证。

### D.13.4 Galerkin 与 support

$$
y_1=(RA_fP)v_c,
\qquad
y_2=A_cv_c.
$$

$$
\epsilon_{\mathrm{Galerkin}}
=
\frac{\lVert y_1-y_2\rVert_2}
{\max(\lVert y_1\rVert_2,\lVert y_2\rVert_2,\epsilon_{\mathrm{floor}})}.
$$

若 `A_c` 是最近邻截断：

$$
\epsilon_{\mathrm{support}}
=
\frac{\lVert(RA_fP-A_c)v\rVert_2}
{\max(\lVert RA_fPv\rVert_2,\epsilon_{\mathrm{floor}})}.
$$

对 Strict ABI，`epsilon_support` 不是可容忍数值误差。只要出现超一跳模型项，必须 fail-closed。

### D.13.5 预条件器质量

对 residual `r` 和 correction `z=M^-1r`：

$$
q_M=\frac{\lVert r-Az\rVert_2}{\lVert r\rVert_2}.
$$

`q_M` 小表示单次预条件强，但必须同时报告 V-cycle 成本。一个强而昂贵的预条件器可能使总 solve 更慢。

### D.13.6 Backward error

若 `||A||` 可估计，则

$$
\eta(x)=
\frac{\lVert b-Ax\rVert_2}
{\lVert A\rVert_2\lVert x\rVert_2+\lVert b\rVert_2}.
$$

报告中写 “tolerance = 1e-8” 时必须说明是 absolute、relative、backward error、Arnoldi estimate 还是 full true residual。

### D.13.7 Krylov 多项式视角

固定预条件器时

$$
e_k=p_k(M^{-1}A)e_0,
\qquad
p_k(0)=1.
$$

SPD CG 的经典上界：

$$
\frac{\lVert e_k\rVert_A}{\lVert e_0\rVert_A}
\le
2\left(
\frac{\sqrt{\kappa(A)}-1}
{\sqrt{\kappa(A)}+1}
\right)^k.
$$

该上界不直接适用于非正规、非 Hermitian、有限精度或变化预条件路径。

### D.13.8 粗空间对低模的覆盖

投影算子

$$
\Pi=P(RP)^{-1}R.
$$

覆盖误差：

$$
\epsilon_{\mathrm{low}}
=
\max_{i\in I}
\frac{\lVert(I-\Pi)u_i\rVert_2}{\lVert u_i\rVert_2}.
$$

选择 block/nvec 时至少同时扫描

$$
(n_v,\text{block volume},E,\epsilon_{\mathrm{orth}},\epsilon_{\mathrm{low}}).
$$

### D.13.9 通信量模型

对四维局部 lattice `Lx*Ly*Lz*Lt`，每个方向有两个 face。若每个 face 有 `n_face` 个复元素，则一次最近邻 halo 的量级为

$$
N_{\mathrm{halo}}
\approx
2n_{\mathrm{face}}
\left(
L_yL_zL_t+L_xL_zL_t+L_xL_yL_t+L_xL_yL_z
\right).
$$

Coarse `Y` 是 dense `E x E` block，因此通信字节数近似随 `E^2` 增长。

### D.13.10 显存顶层预算

$$
M_{\mathrm{peak}}
\simeq
M_{\mathrm{fine}}
+M_{\mathrm{null}}
+M_{\mathrm{coarse}}
+M_{\mathrm{workspace}}
+M_{\mathrm{allocator}}.
$$

最后一项不是 live tensor。加载 cache 前后、setup、first solve、steady 和 close 后都应分别采样。

### D.13.11 谱分解与误差传播

若 `A=VΛV^-1`，则

$$
p(A)=Vp(\Lambda)V^{-1}.
$$

因此误差传播不仅由单个迭代多项式值决定，也由 `V` 的条件数控制。非正规矩阵中，即使所有特征值小于 1，短期 norm 仍可能增长。

### D.13.12 Cycle 的算子形式

令 `C_l^V/W/F` 表示第 `l` 层的 coarse correction，`S_pre/post` 为平滑器：

$$
C_\ell^V
=
S_{\mathrm{post}}
P_\ell C_{\ell+1}^V R_\ell
S_{\mathrm{pre}},
$$

$$
C_\ell^W
=
S_{\mathrm{post}}
P_\ell C_{\ell+1}^W R_\ell
P_\ell C_{\ell+1}^W R_\ell
S_{\mathrm{pre}},
$$

$$
C_\ell^F
=
S_{\mathrm{post}}
P_\ell C_{\ell+1}^F R_\ell
P_\ell C_{\ell+1}^V R_\ell
S_{\mathrm{pre}}.
$$

W-cycle 在同一层调用两次 child，通常粗层成本更高。F-cycle 第一次 child 使用 F、第二次使用 V。K-cycle 使用短 FGMRES 包裹 child preconditioner。

### D.13.13 目标时间比较

若一次 V-cycle 调用 `N_R` 次 restriction、`N_P` 次 prolongation、`N_A` 次 coarse apply，则

$$
T_{\mathrm{Vcycle}}
\approx
N_pT_{\mathrm{pre}}
+N_RT_R
+N_AT_{\mathrm{coarse}}
+N_PT_P
+N_qT_{\mathrm{post}}
+T_{\mathrm{sync}}.
$$

外层总时间还要乘以有效预条件调用次数：

$$
T_{\mathrm{solve}}
\approx
N_{\mathrm{outer}}T_{\mathrm{Vcycle}}
+T_{\mathrm{fine matvec}}
+T_{\mathrm{outer sync}}
+T_{\mathrm{true\ residual}}.
$$

这解释了为什么“外层迭代数更少”不必然意味着总时间更短。

## D.14 公式与算法完整性检查

| 主题 | 已保留内容 |
|---|---|
| 连续与格点 QCD | Lagrangian、准 PDF、两点函数、Wick 收缩、UV/IR |
| Wilson/Clover | covariant shift、hopping、作用量、kappa、Clover block、gamma5-Hermiticity、自由场谱 |
| Schur/MATPC | block elimination、prepare、reconstruct、full residual |
| Transfer/MG | null space、P/R、Galerkin、Strict X/Y/Yhat、33 点、V/W/F/K |
| Krylov | CG、multishift CG、BiCG、CGS、BiCGStab、GMRES/FGMRES、GCR、CA-CG/CA-GCR、Lanczos |
| Smoother | Richardson、MR、Chebyshev、Schwarz、SAP |
| CUDA/MPI | Dslash、halo、overlap、global dot、distributed FGMRES、setup |
| Cache/IO | atomic publish、identity、manifest、digest、rank slab |
| Error/perf | true residual、backward error、Galerkin/support error、strong/weak scaling、break-even、memory |
| 508 historical | Wilson kernel、Clover、quasi-PDF、Schur、CG、BiCGStab、优化、带宽/FLOPs 拟合 |

本表用于证明公式和算法章节没有被压缩掉。若后续源码改变 ABI、归一化、storage 或 reduction order，应先更新对应公式，再更新性能结论。
