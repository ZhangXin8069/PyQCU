# 33-tensor cache 与 Strict cache 对比

本文记录 PyQCU 两条 MultiGrid 粗化路径的 cache 语义。用户询问 cache 内容、选择依据、
HDF5 布局、`set_ptrs` 槽序、加载边界或性能计时口径时读取本文。

## 结论

两者不是同一种 cache 的两种序列化格式：

- **33-tensor cache** 面向 legacy/compact-Schur MG，直接缓存粗算子
  `A_c=P^\dagger S P`。
- **Strict cache** 面向 QUDA-style strict MG，缓存预条件层级资产
  `V/Yhat/(X,X^-1)`，运行时以 full-coarse onsite + 8-link 形式作用。

两者可以描述相关的 Galerkin 物理，但 geometry、槽序、C++ kernel 和生命周期均不同，
不能直接互换。

## 核心对比

| 维度 | 33-tensor cache | Strict cache |
|---|---|---|
| 数学对象 | 显式 `A_c=P^\dagger S P` | `D=X+H`，使用 `Yhat=X^-1Y` |
| 粗格 geometry | compact odd coarse lattice | full coarse lattice |
| parity | 粗层本身为 odd/compact | 仅 fine R/P、MATPC 边界使用 parity |
| 位移类别 | 1 onsite + 8 nearest + 24 diagonal | onsite + 8 个正负方向 link |
| 主要数据 | `lonv/hnn/hdg/sit` | `fine_blocked_v`、`Yhat`、`(X,X^-1)`、递归 `V` |
| raw link | 不适用 | 可选；普通 solve 不驻留 |
| cache 内容 | “算子结果” | “预条件算子因子” |
| 运行期槽位 | `[lonv, hop_nn, hop_diag, sitting]` | `[V, raw Y(optional), Yhat, (X,X^-1)]` |

## 33-tensor cache

每个 transition 保存：

```text
lonv: near-null vectors
sit:  [E,E,Xc,Yc,Zc,Tc]
hnn:  [2,4,E,E,Xc,Yc,Zc,Tc]       # 8 个最近邻
hdg:  [2,2,6,E,E,Xc,Yc,Zc,Tc]     # 24 个双轴对角项
```

`hnn/hdg/sit` 是三个数组，但逻辑上包含 33 个位移 block。矩阵主体大小为：

```text
33 * E^2 * Vc_odd
```

另外保存 `lonv`。当前 simple builder 固定两级、即一条 transition；多层 MG 必须为每条
transition 各保存一套。HDF5 写入只包含 `lonv/hnn/hdg/sit`，构建报告在外部 JSON 中记录
gauge SHA256 和物理参数，调用方必须按身份隔离路径。

实现锚点：

- `pyqcu/tools/_multigrid.py:739`：Python stencil 构建与形状。
- `cpp/cuda/qcu/src/multigrid.cu:858`：wide-stencil CUDA kernel。
- `pyqcu/testing/simple/kupl_multigrid.py:172`：`lonv/hnn/hdg/sit` 读取。
- `pyqcu/testing/simple/kupl_multigrid.py:211`：四 dataset 写入。

## Strict cache

每条 transition 保存：

```text
V:      blocked near-null vectors
Yhat:   [2,4,E,E,Xc,Yc,Zc,Tc]
X/X^-1: [2,E,E,Xc,Yc,Zc,Tc]
```

Strict 的粗层不做 checkerboard crop，`P/R` 只在 fine parity 与 full coarse field
之间转换。其粗化关系为：

```text
D_{l+1} = R (X_l^-1 D_l) P
```

普通 solve 只需要 `Yhat` 和 `(X,X^-1)`；raw `Y` 用于 setup/诊断，默认不驻留。若层级有
`T` 条 transition，runtime cache 通常有：

```text
1 fine V + 2T Yhat/onsite + (T-1) recursive V = 3T logical tensors
```

`T=1` 是 3 个逻辑 tensor，`T=2` 是 6 个。这里的 logical tensor 与 33 个 block 使用不同
计数口径。

实现锚点：

- `pyqcu/solver/_quda_multigrid.py:1506`：`X/Y/Yhat` packed ABI。
- `pyqcu/solver/_quda_multigrid.py:3589`：strict transition 四槽导出。
- `pyqcu/cuda/_strict_cache.py:1`：schema v2 cache 与不可变发布。
- `pyqcu/cuda/_strict_cache.py:1074`：完整验证后加载。
- `cpp/cuda/qcu/src/apply_multigrid_strict.cu:374`：strict coarse apply kernel。

## 存储量

33-path 每 transition 的矩阵 block 数为：

```text
33 * E^2 * Vc_odd
```

Strict runtime 每 transition 为：

```text
(8 Yhat + 2 onsite) * E^2 * Vc_full + 对应 V
```

不能只比较 `33` 与 `10`。标准每轴 block=2 时，通常：

```text
Vc_full ~= 2 * Vc_odd
```

因此 33 block 与 10 full-coarse block 的实际比例要同时考虑 geometry。若 Strict 额外保留
raw `Y`，还会增加 8 个 link block。

## Cache 完整性

当前 legacy simple cache 的契约是四个 dataset 必须存在；物理身份依赖构建报告、gauge
SHA256、目录隔离和调用方约定。

Strict cache schema v2 内建：

- 不可变原子发布；
- identity、metadata、stats、manifest；
- 每个 logical tensor 的流式 SHA256；
- dataset attrs、shape、dtype、nbytes 和内容摘要交叉验证；
- 完整验证后再执行 device transfer；
- 损坏或不匹配时返回明确 miss/fail-closed。

Strict cache hit 会读取一次内容做校验、再读取一次做传输，不能把这段验证 I/O 混入纯 solve
时间。

## 生命周期与 ABI

33-path 使用 legacy coarse slots：

```text
[lonv, hop_nn, hop_diag, sitting]
```

Strict 每 transition 使用：

```text
[V, raw Y(optional), Yhat, (X,X^-1)]
```

Strict 在私有 `LatticeSet` slot 中完成 init、VCycle/FGMRES 和 end，整个生命周期保持
`_SET_INDEX_` 不变。legacy 继续遵循普通事务式生命周期。不可把一个路径的槽序直接交给另一
路径。

## 选择建议

- 需要 compact-Schur、一次 wide kernel 表达完整两跳粗作用：使用 33-tensor cache。
- 需要 QUDA-style full-coarse、strict FGMRES 和更强 cache 完整性：使用 Strict cache。
- 做 QUDA 公平对照：Strict cache 是匹配的层级表示；不要把 33-tensor cache 当成 QUDA
  的 `vec_load`。
- cache-hit 性能应分别报告验证/加载、setup、首次 workspace 分配、steady solve 和真残差。

## 反模式

- 将 `lonv/hnn/hdg/sit` 改名或重排后冒充 strict `V/Yhat/onsite`。
- 对 Strict coarse asset 再做 checkerboard，或把 compact odd geometry 传给 strict。
- 把 `_SET_PTRS_STRICT_*` 槽序用于 legacy 33-tensor kernel，或反向使用。
- 只报 cache-hit solve 而隐藏 33 setup 或 Strict schema-v2 验签/加载开销。
- 在没有 gauge/null-vector/physics identity 的情况下跨配置复用 cache。

## 验证

```bash
python -m pytest -q pyqcu/testing/qcu/strict/quda_comparison/test_strict_runtime_cache.py
python -m pytest -q pyqcu/testing/simple/test_coarse_cache.py
python -m pytest -q pyqcu/testing/python/test_quda_multigrid.py \
  -k 'qcu_stencil_pack_matches_all_33_periodic_slots or qcu_stencil_degenerate_extent_and_strict_support_guard'
```

2026-10-21 基线：分别为 `23 passed`、`2 passed`、`2 passed, 32 deselected`。
