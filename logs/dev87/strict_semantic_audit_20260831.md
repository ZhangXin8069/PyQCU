# PyQCU Strict MultiGrid 语义审计

日期：2026-08-31
对象：`PyQCU` Strict MultiGrid 与 `refer/git-rep/quda` 的 MultiGrid 语义对照。
范围：各层 `P/R`、Clover/Gauge、完整粗算子、奇偶 Schur、预处理、显存和验收闸门。

## 结论摘要

Strict 路径已经在 PyQCU 中形成独立实现，并保留原有 legacy 路径。其核心
结构与 QUDA 的 `DiracCloverPC`、`DiracCoarsePC` 和 coarse-op kernel 一致：

\[
 D_c=R\,(X_f^{-1}D_f)\,P,
 \qquad \widehat D_c=X_c^{-1}D_c=I+X_c^{-1}Y_c .
\]

当前可以确认 Python/CUDA primitive、细层非平凡 Clover MATPC、full-coarse
传输、递归 V-cycle 和正式大格 Strict solve 的闭环。固定的 V100/c64/odd-odd
协议下，正式 benchmark `fair=true` 且 PyQCU 比 QUDA 快约 3.57%；这不是跨设备
的普遍性能承诺。粗层 backward `Yhat` 的存储位置/伴随/坐标偏移仍需要非平凡
Gauge/Clover 的逐项对照，且正式 benchmark 已固定使用同一 MATPC parity。

## 1. 层级与传输

- 每个层级的 coarse 场保持完整 coarse lattice；只有 `R/P` 调用选择 fine
  parity，不能把 coarse 资产本身压成半格。
- blocked `P` 的 fine dof 为 12，Clover/Wilson 的 coarse spin 为 2，因而
  `E=2*nvec`。`R` 是 `P†`，restriction 只读取选定的 fine parity。
- CUDA strict kernel 的递归入口使用 child 的完整 coarse volume；延拓后再
  回到父层的 compact parity。奇偶裁剪发生在 MATPC/prepare/reconstruct
  边界，而不发生在层级资产定义中。
- QUDA 对应 `Transfer::P/R`、`createGeoMap/createSpinMap` 及
  `DiracCoarse` 的 full-coarse 组织。

证据：

- `pyqcu/solver/_quda_multigrid.py`
- `pyqcu/tools/_strict_galerkin.py`
- `cpp/cuda/qcu/src/apply_multigrid_strict.cu`
- `refer/git-rep/quda/lib/dirac_coarse.cpp`
- `refer/git-rep/quda/lib/dirac_clover.cpp`

## 2. Fine Clover、Gauge 与 Schur

细层写成 `D=A-κH`，其中 `A` 是 even/odd Clover onsite，`H` 是带 Gauge
方向的 hopping。目标 parity `p` 的 Schur 为

\[
 S_p=A_p^{-1}\left(A_p-\kappa^2H_{pq}A_q^{-1}H_{qp}\right).
\]

RHS prepare 与另一 parity reconstruct 分别使用同一组 Clover inverse；这与
QUDA 的 Clover MATPC block elimination 一致。Gauge 的 forward link 位于源
点，backward gather 使用前一站点 link 的 dagger：

\[
H\psi(x)=\sum_\mu[(1-\gamma_\mu)U_\mu(x)\psi(x+\hat\mu)
 +(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)].
\]

Clover 的 `m+4=1/(2κ)` 整体归一化差异属于 PyQCU/QUDA 接口约定，不能当作
Clover 系数错误。当前没有发现 fine Gauge 方向必须修正的证据。

## 3. Coarse `X/Y/Yhat` 与预处理方向

粗层先对完整场算子做 Galerkin 投影：

\[
 D_c=R A_fP,
 \quad A_f=X_f^{-1}D_f\ \text{（Strict production path）}.
\]

得到 onsite `X_c` 和 hopping `Y_c` 后，Strict runtime 保存 `X_c`、`X_c^{-1}`
以及 `Yhat=X_c^{-1}Y_c`。粗层 MATPC 再对 `I+Yhat` 做两次跨 parity hopping，
得到 `I-Hhat_pq Hhat_qp`。

QUDA 的粗层 backward link 存放在 `q-μ`，gather 时取 dagger。PyQCU 的 packed
storage 采用同一意图；该方向目前已有 synthetic link 和 CUDA primitive
测试，但最终的非平凡 Gauge/Clover QUDA 数值锚定仍是开放项。

要区分两种预处理：

1. coarse operator 内部是左预处理 `X_c^{-1}D_c`，形成 `Yhat`；
2. Strict 外层 fused solver 是右预处理 FGMRES：先算 `z=M^{-1}v`，再用
   完整细层 `D z`，并以 `z` 更新解。

## 4. 显存策略

Strict runtime 默认不驻留 raw `Y`，只绑定执行所需的 packed transfer、
coarse `Yhat`/onsite 和跨层 blocked `V`。融合外层 Krylov 只创建一个可复用
arena，预算为

```text
(2*m + 5) * B_f + 2 * B_c
```

其中 `B_f` 是 compact fine-parity vector 字节数，`B_c` 是第一粗层完整向量
字节数。该 arena 首次求解懒分配；setup 资产与首次 solve workspace 分开
记账，Python 不再复制一份外层 Krylov 或粗层 I/O workspace。

当前 Strict 对逐层不同 dtype 和多 rank 生产求解采取 fail-closed；c64/c128
同精度 dispatch 已覆盖。不能把 legacy 的 mixed-precision/MPI 结果移作
Strict 证据。

## 5. 验证结果

已通过：

- Strict tier1 快速闸门：CPU 19 项、CUDA Strict 10 项、融合 FGMRES 3 项，
  合计 32 项，实际 CUDA 进程无 skip；
- 双 parity 的 fine nontrivial Clover MATPC、prepare/reconstruct；
- full-coarse `P/R`、33-point stencil、`X/Y/Yhat`、MATPC、持久递归 V-cycle
  和 bounded Krylov arena。
- 定向 collector/cache/QIO 回归 `83 passed`；fresh tier1 为 CPU `19`、CUDA
  Strict `10`、融合 FGMRES `3`，合计 `32 passed`，无 skip/失败。

命令：

```bash
source ./env.sh
python -B examples/qcu/dev87/run_strict_fast.py --tier 1 --fail-fast
```

该环境是 WSL2，PyTorch wheel 对 P100 `sm_60` 给出不兼容警告；这不改变本次
测试的实际返回值，但性能结论必须携带 GPU、CUDA/PyTorch 和编译架构信息。

### 正式大格结果

同一 V100（UUID=`be23deb4-29b1-7bb2-29ef-c4ab7b34f0a8`）、c64、
`16×32×32×48`、同一输入 bundle、odd-odd Schur 下，2 次 warmup 后 5 次
steady solve 的结果如下：

| 侧 | median(s) | MAD(s) | 迭代数 | full-op 真残差 | 收敛 |
|---|---:|---:|---:|---:|---|
| PyQCU Strict | 2.090647 | 0.011224 | 11 | 3.601e-7 | 5/5 |
| QUDA | 2.165289 | 0.021592 | 37 | 7.303e-7 | 5/5 |

正式合并结果为 `comparison.status=pass`、`comparison.fair=true`，
`speedup_pyqcu_over_quda=1.0357026`。Strict runtime cache 为 schema v2 且
命中；PyQCU owned assets=`4,076,863,488 B`（约 3.797 GiB），fused
workspace=`509,607,936 B`（约 0.475 GiB）。独立 device-wide 首次求解峰值为
PyQCU `11,722,362,880 B`（约 10.917 GiB），QUDA 约 `24,530,000,000 B`
（约 22.845 GiB）。设备级峰值与库自有资产分开统计。

## 6. 未决项与验收边界

1. 用非平凡 Gauge/Clover 对 QUDA/PyQCU 粗层 backward `Yhat` 的每个方向、
   storage site、dagger 和周期偏移做数值逐项比较；正式 solve 通过不能替代
   该粒度的 storage 证明。
2. Strict 当前对多 rank 和逐层不同 dtype fail-closed；legacy 的分布式/混合
   精度结果不能移作 Strict 证据。
3. 以 `strict_hopping_parity_kernel` 为首要热点重测 block size 128/256，
   同时记录 kernel 次数、真残差、寄存器和 occupancy；不得以改变收敛阈值
   或粗层迭代数换取不可比的时间。
4. 动态 thin update、MMA/NVSHMEM 与 C++ 完整五项 `verify()` 接口不在本轮
   交付范围内；当前结论只覆盖本审计列出的 Strict 语义和固定正式协议。
