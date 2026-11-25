# KUPL Legacy-MG 修正器路径的历史（2026-11-24/25，已于 2026-11-25 删除）

本文只在有人问起"修正间隔 / 阻尼 / `best_ratio` / MG 为什么慢"这类**历史**问题时
读取。当前实现没有修正器：KUPL Legacy-MG 只有右预条件 FGMRES(m = 10) 一条外层路径，
结构说明见父目录 `SKILL.md` 的"2026-11-25 KUPL Legacy-MG 只有一条外层"一节，删除
记录见仓库 `logs/v20261126.txt`。

**为什么删**：CUDA 与 KUPL 当时跑的是两种不同方法（CUDA 是右预条件 FGMRES + V-cycle
预条件子，KUPL 是细层 BiCGStab + 周期粗修正），任何性能或收敛差异都无法归因到硬件。
2026-11-25 直接把修正器整段删除，KUPL 换成 CUDA 的结构，而不是加开关。

**不可复现**：本节所有数字都测于删除前，当前源码没有产生它们的代码路径。原始推理与
逐轮对话见 `logs/v20261123.txt` §142/§145/§150/§152/§153/§182。

## 1. 删除前的路径长什么样（仅供读旧日志）

```text
outer : 细层 BiCGStab（restart 长度 + 真残差复查）
chunk : min(remaining, correction_interval - since_correction)
        —— 修正间隔就是细层 restart 长度
coarse: 每 chunk 构造一次粗修正，按 1 → 0.5 → 0.25 → 0.125 阻尼线搜索，
        只接受使细层真残差下降的步长，四步全失败则回滚并关闭本 solve 的修正
gate  : KUPL_REQUIRE_MG_CORRECTION=1 拒绝 levels<=1/restart=0、无接受修正
print : MG_LEVEL_ITERS ... level0= level1= ... total= accepted_corrections= fallback_fine=
        MG_CORRECTION_DIAG ... best_damping= best_ratio= overlap= cos_overlap=
                              capture_bound= reduction_bound=
knobs : KUPL_MG_CORRECTION_INTERVAL（间隔）、KUPL_MG_NU_PRE（前缀）、
        KUPL_MG_OUTER（外层选择）、stage-1 smoothing_prefix
```

这些名字、字段和旋钮**全部随修正器删除**。旧日志里出现它们是正常的；把它们当成
现行机制、或按它们复跑，是错的。

## 2. 2026-11-24 记录：修正器路径无法加速

先读 `logs/v20261123.txt` §142（转移审计通过）、§145（`capture_bound` 是代数）、
§150/§152（关闭仪表的三路对照）、§153（间隔即细层 restart，对照仍有混淆）。
以下每条都修正过早期版本的错误结论，不要再引用 §121/§129/§133 的旧措辞。

- **转移对是对的**：逐 dof 审计给出 `mismatches=0`、
  `gather_set_equals_scatter_set=1`、`support == block_sites * E`、`outside_block=0`、
  `roundtrip_err` 2.4e-07–8.3e-07。不存在索引/parity 映射错误——**不要去改
  `restrict()`/`prolong()`**。
- **`capture_bound`/`reduction_bound` 不是测量值**，是代数 `sqrt(E/(e*block))` 与
  `sqrt(1-capture²)`，`E=2, e=12` 时对 block 16/32/256 分别给出
  `0.1020621 / 0.07216878 / 0.02551552`，与当时的"实测"七位吻合。粗化时它下降只是
  block 变大，**与转移无关**，不能作为证据。这两个量由 `diagnose_correction` 打印，
  已随该函数删除。
- **当时的真实测量**（必须关闭 `KUPL_MG_DEBUG`，`diagnose_correction` 计在同一计时器
  内）：一次修正只去掉不到 1% 的残差，而一次细迭代去掉约 45%；`overlap` 0.013–0.046；
  `best_ratio` 0.991–0.993；即使最优阻尼也只保留约 29% 的修正量。这些指标现在不再
  产生，也无法在当前源码上重测。
- **零空间基曾经是错的**：旧循环是 `v -= S⁻¹(S v) = v − v`，存的是内层求解器残差
  （白噪声）。已改为真逆迭代 `v ← S⁻¹v` 并每轮归一化，且缓存身份带
  `nv_builder="inverse-iteration/2"`。**用真基重测，结论不变**（§152）。
- **`E` 不是杠杆**：2→4 让 `best_ratio` 从 0.9941 到 0.9934、`level0` 从 18 到 24
  （更差）。
- **调度的混淆**（§153）：修正间隔**就是**细层 restart 长度
  （`chunk = min(remaining, correction_interval - since_correction)`），所以
  `KUPL_MG_CORRECTION_INTERVAL=2` 同时把 BiCGStab 的重启从 20 变成 2。三路对照里
  `--mg-levels 1` 那一臂其实是**不重启**的 BiCGStab，因此 4.647 / 6.504 / 7.947 s
  比较的是三种 Krylov 策略，**不能**把 1.71× 全部归给 MG。要归因必须先让一个 chunk
  能跨多次重启——这是结构改动，当时未做；两天后修正器整体被删，这个方向作废。
- 结论：删除前那条路径的两级 Legacy-MG 没有净收益。倍数本身在 2026-11-25 的一次
  同分配、同 `restart`、真复用缓存的四臂对照中测得（§3，**MG/plain = 1.68×–1.72×**）；
  在那之前的一切倍数仍然不得引用。

## 3. 2026-11-25 记录：删除前最后一次合法测量

作业 `2258877`，一个 dsub 分配、单节点 `cn12139`、8 rank、`grid 2 2 2 1`、
`mass=+0.1`（`κ=0.121951 < κ_c=0.125`，算符非奇异）、`atol=1e-5`、
`--repeat 3 --warmup 1`，四臂**串行跑在同一分配内**：

| 臂 | ABI 符号 | 秒（3 次中位） | 迭代 | 残差 | 缓存 |
|---|---|---:|---:|---:|---|
| plain BiCGStab | `applyCloverBistabCgQcu` | **1.1856** | 28 | 4.4086e-06 | 未请求 |
| mg-2 + legacy 基 | `applyCloverMultigridQcu` | **1.9922** | `n/a` | 4.8466e-06 | `cache_reused=True` |
| mg-2 + inverse 基 | `applyCloverMultigridQcu` | **2.0421** | `n/a` | 4.6316e-06 | `cache_reused=True` |

```text
MG/plain = 1.680×（legacy 基）   1.722×（inverse 基）
```

**这次为什么可以引用**（逐条对照 §153 的混淆项）：

- **同分配**：四行同一个 dsub、同一节点、同一库 md5 `305a152cdc68`。§114 禁止的跨
  分配比较完全没有出现。
- **`restart` 相同**：四行的 `metrics.restart` **全部为 5**，所以 §153 的"修正间隔即
  restart 长度"混淆没有被引入——两臂的细层 Krylov 重启长度一致。
- **缓存是真复用**：两臂都记录 `cache_reused=True`、`asset_source=cache`、
  `cache_seconds=null`，用的是 `2258741` 产出并按 `--nv-iters 2` 建的产物。
- **差异远超噪声**：同作业内基准漂移 +0.39%（1.1856 → 1.1903），单臂样本离散
  0.13–0.36%，而 68–72% 的 MG 惩罚约为漂移的 **170×**。
- **跨分配一致性佐证**：纯求解残差与另一分配的纯求解**逐位相同**
  （`4.408633403218057e-06`）而墙钟不同——这正是 §114 要防的情形，反过来说明
  残差可比、墙钟不可跨分配比。

**两套基仍然互相不可区分**（1.9922 vs 2.0421，2.5%；另一作业 `2258741` 为 0.25%），
即 **null-vector 构建器的选择不改变代价**。所以"换基能否加速"是否定答案，"MG 是否
值得"对旧路径也是否定答案——机制解释：一次修正去掉不到 1%，一次细迭代去掉约 45%，
一次修正却要付约 2.3 次细层应用的代价。

## 4. 现在该怎么引用

- 旧路径的数字（1.68×–1.72×、`best_ratio` 0.99x、`overlap` 0.0x、阻尼 0.29 等）只能
  作为**历史**引用，并且要说明"2026-11-25 前的修正器路径"。
- 任何按 `KUPL_MG_CORRECTION_INTERVAL`、`KUPL_MG_NU_PRE`、`KUPL_MG_OUTER` 复跑的
  指令都已失效：这些环境变量不再被读取。
- 当前 FGMRES 路径的 MG/plain 倍数与它的门控行为**仍在测**，在测出来之前不得把旧
  倍数搬到新路径上，也不得反过来用旧负结果宣布新路径同样无收益。
