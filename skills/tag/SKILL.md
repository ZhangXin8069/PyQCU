---
name: tag
description: Use when a user asks to create, apply, list, inspect, search, delete, amend, or rewrite Git tags, especially stab/dev/bug/test tags, subversions, snapshots, patches, or remote annotations.
metadata:
  openclaw:
    emoji: 🏷️
---

# tag — Git 四类标签管理技能

## 执行前置

遵循当前目录 `AGENTS.md`「技能执行公共契约」；仅按需读取技能正文与 reference。


## 核心原则

1. **先看后打**：打标签前先分析基线以来的变更，构建诚实、分组的变更清单，不臆造内容。
2. **先看后打，远程同步需明确授权**：先检查当前分支与远程状态；只有用户明确要求同步远程时，
   才在打标签前推送当前分支、打标签后推送标签。未获明确授权时仅执行本地标签操作并在终端摘要说明。
   push 失败按循环重试原则处理，并在终端摘要中说明。
3. **非必要不确认**：创建标签时展示拟用消息后**直接执行**，不阻塞等待确认；仅破坏性操作
   （删除/改写已推送标签）与类型歧义时询问用户。
4. **注释标签**：一律 `git tag -a`，消息格式严格遵循约定（follow 链、编号项以 `;` 结尾、
   运行时 agent 后缀）。模板中的 `[<agent-name>]` 只是占位符，实际输出时必须替换为当前
   正在执行任务的 agent 名称（例如 Codex 使用 `[codex]`，Claude Code 使用 `[claude code]`），
   不得固定为某个 agent 名称，也不得原样输出占位符。
5. **类型推断，歧义一次问清**：按上下文自动推断 stab/dev/bug；无法确定时**一次性列出候选类型与全部歧义点**提问，不逐次追问。
6. **循环尝试直至成功**：push/删除失败、标签冲突、重命名冲突时定位原因重试，
   直至成功或用户终止；失败与重试在终端摘要中说明。
7. **破坏性操作预警**：改写/删除已推送标签会重写他人历史，先警告再执行。

## 历史链自检与自动修正

`tag-chain.sh` 检查所有历史 `stab/dev/bug/test` 标签，而不是只检查最新标签。它把唯一的
`<tag> init,` 注释作为根，先按 peeled commit 的反向拓扑顺序排列，再对同一目标 commit 按
tagger 时间升序排列后核对 `follow` 前缀；这样可发现跨类型漏链、同一旧标签被重复引用和历史
标签中的回指错误。对于合法并行分支，如果当前标签显式 `follow` 一个同时属于两条分支共同祖先的标签，检查器输出
`[BRANCH]` 并保留两个原始目标；只有无法验证共同祖先的非祖先关系才阻断。它同时核对附注类型、peeled
commit 的祖先关系，以及指定远端的标签对象和目标提交。

```bash
# 在仓库根执行；若项目没有本地副本，可把 CHECKER 指向本 configure 技能副本
CHECKER="${TAG_CHAIN_CHECKER:-skills/tag/tag-chain.sh}"
bash "$CHECKER" --check --remote origin

# 只生成全历史修正清单；退出码为 1 表示仍有待处理项
bash "$CHECKER" --repair --dry-run --remote origin

# 仅修正本地标签注释，不改变任何 peeled commit
bash "$CHECKER" --repair --apply

# 明确授权后才改写已推送标签；远端对象漂移或服务器不支持原子推送时拒绝执行
bash "$CHECKER" --repair --apply --remote origin --confirm-remote-rewrite
```

正常创建仍使用 `git tag -a`；历史修正使用 `git mktag` 保留原 tagger 元数据。自动修正只替换可解析的首个 `follow <旧标签>,` 前缀，或给非空正文补上缺失的
`follow <期望标签>, ` 前缀，并保留原 tagger、消息正文和 peeled commit。缺少唯一根、tagger
同一目标的 tagger 时间并列、轻量标签、签名标签、无法验证共同 follow 祖先的目标分叉或提交关系断裂时只报告并停止，
不猜测排序、不移动标签目标。默认 `--check` 覆盖全部历史标签；`--root TAG` 可在历史缺少
`init` 或存在多个 `init` 时显式指定根。

## 默认执行流程（~diff → 历史自检 → ~init → ~tag）

每次 tag 会话**默认**按编排流程执行，首轮直接采用，无需用户逐项确认：

1. **~diff**：调用 diff 技能查看基线以来的改动（`git diff <基线>..HEAD --stat` 概览 + 详情），
   为变更清单构建提供证据；其结果直接复用为原版 Step 1.2，不重复执行；
2. **历史自检**：执行 `tag-chain.sh --check`；若发现可安全修正的 follow 错误，先生成
   `--repair --dry-run` 清单，按用户授权应用修正后再继续；不可推导的历史歧义停止自动修正；
3. **~init**：调用 init 技能同步 AGENTS.md（仓库约定文档与配置同步最新）并归档散落的
   agent 初始化文件（CLAUDE.md/CODEX.md/.claude 等 → `.X.<时间戳>.bak`），保证打标前仓库状态整洁；
4. **~tag（原版）**：执行本技能 Step 1.1–1.6 原版打标操作（类型推断 → 变更清单 →
   按授权同步 → 创建注释标签 → 验证），或用户指定的查看/删除/改写/区间打标操作。

**边界情形**：

- 用户明确指定仅打标/仅查看/仅删除（如"直接打 tag"）→ 跳过 ~diff 与 ~init，直接执行原版操作；
- 只读操作（列出/查看/删除/改写标签）默认也先 ~diff 了解现状；~init 在无初始化需求时可跳过；
- 历史自检发现可修正错误 → 先 dry-run，确认目标对象和消息，再按授权应用；
- 历史自检发现歧义或目标提交断裂 → 不自动改写，列出标签和证据，转人工判断；
- ~diff/~init 执行失败 → 按 debug 技能定位修复后继续，或向用户报告后由用户决定。

## 四类标签

| 类型 | 用途 | 使用时机 |
| --- | --- | --- |
| `stab<N>` | 稳定里程碑 | 某工作**已完成**且稳定。这是里程碑。 |
| `dev<N>` | 开发快照 | 工作**进行中**——继续前保存当前状态。 |
| `bug<N>` | 修复标记 | 发现 **bug** 并已修复。标记修复点。 |
| `test<N>` | 测试标记 | **测试**相关工作已完成（新增/通过/回归）。标记测试点。 |

每类标签有**独立计数器**，从 0 开始。`stab0`、`dev0`、`bug0` 可以同时存在。

## Git 检查

有 Git 且本次产生文件改动时，执行 `git diff --check` 和定向复查；不自动暂存、提交或推送。无 Git 或无本次改动时跳过。

## 触发时机

用户提出以下请求时调用本技能：

- 创建/应用标签（"打标签"、"tag"、"标记一下"、"版本标记"）
  - "stab" / "稳定版" / "完成了" / "完成" → `stab<N>`
  - "dev" / "开发中" / "暂存" / "保存进度" / "快照" → `dev<N>`
  - "bug" / "修复" / "修bug" / "补丁" / "fix" → `bug<N>`
  - "test" / "测试" / "测试完成" / "回归通过" → `test<N>`
  - "stab15_1" / "子版本" / "修订" / "在stab15上补标签" → 该主版本号的子版本（如 `stab15_2`）
- 列出或搜索已有标签（"查看tag"、"list tags"、"有哪些标签"）
- 查看标签详情（"tag详情"、"show tag stab3"）
- 删除标签（"删除tag"、"delete tag stab5"）
- 改写/重写标签（"修改tag"、"amend tag"、"重写tag内容"）

**自动推断**：用户未明确指定类型时，按上下文推断：
- 提到带下划线的完整标签名（如 "stab15_1"、"删掉 dev2_1"）→ 该子版本标签，原样使用
- 提到 "子版本" / "修订" / "sub" + 主版本标签（如 "stab15的子版本"）→ 该主版本的下一子版本
- 提到 "bug"/"修复"/"fix"/"补丁"/"问题" → `bug`
- 提到 "test"/"测试"/"回归"/"验证" → `test`
- 提到 "dev"/"暂存"/"保存"/"快照"/"继续" → `dev`
- 提到 "stab"/"完成"/"稳定"/"版本"/"发布" → `stab`
- 无明确信号 → `dev`（默认）

无法确定时**一次性列出全部候选**提问："Which tag type? [stab/dev/bug/test]"（含全部歧义点，不逐次追问）
- 与其他技能配合：**默认流程**先 diff（查看改动）→ init（AGENTS.md 同步/归档）→ 原版打标操作；
  技能行为异常 → debug；批量生成类打标 → make 编排；历史自检/自动修正由本技能的
  `tag-chain.sh` 负责，远程对象冲突转人工判断

## 标签命名约定

```text
stab0, stab1, stab2, ... stabN, stabN_1, stabN_2, ...   (稳定里程碑)
dev0, dev1, dev2, ... devN, devN_1, ...                 (开发快照)
bug0, bug1, bug2, ... bugN, bugN_1, ...                 (修复标记)
test0, test1, test2, ... testN, testN_1, ...            (测试标记)
```

所有计数器从 0 开始独立递增。每类第一个标签用 `<type>0`。

**完整文法**：`^(stab|dev|bug|test)[0-9]+(_[0-9]+)*$` —— 如 `stab15`、`stab15_1`、`dev3_2_1`。

## 子版本标签（`<type><N>_<M>`）

子版本是**在既有主版本标签之上的后续修订**，不推进主版本计数器。用于主版本标签已放置后的小型定向修复或迭代。

- `stab15_1` = `stab15` 的第一个子版本；属于类型 `stab`、主版本号 `15`。
- 子版本号 `M` 从 **1** 开始（`stab15_1` 是 `stab15` 的第一个子版本；`_0` 从不使用）。
- 消息格式与其他标签完全一致：`follow stab15, 1. 修复...; [<agent-name>].` —— 执行时将
  `[<agent-name>]` 替换为当前 agent 名称；`follow <任意类型的前一标签>` 规则使父主版本（或前一子版本）成为自然前驱。
- 版本排序（`--sort=-v:refname`）正确排序：`stab15 < stab15_1 < stab15_2 < stab16`。
- 子版本**仅显式请求**——自动编号（下文 Case C）总是产生普通主版本标签。若最新标签是 `stab15_1`，自动创建 `stab` 标签得到 `stab16` 而非 `stab15_2`。要再打子版本需显式说明（如 "在 stab15 上再补一个标签"）。

## 标签消息格式

每个标签都是**注释标签**，消息格式如下：

```text
follow <任意类型的前一标签>, 1. 变更说明一; 2. 变更说明二; 3. 变更说明三; [<agent-name>].
```

- 仓库中**第一个标签**使用 `<type>0 init, 1. ...; [<agent-name>].`（无前驱；执行时替换为当前 agent 名称）
- **后续所有标签**（无论类型）使用 `follow <前一标签>, 1. ...;` —— 引用紧邻的前一标签，不论其类型
- 变更项用英文句点 + 空格编号（`1. `、`2. `、`3. `）
- **每项以 `;` 结尾**（英文分号），包括最后一项——无例外
- 所有标点用英文：`.` `,` `;`（内容文字本身可为中文）
- 后缀 ` [<agent-name>].`（前有空格、后接英文句点）始终追加；`<agent-name>` 必须替换为当前实际 agent 名称，不能作为字面量保留
- 消息作为标签注释存储（`git tag -a -m "..."`）

### 变更项编写准则

从 diff 与提交信息构造变更项时：

1. **相关改动分组**——同一子系统/功能的全部改动算一项
2. **按重要性排序**——结构/架构 → 新功能 → 修复 → 清理
3. **每项一句话**——不嵌套列表、不多行
4. **省略琐碎改动**——空白、纯注释、生成文件
5. **用动作导向措辞**——"重构env.sh加载逻辑" 而非 "env.sh被修改了"

示例：

```text
follow stab8, 1. 重构lib目录结构，统一版本化配置模式; 2. 新增cctag Agent技能，替代ccgpush; 3. 修复zshrc中oh-my-zsh插件加载顺序; 4. 清理bin/中过期脚本; [<agent-name>].
```

## 工作流程

详细命令、边界条件与操作输出参考 `references/tag-operations.md`；以下保留主流程与关键安全闸门，执行具体操作前按需读取对应参考小节。

### 1. 创建新标签

这是主要操作。确定标签类型，分析该类型最近标签以来的变更，构建变更清单，创建注释标签。

#### Step 1.1 — 确定标签类型与下一个编号

确定标签类型（`TYPE`）：`stab`、`dev`、`bug`、`test` 之一。按「触发时机」的自动推断规则确定；有歧义时**一次性列出全部候选类型与歧义点**询问用户，不逐次追问。

然后处理三种情形之一：

**Case A — 显式完整标签名**：用户精确指定标签名（如 "打 stab15_1 标签"）。原样使用：

```bash
NEW_TAG="stab15_1"   # 用户指定，必须匹配 ^(stab|dev|bug|test)[0-9]+(_[0-9]+)*$
if git rev-parse -q --verify "refs/tags/${NEW_TAG}" >/dev/null; then
    echo "Tag ${NEW_TAG} already exists — offer to amend it or use the next number"
fi
```

**Case B — 泛化子版本请求**：用户想在既有主版本标签上补打而不指定子版本号（如 "在 stab15 上补一个标签" / "stab15 的子版本"）。计算该主版本的下一子版本号（子版本从 1 开始）：

```bash
TYPE="stab"; MAJOR=15   # 来自用户输入
SUB_TAGS=$(git tag -l "${TYPE}${MAJOR}_[0-9]*" --sort=-v:refname)
if [ -z "$SUB_TAGS" ]; then
    NEXT_MINOR=1
else
    LAST_SUB=$(echo "$SUB_TAGS" | head -1)
    NEXT_MINOR=$(($(echo "$LAST_SUB" | sed "s/^${TYPE}${MAJOR}_//") + 1))
fi
NEW_TAG="${TYPE}${MAJOR}_${NEXT_MINOR}"
```

**Case C — 自动编号**：无显式名称或子版本请求。找出该类型的最近标签，**去掉任何 `_M` 后缀**，取下一个**主版本号**。子版本标签计入其主版本，所以 `stab15_1` 之后自动创建的是 `stab16`：

```bash
TYPE="stab"   # 或 dev、bug、test——由上一步确定
LAST_TAG=$(git tag -l "${TYPE}[0-9]*" --sort=-v:refname | grep -E "^${TYPE}[0-9]+(_[0-9]+)*$" | head -1)
if [ -z "$LAST_TAG" ]; then
    NEW_NUM=0
else
    LAST_NUM=$(echo "$LAST_TAG" | sed "s/^${TYPE}//" | cut -d_ -f1)
    NEW_NUM=$((LAST_NUM + 1))
fi
NEW_TAG="${TYPE}${NEW_NUM}"
```

**边界情形**：共享前缀但非数字后缀的标签（如 `stab-final`）被 `grep -E` 过滤排除，不影响编号。

#### Step 1.2 — 查看基线以来的变更

**基线** = 最近一个**任意类型**（stab/dev/bug/test）的标签。若无标签，用第一个提交。

```bash
# 基线 = 任意类型的最新标签（含子版本标签；版本排序将其排在父标签之上，
# 所以 stab15_1 优先于 stab15 被选中）
BASELINE=$(git tag -l --sort=-v:refname | grep -E '^(stab|dev|bug|test)[0-9]+(_[0-9]+)*$' | head -1)
# 无标签时用第一个提交
if [ -z "$BASELINE" ]; then
    BASELINE=$(git rev-list --max-parents=0 HEAD)
fi
```

查看变更：

```bash
# 基线以来的提交
git log ${BASELINE}..HEAD --oneline --no-merges

# 变更文件
git diff ${BASELINE}..HEAD --stat

# 完整 diff 用于详细分析（限制 500 行）
git diff ${BASELINE}..HEAD -- . | head -500
```

**边界情形 — 无历史提交**：仓库尚无提交时跳过 diff 查看，变更清单直接写"初始提交"。

**边界情形 — 基线以来无变更**：`git diff ${BASELINE}..HEAD --stat` 为空时告知用户没有可打标的新变更。除非用户明确要求，否则**不创建空标签**。

#### Step 1.3 — 构建变更清单

分析收集到的 diff 与提交信息生成标签消息，套用上文变更项编写准则。

- `bug` 标签：聚焦什么坏了、如何修复、影响范围。
- `test` 标签：聚焦测试了什么（新增/回归/覆盖项）、结果如何。
- `dev` 标签：描述已做工作、仍在进行中的部分（可选）与当前状态。
- `stab` 标签：全面描述已完成的工作。

展示拟用消息，然后**直接继续——默认不等待确认**，展示后直接进入 Step 1.4：

```text
Type:    dev
Tag:     dev3
Message: follow stab9, 1. 初步实现xxx功能; 2. 添加yyy模块框架; [<agent-name>].
Changes: 3 files changed, 85 insertions(+), 12 deletions(-)
   → 展示后直接创建并验证，不阻塞等待确认；远程同步须有明确授权。
```

**若用户介入/否决**，询问要改什么——重写某一条、补一条缺失项或删除某项。

#### Step 1.4 — 按授权推送当前提交（打标签前）

**仅当用户明确要求同步远程时执行**：先 push 当前分支，确保远程已包含即将被打标的提交，之后执行 Step 1.5 创建标签；未授权时跳过本步骤并继续本地标签流程。push 失败按循环重试原则处理并在终端摘要中说明，重试仍未成功则暂停远程同步并报告：

```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$BRANCH"
```

**边界情形 — 无远程**：`git remote` 为空时跳过所有 push 步骤并注明未配置远程。

#### Step 1.5 — 创建标签

```bash
git tag -a "$NEW_TAG" -m "$TAG_MESSAGE"
```

验证创建：

```bash
git tag -l "$NEW_TAG"                        # 确认标签存在
git tag -l --format='%(subject)' "$NEW_TAG"  # 显示标签消息
```

#### Step 1.6 — 推送标签（打标签后）

仅当用户明确要求同步远程时，将创建的标签推送到远程（push 失败自动重试并在终端摘要中说明）：

```bash
git push origin "$NEW_TAG"
```

---

### 2. 列出标签

列出标签，可按类型过滤：

```bash
# 列出全部标签（四类全部，含子版本）
git tag -l --sort=-v:refname --format='%(refname:short) | %(taggername) | %(taggerdate:short) | %(subject)' | grep -E '^(stab|dev|bug|test)[0-9]+(_[0-9]+)* \|'

# 按类型过滤（如仅 stab 标签）——子版本包含在内
TYPE="dev"
git tag -l "${TYPE}[0-9]*" --sort=-v:refname --format='%(refname:short) | %(taggerdate:short) | %(subject)'

# 按类型计数（子版本计入其类型）
git tag -l 'stab[0-9]*' | wc -l
git tag -l 'dev[0-9]*'  | wc -l
git tag -l 'bug[0-9]*'  | wc -l
git tag -l 'test[0-9]*' | wc -l
```

**边界情形 — 无标签**：报告 "No cctag tags found in this repository."

---

### 3. 查看单个标签

显示单个标签（任意类型）的详细信息：

```bash
# 显示标签注释
git tag -l --format='%(subject)%0a%(body)' "$TAG_NAME"

# 显示标签作者信息与日期
git tag -l --format='Type: %(refname:short)%0aAuthor: %(taggername) <%(taggeremail)>%0aDate: %(taggerdate:iso)%0aMessage: %(subject)' "$TAG_NAME"

# 显示该标签与其前驱之间的提交（任意类型，含子版本）
PREV_TAG=$(git tag -l --sort=-v:refname | grep -E '^(stab|dev|bug|test)[0-9]+(_[0-9]+)*$' | grep -A1 "^$TAG_NAME$" | tail -1)
if [ -n "$PREV_TAG" ]; then
    git log ${PREV_TAG}..${TAG_NAME} --oneline --no-merges
fi
```

**边界情形 — 标签不存在**：报告 "Tag 'xxx' does not exist. Use 'list' to see available tags."

**边界情形 — 该类型首个标签**：注明该类型没有可对比的前一标签。

---

### 4. 删除标签

在本地与远程同时删除标签：

```bash
# 删除前确认
git tag -l --format='%(subject)' "$TAG_NAME"

# 删除本地
git tag -d "$TAG_NAME"

# 删除远程（若存在于远程）
git push origin :refs/tags/"$TAG_NAME" 2>/dev/null
```

**边界情形 — 本地找不到标签**：检查远程：`git ls-remote --tags origin "$TAG_NAME"`。若远程存在而本地没有，先 fetch 再删除。

**边界情形 — 强制删除**：用户传入 `--force` 或标签已推送时，删除远程标签前必须显式确认——这是影响所有协作者的破坏性操作。

---

### 5. 改写标签

重写既有标签（任意类型）的消息：

#### Step 5.1 — 确定目标标签

```bash
# 用户指定的标签（子版本同理，如 "stab15_1"）
TARGET_TAG="dev2"   # 来自用户输入

# 或改写某类型的最新标签
TYPE="stab"   # 来自用户输入或自动推断
TARGET_TAG=$(git tag -l "${TYPE}[0-9]*" --sort=-v:refname | grep -E "^${TYPE}[0-9]+(_[0-9]+)*$" | head -1)
```

#### Step 5.2 — 显示当前标签内容

```bash
git tag -l --format='%(subject)' "$TARGET_TAG"
```

#### Step 5.3 — 查看标签前驱以来的变更

用任意类型的紧邻前驱标签作为基线（逻辑同 Step 1.2）。

#### Step 5.4 — 构建新消息

提出新消息。可整体重写或仅部分编辑——遵循用户指示。保持同一标签类型。

#### Step 5.5 — 推送当前提交、替换标签、推送标签

**改写已推送标签属破坏性操作，整体流程须先经用户确认（见核心原则与下方警告）；确认范围包含远程同步时才执行推送**：

```bash
# 先推送当前分支
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$BRANCH"

# 删除旧标签（本地）
git tag -d "$TARGET_TAG"

# 删除旧标签（远程）
git push origin :refs/tags/"$TARGET_TAG" 2>/dev/null

# 以新消息重新创建（同一提交）
git tag -a "$TARGET_TAG" -m "$NEW_MESSAGE"

# 推送更新后的标签
git push origin "$TARGET_TAG"
```

**警告**：改写已推送标签会为任何已拉取旧标签的人重写历史。删除远程标签前必须警告用户。

---

## 错误处理

| 场景 | 处理 |
| --- | --- |
| 标签类型有歧义 | **一次性列出全部候选**："Which tag type? [stab/dev/bug/test]" 及全部歧义点 |
| 用户指定子版本标签（如 `stab15_1`） | 原样识别；用于查看/删除/改写/创建 |
| 请求的子版本标签已存在 | 报告；提供改写或下一个子版本号 |
| 子版本标签之后自动编号 | 下一主版本（`stab15_1` → `stab16`）；要另一子版本须显式请求 `stab15_2` |
| 完全没有之前的标签 | 用第一个提交作为基线 |
| 仓库无提交 | 中止；报告无内容可打标签 |
| 基线以来无变更 | 报告；除非用户明确要求，否则不创建空标签 |
| 未配置远程 | 跳过两个 push 步骤；注明无远程 |
| 打标签前的分支推送失败 | 诊断、修复、重试；仍失败则暂停并报告——不创建标签 |
| 用户要求仅本地 | 仅本地创建/改写，跳过 push；在终端摘要中说明 |
| 标签名冲突 | 递增编号重试（自动编号下不应发生） |
| 远程已有同标签 | 本地与远程消息不一致时警告 |
| 改写不存在的标签 | 报告标签不存在；建议列出 |
| push/创建/删除失败（网络、冲突） | 从错误诊断、修复并重试（循环直至成功）；重试在终端摘要中说明 |
| 用户否决变更清单（介入） | 允许重写特定项或整体重写 |

## 注意事项

- 操作结果以实测为准（`git tag -l`、`git push` 输出），不虚报已推送/已创建；
- 本地文件改动按公共 Git 契约检查，未经明确要求不自动提交或推送；仅在用户明确要求同步远程时 push 当前分支与标签，
  push 失败自动重试并在终端摘要中说明；其他机器由用户自行 `git fetch --tags` 同步；
