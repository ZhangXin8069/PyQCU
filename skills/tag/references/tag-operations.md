# tag 操作参考

本文件承载 `tag/SKILL.md` 的详细命令。执行前先读取对应小节，并把命令中的变量替换为已核实的实际值。

## 0. 历史 follow 链自检与修正

`tag-chain.sh` 默认检查全部历史标签。排序以唯一 `init` 标签为根，先按 peeled commit 的
反向拓扑顺序排列，再对同一目标 commit 按 tagger 时间升序排列；它不按编号排序，也不把
commit 时间当作标签创建时间。合法分支若显式 follow 同一个共同祖先会输出 `[BRANCH]` 并保留目标；
无法验证共同祖先的分支或同一目标的 tagger 时间并列时才阻断自动修正。自检同时验证标签类型、peeled commit、祖先关系和可选远端对象。

```bash
CHECKER="${TAG_CHAIN_CHECKER:-skills/tag/tag-chain.sh}"
bash "$CHECKER" --check --remote origin
bash "$CHECKER" --repair --dry-run --remote origin
```

正常创建仍使用 `git tag -a`；历史修正使用 `git mktag` 保留原 tagger 元数据。自动修正边界：仅替换可解析的首个 `follow <旧标签>,` 前缀，或给非空正文补上缺失的
`follow <期望标签>, ` 前缀，保留消息正文、原 tagger 元数据和 peeled commit。以下情形只报告不改写：
根标签缺失或不唯一、同一目标的 tagger 时间并列、轻量标签、
签名标签、目标提交分叉且无法验证共同 follow 祖先、远端对象已漂移。

本地应用与远端应用是两个明确闸门：

```bash
bash "$CHECKER" --repair --apply
bash "$CHECKER" --repair --apply --remote origin --confirm-remote-rewrite
```

第二条命令会使用 `--force-with-lease` 和 `--atomic` 改写远端已有标签，必须在用户明确授权
后执行；远端更新失败时不更新本地 refs，最终会重新执行 `--check` 回读验证。`--repair` 或
`--dry-run` 不写入本地和远端。

## 1. 创建标签

### 1.1 确定类型与编号

标签类型只有 `stab`、`dev`、`bug`、`test`，四类计数器独立。显式完整名称必须匹配：

```text
^(stab|dev|bug|test)[0-9]+(_[0-9]+)*$
```

显式名称直接使用；泛化子版本请求计算指定主版本的下一个 `_M`（从 1 开始）：

```bash
TYPE="stab"; MAJOR=15
SUB_TAGS=$(git tag -l "${TYPE}${MAJOR}_[0-9]*" --sort=-v:refname)
if [ -z "$SUB_TAGS" ]; then
    NEXT_MINOR=1
else
    LAST_SUB=$(printf '%s\n' "$SUB_TAGS" | head -1)
    NEXT_MINOR=$(($(printf '%s\n' "$LAST_SUB" | sed "s/^${TYPE}${MAJOR}_//") + 1))
fi
NEW_TAG="${TYPE}${MAJOR}_${NEXT_MINOR}"
```

无显式名称时，只在同类主版本上递增；子版本不产生新的同主版本标签：

```bash
TYPE="stab"
LAST_TAG=$(git tag -l "${TYPE}[0-9]*" --sort=-v:refname |
    grep -E "^${TYPE}[0-9]+(_[0-9]+)*$" | head -1)
if [ -z "$LAST_TAG" ]; then
    NEW_NUM=0
else
    LAST_NUM=$(printf '%s\n' "$LAST_TAG" | sed "s/^${TYPE}//" | cut -d_ -f1)
    NEW_NUM=$((LAST_NUM + 1))
fi
NEW_TAG="${TYPE}${NEW_NUM}"
```

共享前缀但非数字后缀的标签（如 `stab-final`）由正则排除；已存在的显式标签先报告，不覆盖。

### 1.2 确定变更基线

基线是任意类型中按版本排序最新的标签；没有标签时使用第一个提交。无提交时中止：

```bash
BASELINE=$(git tag -l --sort=-v:refname |
    grep -E '^(stab|dev|bug|test)[0-9]+(_[0-9]+)*$' | head -1)
if [ -z "$BASELINE" ]; then
    BASELINE=$(git rev-list --max-parents=0 HEAD)
fi
git log "${BASELINE}..HEAD" --oneline --no-merges
git diff "${BASELINE}..HEAD" --stat
git diff "${BASELINE}..HEAD" -- . | head -500
```

没有新变更时报告并不创建空标签，除非用户明确要求。

### 1.3 生成消息、推送与创建

消息格式为：

```text
follow <前一标签>, 1. 变更说明一; 2. 变更说明二; [<agent-name>].
```

第一个标签用 `<type>0 init, 1. ...; [<agent-name>].`；执行时将占位符替换为当前实际 agent 名称（例如 Codex 使用 `[codex]`，Claude Code 使用 `[claude code]`）。每项以英文分号结束，引用紧邻的任意类型前一标签。按类型突出：`bug` 写影响与修复，`test` 写覆盖与结果，`dev` 写当前状态，`stab` 写完整成果。

只有用户明确要求同步远程时，才在创建标签前推送当前分支；未授权时只做本地标签操作。无远程则在终端说明跳过：

```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
# 仅在用户明确授权远程同步时执行
git push origin "$BRANCH"
git tag -a "$NEW_TAG" -m "$TAG_MESSAGE"
git tag -l "$NEW_TAG"
git tag -l --format='%(subject)' "$NEW_TAG"
# 仅在用户明确授权远程同步时执行
git push origin "$NEW_TAG"
```

展示拟用消息后，按调用方的授权策略继续；不要把标签创建误报为已经推送。

## 2. 列出标签

```bash
git tag -l --sort=-v:refname \
  --format='%(refname:short) | %(taggerdate:short) | %(subject)' |
  grep -E '^(stab|dev|bug|test)[0-9]+(_[0-9]+)* \|'
TYPE="dev"
git tag -l "${TYPE}[0-9]*" --sort=-v:refname \
  --format='%(refname:short) | %(taggerdate:short) | %(subject)'
git tag -l 'stab[0-9]*' | wc -l
git tag -l 'dev[0-9]*' | wc -l
git tag -l 'bug[0-9]*' | wc -l
git tag -l 'test[0-9]*' | wc -l
```

无标签时报告没有本仓库标签；子版本列出在其父版本附近并计入对应类型。

## 3. 查看标签

```bash
git tag -l --format='%(subject)%0a%(body)' "$TAG_NAME"
git tag -l --format='Type: %(refname:short)%0aAuthor: %(taggername) <%(taggeremail)>%0aDate: %(taggerdate:iso)%0aMessage: %(subject)' "$TAG_NAME"
PREV_TAG=$(git tag -l --sort=-v:refname |
  grep -E '^(stab|dev|bug|test)[0-9]+(_[0-9]+)*$' |
  grep -A1 "^$TAG_NAME$" | tail -1)
if [ -n "$PREV_TAG" ]; then
    git log "${PREV_TAG}..${TAG_NAME}" --oneline --no-merges
fi
```

标签不存在时报告并建议先列出；该类型首个标签没有可比较的前一标签。

## 4. 删除标签

删除前显示目标标签并确认目标。删除本地标签是破坏性操作；已推送标签还会影响协作者，必须取得显式确认。确认后：

```bash
git tag -l --format='%(subject)' "$TAG_NAME"
git tag -d "$TAG_NAME"
git push origin :refs/tags/"$TAG_NAME"
```

本地不存在时先用 `git ls-remote --tags origin "$TAG_NAME"` 检查远程；无远程时只做本地范围内可验证的动作。

## 5. 改写标签

先确定存在的目标标签，显示旧消息并查看其前驱以来的变更；改写已推送标签前必须显式确认。确认后：

```bash
TARGET_TAG="dev2"
git tag -l --format='%(subject)' "$TARGET_TAG"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$BRANCH"
git tag -d "$TARGET_TAG"
git push origin :refs/tags/"$TARGET_TAG"
git tag -a "$TARGET_TAG" -m "$NEW_MESSAGE"
git push origin "$TARGET_TAG"
```

不把未确认的删除、远程覆盖或强制推送作为自动步骤；推送失败时停止后续破坏性动作并记录原因。

## 6. 指定区间

用户给出基线与终点时替换默认 `BASELINE`/`HEAD`，仍执行变更检查、消息构造、远程前置检查和标签验证：

```bash
BASE_REF="stab5"
HEAD_REF="HEAD"
git log "${BASE_REF}..${HEAD_REF}" --oneline --no-merges
git diff "${BASE_REF}..${HEAD_REF}" --stat
```

## 输出与错误

创建或改写后报告类型、标签名、完整消息、实际推送结果和变更统计；列出/查看后报告实际命中内容。任何网络、冲突、编号或验证错误都先保留终端证据，再按 debug 原则定位并重试；无法安全重试时报告遗留。
