# 双远端发布交付与核验

本文件承载 `pyqcu-tag/SKILL.md` 的发布交付细节：发布任务（PyPI 上传、RTD 文档、发布日志）
的提交边界、双远端推送顺序与四端同 hash 核验。标签本身的创建、改写、删除规则仍以父目录
`SKILL.md` 与 `tag-operations.md` 为准；**标签只在用户显式要求时创建**。

## 1. 三个代码托管端的分工

| 端 | 地址 | 角色 |
| --- | --- | --- |
| GitLab | `git@gitlab.itp.ac.cn:zhangxin8069/PyQCU.git`（本检出唯一的 `origin`） | 主开发远端，提交入口 |
| Gitee | `git@gitee.com:zhangxin8069/PyQCU.git` | 国内发布远端，必须显式推送 |
| GitHub | `git@github.com:ZhangXin8069/PyQCU.git` | Gitee 的自动镜像，由 Gitee 同步，不手工推送 |
| RTD | `https://pyqcu.readthedocs.io/{en,zh-cn}/latest/` | 跟踪 GitHub 仓库构建 `docs/en` 与 `docs/zh`，只读发布 |

- 发布语义是**双远端**：GitLab 与 Gitee 都必须收到同一个提交；GitHub 由 Gitee 的仓库同步
  自动镜像，推完 Gitee 后轮询等待，不要另行手工推 GitHub（会造成镜像分叉或回退）。
- 文档能否出现在 RTD 取决于 GitHub 镜像：Gitee 未同步时 GitHub 仍是旧提交，RTD 也看不到新文档。
- 本检出的 `git remote -v` 只有 `origin`（GitLab）。Gitee 用完整 URL 直接推即可，或先登记
  `git remote add gitee git@gitee.com:zhangxin8069/PyQCU.git`；不要为此改写 `origin` 的地址。

## 2. 推送顺序与四端同 hash 核验

顺序固定为 **本地提交 → GitLab → Gitee → 轮询 GitHub 镜像 → 四端比对**：

```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)      # 发布分支，通常 main
HEAD_SHA=$(git rev-parse HEAD)

git push origin "$BRANCH"                                     # 1. GitLab
git push git@gitee.com:zhangxin8069/PyQCU.git "$BRANCH"       # 2. Gitee（触发 GitHub 镜像）

for attempt in $(seq 1 30); do                                # 3. 轮询 GitHub 镜像
    mirror=$(git ls-remote https://github.com/ZhangXin8069/PyQCU.git \
             "refs/heads/$BRANCH" | cut -f1)
    [ "$mirror" = "$HEAD_SHA" ] && break
    sleep 2
done

for url in git@gitlab.itp.ac.cn:zhangxin8069/PyQCU.git \
           git@gitee.com:zhangxin8069/PyQCU.git \
           https://github.com/ZhangXin8069/PyQCU.git; do      # 4. 三端回读
    printf '%s %s\n' "$(git ls-remote "$url" "refs/heads/$BRANCH" | cut -f1)" "$url"
done
printf '%s %s\n' "$HEAD_SHA" "$(git rev-parse --abbrev-ref HEAD)@local"
```

验收：`git rev-parse HEAD` 的输出与三条 `git ls-remote <url> refs/heads/main` 的 hash
**四个值完全相同**；缺一不可报告"已交付"。本次发布提交 `b63e7798b61da…` 的三端均为同一
hash，GitHub 镜像在 Gitee 推送后约 10 s 到数分钟完成同步（0.1.4 的日志提交实测约
5 分钟）。轮询按分钟级设置即可，单次超时不判失败，超时只报
"镜像未同步"，不改写远端、不 force push。

边界与陷阱：

- `git ls-remote` 读的是远端真实 ref；不要用本地 `origin/main`、`FETCH_HEAD` 或
  `git log --all` 冒充三端核验，也不要拿 `git push` 的输出当同步完成证据。
- GitHub 若因 host key 未登记而报 `Host key verification failed`，只读核验改用 HTTPS URL
  （如上面的 `https://github.com/...`），或在推送机器上先登记 `github.com` 主机键；
  不要用 `StrictHostKeyChecking=no` 长期绕过。
- `git ls-remote https://github.com/...` 可能因网络/DNS 连续返回空或超时。不要立刻判
  “未同步”：用 GitHub REST API 做独立只读核验：

  ```bash
  curl -sL https://api.github.com/repos/ZhangXin8069/PyQCU/commits/main \
    | python -c 'import json,sys; d=json.load(sys.stdin); print(d["sha"])'
  ```

  2026-11-09 的 0.1.4 发布中，`git ls-remote` 连续 8 次为空，但 API 返回
  `ceeb04c0…`，RTD 页面标题已是 0.1.4；二者结合即可确认镜像与站点已同步。
  GitLab/Gitee 仍用 `git ls-remote` 核验，API 只作为 GitHub 的补充证据。
- Gitee 推送失败时 GitLab 已就绪，重试 Gitee 即可；重试仍失败则停止后续步骤，并在摘要中
  写明"三端中哪一端未到位"，不隐藏部分交付状态。
- 只推 `main`（或发布分支）与显式标签，不推其它本地分支；不使用 `--force`/`--force-with-lease`。

## 3. 提交边界

- **只暂存本任务文件**：逐条 `git add <具体路径>`，禁止 `git add -A`、`git add .`。暂存后用
  `git status --short` 与 `git diff --cached --stat` 复核，确认没有带入无关改动、未跟踪的
  临时文件或 `data/**` 之外的构建产物。
- 提交消息：一行 `type: 摘要`，正文用 `- ` 分组要点，末尾固定追加协作 trailer：

```text
release: publish pyqcu 0.1.0 and prepare RTD docs

- add PEP 621 metadata, sdist manifest, and package-local native staging
- add bilingual RTD sources, docs example gate, and release log
- document 8x8x16x16 complex64 verification and historical provenance

Co-authored-by: CommandCodeBot <noreply@commandcode.ai>
```

- 提交前按根 `AGENTS.md` 执行 `git diff --check`、大文件闸门与 `bash skills/pyqcu-form/form-audit.sh`；
  发布验证本身的命令见 `pyqcu-testing` 技能的「发布验证（release gate）」小节。
- 已推送的标签不改写、不删除；不经用户明确请求不创建标签，也不把"计划打标签"写成"已打标签"。
  本次发布（`b63e779`）**未创建任何标签**，交付摘要必须与之一致。

## 4. 若未来为发布打标签

标签类型按上下文推断（版本里程碑 `stab`、开发快照 `dev`、修复 `bug`、验证 `test`），消息
格式与 `SKILL.md`「标签消息格式」一致，只是推送同样必须双远端：

```bash
NEW_TAG="stab<N>"     # 由上下文推断类型与编号；子版本形如 stab<N>_<M>
TAG_MESSAGE="follow <前一标签>, 1. 发布 pyqcu <ver> 到 PyPI; 2. 同步 GitLab 与 Gitee 并核验三端同 hash; [<agent-name>]."
git tag -a "$NEW_TAG" -m "$TAG_MESSAGE"
git tag -l --format='%(subject)' "$NEW_TAG"

git push origin "refs/tags/$NEW_TAG"
git push git@gitee.com:zhangxin8069/PyQCU.git "refs/tags/$NEW_TAG"

# 标签也要三端回读 peeled commit
for url in git@gitlab.itp.ac.cn:zhangxin8069/PyQCU.git \
           git@gitee.com:zhangxin8069/PyQCU.git \
           https://github.com/ZhangXin8069/PyQCU.git; do
    git ls-remote "$url" "refs/tags/$NEW_TAG^{}" "refs/tags/$NEW_TAG"
done
```

- 一律 `git tag -a`；`follow <前一标签>,` 前缀、每项以 `;` 结尾、末尾 ` [<agent-name>].` 都不得
  省略，占位符必须替换为当前 agent 名称（如 `[commandcode]`），不得原样输出。
- GitHub 端的标签随 Gitee 镜像同步；三端 `refs/tags/<tag>` 与 peeled commit 必须一致才算交付。
- 已推送标签需要改写时先取得明确授权，并走 `tag-operations.md` 的
  `--repair --apply --remote origin --confirm-remote-rewrite` 闸门；默认只报告不改写。

## 5. 交付检查清单

1. 暂存范围只用显式路径，`git diff --cached --stat` 无越界文件。
2. 提交消息含分组要点与 `Co-authored-by: CommandCodeBot <noreply@commandcode.ai>`。
3. GitLab → Gitee 推送顺序正确，GitHub 镜像轮询到同 hash。
4. 四端（本地 + 三端）`refs/heads/main` hash 一致，逐条贴出实际输出。
5. 未创建/改写/删除标签，除非用户显式要求并已双远端推送。
6. 未验证项（如 RTD 构建、PyPI 传播）在摘要中单列，不写成已通过。
