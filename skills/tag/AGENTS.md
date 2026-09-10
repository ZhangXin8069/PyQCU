# AGENTS.md — tag 技能

本目录为实现 `tag` 技能的 `SKILL.md`（git 四类标签 stab/dev/bug/test 管理，含 frontmatter 与完整操作手册）。SKILL.md 为唯一权威内容，勿复制改动。执行 tag 技能时只使用 Git 提交、标签和终端结果；有 Git 改动时按上级 `AGENTS.md` 公共契约执行 `git diff --check` 与定向复查；未经用户明确要求不自动暂存、提交或推送；无 Git 条件立即跳过。
`tag-chain.sh` 的历史链状态使用临时键值记录，兼容 macOS 自带 Bash 3.2；合法并行分支须显式 follow 共同祖先并输出 `[BRANCH]`，修改后必须运行 `bash skills/tag/tag-chain.test.sh`，保持本地/远端 lease 与 peeled commit 不变量。
