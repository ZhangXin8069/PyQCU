---
name: pyqcu-logs-debug
description: logs/debug 目录的完整生成 skill：逐轮调试/修复日志（fix-log*.md），完成后升级为 logs/fix-report-*.md。
---
# logs/debug

Per-round debug and fix logs generated during development and bug-fixing sessions.

## File Pattern

`fix-log*.md` — per-round fix logs documenting individual bug fixes, root cause analysis, and verification results.

These are temporary/working files — final summaries are promoted to `logs/fix-report-*.md`
（在本目录的上层，见 `skills/pyqcu-logs-results/SKILL.md`）。

当前内容正是三个文件（2026-10-15 核对）：

| 文件 | 对应审查轮次 |
|---|---|
| `fix-log.md` | `logs/dev73/review-2026-07-28.md`（首轮） |
| `fix-log-r2.md` | `logs/dev73/review-2026-07-28-r2.md` |
| `fix-log-r3.md` | `logs/dev73/review-2026-07-28-r3.md` |

**审查报告不在本目录**——三份 `review-2026-07-28*.md` 都在 `logs/dev73/`；本目录只有
对应的 `fix-log*` 逐轮记录，不要在这里找 review。

Methodology notes: worktree timeline bisection and asset-mtime × commit-time
cross-checking (key to locating silent regressions such as bug35).
