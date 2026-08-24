---
name: debug
description: logs/debug 目录的完整生成 skill：逐轮调试/修复日志（fix-log*.md），完成后升级为 logs/fix-report-*.md。
---
# CLAUDE.md — logs/debug

Per-round debug and fix logs generated during development and bug-fixing sessions.

## File Pattern

`fix-log*.md` — per-round fix logs documenting individual bug fixes, root cause analysis, and verification results.

These are temporary/working files — final summaries are promoted to `logs/fix-report-*.md`.

Current contents: `fix-log.md` plus two follow-up rounds r2/r3 (matching
review-2026-07-28-r2/r3). Methodology notes: worktree timeline bisection and
asset-mtime × commit-time cross-checking (key to locating silent regressions such as bug35).
