# CLAUDE.md — logs

Development logs, review reports, bug fix summaries, and solver output. This directory is gitignored — contents are not versioned.

## File Patterns

| Pattern | Purpose |
|---------|---------|
| `dev<N>.md` | Development milestone reports |
| `bug<N>.md` | Bug discovery & code review reports |
| `review-*.md` | Code review findings (e.g., `review-2026-07-28.md`) |
| `fix-report-*.md` | Bug fix summaries |
| `multigrid_report.md` | MG solver performance reports |
| `clover_multigrid.log` | C++ solver convergence output |
| `*.png` | Performance charts, convergence plots |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `debug/` | Per-round fix logs (`fix-log*.md`) |
| `results/` | Final/remaining fix reports |
