# CLAUDE.md — .claude/skills

Agent skills for the PyQCU repository. Each skill is a markdown file with YAML frontmatter. Skills are loaded on demand by Claude Code when their `description` matches the current task — they capture reusable knowledge so it does not have to be re-derived.

## Skill Format

Every skill file must begin with YAML frontmatter:

```yaml
---
name: <kebab-case-slug>          # must match the file name
description: <one-line summary>  # used to decide when the skill applies
---
```

Follow the frontmatter with the skill body: the knowledge, procedures, and conventions it is meant to encapsulate.

## Skills

| File | Skill | Description |
|------|-------|-------------|
| `past-work.md` | `past-work` | Past work history of PyQCU — what was built, optimized, and remains TODO (project phases, current state, known gaps) |

## Adding / Editing a Skill

1. Create or edit `skills/<name>.md`.
2. Keep the frontmatter `name` identical to the file name, and write a `description` specific enough to trigger appropriately.
3. Keep the body focused on one capability.
4. Update the table above so this CLAUDE.md stays accurate.

## Notes

- The full content of the `past-work` skill is reproduced in `../CLAUDE.md` (see "Complete Skill: `past-work`") so it is also available at the parent `.claude/` level.
- This directory is tracked in git (`skills/past-work.md`). Do not put machine-local or ephemeral content here.
