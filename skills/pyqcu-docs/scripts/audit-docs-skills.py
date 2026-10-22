#!/usr/bin/env python3
"""校验 PyQCU docs/ 与 skills/ 的索引、结构和按需披露契约。"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
ROOT_AGENTS = ROOT / "AGENTS.md"
DOCS = ROOT / "docs"
SKILLS = ROOT / "skills"
DOCS_INDEX = DOCS / "AGENTS.md"
SKILLS_INDEX = SKILLS / "AGENTS.md"
DOCS_MAP = SKILLS / "pyqcu-docs" / "references" / "project-docs-map.md"
DOCS_GUIDANCE = SKILLS / "pyqcu-docs" / "references" / "extracted-guidance.md"
DOCS_SKILL = SKILLS / "pyqcu-docs" / "SKILL.md"

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
TRIGGER_RE = re.compile(r"(当|用于|仅当|Use when|触发)")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def fail(errors: list[str], code: str, path: Path, message: str) -> None:
    errors.append(f"[{code}] {rel(path)}: {message}")


def frontmatter(path: Path) -> tuple[str, str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("缺少起始 frontmatter 分隔符")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise ValueError("缺少结束 frontmatter 分隔符") from exc
    block = lines[1:end]
    name = ""
    description_lines: list[str] = []
    in_description = False
    for line in block:
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
            in_description = False
        elif line.startswith("description:"):
            in_description = True
            first = line.split(":", 1)[1].strip()
            if first and first != "|":
                description_lines.append(first)
        elif in_description and (line.startswith(" ") or line.startswith("\t")):
            description_lines.append(line.strip())
        elif in_description:
            in_description = False
    metadata = "\n".join(block)
    return name, " ".join(description_lines).strip(), metadata


def local_links(path: Path) -> Iterable[tuple[str, Path, str]]:
    text = path.read_text(encoding="utf-8")
    for raw in LINK_RE.findall(text):
        target_path, _, fragment = raw.partition("#")
        target = target_path.split("?", 1)[0]
        if not target:
            if fragment:
                yield raw, path.resolve(), fragment
            continue
        if "://" in target or target.startswith("mailto:"):
            continue
        yield raw, (path.parent / target).resolve(), fragment


def balanced_fences(path: Path) -> bool:
    count = sum(
        line.strip().startswith("```")
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    return count % 2 == 0


def h1_count(path: Path) -> int:
    count = 0
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("# "):
            count += 1
    return count


def github_slug(heading: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
    text = text.replace("`", "").strip().lower()
    chars: list[str] = []
    for char in text:
        if char.isspace():
            chars.append("-")
        elif char in "-_" or char.isalnum():
            chars.append(char)
        elif unicodedata.category(char).startswith("L") or unicodedata.category(char).startswith("N"):
            chars.append(char)
    slug = re.sub(r"-+", "-", "".join(chars)).strip("-")
    return slug


def markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            anchors.add(github_slug(match.group(1)))
    return anchors


def check_markdown_file(path: Path, errors: list[str]) -> None:
    if not balanced_fences(path):
        fail(errors, "MARKDOWN", path, "代码围栏数量不平衡")
    for raw, target, fragment in local_links(path):
        if not target.exists():
            fail(errors, "LINK", path, f"本地链接不存在: {raw}")
            continue
        if fragment and target.suffix.lower() == ".md":
            decoded = unquote(fragment)
            if re.fullmatch(r"L\d+(?:-L\d+)?", decoded):
                fail(errors, "LINK", path, f"Markdown 章节链接不能使用代码行锚点: {raw}")
            elif decoded not in markdown_anchors(target):
                fail(errors, "LINK", path, f"Markdown 标题锚点不存在: {raw}")


def check_docs(errors: list[str]) -> int:
    docs = sorted(DOCS.glob("*.md"))
    if not docs:
        fail(errors, "DOCS", DOCS, "未找到 Markdown 文档")
        return 0

    docs_index_text = DOCS_INDEX.read_text(encoding="utf-8")
    map_text = DOCS_MAP.read_text(encoding="utf-8")
    guidance_text = DOCS_GUIDANCE.read_text(encoding="utf-8")
    skill_text = DOCS_SKILL.read_text(encoding="utf-8")

    indexed = set(re.findall(r"^\|\s*`([^`]+\.md)`\s*\|", docs_index_text, re.M))
    mapped = set(re.findall(r"docs/([A-Za-z0-9_.-]+\.md)", map_text))
    covered = set(re.findall(r"docs/([A-Za-z0-9_.-]+\.md)", guidance_text))
    actual = {path.name for path in docs}
    for name in sorted(actual):
        if not re.fullmatch(r"(?:AGENTS|ORGANIZATION|[a-z0-9_]+)\.md", name):
            fail(errors, "DOCS-NAME", DOCS / name, "文件名应为小写下划线、AGENTS.md 或 ORGANIZATION.md")

    docs_count_match = re.search(r"当前 `docs/` 顶层共有 \*\*(\d+) 份", docs_index_text)
    if not docs_count_match:
        fail(errors, "DOCS-COUNT", DOCS_INDEX, "未找到 docs 顶层 Markdown 计数")
    elif int(docs_count_match.group(1)) != len(actual):
        fail(
            errors,
            "DOCS-COUNT",
            DOCS_INDEX,
            f"计数 {docs_count_match.group(1)} != 实际 {len(actual)}",
        )

    index_required = actual - {"AGENTS.md"}
    for code, listed, required in (
        ("DOCS-INDEX", indexed, index_required),
        ("DOCS-MAP", mapped, actual),
        ("DOCS-GUIDANCE", covered, actual),
    ):
        for missing in sorted(required - listed):
            fail(errors, code, DOCS, f"未登记 {missing}")
        for extra in sorted(listed - actual):
            fail(errors, code, DOCS, f"登记了不存在的文档 {extra}")

    for name in sorted(actual - {"AGENTS.md"}):
        index_rows = len(re.findall(
            rf"^\|\s*`{re.escape(name)}`\s*\|", docs_index_text, re.M))
        map_rows = len(re.findall(
            rf"\]\([^)]*docs/{re.escape(name)}(?:#[^)]*)?\)", map_text))
        guidance_rows = len(re.findall(
            rf"^\|\s*`docs/{re.escape(name)}`\s*\|", guidance_text, re.M))
        if index_rows != 1:
            fail(errors, "DOCS-UNIQUE", DOCS_INDEX, f"{name} 应登记 1 次，实际 {index_rows}")
        if map_rows != 1:
            fail(errors, "DOCS-UNIQUE", DOCS_MAP, f"{name} 应登记 1 次，实际 {map_rows}")
        if guidance_rows != 1:
            fail(errors, "DOCS-UNIQUE", DOCS_GUIDANCE, f"{name} 应登记 1 次，实际 {guidance_rows}")

    skill_names = {path.name for path in SKILLS.iterdir() if path.is_dir()}
    for line in map_text.splitlines():
        if not line.startswith("| [`docs/"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            fail(errors, "DOCS-MAP", DOCS_MAP, f"映射行应为 5 列: {line}")
            continue
        primary = re.findall(r"`(pyqcu-[a-z0-9-]+)`", cells[2])
        consumers = re.findall(r"`(pyqcu-[a-z0-9-]+)`", cells[3])
        if len(primary) != 1:
            fail(errors, "DOCS-MAP", DOCS_MAP, f"Primary 必须恰好一个: {cells[0]}")
        elif primary[0] not in skill_names:
            fail(errors, "DOCS-MAP", DOCS_MAP, f"Primary 技能不存在: {primary[0]}")
        for consumer in consumers:
            if consumer not in skill_names:
                fail(errors, "DOCS-MAP", DOCS_MAP, f"Consumer 技能不存在: {consumer}")

    for path, text in ((DOCS_MAP, map_text), (DOCS_SKILL, skill_text)):
        match = re.search(r"(\d+)\s*份.*Markdown", text)
        if not match:
            fail(errors, "DOCS-COUNT", path, "未找到 Markdown 文档计数")
        elif int(match.group(1)) != len(actual):
            fail(
                errors,
                "DOCS-COUNT",
                path,
                f"计数 {match.group(1)} != 实际 {len(actual)}",
            )

    for path in docs:
        check_markdown_file(path, errors)
    return len(actual)


def check_skills(errors: list[str]) -> int:
    skill_dirs = sorted(path for path in SKILLS.iterdir() if path.is_dir())
    names: set[str] = set()
    index_text = SKILLS_INDEX.read_text(encoding="utf-8")
    indexed = set(re.findall(r"^\|\s*`(pyqcu-[a-z0-9-]+)`\s*\|", index_text, re.M))
    descriptions: dict[str, str] = {}

    count_match = re.search(r"技能表（(\d+)/(\d+)）", index_text)
    if not count_match:
        fail(errors, "SKILLS-COUNT", SKILLS_INDEX, "未找到 技能表（N/N） 计数")
    elif int(count_match.group(1)) != len(skill_dirs) or int(count_match.group(2)) != len(skill_dirs):
        fail(
            errors,
            "SKILLS-COUNT",
            SKILLS_INDEX,
            f"计数 {count_match.group(1)}/{count_match.group(2)} != 实际 {len(skill_dirs)}",
        )

    root_text = ROOT_AGENTS.read_text(encoding="utf-8")
    root_count = re.search(r"`skills/`.*?（(\d+) 个技能目录", root_text)
    if not root_count:
        fail(errors, "SKILLS-COUNT", ROOT_AGENTS, "未找到 skills 技能目录计数")
    elif int(root_count.group(1)) != len(skill_dirs):
        fail(
            errors,
            "SKILLS-COUNT",
            ROOT_AGENTS,
            f"计数 {root_count.group(1)} != 实际 {len(skill_dirs)}",
        )

    for skill_dir in skill_dirs:
        name = skill_dir.name
        if not re.fullmatch(r"pyqcu-[a-z0-9-]+", name):
            fail(errors, "SKILL-NAME", skill_dir, "目录名应为 pyqcu- 前缀的小写连字符名")
        if name not in indexed:
            fail(errors, "SKILL-INDEX", SKILLS_INDEX, f"未登记 {name}")
        index_rows = len(re.findall(
            rf"^\|\s*`{re.escape(name)}`\s*\|", index_text, re.M))
        if index_rows != 1:
            fail(errors, "SKILL-UNIQUE", SKILLS_INDEX, f"{name} 应登记 1 次，实际 {index_rows}")
        if name in names:
            fail(errors, "SKILL-NAME", skill_dir, "技能名重复")
        names.add(name)

        skill_file = skill_dir / "SKILL.md"
        agents_file = skill_dir / "AGENTS.md"
        if not skill_file.is_file():
            fail(errors, "SKILL-FILE", skill_dir, "缺少 SKILL.md")
            continue
        if not agents_file.is_file():
            fail(errors, "SKILL-FILE", skill_dir, "缺少 AGENTS.md")

        try:
            declared_name, description, _metadata = frontmatter(skill_file)
        except ValueError as exc:
            fail(errors, "FRONTMATTER", skill_file, str(exc))
            continue
        if declared_name != name:
            fail(errors, "FRONTMATTER", skill_file, f"name={declared_name!r} != {name!r}")
        if not description:
            fail(errors, "DESCRIPTION", skill_file, "description 为空")
        else:
            if not TRIGGER_RE.search(description):
                fail(errors, "DESCRIPTION", skill_file, "description 未包含明确触发条件")
            if "完整生成 skill" in description:
                fail(errors, "DESCRIPTION", skill_file, "description 不应使用工作流摘要式措辞")
            if len(description) > 260:
                fail(errors, "DESCRIPTION", skill_file, f"description 过长: {len(description)} 字符")
            previous = descriptions.get(description)
            if previous is not None:
                fail(errors, "DESCRIPTION", skill_file, f"description 与 {previous} 重复")
            descriptions[description] = name
        lines = len(skill_file.read_text(encoding="utf-8").splitlines())
        if lines >= 500:
            fail(errors, "SKILL-SIZE", skill_file, f"SKILL.md 为 {lines} 行，应拆分到 references/")
        if h1_count(skill_file) != 1:
            fail(errors, "SKILL-TITLE", skill_file, "SKILL.md 必须且只能有一个一级标题")

        check_markdown_file(skill_file, errors)
        if agents_file.is_file():
            check_markdown_file(agents_file, errors)
            agents_text = agents_file.read_text(encoding="utf-8")
            agents_lines = len(agents_text.splitlines())
            expected_title = f"# AGENTS.md — {name}"
            if not agents_text.splitlines() or agents_text.splitlines()[0] != expected_title:
                fail(errors, "AGENTS-TITLE", agents_file, f"首行应为 {expected_title!r}")
            if agents_lines > 12:
                fail(errors, "AGENTS-SIZE", agents_file, f"AGENTS.md 为 {agents_lines} 行，应保持简短导航")
            if "SKILL.md" not in agents_text:
                fail(errors, "AGENTS-POINTER", agents_file, "未指向 SKILL.md")

        pointer_text = skill_file.read_text(encoding="utf-8")
        if agents_file.is_file():
            pointer_text += "\n" + agents_file.read_text(encoding="utf-8")
        allowed_dirs = {"references", "scripts", "assets", "evals"}
        for child in skill_dir.iterdir():
            if child.is_dir() and not child.name.startswith(".") and child.name not in allowed_dirs:
                fail(errors, "SKILL-LAYOUT", child, "未登记的技能子目录")
            if child.is_file() and child.name not in {"SKILL.md", "AGENTS.md"}:
                if child.name not in pointer_text and str(child.relative_to(skill_dir)) not in pointer_text:
                    fail(errors, "POINTER", child, "技能根目录文件未在 SKILL.md/AGENTS.md 中声明")
        for directory in ("references", "scripts", "assets", "evals"):
            payload_dir = skill_dir / directory
            if not payload_dir.exists():
                continue
            for payload in sorted(payload_dir.rglob("*")):
                if (
                    not payload.is_file()
                    or payload.name.startswith(".")
                    or "__pycache__" in payload.parts
                ):
                    continue
                relative = str(payload.relative_to(skill_dir))
                if relative not in pointer_text and payload.name not in pointer_text:
                    fail(errors, "POINTER", payload, "未在 SKILL.md/AGENTS.md 中声明读取分支")

    for listed in sorted(indexed - names):
        fail(errors, "SKILL-INDEX", SKILLS_INDEX, f"登记了不存在的技能 {listed}")

    for bad in sorted(path for path in (ROOT / "docs").rglob("AGENT.md")):
        fail(errors, "AGENTS-NAME", bad, "历史单数文件名必须改为 AGENTS.md")
    for bad in sorted(path for path in SKILLS.rglob("AGENT.md")):
        fail(errors, "AGENTS-NAME", bad, "历史单数文件名必须改为 AGENTS.md")

    return len(skill_dirs)


def main() -> int:
    errors: list[str] = []
    docs_count = check_docs(errors)
    skills_count = check_skills(errors)
    if errors:
        print(f"FAIL docs={docs_count} skills={skills_count} errors={len(errors)}")
        for error in errors:
            print(error)
        return 1
    print(f"PASS docs={docs_count} skills={skills_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
