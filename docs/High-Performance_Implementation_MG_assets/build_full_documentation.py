#!/usr/bin/env python3
"""Build the synchronized LaTeX/PDF view of the PyQCU master document.

The Markdown file is the editable source of truth.  This script converts its
curated body to LaTeX, appends the exact source-file and attachment inventories,
writes the sibling .tex file, and compiles the sibling .pdf twice.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import argparse
import html
import os
import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MASTER = DOCS / "report_pyqcu_full_documentation_20260928.md"
TEX = DOCS / "report_pyqcu_full_documentation_20260928.tex"
PDF = DOCS / "report_pyqcu_full_documentation_20260928.pdf"
ARCHIVE_BEGIN = "<!-- BEGIN GENERATED FULL ARCHIVE -->"
ARCHIVE_END = "<!-- END GENERATED FULL ARCHIVE -->"


def latex_escape(text: str) -> str:
    """Escape plain text while preserving protected math placeholders."""
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_path(path: Path) -> str:
    """Render a path in the main font with discretionary break opportunities."""
    text = latex_escape(str(path))
    for delimiter in ("/", "\\_", "-", "."):
        text = text.replace(delimiter, delimiter + r"\allowbreak{}")
    return text


def protect_inline_math(line: str, store: list[str]) -> str:
    """Protect simple $...$ spans outside inline code spans."""
    out: list[str] = []
    i = 0
    in_code = False
    while i < len(line):
        ch = line[i]
        if ch == "`":
            in_code = not in_code
            out.append(ch)
            i += 1
            continue
        if not in_code and ch == "$" and i + 1 < len(line):
            j = i + 1
            while j < len(line):
                if line[j] == "$" and line[j - 1] != "\\":
                    break
                j += 1
            if j < len(line):
                key = f"PYQCUMATHINLINE{len(store):06d}"
                store.append(line[i + 1 : j])
                out.append(key)
                i = j + 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def protect_math(text: str) -> tuple[str, list[str]]:
    """Replace display and inline math with parser-safe placeholders."""
    lines = text.splitlines()
    store: list[str] = []
    out: list[str] = []
    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```") or line.startswith("~~~~"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if not in_fence and line.strip() == "$$":
            j = i + 1
            block: list[str] = []
            while j < len(lines) and lines[j].strip() != "$$":
                block.append(lines[j])
                j += 1
            if j >= len(lines):
                raise RuntimeError(f"unclosed display math at line {i + 1}")
            key = f"PYQCUMATHBLOCK{len(store):06d}"
            body = " ".join(part.strip() for part in block if part.strip())
            store.append("\\begin{displaymath}\n" + body + "\n\\end{displaymath}")
            out.append(key)
            i = j + 1
            continue
        out.append(line if in_fence else protect_inline_math(line, store))
        i += 1
    return "\n".join(out) + "\n", store


class LatexRenderer:
    def __init__(self) -> None:
        self.out: list[str] = []
        self.seen_first_h1 = False

    def emit(self, text: str) -> None:
        self.out.append(text)

    def render_inline_children(self, children) -> str:
        parts: list[str] = []
        for tok in children or []:
            typ = tok.type
            if typ == "text":
                parts.append(latex_escape(tok.content))
            elif typ == "code_inline":
                parts.append(r"\pycode{" + latex_escape(tok.content) + "}")
            elif typ == "strong_open":
                parts.append(r"\textbf{")
            elif typ == "strong_close":
                parts.append("}")
            elif typ == "em_open":
                parts.append(r"\emph{")
            elif typ == "em_close":
                parts.append("}")
            elif typ == "softbreak":
                parts.append(" ")
            elif typ == "hardbreak":
                parts.append(r"\\")
            elif typ == "link_open":
                href = html.escape(tok.attrGet("href") or "", quote=True)
                parts.append(r"\href{" + href + "}{")
            elif typ == "link_close":
                parts.append("}")
            elif typ == "image":
                src = unquote(tok.attrGet("src") or "")
                alt = tok.content or "image"
                parts.append(
                    r"\begin{figure}[p]\centering"
                    r"\includegraphics[width=0.96\linewidth,height=0.72\textheight,"
                    r"keepaspectratio]{\detokenize{"
                    + src
                    + r"}}\caption{"
                    + latex_escape(alt)
                    + r"}\end{figure}"
                )
            elif typ == "html_inline":
                value = tok.content
                if value.strip().lower() == "<br>":
                    parts.append(r"\\")
                else:
                    parts.append(latex_escape(value))
            else:
                parts.append(latex_escape(tok.content or ""))
        return "".join(parts)

    def render_inline_token(self, tok) -> str:
        return self.render_inline_children(tok.children)

    def render_table(self, tokens, start: int) -> int:
        rows: list[list[str]] = []
        row: list[str] | None = None
        i = start + 1
        while i < len(tokens) and tokens[i].type != "table_close":
            tok = tokens[i]
            if tok.type == "tr_open":
                row = []
            elif tok.type == "tr_close" and row is not None:
                rows.append(row)
                row = None
            elif tok.type == "inline" and row is not None:
                row.append(self.render_inline_token(tok))
            i += 1
        if not rows:
            return i + 1
        width_count = max(len(r) for r in rows)
        if width_count <= 3:
            width, size = 0.30, r"\small"
        elif width_count == 4:
            width, size = 0.23, r"\small"
        elif width_count == 5:
            width, size = 0.18, r"\footnotesize"
        elif width_count == 6:
            width, size = 0.15, r"\scriptsize"
        elif width_count == 7:
            width, size = 0.13, r"\scriptsize"
        else:
            width, size = max(0.08, 0.92 / width_count), r"\tiny"
        spec = "@{}" + "".join(
            rf">{{\raggedright\arraybackslash}}p{{{width:.3f}\textwidth}}" for _ in range(width_count)
        ) + "@{}"
        self.emit(r"\begin{center}" + size + "\n")
        self.emit(r"\setlength{\tabcolsep}{2pt}" + "\n")
        self.emit(r"\begin{longtable}{" + spec + "}\n")
        self.emit(r"\toprule" + "\n")
        head, body = rows[0], rows[1:]
        for idx in range(width_count):
            self.emit((r"\textbf{" + (head[idx] if idx < len(head) else "") + "}"))
            self.emit(" & " if idx + 1 < width_count else r" \\" + "\n")
        self.emit(r"\midrule\endhead" + "\n")
        for record in body:
            for idx in range(width_count):
                self.emit(record[idx] if idx < len(record) else "")
                self.emit(" & " if idx + 1 < width_count else r" \\" + "\n")
        self.emit(r"\bottomrule\end{longtable}" + "\n")
        self.emit(r"\end{center}" + "\n")
        return i + 1

    def render(self, tokens) -> str:
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            typ = tok.type
            if typ == "heading_open":
                level = max(1, min(4, int(tok.tag[1])))
                title = self.render_inline_token(tokens[i + 1])
                title = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", title)
                if level == 1 and not self.seen_first_h1:
                    self.seen_first_h1 = True
                    i += 3
                    continue
                cmd = ["section", "section", "subsection", "subsubsection"][level - 1]
                self.emit(f"\\{cmd}{{{title}}}\n")
                i += 3
                continue
            if typ == "paragraph_open":
                j = i + 1
                while j < len(tokens) and tokens[j].type != "paragraph_close":
                    j += 1
                text = self.render_inline_token(tokens[i + 1])
                self.emit(text + "\n\n")
                i = j + 1
                continue
            if typ == "bullet_list_open":
                self.emit(r"\begin{itemize}" + "\n")
            elif typ == "bullet_list_close":
                self.emit(r"\end{itemize}" + "\n")
            elif typ == "ordered_list_open":
                self.emit(r"\begin{enumerate}" + "\n")
            elif typ == "ordered_list_close":
                self.emit(r"\end{enumerate}" + "\n")
            elif typ == "list_item_open":
                self.emit(r"\item ")
            elif typ == "blockquote_open":
                self.emit(r"\begin{quote}" + "\n")
            elif typ == "blockquote_close":
                self.emit(r"\end{quote}" + "\n")
            elif typ in {"fence", "code_block"}:
                body = tok.content.replace(r"\end{Verbatim}", r"\textbackslash end\{Verbatim\}")
                self.emit(r"\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\scriptsize]" + "\n")
                self.emit(body)
                if not body.endswith("\n"):
                    self.emit("\n")
                self.emit(r"\end{Verbatim}" + "\n")
            elif typ == "table_open":
                i = self.render_table(tokens, i)
                continue
            elif typ == "hr":
                self.emit(r"\par\noindent\rule{\linewidth}{0.4pt}\par" + "\n")
            elif typ == "html_block":
                value = tok.content
                if value.strip().lower() not in {"<details>", "</details>"}:
                    self.emit(latex_escape(value) + "\n\n")
            i += 1
        return "".join(self.out)


def render_markdown_body(markdown_text: str) -> tuple[str, dict[str, str]]:
    protected, store = protect_math(markdown_text)
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(protected)
    body = LatexRenderer().render(tokens)
    placeholders = {
        f"PYQCUMATHINLINE{i:06d}": expr for i, expr in enumerate(store) if expr.startswith("\\begin{displaymath}") is False
    }
    blocks = {
        f"PYQCUMATHBLOCK{i:06d}": expr for i, expr in enumerate(store) if expr.startswith("\\begin{displaymath}")
    }
    for key, value in {**placeholders, **blocks}.items():
        body = body.replace(key, value)
    return body, {**placeholders, **blocks}


def source_inventory(docs: Path) -> tuple[str, str]:
    all_files = sorted(
        p
        for p in docs.rglob("*")
        if p.is_file()
        and p not in {MASTER, TEX, PDF}
        and "__pycache__" not in p.parts
        and p.suffix.lower() != ".pyc"
    )
    sources = [p for p in all_files if p.suffix.lower() in {".tex", ".md"}]
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sources:
        groups[sha256(path.read_bytes()).hexdigest()].append(path)

    def rank(path: Path):
        rel = path.relative_to(docs)
        if rel.parent == Path("."):
            return (0, str(rel))
        if rel.parts[0] == "data":
            return (1, str(rel))
        return (2, str(rel))

    unique = []
    for digest, paths in groups.items():
        paths = sorted(paths, key=rank)
        unique.append((digest, paths[0], paths))
    unique.sort(key=lambda item: rank(item[1]))

    listing = [
        r"\clearpage",
        r"\section*{原文快照}",
        (
            f"共 {len(all_files)} 个文件，{len(sources)} 个 TeX/Markdown 源路径，"
            f"{len(unique)} 份唯一源内容。以下逐字输入每个唯一源文件；"
            "重复内容以 aliases 注明。"
        ),
        r"\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,breakatwhitespace=false,"
        r"columns=fullflexible,keepspaces=true,showstringspaces=false}",
    ]
    for idx, (digest, primary, aliases) in enumerate(unique, 1):
        rel = primary.relative_to(docs)
        listing.extend(
            [
                rf"\subsection*{{E.2.{idx} \pathstyle{{{latex_path(rel)}}}}}",
                rf"\noindent SHA256: \texttt{{{digest}}}; {primary.stat().st_size} bytes.",
            ]
        )
        if len(aliases) > 1:
            alias_text = ", ".join(f"\\pathstyle{{{latex_path(p.relative_to(docs))}}}" for p in aliases[1:])
            listing.append(r"\par\small 完全相同内容也见：" + alias_text + ".")
        listing.append(r"\lstinputlisting{\detokenize{" + str(primary) + "}}")

    manifest = [
        r"\clearpage",
        r"\section*{全附件与全数据 SHA256 清册}",
        rf"以下列出除主文档及其生成 \texttt{{.tex/.pdf}} 外的 {len(all_files)} 个文件。",
        r"\begingroup\tiny",
        r"\begin{longtable}{@{}p{0.44\textwidth}rrp{0.36\textwidth}@{}}",
        r"\toprule 路径 & 字节 & 类型 & SHA256 \\ \midrule \endhead",
    ]
    for path in all_files:
        rel = path.relative_to(docs)
        digest = sha256(path.read_bytes()).hexdigest()
        suffix = path.suffix.lower() or "(none)"
        manifest.append(
            rf"\pathstyle{{{latex_path(rel)}}} & {path.stat().st_size} & "
            rf"\texttt{{{suffix}}} & \texttt{{\seqsplit{{{digest}}}}} \\"
        )
    manifest.extend([r"\bottomrule\end{longtable}", r"\endgroup"])
    return "\n".join(listing) + "\n", "\n".join(manifest) + "\n"


def regenerate_master_archive() -> None:
    """Refresh the collapsible exact-source and full-attachment appendix."""
    complete = MASTER.read_text(encoding="utf-8")
    before, rest = complete.split(ARCHIVE_BEGIN, 1)
    _, after = rest.split(ARCHIVE_END, 1)
    all_files = sorted(
        p
        for p in DOCS.rglob("*")
        if p.is_file()
        and p not in {MASTER, TEX, PDF}
        and "__pycache__" not in p.parts
        and p.suffix.lower() != ".pyc"
    )
    skip_snapshot_dirs = {
        DOCS / "High-Performance_Implementation_MG_assets" / name
        for name in ("paper_build", "beamer_build", "speaker_script_build", "algorithm_build")
    }

    def under_skip(path: Path) -> bool:
        return any(base == path or base in path.parents for base in skip_snapshot_dirs)

    sources = [
        p
        for p in all_files
        if p.suffix.lower() in {".tex", ".md"}
        and p.name != TEX.name
        and not under_skip(p)
    ]
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sources:
        groups[sha256(path.read_bytes()).hexdigest()].append(path)

    def rank(path: Path):
        rel = path.relative_to(DOCS)
        if rel.parent == Path("."):
            return (0, str(rel))
        if rel.parts[0] == "data":
            return (1, str(rel))
        return (2, str(rel))

    unique = []
    for digest, paths in groups.items():
        paths = sorted(paths, key=rank)
        unique.append((digest, paths[0], paths))
    unique.sort(key=lambda item: rank(item[1]))

    lines = [
        "",
        "## 附录 E. 原文快照与全附件清册",
        "",
        "本附录用于满足“明文全文、全附件、全数据”的归档要求。",
        "",
        "- 正文部分是经过裁决和重写的可读总册。",
        "- 本附录中的“原文快照”逐字保留 `docs/**` 下所有 `.tex`、`.md` 源文件；完全相同的内容只保留一次，其余路径列为 aliases。",
        "- “附件清册”列出 `docs/**` 下所有文件的大小、SHA256、类型和路径。",
        "- 主文档自身和由它生成的 `.tex/.pdf` 不进入自身源哈希清单，避免自指循环。",
        "- 仓库根 `data/**`、`logs/**` 的规范实体按项目组织规则保持原位置，并由正文的来源映射建立索引。",
        "",
        "### E.1 源文档去重统计",
        "",
        f"- `docs/**` 总文件数：{len(all_files)}",
        f"- `.tex/.md` 源文件路径数：{len(sources)}",
        f"- 去重后的唯一源内容数：{len(unique)}",
        f"- 唯一源内容总字节数：{sum(p.stat().st_size for _, p, _ in unique)}",
        "",
        "| 唯一内容 SHA256 | 主快照 | 大小 | 行数 | 别名数 |",
        "|---|---|---:|---:|---:|",
    ]
    for digest, primary, aliases in unique:
        line_count = len(primary.read_text(encoding="utf-8", errors="replace").splitlines())
        lines.append(
            f"| `{digest}` | `{primary.relative_to(DOCS)}` | {primary.stat().st_size} | "
            f"{line_count} | {len(aliases) - 1} |"
        )
    lines.extend(["", "### E.2 全文源快照", ""])
    for idx, (digest, primary, aliases) in enumerate(unique, 1):
        rel = primary.relative_to(DOCS)
        body = primary.read_text(encoding="utf-8", errors="replace")
        body = "\n".join(
            line.rstrip(" \t")
            + ("&#9250;" * (len(line) - len(line.rstrip(" "))))
            + ("&#8677;" * (len(line) - len(line.rstrip("\t"))))
            for line in body.splitlines()
        )
        body = body.replace("</details>", "&lt;/details&gt;")
        lang = "tex" if primary.suffix.lower() == ".tex" else "markdown"
        lines.append("<details>")
        lines.append(
            f"<summary><strong>E.2.{idx} {rel}</strong> | SHA256 "
            f"<code>{digest}</code> | {primary.stat().st_size} bytes</summary>"
        )
        lines.append("")
        lines.append(
            "该快照为可读文本视图；行尾空格和制表符分别以 "
            "`&#9250;`、`&#8677;` 显式标出，原始字节可由 SHA256 与路径审计。"
        )
        lines.append("")
        if len(aliases) > 1:
            lines.append(
                "完全相同的内容也见："
                + "、".join(f"`{p.relative_to(DOCS)}`" for p in aliases[1:])
                + "。"
            )
            lines.append("")
        lines.append(f"~~~~{lang}")
        lines.append(body)
        if not body.endswith("\n"):
            lines.append("")
        lines.append("~~~~")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.extend(["### E.3 全附件与全数据清册", ""])

    def category(path: Path) -> str:
        rel = path.relative_to(DOCS)
        if rel.parent == Path("."):
            return "docs 顶层文档"
        if rel.parts[0] == "data":
            return "docs/data 数据与图表"
        if rel.parts[0] == "High-Performance_Implementation_MG_assets":
            return "论文、汇报与算法资产"
        if rel.parts[0].startswith("张鑫 508"):
            return "508 报告解包 Office 资产"
        return "其他"

    by_category: dict[str, list[Path]] = defaultdict(list)
    for path in all_files:
        by_category[category(path)].append(path)
    for cat in sorted(by_category):
        lines.extend(
            [
                "<details>",
                f"<summary><strong>{cat}：{len(by_category[cat])} 个文件</strong></summary>",
                "",
                "| 路径 | 字节 | 类型 | SHA256 |",
                "|---|---:|---|---|",
            ]
        )
        for path in sorted(by_category[cat]):
            digest = sha256(path.read_bytes()).hexdigest()
            lines.append(
                f"| `{path.relative_to(DOCS)}` | {path.stat().st_size} | "
                f"`{path.suffix.lower() or '(none)'}` | `{digest}` |"
            )
        lines.extend(["", "</details>", ""])
    lines.extend(
        [
            "### E.4 归档完整性口径",
            "",
            "- “全文”覆盖 `docs/**` 中所有独立 `.tex` 与 `.md` 文本源，并按内容哈希去重。",
            "- “全附件”覆盖 `docs/**` 中 PDF、PNG、PPTX、Office XML、脚本和构建附属文件，每条记录 SHA256。",
            "- “全数据”覆盖文档目录中的表格、图表和数据描述；仓库根 `data/**`、`logs/**` 的规范实体保持原位置。",
            "- 源文件变化后必须重新运行本构建脚本，再执行公式编译、链接检查、围栏检查和 `git diff --check`。",
        ]
    )
    refreshed = (
        before
        + ARCHIVE_BEGIN
        + "\n\n"
        + "\n".join(lines).rstrip()
        + "\n"
        + ARCHIVE_END
        + after
    )
    MASTER.write_text(refreshed, encoding="utf-8")


def build_tex() -> None:
    complete = MASTER.read_text(encoding="utf-8")
    body_md = complete.split(ARCHIVE_BEGIN, 1)[0].rstrip() + "\n"
    body_md = body_md.replace("<br>", r"\\")
    body_tex, _ = render_markdown_body(body_md)
    source_tex, manifest_tex = source_inventory(DOCS)
    source_digest = sha256(MASTER.read_bytes()).hexdigest()
    preamble = rf"""% Generated by High-Performance_Implementation_MG_assets/build_full_documentation.py
% Source: report_pyqcu_full_documentation_20260928.md
% Source SHA256: {source_digest}
\documentclass[UTF8,a4paper,10pt]{{ctexart}}
\usepackage[a4paper,margin=1.7cm,headheight=15pt]{{geometry}}
\usepackage{{amsmath,amssymb,mathtools}}
\usepackage{{graphicx}}
\usepackage{{longtable,booktabs,array,tabularx}}
\usepackage{{fancyvrb}}
\usepackage{{fvextra}}
\usepackage{{listings}}
\usepackage{{xurl}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage{{enumitem}}
\usepackage{{microtype}}
\usepackage{{newunicodechar}}
\usepackage{{seqsplit}}
\setmonofont{{DejaVu Sans Mono}}
\setCJKmonofont{{FandolFang-Regular.otf}}
\setCJKfamilyfont{{fallback}}{{AR PL UMing CN}}
\DeclareRobustCommand{{\pycode}}[1]{{\texorpdfstring{{\texttt{{\seqsplit{{#1}}}}}}{{#1}}}}
\DeclareRobustCommand{{\pathstyle}}[1]{{\texorpdfstring{{\small #1}}{{#1}}}}
\newunicodechar{{≈}}{{\ensuremath{{\approx}}}}
\newunicodechar{{≤}}{{\ensuremath{{\le}}}}
\newunicodechar{{≥}}{{\ensuremath{{\ge}}}}
\newunicodechar{{×}}{{\ensuremath{{\times}}}}
\newunicodechar{{→}}{{\ensuremath{{\to}}}}
\newunicodechar{{†}}{{\ensuremath{{\dagger}}}}
\newunicodechar{{①}}{{\textcircled{{1}}}}
\newunicodechar{{②}}{{\textcircled{{2}}}}
\newunicodechar{{③}}{{\textcircled{{3}}}}
\newunicodechar{{✓}}{{\ensuremath{{\checkmark}}}}
\newunicodechar{{昇}}{{\CJKfamily{{fallback}}昇}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.4em}}
\setlength{{\emergencystretch}}{{3em}}
\sloppy
\setlist{{topsep=2pt,itemsep=1pt,parsep=0pt}}
\lstset{{basicstyle=\ttfamily\scriptsize,breaklines=true,breakatwhitespace=false,
columns=fullflexible,keepspaces=true,showstringspaces=false,
extendedchars=true,inputencoding=utf8}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{PyQCU Documentation Master}}
\fancyhead[R]{{2026-09-28}}
\fancyfoot[C]{{\thepage}}
\title{{PyQCU 全文汇总：Wilson/Clover、Strict MultiGrid、CUDA/MPI 实现与最终性能对照}}
\author{{PyQCU 文档整合}}
\date{{2026-09-28}}
\begin{{document}}
\maketitle
\tableofcontents
\clearpage
"""
    appendix = (
        "\n\\appendix\n"
        "\\part{原文与附件}\n"
        + source_tex
        + "\n"
        + manifest_tex
    )
    TEX.write_text(preamble + body_tex + appendix + "\n\\end{document}\n", encoding="utf-8")


def compile_pdf() -> None:
    build = Path(tempfile.mkdtemp(prefix="pyqcu-full-tex."))
    for name in (MASTER.name, TEX.name):
        (build / name).symlink_to(DOCS / name)
    for item in DOCS.iterdir():
        if item.name in {MASTER.name, TEX.name, PDF.name}:
            continue
        (build / item.name).symlink_to(item, target_is_directory=item.is_dir())
    command = [
        "xelatex",
        "--shell-escape",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        TEX.name,
    ]
    for _ in range(2):
        result = subprocess.run(
            command,
            cwd=build,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-120:])
            raise RuntimeError(f"xelatex failed with code {result.returncode}\n{tail}")
    produced = build / PDF.name
    if not produced.is_file() or produced.stat().st_size == 0:
        raise RuntimeError("xelatex returned success but produced no PDF")
    shutil.copy2(produced, PDF)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex-only", action="store_true")
    args = parser.parse_args()
    regenerate_master_archive()
    build_tex()
    if not args.tex_only:
        compile_pdf()
    print(f"tex={TEX} bytes={TEX.stat().st_size}")
    if not args.tex_only:
        print(f"pdf={PDF} bytes={PDF.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
