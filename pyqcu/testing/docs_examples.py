"""Execute the Python code blocks in the Read the Docs sources.

The checker is intentionally strict: a block must either contain an
``assert`` (so it verifies its own result) or be marked ``# docs-skip`` /
``# pseudo-code``. Blocks on one page share a namespace and run in page order.

Usage::

    python -m pyqcu.testing.docs_examples
    python -m pyqcu.testing.docs_examples --lang zh --keep-going
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

PAGES = ("overview", "usage", "solvers", "performance", "install",
         "troubleshooting")
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
ROOT = pathlib.Path(__file__).resolve().parents[2]


def blocks(path: pathlib.Path) -> list[str]:
    return BLOCK.findall(path.read_text(encoding="utf-8"))


def run_page(lang: str, page: str, keep_going: bool) -> tuple[int, int, int]:
    path = ROOT / "docs" / lang / f"{page}.md"
    if not path.is_file():
        print(f"{lang}/{page}: missing {path}", file=sys.stderr)
        return 0, 0, 1
    namespace: dict = {"__name__": "__docs_example__"}
    ok = skipped = failed = 0
    for index, code in enumerate(blocks(path)):
        if "# docs-skip" in code or "# pseudo-code" in code:
            skipped += 1
            continue
        if "assert " not in code:
            failed += 1
            print(f"{lang}/{page}#{index}: no assertion", file=sys.stderr)
            if not keep_going:
                break
            continue
        try:
            exec(compile(code, f"{lang}/{page}#{index}", "exec"), namespace)
            ok += 1
        except Exception as error:
            failed += 1
            print(f"{lang}/{page}#{index}: {type(error).__name__}: {error}",
                  file=sys.stderr)
            if not keep_going:
                break
    status = "OK" if failed == 0 else "FAILED"
    print(f"{lang}/{page:<11} blocks ok={ok} skipped={skipped} "
          f"failed={failed} -> {status}")
    return ok, skipped, failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("en", "zh"), default=None)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    languages = (args.lang,) if args.lang else ("en", "zh")
    total_ok = total_skipped = total_failed = 0
    for lang in languages:
        for page in PAGES:
            ok, skipped, failed = run_page(lang, page, args.keep_going)
            total_ok += ok
            total_skipped += skipped
            total_failed += failed
            if failed and not args.keep_going:
                break
        if total_failed and not args.keep_going:
            break
    print(f"total: ok={total_ok} skipped={total_skipped} "
          f"failed={total_failed}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
