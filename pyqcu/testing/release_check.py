"""Release gate for PyQCU.

Checks version consistency, sdist contents, bilingual documentation, Sphinx
builds, and (with ``--online``) the PyPI and Read the Docs endpoints.

Examples::

    python -m pyqcu.testing.release_check --offline
    python -m pyqcu.testing.release_check --online
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("overview", "usage", "solvers", "performance", "install")
BLOCK = re.compile(r"```[a-z]*\n(.*?)```", re.S)
RESULT: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, evidence: str) -> None:
    RESULT.append((name, bool(ok), evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}")


def version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def check_version(ver: str) -> None:
    import pyqcu

    record("version", pyqcu.__version__ == ver,
           f"pyproject={ver} package={pyqcu.__version__}")


def check_sdist(ver: str) -> Path:
    archive = ROOT / "dist" / f"pyqcu-{ver}.tar.gz"
    record("sdist exists", archive.is_file(), str(archive))
    if not archive.is_file():
        return archive
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        sources = tar.extractfile(
            f"pyqcu-{ver}/pyqcu.egg-info/SOURCES.txt")
        sources_text = sources.read().decode() if sources else ""
    record("sdist no testing", not any("pyqcu/testing/" in n for n in names),
           "pyqcu/testing is absent")
    record("sdist no binaries",
           not any(n.endswith((".so", ".so.0", ".o")) for n in names),
           "no .so/.o members")
    record("sdist native sources",
           any("/cpp/kupl/qcu/src/" in n for n in names)
           and any("/cpp/cuda/qcu/src/" in n for n in names),
           "KUPL and CUDA sources present")
    record("sdist no home paths", "/home/" not in sources_text,
           "SOURCES.txt has no absolute home path")
    record("sdist Chinese README",
           any(name.endswith("/README.zh-CN.md") for name in names),
           "README.zh-CN.md is present")
    try:
        check = subprocess.run(
            [sys.executable, "-m", "twine", "check", str(archive)],
            cwd=ROOT, text=True, capture_output=True)
        record("twine check", check.returncode == 0,
               (check.stdout or check.stderr).strip().splitlines()[-1])
    except FileNotFoundError:
        record("twine check", False, "twine is not installed")
    return archive


def check_bilingual() -> None:
    failed = []
    for page in PAGES:
        en = BLOCK.findall(
            (ROOT / "docs" / "en" / f"{page}.md").read_text(encoding="utf-8"))
        zh = BLOCK.findall(
            (ROOT / "docs" / "zh" / f"{page}.md").read_text(encoding="utf-8"))
        if len(en) != len(zh) or any(a != b for a, b in zip(en, zh)):
            failed.append(page)
    record("bilingual blocks", not failed,
           "all pages match" if not failed else f"mismatch: {failed}")


def check_examples() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pyqcu.testing.docs_examples", "--keep-going"],
        cwd=ROOT, text=True, capture_output=True)
    tail = (result.stdout or result.stderr).strip().splitlines()[-1]
    record("docs examples", result.returncode == 0, tail)


def check_sphinx() -> None:
    for lang in ("en", "zh"):
        output = ROOT / "data" / "release_check" / f"sphinx-{lang}"
        doctrees = ROOT / "data" / "release_check" / f"doctrees-{lang}"
        result = subprocess.run(
            [sys.executable, "-m", "sphinx", "-b", "html", "-d", str(doctrees),
             "-W", str(ROOT / "docs" / lang), str(output)],
            cwd=ROOT, text=True, capture_output=True)
        ok = result.returncode == 0 and "build succeeded" in result.stdout
        record(f"sphinx {lang}", ok,
               f"exit={result.returncode}")


def _fetch(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""
    except Exception as error:  # pragma: no cover - network dependent
        return -1, str(error).encode()


def check_pypi(ver: str) -> None:
    status, body = _fetch(f"https://pypi.org/pypi/pyqcu/{ver}/json")
    record("PyPI JSON", status == 200, f"HTTP {status}")
    if status == 200:
        info = json.loads(body)["info"]
        expected = {
            "Documentation", "Homepage", "Issues",
            "国内开发 (GitLab)", "国内发布 (Gitee)", "国外发布 (GitHub)",
        }
        urls = set((info.get("project_urls") or {}).keys())
        record("PyPI project links", expected <= urls,
               f"missing={sorted(expected - urls)}")
        record("PyPI repositories section",
               "## Repositories" in (info.get("description") or ""),
               "README.md contains ## Repositories")
    status, body = _fetch("https://pypi.org/simple/pyqcu/")
    record("PyPI simple index", status == 200
           and f"pyqcu-{ver}.tar.gz" in body.decode(errors="ignore"),
           f"HTTP {status}")


def check_rtd(ver: str) -> None:
    for lang in ("en/latest", "zh-cn/latest"):
        status, body = _fetch(f"https://pyqcu.readthedocs.io/{lang}/")
        text = body.decode(errors="ignore")
        record(f"RTD {lang}", status == 200 and ver in text,
               f"HTTP {status}, version_present={ver in text}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="skip PyPI/RTD network probes")
    parser.add_argument("--online", action="store_true",
                        help="include PyPI/RTD probes")
    parser.add_argument("--skip-sphinx", action="store_true")
    args = parser.parse_args(argv)
    ver = version()
    check_version(ver)
    check_sdist(ver)
    check_bilingual()
    check_examples()
    if not args.skip_sphinx:
        check_sphinx()
    if args.online and not args.offline:
        check_pypi(ver)
        check_rtd(ver)
    failed = [name for name, ok, _ in RESULT if not ok]
    print(f"\n{len(RESULT) - len(failed)}/{len(RESULT)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
