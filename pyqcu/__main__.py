"""Command-line diagnostics for an installed PyQCU.

Run ``python -m pyqcu`` after installation to print the package version, its
location, dependency versions, and the availability of the native KUPL and
CUDA backends. ``python -m pyqcu --json`` is the machine-readable form.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as error:  # pragma: no cover - diagnostic only
        return f"unavailable ({type(error).__name__}: {error})"


def _kupl_status() -> dict:
    try:
        from pyqcu.kupl import qcu

        return {
            "available": True,
            "path": str(qcu.library_path()),
            "symbols": len(qcu._SIGNATURES),
        }
    except Exception as error:  # pragma: no cover - diagnostic only
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def _cuda_status() -> dict:
    try:
        import pyqcu.cuda.qcu as qcu

        path = getattr(qcu, "__file__", None)
        if not path:
            return {"available": False,
                    "error": "Cython bridge is not built in this installation"}
        return {"available": True, "path": path}
    except Exception as error:  # pragma: no cover - diagnostic only
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def collect() -> dict:
    import pyqcu

    return {
        "package": "pyqcu",
        "version": pyqcu.__version__,
        "path": pyqcu.__file__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            name: _module_version(name)
            for name in ("numpy", "torch", "h5py", "mpi4py")
        },
        "backends": {
            "kupl": _kupl_status(),
            "cuda": _cuda_status(),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyqcu",
        description="Show PyQCU version and native-backend availability.")
    parser.add_argument("--json", action="store_true",
                        help="print a machine-readable report")
    args = parser.parse_args(argv)
    report = collect()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"pyqcu       : {report['version']}")
    print(f"location    : {report['path']}")
    print(f"python      : {report['python']} ({report['platform']})")
    for name, value in report["dependencies"].items():
        print(f"{name:<11}: {value}")
    kupl = report["backends"]["kupl"]
    print(f"kupl        : {kupl.get('path', kupl.get('error'))}")
    cuda = report["backends"]["cuda"]
    print(f"cuda bridge : {cuda.get('path', cuda.get('error'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
