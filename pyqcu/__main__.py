"""Command-line diagnostics and self-test for an installed PyQCU.

``pyqcu`` (or ``python -m pyqcu``) prints the package version, dependency
versions, and native-backend availability. ``pyqcu --selftest`` runs the Torch,
KUPL, and CUDA layers with independent residual checks; a missing native
backend is reported as ``SKIP``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import platform
import sys
import warnings
from pathlib import Path

from . import __version__


@contextlib.contextmanager
def _quiet_optional_warnings():
    """Keep the optional-dependency notice out of normal CLI output."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"TileLang is unavailable.*",
            category=RuntimeWarning)
        yield


def _module_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return str(version(name))
    except Exception:
        pass
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as error:  # pragma: no cover - diagnostic only
        return f"unavailable ({type(error).__name__}: {error})"


def _kupl_status() -> dict:
    try:
        spec = importlib.util.find_spec("pyqcu.kupl.qcu")
        if spec is None:
            return {"available": False, "error": "KUPL binding is not installed"}
        import pyqcu.kupl as kupl_package

        candidate = os.environ.get("PYQCU_KUPL_LIB")
        if not candidate:
            candidate = str(Path(kupl_package.__file__).with_name("libqcu.so"))
        if not Path(candidate).is_file():
            return {"available": False, "error": f"library not found: {candidate}"}
        return {"available": True, "path": candidate}
    except Exception as error:  # pragma: no cover - diagnostic only
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def _cuda_library_path(origin: str) -> Path | None:
    override = os.environ.get("PYQCU_CUDA_LIB")
    candidates = [Path(override)] if override else [
        Path(origin).parent / "libqcu.so",
        Path(origin).parent.parent / "libqcu.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _cuda_status() -> dict:
    try:
        spec = importlib.util.find_spec("pyqcu.cuda.qcu")
        if spec is None or not spec.origin:
            return {"available": False,
                    "error": "Cython bridge is not built in this installation"}
        library = _cuda_library_path(spec.origin)
        if library is None:
            return {"available": False,
                    "error": "native library libqcu.so not found",
                    "path": spec.origin}
        return {"available": True, "path": spec.origin,
                "library": str(library)}
    except Exception as error:  # pragma: no cover - diagnostic only
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def collect() -> dict:
    import pyqcu

    mpi_size, mpi_rank = _launcher_info()
    return {
        "package": "pyqcu",
        "version": pyqcu.__version__,
        "path": pyqcu.__file__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mpi_size": mpi_size,
        "mpi_rank": mpi_rank,
        "dependencies": {
            name: _module_version(name)
            for name in ("numpy", "torch", "h5py", "mpi4py")
        },
        "backends": {
            "kupl": _kupl_status(),
            "cuda": _cuda_status(),
        },
    }


def _selftest_exit_code(report: dict) -> int:
    return 0 if report.get("status") == "PASS" else 2


def _one_line(value, limit: int = 180) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@contextlib.contextmanager
def _capture_native_stdout():
    """Keep C++ solver banners out of the CLI's own stdout.

    The native CUDA/KUPL libraries print diagnostics directly to file
    descriptor 1. ``contextlib.redirect_stdout`` cannot capture those, so this
    context manager reserves a temporary file descriptor and restores it in a
    ``finally`` block.
    """
    import os
    import tempfile

    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as capture:
        os.dup2(capture.fileno(), 1)
        try:
            yield capture
        finally:
            sys.stdout.flush()
            try:
                import ctypes

                # Native CUDA/KUPL code uses C printf; flush it before the
                # original file descriptor is restored, or the buffered line
                # would appear after the JSON report.
                ctypes.CDLL(None).fflush(None)
            except Exception:
                pass
            os.dup2(saved, 1)
            os.close(saved)


def _launcher_rank() -> int | None:
    """Read the MPI rank from launcher variables without importing mpi4py."""
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "I_MPI_RANK",
                 "MV2_COMM_WORLD_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _launcher_size() -> int | None:
    """Read the MPI size from launcher variables without importing mpi4py."""
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "I_MPI_SIZE",
                 "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _launcher_info() -> tuple[int, int]:
    rank = _launcher_rank()
    size = _launcher_size()
    if rank is None:
        rank = 0
    if size is None:
        size = 1 if rank == 0 else rank + 1
    return size, rank


def _print_selftest(report: dict) -> None:
    print(f"pyqcu {report['version']} self-test")
    for name, layer in report["layers"].items():
        status = str(layer.get("status", "FAIL"))
        if status == "PASS":
            iterations = layer.get("iterations")
            detail = (
                f"iterations={'n/a' if iterations is None else iterations} "
                f"residual={float(layer.get('true_residual', 0.0)):.3e} "
                f"{layer.get('device', '')}")
        elif status == "SKIP":
            detail = _one_line(layer.get("reason", "unavailable"))
        else:
            detail = _one_line(layer.get("error", "unknown error"))
        print(f"  {name:<6}: {status:<4} {detail}")
    print(f"overall: {report['status']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyqcu",
        description="Show PyQCU version/backends or run the install self-test.")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--json", action="store_true", help="print a machine-readable report")
    parser.add_argument(
        "--selftest", action="store_true",
        help="run Torch/KUPL/CUDA layers with independent residual checks")
    parser.add_argument(
        "--lattice", nargs="*", default=("4", "4", "4", "4"),
        metavar="N",
        help="self-test lattice (four even integers >= 4)")
    argv_list = list(sys.argv[1:] if argv is None else argv)
    rank = _launcher_rank()
    if rank is not None and rank != 0 and any(
            arg in ("--version", "--help", "-h") for arg in argv_list):
        return 0
    args = parser.parse_args(argv_list)

    if args.selftest:
        values = tuple(args.lattice)

        def reject_lattice(message: str) -> int:
            if rank not in (None, 0):
                return 2
            if args.json:
                print(json.dumps({
                    "package": "pyqcu",
                    "version": __version__,
                    "lattice": list(values),
                    "layers": {},
                    "status": "FAIL",
                    "error": message,
                }, indent=2))
                return 2
            parser.error(message)
            return 2

        if len(values) != 4:
            return reject_lattice(
                "self-test lattice requires exactly four integers")
        try:
            lattice = tuple(int(value) for value in values)
        except (TypeError, ValueError):
            return reject_lattice("self-test lattice must be four integers")

        try:
            from . import _selftest
        except Exception as error:
            message = (
                f"self-test dependencies unavailable: "
                f"{type(error).__name__}: {error}")
            if rank not in (None, 0):
                return 2
            if args.json:
                print(json.dumps({
                    "package": "pyqcu",
                    "version": __version__,
                    "lattice": list(args.lattice),
                    "layers": {},
                    "status": "FAIL",
                    "error": message,
                }, indent=2))
                return 2
            print(f"pyqcu: {message}", file=sys.stderr)
            return 2

        try:
            with _quiet_optional_warnings(), _capture_native_stdout() as capture:
                report = _selftest.run_all(lattice)
                capture.seek(0)
                native_output = capture.read()
        except ValueError as error:
            if rank not in (None, 0):
                return 2
            if args.json:
                print(json.dumps({
                    "package": "pyqcu",
                    "version": __version__,
                    "lattice": list(lattice),
                    "layers": {},
                    "status": "FAIL",
                    "error": str(error),
                }, indent=2))
                return 2
            parser.error(str(error))
        except Exception as error:
            message = f"self-test failed: {type(error).__name__}: {error}"
            if rank not in (None, 0):
                return 2
            if args.json:
                print(json.dumps({
                    "package": "pyqcu",
                    "version": __version__,
                    "lattice": list(lattice),
                    "layers": {},
                    "status": "FAIL",
                    "error": message,
                }, indent=2))
                return 2
            print(f"pyqcu: {message}", file=sys.stderr)
            return 2
        if report.get("mpi_rank", 0) != 0:
            # Non-zero ranks keep stdout clean, but a failure must still be
            # visible on stderr and must reach mpirun as a non-zero exit.
            if report["status"] == "FAIL":
                print(
                    f"pyqcu self-test FAIL on MPI rank "
                    f"{report['mpi_rank']}", file=sys.stderr)
                for name, layer in report["layers"].items():
                    if layer.get("status") == "FAIL":
                        print(
                            f"  {name}: "
                            f"{_one_line(layer.get('error', 'failed'))}",
                            file=sys.stderr)
            return _selftest_exit_code(report)
        if native_output.strip() and report["status"] == "FAIL":
            report["native_output"] = _one_line(native_output, 2000)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_selftest(report)
            if report.get("native_output"):
                print("native output:", report["native_output"],
                      file=sys.stderr)
        return _selftest_exit_code(report)

    with _quiet_optional_warnings():
        report = collect()
    if report.get("mpi_rank", 0) != 0:
        return 0
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
