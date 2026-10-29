#!/usr/bin/env python3
"""Contract tests for repository ``env.sh`` QCU defaults."""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENV_SCRIPT = ROOT / "env.sh"
PRINT_KEYS = (
    "PYQCU_ROOT",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "PYQCU_CUDA_LIB",
    "PYQCU_DCU_LIB",
    "PYQCU_KUPL_BUILD_DIR",
    "PYQCU_KUPL_LIB",
    "QCU_KUPL_VARIANT",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "KUPL_OMP_THREADS",
    "KUPL_NUMA_BIND",
    "KUPL_FUSED_BICG_DOTS",
    "KUPL_DISABLE_CLOVER_CACHE",
    "KUPL_ALLOW_OVERSUBSCRIBE",
    "KUPL_HUGEPAGE",
    "OMP_PROC_BIND",
    "OMP_PLACES",
)


def source_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": "/stale/lib/PyQCU-test34_2",
    }
    if extra:
        env.update(extra)
    command = "source \"$1\""
    for key in PRINT_KEYS:
        command += f"; printf '{key}=%s\\n' \"${{{key}-}}\""
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command, "bash",
         str(ENV_SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    result = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


class EnvScriptTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("zsh"), "zsh is not installed")
    def test_env_script_can_be_sourced_by_zsh(self) -> None:
        command = (
            'source "$1" >/dev/null; first=$PYTHONPATH; '
            'source "$1" >/dev/null; test "$first" = "$PYTHONPATH"; '
            'printf "%s\\n%s\\n%s\\n" "$PYQCU_ROOT" '
            '"${PYTHONPATH%%:*}" "$PYQCU_KUPL_LIB"'
        )
        completed = subprocess.run(
            ["zsh", "-f", "-c", command, "zsh", str(ENV_SCRIPT)],
            cwd=ROOT,
            env={"HOME": os.environ.get("HOME", "/tmp"),
                 "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                 "PYTHONPATH": "/stale/lib/PyQCU-test34_2"},
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                str(ROOT),
                str(ROOT),
                str(ROOT / "cpp/kupl/qcu/libqcu.so"),
            ],
        )

    def test_sourcing_prints_start_and_completion_markers(self) -> None:
        completed = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", 'source "$1"',
             "bash", str(ENV_SCRIPT)],
            cwd=ROOT,
            env={"HOME": os.environ.get("HOME", "/tmp"),
                 "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("is sourcing", completed.stdout)
        self.assertIn("is done", completed.stdout)

    def test_public_kupl_library_is_a_stable_symlink(self) -> None:
        library = ROOT / "cpp/kupl/qcu/libqcu.so"
        self.assertTrue(library.is_symlink())
        target = Path(os.readlink(library))
        self.assertFalse(target.is_absolute())
        self.assertEqual(target.name, "libqcu.so")
        if library.is_file():
            self.assertTrue(library.resolve().is_relative_to(ROOT))

    def test_repeated_source_does_not_duplicate_repository_paths(self) -> None:
        command = (
            'source "$1" >/dev/null; first=$PYTHONPATH; '
            'source "$1" >/dev/null; test "$first" = "$PYTHONPATH"; '
            'printf "%s\\n" "$PYTHONPATH"'
        )
        completed = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", command, "bash",
             str(ENV_SCRIPT)],
            cwd=ROOT,
            env={"HOME": os.environ.get("HOME", "/tmp"),
                 "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), str(ROOT))

    def test_repository_and_library_defaults_are_exported(self) -> None:
        values = source_env()
        self.assertEqual(values["PYQCU_ROOT"], str(ROOT))
        self.assertEqual(
            values["PYQCU_CUDA_LIB"],
            str(ROOT / "cpp/cuda/qcu/libqcu.so"))
        self.assertEqual(
            values["PYQCU_DCU_LIB"],
            str(ROOT / "data/build/cuda-qcu-dcu/libqcu.so"))
        self.assertEqual(
            values["PYQCU_KUPL_BUILD_DIR"],
            str(ROOT / "cpp/kupl/qcu/build"))
        self.assertEqual(
            values["PYQCU_KUPL_LIB"],
            str(ROOT / "cpp/kupl/qcu/libqcu.so"))
        self.assertEqual(values["QCU_KUPL_VARIANT"], values["PYQCU_KUPL_LIB"])

    def test_current_repository_precedes_stale_pythonpath(self) -> None:
        values = source_env({
            "PYTHONPATH": f"{ROOT}:/stale/lib/PyQCU-test34_2:{ROOT}:",
        })
        entries = values["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(entries[0], str(ROOT))
        self.assertEqual(entries.count(str(ROOT)), 1)
        self.assertIn("/stale/lib/PyQCU-test34_2", entries[1:])
        ld_entries = values["LD_LIBRARY_PATH"].split(os.pathsep)
        self.assertIn(str(ROOT / "cpp/cuda/qcu"), ld_entries)
        self.assertNotIn("", ld_entries)

    def test_explicit_library_paths_are_preserved(self) -> None:
        values = source_env({
            "PYQCU_CUDA_LIB": "/opt/cuda/libqcu.so",
            "PYQCU_DCU_LIB": "/opt/dcu/libqcu.so",
            "PYQCU_KUPL_LIB": "/opt/kupl/libqcu.so",
        })
        self.assertEqual(values["PYQCU_CUDA_LIB"], "/opt/cuda/libqcu.so")
        self.assertEqual(values["PYQCU_DCU_LIB"], "/opt/dcu/libqcu.so")
        self.assertEqual(values["PYQCU_KUPL_LIB"], "/opt/kupl/libqcu.so")
        self.assertEqual(values["QCU_KUPL_VARIANT"], "/opt/kupl/libqcu.so")

    def test_kupl_public_link_stays_default_for_custom_build_dir(self) -> None:
        values = source_env({"PYQCU_KUPL_BUILD_DIR": "/tmp/kupl-build"})
        self.assertEqual(
            values["PYQCU_KUPL_LIB"], str(ROOT / "cpp/kupl/qcu/libqcu.so"))

    def test_thread_variables_are_synchronized_without_overwriting_both(self) -> None:
        from_kupl = source_env({"KUPL_OMP_THREADS": "6"})
        self.assertEqual(from_kupl["OMP_NUM_THREADS"], "6")
        from_omp = source_env({"OMP_NUM_THREADS": "4"})
        self.assertEqual(from_omp["KUPL_OMP_THREADS"], "4")
        both = source_env({
            "KUPL_OMP_THREADS": "6",
            "OMP_NUM_THREADS": "4",
        })
        self.assertEqual(both["KUPL_OMP_THREADS"], "6")
        self.assertEqual(both["OMP_NUM_THREADS"], "4")

    def test_library_binding_clears_conflicting_omp_placement(self) -> None:
        values = source_env({
            "KUPL_NUMA_BIND": "1",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        })
        self.assertEqual(values["KUPL_NUMA_BIND"], "1")
        self.assertEqual(values["OMP_PROC_BIND"], "")
        self.assertEqual(values["OMP_PLACES"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
