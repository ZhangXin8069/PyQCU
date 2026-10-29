"""Optional TileLang behavior when the dependency cannot be imported."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

_BLOCK_TILELANG = """
import sys

class _BlockTileLang:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tilelang" or fullname.startswith("tilelang."):
            raise ModuleNotFoundError("blocked TileLang for optional-dependency test")
        return None

sys.meta_path.insert(0, _BlockTileLang())
"""


def _run_without_tilelang(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""))
    return subprocess.run(
        [sys.executable, "-B", "-c", _BLOCK_TILELANG + textwrap.dedent(source)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def test_tilelang_probe_warns_once_and_caches_unavailable() -> None:
    completed = _run_without_tilelang(
        """
        from concurrent.futures import ThreadPoolExecutor
        import warnings
        from pyqcu._tilelang import import_tilelang, tilelang_available

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with ThreadPoolExecutor(max_workers=8) as executor:
                values = list(executor.map(
                    lambda _: import_tilelang(), range(16)))
            assert values == [None] * 16
            assert tilelang_available() is False

        messages = [str(item.message) for item in caught
                    if issubclass(item.category, RuntimeWarning)]
        assert len(messages) == 1, messages
        assert "TileLang is unavailable" in messages[0]
        print("tilelang-probe-ok")
        """
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "tilelang-probe-ok" in completed.stdout
    assert "\nError:" not in "\n" + completed.stderr


def test_torch_tools_cpu_path_falls_back_without_tilelang() -> None:
    completed = _run_without_tilelang(
        """
        import warnings
        import torch

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from pyqcu.torch.tools import Eexyzt_exyzt2Exyzt

        left = torch.arange(96, dtype=torch.float32).reshape(2, 3, 2, 2, 2, 2)
        right = torch.arange(48, dtype=torch.float32).reshape(3, 2, 2, 2, 2)
        actual = Eexyzt_exyzt2Exyzt(left, right)
        expected = torch.einsum("Eexyzt,exyzt->Exyzt", left, right)
        torch.testing.assert_close(actual, expected)
        messages = [str(item.message) for item in caught
                    if issubclass(item.category, RuntimeWarning)
                    and "TileLang is unavailable" in str(item.message)]
        assert len(messages) == 1, messages
        print("tilelang-cpu-fallback-ok")
        """
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "tilelang-cpu-fallback-ok" in completed.stdout


def test_dcu_tilelang_entry_skips_instead_of_erroring() -> None:
    completed = _run_without_tilelang(
        """
        import runpy
        runpy.run_path("pyqcu/testing/dcu/conftest_2.py", run_name="__main__")
        """
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "SKIP: TileLang is unavailable" in completed.stdout
    assert "TileLang is unavailable" in completed.stderr
    assert "Traceback" not in output


def test_npu_tilelang_entry_warns_and_returns_zero() -> None:
    completed = _run_without_tilelang(
        """
        import runpy
        runpy.run_path("pyqcu/testing/npu/conftest_1.py", run_name="__main__")
        """
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "TileLang is unavailable" in completed.stderr
    assert "\nError:" not in "\n" + completed.stderr
    assert "Traceback" not in output


def test_tilelang_only_matmul_test_is_skipped() -> None:
    completed = _run_without_tilelang(
        """
        import pytest
        import pyqcu.testing as testing

        try:
            testing.test_matmul()
        except pytest.skip.Exception as exc:
            assert "TileLang is unavailable" in str(exc)
            print("tilelang-matmul-skipped")
        else:
            raise AssertionError("test_matmul() did not skip")
        """
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "tilelang-matmul-skipped" in completed.stdout
    assert "Traceback" not in output


def test_dcu_single_tilelang_test_warns_and_returns_zero() -> None:
    completed = _run_without_tilelang(
        """
        import os
        import runpy
        import torch

        os.environ["PYQCU_ENABLE_DCU_TESTS"] = "1"
        torch.version.hip = "test-hip"
        torch.cuda.is_available = lambda: True
        runpy.run_path("pyqcu/testing/dcu/conftest_1.py", run_name="__main__")
        """
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "TileLang is unavailable" in completed.stderr
    assert "\nError:" not in "\n" + completed.stderr
    assert "Traceback" not in output
