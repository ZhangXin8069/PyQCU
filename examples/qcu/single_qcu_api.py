#!/usr/bin/env python3
"""逐函数单文件 ABI 检查：pyqcu.h、qcu_api.pxd 与 qcu.pyx 的符号集合。"""
from __future__ import annotations
import re
from pathlib import Path

def symbols(path: Path): return set(re.findall(r"\b(?:apply|test)[A-Za-z0-9]+Qcu\b", path.read_text()))
FUNCTION_TO_TEST = {
    "applyInitQcu": "single_qcu_*（各文件生命周期前置）", "applyEndQcu": "single_qcu_*（各文件生命周期收尾）",
    "testWilsonDslashQcu": "single_qcu_wilson_dslash.py", "applyWilsonDslashQcu": "single_qcu_wilson_dslash.py",
    "testCloverDslashQcu": "single_qcu_clover_dslash.py", "applyCloverDslashQcu": "single_qcu_clover_dslash.py",
    "applyWilsonBistabCgQcu": "single_qcu_wilson_bistabcg.py", "applyWilsonBistabCgDslashQcu": "single_qcu_wilson_bistabcg.py",
    "applyWilsonCgQcu": "single_qcu_wilson_cg.py", "applyWilsonCgDslashQcu": "single_qcu_wilson_cg.py",
    "applyLaplacianQcu": "single_qcu_laplacian.py", "applyCloverQcu": "single_qcu_clover.py", "applyCloversQcu": "single_qcu_clover.py",
    "applyDslashQcu": "single_qcu_clover_dslash.py", "applyGaussGaugeQcu": "single_qcu_gauss_gauge.py",
    "applyCloverBistabCgQcu": "single_qcu_clover_bistabcg.py", "applyCloverBistabCgDslashQcu": "single_qcu_clover_bistabcg.py",
    "applyCloverBistabCgPrepareQcu": "single_qcu_clover_bistabcg.py", "applyCloverBistabCgReconstructQcu": "single_qcu_clover_bistabcg.py",
    "applyMultigridRestrictQcu": "single_qcu_multigrid_transfer.py", "applyMultigridProLongQcu": "single_qcu_multigrid_transfer.py",
    "applyMultigridCoarseDslashQcu": "single_qcu_multigrid_coarse.py", "applyMultigridCoarseDslashWideQcu": "single_qcu_multigrid_coarse.py",
    "applyMultigridStrictCoarseQcu": "single_qcu_multigrid_strict.py", "applyMultigridStrictMatPCQcu": "single_qcu_multigrid_strict.py",
    "applyMultigridStrictFineMatPCQcu": "single_qcu_multigrid_strict.py", "applyMultigridStrictPrepareQcu": "single_qcu_multigrid_strict.py",
    "applyMultigridStrictReconstructQcu": "single_qcu_multigrid_strict.py", "applyMultigridStrictRestrictQcu": "single_qcu_multigrid_strict.py",
    "applyMultigridStrictProLongQcu": "single_qcu_multigrid_strict.py", "applyMultigridStrictVCycleQcu": "single_qcu_multigrid_strict.py",
    "applyMultigridStrictInitQcu": "single_qcu_multigrid_strict.py", "applyMultigridStrictEndQcu": "single_qcu_multigrid_strict.py",
    "applyMultigridStrictFgmresQcu": "single_qcu_multigrid_strict.py", "applyCloverMultigridQcu": "single_qcu_clover_multigrid.py",
    "verifyCloverMultigridQcu": "single_qcu_clover_multigrid.py",
}
def main() -> int:
    root = Path(__file__).resolve().parents[2]
    h = symbols(root/"cpp/cuda/qcu/python/pyqcu.h"); p = symbols(root/"pyqcu/cuda/qcu/qcu_api.pxd"); x = symbols(root/"pyqcu/cuda/qcu/qcu.pyx")
    missing = {"header_to_pxd":sorted(h-p),"header_to_pyx":sorted(h-x),"pxd_extra":sorted(p-h)}
    print({"abi": missing, "coverage_missing": sorted(h - set(FUNCTION_TO_TEST))})
    return 0 if not any(missing.values()) and h <= set(FUNCTION_TO_TEST) else 1
if __name__ == "__main__": raise SystemExit(main())
