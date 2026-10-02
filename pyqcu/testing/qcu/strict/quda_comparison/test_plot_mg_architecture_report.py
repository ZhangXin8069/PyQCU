from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).with_name("plot_mg_architecture_report.py").parent
sys.path.insert(0, str(MODULE_DIR))
MODULE_PATH = MODULE_DIR / "plot_mg_architecture_report.py"
SPEC = importlib.util.spec_from_file_location(
    "plot_mg_architecture_report",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ratio_decomposition_preserves_product():
    ratio, iteration_ratio, cost_ratio = MODULE.ratio_components(
        0.15,
        0.72,
        31,
        59,
    )
    assert abs(ratio - 4.8) < 1.0e-12
    assert abs(iteration_ratio - 59 / 31) < 1.0e-12
    assert abs(
        ratio - iteration_ratio * cost_ratio
    ) < 1.0e-12


def test_test27_matrix_contract_and_20261001_reference():
    units = MODULE.load_units(MODULE.DEFAULT_INPUT_DIR)
    memory = MODULE.load_memory(MODULE.DEFAULT_MEMORY_CSV)
    audit = MODULE.audit_units(units, memory)
    assert audit["combined_units"] == 66
    assert audit["side_records"] == 132
    assert audit["level_counts"] == {1: 22, 2: 22, 3: 22}
    unit_rows = MODULE._unit_rows(units, memory)
    assert MODULE.validate_reference(unit_rows, MODULE.DEFAULT_REFERENCE_DIR) == 66


def test_data_build_writes_complete_derived_matrix(tmp_path: Path):
    summary = MODULE.build_data(
        MODULE.DEFAULT_INPUT_DIR,
        MODULE.DEFAULT_MEMORY_CSV,
        MODULE.DEFAULT_REFERENCE_DIR,
        tmp_path,
    )
    assert summary["ratios"]["all_66_median"] == MODULE.statistics.median(
        [
            row["ratio_quda_over_pyqcu"]
            for row in MODULE._unit_rows(
                MODULE.load_units(MODULE.DEFAULT_INPUT_DIR),
                MODULE.load_memory(MODULE.DEFAULT_MEMORY_CSV),
            )
        ]
    )
    for name in (
        "unit_analysis.csv",
        "group_analysis.csv",
        "algorithm_profiles.csv",
        "summary.json",
        "source_hashes.json",
    ):
        path = tmp_path / name
        assert path.is_file()
        assert path.stat().st_size > 0
