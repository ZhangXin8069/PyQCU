from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("plot_mg_absolute_times.py")
SPEC = importlib.util.spec_from_file_location("plot_mg_absolute_times", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _unit(levels: int, side_levels: dict[str, list[dict[str, object]]]):
    return {
        "levels": levels,
        "path": Path(f"v100__8x8x8x16__c64__l{levels}__trace-on.json"),
        "document": {
            "sides": {
                side: {"status": "ok", "mg_levels": rows}
                for side, rows in side_levels.items()
            }
        },
    }


def test_stage_rows_drop_zero_quda_placeholder():
    unit = _unit(
        3,
        {
            "quda": [
                {
                    "level": 0,
                    "pre_smoother_seconds": 1.0,
                    "coarse_solver_seconds": 4.0,
                },
                {
                    "level": 1,
                    "pre_smoother_seconds": 2.0,
                    "coarse_solver_seconds": 3.0,
                },
                {
                    "level": 2,
                    "pre_smoother_seconds": 0.0,
                    "coarse_solver_seconds": 0.0,
                },
            ]
        },
    )
    rows = MODULE.stage_rows(unit, "quda")
    assert [row["level"] for row in rows] == [0, 1]
    assert rows[-1]["seconds"]["coarse"] == 3.0


def test_steady_seconds_uses_reference_solver_for_mg1():
    unit = {
        "levels": 1,
        "path": Path("v100__8x8x8x16__c64__l1__trace-off.json"),
        "document": {
            "sides": {
                "pyqcu": {
                    "status": "ok",
                    "timing": {"steady": {"median_seconds": 99.0}},
                    "reference_solver": {
                        "steady": {"median_seconds": 1.25}
                    },
                }
            }
        },
    }
    assert MODULE.steady_seconds(unit, "pyqcu") == 1.25


def test_steady_seconds_uses_mg_timing_for_mg2():
    unit = {
        "levels": 2,
        "path": Path("v100__8x8x8x16__c64__l2__trace-off.json"),
        "document": {
            "sides": {
                "quda": {
                    "status": "ok",
                    "timing": {"steady": {"median_seconds": 2.5}},
                    "reference_solver": {
                        "steady": {"median_seconds": 99.0}
                    },
                }
            }
        },
    }
    assert MODULE.steady_seconds(unit, "quda") == 2.5
