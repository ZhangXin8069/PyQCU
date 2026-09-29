from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).with_name("plot_mg_linear_report.py").parent
sys.path.insert(0, str(MODULE_DIR))
MODULE_PATH = MODULE_DIR / "plot_mg_linear_report.py"
SPEC = importlib.util.spec_from_file_location("plot_mg_linear_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _unit(device: str = "v100") -> dict[str, object]:
    return {
        "device": device,
        "precision": "c64",
        "lattice": "8x8x8x16",
        "levels": 2,
        "solver": "mg-2",
        "trace": "off",
        "path": Path(
            f"{device}__8x8x8x16__c64__l2__trace-off.json"
        ),
        "document": {
            "protocol": {
                "mass": 0.05,
                "kappa": 0.1234567901234568,
                "block_xyzt": [2, 2, 2, 2],
                "nvec": 12,
                "lattice_xyzt": [8, 8, 8, 16],
            }
        },
    }


def test_metadata_label_contains_test_contract():
    label = MODULE._metadata_label([_unit()])
    assert "mass=0.050" in label
    assert "block=2x2x2x2" in label
    assert "nvec=12" in label
    assert "kappa=" not in label


def test_case_label_contains_lattice_without_mass():
    label = MODULE._unit_case_label(_unit())
    assert "8x8x8x16" in label
    assert "mass=" not in label


def test_linear_limits_do_not_introduce_log_scale():
    lower, upper = MODULE._linear_limits([0.1, 1.0])
    assert lower == 0.0
    assert upper > 1.0


def test_plot_source_avoids_redundant_scale_suffixes():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert ", linear scale" not in source
    assert "only the residual norm uses a log axis" not in source
