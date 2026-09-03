"""8^4 QUDA MultiGrid setup-only 闸门的 CPU/协议快速测试。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import smoke_quda_multigrid as smoke


class _CopyOnReadParam:
    """模拟 PyQUDA：getter 返回副本，setter 才会写入 backing struct。"""

    def __init__(self) -> None:
        self._values = {
            "n_level": 2,
            "geo_block_size": [[2, 2, 2, 2, 1, 1], [4, 4, 4, 4, 1, 1]],
            "spin_block_size": [2, 1, 1, 1, 1],
            "n_vec": [24, 24, 24, 24, 24],
            "n_block_ortho": [1, 1, 1, 1, 1],
            "nu_pre": [0, 0, 0, 0, 0],
            "nu_post": [8, 8, 8, 8, 8],
            "setup_use_mma": [1, 1, 1, 1, 1],
            "dslash_use_mma": [1, 1, 1, 1, 1],
            "transfer_use_mma": [1, 1, 1, 1, 1],
            "vec_load": [0, 0, 0, 0, 0],
            "compute_null_vector": 1,
            "generate_all_levels": 1,
            "precision_null": [4, 4, 4, 4, 4],
        }

    def __getattr__(self, name: str):
        try:
            return copy.deepcopy(self._values[name])
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        if name == "_values" or "_values" not in self.__dict__:
            object.__setattr__(self, name, value)
        elif name in self._values:
            self._values[name] = copy.deepcopy(value)
        else:
            object.__setattr__(self, name, value)


def test_fixed_protocol_is_not_cli_overridable():
    args = smoke._parser().parse_args([])
    assert not hasattr(args, "lat")
    assert smoke.LATTICE == (8, 8, 8, 8)
    assert smoke.NVECS == 12
    assert smoke.COARSE_SPIN == 2
    assert smoke.COARSE_COLOR == 24
    assert smoke.BLOCK == (2, 2, 2, 2)
    assert smoke.LEVELS == 2
    assert smoke.NU_PRE == 0
    assert smoke.NU_POST == 2
    assert args.resource_path == smoke.DEFAULT_RESOURCE_PATH


def test_whole_column_writeback_handles_copying_getters():
    param = _CopyOnReadParam()
    detached = param.n_vec
    detached[0] = 999
    assert param.n_vec[0] == 24

    readback = smoke._set_array_item_whole_column(param, "n_vec", 0, 12)
    assert readback == [12, 24, 24, 24, 24]
    assert param.n_vec == readback


def test_configure_and_post_setup_contract_are_fail_closed():
    param = _CopyOnReadParam()
    writes = smoke._configure_multigrid(
        param, boolean_false=0, boolean_true=1, compute_null_vector_yes=1)
    snapshot = smoke._snapshot_multigrid(param)
    checks = smoke._evaluate_multigrid_contract(snapshot)
    assert writes["n_vec"] == [12, 12, 12, 12, 12]
    assert snapshot["derived_coarse_spin"] == 2
    assert snapshot["derived_coarse_color"] == 24
    assert all(checks.values())

    broken = copy.deepcopy(snapshot)
    broken["n_vec"][0] = 11
    broken["derived_coarse_color"] = 22
    broken_checks = smoke._evaluate_multigrid_contract(broken)
    assert not broken_checks["n_vec"]
    assert not broken_checks["coarse_color"]


def test_unit_qdp_gauge_is_contiguous_complex128():
    np = pytest.importorskip("numpy")
    gauge = smoke._unit_gauge_qdp(np)
    assert gauge.shape == (4, 8, 8, 8, 8, 3, 3)
    assert gauge.dtype == np.dtype("complex128")
    assert gauge.flags.c_contiguous
    assert np.all(gauge == np.eye(3, dtype=np.complex128)[None, None, None, None, None])


def test_resource_path_is_repository_data_scoped():
    inside = smoke._resource_path("mg8-nc24-test")
    assert inside == (smoke.DATA_DIR / "mg8-nc24-test").resolve()
    with pytest.raises(smoke.SmokeFailure) as caught:
        smoke._resource_path("/tmp/pyqcu-mg8-nc24")
    assert caught.value.code == "resource_outside_repository_data"


def test_compile_instance_parser_and_failure_json():
    assert smoke._parse_instance_list("12,24") == [12, 24]
    assert smoke._parse_instance_list("12; 24; 12") == [12, 24]
    record = smoke._base_failure(
        smoke._parser().parse_args([]), smoke.SmokeFailure("test", "expected"))
    json.dumps(record, allow_nan=False)
    assert record["status"] == "failed"
    assert record["pass_marker"] is None


def test_setup_script_contains_no_solve_call():
    source = (HERE / "smoke_quda_multigrid.py").read_text(encoding="utf-8")
    assert ".invert(" not in source
