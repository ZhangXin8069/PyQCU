"""Static and mock regressions for the CUDA QCU Python call boundary."""

from __future__ import annotations

import ast
import re
import threading
from pathlib import Path

import pytest
import torch

from pyqcu.cuda import QcuRuntime, define


_ROOT = Path(__file__).resolve().parents[3]
_CUDA_QCU = _ROOT / "pyqcu/cuda/qcu"
_HEADER = _ROOT / "cpp/cuda/qcu/python/pyqcu.h"
_DECLARATION = re.compile(
    r"\b(?P<return_type>void|int)\s+"
    r"(?P<name>(?:apply|test|get|verify)[A-Za-z0-9]+Qcu)\s*"
    r"\((?P<arguments>.*?)\)\s*;",
    re.DOTALL,
)
_PXD_DECLARATION = re.compile(
    r"\b(?P<return_type>void|int)\s+"
    r"(?P<name>(?:apply|test|get|verify)[A-Za-z0-9]+Qcu)\s*"
    r"\((?P<arguments>.*?)\)\s*(?:except\s*\+)?\s*nogil",
    re.DOTALL,
)
_PYX_DEFINITION = re.compile(
    r"^def\s+(?P<name>\w+)\s*\((?P<arguments>.*?)\):",
    re.MULTILINE | re.DOTALL,
)
_NONE_WRAPPERS = {
    "applyInitQcu",
    "applyEndQcu",
    "testWilsonDslashQcu",
    "applyWilsonDslashQcu",
    "testCloverDslashQcu",
    "applyCloverDslashQcu",
    "applyWilsonBistabCgQcu",
    "applyWilsonBistabCgDslashQcu",
    "applyWilsonCgQcu",
    "applyWilsonCgDslashQcu",
    "applyLaplacianQcu",
    "applyCloverQcu",
    "applyCloversQcu",
    "applyDslashQcu",
    "applyGaussGaugeQcu",
    "applyCloverBistabCgQcu",
    "applyCloverBistabCgDslashQcu",
    "applyCloverBistabCgPrepareQcu",
    "applyCloverBistabCgReconstructQcu",
    "applyMultigridRestrictQcu",
    "applyMultigridProLongQcu",
    "applyMultigridCoarseDslashQcu",
    "applyMultigridCoarseDslashWideQcu",
    "applyMultigridStrictCoarseQcu",
    "applyMultigridStrictMatPCQcu",
    "applyMultigridStrictFineMatPCQcu",
    "applyMultigridStrictPrepareQcu",
    "applyMultigridStrictReconstructQcu",
    "applyMultigridStrictRestrictQcu",
    "applyMultigridStrictProLongQcu",
    "applyMultigridStrictEndQcu",
    "applyCloverMultigridQcu",
}


def _normalize_argument(argument: str) -> str:
    normalized = " ".join(argument.split())
    normalized = re.sub(r"\s+[A-Za-z_]\w*$", "", normalized)
    normalized = re.sub(r"\*[A-Za-z_]\w*$", "*", normalized)
    return re.sub(r"\s*\*\s*", "*", normalized)


def _declarations(pattern: re.Pattern[str], path: Path):
    result = {}
    for match in pattern.finditer(path.read_text(encoding="utf-8")):
        arguments = tuple(
            _normalize_argument(argument)
            for argument in match.group("arguments").split(","))
        result[match.group("name")] = (
            match.group("return_type"), arguments)
    return result


def _parameter_names(arguments: str) -> tuple[str, ...]:
    names = []
    for argument in arguments.split(","):
        declaration = argument.split("=", 1)[0].strip()
        names.append(declaration.split()[-1])
    return tuple(names)


def _without_leading_underscore(value: str) -> str:
    return value[1:] if value.startswith("_") else value


def _unparse_annotation(node: ast.AST | None) -> str:
    if node is None:
        return "None"
    unparse = getattr(ast, "unparse", None)
    if unparse is not None:
        return unparse(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute):
        return f"{_unparse_annotation(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return (f"{_unparse_annotation(node.value)}["
                f"{_unparse_annotation(node.slice)}]")
    if isinstance(node, ast.Tuple):
        return ", ".join(_unparse_annotation(item) for item in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (f"{_unparse_annotation(node.left)} | "
                f"{_unparse_annotation(node.right)}")
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def test_cuda_c_header_and_cython_declarations_match_exactly() -> None:
    header = _declarations(_DECLARATION, _HEADER)
    qcu_api = _declarations(
        _PXD_DECLARATION, _CUDA_QCU / "qcu_api.pxd")
    qcu_public = _declarations(_PXD_DECLARATION, _CUDA_QCU / "qcu.pxd")

    assert len(header) == 40
    assert qcu_api == header
    assert qcu_public == header
    for path in (_CUDA_QCU / "qcu_api.pxd", _CUDA_QCU / "qcu.pxd"):
        source = path.read_text(encoding="utf-8")
        assert "void applyInitQcu" in source
        assert "except + nogil" in source
    assert header["applyMultigridStrictInitQcu"] == (
        "int", ("long long", "long long", "int", "unsigned long long*"))
    assert header["applyMultigridStrictFgmresQcu"][1][-1] == (
        "unsigned long long*")


def test_pyx_and_pyi_export_every_cuda_entry_with_matching_parameters() -> None:
    header = _declarations(_DECLARATION, _HEADER)
    pyx_source = (_CUDA_QCU / "qcu.pyx").read_text(encoding="utf-8")
    pyx = {
        match.group("name"): _parameter_names(match.group("arguments"))
        for match in _PYX_DEFINITION.finditer(pyx_source)
    }
    stub_tree = ast.parse(
        (_CUDA_QCU / "qcu.pyi").read_text(encoding="utf-8"))
    pyi = {
        node.name: tuple(
            _without_leading_underscore(argument.arg)
            for argument in node.args.args)
        for node in stub_tree.body if isinstance(node, ast.FunctionDef)
    }

    assert set(pyx) == set(header)
    assert pyi == {
        name: tuple(
            _without_leading_underscore(argument)
            for argument in arguments)
        for name, arguments in pyx.items()
    }
    assert pyx_source.count("with nogil:") == len(header)
    assert ".contiguous().data_ptr()" not in pyx_source


def test_cuda_wrapper_return_types_match_python_contract() -> None:
    pyx_source = (_CUDA_QCU / "qcu.pyx").read_text(encoding="utf-8")
    matches = list(_PYX_DEFINITION.finditer(pyx_source))
    bodies = {
        match.group("name"): pyx_source[
            match.end():matches[index + 1].start()
            if index + 1 < len(matches) else len(pyx_source)]
        for index, match in enumerate(matches)
    }
    for name in _NONE_WRAPPERS:
        assert "return " not in bodies[name], name
    for name in (
            "getCloverBistabCgIterationsQcu",
            "applyMultigridStrictVCycleQcu",
            "applyMultigridStrictInitQcu",
            "verifyCloverMultigridQcu"):
        assert "return " in bodies[name], name
    assert "return {" in bodies["applyMultigridStrictFgmresQcu"]

    stub_tree = ast.parse(
        (_CUDA_QCU / "qcu.pyi").read_text(encoding="utf-8"))
    annotations = {
        node.name: (
            "None" if isinstance(node.returns, ast.Constant)
            and node.returns.value is None else _unparse_annotation(node.returns))
        for node in stub_tree.body if isinstance(node, ast.FunctionDef)
    }
    for name in _NONE_WRAPPERS:
        assert annotations[name] == "None"
    for name in (
            "getCloverBistabCgIterationsQcu",
            "applyMultigridStrictVCycleQcu",
            "applyMultigridStrictInitQcu",
            "verifyCloverMultigridQcu"):
        assert annotations[name] == "int"
    assert annotations["applyMultigridStrictFgmresQcu"].startswith("dict[")


def _runtime_fixture(fake_qcu) -> tuple[QcuRuntime, torch.Tensor]:
    params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    params[define._DATA_TYPE_] = define._LAT_C64_
    params[define._SET_INDEX_] = 0
    argv = torch.zeros(define._ARGV_SIZE_, dtype=torch.float32)
    set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
    runtime = QcuRuntime(
        params=params,
        argv=argv,
        set_ptrs=set_ptrs,
        device=torch.device("cpu"),
        lattice=(4, 4, 4, 4),
        mass=0.1,
        csw=1.0,
        qcu_module=fake_qcu,
        slot_limit=2,
    )
    operand = torch.zeros(2, dtype=torch.complex64)
    return runtime, operand


def test_qcu_runtime_keeps_index_until_end_then_recycles_slot() -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            index = int(params[define._SET_INDEX_])
            events.append(("init", index))
            set_ptrs[index] = 11

        def applyEndQcu(self, set_ptrs, params):
            index = int(params[define._SET_INDEX_])
            events.append(("end", index))
            set_ptrs[index] = 0

        def applyCloverQcu(self, output, gauge, set_ptrs, params):
            events.append(("operation", int(params[define._SET_INDEX_])))

    fake = FakeQcu()
    runtime, operand = _runtime_fixture(fake)
    runtime.call("applyCloverQcu", operand, operand)

    assert events == [("init", 0), ("operation", 0), ("end", 0)]
    assert runtime.last_set_index == 0
    assert int(runtime.set_ptrs[0]) == 0


def test_qcu_runtime_session_reuses_one_native_initialization() -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            index = int(params[define._SET_INDEX_])
            events.append(("init", index))
            set_ptrs[index] = 17

        def applyEndQcu(self, set_ptrs, params):
            index = int(params[define._SET_INDEX_])
            events.append(("end", index))
            set_ptrs[index] = 0

        def applyCloverQcu(self, output, gauge, set_ptrs, params):
            events.append(("operation", int(params[define._SET_INDEX_])))

        def getCloverBistabCgIterationsQcu(self, set_ptrs, params):
            events.append(("getter", int(params[define._SET_INDEX_])))
            return 7

    runtime, operand = _runtime_fixture(FakeQcu())
    with runtime.session() as session:
        assert session.active and session.set_index == 0
        session("applyCloverQcu", operand, operand)
        assert session.read("getCloverBistabCgIterationsQcu") == 7
        session("applyCloverQcu", operand, operand)
        with pytest.raises(RuntimeError, match="not reentrant"):
            runtime.call("applyCloverQcu", operand, operand)

    assert events == [
        ("init", 0),
        ("operation", 0),
        ("getter", 0),
        ("operation", 0),
        ("end", 0),
    ]
    assert not session.active and session.set_index is None
    assert int(runtime.set_ptrs[0]) == 0
    with pytest.raises(RuntimeError, match="not active"):
        session("applyCloverQcu", operand, operand)


def test_qcu_runtime_session_poisoning_survives_caught_operation_error() -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            index = int(params[define._SET_INDEX_])
            events.append(("init", index))
            set_ptrs[index] = 19

        def applyEndQcu(self, set_ptrs, params):
            index = int(params[define._SET_INDEX_])
            events.append(("end", index))
            set_ptrs[index] = 0

        def applyCloverQcu(self, *args):
            events.append("operation")
            raise RuntimeError("solver failed")

    runtime, operand = _runtime_fixture(FakeQcu())
    with runtime.session() as session:
        with pytest.raises(RuntimeError, match="solver failed"):
            session("applyCloverQcu", operand, operand)
        assert session.poisoned
        with pytest.raises(RuntimeError, match="poisoned"):
            session("applyCloverQcu", operand, operand)

    assert events == [("init", 0), "operation", ("end", 0)]
    assert int(runtime.set_ptrs[0]) == 0


@pytest.mark.parametrize(
    "name",
    ("applyInitQcu", "applyEndQcu",
     "applyMultigridStrictInitQcu", "applyMultigridStrictEndQcu"),
)
def test_qcu_runtime_rejects_nested_lifecycle_operations(name: str) -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            events.append("init")
            set_ptrs[int(params[define._SET_INDEX_])] = 19

        def applyEndQcu(self, set_ptrs, params):
            events.append("end")
            set_ptrs[int(params[define._SET_INDEX_])] = 0

    runtime, operand = _runtime_fixture(FakeQcu())
    with pytest.raises(ValueError, match="manages the QCU lifecycle"):
        runtime.call(name)

    with runtime.session() as session:
        with pytest.raises(ValueError, match="manages the QCU lifecycle"):
            session(name)

    assert events == ["init", "end"]
    assert int(runtime.set_ptrs[0]) == 0


def test_qcu_runtime_session_poisons_when_device_changes() -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            set_ptrs[int(params[define._SET_INDEX_])] = 19

        def applyEndQcu(self, set_ptrs, params):
            events.append("end")
            set_ptrs[int(params[define._SET_INDEX_])] = 0

        def applyCloverQcu(self, *args):
            events.append("operation")

    runtime, operand = _runtime_fixture(FakeQcu())
    with runtime.session() as session:
        runtime.device = torch.device("meta")
        with pytest.raises(RuntimeError, match="device changed"):
            session("applyCloverQcu", operand, operand)
        assert session.poisoned

    assert events == ["end"]


@pytest.mark.parametrize(
    ("drift", "expected"),
    (("params", "params changed"),
     ("argv", "argv changed"),
     ("index", "_SET_INDEX_ changed"),
     ("pointer", "active slot pointer changed")),
)
def test_qcu_runtime_session_fails_closed_on_control_drift(
        drift: str, expected: str) -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            index = int(params[define._SET_INDEX_])
            events.append(("init", index))
            set_ptrs[index] = 19

        def applyEndQcu(self, set_ptrs, params):
            events.append(("end", int(params[define._SET_INDEX_])))
            set_ptrs[int(params[define._SET_INDEX_])] = 0

        def applyCloverQcu(self, *args):
            events.append("operation")

    runtime, operand = _runtime_fixture(FakeQcu())
    with runtime.session() as session:
        index = int(session.set_index)
        if drift == "params":
            runtime.params[define._SET_PLAN_] = 7
        elif drift == "argv":
            runtime.argv[define._MASS_] = 0.25
        elif drift == "index":
            runtime.params[define._SET_INDEX_] = 1
        else:
            runtime.set_ptrs[index] = 23
        with pytest.raises(RuntimeError, match=expected):
            session("applyCloverQcu", operand, operand)
        assert session.poisoned
        with pytest.raises(RuntimeError, match="poisoned"):
            session("applyCloverQcu", operand, operand)

    assert events == [("init", 0), ("end", 0)]
    assert int(runtime.set_ptrs[0]) == 0


def test_qcu_runtime_session_rejects_cross_thread_use_and_poisons() -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            set_ptrs[int(params[define._SET_INDEX_])] = 29

        def applyEndQcu(self, set_ptrs, params):
            events.append("end")
            set_ptrs[int(params[define._SET_INDEX_])] = 0

        def applyCloverQcu(self, *args):
            events.append("operation")

    runtime, operand = _runtime_fixture(FakeQcu())
    worker_errors = []
    with runtime.session() as session:
        def use_session() -> None:
            try:
                session("applyCloverQcu", operand, operand)
            except BaseException as exc:
                worker_errors.append(exc)

        worker = threading.Thread(target=use_session)
        worker.start()
        worker.join()

        assert len(worker_errors) == 1
        assert "owner thread" in str(worker_errors[0])
        assert session.poisoned
        with pytest.raises(RuntimeError, match="poisoned"):
            session("applyCloverQcu", operand, operand)

    assert events == ["end"]


def test_qcu_runtime_session_exit_restores_tampered_controls_and_pointer() -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            set_ptrs[int(params[define._SET_INDEX_])] = 31

        def applyEndQcu(self, set_ptrs, params):
            index = int(params[define._SET_INDEX_])
            events.append(("end", index, int(set_ptrs[index])))
            set_ptrs[index] = 0

    runtime, _ = _runtime_fixture(FakeQcu())
    original_plan = int(runtime.params[define._SET_PLAN_])
    original_mass = float(runtime.argv[define._MASS_])
    with runtime.session() as session:
        index = int(session.set_index)
        runtime.params[define._SET_INDEX_] = 1
        runtime.params[define._SET_PLAN_] = 9
        runtime.argv[define._MASS_] = 0.75
        runtime.set_ptrs[index] = 37

    assert events == [("end", 0, 31)]
    assert int(runtime.params[define._SET_INDEX_]) == 0
    assert int(runtime.params[define._SET_PLAN_]) == original_plan
    assert float(runtime.argv[define._MASS_]) == original_mass
    assert int(runtime.set_ptrs[0]) == 0


def test_qcu_runtime_releases_slot_when_initialization_fails() -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            index = int(params[define._SET_INDEX_])
            events.append(("init", index))
            set_ptrs[index] = 23
            raise RuntimeError("setup failed")

        def applyEndQcu(self, set_ptrs, params):
            index = int(params[define._SET_INDEX_])
            events.append(("end", index))
            set_ptrs[index] = 0

        def applyCloverQcu(self, *args):
            raise AssertionError("operation must not run")

    runtime, operand = _runtime_fixture(FakeQcu())
    with pytest.raises(RuntimeError, match="setup failed"):
        runtime.call("applyCloverQcu", operand, operand)

    assert events == [("init", 0), ("end", 0)]
    assert int(runtime.set_ptrs[0]) == 0


def test_qcu_runtime_session_cleans_failed_initialization() -> None:
    events = []

    class FakeQcu:
        fail_init = True
        fail_end_once = True

        def applyInitQcu(self, set_ptrs, params, argv):
            index = int(params[define._SET_INDEX_])
            events.append(("init", index))
            set_ptrs[index] = 23
            if self.fail_init:
                raise RuntimeError("session setup failed")

        def applyEndQcu(self, set_ptrs, params):
            index = int(params[define._SET_INDEX_])
            events.append(("end", index, int(set_ptrs[index])))
            if self.fail_end_once:
                self.fail_end_once = False
                raise RuntimeError("cleanup end failed")
            set_ptrs[index] = 0

        def applyCloverQcu(self, *args):
            events.append("operation")

    fake = FakeQcu()
    runtime, operand = _runtime_fixture(fake)
    with pytest.raises(RuntimeError, match="session setup failed"):
        with runtime.session():
            raise AssertionError("failed session must not become active")

    assert events == [("init", 0), ("end", 0, 23)]
    assert int(runtime.set_ptrs[0]) == 0

    fake.fail_init = False
    runtime.call("applyCloverQcu", operand, operand)
    assert events[2:] == [("init", 1), "operation", ("end", 1, 23)]
    assert int(runtime.set_ptrs[1]) == 0


def test_qcu_runtime_session_end_failure_clears_state_and_lock() -> None:
    class FakeQcu:
        @staticmethod
        def applyInitQcu(set_ptrs, params, argv):
            set_ptrs[int(params[define._SET_INDEX_])] = 29

        @staticmethod
        def applyEndQcu(set_ptrs, params):
            raise RuntimeError("session end failed")

    runtime, _ = _runtime_fixture(FakeQcu())
    session = runtime.session()
    with pytest.raises(RuntimeError, match="session end failed"):
        with session:
            pass

    assert not session.active
    assert session.set_index is None
    assert int(runtime.set_ptrs[0]) == 0
    assert runtime._operation_lock.acquire(blocking=False)
    runtime._operation_lock.release()


def test_qcu_runtime_rejects_operand_dtype_and_device_before_operation() -> None:
    events = []

    class FakeQcu:
        def applyInitQcu(self, set_ptrs, params, argv):
            set_ptrs[int(params[define._SET_INDEX_])] = 1

        def applyEndQcu(self, set_ptrs, params):
            events.append("end")
            set_ptrs[int(params[define._SET_INDEX_])] = 0

        def applyCloverQcu(self, *args):
            raise AssertionError("invalid operands reached native operation")

    runtime, operand = _runtime_fixture(FakeQcu())
    with pytest.raises(TypeError, match="complex64"):
        runtime.call("applyCloverQcu", torch.zeros(2), operand)

    runtime.device = torch.device("meta")
    with pytest.raises(ValueError, match="must be on meta"):
        runtime.call("applyCloverQcu", operand, operand)
    assert events == ["end", "end"]


def test_qcu_runtime_rejects_invalid_controls_and_reserved_slots() -> None:
    class NoopQcu:
        def applyInitQcu(self, *args):
            raise AssertionError("invalid control reached native initialization")

        def applyCloverQcu(self, *args):
            raise AssertionError("invalid slot reached native operation")

    params = torch.zeros(define._PARAMS_SIZE_, dtype=torch.int32)
    params[define._DATA_TYPE_] = define._LAT_C64_
    argv = torch.zeros(define._ARGV_SIZE_, dtype=torch.float32)
    set_ptrs = torch.zeros(define._SET_PTRS_SIZE_, dtype=torch.int64)
    with pytest.raises(TypeError, match="set_ptrs"):
        QcuRuntime(
            params=params, argv=argv,
            set_ptrs=set_ptrs.to(torch.int32), device=torch.device("cpu"),
            lattice=(4, 4, 4, 4), mass=0.1, csw=1.0, qcu_module=NoopQcu())
    with pytest.raises(ValueError, match="slot_limit"):
        QcuRuntime(
            params=params, argv=argv, set_ptrs=set_ptrs,
            device=torch.device("cpu"), lattice=(4, 4, 4, 4), mass=0.1,
            csw=1.0, qcu_module=NoopQcu(), slot_limit=31)

    runtime = QcuRuntime(
        params=params, argv=argv, set_ptrs=set_ptrs,
        device=torch.device("cpu"), lattice=(4, 4, 4, 4), mass=0.1,
        csw=1.0, qcu_module=NoopQcu(), slot_limit=2)
    runtime.slot_limit = 31
    with pytest.raises(ValueError, match="slot_limit"):
        runtime.call("applyCloverQcu", torch.zeros(1, dtype=torch.complex64))
