"""CPU-only tests for the strict MPI safety preflight layer."""

from __future__ import annotations

import builtins
import ctypes
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI
import pytest

from pyqcu.cuda._strict_mpi import (
    STRICT_MPI_CACHE_SCHEMA,
    STRICT_MPI_CACHE_SCHEMA_VERSION,
    StrictCacheShardMetadata,
    StrictMpiCapabilityError,
    StrictMpiPreflightError,
    collective_validate_strict_mpi,
    collective_validate_strict_runtime,
    expected_strict_cache_asset_shapes,
    make_strict_cache_shard_metadata,
    strict_mpi_world_communicator,
    strict_mpi_capabilities,
    validate_strict_cache_shard_assets,
    validate_strict_mpi_geometry,
)


WORLD = MPI.COMM_WORLD
FINE = (8, 8, 8, 8)
COARSE = (4, 4, 4, 4)
BLOCK = (2, 2, 2, 2)
PARAM_COUNT = 58
LAT_INDICES = (0, 1, 2, 3)
GRID_INDICES = (5, 6, 7, 8)
NODE_RANK_INDEX = 10
NODE_SIZE_INDEX = 11


@dataclass(frozen=True)
class _ShapeOnly:
    shape: tuple[int, ...]
    dtype: str = "complex64"


def _metadata_and_assets(geometry):
    metadata = make_strict_cache_shard_metadata(
        geometry,
        gauge_fingerprint="sha256:gauge-fixture",
        operator_fingerprint="sha256:clover-mass-boundary-fixture",
        boundary=("periodic", "periodic", "periodic", "anti-periodic"),
        dtype="complex64",
        target_parity=1,
        dofs=(12, 8),
    )
    expected = expected_strict_cache_asset_shapes(metadata)
    V = tuple(_ShapeOnly(item.V) for item in expected)
    Yhat = tuple(_ShapeOnly(item.Yhat) for item in expected)
    onsite = tuple(_ShapeOnly(item.onsite) for item in expected)
    return metadata, V, Yhat, onsite


@pytest.mark.skipif(WORLD.Get_size() != 1, reason="single-rank test")
def test_single_rank_valid_geometry_cache_key_assets_and_capability():
    geometry = validate_strict_mpi_geometry(
        comm_size=1,
        comm_rank=0,
        process_grid=(1, 1, 1, 1),
        global_shapes=(FINE, COARSE),
        block_sizes=(BLOCK,),
        level_process_grids=((1, 1, 1, 1), (1, 1, 1, 1)),
    )
    assert geometry.local_shapes == (FINE, COARSE)
    assert geometry.local_origins == ((0, 0, 0, 0),) * 2
    assert geometry.parity_origins == (0, 0)

    metadata, V, Yhat, onsite = _metadata_and_assets(geometry)
    payload = metadata.to_dict()
    required = {
        "schema", "schema_version", "cache_key", "gauge_fingerprint",
        "operator_fingerprint", "boundary", "dtype", "target_parity",
        "block_sizes", "dofs", "global_shapes", "local_shapes",
        "process_grid", "rank_coordinate",
    }
    assert set(payload) == required
    assert payload["schema"] == STRICT_MPI_CACHE_SCHEMA
    assert payload["schema_version"] == STRICT_MPI_CACHE_SCHEMA_VERSION
    restored = StrictCacheShardMetadata.from_mapping(payload)
    assert restored == metadata
    assert restored.cache_key == metadata.cache_key

    shapes = validate_strict_cache_shard_assets(
        restored, V=V, Yhat=Yhat, onsite=onsite)
    assert shapes[0].V == (8, 12, 4, 2, 4, 2, 4, 2, 4, 2)
    assert shapes[0].Yhat == (2, 4, 8, 8, 4, 4, 4, 4)
    assert shapes[0].onsite == (2, 8, 8, 4, 4, 4, 4)

    result = collective_validate_strict_mpi(
        MPI.COMM_SELF,
        process_grid=(1, 1, 1, 1),
        global_shapes=(FINE, COARSE),
        block_sizes=(BLOCK,),
        cache_metadata=payload,
        null_vectors=V,
        preconditioned_links=Yhat,
        onsite_pair=onsite,
        require_backend_ready=True,
    )
    assert result.backend_ready
    assert result.strict_coarse_halo is False
    capabilities = strict_mpi_capabilities()
    distributed = {
        "setup_halo", "full_halo", "compact_halo",
        "global_reduction", "fused_fgmres",
    }
    assert distributed.issubset(capabilities)
    assert capabilities["global_reduction"] is True
    incomplete = distributed - {"global_reduction"}
    assert all(capabilities[name] is False for name in incomplete)
    assert capabilities["strict_coarse_halo"] is False


@pytest.mark.skipif(WORLD.Get_size() != 1, reason="single-rank test")
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"comm_size": 2, "process_grid": (1, 1, 1, 1)}, "乘积"),
        ({
            "comm_size": 2,
            "process_grid": (2, 1, 1, 1),
            "global_shapes": ((7, 8, 8, 8),),
        }, "不能被"),
        ({"global_shapes": ((5, 8, 8, 8),)}, "正偶数"),
        ({
            "comm_size": 2,
            "process_grid": (2, 1, 1, 1),
            "global_shapes": ((24, 8, 8, 8), (3, 4, 4, 4)),
            "block_sizes": ((8, 2, 2, 2),),
        }, "aggregate 会跨 rank"),
        ({
            "global_shapes": (FINE, (2, 4, 4, 4)),
            "block_sizes": (BLOCK,),
        }, "global_shape 应为"),
        ({
            "comm_size": 2,
            "process_grid": (2, 1, 1, 1),
            "global_shapes": (FINE, COARSE),
            "block_sizes": (BLOCK,),
            "level_process_grids": (
                (2, 1, 1, 1), (1, 1, 1, 2)),
        }, "层间改变 process grid"),
        ({"local_parity_origins": (1,)}, "parity origin 不一致"),
    ],
)
def test_single_rank_invalid_geometry_is_rejected(kwargs, message):
    values = {
        "comm_size": 1,
        "comm_rank": 0,
        "process_grid": (1, 1, 1, 1),
        "global_shapes": (FINE,),
        "block_sizes": (),
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_strict_mpi_geometry(**values)


@pytest.mark.skipif(WORLD.Get_size() != 1, reason="single-rank test")
def test_cache_key_and_each_local_asset_shape_are_strictly_checked():
    geometry = validate_strict_mpi_geometry(
        comm_size=1,
        comm_rank=0,
        process_grid=(1, 1, 1, 1),
        global_shapes=(FINE, COARSE),
        block_sizes=(BLOCK,),
    )
    metadata, V, Yhat, onsite = _metadata_and_assets(geometry)

    stale = metadata.to_dict()
    stale["operator_fingerprint"] = "sha256:different-operator"
    with pytest.raises(ValueError, match="key .*不一致"):
        StrictCacheShardMetadata.from_mapping(stale)

    bad_groups = (
        ("null_vectors", (_ShapeOnly(V[0].shape[:-1] + (3,)),), Yhat, onsite),
        ("preconditioned_links", V,
         (_ShapeOnly(Yhat[0].shape[:-1] + (3,)),), onsite),
        ("onsite_pair", V, Yhat,
         (_ShapeOnly(onsite[0].shape[:-1] + (3,)),)),
    )
    for name, bad_V, bad_Yhat, bad_onsite in bad_groups:
        with pytest.raises(ValueError, match=rf"{name} shape 应为"):
            validate_strict_cache_shard_assets(
                metadata,
                null_vectors=bad_V,
                preconditioned_links=bad_Yhat,
                onsite_pair=bad_onsite,
            )

    wrong_dtype = (_ShapeOnly(V[0].shape, dtype="complex128"),)
    with pytest.raises(ValueError, match="dtype 应为 complex64"):
        validate_strict_cache_shard_assets(
            metadata,
            null_vectors=wrong_dtype,
            preconditioned_links=Yhat,
            onsite_pair=onsite,
        )


@pytest.mark.skipif(WORLD.Get_size() != 1, reason="single-rank test")
def test_three_level_asset_shapes_follow_each_local_transition():
    geometry = validate_strict_mpi_geometry(
        comm_size=1,
        comm_rank=0,
        process_grid=(1, 1, 1, 1),
        global_shapes=(FINE, COARSE, (2, 2, 2, 2)),
        block_sizes=(BLOCK, BLOCK),
    )
    metadata = make_strict_cache_shard_metadata(
        geometry,
        gauge_fingerprint="gauge-three-level",
        operator_fingerprint="operator-three-level",
        boundary="periodic",
        dtype="complex128",
        target_parity=0,
        dofs=(12, 8, 4),
    )
    expected = expected_strict_cache_asset_shapes(metadata)
    assert len(expected) == 2
    assert expected[1].V == (4, 8, 2, 2, 2, 2, 2, 2, 2, 2)
    assert expected[1].Yhat == (2, 4, 4, 4, 2, 2, 2, 2)
    validate_strict_cache_shard_assets(
        metadata,
        V=tuple(_ShapeOnly(item.V, "complex128") for item in expected),
        Yhat=tuple(_ShapeOnly(item.Yhat, "complex128") for item in expected),
        onsite=tuple(_ShapeOnly(item.onsite, "complex128") for item in expected),
    )


def _runtime_descriptor(process_grid, local_shape, *, node_size, node_rank):
    coarse = tuple(local_shape[axis] // BLOCK[axis] for axis in range(4))
    return {
        "process_grid": process_grid,
        "node_size": node_size,
        "node_rank": node_rank,
        "local_shapes": (local_shape, coarse),
        "block_sizes": (BLOCK,),
    }


@pytest.mark.skipif(WORLD.Get_size() != 1, reason="single-rank test")
def test_single_rank_without_mpi4py_uses_serial_preflight(monkeypatch):
    real_import = builtins.__import__

    def without_mpi4py(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise ModuleNotFoundError(
                "test fixture hides mpi4py", name="mpi4py")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_mpi4py)
    comm = strict_mpi_world_communicator()
    result = collective_validate_strict_runtime(
        comm,
        lambda: _runtime_descriptor(
            (1, 1, 1, 1), FINE, node_size=1, node_rank=0),
        require_backend_ready=True,
    )
    assert comm.Get_size() == 1
    assert result.backend_ready


class _SetupEntered(RuntimeError):
    pass


class _HierarchySetupProbe:
    def __init__(self, local_shape, setup_mode):
        self._strict_quda = True
        self.target_parity = 1
        self.fine_shape = tuple(local_shape)
        self._transition_count = 1
        self._block_sizes = (BLOCK,)
        self.strict_galerkin_mode = setup_mode
        self.setup_calls = 0

    def setup(self):
        self.setup_calls += 1
        raise _SetupEntered("hierarchy setup was entered")


def _constructor_params(local_shape, process_grid):
    values = [0] * PARAM_COUNT
    for index, extent in zip(LAT_INDICES, local_shape):
        values[index] = extent
    for index, extent in zip(GRID_INDICES, process_grid):
        values[index] = extent
    values[NODE_RANK_INDEX] = WORLD.Get_rank()
    values[NODE_SIZE_INDEX] = WORLD.Get_size()
    return values


@pytest.mark.skipif(WORLD.Get_size() != 1, reason="single-rank test")
def test_production_constructor_allows_serial_preflight_before_setup():
    from pyqcu.cuda._strict_multigrid import CudaStrictMultigridSolver

    hierarchy = _HierarchySetupProbe(FINE, "column")
    params = _constructor_params(FINE, (1, 1, 1, 1))
    with pytest.raises(_SetupEntered, match="setup was entered"):
        CudaStrictMultigridSolver(
            hierarchy, None, None, None, None, None, None, params)
    assert hierarchy.setup_calls == 1


@pytest.mark.skipif(WORLD.Get_size() != 2, reason="requires mpirun -np 2")
@pytest.mark.parametrize("setup_mode", ("column", "site-batch"))
def test_production_constructor_collectively_blocks_setup_modes(setup_mode):
    from pyqcu.cuda._strict_multigrid import CudaStrictMultigridSolver

    grid = (2, 1, 1, 1)
    local_shape = tuple(FINE[axis] // grid[axis] for axis in range(4))
    hierarchy = _HierarchySetupProbe(local_shape, setup_mode)
    params = _constructor_params(local_shape, grid)
    with pytest.raises(StrictMpiPreflightError) as caught:
        CudaStrictMultigridSolver(
            hierarchy, None, None, None, None, None, None, params)
    messages = WORLD.allgather(str(caught.value))
    setup_calls = WORLD.allgather(hierarchy.setup_calls)
    assert len(set(messages)) == 1
    assert setup_calls == [0, 0]
    assert "backend_ready=False" in messages[0]
    assert "setup_halo=False" in messages[0]


@pytest.mark.skipif(WORLD.Get_size() != 2, reason="requires mpirun -np 2")
@pytest.mark.parametrize(
    "process_grid",
    ((2, 1, 1, 1), (1, 1, 1, 2)),
    ids=("x-topology", "t-topology"),
)
def test_two_rank_x_t_topologies_report_identical_collective_error(process_grid):
    # 先证明该 x/t 分区本身合法；仅 global reduction 就绪仍不足以求解。
    valid = collective_validate_strict_mpi(
        WORLD,
        process_grid=process_grid,
        global_shapes=(FINE, COARSE),
        block_sizes=(BLOCK,),
    )
    assert valid.geometry.local_shapes[0] in (
        (4, 8, 8, 8), (8, 8, 8, 4))
    assert not valid.backend_ready
    with pytest.raises(StrictMpiCapabilityError, match="backend_ready=False"):
        valid.require_backend_ready()

    with pytest.raises(StrictMpiPreflightError) as capability_gate:
        collective_validate_strict_mpi(
            WORLD,
            process_grid=process_grid,
            global_shapes=(FINE, COARSE),
            block_sizes=(BLOCK,),
            require_backend_ready=True,
        )
    capability_messages = WORLD.allgather(str(capability_gate.value))
    assert len(set(capability_messages)) == 1
    assert tuple(rank for rank, _, _ in capability_gate.value.failures) == (0, 1)

    # 只让 rank 0 带入错误 parity convention。collective preflight 必须让
    # rank 1 也在同一 allgather 后抛出完全相同的错误，不能进入后端。
    local_parity = (1, 0) if WORLD.Get_rank() == 0 else (0, 0)
    with pytest.raises(StrictMpiPreflightError) as caught:
        collective_validate_strict_mpi(
            WORLD,
            process_grid=process_grid,
            global_shapes=(FINE, COARSE),
            block_sizes=(BLOCK,),
            local_parity_origins=local_parity,
        )
    message = str(caught.value)
    gathered = WORLD.allgather(message)
    assert len(set(gathered)) == 1
    assert "rank 0" in message
    assert "global/local parity origin 不一致" in message
    assert tuple(rank for rank, _, _ in caught.value.failures) == (0,)


@pytest.mark.skipif(WORLD.Get_size() != 2, reason="requires mpirun -np 2")
def test_two_rank_collective_rejects_individually_valid_cache_mismatch():
    grid = (2, 1, 1, 1)

    def metadata_factory(geometry):
        return make_strict_cache_shard_metadata(
            geometry,
            gauge_fingerprint=f"gauge-rank-{WORLD.Get_rank()}",
            operator_fingerprint="same-operator",
            boundary="periodic",
            dtype="complex64",
            target_parity=1,
            dofs=(12, 8),
        )

    with pytest.raises(StrictMpiPreflightError, match="CollectiveMismatch") as caught:
        collective_validate_strict_mpi(
            WORLD,
            process_grid=grid,
            global_shapes=(FINE, COARSE),
            block_sizes=(BLOCK,),
            cache_metadata=metadata_factory,
        )
    messages = WORLD.allgather(str(caught.value))
    assert len(set(messages)) == 1


def _load_strict_backend_library():
    library_path = (
        Path(__file__).resolve().parents[3] / "cpp" / "cuda" / "qcu" /
        "libqcu.so"
    )
    if not library_path.is_file():
        pytest.skip(f"strict backend library 不存在：{library_path}")
    try:
        library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        pytest.skip(f"strict backend library 无法加载：{exc}")

    ll = ctypes.c_longlong
    integer = ctypes.c_int
    ull = ctypes.c_ulonglong
    real = ctypes.c_double
    integer_ptr = ctypes.POINTER(integer)
    ull_ptr = ctypes.POINTER(ull)
    real_ptr = ctypes.POINTER(real)
    signatures = {
        "testMultigridStrictGlobalReductionQcu": (
            [integer] + [real] * 4 + [real_ptr] * 3 +
            [integer_ptr] * 2),
        "applyMultigridStrictVCycleQcu": (
            [ll] * 4 + [integer, ull_ptr]),
        "applyMultigridStrictInitQcu": (
            [ll] * 2 + [integer, ull_ptr]),
        "applyMultigridStrictEndQcu": [ll] * 2,
        "applyMultigridStrictFgmresQcu": (
            [ll] * 10 + [integer] * 13 + [real] + [integer] * 2 + [ull] +
            [integer_ptr, integer_ptr, real_ptr, ull_ptr]),
        "applyMultigridStrictCoarseQcu": [ll] * 6 + [integer] * 6,
        "applyMultigridStrictMatPCQcu": [ll] * 6 + [integer] * 6,
        "applyMultigridStrictPrepareQcu": [ll] * 7 + [integer] * 6,
        "applyMultigridStrictReconstructQcu": [ll] * 8 + [integer] * 6,
        "applyMultigridStrictRestrictQcu": [ll] * 5 + [integer] * 11,
        "applyMultigridStrictProLongQcu": [ll] * 5 + [integer] * 11,
    }
    for name, argtypes in signatures.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = integer
    return library


def _call_global_reduction_probe(
        library, data_type, local_dot, local_norm2, threshold):
    global_real = ctypes.c_double()
    global_imag = ctypes.c_double()
    global_norm = ctypes.c_double()
    converged = ctypes.c_int()
    collective_calls = ctypes.c_int()
    status = library.testMultigridStrictGlobalReductionQcu(
        data_type,
        local_dot.real,
        local_dot.imag,
        local_norm2,
        threshold,
        ctypes.byref(global_real),
        ctypes.byref(global_imag),
        ctypes.byref(global_norm),
        ctypes.byref(converged),
        ctypes.byref(collective_calls),
    )
    return (
        status,
        global_real.value,
        global_imag.value,
        global_norm.value,
        converged.value,
        collective_calls.value,
    )


@pytest.mark.skipif(
    WORLD.Get_size() not in (1, 2), reason="requires one or two MPI ranks")
@pytest.mark.parametrize(
    ("data_type", "absolute_tolerance"),
    ((2, 1.0e-6), (3, 1.0e-12)),
    ids=("complex64", "complex128"),
)
@pytest.mark.parametrize(
    ("threshold_offset", "expected_converged"),
    ((0.0, 1), (-0.25, 0)),
    ids=("at-threshold", "above-threshold"),
)
def test_strict_global_reduction_numeric_gate(
        data_type, absolute_tolerance, threshold_offset,
        expected_converged):
    library = _load_strict_backend_library()
    local_dots = (complex(1.25, -0.5), complex(-0.25, 2.0))
    local_norm2 = (4.0, 5.0)
    active = WORLD.Get_size()
    rank = WORLD.Get_rank()
    expected_dot = sum(local_dots[:active])
    expected_norm = sum(local_norm2[:active]) ** 0.5
    result = _call_global_reduction_probe(
        library,
        data_type,
        local_dots[rank],
        local_norm2[rank],
        expected_norm + threshold_offset,
    )

    status, global_real, global_imag, global_norm, converged, calls = result
    assert status == 0
    assert global_real == pytest.approx(
        expected_dot.real, abs=absolute_tolerance)
    assert global_imag == pytest.approx(
        expected_dot.imag, abs=absolute_tolerance)
    assert global_norm == pytest.approx(
        expected_norm, abs=absolute_tolerance)
    assert converged == expected_converged
    # Multi-rank calls include a symmetric validity/type preflight and a
    # threshold-consistency Allgather before the two numerical reductions.
    assert calls == (0 if active == 1 else 4)

    # 每个 rank 必须从同一全局 Arnoldi/真残差标量导出同一停机决定。
    gathered = WORLD.allgather(result)
    assert len(set(gathered)) == 1


@pytest.mark.skipif(
    WORLD.Get_size() not in (1, 2), reason="requires one or two MPI ranks")
def test_strict_global_reduction_invalid_dtype_fails_identically(capfd):
    library = _load_strict_backend_library()
    result = _call_global_reduction_probe(
        library, 999, complex(1.0, -2.0), 1.0, 1.0)
    error = capfd.readouterr().err
    assert result[0] == 1
    assert result[-1] == (0 if WORLD.Get_size() == 1 else 1)
    assert "supports complex64/complex128" in error
    gathered = WORLD.allgather((result[0], result[-1], error))
    assert len(set(gathered)) == 1


@pytest.mark.skipif(
    WORLD.Get_size() != 2, reason="requires exactly two MPI ranks")
@pytest.mark.parametrize(
    ("case", "expected_calls", "message"),
    (
        ("rank-local-invalid", 1, "input is invalid on at least one rank"),
        ("dtype-mismatch", 1, "data_type differs between MPI ranks"),
        ("threshold-mismatch", 2, "threshold differs between MPI ranks"),
    ),
)
def test_strict_global_reduction_rank_local_error_is_collective(
        case, expected_calls, message, capfd):
    library = _load_strict_backend_library()
    rank = WORLD.Get_rank()
    data_type = 2
    local_norm2 = 1.0
    threshold = 2.0
    if case == "rank-local-invalid" and rank == 0:
        local_norm2 = -1.0
    elif case == "dtype-mismatch" and rank == 1:
        data_type = 3
    elif case == "threshold-mismatch" and rank == 1:
        threshold = 1.0

    result = _call_global_reduction_probe(
        library, data_type, complex(1.0, -2.0), local_norm2, threshold)
    error = capfd.readouterr().err
    assert result[0] == 1
    assert result[-1] == expected_calls
    assert message in error
    gathered = WORLD.allgather((result[0], result[-1], error))
    assert len(set(gathered)) == 1


def _abi_params(process_grid):
    values = (ctypes.c_int * PARAM_COUNT)()
    for index, extent in zip(GRID_INDICES, process_grid):
        values[index] = extent
    values[NODE_RANK_INDEX] = WORLD.Get_rank()
    values[NODE_SIZE_INDEX] = WORLD.Get_size()
    return values


def _call_all_strict_abis(library, params):
    pointer = ctypes.addressof(params)
    zero = 0
    statuses = {
        "vcycle": library.applyMultigridStrictVCycleQcu(
            zero, zero, zero, pointer, 1, None),
        "hierarchy": library.applyMultigridStrictInitQcu(
            zero, pointer, 1, None),
        "end": library.applyMultigridStrictEndQcu(zero, pointer),
        "fgmres": library.applyMultigridStrictFgmresQcu(
            *([zero] * 9), pointer,
            12, 4, 4, 4, 4, 4, 2, 2, 2, 2, 8, 1, 1,
            1.0e-6, 0, 0, 0, None, None, None, None),
        "coarse": library.applyMultigridStrictCoarseQcu(
            *([zero] * 5), pointer, 4, 2, 2, 2, 2, -1),
        "matpc": library.applyMultigridStrictMatPCQcu(
            *([zero] * 5), pointer, 4, 2, 2, 2, 2, 1),
        "prepare": library.applyMultigridStrictPrepareQcu(
            *([zero] * 6), pointer, 4, 2, 2, 2, 2, 1),
        "reconstruct": library.applyMultigridStrictReconstructQcu(
            *([zero] * 7), pointer, 4, 2, 2, 2, 2, 1),
        "restrict": library.applyMultigridStrictRestrictQcu(
            *([zero] * 4), pointer,
            4, 12, 4, 4, 4, 4, 2, 2, 2, 2, 1),
        "prolong": library.applyMultigridStrictProLongQcu(
            *([zero] * 4), pointer,
            4, 12, 4, 4, 4, 4, 2, 2, 2, 2, 1),
    }
    return statuses


@pytest.mark.skipif(WORLD.Get_size() != 1, reason="single-rank test")
def test_strict_c_abi_gate_accepts_world_one_and_checks_params_first(capfd):
    library = _load_strict_backend_library()
    params = _abi_params((1, 1, 1, 1))
    status = library.applyMultigridStrictInitQcu(
        0, ctypes.addressof(params), 1, None)
    serial_error = capfd.readouterr().err
    assert status == 1
    assert "strict hierarchy byte output is null" in serial_error
    assert "strict MPI fail-closed" not in serial_error

    params[NODE_SIZE_INDEX] = 2
    status = library.applyMultigridStrictInitQcu(
        0, ctypes.addressof(params), 1, None)
    params_error = capfd.readouterr().err
    assert status == 1
    assert "strict MPI fail-closed" in params_error


@pytest.mark.skipif(WORLD.Get_size() != 2, reason="requires mpirun -np 2")
def test_all_strict_c_abis_fail_before_null_algorithm_inputs(capfd):
    library = _load_strict_backend_library()
    params = _abi_params((2, 1, 1, 1))
    statuses = _call_all_strict_abis(library, params)
    errors = capfd.readouterr().err

    assert set(statuses) == {
        "vcycle", "hierarchy", "end", "fgmres", "coarse", "matpc",
        "prepare", "reconstruct", "restrict", "prolong",
    }
    assert all(status == 1 for status in statuses.values())
    gate_count = errors.count("strict MPI fail-closed")
    assert gate_count == len(statuses)
    summaries = WORLD.allgather((tuple(statuses.items()), errors))
    assert len(set(summaries)) == 1
