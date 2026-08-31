"""Strict MultiGrid 的 MPI 安全前置层。

本模块只验证分布式几何、cache shard 元数据和运行期资产形状。它不包含
coarse halo，也不会把通过 preflight 解释为多 rank strict 后端已经可用。
调用方在进入任何可能包含 halo/collective 的后端之前，应使用
``collective_validate_strict_mpi``，使所有 rank 先完成同一次错误汇聚。

该模块有意只依赖 Python 标准库；传入的 communicator 仅需提供 mpi4py
风格的 ``Get_size``、``Get_rank`` 和 ``allgather`` 方法。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import prod
from numbers import Integral
from typing import Any, NoReturn, TypeVar


Shape4 = tuple[int, int, int, int]
STRICT_MPI_CACHE_SCHEMA = "pyqcu.strict-mpi-cache-shard"
STRICT_MPI_CACHE_SCHEMA_VERSION = 1


class StrictMpiPreflightError(RuntimeError):
    """所有 rank 都会看到相同文本的 collective preflight 错误。"""

    def __init__(self, failures: Sequence[tuple[int, str, str]]):
        self.failures = tuple(
            (int(rank), str(kind), str(message))
            for rank, kind, message in failures
        )
        details = "\n".join(
            f"  rank {rank}: {kind}: {message}"
            for rank, kind, message in self.failures
        )
        super().__init__(
            "strict MPI collective preflight failed on "
            f"{len(self.failures)} rank(s):\n{details}"
        )


class StrictMpiCapabilityError(NotImplementedError):
    """请求了本安全层明确未实现的 strict MPI capability。"""


class _SerialCommunicator:
    """mpi4py 不可用时供真实单进程使用的最小 communicator。"""

    @staticmethod
    def Get_size() -> int:
        return 1

    @staticmethod
    def Get_rank() -> int:
        return 0

    @staticmethod
    def allgather(value: Any) -> list[Any]:
        return [value]


_SERIAL_COMMUNICATOR = _SerialCommunicator()


def strict_mpi_world_communicator() -> Any:
    """返回 ``MPI.COMM_WORLD``；未安装 mpi4py 时退化为单进程通信器。

    该退化只表达“当前 Python 进程没有 MPI Python 绑定”。production
    preflight 仍会核对 ``params`` 的 node/grid 协议；C++ strict ABI 还会
    独立查询真实 ``MPI_COMM_WORLD``，因此该便利入口不会放宽后端闸门。
    """

    try:
        from mpi4py import MPI
    except ModuleNotFoundError as exc:
        if exc.name not in ("mpi4py", "mpi4py.MPI"):
            raise
        return _SERIAL_COMMUNICATOR
    return MPI.COMM_WORLD


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} 必须是正整数，得到 {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} 必须是正整数，得到 {result}")
    return result


def _shape4(value: Sequence[int], name: str) -> Shape4:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} 必须包含四个正整数")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} 必须包含四个正整数") from exc
    if len(items) != 4:
        raise ValueError(f"{name} 必须包含四个正整数，得到 {items!r}")
    return tuple(
        _positive_int(item, f"{name}[{axis}]")
        for axis, item in enumerate(items)
    )  # type: ignore[return-value]


def _coordinate4(value: Sequence[int], name: str) -> Shape4:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} 必须包含四个非负整数")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} 必须包含四个非负整数") from exc
    if len(items) != 4:
        raise ValueError(f"{name} 必须包含四个非负整数，得到 {items!r}")
    result: list[int] = []
    for axis, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"{name}[{axis}] 必须是非负整数，得到 {item!r}")
        coordinate = int(item)
        if coordinate < 0:
            raise ValueError(
                f"{name}[{axis}] 必须是非负整数，得到 {coordinate}")
        result.append(coordinate)
    return tuple(result)  # type: ignore[return-value]


def _shape_levels(value: Sequence[Sequence[int]], name: str) -> tuple[Shape4, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} 必须是 4D shape 序列")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} 必须是 4D shape 序列") from exc
    if not items:
        raise ValueError(f"{name} 至少需要一层")
    # 单层调用允许直接传 ``(X,Y,Z,T)``。
    if len(items) == 4 and all(isinstance(item, Integral) for item in items):
        return (_shape4(items, f"{name}[0]"),)
    return tuple(_shape4(item, f"{name}[{level}]")
                 for level, item in enumerate(items))


def _block_levels(value: Sequence[Sequence[int]], count: int) -> tuple[Shape4, ...]:
    if count == 0:
        if tuple(value):
            raise ValueError("单层 hierarchy 的 block_sizes 必须为空")
        return ()
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError("block_sizes 必须是 4D block 序列") from exc
    if count == 1 and len(items) == 4 and all(
            isinstance(item, Integral) for item in items):
        items = (items,)
    if len(items) != count:
        raise ValueError(
            f"block_sizes 应有 {count} 项，得到 {len(items)} 项")
    return tuple(_shape4(item, f"block_sizes[{level}]")
                 for level, item in enumerate(items))


def _rank_coordinate(rank: int, grid: Shape4) -> Shape4:
    remainder = int(rank)
    coordinate: list[int] = []
    for axis in range(4):
        stride = prod(grid[axis + 1:])
        coordinate.append(remainder // stride)
        remainder %= stride
    return tuple(coordinate)  # type: ignore[return-value]


def _linear_rank(coordinate: Shape4, grid: Shape4) -> int:
    rank = 0
    for axis in range(4):
        if coordinate[axis] < 0 or coordinate[axis] >= grid[axis]:
            raise ValueError(
                f"rank_coordinate={coordinate} 超出 process_grid={grid}")
        rank = rank * grid[axis] + coordinate[axis]
    return rank


@dataclass(frozen=True)
class StrictMpiLevelGeometry:
    level: int
    process_grid: Shape4
    rank_coordinate: Shape4
    global_shape: Shape4
    local_shape: Shape4
    local_origin: Shape4
    global_parity_origin: int
    local_parity_origin: int


@dataclass(frozen=True)
class StrictMpiGeometry:
    """通过校验的、所有层使用同一 process grid 的 rank-local 几何。"""

    comm_size: int
    comm_rank: int
    process_grid: Shape4
    rank_coordinate: Shape4
    block_sizes: tuple[Shape4, ...]
    levels: tuple[StrictMpiLevelGeometry, ...]

    @property
    def global_shapes(self) -> tuple[Shape4, ...]:
        return tuple(level.global_shape for level in self.levels)

    @property
    def local_shapes(self) -> tuple[Shape4, ...]:
        return tuple(level.local_shape for level in self.levels)

    @property
    def local_origins(self) -> tuple[Shape4, ...]:
        return tuple(level.local_origin for level in self.levels)

    @property
    def parity_origins(self) -> tuple[int, ...]:
        return tuple(level.global_parity_origin for level in self.levels)


def validate_strict_mpi_geometry(
    *,
    comm_size: int,
    comm_rank: int,
    process_grid: Sequence[int],
    global_shapes: Sequence[Sequence[int]],
    block_sizes: Sequence[Sequence[int]],
    level_process_grids: Sequence[Sequence[int]] | None = None,
    rank_coordinate: Sequence[int] | None = None,
    local_parity_origins: Sequence[int] | None = None,
) -> StrictMpiGeometry:
    """验证 strict full-coarse hierarchy 的 MPI 分区不变量。

    当前 checkerboard kernel 把每个 rank 的 local 原点视为偶点。因此除了
    每层 local 四维都必须为正偶数外，本函数还显式核对 local 原点在全局
    坐标中的 parity。aggregate 必须完整留在一个 rank 内；首版不允许层间
    改变 process grid 或做 rank agglomeration。
    """

    size = _positive_int(comm_size, "comm_size")
    if isinstance(comm_rank, bool) or not isinstance(comm_rank, Integral):
        raise TypeError("comm_rank 必须是整数")
    rank = int(comm_rank)
    if rank < 0 or rank >= size:
        raise ValueError(f"comm_rank={rank} 超出 communicator size={size}")

    globals_ = _shape_levels(global_shapes, "global_shapes")
    blocks = _block_levels(block_sizes, len(globals_) - 1)
    grid = _shape4(process_grid, "process_grid")
    if prod(grid) != size:
        raise ValueError(
            f"process_grid={grid} 的乘积 {prod(grid)} != comm size {size}")

    if level_process_grids is None:
        grids = (grid,) * len(globals_)
    else:
        try:
            grid_items = tuple(level_process_grids)
        except TypeError as exc:
            raise TypeError("level_process_grids 必须是 4D grid 序列") from exc
        if len(grid_items) != len(globals_):
            raise ValueError(
                "level_process_grids 必须与 global_shapes 层数一致："
                f"{len(grid_items)} != {len(globals_)}")
        grids = tuple(_shape4(item, f"level_process_grids[{level}]")
                      for level, item in enumerate(grid_items))
        for level, level_grid in enumerate(grids):
            if level_grid != grid:
                raise ValueError(
                    "strict MPI 暂不支持层间改变 process grid："
                    f"level 0={grid}, level {level}={level_grid}")

    derived_coordinate = _rank_coordinate(rank, grid)
    if rank_coordinate is None:
        coordinate = derived_coordinate
    else:
        coordinate = _coordinate4(rank_coordinate, "rank_coordinate")
        if any(coordinate[axis] >= grid[axis] for axis in range(4)):
            raise ValueError(
                f"rank_coordinate={coordinate} 超出 process_grid={grid}")
        if coordinate != derived_coordinate:
            raise ValueError(
                f"rank {rank} 的 row-major coordinate 应为 {derived_coordinate}，"
                f"得到 {coordinate}")

    if local_parity_origins is None:
        parity_origins = (0,) * len(globals_)
    else:
        try:
            parity_origins = tuple(int(value) for value in local_parity_origins)
        except (TypeError, ValueError) as exc:
            raise TypeError("local_parity_origins 必须是 0/1 序列") from exc
        if len(parity_origins) != len(globals_):
            raise ValueError(
                "local_parity_origins 必须与层数一致："
                f"{len(parity_origins)} != {len(globals_)}")
        if any(value not in (0, 1) for value in parity_origins):
            raise ValueError("local_parity_origins 只能包含 0 或 1")

    levels: list[StrictMpiLevelGeometry] = []
    for level, global_shape in enumerate(globals_):
        level_grid = grids[level]
        if any(global_shape[axis] % level_grid[axis] for axis in range(4)):
            raise ValueError(
                f"level {level} global_shape={global_shape} 不能被 "
                f"process_grid={level_grid} 整除")
        local_shape = tuple(
            global_shape[axis] // level_grid[axis] for axis in range(4)
        )
        if any(extent <= 0 or extent % 2 for extent in local_shape):
            raise ValueError(
                f"level {level} local_shape={local_shape} 必须四维均为正偶数")
        origin = tuple(
            coordinate[axis] * local_shape[axis] for axis in range(4)
        )
        global_parity = sum(origin) & 1
        if parity_origins[level] != global_parity:
            raise ValueError(
                f"level {level} global/local parity origin 不一致："
                f"global origin={origin} 的 parity={global_parity}，"
                f"local convention={parity_origins[level]}")

        levels.append(StrictMpiLevelGeometry(
            level=level,
            process_grid=level_grid,
            rank_coordinate=coordinate,
            global_shape=global_shape,
            local_shape=local_shape,  # type: ignore[arg-type]
            local_origin=origin,  # type: ignore[arg-type]
            global_parity_origin=global_parity,
            local_parity_origin=parity_origins[level],
        ))

        if level == len(globals_) - 1:
            continue
        block = blocks[level]
        if any(global_shape[axis] % block[axis] for axis in range(4)):
            raise ValueError(
                f"level {level} global_shape={global_shape} 不能被 "
                f"block_size={block} 整除")
        expected_next = tuple(
            global_shape[axis] // block[axis] for axis in range(4)
        )
        if globals_[level + 1] != expected_next:
            raise ValueError(
                f"level {level + 1} global_shape 应为 {expected_next}，"
                f"得到 {globals_[level + 1]}")
        if any(local_shape[axis] % block[axis] for axis in range(4)):
            raise ValueError(
                f"level {level} local_shape={local_shape} 不能被 "
                f"block_size={block} 整除；aggregate 会跨 rank 边界")
        if any(origin[axis] % block[axis] for axis in range(4)):
            raise ValueError(
                f"level {level} local_origin={origin} 未与 block_size={block} "
                "对齐；aggregate 会跨 rank 边界")

    return StrictMpiGeometry(
        comm_size=size,
        comm_rank=rank,
        process_grid=grid,
        rank_coordinate=coordinate,
        block_sizes=blocks,
        levels=tuple(levels),
    )


def _normalise_text(value: Any, name: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} 不能为空")
    return result


def _normalise_boundary(
    value: Sequence[str] | str | bytes,
) -> tuple[str, str, str, str]:
    values = (value,) * 4 if isinstance(value, (str, bytes)) else tuple(value)
    if len(values) != 4:
        raise ValueError("boundary 必须包含 x/y/z/t 四项")
    return tuple(
        _normalise_text(item, f"boundary[{axis}]").lower()
        for axis, item in enumerate(values)
    )  # type: ignore[return-value]


def _normalise_dtype(value: Any) -> str:
    name = _normalise_text(value, "dtype").lower()
    for prefix in ("torch.", "numpy.", "np."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    aliases = {
        "c64": "complex64",
        "complex64": "complex64",
        "c128": "complex128",
        "complex128": "complex128",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(
            f"strict cache dtype 仅支持 complex64/complex128，得到 {value!r}") from exc


@dataclass(frozen=True)
class StrictCacheShardMetadata:
    """一个 rank 的 strict hierarchy cache shard 身份与布局。"""

    gauge_fingerprint: str
    operator_fingerprint: str
    boundary: tuple[str, str, str, str]
    dtype: str
    target_parity: int
    block_sizes: tuple[Shape4, ...]
    dofs: tuple[int, ...]
    global_shapes: tuple[Shape4, ...]
    local_shapes: tuple[Shape4, ...]
    process_grid: Shape4
    rank_coordinate: Shape4
    schema: str = STRICT_MPI_CACHE_SCHEMA
    schema_version: int = STRICT_MPI_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema = _normalise_text(self.schema, "schema")
        version = _positive_int(self.schema_version, "schema_version")
        if schema != STRICT_MPI_CACHE_SCHEMA:
            raise ValueError(
                f"cache schema 应为 {STRICT_MPI_CACHE_SCHEMA!r}，得到 {schema!r}")
        if version != STRICT_MPI_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"cache schema version 应为 {STRICT_MPI_CACHE_SCHEMA_VERSION}，"
                f"得到 {version}")

        global_shapes = _shape_levels(self.global_shapes, "global_shapes")
        local_shapes = _shape_levels(self.local_shapes, "local_shapes")
        if len(local_shapes) != len(global_shapes):
            raise ValueError("cache local_shapes/global_shapes 层数不一致")
        blocks = _block_levels(self.block_sizes, len(global_shapes) - 1)
        grid = _shape4(self.process_grid, "process_grid")
        coordinate = _coordinate4(self.rank_coordinate, "rank_coordinate")
        rank = _linear_rank(coordinate, grid)
        geometry = validate_strict_mpi_geometry(
            comm_size=prod(grid),
            comm_rank=rank,
            process_grid=grid,
            global_shapes=global_shapes,
            block_sizes=blocks,
            rank_coordinate=coordinate,
        )
        if geometry.local_shapes != local_shapes:
            raise ValueError(
                f"cache local_shapes={local_shapes} 与几何推导值 "
                f"{geometry.local_shapes} 不一致")

        try:
            dofs = tuple(_positive_int(value, f"dofs[{level}]")
                         for level, value in enumerate(self.dofs))
        except TypeError as exc:
            raise TypeError("dofs 必须是每层自由度序列") from exc
        if len(dofs) != len(global_shapes):
            raise ValueError(
                f"dofs 应有 {len(global_shapes)} 项，得到 {len(dofs)} 项")
        if any(dof % 2 for dof in dofs[1:]):
            raise ValueError("strict coarse dof 必须为 2*nvec，因而必须是偶数")

        parity = int(self.target_parity)
        if parity not in (0, 1):
            raise ValueError("target_parity 必须为 0 或 1")

        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "gauge_fingerprint", _normalise_text(
            self.gauge_fingerprint, "gauge_fingerprint"))
        object.__setattr__(self, "operator_fingerprint", _normalise_text(
            self.operator_fingerprint, "operator_fingerprint"))
        object.__setattr__(self, "boundary", _normalise_boundary(self.boundary))
        object.__setattr__(self, "dtype", _normalise_dtype(self.dtype))
        object.__setattr__(self, "target_parity", parity)
        object.__setattr__(self, "block_sizes", blocks)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "global_shapes", global_shapes)
        object.__setattr__(self, "local_shapes", local_shapes)
        object.__setattr__(self, "process_grid", grid)
        object.__setattr__(self, "rank_coordinate", coordinate)

    def _key_payload(self) -> dict[str, Any]:
        # 每个字段都进入 canonical JSON；rank coordinate/local shape 因而会
        # 产生真正的 shard key，而不是只标识全局 hierarchy 的 key。
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "gauge_fingerprint": self.gauge_fingerprint,
            "operator_fingerprint": self.operator_fingerprint,
            "boundary": list(self.boundary),
            "dtype": self.dtype,
            "target_parity": self.target_parity,
            "block_sizes": [list(shape) for shape in self.block_sizes],
            "dofs": list(self.dofs),
            "global_shapes": [list(shape) for shape in self.global_shapes],
            "local_shapes": [list(shape) for shape in self.local_shapes],
            "process_grid": list(self.process_grid),
            "rank_coordinate": list(self.rank_coordinate),
        }

    @property
    def cache_key(self) -> str:
        encoded = dumps(
            self._key_payload(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return f"strict-mpi-v{self.schema_version}-{sha256(encoded).hexdigest()}"

    @property
    def collective_key(self) -> str:
        payload = self._key_payload()
        payload.pop("rank_coordinate")
        encoded = dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result = self._key_payload()
        result["cache_key"] = self.cache_key
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StrictCacheShardMetadata":
        required = {
            "schema", "schema_version", "cache_key", "gauge_fingerprint",
            "operator_fingerprint", "boundary", "dtype", "target_parity",
            "block_sizes", "dofs", "global_shapes", "local_shapes",
            "process_grid", "rank_coordinate",
        }
        keys = set(value)
        missing = sorted(required - keys)
        unknown = sorted(keys - required)
        if missing:
            raise ValueError(f"strict cache metadata 缺少字段：{missing}")
        if unknown:
            raise ValueError(f"strict cache metadata 含未知字段：{unknown}")
        metadata = cls(
            schema=value["schema"],
            schema_version=value["schema_version"],
            gauge_fingerprint=value["gauge_fingerprint"],
            operator_fingerprint=value["operator_fingerprint"],
            boundary=value["boundary"],
            dtype=value["dtype"],
            target_parity=value["target_parity"],
            block_sizes=value["block_sizes"],
            dofs=value["dofs"],
            global_shapes=value["global_shapes"],
            local_shapes=value["local_shapes"],
            process_grid=value["process_grid"],
            rank_coordinate=value["rank_coordinate"],
        )
        supplied_key = _normalise_text(value["cache_key"], "cache_key")
        if supplied_key != metadata.cache_key:
            raise ValueError(
                "strict cache key 与 metadata 内容不一致；cache 可能陈旧或损坏")
        return metadata


def make_strict_cache_shard_metadata(
    geometry: StrictMpiGeometry,
    *,
    gauge_fingerprint: str,
    operator_fingerprint: str,
    boundary: Sequence[str] | str | bytes,
    dtype: Any,
    target_parity: int,
    dofs: Sequence[int],
) -> StrictCacheShardMetadata:
    """由已验证几何构造包含 deterministic shard key 的 metadata。"""

    return StrictCacheShardMetadata(
        gauge_fingerprint=gauge_fingerprint,
        operator_fingerprint=operator_fingerprint,
        boundary=_normalise_boundary(boundary),
        dtype=_normalise_dtype(dtype),
        target_parity=target_parity,
        block_sizes=geometry.block_sizes,
        dofs=tuple(dofs),
        global_shapes=geometry.global_shapes,
        local_shapes=geometry.local_shapes,
        process_grid=geometry.process_grid,
        rank_coordinate=geometry.rank_coordinate,
    )


def validate_strict_cache_shard_metadata(
    value: StrictCacheShardMetadata | Mapping[str, Any],
    *,
    geometry: StrictMpiGeometry | None = None,
) -> StrictCacheShardMetadata:
    metadata = (value if isinstance(value, StrictCacheShardMetadata)
                else StrictCacheShardMetadata.from_mapping(value))
    if geometry is not None:
        comparisons = {
            "process_grid": (metadata.process_grid, geometry.process_grid),
            "rank_coordinate": (
                metadata.rank_coordinate, geometry.rank_coordinate),
            "global_shapes": (metadata.global_shapes, geometry.global_shapes),
            "local_shapes": (metadata.local_shapes, geometry.local_shapes),
            "block_sizes": (metadata.block_sizes, geometry.block_sizes),
        }
        for name, (cached, current) in comparisons.items():
            if cached != current:
                raise ValueError(
                    f"strict cache {name}={cached} 与当前几何 {current} 不一致")
    return metadata


@dataclass(frozen=True)
class StrictCacheAssetShapes:
    transition: int
    null_vectors: tuple[int, ...]
    preconditioned_links: tuple[int, ...]
    onsite_pair: tuple[int, ...]

    @property
    def V(self) -> tuple[int, ...]:
        return self.null_vectors

    @property
    def Yhat(self) -> tuple[int, ...]:
        return self.preconditioned_links

    @property
    def onsite(self) -> tuple[int, ...]:
        return self.onsite_pair


def expected_strict_cache_asset_shapes(
    metadata: StrictCacheShardMetadata | Mapping[str, Any],
) -> tuple[StrictCacheAssetShapes, ...]:
    metadata = validate_strict_cache_shard_metadata(metadata)
    result: list[StrictCacheAssetShapes] = []
    for level, block in enumerate(metadata.block_sizes):
        fine_dof = metadata.dofs[level]
        coarse_dof = metadata.dofs[level + 1]
        coarse_shape = metadata.local_shapes[level + 1]
        null_shape: list[int] = [coarse_dof, fine_dof]
        for axis in range(4):
            null_shape.extend((coarse_shape[axis], block[axis]))
        result.append(StrictCacheAssetShapes(
            transition=level,
            null_vectors=tuple(null_shape),
            preconditioned_links=(
                2, 4, coarse_dof, coarse_dof, *coarse_shape),
            onsite_pair=(2, coarse_dof, coarse_dof, *coarse_shape),
        ))
    return tuple(result)


def _asset_group(value: Any, count: int, name: str) -> tuple[Any, ...]:
    if count == 1 and hasattr(value, "shape"):
        return (value,)
    if isinstance(value, Mapping):
        result: list[Any] = []
        for level in range(count):
            candidates = (level, str(level), f"transition_{level}")
            found = [key for key in candidates if key in value]
            if len(found) != 1:
                raise ValueError(
                    f"{name} mapping 必须为 transition {level} 提供唯一条目")
            result.append(value[found[0]])
        if len(value) != count:
            raise ValueError(f"{name} mapping 含未识别 transition 条目")
        return tuple(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(value)
        if len(result) != count:
            raise ValueError(f"{name} 应有 {count} 项，得到 {len(result)} 项")
        return result
    raise TypeError(f"{name} 必须是每条 transition 的资产序列或 mapping")


def _asset_shape(value: Any, name: str) -> tuple[int, ...]:
    if not hasattr(value, "shape"):
        raise TypeError(f"{name} 必须提供 shape 属性")
    try:
        shape = tuple(int(extent) for extent in value.shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name}.shape 必须是整数序列") from exc
    if any(extent <= 0 for extent in shape):
        raise ValueError(f"{name}.shape 必须全为正数，得到 {shape}")
    return shape


def _select_asset(canonical: Any, alias: Any, canonical_name: str,
                  alias_name: str) -> Any:
    if canonical is not None and alias is not None:
        raise ValueError(f"{canonical_name} 与别名 {alias_name} 不能同时传入")
    result = canonical if canonical is not None else alias
    if result is None:
        raise ValueError(f"缺少 strict cache 资产 {canonical_name}/{alias_name}")
    return result


def validate_strict_cache_shard_assets(
    metadata: StrictCacheShardMetadata | Mapping[str, Any],
    *,
    null_vectors: Any = None,
    preconditioned_links: Any = None,
    onsite_pair: Any = None,
    V: Any = None,
    Yhat: Any = None,
    onsite: Any = None,
) -> tuple[StrictCacheAssetShapes, ...]:
    """验证 rank-local ``V/Yhat/onsite`` 的每层精确 shape 与 dtype。"""

    metadata = validate_strict_cache_shard_metadata(metadata)
    expected = expected_strict_cache_asset_shapes(metadata)
    null_group = _asset_group(
        _select_asset(null_vectors, V, "null_vectors", "V"),
        len(expected), "null_vectors")
    links_group = _asset_group(
        _select_asset(preconditioned_links, Yhat,
                      "preconditioned_links", "Yhat"),
        len(expected), "preconditioned_links")
    onsite_group = _asset_group(
        _select_asset(onsite_pair, onsite, "onsite_pair", "onsite"),
        len(expected), "onsite_pair")

    for item, null_value, links_value, onsite_value in zip(
            expected, null_group, links_group, onsite_group):
        triples = (
            ("null_vectors", null_value, item.null_vectors),
            ("preconditioned_links", links_value,
             item.preconditioned_links),
            ("onsite_pair", onsite_value, item.onsite_pair),
        )
        for name, asset, expected_shape in triples:
            label = f"transition {item.transition} {name}"
            actual_shape = _asset_shape(asset, label)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{label} shape 应为 {expected_shape}，得到 {actual_shape}")
            if hasattr(asset, "dtype"):
                actual_dtype = _normalise_dtype(asset.dtype)
                if actual_dtype != metadata.dtype:
                    raise ValueError(
                        f"{label} dtype 应为 {metadata.dtype}，得到 {actual_dtype}")
    return expected


def validate_strict_cache_shard(
    metadata: StrictCacheShardMetadata | Mapping[str, Any],
    *,
    geometry: StrictMpiGeometry | None = None,
    null_vectors: Any,
    preconditioned_links: Any,
    onsite_pair: Any,
) -> StrictCacheShardMetadata:
    metadata = validate_strict_cache_shard_metadata(
        metadata, geometry=geometry)
    validate_strict_cache_shard_assets(
        metadata,
        null_vectors=null_vectors,
        preconditioned_links=preconditioned_links,
        onsite_pair=onsite_pair,
    )
    return metadata


@dataclass(frozen=True)
class StrictMpiCapabilities:
    """strict 分布式后端的真实能力；单 rank 不需要这些通信能力。"""

    setup_halo: bool = False
    full_halo: bool = False
    compact_halo: bool = False
    global_reduction: bool = True
    fused_fgmres: bool = False

    @property
    def strict_coarse_halo(self) -> bool:
        """兼容旧能力名；full/compact halo 均完成后才可为真。"""

        return self.full_halo and self.compact_halo

    @property
    def distributed_backend_ready(self) -> bool:
        return all((
            self.setup_halo,
            self.full_halo,
            self.compact_halo,
            self.global_reduction,
            self.fused_fgmres,
        ))

    def to_dict(self) -> dict[str, bool]:
        return {
            "setup_halo": self.setup_halo,
            "full_halo": self.full_halo,
            "compact_halo": self.compact_halo,
            "global_reduction": self.global_reduction,
            "fused_fgmres": self.fused_fgmres,
            "strict_coarse_halo": self.strict_coarse_halo,
        }


STRICT_MPI_CAPABILITIES = StrictMpiCapabilities()


def strict_mpi_capabilities() -> dict[str, bool]:
    return STRICT_MPI_CAPABILITIES.to_dict()


def require_strict_coarse_halo() -> NoReturn:
    raise StrictMpiCapabilityError(
        "strict_coarse_halo=False：本模块仅实现安全前置校验，"
        "尚未实现或伪装 coarse halo")


def require_strict_mpi_backend() -> NoReturn:
    missing = ", ".join(
        f"{name}=False" for name, enabled in
        STRICT_MPI_CAPABILITIES.to_dict().items()
        if name != "strict_coarse_halo" and not enabled
    )
    raise StrictMpiCapabilityError(
        "strict MPI backend_ready=False：" + missing + "；"
        "阶段 1 已独立验证 fused FGMRES 标量全局归约；"
        "halo 与完整分布式 solve 尚未实现")


@dataclass(frozen=True)
class StrictMpiPreflightResult:
    geometry: StrictMpiGeometry
    cache_metadata: StrictCacheShardMetadata | None
    capabilities: StrictMpiCapabilities = STRICT_MPI_CAPABILITIES

    @property
    def strict_coarse_halo(self) -> bool:
        return self.capabilities.strict_coarse_halo

    @property
    def setup_halo(self) -> bool:
        return self.capabilities.setup_halo

    @property
    def full_halo(self) -> bool:
        return self.capabilities.full_halo

    @property
    def compact_halo(self) -> bool:
        return self.capabilities.compact_halo

    @property
    def global_reduction(self) -> bool:
        return self.capabilities.global_reduction

    @property
    def fused_fgmres(self) -> bool:
        return self.capabilities.fused_fgmres

    @property
    def backend_ready(self) -> bool:
        # 单 rank 不需要分布式 halo/reduction；多 rank 必须全部能力就绪。
        return (
            self.geometry.comm_size == 1 or
            self.capabilities.distributed_backend_ready
        )

    def require_backend_ready(self) -> None:
        if not self.backend_ready:
            require_strict_mpi_backend()

    def _collective_descriptor(self) -> dict[str, Any]:
        return {
            "process_grid": self.geometry.process_grid,
            "global_shapes": self.geometry.global_shapes,
            "local_shapes": self.geometry.local_shapes,
            "block_sizes": self.geometry.block_sizes,
            "cache_collective_key": (
                None if self.cache_metadata is None
                else self.cache_metadata.collective_key),
            "capabilities": self.capabilities.to_dict(),
        }


ResultT = TypeVar("ResultT")


def collective_strict_preflight(
    comm: Any,
    check: Callable[[], ResultT],
) -> ResultT:
    """先 ``allgather`` 本地错误，再在所有 rank 统一抛出。

    ``check`` 内只能做不会进入后端通信的数据/几何校验。即使一个 rank
    失败，其余 rank 也仍会进入同一次 allgather，避免后续出现部分 rank
    已进入 halo、部分 rank 已抛异常的死锁模式。
    """

    rank = int(comm.Get_rank())
    local_result: ResultT | None = None
    local_failure: tuple[int, str, str] | None = None
    try:
        local_result = check()
    except Exception as exc:  # 不吞 KeyboardInterrupt/SystemExit
        local_failure = (rank, type(exc).__name__, str(exc))

    gathered = tuple(comm.allgather(local_failure))
    failures = tuple(failure for failure in gathered if failure is not None)
    if failures:
        raise StrictMpiPreflightError(failures)
    # 所有 rank 的 check 均成功，因此本 rank 一定有结果；
    # 结果对象本身允许为 None。
    return local_result  # type: ignore[return-value]


def collective_validate_strict_mpi(
    comm: Any,
    *,
    process_grid: Sequence[int],
    global_shapes: Sequence[Sequence[int]],
    block_sizes: Sequence[Sequence[int]],
    level_process_grids: Sequence[Sequence[int]] | None = None,
    rank_coordinate: Sequence[int] | None = None,
    local_parity_origins: Sequence[int] | None = None,
    cache_metadata: (
        StrictCacheShardMetadata | Mapping[str, Any] |
        Callable[[StrictMpiGeometry], StrictCacheShardMetadata | Mapping[str, Any]] |
        None
    ) = None,
    null_vectors: Any = None,
    preconditioned_links: Any = None,
    onsite_pair: Any = None,
    require_backend_ready: bool = False,
) -> StrictMpiPreflightResult:
    """collective 地验证几何、可选 cache shard 和真实 capability。

    对可能不合法的持久化 metadata，推荐传原始 mapping 或 ``factory``；其
    解析会发生在 collective 错误汇聚内部。若 ``require_backend_ready`` 为
    真，多 rank 当前会在所有 rank 一致失败，因为 coarse halo 尚未实现。
    """

    size = int(comm.Get_size())
    rank = int(comm.Get_rank())

    def local_check() -> StrictMpiPreflightResult:
        geometry = validate_strict_mpi_geometry(
            comm_size=size,
            comm_rank=rank,
            process_grid=process_grid,
            global_shapes=global_shapes,
            block_sizes=block_sizes,
            level_process_grids=level_process_grids,
            rank_coordinate=rank_coordinate,
            local_parity_origins=local_parity_origins,
        )
        metadata_value = (
            cache_metadata(geometry) if callable(cache_metadata)
            else cache_metadata
        )
        metadata: StrictCacheShardMetadata | None = None
        any_assets = any(value is not None for value in (
            null_vectors, preconditioned_links, onsite_pair))
        if metadata_value is not None:
            metadata = validate_strict_cache_shard_metadata(
                metadata_value, geometry=geometry)
            if any_assets:
                if not all(value is not None for value in (
                        null_vectors, preconditioned_links, onsite_pair)):
                    raise ValueError(
                        "验证 cache 资产时必须同时提供 V/Yhat/onsite 三组")
                validate_strict_cache_shard_assets(
                    metadata,
                    null_vectors=null_vectors,
                    preconditioned_links=preconditioned_links,
                    onsite_pair=onsite_pair,
                )
        elif any_assets:
            raise ValueError("提供 cache 资产时必须同时提供 cache_metadata")

        result = StrictMpiPreflightResult(
            geometry=geometry, cache_metadata=metadata)
        if require_backend_ready:
            result.require_backend_ready()
        return result

    result = collective_strict_preflight(comm, local_check)

    # 第一轮 allgather 已保证所有本地校验成功。第二轮只核对 rank 间公共
    # 配置一致，防止每个 shard 各自合法、但 gauge/operator 等身份不同。
    descriptors = tuple(comm.allgather(result._collective_descriptor()))
    canonical = tuple(
        dumps(value, sort_keys=True, separators=(",", ":"), default=list)
        for value in descriptors
    )
    if any(value != canonical[0] for value in canonical[1:]):
        digests = [sha256(value.encode("utf-8")).hexdigest()[:12]
                   for value in canonical]
        failures = tuple(
            (rank_id, "CollectiveMismatch",
             f"preflight descriptor digest={digest}; all ranks must agree")
            for rank_id, digest in enumerate(digests)
        )
        raise StrictMpiPreflightError(failures)
    return result


def collective_validate_strict_runtime(
    comm: Any,
    descriptor_factory: Callable[[], Mapping[str, Any]],
    *,
    require_backend_ready: bool = True,
) -> StrictMpiPreflightResult:
    """校验 production strict runtime 描述并复用几何/capability 闸门。

    ``descriptor_factory`` 在 collective 错误汇聚内部执行，必须返回：
    ``process_grid``、``node_size``、``node_rank``、``local_shapes`` 和
    ``block_sizes``。这样即使某个 rank 的 hierarchy/params 描述损坏，
    其他 rank 也不会越过 preflight 进入 setup 或 CUDA 分配。
    """

    size = int(comm.Get_size())
    rank = int(comm.Get_rank())

    def local_runtime_check() -> dict[str, Any]:
        raw = descriptor_factory()
        if not isinstance(raw, Mapping):
            raise TypeError("strict runtime descriptor 必须是 mapping")
        required = {
            "process_grid", "node_size", "node_rank",
            "local_shapes", "block_sizes",
        }
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(
                f"strict runtime descriptor 缺少字段：{missing}")

        node_size = _positive_int(raw["node_size"], "params node_size")
        node_rank_value = raw["node_rank"]
        if (isinstance(node_rank_value, bool) or
                not isinstance(node_rank_value, Integral)):
            raise TypeError("params node_rank 必须是整数")
        node_rank = int(node_rank_value)
        if node_size != size or node_rank != rank:
            raise ValueError(
                "params node 协议与 MPI.COMM_WORLD 不一致："
                f"params size/rank={node_size}/{node_rank}, "
                f"world size/rank={size}/{rank}")

        grid = _shape4(raw["process_grid"], "process_grid")
        local_shapes = _shape_levels(raw["local_shapes"], "local_shapes")
        blocks = _block_levels(raw["block_sizes"], len(local_shapes) - 1)
        for level, block in enumerate(blocks):
            current = local_shapes[level]
            if any(current[axis] % block[axis] for axis in range(4)):
                raise ValueError(
                    f"level {level} local_shape={current} 不能被 "
                    f"block_size={block} 整除")
            expected = tuple(
                current[axis] // block[axis] for axis in range(4))
            if local_shapes[level + 1] != expected:
                raise ValueError(
                    f"level {level + 1} local_shape 应为 {expected}，"
                    f"得到 {local_shapes[level + 1]}")

        global_shapes = tuple(
            tuple(local[axis] * grid[axis] for axis in range(4))
            for local in local_shapes
        )
        return {
            "process_grid": grid,
            "global_shapes": global_shapes,
            "block_sizes": blocks,
            "level_process_grids": (grid,) * len(global_shapes),
            # strict CUDA checkerboard decode 当前把每个 local origin 当偶点。
            "local_parity_origins": (0,) * len(global_shapes),
        }

    runtime = collective_strict_preflight(comm, local_runtime_check)
    return collective_validate_strict_mpi(
        comm,
        process_grid=runtime["process_grid"],
        global_shapes=runtime["global_shapes"],
        block_sizes=runtime["block_sizes"],
        level_process_grids=runtime["level_process_grids"],
        local_parity_origins=runtime["local_parity_origins"],
        require_backend_ready=require_backend_ready,
    )


__all__ = [
    "STRICT_MPI_CACHE_SCHEMA",
    "STRICT_MPI_CACHE_SCHEMA_VERSION",
    "STRICT_MPI_CAPABILITIES",
    "StrictCacheAssetShapes",
    "StrictCacheShardMetadata",
    "StrictMpiCapabilities",
    "StrictMpiCapabilityError",
    "StrictMpiGeometry",
    "StrictMpiLevelGeometry",
    "StrictMpiPreflightError",
    "StrictMpiPreflightResult",
    "collective_strict_preflight",
    "collective_validate_strict_mpi",
    "collective_validate_strict_runtime",
    "expected_strict_cache_asset_shapes",
    "make_strict_cache_shard_metadata",
    "require_strict_coarse_halo",
    "require_strict_mpi_backend",
    "strict_mpi_world_communicator",
    "strict_mpi_capabilities",
    "validate_strict_cache_shard",
    "validate_strict_cache_shard_assets",
    "validate_strict_cache_shard_metadata",
    "validate_strict_mpi_geometry",
]
