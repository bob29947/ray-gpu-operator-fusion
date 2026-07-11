"""Partition materialization, grouping, and map-group-partitions transforms."""

from __future__ import annotations

import collections.abc
import numbers
from dataclasses import dataclass
from typing import Any, Iterator

from ray.data._internal.execution.gpu.parquet_types import ParquetRangeWork, RangeData
from ray.data._internal.execution.gpu.transforms._runtime_support import (
    CudfRuntimeContext,
)

SORT_PARTITION_KIND = "sort_partition"
BUILD_GROUP_BOUNDARIES_KIND = "build_group_boundaries"
MAP_GROUP_PARTITIONS_KIND = "map_group_partitions"


@dataclass(frozen=True)
class SortPartitionConfig:
    group_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_keys", tuple(self.group_keys))
        if not self.group_keys:
            raise ValueError("SortPartitionConfig requires group keys")


@dataclass(frozen=True)
class BuildGroupBoundariesConfig:
    group_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_keys", tuple(self.group_keys))
        if not self.group_keys:
            raise ValueError("BuildGroupBoundariesConfig requires group keys")


@dataclass(frozen=True, eq=False)
class MapGroupPartitionsConfig:
    partition_fn: Any
    group_keys: tuple[str, ...]
    target_max_block_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_keys", tuple(self.group_keys))
        if not self.group_keys:
            raise ValueError("MapGroupPartitionsConfig requires group keys")
        if (
            not isinstance(self.target_max_block_size, numbers.Integral)
            or isinstance(self.target_max_block_size, bool)
            or self.target_max_block_size <= 0
        ):
            raise ValueError("target_max_block_size must be a positive integer")


@dataclass(frozen=True, eq=False)
class GroupedPartitionData:
    work: ParquetRangeWork
    frame: Any
    boundaries: tuple[int, ...]


def is_sorted(frame: Any, keys: tuple[str, ...]) -> bool:
    if len(frame) < 2:
        return True
    try:
        import pylibcudf as plc

        table, _ = frame.loc[:, list(keys)].to_pylibcudf(copy=False)
        return bool(
            plc.sorting.is_sorted(
                table,
                [plc.types.Order.ASCENDING] * len(keys),
                [plc.types.NullOrder.AFTER] * len(keys),
            )
        )
    except (AttributeError, ImportError, TypeError):
        return False


def group_boundaries(
    frame: Any,
    keys: tuple[str, ...],
    cupy: Any,
) -> tuple[int, ...]:
    if len(frame) == 0:
        return (0,)
    changed = cupy.zeros(len(frame), dtype=cupy.bool_)
    changed[0] = True
    for key in keys:
        values = frame[key]
        previous = values.shift(1)
        missing = values.isnull()
        previous_missing = previous.isnull()
        changed |= ((values != previous) & ~(missing & previous_missing)).fillna(
            False
        ).to_cupy() | (missing != previous_missing).to_cupy()
    return tuple(int(index) for index in cupy.flatnonzero(changed).get()) + (
        len(frame),
    )


def split_output(batch: Any, target_bytes: int) -> Iterator[Any]:
    """Split a user batch without altering row order or schema."""

    if isinstance(batch, dict):
        values = tuple(batch.values())
        rows = len(values[0]) if values else 0
        if any(len(value) != rows for value in values):
            raise ValueError("map_group_partitions output columns have unequal lengths")
        size = sum(int(getattr(value, "nbytes", 0)) for value in values)

        def take(start: int, end: int) -> Any:
            return {key: value[start:end] for key, value in batch.items()}

    else:
        rows = len(batch)
        memory_usage = getattr(batch, "memory_usage", None)
        size = (
            int(memory_usage(deep=True).sum())
            if memory_usage is not None
            else int(getattr(batch, "nbytes", 0))
        )

        def take(start: int, end: int) -> Any:
            if hasattr(batch, "iloc"):
                return batch.iloc[start:end]
            return batch.slice(start, end - start)

    if rows == 0:
        yield batch
        return
    rows_per_output = max(1, int(rows * target_bytes / size)) if size else rows
    for start in range(0, rows, rows_per_output):
        yield take(start, min(rows, start + rows_per_output))


def _one_or_generator(value: Any) -> Iterator[Any]:
    return value if isinstance(value, collections.abc.Iterator) else iter((value,))


class SortPartitionRuntime:
    def __init__(
        self,
        config: SortPartitionConfig,
        context: CudfRuntimeContext,
    ) -> None:
        if not isinstance(config, SortPartitionConfig):
            raise TypeError("sort_partition requires SortPartitionConfig")
        self._config = config
        self._context = context

    def initialize(self) -> None:
        self._context.ensure_cudf()

    def apply(self, value: Any, task_context: Any) -> RangeData:
        del task_context
        if not isinstance(value, RangeData):
            raise TypeError("sort_partition expected actor-local RangeData")
        frames = list(value.value)
        if not frames:
            return RangeData(value.work, None)
        frame = (
            frames[0]
            if len(frames) == 1
            else self._context.cudf.concat(frames, ignore_index=True)
        )
        frames.clear()
        if any(key not in frame for key in self._config.group_keys):
            raise ValueError("The GPU map_batches UDF removed a group key")
        if not is_sorted(frame, self._config.group_keys):
            frame = frame.sort_values(
                list(self._config.group_keys),
                ignore_index=True,
                na_position="last",
            )
        return RangeData(value.work, frame)


class BuildGroupBoundariesRuntime:
    def __init__(
        self,
        config: BuildGroupBoundariesConfig,
        context: CudfRuntimeContext,
    ) -> None:
        if not isinstance(config, BuildGroupBoundariesConfig):
            raise TypeError(
                "build_group_boundaries requires BuildGroupBoundariesConfig"
            )
        self._config = config
        self._context = context

    def initialize(self) -> None:
        self._context.ensure_cudf()

    def apply(self, value: Any, task_context: Any) -> GroupedPartitionData:
        del task_context
        if not isinstance(value, RangeData):
            raise TypeError("group-boundary construction expected RangeData")
        if value.value is None:
            return GroupedPartitionData(value.work, None, (0,))
        boundaries = group_boundaries(
            value.value, self._config.group_keys, self._context.cupy
        )
        return GroupedPartitionData(value.work, value.value, boundaries)


class MapGroupPartitionsRuntime:
    def __init__(self, config: MapGroupPartitionsConfig) -> None:
        if not isinstance(config, MapGroupPartitionsConfig):
            raise TypeError("map_group_partitions requires MapGroupPartitionsConfig")
        self._config = config

    def initialize(self) -> None:
        pass

    def _apply_partition(self, value: GroupedPartitionData) -> Iterator[Any]:
        if value.frame is None:
            return
        from ray.data.grouped_data import MapGroupPartitionsContext

        context = MapGroupPartitionsContext(
            group_keys=self._config.group_keys,
            input_group_boundaries=value.boundaries,
        )
        result = self._config.partition_fn(value.frame, context)
        from ray.data.block import BlockAccessor

        for output in _one_or_generator(result):
            for batch in split_output(output, int(self._config.target_max_block_size)):
                yield BlockAccessor.batch_to_block(batch)

    def apply(self, value: Any, task_context: Any) -> Iterator[Any]:
        del task_context
        if not isinstance(value, GroupedPartitionData):
            raise TypeError("map_group_partitions expected GroupedPartitionData")
        return self._apply_partition(value)
