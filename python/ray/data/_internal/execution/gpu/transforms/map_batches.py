"""Actor-local cuDF map-batches and range-ownership validation transforms."""

from __future__ import annotations

import collections
import collections.abc
import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from ray.data._internal.execution.gpu.payloads import RangeData
from ray.data._internal.execution.gpu.transforms._runtime_support import (
    CudfRuntimeContext,
)
from ray.data._internal.execution.gpu.transforms._user_code import (
    invoke_user_code,
    iterate_user_code,
    raise_user_code_exception,
)

MAP_BATCHES_KIND = "map_batches"
VALIDATE_RANGE_OWNERSHIP_KIND = "validate_range_ownership"


@dataclass(frozen=True, eq=False)
class MapBatchesConfig:
    """Immutable execution arguments for one cuDF batch UDF."""

    udf_cls: Any
    batch_size: int
    zero_copy_batch: bool
    fn_args: tuple[Any, ...] = ()
    fn_kwargs: tuple[tuple[str, Any], ...] = ()
    fn_constructor_args: tuple[Any, ...] = ()
    fn_constructor_kwargs: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fn_args", tuple(self.fn_args))
        object.__setattr__(self, "fn_kwargs", tuple(self.fn_kwargs))
        object.__setattr__(self, "fn_constructor_args", tuple(self.fn_constructor_args))
        object.__setattr__(
            self, "fn_constructor_kwargs", tuple(self.fn_constructor_kwargs)
        )
        if (
            not isinstance(self.batch_size, numbers.Integral)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise ValueError("MapBatchesConfig.batch_size must be a positive integer")
        if len({key for key, _ in self.fn_kwargs}) != len(self.fn_kwargs):
            raise ValueError("MapBatchesConfig.fn_kwargs contains duplicate keys")
        if len({key for key, _ in self.fn_constructor_kwargs}) != len(
            self.fn_constructor_kwargs
        ):
            raise ValueError(
                "MapBatchesConfig.fn_constructor_kwargs contains duplicate keys"
            )

    @classmethod
    def from_logical_op(cls, op: Any) -> "MapBatchesConfig":
        return cls(
            udf_cls=op.fn,
            batch_size=int(op.batch_size),
            zero_copy_batch=bool(op.zero_copy_batch),
            fn_args=tuple(op.fn_args or ()),
            fn_kwargs=tuple((op.fn_kwargs or {}).items()),
            fn_constructor_args=tuple(op.fn_constructor_args or ()),
            fn_constructor_kwargs=tuple((op.fn_constructor_kwargs or {}).items()),
        )


@dataclass(frozen=True)
class ValidateRangeOwnershipConfig:
    range_key: str
    group_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_keys", tuple(self.group_keys))
        if not self.range_key or not self.group_keys:
            raise ValueError("Range and group keys must be nonempty")
        if self.group_keys[0] != self.range_key:
            raise ValueError("The leading group key must be the range key")


def rebatched_frames(
    frames: Iterable[Any],
    batch_size: int,
    cudf: Any,
) -> Iterator[Any]:
    """Yield exact-size batches without repeatedly concatenating a tail."""

    pending: collections.deque[list[Any]] = collections.deque()
    available = 0

    def take(rows: int) -> Any:
        nonlocal available
        pieces = []
        remaining = rows
        while remaining:
            frame, offset = pending[0]
            count = min(remaining, len(frame) - offset)
            pieces.append(frame.iloc[offset : offset + count])
            offset += count
            remaining -= count
            available -= count
            if offset == len(frame):
                pending.popleft()
            else:
                pending[0][1] = offset
        if len(pieces) == 1:
            return pieces[0].reset_index(drop=True)
        return cudf.concat(pieces, ignore_index=True)

    for frame in frames:
        if len(frame) == 0:
            continue
        pending.append([frame, 0])
        available += len(frame)
        while available >= batch_size:
            yield take(batch_size)
    if available:
        yield take(available)


def one_or_generator(value: Any) -> Iterator[Any]:
    return value if isinstance(value, collections.abc.Iterator) else iter((value,))


class MapBatchesRuntime:
    def __init__(
        self,
        config: MapBatchesConfig,
        context: CudfRuntimeContext,
    ) -> None:
        if not isinstance(config, MapBatchesConfig):
            raise TypeError("map_batches requires MapBatchesConfig")
        self._config = config
        self._context = context
        self._udf: Any = None

    def initialize(self) -> None:
        self._context.ensure_cudf()
        self._udf = invoke_user_code(
            self._config.udf_cls,
            *self._config.fn_constructor_args,
            **dict(self._config.fn_constructor_kwargs),
        )

    def _apply_frames(self, frames: Iterable[Any]) -> Iterator[Any]:
        cudf = self._context.cudf
        for raw in rebatched_frames(frames, int(self._config.batch_size), cudf):
            udf_input = raw if self._config.zero_copy_batch else raw.copy(deep=True)
            result = invoke_user_code(
                self._udf,
                udf_input,
                *self._config.fn_args,
                **dict(self._config.fn_kwargs),
            )
            for output in iterate_user_code(one_or_generator(result)):
                try:
                    if not isinstance(output, cudf.DataFrame):
                        raise TypeError(
                            "The GPU map_batches UDF must return cuDF DataFrames"
                        )
                    output = output.reset_index(drop=True)
                    if len(output):
                        yield output
                except Exception as error:
                    raise_user_code_exception(error)

    def apply(self, value: Any, task_context: Any) -> RangeData:
        del task_context
        if not isinstance(value, RangeData):
            raise TypeError("map_batches expected actor-local RangeData")
        return RangeData(value.partition, self._apply_frames(value.value))


class ValidateRangeOwnershipRuntime:
    def __init__(self, config: ValidateRangeOwnershipConfig) -> None:
        if not isinstance(config, ValidateRangeOwnershipConfig):
            raise TypeError(
                "validate_range_ownership requires ValidateRangeOwnershipConfig"
            )
        self._config = config

    def initialize(self) -> None:
        pass

    def _validate(self, value: RangeData) -> Iterator[Any]:
        for frame in value.value:
            # Iterator advancement stays outside this try block so upstream
            # I/O/system failures are not mislabeled as user contract errors.
            try:
                if not len(frame):
                    continue
                if any(key not in frame for key in self._config.group_keys):
                    raise ValueError("The GPU map_batches UDF removed a group key")
                range_values = frame[self._config.range_key]
                range_owned = not bool(range_values.isnull().any()) and bool(
                    (
                        (range_values >= value.partition.lower)
                        & (range_values <= value.partition.upper)
                    ).all()
                )
                if not range_owned:
                    raise ValueError(
                        "The GPU map_batches UDF produced a leading group key "
                        "outside its assigned range"
                    )
            except Exception as error:
                raise_user_code_exception(error)
            yield frame

    def apply(self, value: Any, task_context: Any) -> RangeData:
        del task_context
        if not isinstance(value, RangeData):
            raise TypeError("range validation expected actor-local RangeData")
        return RangeData(value.partition, self._validate(value))
