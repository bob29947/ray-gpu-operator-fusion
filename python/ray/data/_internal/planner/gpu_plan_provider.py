"""Driver-side provider dispatch for opt-in GPU physical-plan fragments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Protocol, Tuple, Union

from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators import MapGroupPartitions
from ray.data.context import DataContext


@dataclass(frozen=True)
class GPUPlanFragment:
    """A complete physical fragment returned by a GPU plan provider."""

    root: PhysicalOperator
    new_operators: Tuple[PhysicalOperator, ...]
    consumed_logical_operators: Tuple[LogicalOperator, ...]
    op_map: Mapping[PhysicalOperator, LogicalOperator]
    region_id: str


@dataclass(frozen=True)
class GPUProviderNotApplicable:
    pass


@dataclass(frozen=True)
class GPUProviderRejected:
    reason: str


@dataclass(frozen=True)
class GPUProviderAccepted:
    fragment: GPUPlanFragment


GPUProviderResult = Union[
    GPUProviderNotApplicable,
    GPUProviderRejected,
    GPUProviderAccepted,
]


class GPUPlanProvider(Protocol):
    def plan(
        self,
        logical_op: LogicalOperator,
        data_context: DataContext,
        consumer_counts: Mapping[LogicalOperator, int],
    ) -> GPUProviderResult: ...


GPUPlanProviderLoader = Callable[[], GPUPlanProvider]


def _load_parquet_cudf_map_group_partitions_provider() -> GPUPlanProvider:
    # Keep Parquet, PyArrow, and RAPIDS imports off the stock planner path.
    from ray.data._internal.planner.parquet_cudf_map_group_partitions_provider import (
        ParquetCudfMapGroupPartitionsProvider,
    )

    return ParquetCudfMapGroupPartitionsProvider()


_GPU_PROVIDER_LOADERS_BY_EXACT_ROOT: Dict[
    type[LogicalOperator], Tuple[GPUPlanProviderLoader, ...]
] = {
    MapGroupPartitions: (_load_parquet_cudf_map_group_partitions_provider,),
}


def has_gpu_plan_provider(logical_op: LogicalOperator) -> bool:
    """Return whether an exact logical root has any GPU provider candidates."""

    return type(logical_op) in _GPU_PROVIDER_LOADERS_BY_EXACT_ROOT


def plan_with_gpu_provider(
    logical_op: LogicalOperator,
    data_context: DataContext,
    consumer_counts: Mapping[LogicalOperator, int],
) -> GPUPlanFragment | None:
    """Try exact-root providers without silently falling back after rejection."""

    loaders = _GPU_PROVIDER_LOADERS_BY_EXACT_ROOT.get(type(logical_op), ())
    for loader in loaders:
        result = loader().plan(logical_op, data_context, consumer_counts)
        if isinstance(result, GPUProviderNotApplicable):
            continue
        if isinstance(result, GPUProviderRejected):
            raise ValueError(
                "GPU operator planning rejected the requested pipeline: "
                f"{result.reason}"
            ) from None
        assert isinstance(result, GPUProviderAccepted), result
        return result.fragment
    return None
