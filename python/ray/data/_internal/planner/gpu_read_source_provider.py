"""Read-source contract and the V1 Parquet range implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple

from ray.data._internal.execution.gpu.operator import GPUOperatorCreationOptions
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionRequirements,
    GPUOperatorSpec,
)
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)


@dataclass(frozen=True)
class GPUReadSourceDemand:
    group_keys: Tuple[str, ...]
    target_ranges: int
    num_workers: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_keys", tuple(self.group_keys))
        if not self.group_keys:
            raise ValueError("GPU Read source demand requires group keys")
        if self.target_ranges <= 0 or self.num_workers <= 0:
            raise ValueError("GPU Read source demand counts must be positive")


@dataclass(frozen=True)
class GPUReadSourceAnalysis:
    accepted: bool
    reason: str = ""
    provider_state: Any = None
    execution_requirements: Optional[GPUExecutionRequirements] = None

    def __post_init__(self) -> None:
        if self.accepted == bool(self.reason):
            raise ValueError(
                "accepted source support must have no reason and rejection must "
                "have one"
            )


@dataclass(frozen=True)
class GPUReadLoweringRequest:
    provider_state: Any
    execution_requirements: GPUExecutionRequirements
    target_max_block_size: int
    retried_io_errors: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "retried_io_errors", tuple(self.retried_io_errors))
        if self.target_max_block_size <= 0:
            raise ValueError("target_max_block_size must be positive")


@dataclass(frozen=True)
class GPUReadSourceArtifacts:
    """Generic external input factory and opaque provider-owned artifacts."""

    input_data_factory: Callable[[int], list[Any]]
    provider_artifacts: Any = None

    def __post_init__(self) -> None:
        if not callable(self.input_data_factory):
            raise TypeError("GPU Read input_data_factory must be callable")


@dataclass(frozen=True)
class GPUReadSourceLowering:
    spec: GPUOperatorSpec
    creation_options: GPUOperatorCreationOptions
    artifacts: GPUReadSourceArtifacts


class GPUReadSourceProvider(Protocol):
    provider_name: str

    def analyze(
        self,
        logical_op: LogicalOperator,
        demands: Sequence[Any],
    ) -> GPUReadSourceAnalysis: ...

    def lower(
        self,
        logical_op: LogicalOperator,
        request: GPUReadLoweringRequest,
    ) -> GPUReadSourceLowering: ...


class ParquetRangeGPUReadProvider:
    """The only V1 source: a direct Parquet/cuDF range Read."""

    provider_name = "parquet_range"

    def analyze(
        self,
        logical_op: LogicalOperator,
        demands: Sequence[Any],
    ) -> GPUReadSourceAnalysis:
        from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
        from ray.data._internal.logical.operators import Read
        from ray.data._internal.planner.gpu_execution_profile import (
            reconcile_read_remote_args,
        )
        from ray.data._internal.planner.gpu_requirement_analysis import (
            RangePartitionDemand,
        )

        if type(logical_op) is not Read:
            return GPUReadSourceAnalysis(False, "logical_read")
        datasource = logical_op.datasource
        if (
            type(datasource) is not ParquetDatasource
            or logical_op.datasource_or_legacy_reader is not datasource
        ):
            return GPUReadSourceAnalysis(False, "parquet_datasource")
        range_demands = tuple(
            demand for demand in demands if isinstance(demand, RangePartitionDemand)
        )
        if len(range_demands) != 1:
            return GPUReadSourceAnalysis(False, "range_partition_demand")
        demand = range_demands[0]
        try:
            provider_state = GPUReadSourceDemand(
                demand.keys,
                *demand.resolved_counts(),
            )
        except GPUPlanningError as error:
            return GPUReadSourceAnalysis(False, error.detail)
        execution = demand.execution_requirements
        assert execution is not None
        return GPUReadSourceAnalysis(
            True,
            provider_state=provider_state,
            execution_requirements=reconcile_read_remote_args(
                execution,
                logical_op.ray_remote_args,
                consumer=type(logical_op).__name__,
            ),
        )

    def lower(
        self,
        logical_op: LogicalOperator,
        request: GPUReadLoweringRequest,
    ) -> GPUReadSourceLowering:
        # This method is called only by the generic Read lowerer's lower()
        # dispatch.  Source-specific planning cannot leak into analysis.
        from ray.data._internal.execution.gpu.operator_spec import (
            GPUTransformSpec,
            RANGE_DATA,
            RANGE_WORK,
        )
        from ray.data._internal.execution.gpu.properties import RangePartitioning
        from ray.data._internal.execution.gpu.transforms.parquet_read import (
            READ_PARQUET_RANGE_KIND,
            ReadParquetRangeConfig,
        )
        from ray.data._internal.planner.parquet_cudf_range_planner import (
            ParquetCudfPlanningError,
            plan_parquet_ranges,
        )
        from ray.data._internal.planner.parquet_cudf_read_recognizer import (
            recognize_parquet_cudf_read,
        )

        demand = request.provider_state
        if not isinstance(demand, GPUReadSourceDemand):
            raise _unsupported(
                "Parquet range source received incompatible provider analysis",
            )
        requirement = (
            f"consumer={type(logical_op).__name__}; "
            f"required_keys={demand.group_keys!r}; "
            f"required_count={demand.target_ranges}"
        )
        recognition = recognize_parquet_cudf_read(logical_op, demand.group_keys)
        if not recognition.accepted:
            raise _unsupported(
                f"{requirement}; specialized "
                "Parquet/cuDF Read recognition failed "
                f"({recognition.reason})",
            )
        descriptor = recognition.descriptor
        assert descriptor is not None
        try:
            range_plan = plan_parquet_ranges(
                descriptor,
                range_key=demand.group_keys[0],
                target_ranges=demand.target_ranges,
                num_workers=demand.num_workers,
                retried_io_errors=request.retried_io_errors,
            )
        except ParquetCudfPlanningError as error:
            raise _unsupported(
                f"{requirement}; Parquet range planning "
                f"failed ({error.reason}): {error}",
            ) from error

        from ray.data._internal.execution.gpu.transforms.boundary import (
            descriptor_bundles,
        )

        work = tuple(range_plan.work)
        artifacts = GPUReadSourceArtifacts(
            input_data_factory=lambda _target_size, work=work: descriptor_bundles(work),
            provider_artifacts=(descriptor, range_plan),
        )
        transform = GPUTransformSpec(
            kind=READ_PARQUET_RANGE_KIND,
            input_payload=RANGE_WORK,
            output_payload=RANGE_DATA,
            config=ReadParquetRangeConfig(
                source_kind=descriptor.source_kind,
                region=descriptor.region,
                projection=descriptor.projection,
                range_key=range_plan.range_key,
                retried_io_errors=request.retried_io_errors,
            ),
        )
        spec = GPUOperatorSpec(
            transforms=(transform,),
            execution_requirements=_requirements_for_source(
                request.execution_requirements,
                descriptor,
                request.retried_io_errors,
            ),
            output_facts=(
                RangePartitioning((range_plan.range_key,), range_plan.scheme_id),
            ),
        )
        return GPUReadSourceLowering(
            spec=spec,
            creation_options=GPUOperatorCreationOptions(
                name="GPUReadParquetRange",
                min_rows_per_bundle=1,
                target_max_block_size_override=request.target_max_block_size,
            ),
            artifacts=artifacts,
        )


def _requirements_for_source(
    requirements: GPUExecutionRequirements,
    descriptor: Any,
    retried_io_errors: Tuple[str, ...],
) -> GPUExecutionRequirements:
    key = requirements.compatibility_key
    if key is None:
        raise _unsupported("GPU Read requires a concrete execution profile")
    bootstrap = dict(
        key.actor_bootstrap,
        source_kind=descriptor.source_kind,
        region=descriptor.region or "",
    )
    return replace(
        requirements,
        compatibility_key=replace(
            key,
            actor_bootstrap=tuple(sorted(bootstrap.items())),
            retry_policy=replace(
                key.retry_policy,
                retried_io_errors=tuple(retried_io_errors),
            ),
        ),
    )


def _unsupported(detail: str) -> GPUPlanningError:
    return GPUPlanningError(GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY, detail)


__all__ = [
    "GPUReadLoweringRequest",
    "GPUReadSourceAnalysis",
    "GPUReadSourceArtifacts",
    "GPUReadSourceDemand",
    "GPUReadSourceLowering",
    "GPUReadSourceProvider",
    "ParquetRangeGPUReadProvider",
]
