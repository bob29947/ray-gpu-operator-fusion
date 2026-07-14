"""Read-source contract and the V1 Parquet range implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, Tuple

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
class GPUReadSourceSupport:
    accepted: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.accepted == bool(self.reason):
            raise ValueError(
                "accepted source support must have no reason and rejection must "
                "have one"
            )


@dataclass(frozen=True)
class GPUReadLoweringRequest:
    logical_node_id: str
    group_keys: Tuple[str, ...]
    execution_requirements: GPUExecutionRequirements
    target_max_block_size: int
    retried_io_errors: Tuple[str, ...]
    target_ranges: int
    num_workers: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_keys", tuple(self.group_keys))
        object.__setattr__(self, "retried_io_errors", tuple(self.retried_io_errors))
        if not self.logical_node_id:
            raise ValueError("logical_node_id must be nonempty")
        if not self.group_keys:
            raise ValueError("GPU Read lowering requires group keys")
        if self.target_max_block_size <= 0:
            raise ValueError("target_max_block_size must be positive")
        if self.target_ranges <= 0 or self.num_workers <= 0:
            raise ValueError("GPU Read lowering counts must be positive")

    def transform_id(self, ordinal: int, kind: str) -> str:
        if ordinal < 0 or not kind:
            raise ValueError("transform identity requires ordinal and kind")
        return f"{self.logical_node_id}/{ordinal}/{kind}"


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

    def supports(
        self,
        logical_op: LogicalOperator,
        demand: GPUReadSourceDemand,
    ) -> GPUReadSourceSupport: ...

    def lower(
        self,
        logical_op: LogicalOperator,
        request: GPUReadLoweringRequest,
    ) -> GPUReadSourceLowering: ...


class ParquetRangeGPUReadProvider:
    """The only V1 source: a direct Parquet/cuDF range Read."""

    provider_name = "parquet_range"

    def supports(
        self,
        logical_op: LogicalOperator,
        demand: GPUReadSourceDemand,
    ) -> GPUReadSourceSupport:
        del demand
        # This is intentionally only a no-I/O shape probe.  Full recognition,
        # descriptor construction, footer reads, and ranges occur in lower().
        from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
        from ray.data._internal.logical.operators import Read

        if type(logical_op) is not Read:
            return GPUReadSourceSupport(False, "logical_read")
        datasource = logical_op.datasource
        if (
            type(datasource) is not ParquetDatasource
            or logical_op.datasource_or_legacy_reader is not datasource
        ):
            return GPUReadSourceSupport(False, "parquet_datasource")
        return GPUReadSourceSupport(True)

    def lower(
        self,
        logical_op: LogicalOperator,
        request: GPUReadLoweringRequest,
    ) -> GPUReadSourceLowering:
        # This method is called only by the generic Read lowerer's lower()
        # dispatch.  Source-specific planning cannot leak into analysis.
        from ray.data._internal.execution.gpu import (
            CompletePartitionBundles,
            Evidence,
            GPUTransformSpec,
            PayloadKind,
            PropertyFact,
            RANGE_DATA_BOUNDARY_ADAPTER_KEY,
            RANGE_DATA_TERMINAL_ADAPTER_KEY,
            RANGE_WORK_ADAPTER_KEY,
            RangePartitioning,
        )
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

        recognition = recognize_parquet_cudf_read(logical_op, request.group_keys)
        if not recognition.accepted:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={type(logical_op).__name__}; "
                f"required_keys={request.group_keys!r}; "
                f"required_count={request.target_ranges}; specialized "
                "Parquet/cuDF Read recognition failed "
                f"({recognition.reason})",
            )
        descriptor = recognition.descriptor
        assert descriptor is not None
        try:
            range_plan = plan_parquet_ranges(
                descriptor,
                range_key=request.group_keys[0],
                target_ranges=request.target_ranges,
                num_workers=request.num_workers,
                retried_io_errors=request.retried_io_errors,
            )
        except ParquetCudfPlanningError as error:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={type(logical_op).__name__}; "
                f"required_keys={request.group_keys!r}; "
                f"required_count={request.target_ranges}; Parquet range planning "
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
        requirements = _requirements_for_source(
            request.execution_requirements,
            descriptor,
            request.retried_io_errors,
        )
        transform = GPUTransformSpec(
            transform_id=request.transform_id(0, READ_PARQUET_RANGE_KIND),
            kind=READ_PARQUET_RANGE_KIND,
            input_payload=PayloadKind.RANGE_WORK,
            output_payload=PayloadKind.RANGE_DATA,
            config=ReadParquetRangeConfig(
                source_kind=descriptor.source_kind,
                region=descriptor.region,
                projection=descriptor.projection,
                range_key=range_plan.range_key,
                retried_io_errors=request.retried_io_errors,
            ),
        )
        spec = GPUOperatorSpec(
            spec_instance_id=request.logical_node_id,
            transforms=(transform,),
            external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
            external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_input_payload=PayloadKind.RANGE_WORK,
            native_output_payload=PayloadKind.RANGE_DATA,
            input_adapter_key=RANGE_WORK_ADAPTER_KEY,
            output_adapter_key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
            terminal_output_adapter_key=RANGE_DATA_TERMINAL_ADAPTER_KEY,
            execution_requirements=requirements,
            output_properties=(
                PropertyFact(
                    RangePartitioning((range_plan.range_key,), range_plan.scheme_id),
                    Evidence.OPERATOR_GUARANTEE,
                ),
                PropertyFact(
                    CompletePartitionBundles(),
                    Evidence.OPERATOR_GUARANTEE,
                ),
            ),
            logical_provenance=(request.logical_node_id,),
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
    bootstrap = dict(key.actor_bootstrap)
    bootstrap.update(
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


__all__ = [
    "GPUReadLoweringRequest",
    "GPUReadSourceArtifacts",
    "GPUReadSourceDemand",
    "GPUReadSourceLowering",
    "GPUReadSourceProvider",
    "GPUReadSourceSupport",
    "ParquetRangeGPUReadProvider",
]
