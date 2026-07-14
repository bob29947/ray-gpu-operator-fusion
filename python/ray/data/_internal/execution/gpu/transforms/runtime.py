"""Built-in payload-exact transform runtime registrations."""

from __future__ import annotations

from typing import Any, Type

from ray.data._internal.execution.gpu.operator_spec import (
    GROUPED_RANGE_DATA,
    RANGE_DATA,
    RANGE_PARTITION_DATA,
    RANGE_WORK,
    RAY_BLOCK_STREAM,
    RuntimeKey,
)
from ray.data._internal.execution.gpu.runtime import (
    GPUTransformRuntimeContext,
    TransformRuntimeRegistration,
    register_gpu_transform_runtime,
)
from ray.data._internal.execution.gpu.transforms._runtime_support import (
    CudfRuntimeContext,
)
from ray.data._internal.execution.gpu.transforms.boundary import (
    EXPORT_RANGE_DATA_KIND,
    IMPORT_RANGE_WORK_KIND,
    ExportRangeDataRuntime,
    ImportRangeWorkRuntime,
)
from ray.data._internal.execution.gpu.transforms.grouped_partitions import (
    BUILD_GROUP_BOUNDARIES_KIND,
    MAP_GROUP_PARTITIONS_KIND,
    SORT_PARTITION_KIND,
    BuildGroupBoundariesConfig,
    BuildGroupBoundariesRuntime,
    MapGroupPartitionsConfig,
    MapGroupPartitionsRuntime,
    SortPartitionConfig,
    SortPartitionRuntime,
)
from ray.data._internal.execution.gpu.transforms.map_batches import (
    MAP_BATCHES_KIND,
    VALIDATE_RANGE_OWNERSHIP_KIND,
    MapBatchesConfig,
    MapBatchesRuntime,
    ValidateRangeOwnershipConfig,
    ValidateRangeOwnershipRuntime,
)
from ray.data._internal.execution.gpu.transforms.parquet_read import (
    READ_PARQUET_RANGE_KIND,
    ReadParquetRangeConfig,
    ReadParquetRangeRuntime,
)


def _is_none(config: Any) -> bool:
    return config is None


def _is_instance(config_type: Type[Any]):
    def validate(config: Any) -> bool:
        return isinstance(config, config_type)

    return validate


def _boundary_factory(transform_spec: Any, context: Any) -> Any:
    del context
    if transform_spec.kind == IMPORT_RANGE_WORK_KIND:
        return ImportRangeWorkRuntime()
    if transform_spec.kind == EXPORT_RANGE_DATA_KIND:
        return ExportRangeDataRuntime()
    raise ValueError(f"Unknown GPU boundary transform: {transform_spec.kind!r}")


def _cudf_factory(
    transform_spec: Any,
    shared_context: GPUTransformRuntimeContext,
) -> Any:
    context = shared_context.get_or_create("cudf_runtime", CudfRuntimeContext)
    config = transform_spec.config
    if transform_spec.kind == READ_PARQUET_RANGE_KIND:
        return ReadParquetRangeRuntime(config, context)
    if transform_spec.kind == MAP_BATCHES_KIND:
        return MapBatchesRuntime(config, context)
    if transform_spec.kind == VALIDATE_RANGE_OWNERSHIP_KIND:
        return ValidateRangeOwnershipRuntime(config)
    if transform_spec.kind == SORT_PARTITION_KIND:
        return SortPartitionRuntime(config, context)
    if transform_spec.kind == BUILD_GROUP_BOUNDARIES_KIND:
        return BuildGroupBoundariesRuntime(config, context)
    if transform_spec.kind == MAP_GROUP_PARTITIONS_KIND:
        return MapGroupPartitionsRuntime(config, context)
    raise ValueError(f"Unknown GPU transform kind: {transform_spec.kind!r}")


def _register(entry: tuple[Any, ...]) -> None:
    kind, input_payload, output_payload, validator, factory = entry
    register_gpu_transform_runtime(
        RuntimeKey("ray-data-gpu-v1", kind, input_payload, output_payload),
        TransformRuntimeRegistration(validator, factory),
    )


for _entry in (
    (IMPORT_RANGE_WORK_KIND, RAY_BLOCK_STREAM, RANGE_WORK, _is_none, _boundary_factory),
    (
        READ_PARQUET_RANGE_KIND,
        RANGE_WORK,
        RANGE_DATA,
        _is_instance(ReadParquetRangeConfig),
        _cudf_factory,
    ),
    (
        MAP_BATCHES_KIND,
        RANGE_DATA,
        RANGE_DATA,
        _is_instance(MapBatchesConfig),
        _cudf_factory,
    ),
    (
        VALIDATE_RANGE_OWNERSHIP_KIND,
        RANGE_DATA,
        RANGE_DATA,
        _is_instance(ValidateRangeOwnershipConfig),
        _cudf_factory,
    ),
    (
        SORT_PARTITION_KIND,
        RANGE_DATA,
        RANGE_PARTITION_DATA,
        _is_instance(SortPartitionConfig),
        _cudf_factory,
    ),
    (
        BUILD_GROUP_BOUNDARIES_KIND,
        RANGE_PARTITION_DATA,
        GROUPED_RANGE_DATA,
        _is_instance(BuildGroupBoundariesConfig),
        _cudf_factory,
    ),
    (
        MAP_GROUP_PARTITIONS_KIND,
        GROUPED_RANGE_DATA,
        RANGE_DATA,
        _is_instance(MapGroupPartitionsConfig),
        _cudf_factory,
    ),
    (EXPORT_RANGE_DATA_KIND, RANGE_DATA, RAY_BLOCK_STREAM, _is_none, _boundary_factory),
):
    _register(_entry)


__all__ = ["CudfRuntimeContext"]
