"""Registry-friendly construction of actor-local GPU transform runtimes."""

from __future__ import annotations

from typing import Any

from ray.data._internal.execution.gpu.transforms._runtime_support import (
    CudfRuntimeContext,
)
from ray.data._internal.execution.gpu.runtime import (
    GPUTransformRuntimeContext,
    register_gpu_transform_runtime,
)


def _create_cudf_transform_runtime(
    kind: str,
    config: Any,
    context: CudfRuntimeContext,
) -> Any:
    """Construct one initialized transform callable from a declarative config."""

    if kind == "read_parquet_range":
        from ray.data._internal.execution.gpu.transforms.parquet_read import (
            ReadParquetRangeRuntime,
        )

        return ReadParquetRangeRuntime(config, context)
    if kind == "map_batches":
        from ray.data._internal.execution.gpu.transforms.map_batches import (
            MapBatchesRuntime,
        )

        return MapBatchesRuntime(config, context)
    if kind == "validate_range_ownership":
        from ray.data._internal.execution.gpu.transforms.map_batches import (
            ValidateRangeOwnershipRuntime,
        )

        return ValidateRangeOwnershipRuntime(config)
    if kind == "sort_partition":
        from ray.data._internal.execution.gpu.transforms.grouped_partitions import (
            SortPartitionRuntime,
        )

        return SortPartitionRuntime(config, context)
    if kind == "build_group_boundaries":
        from ray.data._internal.execution.gpu.transforms.grouped_partitions import (
            BuildGroupBoundariesRuntime,
        )

        return BuildGroupBoundariesRuntime(config, context)
    if kind == "map_group_partitions":
        from ray.data._internal.execution.gpu.transforms.grouped_partitions import (
            MapGroupPartitionsRuntime,
        )

        return MapGroupPartitionsRuntime(config, context)
    raise ValueError(f"Unknown GPU transform kind: {kind!r}")


def _registered_runtime_factory(
    transform_spec: Any, shared_context: GPUTransformRuntimeContext
) -> Any:
    """Bridge the durable registry to one cuDF context shared by the actor."""

    cudf_context = shared_context.get_or_create("cudf_runtime", CudfRuntimeContext)
    return _create_cudf_transform_runtime(
        transform_spec.kind, transform_spec.config, cudf_context
    )


for _kind in (
    "read_parquet_range",
    "map_batches",
    "validate_range_ownership",
    "sort_partition",
    "build_group_boundaries",
    "map_group_partitions",
):
    register_gpu_transform_runtime(
        _kind, _registered_runtime_factory, replace_existing=True
    )


__all__ = ["CudfRuntimeContext"]
