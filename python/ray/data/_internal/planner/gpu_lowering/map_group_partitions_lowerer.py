from ray.data._internal.execution.gpu import (
    GPUBoundaryPolicy,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    KeysColocated,
    PayloadKind,
)
from ray.data._internal.execution.gpu.transforms.grouped_partitions import (
    BUILD_GROUP_BOUNDARIES_KIND,
    MAP_GROUP_PARTITIONS_KIND,
    SORT_PARTITION_KIND,
    BuildGroupBoundariesConfig,
    MapGroupPartitionsConfig,
    SortPartitionConfig,
)
from ray.data._internal.logical.operators import MapGroupPartitions
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPULoweredOperator,
    GPULoweringContext,
)


class GPUMapGroupPartitionsLowerer:
    def lower(
        self, logical_op: MapGroupPartitions, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not MapGroupPartitions:
            raise TypeError(
                "GPUMapGroupPartitionsLowerer requires an exact MapGroupPartitions"
            )
        transforms = (
            GPUTransformSpec(
                transform_id="sort_partition",
                kind=SORT_PARTITION_KIND,
                input_payload=PayloadKind.CUDF_BATCH_STREAM,
                output_payload=PayloadKind.CUDF_PARTITION,
                config=SortPartitionConfig(context.group_keys),
                requires_resolved_obligations=True,
            ),
            GPUTransformSpec(
                transform_id="build_group_boundaries",
                kind=BUILD_GROUP_BOUNDARIES_KIND,
                input_payload=PayloadKind.CUDF_PARTITION,
                output_payload=PayloadKind.GROUPED_CUDF_PARTITION,
                config=BuildGroupBoundariesConfig(context.group_keys),
            ),
            GPUTransformSpec(
                transform_id="map_group_partitions",
                kind=MAP_GROUP_PARTITIONS_KIND,
                input_payload=PayloadKind.GROUPED_CUDF_PARTITION,
                output_payload=PayloadKind.OUTPUT_BATCH_STREAM,
                config=MapGroupPartitionsConfig(
                    partition_fn=logical_op.fn,
                    group_keys=context.group_keys,
                    target_max_block_size=context.target_max_block_size,
                ),
            ),
        )
        spec = GPUOperatorSpec(
            transforms=transforms,
            input_payload=PayloadKind.CUDF_BATCH_STREAM,
            output_payload=PayloadKind.OUTPUT_BATCH_STREAM,
            execution_requirements=context.execution_requirements,
            input_requirements=(KeysColocated(context.group_keys),),
            input_boundary=GPUBoundaryPolicy.ELIDABLE,
            output_boundary=GPUBoundaryPolicy.REQUIRED,
            logical_provenance=("MapGroupPartitions",),
            required_logical_provenance=context.required_logical_provenance,
            region_id=context.region_id,
            requires_complete_fusion=True,
        )
        return GPULoweredOperator(
            spec,
            GPUOperatorCreationOptions(
                name="GPUMapGroupPartitions",
                target_max_block_size_override=context.target_max_block_size,
            ),
        )
