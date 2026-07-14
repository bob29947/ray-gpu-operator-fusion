"""Independent GPU lowering for MapGroupPartitions."""

from ray.data._internal.execution.gpu.operator import GPUOperatorCreationOptions
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUTransformSpec,
    GROUPED_RANGE_DATA,
    RANGE_DATA,
    RANGE_PARTITION_DATA,
)
from ray.data._internal.execution.gpu.properties import KeysColocated
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
    GPULogicalAnalysis,
    GPULoweredOperator,
    GPULoweringContext,
    GPURequirementDemand,
    is_positive_int,
    is_synchronous_callable,
)
from ray.data._internal.planner.gpu_requirement_analysis import RangePartitionDemand


class GPUMapGroupPartitionsLowerer:
    def analyze(
        self, logical_op: MapGroupPartitions, data_context
    ) -> GPULogicalAnalysis:
        del data_context
        if type(logical_op) is not MapGroupPartitions:
            return GPULogicalAnalysis(False, "logical_map_group_partitions")
        partition_keys = _partition_keys(logical_op)
        if not partition_keys or any(
            not isinstance(key, str) or not key for key in partition_keys
        ):
            return GPULogicalAnalysis(False, "group_keys")
        requested_partitions = logical_op.num_partitions
        if requested_partitions is not None and not is_positive_int(
            requested_partitions
        ):
            return GPULogicalAnalysis(False, "num_partitions")
        if isinstance(logical_op.fn, type) or not is_synchronous_callable(
            logical_op.fn
        ):
            return GPULogicalAnalysis(
                False,
                "MapGroupPartitions requires a synchronous function, not a class",
            )
        return GPULogicalAnalysis(True)

    def transfer_demands(
        self,
        logical_op: MapGroupPartitions,
        downstream_demands: tuple[GPURequirementDemand, ...],
        analysis: GPULogicalAnalysis,
    ) -> tuple[tuple[GPURequirementDemand, ...], ...]:
        del downstream_demands
        if len(logical_op.input_dependencies) != 1:
            return tuple(() for _ in logical_op.input_dependencies)
        demand = RangePartitionDemand(
            _partition_keys(logical_op), logical_op.num_partitions
        )
        return ((demand,),)

    def lower(
        self, logical_op: MapGroupPartitions, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not MapGroupPartitions:
            raise TypeError(
                "GPUMapGroupPartitionsLowerer requires an exact MapGroupPartitions"
            )
        requirements = context.execution_requirements or GPUExecutionRequirements()
        group_keys = _partition_keys(logical_op)
        spec = GPUOperatorSpec(
            transforms=(
                GPUTransformSpec(
                    kind=SORT_PARTITION_KIND,
                    input_payload=RANGE_DATA,
                    output_payload=RANGE_PARTITION_DATA,
                    config=SortPartitionConfig(group_keys),
                ),
                GPUTransformSpec(
                    kind=BUILD_GROUP_BOUNDARIES_KIND,
                    input_payload=RANGE_PARTITION_DATA,
                    output_payload=GROUPED_RANGE_DATA,
                    config=BuildGroupBoundariesConfig(group_keys),
                ),
                GPUTransformSpec(
                    kind=MAP_GROUP_PARTITIONS_KIND,
                    input_payload=GROUPED_RANGE_DATA,
                    output_payload=RANGE_DATA,
                    config=MapGroupPartitionsConfig(
                        partition_fn=logical_op.fn,
                        group_keys=group_keys,
                        target_max_block_size=context.target_max_block_size,
                    ),
                ),
            ),
            execution_requirements=requirements,
            input_requirements=(KeysColocated(group_keys),),
            output_facts=(),
        )
        return GPULoweredOperator(
            spec=spec,
            creation_options=GPUOperatorCreationOptions(
                name="GPUMapGroupPartitions",
                target_max_block_size_override=context.target_max_block_size,
            ),
        )


def _partition_keys(logical_op: MapGroupPartitions) -> tuple[str, ...]:
    return (
        (logical_op.key,) if isinstance(logical_op.key, str) else tuple(logical_op.key)
    )
