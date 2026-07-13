import functools
import inspect

from ray.data._internal.execution.gpu import (
    CompletePartitionBundles,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    KeysColocated,
    PayloadKind,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
)
from ray.data._internal.execution.gpu.properties import (
    PartitioningPreservationError,
    readvertise_preserved_properties,
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
    GPUComponentRole,
    GPULogicalAnalysis,
    GPULoweredOperator,
    GPULoweringContext,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)


class GPUMapGroupPartitionsLowerer:
    component_role = GPUComponentRole.PARTITION_CONSUMER

    def analyze(
        self, logical_op: MapGroupPartitions, data_context
    ) -> GPULogicalAnalysis:
        del data_context
        if type(logical_op) is not MapGroupPartitions:
            return GPULogicalAnalysis(False, "logical_map_group_partitions")
        partition_keys = (
            (logical_op.key,)
            if isinstance(logical_op.key, str)
            else tuple(logical_op.key)
        )
        if not partition_keys or any(
            not isinstance(key, str) or not key for key in partition_keys
        ):
            return GPULogicalAnalysis(False, "group_keys")
        requested_partitions = logical_op.num_partitions
        if requested_partitions is not None and (
            not isinstance(requested_partitions, int)
            or isinstance(requested_partitions, bool)
            or requested_partitions <= 0
        ):
            return GPULogicalAnalysis(False, "num_partitions")
        if inspect.isclass(logical_op.fn) or not _is_synchronous(logical_op.fn):
            return GPULogicalAnalysis(
                False,
                "MapGroupPartitions requires a synchronous function, not a class",
            )
        return GPULogicalAnalysis(
            True,
            None,
            preserves_partitioning=bool(
                logical_op.udf_properties is not None
                and logical_op.udf_properties.preserves_partitioning
            ),
            partition_keys=partition_keys,
            requested_partitions=requested_partitions,
        )

    def lower(
        self, logical_op: MapGroupPartitions, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not MapGroupPartitions:
            raise TypeError(
                "GPUMapGroupPartitionsLowerer requires an exact MapGroupPartitions"
            )
        if context.source_artifacts is None:
            raise ValueError("GPU MGP lowering requires Read source artifacts")
        logical_group_keys = (
            (logical_op.key,)
            if isinstance(logical_op.key, str)
            else tuple(logical_op.key)
        )
        if logical_group_keys != context.group_keys:
            raise ValueError(
                "MGP lowering context group_keys must match the logical operator"
            )

        preserves = bool(
            logical_op.udf_properties is not None
            and logical_op.udf_properties.preserves_partitioning
        )
        try:
            readvertisement = readvertise_preserved_properties(
                context.input_properties,
                preserves_partitioning=preserves,
                # MGP can advertise a complete output partition only when the
                # partition identity itself survives its UDF.
                preserves_complete_partition_bundles=preserves,
            )
        except PartitioningPreservationError as error:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer=MapGroupPartitions({logical_op.name}); {error}",
            ) from error

        transforms = [
            GPUTransformSpec(
                transform_id=context.transform_id(0, SORT_PARTITION_KIND),
                kind=SORT_PARTITION_KIND,
                input_payload=PayloadKind.RANGE_DATA,
                output_payload=PayloadKind.RANGE_PARTITION_DATA,
                config=SortPartitionConfig(context.group_keys),
                requires_resolved_obligations=True,
            ),
            GPUTransformSpec(
                transform_id=context.transform_id(1, BUILD_GROUP_BOUNDARIES_KIND),
                kind=BUILD_GROUP_BOUNDARIES_KIND,
                input_payload=PayloadKind.RANGE_PARTITION_DATA,
                output_payload=PayloadKind.GROUPED_RANGE_DATA,
                config=BuildGroupBoundariesConfig(context.group_keys),
            ),
            GPUTransformSpec(
                transform_id=context.transform_id(2, MAP_GROUP_PARTITIONS_KIND),
                kind=MAP_GROUP_PARTITIONS_KIND,
                input_payload=PayloadKind.GROUPED_RANGE_DATA,
                output_payload=PayloadKind.RANGE_DATA,
                config=MapGroupPartitionsConfig(
                    partition_fn=logical_op.fn,
                    group_keys=context.group_keys,
                    target_max_block_size=context.target_max_block_size,
                ),
                creates_obligations=readvertisement.created_obligations,
            ),
        ]

        for ordinal, validation in enumerate(
            readvertisement.partitioning_validations,
            start=3,
        ):
            transforms.append(
                GPUTransformSpec(
                    transform_id=context.transform_id(ordinal, validation.kind),
                    kind=validation.kind,
                    input_payload=PayloadKind.RANGE_DATA,
                    output_payload=PayloadKind.RANGE_DATA,
                    config=validation.config,
                    discharges_obligations=validation.obligations,
                )
            )

        spec = GPUOperatorSpec(
            spec_instance_id=context.logical_node_id,
            transforms=tuple(transforms),
            external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
            external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_input_payload=PayloadKind.RANGE_DATA,
            native_output_payload=PayloadKind.RANGE_DATA,
            input_adapter_key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
            output_adapter_key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
            terminal_output_adapter_key=RANGE_DATA_TERMINAL_ADAPTER_KEY,
            execution_requirements=context.execution_requirements,
            input_requirements=(
                KeysColocated(context.group_keys),
                CompletePartitionBundles(),
            ),
            output_properties=readvertisement.output_properties,
            logical_provenance=(context.logical_node_id,),
        )
        downstream_context = context.for_lowered_output(spec)
        return GPULoweredOperator(
            spec=spec,
            creation_options=GPUOperatorCreationOptions(
                name="GPUMapGroupPartitions",
                target_max_block_size_override=context.target_max_block_size,
            ),
            downstream_context=downstream_context,
            source_artifacts=context.source_artifacts,
        )


def _is_synchronous(callable_) -> bool:
    if not callable(callable_):
        return False
    if inspect.isroutine(callable_) or isinstance(callable_, functools.partial):
        function = callable_
    else:
        function = callable_.__call__
    return not inspect.iscoroutinefunction(function) and not inspect.isasyncgenfunction(
        function
    )
